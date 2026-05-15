"""
trade_manager.py — Active position management after entry.

Three mechanisms (all configurable in config.yaml → trade_management):

1. 3-STAGE PARTIAL CLOSE  — close 30% at 1R, 30% at 2R, 40% at 3R
                             Locks real cash progressively.

2. BREAK-EVEN             — move SL to entry+buffer at 1R.
                             Worst case becomes scratch, not a loss.

3. ATR-BASED TRAILING     — after 1.5R, trail by ATR × multiplier.
                             Dynamically wide in volatile markets.

4. TIME-BASED EXIT        — close stale trades after N bars if still not in profit.
                             Eliminates zombie positions tying up margin.
"""

import logging
from datetime import datetime
from typing import Dict, Set

import MetaTrader5 as mt5

logger = logging.getLogger('AI-Trade')


class TradeManager:
    def __init__(self, config: dict):
        self._cfg = config.get('trade_management', {})

        self._be_applied:  Set[int] = set()   # tickets where BE was moved
        self._pc_stages:   Dict[int, int] = {}  # ticket -> last stage applied (0,1,2)
        self._open_time:   Dict[int, datetime] = {}  # ticket -> open datetime
        self._open_atr:    Dict[int, float] = {}  # ticket -> ATR at entry

    # ── Public API ────────────────────────────────────────────────────────────

    def is_at_breakeven(self, ticket: int) -> bool:
        """Returns True only after TradeManager has physically moved SL to breakeven."""
        return ticket in self._be_applied

    def manage(self, positions: list) -> None:
        open_tickets = {p.ticket for p in positions}

        self._be_applied -= (self._be_applied - open_tickets)
        for t in list(self._pc_stages.keys()):
            if t not in open_tickets:
                del self._pc_stages[t]
        for t in list(self._open_time.keys()):
            if t not in open_tickets:
                del self._open_time[t]
        for t in list(self._open_atr.keys()):
            if t not in open_tickets:
                del self._open_atr[t]

        for pos in positions:
            try:
                self._manage_one(pos)
            except Exception as exc:
                logger.error(f"TradeManager error ticket={pos.ticket}: {exc}")

    # ── Per-position logic ────────────────────────────────────────────────────

    def _manage_one(self, pos) -> None:
        if pos.sl == 0:
            return

        sym_info = mt5.symbol_info(pos.symbol)
        tick     = mt5.symbol_info_tick(pos.symbol)
        if sym_info is None or tick is None:
            return

        digits = sym_info.digits
        point  = sym_info.point if sym_info.point > 0 else 1e-5
        entry  = pos.price_open
        sl     = pos.sl
        tp     = pos.tp

        is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
        price  = tick.bid if is_buy else tick.ask

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return

        profit_r = (price - entry) / sl_dist if is_buy else (entry - price) / sl_dist

        # Track open time
        if pos.ticket not in self._open_time:
            try:
                self._open_time[pos.ticket] = datetime.fromtimestamp(pos.time)
            except Exception:
                self._open_time[pos.ticket] = datetime.now()

        # Track ATR at entry (use current ATR as proxy if not set)
        if pos.ticket not in self._open_atr:
            self._open_atr[pos.ticket] = sl_dist / self._cfg.get('atr_sl_multiplier', 2.0)

        # ── Step 1: 3-stage partial close ─────────────────────────────────────
        if self._cfg.get('partial_close', True):
            stages = self._cfg.get('partial_close_stages', [
                {'r': 1.0, 'pct': 0.30},
                {'r': 2.0, 'pct': 0.30},
                {'r': 3.0, 'pct': 0.40},
            ])
            current_stage = self._pc_stages.get(pos.ticket, 0)

            for i, stage in enumerate(stages):
                stage_idx = i + 1
                if stage_idx <= current_stage:
                    continue
                if profit_r >= stage['r']:
                    vol   = round(pos.volume * stage['pct'], 2)
                    vol   = max(float(sym_info.volume_min), vol)
                    step  = float(sym_info.volume_step)
                    vol   = round(vol / step) * step

                    if vol < pos.volume and self._partial_close(pos, vol, sym_info, tick):
                        self._pc_stages[pos.ticket] = stage_idx
                        logger.info(
                            f"[TM] Stage-{stage_idx} partial close ticket={pos.ticket} "
                            f"vol={vol:.2f} @ {price:.{digits}f} ({profit_r:.2f}R)"
                        )
                    break  # only one stage per cycle

        # ── Step 2: Break-even ────────────────────────────────────────────────
        be_r  = self._cfg.get('breakeven_r', 1.0)
        buf   = self._cfg.get('breakeven_buffer_pts', 5) * point

        if pos.ticket not in self._be_applied and profit_r >= be_r:
            new_sl      = (entry + buf) if is_buy else (entry - buf)
            sl_improved = (new_sl > sl) if is_buy else (new_sl < sl)

            if sl_improved and self._modify_sl(pos, new_sl, digits, tp):
                self._be_applied.add(pos.ticket)
                logger.info(
                    f"[TM] Break-even ticket={pos.ticket} "
                    f"new_sl={new_sl:.{digits}f} ({profit_r:.2f}R)"
                )

        # ── Step 3: ATR-based trailing stop ───────────────────────────────────
        trail_r      = self._cfg.get('trailing_r', 1.5)
        atr_at_entry = self._open_atr.get(pos.ticket, sl_dist)
        atr_mult     = self._cfg.get('trailing_atr_mult', 1.0)
        trail_dist   = atr_at_entry * atr_mult   # dynamic trail distance

        if pos.ticket in self._be_applied and profit_r >= trail_r:
            if is_buy:
                candidate_sl = price - trail_dist
                if candidate_sl > pos.sl + point:
                    if self._modify_sl(pos, candidate_sl, digits, tp):
                        logger.info(
                            f"[TM] Trail UP ticket={pos.ticket} "
                            f"sl={candidate_sl:.{digits}f} ({profit_r:.2f}R)"
                        )
            else:
                candidate_sl = price + trail_dist
                if candidate_sl < pos.sl - point:
                    if self._modify_sl(pos, candidate_sl, digits, tp):
                        logger.info(
                            f"[TM] Trail DOWN ticket={pos.ticket} "
                            f"sl={candidate_sl:.{digits}f} ({profit_r:.2f}R)"
                        )

        # ── Step 4: Time-based exit ───────────────────────────────────────────
        if self._cfg.get('time_exit_enabled', True):
            self._check_time_exit(pos, profit_r, price, digits, sym_info, tick)

    def _check_time_exit(
        self, pos, profit_r: float, price: float, digits: int, sym_info, tick
    ) -> None:
        max_bars   = self._cfg.get('time_exit_bars', 48)
        min_r      = self._cfg.get('time_exit_min_r', -0.3)
        open_ts    = self._open_time.get(pos.ticket)
        if open_ts is None:
            return

        elapsed_minutes = (datetime.now() - open_ts).total_seconds() / 60.0
        # Approximate bars elapsed (assume M15 = 15-min bars)
        bars_elapsed = elapsed_minutes / 15.0

        if bars_elapsed >= max_bars and profit_r < min_r:
            logger.info(
                f"[TM] Time-exit ticket={pos.ticket} "
                f"bars≈{bars_elapsed:.0f} profit_r={profit_r:.2f} — closing stale trade"
            )
            self._market_close(pos, price, sym_info, tick)

    # ── MT5 helpers ───────────────────────────────────────────────────────────

    def _modify_sl(self, pos, new_sl: float, digits: int, tp: float) -> bool:
        result = mt5.order_send({
            'action':   mt5.TRADE_ACTION_SLTP,
            'position': pos.ticket,
            'symbol':   pos.symbol,
            'sl':       round(new_sl, digits),
            'tp':       tp,
        })
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        logger.warning(
            f"_modify_sl failed ticket={pos.ticket} "
            f"retcode={result.retcode if result else 'None'}"
        )
        return False

    def _filling_mode(self, sym_info) -> int:
        mode       = sym_info.filling_mode
        SYM_FOK    = getattr(mt5, 'SYMBOL_FILLING_FOK',    1)
        SYM_IOC    = getattr(mt5, 'SYMBOL_FILLING_IOC',    2)
        ORD_FOK    = getattr(mt5, 'ORDER_FILLING_FOK',    0)
        ORD_IOC    = getattr(mt5, 'ORDER_FILLING_IOC',    1)
        ORD_RETURN = getattr(mt5, 'ORDER_FILLING_RETURN', 2)
        return (ORD_IOC    if mode & SYM_IOC else
                ORD_FOK    if mode & SYM_FOK else
                ORD_RETURN)

    def _partial_close(self, pos, volume: float, sym_info, tick) -> bool:
        return self._send_close_order(pos, volume, sym_info, tick, 'partial-tp')

    def _market_close(self, pos, price: float, sym_info, tick) -> bool:
        return self._send_close_order(pos, pos.volume, sym_info, tick, 'time-exit')

    def _send_close_order(self, pos, volume: float, sym_info, tick, comment: str) -> bool:
        is_buy     = (pos.type == mt5.ORDER_TYPE_BUY)
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price      = tick.bid if is_buy else tick.ask

        result = mt5.order_send({
            'action':       mt5.TRADE_ACTION_DEAL,
            'symbol':       pos.symbol,
            'volume':       float(volume),
            'type':         close_type,
            'position':     pos.ticket,
            'price':        price,
            'deviation':    20,
            'magic':        pos.magic,
            'comment':      comment,
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': self._filling_mode(sym_info),
        })

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"[TM] Close OK ticket={pos.ticket} "
                f"vol={volume:.2f} @ {result.price} comment={comment}"
            )
            return True

        logger.error(
            f"[TM] Close FAILED ticket={pos.ticket} comment={comment}: "
            f"{mt5.last_error() if result is None else result.comment}"
        )
        return False
