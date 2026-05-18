"""
main.py — Trading engine entry point.

Execution flow (each cycle, default 60 s):
  1. Fetch account info & update risk state
  2. Enforce global drawdown / daily-loss limits
  3. For each symbol:
       a. Fetch OHLCV -> compute indicators
       b. Optionally retrain AI (every N hours)
       c. Run strategy -> get signal
       d. Run AI -> get bias + confidence
       e. Cross-check AI filter
       f. Calculate SL / TP / lot-size
       g. Execute order via MT5
  4. Sync closed positions to DB
  5. Write dashboard state JSON
  6. Sleep until next cycle

Usage
-----
  1. Open MetaTrader 5 and log in to your account.
  2. python main.py
"""

import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yaml

from utils import (
    setup_logging, compute_indicators, init_db,
    insert_trade, close_trade_db, record_equity,
    write_state, get_trade_stats, log_activity,
)
from strategy      import generate_signal, detect_regime
from risk          import RiskManager
from execution_mt5 import MT5Executor
from ai_model      import AIModel
from trade_manager import TradeManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def rates_to_df(rates) -> pd.DataFrame:
    """Convert MT5 rates array to a standard OHLCV DataFrame."""
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']].copy()


# ── Engine ────────────────────────────────────────────────────────────────────

class TradingEngine:
    CYCLE_SECONDS = 60   # main loop sleep interval

    def __init__(self):
        self.config = load_config()
        self.logger = setup_logging(self.config)
        init_db()

        self.risk         = RiskManager(self.config)
        self.executor     = MT5Executor(self.config)
        self.ai           = AIModel(self.config)
        self.trade_manager = TradeManager(self.config)

        self.running              = False
        self._ai_last_train_ts    = 0.0
        self._known_tickets: set  = set()            # tickets we opened this session
        self._terminal: dict      = {}               # per-symbol terminal data
        self._last_trade_ts: dict = {}               # symbol -> timestamp of last trade open
        self._ticket_direction: dict   = {}            # ticket -> 'BUY'|'SELL'
        self._ticket_confidence: dict  = {}            # ticket -> ai_confidence int
        # direction_ban: symbol -> {'direction': 'BUY'|'SELL', 'until': float}
        self._direction_ban: dict = {}
        # per-symbol consecutive losses per direction: symbol -> {'BUY': int, 'SELL': int}
        self._dir_loss_streak: dict = {}

        self.logger.info("=" * 60)
        self.logger.info("  AI-Trade Engine  |  starting up")
        self.logger.info("=" * 60)

    # ── MT5 connection ────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to the MT5 terminal that is already open on this PC."""

        return self.executor.connect()

    # ── Terminal data (for live dashboard) ───────────────────────────────────

    def _collect_terminal_data(
        self, symbol: str, df: pd.DataFrame,
        signal: str, atr: float, ai_bias: str, ai_confidence: int,
        regime: str = 'TREND',
    ) -> None:
        """Compute all indicator readings and store for the dashboard terminal."""
        try:
            latest = df.iloc[-1]
            prev   = df.iloc[-2]
            rsi    = float(latest['rsi'])
            close  = float(latest['close'])
            ema200 = float(latest['ema200'])

            # RSI label
            if   rsi < 30:  rsi_lbl, rsi_col = 'Oversold',   'red'
            elif rsi < 50:  rsi_lbl, rsi_col = 'Bearish',    'yellow'
            elif rsi <= 70: rsi_lbl, rsi_col = 'Neutral',    'green'
            else:           rsi_lbl, rsi_col = 'Overbought', 'red'

            # MACD status
            pl = float(prev['macd_line']);   ps = float(prev['macd_signal'])
            cl = float(latest['macd_line']); cs = float(latest['macd_signal'])
            if   pl < ps and cl >= cs: macd_st, macd_col = 'Bullish Cross', 'green'
            elif pl > ps and cl <= cs: macd_st, macd_col = 'Bearish Cross', 'red'
            elif float(latest['macd_hist']) > 0: macd_st, macd_col = 'Bullish', 'green'
            else:                                macd_st, macd_col = 'Bearish', 'red'

            # EMA200 status
            ema_st  = 'Above Trend' if close > ema200 else 'Below Trend'
            ema_col = 'white'       if close > ema200 else 'red'

            # AI label
            if signal == 'BUY':
                ai_lbl = 'Strong Buy' if ai_confidence >= 80 else ('Buy' if ai_confidence >= 60 else 'Watching')
            elif signal == 'SELL':
                ai_lbl = 'Strong Sell' if ai_confidence >= 80 else ('Sell' if ai_confidence >= 60 else 'Watching')
            else:
                ai_lbl = 'Neutral'

            # Live price + spread
            tick     = self.executor.get_tick(symbol)
            price    = float(tick.bid) if tick else close
            sym_info = self.executor.get_symbol_info(symbol)
            spread_pips = 0.0
            if sym_info and tick and sym_info.point > 0:
                spread_pips = round((tick.ask - tick.bid) / sym_info.point / 10, 1)

            avg_atr = float(df['atr'].tail(20).mean())
            vol_pct = round(atr / avg_atr * 100, 1) if avg_atr > 0 else 0
            adx_val = round(float(latest.get('adx', 0.0)), 1) if 'adx' in df.columns else 0.0

            self._terminal[symbol] = {
                'price':         round(price, 5),
                'spread_pips':   spread_pips,
                'signal':        signal,
                'regime':        regime,
                'adx':           adx_val,
                'rsi':           round(rsi, 1),
                'rsi_label':     rsi_lbl,
                'rsi_color':     rsi_col,
                'macd_status':   macd_st,
                'macd_color':    macd_col,
                'macd_hist':     round(float(latest.get('macd_hist', 0.0)), 5),
                'ema50':         round(float(latest.get('ema50',  0.0)), 2),
                'ema200':        round(float(latest.get('ema200', 0.0)), 2),
                'stoch_k':       round(float(latest.get('stoch_k', 50.0)), 1),
                'ema200_status': ema_st,
                'ema200_color':  ema_col,
                'ai_confidence': ai_confidence,
                'ai_label':      ai_lbl,
                'ai_bias':       ai_bias,
                'volatility_pct':vol_pct,
                'volatility_ok': atr >= avg_atr * 0.5,
                'atr':           round(atr, 5),
                'updated':       datetime.now().strftime('%H:%M:%S'),
            }
        except Exception as exc:
            self.logger.debug(f"_collect_terminal_data({symbol}): {exc}")

    # ── News blackout filter ──────────────────────────────────────────────────

    def _is_news_blackout(self) -> bool:
        """
        Block trading before/after high-impact news using MT5 calendar.
        Configurable via trade_management.news_filter_enabled.
        """
        tm = self.config.get('trade_management', {})
        if not tm.get('news_filter_enabled', True):
            return False
        before_min = tm.get('news_blackout_before', 30)
        after_min  = tm.get('news_blackout_after',  15)
        try:
            import MetaTrader5 as mt5
            now    = datetime.now(timezone.utc)
            fr     = (now - timedelta(minutes=after_min)).timestamp()
            to     = (now + timedelta(minutes=before_min)).timestamp()
            events = mt5.calendar_query(fr, to) or []
            for ev in events:
                if getattr(ev, 'importance', 0) >= 3:
                    self.logger.info(
                        f"News blackout: {getattr(ev, 'name', '?')} — skipping"
                    )
                    return True
        except Exception:
            pass
        return False

    # ── AI retraining ─────────────────────────────────────────────────────────

    def _maybe_retrain(self, df: pd.DataFrame) -> None:
        interval = self.config['ai']['retrain_interval'] * 60
        if time.time() - self._ai_last_train_ts >= interval:
            ok = self.ai.train(df)
            self._ai_last_train_ts = time.time()
            # Auto-disable only when AUC is strongly inversely predictive.
            # min_confidence=62% is the main guard — a model at ~50% AUC
            # won't reach 62% confidence and won't trigger trades.
            if ok and hasattr(self.ai, '_last_auc'):
                auc = self.ai._last_auc
                if auc < 0.44:
                    self.logger.warning(
                        f"AI auto-disabled: AUC={auc:.3f} < 0.44 after retrain. "
                        "Running on technical signals only."
                    )
                    self.config['ai']['enabled'] = False
                elif not self.config['ai']['enabled']:
                    self.logger.info(
                        f"AI auto-enabled: AUC={auc:.3f} >= 0.44 after retrain."
                    )
                    self.config['ai']['enabled'] = True

    # ── Higher-timeframe trend bias ───────────────────────────────────────────

    def _fetch_htf_bias(self, symbol: str) -> str:
        """
        Two-level trend filter: H4 EMA200 + D1 EMA50.
        Both must agree (or at least not conflict) before returning a directional bias.
        Returns 'BUY' | 'SELL' | 'NEUTRAL'.
        """
        htf_cfg = self.config.get('htf_filter', {})
        if not htf_cfg.get('enabled', False):
            return 'NEUTRAL'

        tf    = htf_cfg.get('timeframe', 'H4')
        bars  = htf_cfg.get('bars', 250)
        ema_p = htf_cfg.get('ema_period', 200)
        # Wide neutral zone: price must be clearly above/below EMA200 before taking a side.
        # 0.5% on gold ≈ $22 buffer — prevents trading during EMA200 retests.
        zone  = htf_cfg.get('neutral_zone', 0.005)

        try:
            # ── Level 1: H4 EMA200 ───────────────────────────────────────────
            rates = self.executor.get_ohlcv(symbol, tf, bars)
            if rates is None or len(rates) < ema_p + 10:
                return 'NEUTRAL'

            df_h4  = rates_to_df(rates)
            closes = df_h4['close']
            ema200 = closes.ewm(span=ema_p, adjust=False).mean().iloc[-1]
            price  = float(closes.iloc[-1])

            if price > ema200 * (1 + zone):
                h4_bias = 'BUY'
            elif price < ema200 * (1 - zone):
                h4_bias = 'SELL'
            else:
                h4_bias = 'NEUTRAL'   # inside buffer → do not trade

            # ── Level 2: D1 EMA50 (macro trend direction) ────────────────────
            d1_rates = self.executor.get_ohlcv(symbol, 'D1', 80)
            d1_bias  = 'NEUTRAL'
            if d1_rates is not None and len(d1_rates) >= 55:
                df_d1    = rates_to_df(d1_rates)
                d1_close = float(df_d1['close'].iloc[-1])
                d1_ema50 = float(df_d1['close'].ewm(span=50, adjust=False).mean().iloc[-1])
                if d1_close > d1_ema50 * 1.001:
                    d1_bias = 'BUY'
                elif d1_close < d1_ema50 * 0.999:
                    d1_bias = 'SELL'

            self.logger.debug(
                f"HTF({symbol}): H4={h4_bias}(price={price:.1f} ema200={ema200:.1f}) "
                f"D1={d1_bias}(ema50={d1_ema50 if d1_rates is not None and len(d1_rates)>=55 else 'N/A':.1f})"
            )

            # ── Combine: block if H4 and D1 conflict ─────────────────────────
            if h4_bias != 'NEUTRAL' and d1_bias != 'NEUTRAL' and h4_bias != d1_bias:
                # H4 and D1 disagree → market is transitioning → stay flat
                return 'NEUTRAL'

            # If H4 is NEUTRAL (EMA retest zone), use D1 as tiebreaker but be conservative
            if h4_bias == 'NEUTRAL':
                return 'NEUTRAL'   # always flat when H4 is in neutral zone

            return h4_bias

        except Exception as exc:
            self.logger.debug(f"_fetch_htf_bias({symbol}): {exc}")
            return 'NEUTRAL'

    # ── Direction ban ─────────────────────────────────────────────────────────

    def _is_direction_banned(self, symbol: str, direction: str) -> bool:
        ban_cfg = self.config.get('direction_ban', {})
        if not ban_cfg.get('enabled', False):
            return False

        entry = self._direction_ban.get(symbol)
        if not entry:
            return False
        if entry['direction'] != direction:
            return False
        if time.time() > entry['until']:
            del self._direction_ban[symbol]
            return False

        remaining = (entry['until'] - time.time()) / 3600
        self.logger.info(
            f"{symbol}: direction {direction} BANNED for {remaining:.1f}h more "
            f"(consecutive {direction} losses)"
        )
        return True

    def _update_direction_streak(self, symbol: str, direction: str, profit: float) -> None:
        ban_cfg = self.config.get('direction_ban', {})
        if not ban_cfg.get('enabled', False):
            return

        if symbol not in self._dir_loss_streak:
            self._dir_loss_streak[symbol] = {'BUY': 0, 'SELL': 0}

        if profit < 0:
            self._dir_loss_streak[symbol][direction] += 1
            # Reset opposite direction streak
            opp = 'SELL' if direction == 'BUY' else 'BUY'
            self._dir_loss_streak[symbol][opp] = 0

            max_losses = ban_cfg.get('max_same_dir_losses', 2)
            if self._dir_loss_streak[symbol][direction] >= max_losses:
                ban_hours = ban_cfg.get('ban_hours', 4)
                self._direction_ban[symbol] = {
                    'direction': direction,
                    'until': time.time() + ban_hours * 3600,
                }
                self._dir_loss_streak[symbol][direction] = 0
                self.logger.warning(
                    f"{symbol}: {direction} direction BANNED for {ban_hours}h "
                    f"after {max_losses} consecutive losses"
                )
                log_activity(
                    symbol,
                    f"แบน {direction} {ban_hours} ชั่วโมง — แพ้ซ้อน {max_losses} ครั้ง",
                    'warning',
                )
        else:
            # Win resets the streak for that direction
            if symbol in self._dir_loss_streak:
                self._dir_loss_streak[symbol][direction] = 0

    # ── Per-symbol logic ──────────────────────────────────────────────────────

    def _process_symbol(self, symbol: str, balance: float) -> None:
        cfg   = self.config
        magic = cfg['trading']['magic_number']

        # ── 1. Fetch data ─────────────────────────────────────────────────────
        min_bars = max(cfg['strategy']['ema_slow'] + 300,
                       cfg.get('ai', {}).get('training_bars', 2000))
        rates = self.executor.get_ohlcv(symbol, cfg['trading']['timeframe'], min_bars)
        if rates is None or len(rates) < 250:
            self.logger.warning(f"{symbol}: not enough bars ({len(rates) if rates is not None else 0})")
            return

        df = rates_to_df(rates)
        df = compute_indicators(df, cfg)
        if len(df) < 10:
            return

        # ── 2. AI retrain + online model update ──────────────────────────────
        self._maybe_retrain(df)
        self.ai.update_online_model(df)

        # ── 2b. Always collect terminal data so dashboard stays live ──────────
        _sig_preview, _atr_preview, _, _ = generate_signal(df, cfg)
        _regime_preview = detect_regime(df, cfg)
        _ai_bias_p, _ai_conf_p = self.ai.predict(df)
        self._collect_terminal_data(
            symbol, df, _sig_preview, _atr_preview,
            _ai_bias_p, _ai_conf_p, _regime_preview,
        )

        # ── 3. Multi-trade check: allow up to max_concurrent_trades per symbol ──
        existing_sym_pos = [
            p for p in self.executor.get_all_open_positions()
            if p.symbol == symbol and p.magic == magic
        ]
        sym_max = cfg['risk']['max_concurrent_trades']
        if len(existing_sym_pos) >= sym_max:
            self.logger.debug(
                f"{symbol}: {len(existing_sym_pos)}/{sym_max} trades open — skip"
            )
            return

        # ── 4. Per-symbol risk checks ─────────────────────────────────────────
        if not self.risk.check_daily_loss_limit(balance):
            return
        if not self.risk.check_drawdown_limit(balance):
            self.running = False
            return

        all_positions = self.executor.get_all_open_positions()
        if not self.risk.can_open_trade(len(all_positions)):
            return

        # ── 5. Cooldown filter ───────────────────────────────────────────────
        cooldown_sec = cfg.get('trade_management', {}).get('cooldown_minutes', 90) * 60
        last_trade   = self._last_trade_ts.get(symbol, 0)
        elapsed      = time.time() - last_trade
        if elapsed < cooldown_sec:
            remaining_min = (cooldown_sec - elapsed) / 60
            self.logger.info(
                f"{symbol}: cooldown active — {remaining_min:.0f} min remaining "
                f"(next entry after {cooldown_sec/60:.0f} min gap)"
            )
            return

        # ── 5b. News blackout filter ─────────────────────────────────────────
        if self._is_news_blackout():
            log_activity(symbol, "News blackout — หยุดเทรดช่วงข่าวสำคัญ", 'warning')
            return

        # ── 6. Spread filter ─────────────────────────────────────────────────
        tick_pre = self.executor.get_tick(symbol)
        sym_pre  = self.executor.get_symbol_info(symbol)
        if tick_pre and sym_pre and sym_pre.point > 0:
            spread_pips = (tick_pre.ask - tick_pre.bid) / sym_pre.point / 10
            is_gold = 'XAU' in symbol.upper()
            max_sp  = cfg['trade_management'].get(
                'max_spread_gold' if is_gold else 'max_spread_forex', 5.0
            )
            if spread_pips > max_sp:
                log_activity(
                    symbol,
                    f"Spread {spread_pips:.1f} pips > max {max_sp} — skip",
                    'warning',
                )
                self.logger.info(
                    f"{symbol}: spread {spread_pips:.1f} pips > limit {max_sp} — skipping"
                )
                return

        # ── 7. Generate signal (with HTF bias) ───────────────────────────────
        log_activity(symbol, f"สแกนคู่เงิน {symbol}...", 'scan')
        htf_bias = self._fetch_htf_bias(symbol)
        signal, atr, last_sh, last_sl = generate_signal(df, cfg, htf_bias=htf_bias)
        regime = detect_regime(df, cfg)

        # ── 8. AI prediction ──────────────────────────────────────────────────
        ai_bias, ai_confidence = self.ai.predict(df)

        # Update terminal with final signal/AI (2b already set preview; this overwrites with final)
        self._collect_terminal_data(symbol, df, signal, atr, ai_bias, ai_confidence, regime)

        self.logger.info(
            f"{symbol}: signal={signal:<4} | ATR={atr:.5f} | "
            f"regime={regime} | AI={ai_bias}({ai_confidence}%)"
        )

        # Log indicators to activity feed
        latest = df.iloc[-1]
        log_activity(
            symbol,
            f"RSI:{float(latest['rsi']):.1f} | "
            f"MACD:{self._terminal.get(symbol,{}).get('macd_status','—')} | "
            f"EMA200:{self._terminal.get(symbol,{}).get('ema200_status','—')}",
            'indicator',
        )
        log_activity(
            symbol,
            f"AI Confidence: {ai_confidence}% — "
            f"{self._terminal.get(symbol,{}).get('ai_label','—')}",
            'ai',
        )

        if signal == 'HOLD':
            log_activity(symbol, "รอจังหวะ — ยังไม่มีสัญญาณ", 'info')
            return

        # ── Direction ban check ───────────────────────────────────────────────
        if self._is_direction_banned(symbol, signal):
            log_activity(
                symbol,
                f"Direction ban: {signal} ถูกแบน — เว้นจากการเทรดทิศนี้",
                'warning',
            )
            return

        log_activity(symbol, f"สัญญาณ {signal} — H4 bias={htf_bias} — รอยืนยัน AI", 'signal')

        if cfg['ai']['enabled'] and self.ai.is_trained:
            # Block ONLY when AI confidently predicts the OPPOSITE direction.
            # Neutral AI = let strategy signal through.
            # Agreeing AI = definitely let through.
            opposite_bias = 'bearish' if signal == 'BUY' else 'bullish'
            if ai_bias == opposite_bias and ai_confidence >= cfg['ai']['min_confidence']:
                self.logger.info(
                    f"{symbol}: AI filter blocked — AI predicts {ai_bias}({ai_confidence}%) "
                    f"against signal={signal}"
                )
                log_activity(
                    symbol,
                    f"AI บล็อก: คาด {ai_bias} {ai_confidence}% สวนทาง {signal}",
                    'warning',
                )
                return

        # ── 9. SL / TP calculation ────────────────────────────────────────────
        sym_info = self.executor.get_symbol_info(symbol)
        if sym_info is None:
            return

        close      = float(df['close'].iloc[-1])
        sl_mult    = cfg['risk']['atr_sl_multiplier']
        tp_mult    = cfg['risk']['atr_tp_multiplier']

        if signal == 'BUY':
            sl_price = close - atr * sl_mult
            tp_price = close + atr * tp_mult
        else:
            sl_price = close + atr * sl_mult
            tp_price = close - atr * tp_mult

        # Enforce minimum absolute SL distance — prevents instant stop-outs
        min_sl_pts = cfg.get('trade_management', {}).get('min_sl_points', 12.0)
        sl_dist    = max(abs(close - sl_price), min_sl_pts)
        if signal == 'BUY':
            sl_price = close - sl_dist
            tp_price = close + sl_dist * cfg['risk']['atr_tp_multiplier'] / cfg['risk']['atr_sl_multiplier']
        else:
            sl_price = close + sl_dist
            tp_price = close - sl_dist * cfg['risk']['atr_tp_multiplier'] / cfg['risk']['atr_sl_multiplier']

        tp_dist = abs(tp_price - close)
        rr      = tp_dist / sl_dist if sl_dist > 0 else 0.0

        if rr < cfg['risk']['min_rr_ratio']:
            self.logger.info(
                f"{symbol}: RR={rr:.2f} < min {cfg['risk']['min_rr_ratio']} — skip"
            )
            return

        # ── 10. Position sizing (with Kelly from trade history) ──────────────
        stats    = get_trade_stats()
        lot_size = self.risk.calculate_lot_size(
            balance, sl_dist, sym_info,
            win_rate = stats.get('win_rate') or None,
            avg_win  = (stats.get('total_profit', 0) / max(1, int(stats.get('win_rate', 0) / 100 * stats.get('total_trades', 1)))) if stats.get('win_rate') else None,
            avg_loss = None,
        )

        # ── 11. Execute ───────────────────────────────────────────────────────
        result = self.executor.place_market_order(
            symbol    = symbol,
            direction = signal,
            lot_size  = lot_size,
            sl_price  = sl_price,
            tp_price  = tp_price,
            magic     = magic,
            comment   = f"AI-Trade|{signal}|{ai_confidence}%",
        )

        if result:
            self._last_trade_ts[symbol] = time.time()   # start cooldown
            log_activity(
                symbol,
                f"ส่งคำสั่ง {signal} @ {result['price']:.5f} | Lot={lot_size:.2f} | "
                f"SL={sl_price:.5f} TP={tp_price:.5f} | H4={htf_bias}",
                'order',
            )
            self._known_tickets.add(result['ticket'])
            self._ticket_direction[result['ticket']]  = signal          # track for direction ban
            self._ticket_confidence[result['ticket']] = ai_confidence   # track for dashboard
            # Store entry features for trade-outcome feedback learning
            self.ai.record_trade_entry(result['ticket'], df, signal)
            insert_trade({
                'ticket':       result['ticket'],
                'symbol':       symbol,
                'direction':    signal,
                'lot_size':     lot_size,
                'entry_price':  result['price'],
                'sl_price':     sl_price,
                'tp_price':     tp_price,
                'open_time':    datetime.now().isoformat(timespec='seconds'),
                'ai_confidence': ai_confidence,
            })

    # ── Closed-position sync ──────────────────────────────────────────────────

    def _sync_closed_positions(self) -> None:
        """Detect positions that were closed (SL/TP hit) and update DB."""
        magic      = self.config['trading']['magic_number']
        now        = datetime.now(timezone.utc)
        from_ts    = (now - timedelta(hours=24)).timestamp()

        deals = self.executor.get_closed_deals(from_ts, now.timestamp(), magic)
        for d in deals:
            if d.order in self._known_tickets:
                close_trade_db(
                    ticket      = d.order,
                    close_price = d.price,
                    close_time  = datetime.fromtimestamp(d.time).isoformat(),
                    profit      = d.profit,
                )
                self.risk.record_trade_result(d.profit)
                # Feed trade outcome back to AI for self-learning
                direction = self._ticket_direction.pop(d.order, '')
                self.ai.label_closed_trade(d.order, d.profit, direction)
                # Update per-direction loss streak for direction ban
                if direction and d.symbol:
                    self._update_direction_streak(d.symbol, direction, d.profit)
                elif direction:
                    # Fallback: apply to all tracked symbols (single-symbol setup)
                    for sym in self.config['trading']['symbols']:
                        self._update_direction_streak(sym, direction, d.profit)
                self._ticket_confidence.pop(d.order, None)
                self._known_tickets.discard(d.order)

    # ── Dashboard state update ────────────────────────────────────────────────

    def _write_ai_insights(self) -> None:
        """Write AI prediction breakdown + learning stats to JSON for dashboard."""
        import json
        from pathlib import Path

        insights_path = Path('data/ai_insights.json')
        learn_path    = Path('data/learning_stats.json')
        Path('data').mkdir(exist_ok=True)

        try:
            insights = self.ai.get_ai_insights()
            tmp = insights_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(insights))
            tmp.replace(insights_path)
        except Exception as exc:
            self.logger.debug(f"AI insights write failed: {exc}")

        try:
            # Build learning stats
            rl_status  = {}
            mem_stats  = {}
            online_stats = {}

            if hasattr(self.ai, '_rl') and self.ai._rl is not None:
                rl_status = self.ai._rl.status()

            if hasattr(self.ai, '_memory') and self.ai._memory is not None:
                mem_stats = self.ai._memory.stats()

            if hasattr(self.ai, '_online') and self.ai._online is not None:
                ol = self.ai._online
                online_stats = {
                    'n_upd':    ol._n_upd,
                    'accuracy': round(ol.recent_accuracy, 4),
                    'weight':   round(ol.weight, 4),
                }

            interval_min = self.config['ai'].get('retrain_interval', 360)
            elapsed_min  = (time.time() - self._ai_last_train_ts) / 60
            retrain_in   = f"{max(0, interval_min - elapsed_min):.0f} min"

            learn = {
                'rl':       rl_status,
                'memory':   mem_stats,
                'online':   online_stats,
                'retrain_in': retrain_in,
            }
            tmp2 = learn_path.with_suffix('.tmp')
            tmp2.write_text(json.dumps(learn))
            tmp2.replace(learn_path)
        except Exception as exc:
            self.logger.debug(f"Learning stats write failed: {exc}")

    def _update_dashboard_state(self, account_info) -> None:
        open_pos = self.executor.get_all_open_positions()
        positions_data = []
        for p in open_pos:
            positions_data.append({
                'ticket':        p.ticket,
                'symbol':        p.symbol,
                'direction':     'BUY' if p.type == 0 else 'SELL',
                'lot_size':      p.volume,
                'entry_price':   p.price_open,
                'current_price': p.price_current,
                'sl_price':      p.sl,
                'tp_price':      p.tp,
                'profit':        round(p.profit, 2),
                'ai_confidence': self._ticket_confidence.get(p.ticket, 0),
            })

        # Build symbols dict for dashboard (indicator values)
        symbols_dash = {}
        for sym, term in self._terminal.items():
            symbols_dash[sym] = {
                'close':        term.get('price', 0),
                'spread':       term.get('spread_pips', 0),
                'adx':          term.get('adx', 0),
                'rsi':          term.get('rsi', 50),
                'macd_hist':    term.get('macd_hist', 0),
                'ema50':        term.get('ema50',     0),
                'ema200':       term.get('ema200',    0),
                'stoch_k':      term.get('stoch_k',   0),
                'ai_bias':      term.get('ai_bias', 'neutral'),
                'ai_confidence': term.get('ai_confidence', 0),
                'regime':       term.get('regime', 'TREND'),
                'signal':       term.get('signal', 'HOLD'),
                'atr':          term.get('atr', 0),
            }

        stats = get_trade_stats()

        write_state({
            'timestamp':           datetime.now().isoformat(timespec='seconds'),
            'balance':             account_info.balance,
            'equity':              account_info.equity,
            'initial_balance':     self.risk.initial_balance,
            'peak_balance':        self.risk.peak_balance,
            'margin':              account_info.margin,
            'free_margin':         account_info.margin_free,
            'terminal':            self._terminal,
            'symbols':             symbols_dash,
            'open_trades':         positions_data,
            'drawdown_pct':        round(
                self.risk.current_drawdown(account_info.balance) * 100, 2
            ),
            'daily_pnl':           round(
                self.risk.daily_pnl(account_info.balance), 2
            ),
            'daily_start_balance': self.risk.daily_start_balance,
            'stats': {
                'total_trades': stats.get('total_trades', 0),
                'wins':         stats.get('wins', 0),
                'losses':       stats.get('losses', 0),
                'win_rate':     round(stats.get('win_rate', 0) / 100, 4),
                'today_pnl':    round(self.risk.daily_pnl(account_info.balance), 2),
                'total_profit': stats.get('total_profit', 0),
            },
        })

        # Write AI insights + learning stats separately
        self._write_ai_insights()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.running = True
        symbols      = self.config['trading']['symbols']

        self.logger.info(f"Symbols  : {symbols}")
        self.logger.info(f"Timeframe: {self.config['trading']['timeframe']}")
        self.logger.info(f"Cycle    : {self.CYCLE_SECONDS}s")

        while self.running:
            try:
                # Account heartbeat
                account_info = self.executor.get_account_info()
                if account_info is None:
                    self.logger.error("Cannot reach account info — retry in 30s")
                    time.sleep(30)
                    continue

                balance = account_info.balance
                self.risk.update_balance(balance)
                record_equity(balance, account_info.equity)

                # Global drawdown halt
                if not self.risk.check_drawdown_limit(balance):
                    self.logger.critical(
                        "DRAWDOWN LIMIT REACHED — engine stopped. "
                        "Review account before restarting."
                    )
                    self.running = False
                    break

                # Process each symbol
                for symbol in symbols:
                    try:
                        self._process_symbol(symbol, balance)
                    except Exception as exc:
                        self.logger.error(
                            f"Error processing {symbol}: {exc}", exc_info=True
                        )

                # Active trade management (break-even / trailing / partial close)
                all_open = self.executor.get_all_open_positions()
                self.trade_manager.manage(all_open)

                # Housekeeping
                self._sync_closed_positions()
                self._update_dashboard_state(account_info)

                dd_pct = self.risk.current_drawdown(balance) * 100
                self.logger.info(
                    f"Cycle OK | balance={balance:.2f} | "
                    f"equity={account_info.equity:.2f} | "
                    f"drawdown={dd_pct:.2f}% | "
                    f"open={len(self.executor.get_all_open_positions())}"
                )

                time.sleep(self.CYCLE_SECONDS)

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt — shutting down cleanly")
                break
            except Exception as exc:
                self.logger.error(f"Unhandled main-loop error: {exc}", exc_info=True)
                time.sleep(30)

        self.executor.disconnect()
        self.logger.info("Engine stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    engine = TradingEngine()
    if engine.connect():
        engine.run()
    else:
        sys.exit(1)
