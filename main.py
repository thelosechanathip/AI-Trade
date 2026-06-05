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
import json
import logging
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yaml

from utils import (
    setup_logging, compute_indicators, init_db,
    insert_trade, close_trade_db, record_equity,
    write_state, get_trade_stats, get_today_trade_stats, log_activity,
    get_open_trades_from_db, compute_context_penalty,
)
from strategy          import generate_signal, detect_regime
from risk              import RiskManager
from execution_mt5     import MT5Executor
from ai_model          import AIModel
from trade_manager     import TradeManager
from market_brain         import MarketBrain, BrainContext, BrainDecision
from exit_intelligence    import ExitIntelligence
from brain_memory         import BrainMemory
from confidence_bootstrap import ConfidenceBootstrap
from cold_start_manager   import ColdStartManager
from signal_stability     import SignalStabilityTracker
from noise_filter         import NoiseFilter
from anti_chase           import AntiChaseEngine
from context_persistence  import ContextPersistenceEngine
from strategy_versioning  import StrategyVersionManager
from committee_guard      import InvestmentCommitteeGuard
from execution_controls   import ExecutionControlStore, build_order_plan


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


# ── Lightweight MI narrative proxy built from terminal dict ──────────────────

class _TerminalNarrative:
    """
    Wraps the per-symbol terminal dict so ExitIntelligence can read
    MI narrative fields without requiring the full MarketNarrative object.
    """
    def __init__(self, term: dict):
        signals = term.get('mi_signals', []) or []
        self.regime               = term.get('mi_regime',     '')
        self.narrative            = term.get('mi_narrative',  '')
        self.signals_active       = signals
        self.setup_quality        = float(term.get('mi_quality', 0.5))
        self.block_buy            = bool(term.get('mi_block_buy',  False))
        self.block_sell           = bool(term.get('mi_block_sell', False))
        self.block_reason         = ''
        self.confidence_adjustment= 0.0
        self.uncertainty          = 0.0
        # Populate signal flags from active signal list
        s = set(signals)
        self.rsi_divergence_bull  = 'RSI_DIV_BULL'  in s
        self.rsi_divergence_bear  = 'RSI_DIV_BEAR'  in s
        self.macd_divergence_bull = 'MACD_DIV_BULL' in s
        self.macd_divergence_bear = 'MACD_DIV_BEAR' in s
        self.displacement_bull    = 'DISP_BULL'     in s
        self.displacement_bear    = 'DISP_BEAR'     in s
        self.liquidity_sweep_bull = 'LIQ_SWEEP_BULL' in s
        self.liquidity_sweep_bear = 'LIQ_SWEEP_BEAR' in s
        self.volatility_climax    = 'VOL_CLIMAX'    in s
        self.reversal_detected    = 'REVERSAL_UP' in s or 'REVERSAL_DOWN' in s
        self.reversal_direction   = ('UP' if 'REVERSAL_UP' in s
                                     else ('DOWN' if 'REVERSAL_DOWN' in s else ''))
        self.bos_choch: dict = {
            'bos_bull':   'BOS_BULL'   in s,
            'bos_bear':   'BOS_BEAR'   in s,
            'choch_bull': 'CHOCH_BULL' in s,
            'choch_bear': 'CHOCH_BEAR' in s,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class TradingEngine:
    CYCLE_SECONDS = 60   # main loop sleep interval

    def __init__(self):
        self.config = load_config()
        self.logger = setup_logging(self.config)
        init_db()

        self.risk          = RiskManager(self.config)
        self.executor      = MT5Executor(self.config)
        self.ai            = AIModel(self.config)
        self.trade_manager = TradeManager(self.config)
        self.brain_memory         = BrainMemory()
        self.market_brain         = MarketBrain(self.config, self.brain_memory)
        self.exit_intel           = ExitIntelligence(self.config)
        self.confidence_bootstrap = ConfidenceBootstrap()
        self.cold_start_manager   = ColdStartManager(self.config)
        # Stability layer
        entry_cfg = self.config.get('entry_filters', {})
        self.signal_stability     = SignalStabilityTracker(
            required_cycles = int(entry_cfg.get('signal_stability_cycles', 2)),
            window_size     = int(entry_cfg.get('signal_stability_window', 5)),
        )
        self.noise_filter         = NoiseFilter(
            threshold=float(entry_cfg.get('noise_threshold', 0.60))
        )
        self.anti_chase           = AntiChaseEngine(
            chase_threshold=float(entry_cfg.get('chase_threshold', 0.60))
        )
        self.context_persistence  = ContextPersistenceEngine(
            ema_alpha=0.20, flip_threshold=0.72, min_cycles_before_flip=3
        )
        self.committee_guard      = InvestmentCommitteeGuard(self.config)
        self.execution_controls   = ExecutionControlStore(self.config)

        self.running              = False
        self._ai_last_train_ts    = 0.0
        self._known_tickets: set  = set()            # tickets we opened this session
        self._terminal: dict      = {}               # per-symbol terminal data
        self._last_trade_ts: dict = {}               # symbol -> timestamp of last trade open
        self._ticket_direction: dict   = {}            # ticket -> 'BUY'|'SELL'
        self._ticket_confidence: dict  = {}            # ticket -> ai_confidence int
        self._ticket_brain: dict       = {}            # ticket -> BrainDecision
        self._ticket_committee: dict   = {}            # ticket -> CommitteeVerdict
        self._ticket_entry_bar: dict   = {}            # ticket -> bar count at entry
        self._account_margin_level: float = 0.0
        # direction_ban: symbol -> {'direction': 'BUY'|'SELL', 'until': float}
        self._direction_ban: dict = {}
        # per-symbol consecutive losses per direction: symbol -> {'BUY': int, 'SELL': int}
        self._dir_loss_streak: dict = {}
        # symbol -> timestamp of last loss (any direction) — recent-loss penalty
        self._last_loss_ts: dict = {}
        # symbol -> {'direction': 'BUY'|'SELL', 'until': float} — 30min cooldown for
        # the OPPOSITE direction after a loss (prevents immediate flip trading)
        self._opposite_cooldown: dict = {}
        self._guard_state_file = Path('data/trade_guard_state.json')
        self._load_trade_guard_state()

        # Strategy version manager — seeds v1 from config on first run
        self.version_manager = StrategyVersionManager(self.config)
        self.logger.info(
            f"StrategyVersioning: current version = "
            f"v{self.version_manager.get_current_version().get('version_id', '?')}"
        )

        self._restore_open_trades()

        self.logger.info("=" * 60)
        self.logger.info("  AI-Trade Engine  |  starting up")
        self.logger.info("=" * 60)

    # ── Session state restore ─────────────────────────────────────────────────

    def _load_trade_guard_state(self) -> None:
        """Restore cooldown/ban state so a restart cannot erase loss protection."""
        if not self._guard_state_file.exists():
            return
        try:
            state = json.loads(self._guard_state_file.read_text(encoding='utf-8'))
            now = time.time()
            self._last_trade_ts = {
                str(k): float(v) for k, v in state.get('last_trade_ts', {}).items()
            }
            self._last_loss_ts = {
                str(k): float(v) for k, v in state.get('last_loss_ts', {}).items()
            }
            self._dir_loss_streak = {
                str(sym): {
                    'BUY': int(vals.get('BUY', 0)),
                    'SELL': int(vals.get('SELL', 0)),
                }
                for sym, vals in state.get('dir_loss_streak', {}).items()
                if isinstance(vals, dict)
            }
            self._direction_ban = {
                str(sym): entry
                for sym, entry in state.get('direction_ban', {}).items()
                if isinstance(entry, dict) and float(entry.get('until', 0)) > now
            }
            self._opposite_cooldown = {
                str(sym): entry
                for sym, entry in state.get('opposite_cooldown', {}).items()
                if isinstance(entry, dict) and float(entry.get('until', 0)) > now
            }
            self.logger.info("Restored trade guard state from disk")
        except Exception as exc:
            self.logger.warning(f"Could not restore trade guard state: {exc}")

    def _save_trade_guard_state(self) -> None:
        """Persist entry cooldowns and direction loss guards."""
        try:
            self._guard_state_file.parent.mkdir(parents=True, exist_ok=True)
            self._guard_state_file.write_text(json.dumps({
                'last_trade_ts': self._last_trade_ts,
                'last_loss_ts': self._last_loss_ts,
                'dir_loss_streak': self._dir_loss_streak,
                'direction_ban': self._direction_ban,
                'opposite_cooldown': self._opposite_cooldown,
            }, indent=2), encoding='utf-8')
        except Exception as exc:
            self.logger.debug(f"Could not save trade guard state: {exc}")

    def _restore_open_trades(self) -> None:
        """Re-populate _known_tickets from DB open trades so restarts don't lose tracking."""
        open_trades = get_open_trades_from_db()
        for t in open_trades:
            ticket = t['ticket']
            self._known_tickets.add(ticket)
            if t.get('direction'):
                self._ticket_direction[ticket] = t['direction']
            if t.get('ai_confidence') is not None:
                self._ticket_confidence[ticket] = int(t['ai_confidence'])
        if open_trades:
            self.logger.info(
                f"Restored {len(open_trades)} open trade(s) from DB: "
                f"{[t['ticket'] for t in open_trades]}"
            )

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

    @staticmethod
    def _scale_lot_to_symbol(lot_size: float, scale: float, symbol_info) -> float:
        """Scale lot size while respecting broker min/max/step constraints."""
        scale = max(0.0, min(1.0, float(scale)))
        raw = max(0.0, float(lot_size) * scale)
        try:
            step = float(symbol_info.volume_step)
            vol_min = float(symbol_info.volume_min)
            vol_max = float(symbol_info.volume_max)
            if step <= 0:
                return round(raw, 2)
            scaled = (raw // step) * step
            if scaled < vol_min:
                return 0.0
            scaled = min(vol_max, scaled)
            return round(scaled, 2)
        except Exception:
            return round(raw, 2)

    def _portfolio_open_risk(self, positions: list) -> float:
        """Estimate cash-at-stop for strategy positions; fail closed if unverifiable."""
        total = 0.0
        for pos in positions:
            sl_price = float(getattr(pos, 'sl', 0.0) or 0.0)
            entry_price = float(getattr(pos, 'price_open', 0.0) or 0.0)
            if sl_price <= 0 or entry_price <= 0:
                return float('inf')
            symbol = self.executor.canonical_symbol(str(getattr(pos, 'symbol', '')))
            symbol_info = self.executor.get_symbol_info(symbol)
            amount = self.risk.estimate_risk_amount(
                float(getattr(pos, 'volume', 0.0) or 0.0),
                abs(entry_price - sl_price),
                symbol_info,
            )
            if not math.isfinite(amount):
                return float('inf')
            total += amount
        return total

    def _cap_lot_to_portfolio_risk(
        self,
        lot_size: float,
        equity: float,
        sl_distance: float,
        symbol_info,
        positions: list,
    ) -> float:
        """Cap aggregate new-entry lot so total open cash-at-stop stays bounded."""
        max_fraction = float(self.config['risk'].get('max_total_open_risk', 0.03))
        max_amount = max(0.0, float(equity)) * max_fraction
        open_amount = self._portfolio_open_risk(positions)
        per_lot_amount = self.risk.estimate_risk_amount(1.0, sl_distance, symbol_info)
        if (
            max_amount <= 0
            or not math.isfinite(open_amount)
            or not math.isfinite(per_lot_amount)
            or per_lot_amount <= 0
        ):
            return 0.0
        remaining = max(0.0, max_amount - open_amount)
        risk_cap_lot = remaining / per_lot_amount
        if risk_cap_lot >= lot_size:
            return lot_size
        return self._scale_lot_to_symbol(
            lot_size,
            risk_cap_lot / max(float(lot_size), 1e-12),
            symbol_info,
        )

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

    def _fetch_htf_bias(self, symbol: str) -> tuple:
        """
        Three-level Higher Timeframe trend filter.

        Levels:
          H4  EMA200 — intermediate trend (200 × 4h ≈ 33 trading days)
          D1  EMA50  — macro trend direction
          H1  EMA50  — intraday momentum bridge between M15 and H4

        Voting logic:
          • All three must not conflict to return a directional bias.
          • Any level alone returning NEUTRAL makes the overall bias NEUTRAL.
          • H4 and D1 conflicting → NEUTRAL (transitioning market).
          • H1 used as a confirmation bridge: if H4+D1 agree but H1 opposes,
            downgrade confidence (still return direction but low strength).

        Returns:
          (bias: str, strength: float)
          bias     — 'BUY' | 'SELL' | 'NEUTRAL'
          strength — 0.0–1.0  (how strongly all levels agree)
        """
        htf_cfg = self.config.get('htf_filter', {})
        if not htf_cfg.get('enabled', False):
            return 'NEUTRAL', 0.0

        tf    = htf_cfg.get('timeframe', 'H4')
        bars  = htf_cfg.get('bars', 250)
        ema_p = htf_cfg.get('ema_period', 200)
        zone  = htf_cfg.get('neutral_zone', 0.005)   # 0.5% ≈ $22 buffer

        try:
            # ── Level 1: H4 EMA200 ───────────────────────────────────────────
            rates = self.executor.get_ohlcv(symbol, tf, bars)
            if rates is None or len(rates) < ema_p + 10:
                return 'NEUTRAL', 0.0

            df_h4  = rates_to_df(rates)
            closes = df_h4['close']
            ema200_h4 = float(closes.ewm(span=ema_p, adjust=False).mean().iloc[-1])
            price     = float(closes.iloc[-1])
            h4_dist   = (price - ema200_h4) / max(ema200_h4, 1.0)

            if price > ema200_h4 * (1 + zone):
                h4_bias = 'BUY'
            elif price < ema200_h4 * (1 - zone):
                h4_bias = 'SELL'
            else:
                h4_bias = 'NEUTRAL'

            # ── Level 2: D1 EMA50 (macro) ────────────────────────────────────
            d1_rates  = self.executor.get_ohlcv(symbol, 'D1', 80)
            d1_bias   = 'NEUTRAL'
            d1_ema50  = None
            if d1_rates is not None and len(d1_rates) >= 55:
                df_d1    = rates_to_df(d1_rates)
                d1_close = float(df_d1['close'].iloc[-1])
                d1_ema50 = float(df_d1['close'].ewm(span=50, adjust=False).mean().iloc[-1])
                if d1_close > d1_ema50 * 1.001:
                    d1_bias = 'BUY'
                elif d1_close < d1_ema50 * 0.999:
                    d1_bias = 'SELL'

            # ── Level 3: H1 EMA50 (intraday bridge) ──────────────────────────
            h1_rates = self.executor.get_ohlcv(symbol, 'H1', 80)
            h1_bias  = 'NEUTRAL'
            h1_ema50 = None
            if h1_rates is not None and len(h1_rates) >= 55:
                df_h1    = rates_to_df(h1_rates)
                h1_close = float(df_h1['close'].iloc[-1])
                h1_ema50 = float(df_h1['close'].ewm(span=50, adjust=False).mean().iloc[-1])
                if h1_close > h1_ema50 * 1.0005:
                    h1_bias = 'BUY'
                elif h1_close < h1_ema50 * 0.9995:
                    h1_bias = 'SELL'

            self.logger.debug(
                f"HTF({symbol}): H4={h4_bias}(dist={h4_dist:+.3%}) "
                f"D1={d1_bias}(ema50={d1_ema50:.1f} if d1_ema50 else 'N/A') "
                f"H1={h1_bias}(ema50={h1_ema50:.1f} if h1_ema50 else 'N/A')"
            )

            # ── Combine: H4+D1 are the primary signals ────────────────────────
            # If H4 is in the neutral zone, we cannot trade — price is at EMA retest
            if h4_bias == 'NEUTRAL':
                return 'NEUTRAL', 0.0

            # H4 and D1 conflict → transitioning market → stay flat
            if d1_bias != 'NEUTRAL' and d1_bias != h4_bias:
                return 'NEUTRAL', 0.0

            # Both H4 and D1 agree (or D1 is NEUTRAL) → direction is h4_bias
            bias = h4_bias

            # ── Calculate trend strength (0.0–1.0) ───────────────────────────
            # Score: H4 dist from EMA200, D1 agreement, H1 confirmation
            dist_score = min(abs(h4_dist) / 0.03, 1.0)   # 3% = max strength
            d1_score   = 1.0 if (d1_bias == bias) else 0.5
            h1_score   = 1.0 if (h1_bias == bias) else (0.5 if h1_bias == 'NEUTRAL' else 0.2)
            strength   = float((dist_score * 0.5 + d1_score * 0.3 + h1_score * 0.2))

            # If H1 strongly opposes H4+D1, downgrade but don't block
            if h1_bias != 'NEUTRAL' and h1_bias != bias:
                strength *= 0.6   # H1 momentum hasn't turned yet — lower confidence

            return bias, round(min(strength, 1.0), 3)

        except Exception as exc:
            self.logger.debug(f"_fetch_htf_bias({symbol}): {exc}")
            return 'NEUTRAL', 0.0

    # ── Direction ban ─────────────────────────────────────────────────────────

    def _is_direction_banned(self, symbol: str, direction: str) -> bool:
        """Return True if direction is currently under a soft or hard ban."""
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
            self._save_trade_guard_state()
            return False

        remaining  = (entry['until'] - time.time()) / 3600
        ban_type = entry.get('type', 'hard')
        if ban_type == 'soft':
            self.logger.info(
                f"{symbol}: direction {direction} SOFT guard active "
                f"for {remaining:.1f}h more (lot/score penalties still apply)"
            )
            return False

        self.logger.info(
            f"{symbol}: direction {direction} HARD BANNED "
            f"for {remaining:.1f}h more"
        )
        return True

    def _get_direction_lot_scale(self, symbol: str, direction: str) -> float:
        """
        Return lot size multiplier based on current direction penalty state.

        Soft ban → 0.5× (half size, still allowed to trade after ban expires)
        No ban   → 1.0×
        Hard ban → caller should never reach here (blocked by _is_direction_banned)
        """
        streak = self._dir_loss_streak.get(symbol, {}).get(direction, 0)
        ban_cfg = self.config.get('direction_ban', {})
        soft_threshold = max(1, ban_cfg.get('max_same_dir_losses', 2) - 1)
        if streak >= soft_threshold:
            return 0.5
        return 1.0

    def _update_direction_streak(self, symbol: str, direction: str, profit: float) -> None:
        """
        Progressive direction penalty system.

        Loss count → Action:
          1 loss  → warning only (no ban)
          2 losses → short ban (ban_hours_soft, default 2h) + lot scale 0.5×
          3+ losses → hard ban (ban_hours_hard, default 6h)

        A win resets the streak AND any soft lot scaling for that direction.
        """
        ban_cfg = self.config.get('direction_ban', {})
        if not ban_cfg.get('enabled', False):
            return

        if symbol not in self._dir_loss_streak:
            self._dir_loss_streak[symbol] = {'BUY': 0, 'SELL': 0}

        if profit < 0:
            self._dir_loss_streak[symbol][direction] += 1
            streak = self._dir_loss_streak[symbol][direction]

            # Reset opposite direction's streak on a loss (focus on this direction)
            opp = 'SELL' if direction == 'BUY' else 'BUY'
            self._dir_loss_streak[symbol][opp] = 0

            hard_losses = ban_cfg.get('max_same_dir_losses', 2)       # default 2
            soft_losses = max(1, hard_losses - 1)                      # 1 before hard ban
            ban_hard    = ban_cfg.get('ban_hours', 4)
            ban_soft    = ban_cfg.get('ban_hours_soft', 2)

            if streak >= hard_losses:
                # Hard ban: no trades in this direction for ban_hard hours
                self._direction_ban[symbol] = {
                    'direction': direction,
                    'until':     time.time() + ban_hard * 3600,
                    'type':      'hard',
                }
                self._dir_loss_streak[symbol][direction] = 0
                self.logger.warning(
                    f"{symbol}: {direction} HARD BAN {ban_hard}h "
                    f"after {streak} consecutive losses"
                )
                log_activity(
                    symbol,
                    f"HARD BAN {direction} {ban_hard}h — แพ้ซ้อน {streak} ครั้ง",
                    'warning',
                )
            elif streak >= soft_losses:
                # Soft ban: shorter cooldown + lot halving flagged via direction_ban
                self._direction_ban[symbol] = {
                    'direction': direction,
                    'until':     time.time() + ban_soft * 3600,
                    'type':      'soft',
                }
                self.logger.info(
                    f"{symbol}: {direction} SOFT BAN {ban_soft}h after {streak} loss"
                )
                log_activity(
                    symbol,
                    f"Soft ban {direction} {ban_soft}h — แพ้ {streak} ครั้งติด",
                    'warning',
                )
        else:
            # Win: fully reset this direction's penalty state
            if symbol in self._dir_loss_streak:
                self._dir_loss_streak[symbol][direction] = 0
            if (symbol in self._direction_ban
                    and self._direction_ban[symbol].get('direction') == direction):
                del self._direction_ban[symbol]
                self.logger.info(
                    f"{symbol}: {direction} direction ban lifted after a winning trade"
                )
        self._save_trade_guard_state()

    # ── Session classifier ────────────────────────────────────────────────────

    def _get_current_session(self) -> str:
        """Return a session label for the current UTC time."""
        sess = self.config.get('sessions', {})
        now  = datetime.utcnow().strftime('%H:%M')

        def in_range(start: str, end: str, t: str) -> bool:
            return start <= t < end

        london_start = sess.get('london', {}).get('start', '07:00')
        london_end   = sess.get('london', {}).get('end',   '16:00')
        ny_start     = sess.get('new_york', {}).get('start', '12:00')
        ny_end       = sess.get('new_york', {}).get('end',   '21:00')

        in_london = in_range(london_start, london_end, now)
        in_ny     = in_range(ny_start,     ny_end,     now)

        if in_london and in_ny:
            return 'LONDON_NY'
        if in_london:
            return 'LONDON'
        if in_ny:
            return 'NEW_YORK'
        if '00:00' <= now < '04:00':
            return 'SYDNEY'
        if '02:00' <= now < '09:00':
            return 'TOKYO'
        return 'OFF_HOURS'

    def _is_opposite_dir_cooled(self, symbol: str, direction: str) -> bool:
        """Return True if the OPPOSITE-direction cooldown is still active."""
        entry = self._opposite_cooldown.get(symbol)
        if not entry:
            return False
        if entry['direction'] != direction:
            return False
        if time.time() > entry['until']:
            del self._opposite_cooldown[symbol]
            self._save_trade_guard_state()
            return False
        remaining_min = (entry['until'] - time.time()) / 60
        self.logger.info(
            f"{symbol}: opposite-direction cooldown for {direction} — "
            f"{remaining_min:.0f} min remaining after last loss"
        )
        return True

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
        _sig_preview, _atr_preview, _, _, _ = generate_signal(df, cfg)
        _regime_preview = detect_regime(df, cfg)
        _ai_bias_p, _ai_conf_p = self.ai.predict(df)
        self._collect_terminal_data(
            symbol, df, _sig_preview, _atr_preview,
            _ai_bias_p, _ai_conf_p, _regime_preview,
        )
        controls = self.execution_controls.get()
        if symbol in self._terminal:
            self._terminal[symbol]['execution_controls_revision'] = controls['revision']
        if not controls.get('trading_enabled', False):
            self.logger.info(f"{symbol}: operator control has disabled new entries")
            return
        min_margin_level = float(cfg['risk'].get('min_margin_level', 300.0))
        if (
            self._account_margin_level > 0
            and self._account_margin_level < min_margin_level
        ):
            self.logger.warning(
                f"{symbol}: margin level {self._account_margin_level:.1f}% below "
                f"minimum {min_margin_level:.1f}%"
            )
            return
        # Fetch HTF bias early so we can pass it to the preview signal too
        # (actual signal uses it below after all risk checks pass)

        # ── 3. Multi-trade check: allow up to max_concurrent_trades per symbol ──
        existing_sym_pos = self.executor.get_positions_for_symbol(symbol, magic)
        sym_max = cfg['risk']['max_concurrent_trades']
        if len(existing_sym_pos) >= sym_max:
            self.logger.debug(
                f"{symbol}: {len(existing_sym_pos)}/{sym_max} trades open — skip"
            )
            return

        # ── 4. Per-symbol risk checks ─────────────────────────────────────────
        if not self.risk.check_daily_loss_limit(balance):
            return
        if not self.risk.check_weekly_loss_limit(balance):
            log_activity(symbol, "Weekly loss limit reached — หยุดเทรดสัปดาห์นี้", 'warning')
            return
        if not self.risk.check_drawdown_limit(balance):
            self.running = False
            return
        if not self.risk.check_adaptive_cooldown():
            log_activity(symbol, "Adaptive cooldown — พักการเทรดหลังขาดทุนต่อเนื่อง", 'warning')
            return

        all_positions = self.executor.get_magic_positions(magic)
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

        # ── 7. Generate signal (with HTF bias + strength) ────────────────────
        log_activity(symbol, f"สแกนคู่เงิน {symbol}...", 'scan')
        htf_bias, htf_strength = self._fetch_htf_bias(symbol)
        signal, atr, last_sh, last_sl, mi_narrative = generate_signal(
            df, cfg,
            htf_bias     = htf_bias,
            htf_strength = htf_strength,
        )
        regime = detect_regime(df, cfg)

        # ── 8. AI prediction ──────────────────────────────────────────────────
        ai_bias, ai_confidence = self.ai.predict(df)

        # Update terminal with final signal/AI (2b set preview; this overwrites with final)
        self._collect_terminal_data(symbol, df, signal, atr, ai_bias, ai_confidence, regime)
        # Append MI data to terminal dict (non-breaking addition)
        if symbol in self._terminal:
            self._terminal[symbol]['mi_regime']    = mi_narrative.regime
            self._terminal[symbol]['mi_narrative'] = mi_narrative.narrative
            self._terminal[symbol]['mi_signals']   = mi_narrative.signals_active
            self._terminal[symbol]['mi_quality']   = round(mi_narrative.setup_quality, 2)
            self._terminal[symbol]['mi_block_buy']  = mi_narrative.block_buy
            self._terminal[symbol]['mi_block_sell'] = mi_narrative.block_sell

        self.logger.info(
            f"{symbol}: signal={signal:<4} | ATR={atr:.5f} | "
            f"regime={regime} | MI={mi_narrative.regime} | "
            f"HTF={htf_bias}({htf_strength:.2f}) | "
            f"AI={ai_bias}({ai_confidence}%)"
        )
        if mi_narrative.signals_active:
            self.logger.info(
                f"{symbol}: MI signals={mi_narrative.signals_active} "
                f"quality={mi_narrative.setup_quality:.2f} "
                f"block=[buy={mi_narrative.block_buy},sell={mi_narrative.block_sell}]"
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

        # ── 7b. Signal Stability: record current signal each cycle ───────────
        # Record BEFORE Brain so even HOLD cycles count toward stability tracking.
        self.signal_stability.record(symbol, signal)

        # ── 8b. Market Brain — Context Analysis ──────────────────────────────
        # Architecture: Rule Engine = Primary Decision Layer
        #               Brain      = Context Modifier + Risk Intelligence
        #
        # The Brain OVERRIDES strategy only at autonomy level ≥ 3.
        # At levels 0-2, it adjusts risk/sizing without blocking valid entries.
        # Emergency block fires at ALL levels (only for extreme rev+uncertainty).

        autonomy_level = self.cold_start_manager.current_level
        latest_row     = df.iloc[-1]
        adx_val  = float(latest_row.get('adx', 0.0)) if 'adx' in df.columns else 0.0
        rsi_val  = float(latest_row['rsi']) if 'rsi' in df.columns else 50.0
        mh_val   = float(latest_row.get('macd_hist', 0.0)) if 'macd_hist' in df.columns else 0.0
        stk_val  = float(latest_row.get('stoch_k', 50.0)) if 'stoch_k' in df.columns else 50.0

        ctx = BrainContext(
            df            = df,
            mi_narrative  = mi_narrative,
            htf_bias      = htf_bias,
            htf_strength  = htf_strength,
            adx           = adx_val,
            rsi           = rsi_val,
            macd_hist     = mh_val,
            stoch_k       = stk_val,
            regime        = regime,
            ema_trend     = signal,
            ai_bias       = ai_bias,
            ai_confidence = float(ai_confidence),
            rule_signal   = signal,
            symbol        = symbol,
        )

        open_count     = len(existing_sym_pos)
        brain_decision = self.market_brain.decide(
            ctx,
            open_trade_count = open_count,
            max_trades       = cfg['risk']['max_concurrent_trades'],
            autonomy_level   = autonomy_level,
        )

        self.logger.info(
            f"{symbol}: Brain={brain_decision.decision} "
            f"conf={brain_decision.confidence:.0%} "
            f"unc={brain_decision.uncertainty:.0%} "
            f"quality={brain_decision.entry_quality:.0%} "
            f"rev={brain_decision.reversal_probability:.0%} "
            f"L{autonomy_level}({self.cold_start_manager.level_name})"
        )

        try:
            self.brain_memory.record_decision(brain_decision, symbol)
        except Exception:
            pass

        # ── Emergency block — respected at ALL autonomy levels ────────────────
        if brain_decision.emergency_block:
            reason = (brain_decision.hold_reasons[-1]
                      if brain_decision.hold_reasons else "extreme reversal + uncertainty")
            log_activity(symbol, f"Brain emergency block: {reason}", 'warning')
            return

        # ── L3-4: Brain is primary — use its decision directly ────────────────
        if autonomy_level >= 3:
            signal = brain_decision.decision
            if signal == 'HOLD':
                hold_summary = '; '.join(brain_decision.hold_reasons[:2]) or 'no setup'
                log_activity(symbol, f"Brain HOLD (L{autonomy_level}): {hold_summary}", 'info')
                return

        # ── L0-2: Strategy is primary — Brain is context advisor ─────────────
        # Respect the strategy's HOLD signal.
        if signal == 'HOLD':
            log_activity(symbol, "รอจังหวะ — ยังไม่มีสัญญาณ", 'info')
            return

        ai_signal = {
            'bullish': 'BUY',
            'bearish': 'SELL',
        }.get(str(ai_bias).lower())
        ai_min_conf = float(cfg.get('ai', {}).get('min_confidence', 52))
        ai_conflict_block = float(cfg.get('ai', {}).get('conflict_block_confidence', 75))
        if ai_signal and ai_signal != signal and ai_confidence >= ai_conflict_block:
            log_activity(
                symbol,
                f"AI conflict: rule={signal} แต่ AI={ai_signal} {ai_confidence}% — skip",
                'warning',
            )
            return
        ai_alignment_bonus = (
            0.03 if ai_signal == signal and ai_confidence >= ai_min_conf else 0.0
        )

        # ── Bootstrap Confidence + Final Trade Score ──────────────────────────
        # Blends bootstrap (technical) confidence with Brain (AI) confidence
        # to produce a single quality gate that works even during cold-start.
        bootstrap = self.confidence_bootstrap.compute(
            signal       = signal,
            htf_bias     = htf_bias,
            htf_strength = htf_strength,
            adx          = adx_val,
            rsi          = rsi_val,
            macd_hist    = mh_val,
            stoch_k      = stk_val,
            regime       = regime,
            mi_narrative = mi_narrative,
        )

        bw               = self.cold_start_manager.get_bootstrap_weight()
        effective_ai     = bootstrap.normalized * bw + brain_decision.confidence * (1.0 - bw)
        setup_quality    = float(getattr(mi_narrative, 'setup_quality', 0.5))
        final_score      = min(1.0, setup_quality * 0.60 + effective_ai * 0.40 + ai_alignment_bonus)
        min_score        = self.cold_start_manager.get_min_score_threshold()

        # ── Same-direction loss penalty ───────────────────────────────────────
        # Each loss in the same direction raises the quality bar (0.06 per loss),
        # making the system more selective before the hard direction ban fires.
        dir_streak   = self._dir_loss_streak.get(symbol, {}).get(signal, 0)
        dir_penalty  = dir_streak * 0.06
        if dir_penalty > 0:
            self.logger.info(
                f"{symbol}: same-direction loss penalty -{dir_penalty:.0%} "
                f"(streak={dir_streak})"
            )
            final_score = max(0.0, final_score - dir_penalty)

        # ── Recent loss penalty ───────────────────────────────────────────────
        # If a loss occurred recently on this symbol, apply a small score
        # penalty. Keep this configurable so frequency can be tuned without
        # weakening the hard safety blockers.
        tm_cfg = cfg.get('trade_management', {})
        recent_loss_window = float(
            tm_cfg.get('recent_loss_penalty_minutes', 45)
        ) * 60
        recent_loss_penalty = float(
            tm_cfg.get('recent_loss_penalty_score', 0.06)
        )
        last_loss = self._last_loss_ts.get(symbol, 0)
        if time.time() - last_loss < recent_loss_window:
            elapsed_min = (time.time() - last_loss) / 60
            self.logger.info(
                f"{symbol}: recent loss penalty -{recent_loss_penalty:.0%} "
                f"(last loss {elapsed_min:.0f} min ago)"
            )
            final_score = max(0.0, final_score - recent_loss_penalty)

        # ── Context-based learning penalty ───────────────────────────────────
        # Uses historical closed-trade stats by session/regime/direction.
        # Only fires when enough data exists (min 8 trades per context).
        ctx_penalty, ctx_block, ctx_reasons = compute_context_penalty(
            session     = self._get_current_session(),
            regime      = regime,
            direction   = signal,
            final_score = final_score,
            lookback_days = cfg.get('learning_analytics', {}).get('lookback_days', 60),
            min_trades  = cfg.get('learning_analytics', {}).get('min_trades', 8),
            block_wr    = cfg.get('learning_analytics', {}).get('block_win_rate', 0.20),
            penalty_wr  = cfg.get('learning_analytics', {}).get('penalty_win_rate', 0.38),
        )
        if ctx_reasons:
            for r in ctx_reasons:
                self.logger.info(f"{symbol}: context analytics — {r}")

        if ctx_block:
            log_activity(
                symbol,
                f"Context block: สถิติประวัติบ่งชี้ risk สูงใน context นี้ — {ctx_reasons[0]}",
                'warning',
            )
            return

        if ctx_penalty > 0:
            final_score = max(0.0, final_score - ctx_penalty)

        self.logger.info(
            f"{symbol}: final_score={final_score:.0%} "
            f"(setup={setup_quality:.0%} ai_eff={effective_ai:.0%} "
            f"bootstrap={bootstrap.score:.0f}/100 bw={bw:.0%} "
            f"ai_bonus={ai_alignment_bonus:.0%} "
            f"dir_pen={dir_penalty:.0%} ctx_pen={ctx_penalty:.0%} "
            f"recent_loss={'yes' if time.time()-last_loss<recent_loss_window else 'no'}) "
            f"min={min_score:.0%}"
        )

        if final_score < min_score:
            log_activity(
                symbol,
                f"Setup quality {final_score:.0%} < min {min_score:.0%} (L{autonomy_level})",
                'info',
            )
            return

        # ── Context Persistence: update stable bias + stability bonus ─────────
        persistence = self.context_persistence.update(
            symbol        = symbol,
            raw_signal    = signal,
            htf_bias      = htf_bias,
            htf_strength  = htf_strength,
            ai_confidence = brain_decision.confidence,
            final_score   = final_score,
        )
        # Apply stability bonus/penalty to final_score
        adjusted_score = max(0.0, min(1.0, final_score + persistence.stability_bonus))

        if not persistence.signal_matches and persistence.cycles_held >= 3:
            log_activity(
                symbol,
                f"Bias mismatch: signal={signal} but stable bias={persistence.stable_bias} "
                f"({persistence.cycles_held} cycles) — skipping",
                'info',
            )
            return

        self.logger.debug(
            f"{symbol}: persistence bias={persistence.stable_bias} "
            f"held={persistence.cycles_held} bonus={persistence.stability_bonus:+.3f} "
            f"flipped={persistence.just_flipped}"
        )

        # ── Signal Stability gate: require N consecutive same-direction cycles ─
        stability = self.signal_stability.check(symbol, signal)
        if not stability.is_stable:
            log_activity(
                symbol,
                f"Signal not stable yet: {signal} seen {stability.count}/{stability.required} cycles "
                f"(flips={stability.flip_count})",
                'info',
            )
            return

        self.logger.debug(
            f"{symbol}: signal stability OK — {signal} {stability.count}/{stability.required} cycles"
        )

        # ── Noise Filter: reject low-quality price action ──────────────────────
        spread_pts = 0.0
        tick_ns = self.executor.get_tick(symbol)
        if tick_ns:
            spread_pts = float(tick_ns.ask) - float(tick_ns.bid)

        noise = self.noise_filter.assess(df, signal, spread_pts=spread_pts)
        if noise.is_noisy:
            log_activity(
                symbol,
                f"Noise filter blocked: score={noise.noise_score:.0%} — "
                + "; ".join(noise.reasons[:2]),
                'info',
            )
            return

        if noise.noise_score > 0.30:
            self.logger.debug(
                f"{symbol}: noise={noise.noise_score:.0%} (below threshold) "
                f"reasons={noise.reasons}"
            )

        # ── Anti-Chase gate: block chasing exhausted moves ─────────────────────
        chase = self.anti_chase.assess(df, signal)
        if chase.is_chasing:
            log_activity(
                symbol,
                f"Anti-chase blocked: score={chase.chase_score:.0%} — "
                + "; ".join(chase.reasons[:2]),
                'warning',
            )
            return

        if chase.chase_score > 0.30:
            self.logger.debug(
                f"{symbol}: chase={chase.chase_score:.0%} reasons={chase.reasons}"
            )

        # ── Setup grade from adjusted_score ──────────────────────────────────
        # A+ = full size, A/B = config-scaled size, C = HOLD
        entry_cfg = cfg.get('entry_filters', {})
        grade_a_plus_at = float(entry_cfg.get('grade_a_plus_score', 0.70))
        grade_a_at      = float(entry_cfg.get('grade_a_score', 0.58))
        grade_b_at      = float(entry_cfg.get('grade_b_score', 0.45))
        grade_a_scale   = float(entry_cfg.get('grade_a_scale', 0.80))
        grade_b_scale   = float(entry_cfg.get('grade_b_scale', 0.55))

        if   adjusted_score >= grade_a_plus_at: grade, grade_scale = 'A+', 1.00
        elif adjusted_score >= grade_a_at:      grade, grade_scale = 'A',  grade_a_scale
        elif adjusted_score >= grade_b_at:      grade, grade_scale = 'B',  grade_b_scale
        else:
            log_activity(symbol, f"Grade C setup ({adjusted_score:.0%}) — skip", 'info')
            return

        self.logger.info(
            f"{symbol}: grade={grade} (adjusted_score={adjusted_score:.0%} "
            f"scale={grade_scale:.0%})"
        )

        # Confidence scale for lot sizing uses adjusted_score
        conf_full_at  = float(entry_cfg.get('conf_full_score', 0.65))
        conf_mid_at   = float(entry_cfg.get('conf_mid_score', 0.55))
        conf_low_at   = float(entry_cfg.get('conf_low_score', 0.45))
        conf_mid_mult = float(entry_cfg.get('conf_mid_scale', 0.85))
        conf_low_mult = float(entry_cfg.get('conf_low_scale', 0.70))
        if   adjusted_score >= conf_full_at: conf_scale = 1.00
        elif adjusted_score >= conf_mid_at:  conf_scale = conf_mid_mult
        elif adjusted_score >= conf_low_at:  conf_scale = conf_low_mult
        else:                                conf_scale = 0.55

        # ── Direction ban check ───────────────────────────────────────────────
        if self._is_direction_banned(symbol, signal):
            log_activity(
                symbol,
                f"Direction ban: {signal} ถูกแบน — เว้นจากการเทรดทิศนี้",
                'warning',
            )
            return

        # ── Opposite-direction cooldown (30 min after a loss before flipping) ─
        if self._is_opposite_dir_cooled(symbol, signal):
            log_activity(
                symbol,
                f"Opposite-direction cooldown: รอก่อนเปลี่ยนทิศเป็น {signal}",
                'warning',
            )
            return

        # ── Per-direction stacking guard ──────────────────────────────────────
        _mt5_type = {'BUY': 0, 'SELL': 1}
        same_dir_pos = [
            p for p in existing_sym_pos
            if p.type == _mt5_type.get(signal, -1)
        ]
        max_per_dir = cfg['risk'].get('max_per_direction', 2)
        if len(same_dir_pos) >= max_per_dir:
            self.logger.debug(
                f"{symbol}: {len(same_dir_pos)}/{max_per_dir} {signal} trades open — skip"
            )
            return
        if same_dir_pos and all(p.profit < 0 for p in same_dir_pos):
            total_loss = sum(p.profit for p in same_dir_pos)
            self.logger.info(
                f"{symbol}: All {signal} positions in drawdown (total={total_loss:.2f}) — no stacking"
            )
            log_activity(symbol, f"งดเพิ่มไม้ {signal} — ทุกไม้ทิศนี้ขาดทุนอยู่", 'warning')
            return

        log_activity(
            symbol,
            f"{'Brain' if autonomy_level >= 3 else 'Strategy'}: {signal} "
            f"score={final_score:.0%} "
            f"conf={brain_decision.confidence:.0%} | H4={htf_bias} L{autonomy_level}",
            'signal',
        )

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

        # ── 10. Position sizing (Kelly + direction + Brain risk scaling) ──────
        stats    = get_trade_stats()
        lot_size = self.risk.calculate_lot_size(
            balance, sl_dist, sym_info,
            win_rate = stats.get('win_rate') or None,
            avg_win  = stats.get('avg_win') or None,
            avg_loss = stats.get('avg_loss') or None,
            sample_count = stats.get('total_trades', 0),
        )
        # Apply direction penalty scale (streak-based lot reduction)
        dir_scale = self._get_direction_lot_scale(symbol, signal)
        if dir_scale < 1.0:
            self.logger.info(
                f"{symbol}: direction penalty scale={dir_scale:.2f} "
                f"(streak={self._dir_loss_streak.get(symbol, {}).get(signal, 0)})"
            )
            lot_size = round(lot_size * dir_scale, 2)
        # Brain risk state adjustment (never zeros at advisory levels — clamped to 0.40)
        brain_risk_adj = brain_decision.risk_multiplier_adj
        if brain_risk_adj < 1.0:
            lot_size = round(lot_size * brain_risk_adj, 2)
            self.logger.info(
                f"{symbol}: Brain risk adj={brain_risk_adj:.2f} "
                f"(state={brain_decision.risk_state})"
            )
        # Cold-start + confidence + grade quality scales
        cold_scale = self.cold_start_manager.get_risk_scale()
        lot_size   = round(lot_size * cold_scale * conf_scale * grade_scale, 2)
        if cold_scale < 1.0 or conf_scale < 1.0 or grade_scale < 1.0:
            self.logger.info(
                f"{symbol}: lot scaled cold={cold_scale:.2f} "
                f"conf={conf_scale:.2f} grade={grade_scale:.2f} → {lot_size:.2f}"
            )
        if lot_size <= 0:
            log_activity(symbol, "Risk sizing blocked: lot budget below broker minimum", 'warning')
            return

        portfolio_lot = self._cap_lot_to_portfolio_risk(
            lot_size, balance, sl_dist, sym_info, all_positions
        )
        if portfolio_lot <= 0:
            log_activity(
                symbol,
                "Portfolio risk cap reached or open risk cannot be verified",
                'warning',
            )
            return
        if portfolio_lot < lot_size:
            self.logger.info(
                f"{symbol}: portfolio risk cap reduced lot {lot_size:.2f} -> "
                f"{portfolio_lot:.2f}"
            )
            lot_size = portfolio_lot

        # Final institutional committee gate before live execution.
        committee = self.committee_guard.review(
            symbol          = symbol,
            signal          = signal,
            final_score     = final_score,
            adjusted_score  = adjusted_score,
            grade           = grade,
            bootstrap_score = bootstrap.score,
            effective_ai    = effective_ai,
            brain_decision  = brain_decision,
            mi_narrative    = mi_narrative,
            htf_bias        = htf_bias,
            htf_strength    = htf_strength,
            adx             = adx_val,
            rsi             = rsi_val,
            atr             = atr,
            close_price     = close,
            spread_price    = spread_pts,
            rr              = rr,
            lot_size        = lot_size,
            open_count      = len(all_positions),
            max_trades      = cfg['risk']['max_concurrent_trades'],
            noise_score     = noise.noise_score,
            chase_score     = chase.chase_score,
            drawdown        = self.risk.current_drawdown(balance),
            session         = self._get_current_session(),
            autonomy_level  = autonomy_level,
        )
        if symbol in self._terminal:
            self._terminal[symbol]['committee'] = committee.to_dict()

        self.logger.info(
            f"{symbol}: Committee={committee.verdict} "
            f"score={committee.score:.0%}/{committee.min_score:.0%} "
            f"risk_mult={committee.risk_multiplier:.2f}"
        )

        if not committee.approved:
            reason = committee.blockers[0] if committee.blockers else committee.verdict
            log_activity(symbol, f"Committee block: {reason}", 'warning')
            return

        if committee.risk_multiplier < 1.0:
            old_lot = lot_size
            lot_size = self._scale_lot_to_symbol(
                lot_size, committee.risk_multiplier, sym_info
            )
            self.logger.info(
                f"{symbol}: committee risk scale "
                f"{committee.risk_multiplier:.2f} | lot {old_lot:.2f} -> {lot_size:.2f}"
            )

        # ── 11. Execute guarded batch plan ────────────────────────────────────
        controls = self.execution_controls.get()
        global_capacity = max(
            0, int(cfg['risk']['max_concurrent_trades']) - len(all_positions)
        )
        direction_capacity = max(0, int(max_per_dir) - len(same_dir_pos))
        capacity = min(global_capacity, direction_capacity)
        order_plan = build_order_plan(lot_size, controls, sym_info, capacity)
        if not order_plan:
            log_activity(
                symbol,
                "Execution plan blocked by operator controls, capacity, or lot guardrails",
                'warning',
            )
            return

        plan_total = round(sum(order_plan), 8)
        if symbol in self._terminal:
            self._terminal[symbol]['execution_plan'] = {
                'controls_revision': controls['revision'],
                'lot_mode': controls['lot_mode'],
                'requested_orders': controls['order_count'],
                'planned_orders': len(order_plan),
                'lots': order_plan,
                'total_lot': plan_total,
            }
        self.logger.info(
            f"{symbol}: execution plan rev={controls['revision']} "
            f"mode={controls['lot_mode']} orders={len(order_plan)} "
            f"lots={order_plan} total={plan_total:.2f}"
        )

        results = []
        entry_spread = spread_pts / (sym_info.point * 10) if sym_info.point > 0 else 0.0
        for index, order_lot in enumerate(order_plan, start=1):
            result = self.executor.place_market_order(
                symbol    = symbol,
                direction = signal,
                lot_size  = order_lot,
                sl_price  = sl_price,
                tp_price  = tp_price,
                magic     = magic,
                comment   = (
                    f"AI|{signal}|B{index}/{len(order_plan)}|"
                    f"C{int(committee.score * 100)}"
                ),
            )
            if not result:
                log_activity(
                    symbol,
                    f"Batch stopped after {len(results)}/{len(order_plan)} orders",
                    'warning',
                )
                break

            results.append(result)
            ticket = result['ticket']
            self._known_tickets.add(ticket)
            self._ticket_direction[ticket]  = signal
            self._ticket_confidence[ticket] = ai_confidence
            self._ticket_brain[ticket]      = brain_decision
            self._ticket_committee[ticket]  = committee
            self._ticket_entry_bar[ticket]  = len(df)
            self.ai.record_trade_entry(ticket, df, signal)
            try:
                self.brain_memory.record_trade_open(
                    ticket, brain_decision, result['price'], symbol
                )
            except Exception:
                pass
            insert_trade({
                'ticket':        ticket,
                'symbol':        symbol,
                'direction':     signal,
                'lot_size':      order_lot,
                'entry_price':   result['price'],
                'sl_price':      sl_price,
                'tp_price':      tp_price,
                'open_time':     datetime.now().isoformat(timespec='seconds'),
                'ai_confidence': ai_confidence,
                'session':       self._get_current_session(),
                'regime':        regime,
                'final_score':   round(adjusted_score, 4),
                'spread_pips':   round(entry_spread, 2),
                'committee_verdict': committee.verdict,
                'committee_score':   round(committee.score, 4),
                'committee_risk_multiplier': round(committee.risk_multiplier, 4),
            })

        if results:
            self._last_trade_ts[symbol] = time.time()
            self._save_trade_guard_state()
            self.signal_stability.reset(symbol)
            actual_total = sum(float(r['lot_size']) for r in results)
            log_activity(
                symbol,
                f"Batch {signal}: {len(results)}/{len(order_plan)} orders | "
                f"total lot={actual_total:.2f} | SL={sl_price:.5f} TP={tp_price:.5f}",
                'order',
            )

    # ── Progressive autonomy level management ────────────────────────────────

    def _check_autonomy_level(self, balance: float) -> None:
        """
        Called each main-loop cycle to evaluate whether the system is ready
        to upgrade to a higher autonomy level (or should downgrade).
        Also syncs live performance to the current strategy version.
        """
        try:
            stats = get_trade_stats()
            total = int(stats.get('total_trades', 0))
            wr    = float(stats.get('win_rate', 0)) / 100.0
            auc   = float(getattr(self.ai, '_last_auc', 0.0) or 0.0)
            dd    = self.risk.current_drawdown(balance)
            loss_streak = self.risk._global_loss_streak

            self.cold_start_manager.check_upgrade(total, wr, auc, dd)
            self.cold_start_manager.check_downgrade(loss_streak, dd)

            # Update live metrics for current strategy version (every 10 trades)
            if total > 0 and total % 10 == 0:
                current_vid = self.version_manager.get_current_version().get('version_id')
                if current_vid:
                    self.version_manager.update_live_metrics(current_vid, {
                        'total_trades': total,
                        'win_rate':     round(wr, 4),
                        'profit_factor':stats.get('profit_factor', 0.0),
                        'total_profit': stats.get('total_profit',  0.0),
                        'drawdown_pct': round(dd * 100, 2),
                        'updated':      datetime.now().isoformat(timespec='seconds'),
                    })
        except Exception as exc:
            self.logger.debug(f"_check_autonomy_level: {exc}")

    # ── Exit Intelligence: re-evaluate all open positions ────────────────────

    def _re_evaluate_positions(self) -> None:
        """
        Called every cycle.  For each open position, ask ExitIntelligence
        whether the narrative has shifted enough to warrant early exit.
        Executes closes/reductions via executor.
        """
        magic      = self.config['trading']['magic_number']
        all_open   = self.executor.get_all_open_positions()
        if not all_open:
            return

        # ── Spread spike guard: close all positions for any symbol where the
        # live spread exceeds 15 pips, protecting SL integrity during spikes.
        _spike_limit = self.config.get('trade_management', {}).get(
            'max_spread_gold', 10.0
        ) * 1.5   # 1.5× the entry spread limit = spike threshold
        _spiked_symbols: set = set()
        for pos in all_open:
            if pos.magic != magic:
                continue
            sym = pos.symbol
            if sym in _spiked_symbols:
                continue
            tick = self.executor.get_tick(sym)
            sym_info = self.executor.get_symbol_info(sym)
            if tick and sym_info and sym_info.point > 0:
                live_spread = (tick.ask - tick.bid) / sym_info.point / 10
                if live_spread > _spike_limit:
                    _spiked_symbols.add(sym)
                    self.logger.warning(
                        f"SPREAD SPIKE on {sym}: {live_spread:.1f} pips > "
                        f"limit {_spike_limit:.1f} — emergency close all {sym} positions"
                    )
                    log_activity(
                        sym,
                        f"Spread spike {live_spread:.1f} pips — ปิดสถานะทั้งหมดฉุกเฉิน",
                        'warning',
                    )

        for sym in _spiked_symbols:
            for pos in all_open:
                if pos.symbol == sym and pos.magic == magic:
                    self.executor.close_by_ticket(pos.ticket)

        for pos in all_open:
            if pos.magic != magic:
                continue

            ticket    = pos.ticket
            direction = 'BUY' if pos.type == 0 else 'SELL'
            symbol    = pos.symbol

            # Need the latest MI narrative for this symbol
            term       = self._terminal.get(symbol, {})
            # Reconstruct a minimal mi_narrative proxy from terminal data
            # (the full object is not stored; use a lightweight holder)
            mi_narr = _TerminalNarrative(term)

            bd        = self._ticket_brain.get(ticket)
            atr       = float(term.get('atr', 0.0))
            close     = float(term.get('price', pos.price_current))
            sl_dist   = abs(close - pos.sl) if pos.sl else atr * 1.5
            profit_r  = (pos.profit / (sl_dist * pos.volume * 100)) if sl_dist > 0 else 0.0

            exit_sig = self.exit_intel.evaluate(
                ticket        = ticket,
                direction     = direction,
                entry_price   = pos.price_open,
                current_price = close,
                sl_price      = pos.sl,
                tp_price      = pos.tp,
                atr           = atr,
                mi_narrative  = mi_narr,
                brain_decision= bd,
                bars_held     = 0,
                profit_r      = profit_r,
            )

            if not exit_sig.should_act:
                continue

            self.logger.info(
                f"ExitIntelligence #{ticket} {direction}: "
                f"{exit_sig.action} — {exit_sig.reason}"
            )
            log_activity(
                symbol,
                f"ExitAI #{ticket}: {exit_sig.action} — {exit_sig.reason}",
                'warning',
            )

            if exit_sig.action == 'CLOSE':
                self.executor.close_by_ticket(ticket)

            elif exit_sig.action in ('REDUCE_50', 'REDUCE_30'):
                pct    = 0.50 if exit_sig.action == 'REDUCE_50' else 0.30
                volume = round(pos.volume * pct, 2)
                if volume >= 0.01:
                    self.executor.partial_close(ticket, volume)

            elif exit_sig.action == 'TIGHTEN_SL' and exit_sig.new_sl > 0:
                self.executor.modify_sl(ticket, exit_sig.new_sl)

    # ── Closed-position sync ──────────────────────────────────────────────────

    def _sync_closed_positions(self) -> None:
        """Detect positions that were closed (SL/TP hit) and update DB."""
        magic      = self.config['trading']['magic_number']
        now        = datetime.now(timezone.utc)
        from_dt    = now - timedelta(days=7)

        # After the engine has been offline for several days, DB rows can still
        # be marked open even though MT5 closed them long ago. Query back to the
        # oldest still-open DB ticket, capped at 60 days, so restart sync can
        # repair stale dashboard/learning data without an unbounded history scan.
        try:
            open_times = []
            for row in get_open_trades_from_db():
                raw = row.get('open_time')
                if not raw:
                    continue
                try:
                    dt = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    open_times.append(dt)
                except Exception:
                    continue
            if open_times:
                earliest = min(open_times) - timedelta(days=1)
                from_dt = min(from_dt, max(earliest, now - timedelta(days=60)))
        except Exception as exc:
            self.logger.debug(f"_sync_closed_positions history window: {exc}")

        from_ts = from_dt.timestamp()

        deals = self.executor.get_closed_deals(from_ts, now.timestamp(), magic)
        for d in deals:
            ticket = int(getattr(d, 'position_id', 0) or getattr(d, 'position', 0) or d.order)
            if ticket in self._known_tickets:
                # Deal reason: 3=SL, 4=TP, 0/1=manual or exit-intel
                _reason_map = {3: 'SL', 4: 'TP', 0: 'MANUAL', 1: 'MANUAL'}
                close_reason = _reason_map.get(getattr(d, 'reason', -1), 'EXIT_INTEL')
                close_trade_db(
                    ticket       = ticket,
                    close_price  = d.price,
                    close_time   = datetime.fromtimestamp(d.time).isoformat(),
                    profit       = d.profit,
                    close_reason = close_reason,
                )
                self.risk.record_trade_result(d.profit)
                # Feed trade outcome back to AI for self-learning
                direction = self._ticket_direction.pop(ticket, '')
                self.ai.label_closed_trade(ticket, d.profit, direction)
                # Update per-direction loss streak for direction ban
                sym_for_ban = self.executor.canonical_symbol(d.symbol or '')
                if direction and sym_for_ban:
                    self._update_direction_streak(sym_for_ban, direction, d.profit)
                elif direction:
                    sym_for_ban = self.config['trading']['symbols'][0]
                    for sym in self.config['trading']['symbols']:
                        self._update_direction_streak(sym, direction, d.profit)
                # Track recent-loss timestamp and set opposite-direction cooldown
                if d.profit < 0 and sym_for_ban:
                    self._last_loss_ts[sym_for_ban] = time.time()
                    if direction:
                        opp = 'SELL' if direction == 'BUY' else 'BUY'
                        cooldown_min = self.config.get(
                            'direction_ban', {}
                        ).get('opposite_cooldown_min', 30)
                        self._opposite_cooldown[sym_for_ban] = {
                            'direction': opp,
                            'until':     time.time() + cooldown_min * 60,
                        }
                        self._save_trade_guard_state()
                        self.logger.info(
                            f"{sym_for_ban}: opposite-direction ({opp}) cooldown "
                            f"{cooldown_min}min after {direction} loss"
                        )
                # Brain memory: record close + self-review
                try:
                    outcome = 'win' if d.profit > 0 else ('breakeven' if d.profit == 0 else 'loss')
                    entry_bar = self._ticket_entry_bar.pop(ticket, 0)
                    bars_held = 0  # approximate
                    self.brain_memory.record_trade_close(
                        ticket, d.price, d.profit, outcome, bars_held
                    )
                    bd = self._ticket_brain.get(ticket)
                    if bd:
                        self.brain_memory.update_learning_feedback(
                            bd.market_regime, outcome, d.profit,
                            bd.confidence, bars_held
                        )
                        if outcome == 'loss':
                            self.brain_memory.record_failure_pattern(
                                d.symbol or '', direction, bd.market_regime,
                                bd.signals_active, d.profit, bars_held
                            )
                    self.brain_memory.self_review(ticket)
                except Exception as exc:
                    self.logger.debug(f"Brain memory post-trade update failed: {exc}")
                self._ticket_confidence.pop(ticket, None)
                self._ticket_brain.pop(ticket, None)
                self._ticket_committee.pop(ticket, None)
                self._known_tickets.discard(ticket)

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
                'brain_decision': (
                    self._ticket_brain[p.ticket].decision
                    if p.ticket in self._ticket_brain else ''
                ),
                'brain_confidence': (
                    round(self._ticket_brain[p.ticket].confidence, 3)
                    if p.ticket in self._ticket_brain else 0.0
                ),
                'brain_regime': (
                    self._ticket_brain[p.ticket].market_regime
                    if p.ticket in self._ticket_brain else ''
                ),
                'committee_verdict': (
                    self._ticket_committee[p.ticket].verdict
                    if p.ticket in self._ticket_committee else ''
                ),
                'committee_score': (
                    round(self._ticket_committee[p.ticket].score, 3)
                    if p.ticket in self._ticket_committee else 0.0
                ),
                'committee_risk_multiplier': (
                    round(self._ticket_committee[p.ticket].risk_multiplier, 3)
                    if p.ticket in self._ticket_committee else 1.0
                ),
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
                'regime':          term.get('regime', 'TREND'),
                'signal':          term.get('signal', 'HOLD'),
                'atr':             term.get('atr', 0),
                'mi_regime':       term.get('mi_regime', ''),
                'mi_narrative':    term.get('mi_narrative', ''),
                'mi_signals':      term.get('mi_signals', []),
                'mi_quality':      term.get('mi_quality', 0.0),
                'mi_block_buy':    term.get('mi_block_buy', False),
                'mi_block_sell':   term.get('mi_block_sell', False),
                'committee':       term.get('committee', {}),
            }

        all_time_stats = get_trade_stats()
        today_stats = get_today_trade_stats()

        bal = account_info.balance
        risk_equity = account_info.equity
        controls = self.execution_controls.get()
        daily_loss_pct   = (self.risk.daily_start_balance - risk_equity) / max(self.risk.daily_start_balance, 1)
        weekly_loss_pct  = (self.risk.weekly_start_balance - risk_equity) / max(self.risk.weekly_start_balance, 1)
        daily_limit_hit  = daily_loss_pct  >= self.risk._cfg.get('max_daily_loss',  0.04)
        weekly_limit_hit = weekly_loss_pct >= self.risk._cfg.get('max_weekly_loss', 0.10)
        dd_limit_hit     = self.risk.current_drawdown(risk_equity) >= self.risk._cfg.get('max_drawdown', 0.08)

        if self.risk._trading_halted:
            risk_lock = 'HALTED'
        elif not controls.get('trading_enabled', False):
            risk_lock = 'OPERATOR_DISABLED'
        elif dd_limit_hit:
            risk_lock = 'DD_BREACH'
        elif weekly_limit_hit:
            risk_lock = 'WEEKLY_LIMIT'
        elif daily_limit_hit:
            risk_lock = 'DAILY_LIMIT'
        elif not self.risk.check_adaptive_cooldown():
            risk_lock = 'COOLDOWN'
        else:
            risk_lock = 'OK'

        write_state({
            'timestamp':           datetime.now().isoformat(timespec='seconds'),
            'balance':             bal,
            'equity':              account_info.equity,
            'initial_balance':     self.risk.initial_balance,
            'peak_balance':        self.risk.peak_balance,
            'margin':              account_info.margin,
            'margin_level':        getattr(account_info, 'margin_level', 0.0),
            'free_margin':         account_info.margin_free,
            'terminal':            self._terminal,
            'symbols':             symbols_dash,
            'open_trades':         positions_data,
            'drawdown_pct':        round(self.risk.current_drawdown(risk_equity) * 100, 2),
            'daily_pnl':           round(self.risk.daily_pnl(risk_equity), 2),
            'weekly_pnl':          round(self.risk.weekly_pnl(risk_equity), 2),
            'daily_start_balance': self.risk.daily_start_balance,
            'weekly_start_balance':self.risk.weekly_start_balance,
            'risk_lock':           risk_lock,
            'trading_halted':      self.risk._trading_halted,
            'stats': {
                'total_trades': today_stats.get('total_trades', 0),
                'wins':         today_stats.get('wins', 0),
                'losses':       today_stats.get('losses', 0),
                'closed_trades': today_stats.get('closed_trades', 0),
                'open_trades_today': today_stats.get('open_trades_today', 0),
                'win_rate':     today_stats.get('win_rate', 0.0),
                'today_pnl':    today_stats.get('today_pnl', round(self.risk.daily_pnl(risk_equity), 2)),
                'weekly_pnl':   round(self.risk.weekly_pnl(risk_equity), 2),
                'total_profit': today_stats.get('total_profit', 0),
                'profit_factor':today_stats.get('profit_factor', 0),
            },
            'all_time_stats': all_time_stats,
            'autonomy': self.cold_start_manager.get_status(),
            'execution_controls': controls,
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
                risk_equity = account_info.equity
                self._account_margin_level = float(
                    getattr(account_info, 'margin_level', 0.0) or 0.0
                )
                self.risk.update_balance(risk_equity)
                record_equity(balance, risk_equity)

                # Global drawdown halt
                if not self.risk.check_drawdown_limit(risk_equity):
                    self.logger.critical(
                        "DRAWDOWN LIMIT REACHED — engine stopped. "
                        "Review account before restarting."
                    )
                    self.running = False
                    break

                # Process each symbol
                for symbol in symbols:
                    try:
                        self._process_symbol(symbol, risk_equity)
                    except Exception as exc:
                        self.logger.error(
                            f"Error processing {symbol}: {exc}", exc_info=True
                        )

                # Active trade management (break-even / trailing / partial close)
                all_open = self.executor.get_all_open_positions()
                self.trade_manager.manage(all_open)

                # Exit Intelligence: narrative-driven early exits
                self._re_evaluate_positions()

                # Housekeeping
                self._sync_closed_positions()
                self._update_dashboard_state(account_info)

                # Progressive autonomy: check upgrade / downgrade each cycle
                self._check_autonomy_level(risk_equity)

                dd_pct = self.risk.current_drawdown(risk_equity) * 100
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
