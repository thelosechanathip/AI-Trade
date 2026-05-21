"""
strategy.py — Multi-layer confluence signal engine with market regime detection.

Regimes (ADX-based):
  TREND    : ADX >= 25 — use trend-following confluence scoring
  RANGE    : ADX < 20  — use mean-reversion (BB + Stochastic + RSI extremes)
  HIGH_VOL : ADX >= 42 — trend-following with reduced size

Signal pipeline (v2.5 — with MarketIntelligence):
  1. Session filter            (REQUIRED)
  2. Market context analysis   → trend direction, strength, exhaustion risk
  3. MarketIntelligence analysis → institutional signals, divergence, BOS/CHOCH
  4. Trend dominance protection (HARD BLOCK) — prevents counter-trend in strong moves
  5. EMA trend filter          (REQUIRED, HTF-aware)
  6. Entry momentum gate       (HARD BLOCK) — 2-bar candle direction
  7. RSI recovery/rejection gate (HARD BLOCK)
  8. Confluence scoring        (5 conditions, need >= min_confluence)
  9. HTF bias guard            (HARD BLOCK) — final check against H4+D1 trend

Returns: (signal, atr, last_swing_high, last_swing_low, mi_narrative)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, time as dt_time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils import compute_indicators, detect_market_structure
from market_intelligence import MarketIntelligence, MarketNarrative

logger = logging.getLogger('AI-Trade')

# Module-level singleton — instantiated once, reused every cycle
_mi = MarketIntelligence()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MarketContext:
    """Rich market state snapshot used throughout signal generation."""
    regime: str = 'TREND'           # 'TREND' | 'RANGE' | 'HIGH_VOL'
    trend_dir: str = 'NEUTRAL'      # dominant M15 trend direction
    trend_strength: float = 0.0     # 0.0–1.0  (ADX + EMA distance combined)
    adx: float = 0.0
    rsi: float = 50.0
    rsi_prev: float = 50.0          # RSI one bar ago (for recovery detection)
    ema50: float = 0.0
    ema200: float = 0.0
    close: float = 0.0
    atr: float = 0.0
    exhaustion_long: bool = False   # price very far above EMA200 → longs exhausted
    exhaustion_short: bool = False  # price very far below EMA200 → shorts exhausted
    rsi_recovering: bool = False    # RSI rising from oversold  (bullish recovery)
    rsi_rejecting: bool = False     # RSI falling from overbought (bearish rejection)
    macd_hist: float = 0.0
    macd_hist_prev: float = 0.0

    @property
    def macd_momentum_bull(self) -> bool:
        return self.macd_hist > self.macd_hist_prev

    @property
    def macd_momentum_bear(self) -> bool:
        return self.macd_hist < self.macd_hist_prev


# ── Session filter ────────────────────────────────────────────────────────────

def _parse_time(s: str) -> dt_time:
    h, m = map(int, s.split(':'))
    return dt_time(h, m)


def check_session_filter(config: dict, bar_time: Optional[datetime] = None) -> bool:
    sessions = config.get('sessions', {})
    if not sessions.get('enabled', True):
        return True

    if bar_time is not None:
        if hasattr(bar_time, 'tzinfo') and bar_time.tzinfo is None:
            now = bar_time.time()
        else:
            try:
                now = bar_time.astimezone(timezone.utc).time()
            except Exception:
                now = bar_time.time()
    else:
        now = datetime.now(timezone.utc).time()

    def in_window(start_s: str, end_s: str) -> bool:
        return _parse_time(start_s) <= now <= _parse_time(end_s)

    return (
        in_window(sessions['london']['start'],   sessions['london']['end']) or
        in_window(sessions['new_york']['start'], sessions['new_york']['end'])
    )


# ── Market regime detection ───────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame, config: dict) -> str:
    """
    Classify market regime from ADX.
    Returns: 'TREND' | 'RANGE' | 'HIGH_VOL'
    """
    s            = config['strategy']
    adx_trend    = s.get('adx_trend_threshold',    25)
    adx_range    = s.get('adx_range_threshold',    20)
    adx_high_vol = s.get('adx_high_vol_threshold', 42)

    if 'adx' not in df.columns or len(df) < 1:
        return 'TREND'

    adx = float(df['adx'].iloc[-1])

    if adx >= adx_high_vol:
        return 'HIGH_VOL'
    if adx >= adx_trend:
        return 'TREND'
    if adx < adx_range:
        return 'RANGE'
    return 'TREND'


# ── Market context builder ────────────────────────────────────────────────────

def build_market_context(df: pd.DataFrame, config: dict) -> MarketContext:
    """
    Build a rich MarketContext from M15 OHLCV+indicators.

    Computes trend direction, strength, exhaustion flags, and RSI recovery/rejection
    signals that are used as hard gates and scoring bonuses in generate_signal().
    """
    s      = config['strategy']
    td_cfg = config.get('trend_dominance', {})

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else latest

    close      = float(latest['close'])
    ema50      = float(latest['ema50'])
    ema200     = float(latest['ema200'])
    adx        = float(latest.get('adx', 0.0))
    rsi        = float(latest.get('rsi', 50.0))
    rsi_prev   = float(prev.get('rsi', 50.0))
    atr        = float(latest.get('atr', 0.0))
    macd_hist  = float(latest.get('macd_hist', 0.0))
    mh_prev    = float(prev.get('macd_hist', macd_hist))

    regime = detect_regime(df, config)

    # ── Trend direction from EMA alignment ───────────────────────────────────
    if close > ema50 and ema50 > ema200:
        trend_dir = 'BUY'
    elif close < ema50 and ema50 < ema200:
        trend_dir = 'SELL'
    elif close > ema50 and close > ema200:
        trend_dir = 'BUY'
    elif close < ema50 and close < ema200:
        trend_dir = 'SELL'
    else:
        trend_dir = 'NEUTRAL'

    # ── Trend strength: blend ADX (0-50 → 0-1) and EMA distance ─────────────
    adx_norm      = min(adx / 50.0, 1.0)
    ema_dist_pct  = abs(close - ema200) / max(ema200, 1.0)
    trend_strength = float(np.clip(adx_norm * 0.6 + ema_dist_pct * 10 * 0.4, 0.0, 1.0))

    # ── Exhaustion: price very far from EMA200 → trend may be overextended ───
    exhaustion_pct = td_cfg.get('exhaustion_ema200_pct', 0.025)  # 2.5% default
    exhaustion_long  = close > ema200 * (1 + exhaustion_pct)
    exhaustion_short = close < ema200 * (1 - exhaustion_pct)

    # ── RSI momentum states ───────────────────────────────────────────────────
    # RSI recovering: was oversold, now rising — suggests bearish move exhausted
    rsi_oversold_level   = td_cfg.get('rsi_recovery_level',   38)
    rsi_overbought_level = td_cfg.get('rsi_rejection_level',  62)
    rsi_min_swing        = td_cfg.get('rsi_momentum_swing',    5)  # must move ≥5 pts

    rsi_recovering = (
        rsi_prev < rsi_oversold_level
        and rsi > rsi_prev
        and (rsi - rsi_prev) >= rsi_min_swing
    )
    rsi_rejecting = (
        rsi_prev > rsi_overbought_level
        and rsi < rsi_prev
        and (rsi_prev - rsi) >= rsi_min_swing
    )

    return MarketContext(
        regime           = regime,
        trend_dir        = trend_dir,
        trend_strength   = trend_strength,
        adx              = adx,
        rsi              = rsi,
        rsi_prev         = rsi_prev,
        ema50            = ema50,
        ema200           = ema200,
        close            = close,
        atr              = atr,
        exhaustion_long  = exhaustion_long,
        exhaustion_short = exhaustion_short,
        rsi_recovering   = rsi_recovering,
        rsi_rejecting    = rsi_rejecting,
        macd_hist        = macd_hist,
        macd_hist_prev   = mh_prev,
    )


# ── Trend dominance protection ────────────────────────────────────────────────

def _trend_dominance_blocked(
    signal: str,
    ctx: MarketContext,
    htf_bias: str,
    config: dict,
) -> Tuple[bool, str]:
    """
    Hard block for counter-trend trades when a strong trend is active.

    Three-layer check:
      1. HTF says opposite direction with ADX confirming strength
      2. Price is exhausted on the signal side (already over-extended)
      3. RSI shows momentum reversal against the proposed signal

    Returns (blocked: bool, reason: str)
    """
    td_cfg = config.get('trend_dominance', {})
    if not td_cfg.get('enabled', True):
        return False, ''

    adx_threshold  = td_cfg.get('adx_strong_threshold',    28)
    strength_block = td_cfg.get('trend_strength_block',  0.55)

    # ── Layer 1: Strong HTF trend opposes signal ──────────────────────────────
    # When HTF clearly disagrees AND ADX confirms the HTF trend is strong
    if htf_bias != 'NEUTRAL' and htf_bias != signal:
        if ctx.adx >= adx_threshold:
            return True, (
                f"HTF={htf_bias} opposes {signal} with ADX={ctx.adx:.1f} "
                f"(strong trend, ≥{adx_threshold})"
            )

    # ── Layer 2: Price exhaustion on signal side ──────────────────────────────
    # SELL while longs are exhausted (price far above EMA200) → already moved too far down?
    # Actually: exhaustion_short = price far BELOW EMA200 → shorts are exhausted → block SELL
    if signal == 'SELL' and ctx.exhaustion_short:
        return True, (
            f"Short exhaustion: price is already {((ctx.ema200 - ctx.close)/ctx.ema200*100):.1f}% "
            f"below EMA200 — shorts likely exhausted"
        )
    if signal == 'BUY' and ctx.exhaustion_long:
        return True, (
            f"Long exhaustion: price is already {((ctx.close - ctx.ema200)/ctx.ema200*100):.1f}% "
            f"above EMA200 — longs likely exhausted"
        )

    # ── Layer 3: RSI momentum reversal ───────────────────────────────────────
    # RSI recovering from oversold → bearish momentum is reversing → block SELL
    if signal == 'SELL' and ctx.rsi_recovering:
        return True, (
            f"RSI recovery: RSI rose {ctx.rsi - ctx.rsi_prev:.1f}pts from oversold "
            f"({ctx.rsi_prev:.1f}→{ctx.rsi:.1f}) — blocking SELL"
        )
    # RSI rejecting from overbought → bullish momentum is reversing → block BUY
    if signal == 'BUY' and ctx.rsi_rejecting:
        return True, (
            f"RSI rejection: RSI fell {ctx.rsi_prev - ctx.rsi:.1f}pts from overbought "
            f"({ctx.rsi_prev:.1f}→{ctx.rsi:.1f}) — blocking BUY"
        )

    return False, ''


# ── Range (mean-reversion) signal ─────────────────────────────────────────────

def _range_signal(
    df: pd.DataFrame,
    config: dict,
    ctx: MarketContext,
) -> Tuple[str, float, float, float]:
    """
    Mean-reversion entry when ADX < range threshold.

    Uses Bollinger Bands pct_b + Stochastic + RSI extremes.
    Requires 2/3 conditions per side.
    Also respects exhaustion flags: exhaustion_short blocks SELL even in RANGE.
    """
    s      = config['strategy']
    latest = df.iloc[-1]
    atr    = ctx.atr

    bb_pct  = float(latest.get('bb_pct',  0.5))
    stoch_k = float(latest.get('stoch_k', 50.0))
    stoch_d = float(latest.get('stoch_d', 50.0))
    rsi     = ctx.rsi

    bb_buy  = s.get('range_bb_pct_buy',     0.12)
    bb_sell = s.get('range_bb_pct_sell',    0.88)
    rsi_os  = s.get('range_rsi_oversold',   33)
    rsi_ob  = s.get('range_rsi_overbought', 67)

    buy_score = sum([
        bb_pct <= bb_buy,
        stoch_k < 20 and stoch_d < 20,
        rsi <= rsi_os,
    ])
    sell_score = sum([
        bb_pct >= bb_sell,
        stoch_k > 80 and stoch_d > 80,
        rsi >= rsi_ob,
    ])

    if buy_score >= 2 and not ctx.exhaustion_long:
        return 'BUY', atr, 0.0, 0.0
    if sell_score >= 2 and not ctx.exhaustion_short:
        return 'SELL', atr, 0.0, 0.0

    return 'HOLD', atr, 0.0, 0.0


# ── Signal generation ─────────────────────────────────────────────────────────

_NULL_NARRATIVE = MarketNarrative(narrative='no data')


def generate_signal(
    df: pd.DataFrame,
    config: dict,
    bar_time: Optional[datetime] = None,
    htf_bias: str = 'NEUTRAL',
    htf_strength: float = 0.0,
) -> Tuple[str, float, float, float, MarketNarrative]:
    """
    Regime-adaptive, bias-balanced signal generation (v2.5).

    Pipeline (all layers in order — any HARD BLOCK returns HOLD immediately):
      1. Session filter
      2. Market context build (regime, trend_dir, strength, exhaustion, RSI states)
      3. MarketIntelligence analysis (institutional signals, divergence, BOS/CHOCH)
      4. Trend dominance protection (hard block counter-trend in strong moves)
      5. EMA trend filter (HTF-aware: relaxed when HTF confirms direction)
      6. RANGE mode branch (mean-reversion path)
      7. Entry momentum gate (2-bar candle direction check)
      8. RSI recovery/rejection gate
      9. Confluence scoring (5 conditions, need >= min_confluence)
     10. Final HTF guard

    Args:
        df:           M15 OHLCV + computed indicators
        config:       full config dict
        bar_time:     current bar time (for session filter)
        htf_bias:     'BUY' | 'SELL' | 'NEUTRAL' from higher-timeframe filter
        htf_strength: 0.0-1.0, strength of HTF trend

    Returns:
        (signal, atr, last_swing_high, last_swing_low, mi_narrative)
    """
    s = config['strategy']

    min_bars = s['ema_slow'] + s['atr_period'] + s['structure_lookback'] * 4 + 20
    if len(df) < min_bars:
        return 'HOLD', 0.0, 0.0, 0.0, _NULL_NARRATIVE

    df = compute_indicators(df, config)
    if len(df) < 4:
        return 'HOLD', 0.0, 0.0, 0.0, _NULL_NARRATIVE

    latest = df.iloc[-1]

    # ── 1. Session filter ─────────────────────────────────────────────────────
    if not check_session_filter(config, bar_time):
        return 'HOLD', float(latest.get('atr', 0.0)), 0.0, 0.0, _NULL_NARRATIVE

    # ── 2. Build market context ───────────────────────────────────────────────
    ctx = build_market_context(df, config)
    htf_upper = htf_bias.upper()

    # ── 3. MarketIntelligence analysis ────────────────────────────────────────
    mi_narrative = _mi.analyze(df, config, htf_bias=htf_upper, htf_strength=htf_strength)

    # ── 4. EMA trend filter (direction-gating) ────────────────────────────────
    close  = ctx.close
    ema50  = ctx.ema50
    ema200 = ctx.ema200

    # When HTF confirms the direction, relax EMA200 requirement.
    if htf_upper == 'BUY':
        trend_up   = close > ema50
        trend_down = (close < ema50) and (close < ema200)
    elif htf_upper == 'SELL':
        trend_up   = (close > ema50) and (close > ema200)
        trend_down = close < ema50
    else:
        trend_up   = (close > ema50) and (close > ema200)
        trend_down = (close < ema50) and (close < ema200)

    # ── RANGE mode: mean-reversion path ──────────────────────────────────────
    if ctx.regime == 'RANGE':
        raw_sig, raw_atr, sh, sl = _range_signal(df, config, ctx)
        if raw_sig == 'HOLD':
            return 'HOLD', raw_atr, sh, sl, mi_narrative

        # MI block check for RANGE signals
        if raw_sig == 'BUY' and mi_narrative.block_buy:
            logger.debug(f"MI blocked RANGE BUY: {mi_narrative.block_reason}")
            return 'HOLD', raw_atr, sh, sl, mi_narrative
        if raw_sig == 'SELL' and mi_narrative.block_sell:
            logger.debug(f"MI blocked RANGE SELL: {mi_narrative.block_reason}")
            return 'HOLD', raw_atr, sh, sl, mi_narrative

        # Apply trend dominance check even in RANGE
        blocked, reason = _trend_dominance_blocked(raw_sig, ctx, htf_upper, config)
        if blocked:
            logger.debug(f"RANGE trend-dominance blocked {raw_sig}: {reason}")
            return 'HOLD', raw_atr, sh, sl, mi_narrative

        # HTF guard for range signals
        if htf_upper != 'NEUTRAL' and raw_sig != htf_upper:
            logger.debug(f"HTF guard (RANGE): blocked {raw_sig}, H4 bias={htf_upper}")
            return 'HOLD', raw_atr, sh, sl, mi_narrative

        return raw_sig, raw_atr, sh, sl, mi_narrative

    # ── TREND / HIGH_VOL: require price on correct side of EMA ───────────────
    if not trend_up and not trend_down:
        logger.debug(
            f"HOLD: price between EMAs "
            f"(close={close:.1f} ema50={ema50:.1f} ema200={ema200:.1f})"
        )
        return 'HOLD', ctx.atr, 0.0, 0.0, mi_narrative

    # Determine candidate direction from EMA filter
    candidate = 'BUY' if trend_up else 'SELL'

    # ── MarketIntelligence hard blocks ────────────────────────────────────────
    if candidate == 'BUY' and mi_narrative.block_buy:
        logger.debug(f"MI blocked BUY: {mi_narrative.block_reason}")
        return 'HOLD', ctx.atr, 0.0, 0.0, mi_narrative
    if candidate == 'SELL' and mi_narrative.block_sell:
        logger.debug(f"MI blocked SELL: {mi_narrative.block_reason}")
        return 'HOLD', ctx.atr, 0.0, 0.0, mi_narrative

    # ── Trend dominance protection (hard block) ───────────────────────────────
    blocked, reason = _trend_dominance_blocked(candidate, ctx, htf_upper, config)
    if blocked:
        logger.debug(f"Trend dominance blocked {candidate}: {reason}")
        return 'HOLD', ctx.atr, 0.0, 0.0, mi_narrative

    # ── 5. Entry momentum gate (hard block) ───────────────────────────────────
    if len(df) >= 4:
        b1 = df.iloc[-3]
        b2 = df.iloc[-2]
        b1_bull = float(b1['close']) > float(b1['open'])
        b2_bull = float(b2['close']) > float(b2['open'])
        if candidate == 'SELL' and b1_bull and b2_bull:
            logger.debug(
                f"Momentum gate: blocked SELL — 2 consecutive bullish bars "
                f"(close={close:.1f})"
            )
            return 'HOLD', ctx.atr, 0.0, 0.0, mi_narrative
        if candidate == 'BUY' and (not b1_bull) and (not b2_bull):
            logger.debug(
                f"Momentum gate: blocked BUY — 2 consecutive bearish bars "
                f"(close={close:.1f})"
            )
            return 'HOLD', ctx.atr, 0.0, 0.0, mi_narrative

    # ── 6. RSI recovery / rejection gate (hard block) ─────────────────────────
    if candidate == 'SELL' and ctx.rsi_recovering:
        logger.debug(
            f"RSI recovery gate: blocked SELL "
            f"(RSI {ctx.rsi_prev:.1f}->{ctx.rsi:.1f}, recovering from oversold)"
        )
        return 'HOLD', ctx.atr, 0.0, 0.0, mi_narrative
    if candidate == 'BUY' and ctx.rsi_rejecting:
        logger.debug(
            f"RSI rejection gate: blocked BUY "
            f"(RSI {ctx.rsi_prev:.1f}->{ctx.rsi:.1f}, rejecting from overbought)"
        )
        return 'HOLD', ctx.atr, 0.0, 0.0, mi_narrative

    # ── 7. Confluence scoring ─────────────────────────────────────────────────
    rsi       = ctx.rsi
    macd_hist = ctx.macd_hist
    macd_line = float(latest.get('macd_line',   0.0))
    macd_sig  = float(latest.get('macd_signal', 0.0))

    rsi_bull_side = rsi >= 50
    rsi_bear_side = rsi <= 50

    macd_raw_bull         = (macd_line > macd_sig) and (macd_hist > 0)
    macd_raw_bear         = (macd_line < macd_sig) and (macd_hist < 0)
    macd_rsi_aligned_bull = macd_raw_bull and rsi_bull_side
    macd_rsi_aligned_bear = macd_raw_bear and rsi_bear_side

    rsi_buy_ok  = s['rsi_buy_min']  <= rsi <= s['rsi_buy_max']
    rsi_sell_ok = s['rsi_sell_min'] <= rsi <= s['rsi_sell_max']

    structure_up, structure_down, last_sh, last_sl = detect_market_structure(
        df,
        lookback = s['structure_lookback'],
        strength = s['swing_strength'],
    )
    last_sh = last_sh or 0.0
    last_sl = last_sl or 0.0

    avg_atr       = float(df['atr'].tail(20).mean())
    volatility_ok = ctx.atr >= avg_atr * s['atr_threshold_multiplier']

    buy_conds = {
        'macd_rsi_align': macd_rsi_aligned_bull,
        'rsi_range':      rsi_buy_ok,
        'structure':      structure_up,
        'volatility':     volatility_ok,
        'macd_momentum':  ctx.macd_momentum_bull,
    }
    sell_conds = {
        'macd_rsi_align': macd_rsi_aligned_bear,
        'rsi_range':      rsi_sell_ok,
        'structure':      structure_down,
        'volatility':     volatility_ok,
        'macd_momentum':  ctx.macd_momentum_bear,
    }

    min_score  = s.get('min_confluence', 3)
    buy_score  = sum(buy_conds.values())
    sell_score = sum(sell_conds.values())

    buy_ok  = trend_up   and buy_score  >= min_score
    sell_ok = trend_down and sell_score >= min_score

    direction = 'UP' if trend_up else 'DOWN'
    conds     = buy_conds if trend_up else sell_conds
    score     = buy_score if trend_up else sell_score
    passed    = [k for k, v in conds.items() if v]
    failed    = [k for k, v in conds.items() if not v]
    logger.debug(
        f"Signal ({direction}|{ctx.regime}) RSI={rsi:.1f} ADX={ctx.adx:.1f} "
        f"strength={ctx.trend_strength:.2f} MI_regime={mi_narrative.regime} "
        f"MACD_hist={macd_hist:.5f}(d{macd_hist - ctx.macd_hist_prev:+.5f}) "
        f"score={score}/{len(conds)} need={min_score} "
        f"passed={passed} failed={failed}"
    )

    # ── 8. Final HTF guard ────────────────────────────────────────────────────
    if buy_ok:
        if htf_upper == 'SELL':
            logger.debug(f"HTF guard: blocked BUY, H4 bias=SELL strength={htf_strength:.2f}")
            return 'HOLD', ctx.atr, last_sh, last_sl, mi_narrative
        return 'BUY', ctx.atr, last_sh, last_sl, mi_narrative

    if sell_ok:
        if htf_upper == 'BUY':
            logger.debug(f"HTF guard: blocked SELL, H4 bias=BUY strength={htf_strength:.2f}")
            return 'HOLD', ctx.atr, last_sh, last_sl, mi_narrative
        return 'SELL', ctx.atr, last_sh, last_sl, mi_narrative

    return 'HOLD', ctx.atr, last_sh, last_sl, mi_narrative
