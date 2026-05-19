"""
strategy.py — Multi-layer confluence signal engine with market regime detection.

Regimes (ADX-based):
  TREND    : ADX >= 25 — use trend-following confluence scoring
  RANGE    : ADX < 20  — use mean-reversion (BB + Stochastic + RSI extremes)
  HIGH_VOL : ADX >= 40 — reduce size, widen SL; still use trend logic

Signal generation:
  REQUIRED (must pass both):
    1. Session filter     — London / New York only
    2. Trend filter       — close above/below both EMA50 and EMA200

  SCORED (need >= min_confluence of 4 in TREND; different set in RANGE):
    3. MACD + RSI aligned — MACD histogram direction AND RSI agree on side
    4. RSI in range       — not overbought/oversold
    5. Market structure   — HH/HL or LH/LL detected
    6. Volatility filter  — ATR above threshold

Returns: signal ('BUY' | 'SELL' | 'HOLD'), atr, last_swing_high, last_swing_low,
         regime ('TREND' | 'RANGE' | 'HIGH_VOL')
"""

import logging
from datetime import datetime, timezone, time as dt_time
from typing import Optional, Tuple

import pandas as pd

from utils import compute_indicators, detect_market_structure

logger = logging.getLogger('AI-Trade')


# ── Session filter ────────────────────────────────────────────────────────────

def _parse_time(s: str) -> dt_time:
    h, m = map(int, s.split(':'))
    return dt_time(h, m)


def check_session_filter(config: dict, bar_time: Optional[datetime] = None) -> bool:
    sessions = config.get('sessions', {})
    # sessions.enabled = false → เทรด 24 ชั่วโมง ไม่กรอง session
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
    Classify market regime from ADX reading.

    Returns: 'TREND' | 'RANGE' | 'HIGH_VOL'
    """
    s = config['strategy']
    adx_trend    = s.get('adx_trend_threshold',    25)
    adx_range    = s.get('adx_range_threshold',    20)
    adx_high_vol = s.get('adx_high_vol_threshold', 40)

    if 'adx' not in df.columns or len(df) < 1:
        return 'TREND'

    adx = float(df['adx'].iloc[-1])

    if adx >= adx_high_vol:
        return 'HIGH_VOL'
    if adx >= adx_trend:
        return 'TREND'
    if adx < adx_range:
        return 'RANGE'
    return 'TREND'   # between range and trend thresholds → treat as trend


# ── Range (mean-reversion) signal ─────────────────────────────────────────────

def _range_signal(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[str, float, float, float]:
    """
    Mean-reversion entry when the market is ranging (ADX < threshold).
    Uses Bollinger Bands + Stochastic + RSI extremes.
    """
    s      = config['strategy']
    latest = df.iloc[-1]
    atr    = float(latest.get('atr', 0.0))

    close   = float(latest['close'])
    bb_pct  = float(latest.get('bb_pct', 0.5))
    stoch_k = float(latest.get('stoch_k', 50.0))
    stoch_d = float(latest.get('stoch_d', 50.0))
    rsi     = float(latest.get('rsi', 50.0))

    bb_buy  = s.get('range_bb_pct_buy',     0.15)
    bb_sell = s.get('range_bb_pct_sell',    0.85)
    rsi_os  = s.get('range_rsi_oversold',   35)
    rsi_ob  = s.get('range_rsi_overbought', 65)

    # Buy: price near lower band + oversold stoch + oversold RSI
    buy_score = sum([
        bb_pct <= bb_buy,
        stoch_k < 20 and stoch_d < 20,
        rsi <= rsi_os,
    ])

    # Sell: price near upper band + overbought stoch + overbought RSI
    sell_score = sum([
        bb_pct >= bb_sell,
        stoch_k > 80 and stoch_d > 80,
        rsi >= rsi_ob,
    ])

    if buy_score >= 2:
        return 'BUY', atr, 0.0, 0.0
    if sell_score >= 2:
        return 'SELL', atr, 0.0, 0.0

    return 'HOLD', atr, 0.0, 0.0


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signal(
    df: pd.DataFrame,
    config: dict,
    bar_time: Optional[datetime] = None,
    htf_bias: str = 'NEUTRAL',
) -> Tuple[str, float, float, float]:
    """
    Regime-adaptive signal generation.

    In TREND / HIGH_VOL mode: score-based confluence (same as before).
    In RANGE mode: mean-reversion using BB + Stochastic + RSI extremes.

    htf_bias: 'BUY' | 'SELL' | 'NEUTRAL'
      When not NEUTRAL, blocks signals that oppose the higher-timeframe trend.

    Returns: signal, atr, last_swing_high, last_swing_low
    """
    s        = config['strategy']
    min_bars = s['ema_slow'] + s['atr_period'] + s['structure_lookback'] * 4 + 20

    if len(df) < min_bars:
        return 'HOLD', 0.0, 0.0, 0.0

    df = compute_indicators(df, config)
    if len(df) < 3:
        return 'HOLD', 0.0, 0.0, 0.0

    latest = df.iloc[-1]
    atr    = float(latest['atr'])

    # ── 1. Session filter (REQUIRED) ─────────────────────────────────────────
    if not check_session_filter(config, bar_time):
        return 'HOLD', atr, 0.0, 0.0

    # ── Detect regime ─────────────────────────────────────────────────────────
    regime = detect_regime(df, config)

    # ── HTF bias guard (REQUIRED when htf_filter.enabled) ────────────────────
    htf_cfg = config.get('htf_filter', {})
    htf_enabled = htf_cfg.get('enabled', False)
    htf_bias_upper = htf_bias.upper()

    # ── 2. Trend filter (REQUIRED in all regimes) ─────────────────────────────
    close  = float(latest['close'])
    ema50  = float(latest['ema50'])
    ema200 = float(latest['ema200'])

    # When HTF confirms the direction, relax EMA200 requirement (EMA50 sufficient).
    # This lets BUY signals fire at the start of an upswing before M15 EMA200 turns.
    if htf_bias_upper == 'BUY':
        trend_up   = close > ema50
        trend_down = (close < ema50) and (close < ema200)
    elif htf_bias_upper == 'SELL':
        trend_up   = (close > ema50) and (close > ema200)
        trend_down = close < ema50
    else:
        trend_up   = (close > ema50) and (close > ema200)
        trend_down = (close < ema50) and (close < ema200)

    # ── RANGE regime: mean-reversion, respect HTF bias ───────────────────────
    if regime == 'RANGE':
        raw_sig, raw_atr, sh, sl = _range_signal(df, config)
        if (htf_enabled and htf_bias_upper != 'NEUTRAL'
                and raw_sig != 'HOLD' and raw_sig != htf_bias_upper):
            logger.debug(
                f"HTF guard (RANGE): blocked {raw_sig}, H4 bias={htf_bias_upper}"
            )
            return 'HOLD', raw_atr, sh, sl
        return raw_sig, raw_atr, sh, sl

    # ── TREND / HIGH_VOL: require clear trend ─────────────────────────────────
    if not trend_up and not trend_down:
        logger.debug(
            f"HOLD: price between EMAs "
            f"(close={close:.2f} ema50={ema50:.2f} ema200={ema200:.2f})"
        )
        return 'HOLD', atr, 0.0, 0.0

    # ── Indicators ───────────────────────────────────────────────────────────
    rsi       = float(latest['rsi'])
    macd_hist = float(latest['macd_hist'])
    macd_line = float(latest['macd_line'])
    macd_sig  = float(latest['macd_signal'])

    # MACD histogram previous bar (for momentum direction)
    prev       = df.iloc[-2]
    prev_mhist = float(prev.get('macd_hist', macd_hist))

    # ── Entry momentum gate (HARD BLOCK) ─────────────────────────────────────
    # Block SELL if the last 2 completed bars are BOTH bullish (gold bouncing).
    # Block BUY  if the last 2 completed bars are BOTH bearish (gold collapsing).
    if len(df) >= 4:
        b1 = df.iloc[-3]   # 2 bars ago
        b2 = df.iloc[-2]   # 1 bar ago (last completed)
        b1_bull = float(b1['close']) > float(b1['open'])
        b2_bull = float(b2['close']) > float(b2['open'])
        if trend_down and b1_bull and b2_bull:
            logger.debug(
                f"Momentum gate: blocked SELL — last 2 bars both bullish (bounce)"
            )
            return 'HOLD', atr, 0.0, 0.0
        if trend_up and (not b1_bull) and (not b2_bull):
            logger.debug(
                f"Momentum gate: blocked BUY — last 2 bars both bearish (dump)"
            )
            return 'HOLD', atr, 0.0, 0.0

    # ── 3. MACD + RSI alignment (SCORED) ─────────────────────────────────────
    rsi_bull_side = rsi >= 50
    rsi_bear_side = rsi <= 50

    macd_raw_bull = (macd_line > macd_sig) and (macd_hist > 0)
    macd_raw_bear = (macd_line < macd_sig) and (macd_hist < 0)

    macd_rsi_aligned_bull = macd_raw_bull and rsi_bull_side
    macd_rsi_aligned_bear = macd_raw_bear and rsi_bear_side

    # ── 4. RSI in tradeable range (SCORED) ────────────────────────────────────
    rsi_buy_ok  = s['rsi_buy_min']  <= rsi <= s['rsi_buy_max']
    rsi_sell_ok = s['rsi_sell_min'] <= rsi <= s['rsi_sell_max']

    # ── 5. Market structure (SCORED) ─────────────────────────────────────────
    structure_up, structure_down, last_sh, last_sl = detect_market_structure(
        df,
        lookback=s['structure_lookback'],
        strength=s['swing_strength'],
    )
    last_sh = last_sh or 0.0
    last_sl = last_sl or 0.0

    # ── 6. Volatility filter (SCORED) ────────────────────────────────────────
    avg_atr       = float(df['atr'].tail(20).mean())
    volatility_ok = atr >= avg_atr * s['atr_threshold_multiplier']

    # ── 7. MACD histogram momentum (SCORED bonus) ────────────────────────────
    # Histogram growing in signal direction = momentum strengthening
    macd_hist_bull = macd_hist > prev_mhist   # getting more positive
    macd_hist_bear = macd_hist < prev_mhist   # getting more negative

    # ── Confluence scoring (5 conditions) ────────────────────────────────────
    min_score = s.get('min_confluence', 3)

    buy_conds = {
        'macd_rsi_align': macd_rsi_aligned_bull,
        'rsi_range':      rsi_buy_ok,
        'structure':      structure_up,
        'volatility':     volatility_ok,
        'macd_momentum':  macd_hist_bull,
    }
    sell_conds = {
        'macd_rsi_align': macd_rsi_aligned_bear,
        'rsi_range':      rsi_sell_ok,
        'structure':      structure_down,
        'volatility':     volatility_ok,
        'macd_momentum':  macd_hist_bear,
    }

    buy_score  = sum(buy_conds.values())
    sell_score = sum(sell_conds.values())

    buy_ok  = trend_up   and buy_score  >= min_score
    sell_ok = trend_down and sell_score >= min_score

    direction = 'UP' if trend_up else 'DOWN'
    conds     = buy_conds if trend_up else sell_conds
    score     = buy_score if trend_up else sell_score
    passed    = [k for k, v in conds.items() if v]
    failed    = [k for k, v in conds.items() if not v]
    adx_val   = float(latest.get('adx', 0.0))
    logger.debug(
        f"Signal ({direction}|{regime}) RSI={rsi:.1f} ADX={adx_val:.1f} "
        f"MACD_hist={macd_hist:.5f}(prev={prev_mhist:.5f}) "
        f"score={score}/{len(conds)} need={min_score} "
        f"passed={passed} failed={failed}"
    )

    if buy_ok:
        if htf_enabled and htf_bias_upper == 'SELL':
            logger.debug(f"HTF guard: blocked BUY, H4 bias=SELL")
            return 'HOLD', atr, last_sh, last_sl
        return 'BUY', atr, last_sh, last_sl
    if sell_ok:
        if htf_enabled and htf_bias_upper == 'BUY':
            logger.debug(f"HTF guard: blocked SELL, H4 bias=BUY")
            return 'HOLD', atr, last_sh, last_sl
        return 'SELL', atr, last_sh, last_sl

    return 'HOLD', atr, last_sh, last_sl
