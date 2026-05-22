"""
noise_filter.py — Market Noise Filter.

Identifies low-quality price action that should not trigger entries:
  - Doji / indecision candles (body < 30% of range)
  - Micro candles (body < 0.25 × ATR)
  - Parabolic / blow-off moves (3+ consecutive large candles same direction)
  - Spread-to-ATR ratio too high (cost > 15% of ATR)

Returns a NoiseAssessment with a composite noise_score 0–1.
A score >= 0.60 means the market is too noisy to trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class NoiseAssessment:
    noise_score:   float          # 0.0 (clean) – 1.0 (noisy)
    is_noisy:      bool           # True = skip entry
    reasons:       List[str]
    # Component scores
    doji_score:     float = 0.0   # indecision candle penalty
    micro_score:    float = 0.0   # body too small vs ATR
    parabolic_score: float = 0.0  # blow-off / exhaustion penalty
    spread_score:   float = 0.0   # spread cost vs ATR


class NoiseFilter:
    """
    Evaluates the last few candles for noise / low-quality price action.

    All checks use the M15 OHLCV DataFrame that main.py already computes,
    so there is zero extra API cost.
    """

    # Noise score threshold above which we suppress the entry
    NOISE_THRESHOLD = 0.60

    def __init__(self, threshold: float = NOISE_THRESHOLD):
        self.threshold = threshold

    def assess(
        self,
        df:         pd.DataFrame,
        signal:     str,
        spread_pts: float = 0.0,   # current spread in price points
    ) -> NoiseAssessment:
        """
        df      — OHLCV + indicators DataFrame (most recent bar is iloc[-1])
        signal  — 'BUY' or 'SELL' (HOLD is not assessed — call is skipped)
        spread_pts — live spread in the same units as price
        """
        reasons: List[str] = []
        doji_s = micro_s = para_s = spread_s = 0.0

        if len(df) < 5:
            return NoiseAssessment(0.0, False, [])

        c = df.iloc[-1]
        atr = float(c.get('atr', 0.0)) if 'atr' in df.columns else 0.0
        if atr <= 0:
            return NoiseAssessment(0.0, False, [])

        body  = abs(float(c['close']) - float(c['open']))
        rng   = float(c['high']) - float(c['low'])

        # ── Doji / indecision ─────────────────────────────────────────────────
        if rng > 0:
            body_ratio = body / rng
            if body_ratio < 0.18:       # nearly perfect doji
                doji_s = 1.0
                reasons.append(f"doji (body/range={body_ratio:.0%})")
            elif body_ratio < 0.30:     # weak body
                doji_s = 0.6
                reasons.append(f"weak body (body/range={body_ratio:.0%})")

        # ── Micro candle (body vs ATR) ────────────────────────────────────────
        body_atr = body / atr
        if body_atr < 0.15:
            micro_s = 1.0
            reasons.append(f"micro candle (body={body_atr:.0%}×ATR)")
        elif body_atr < 0.25:
            micro_s = 0.5
            reasons.append(f"small candle (body={body_atr:.0%}×ATR)")

        # ── Parabolic / blow-off detection ────────────────────────────────────
        # Three or more consecutive large (>1.2×ATR) same-direction candles
        para_s = self._parabolic_score(df, signal, atr)
        if para_s >= 0.7:
            reasons.append("parabolic move — chasing exhaustion")
        elif para_s >= 0.4:
            reasons.append("extended move — momentum stretched")

        # ── Spread cost vs ATR ────────────────────────────────────────────────
        if spread_pts > 0 and atr > 0:
            spread_ratio = spread_pts / atr
            if spread_ratio > 0.20:
                spread_s = min(1.0, spread_ratio * 3)
                reasons.append(f"high spread cost ({spread_ratio:.0%} of ATR)")

        # ── Composite score (weighted) ────────────────────────────────────────
        noise = (
            doji_s   * 0.30 +
            micro_s  * 0.30 +
            para_s   * 0.25 +
            spread_s * 0.15
        )
        noise = round(min(noise, 1.0), 3)

        return NoiseAssessment(
            noise_score    = noise,
            is_noisy       = noise >= self.threshold,
            reasons        = reasons,
            doji_score     = round(doji_s, 3),
            micro_score    = round(micro_s, 3),
            parabolic_score= round(para_s, 3),
            spread_score   = round(spread_s, 3),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parabolic_score(
        self, df: pd.DataFrame, signal: str, atr: float
    ) -> float:
        """
        Score 0–1 based on how many consecutive same-direction large candles
        precede the current bar. Used to detect chasing exhaustion.

        signal='BUY'  checks for consecutive bullish (green) large candles
        signal='SELL' checks for consecutive bearish (red) large candles
        """
        tail = df.tail(6)  # look at last 6 bars including current
        count_large = 0

        for i in range(len(tail) - 1, -1, -1):
            row   = tail.iloc[i]
            body  = float(row['close']) - float(row['open'])
            is_bull = body > 0
            is_large = abs(body) >= 1.0 * atr  # >= 1×ATR = large candle

            if not is_large:
                break  # chain broken

            if signal == 'BUY' and is_bull:
                count_large += 1
            elif signal == 'SELL' and not is_bull:
                count_large += 1
            else:
                break  # wrong direction — chain broken

        if count_large >= 4:
            return 1.0
        elif count_large == 3:
            return 0.75
        elif count_large == 2:
            return 0.30
        return 0.0
