"""
anti_chase.py — Anti-Chase Engine.

Prevents the system from entering a trade in the direction of a move that
has already happened — the single most common retail mistake.

Checks:
  1. Overextension: price is > N×ATR from EMA50/EMA200
  2. Momentum exhaustion: RSI extreme (>75 for BUY, <25 for SELL)
  3. Consecutive large same-direction candles (bodies > 1.5×ATR) — momentum stretch
  4. Volatility climax: current ATR > mean + 2×std of recent ATR series

A ChaseAssessment.chase_score >= 0.60 means we are chasing; block the entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class ChaseAssessment:
    chase_score:     float        # 0.0 (safe) – 1.0 (full chase)
    is_chasing:      bool
    reasons:         List[str]
    # Components
    overextension:   float = 0.0
    rsi_extreme:     float = 0.0
    momentum_stretch: float = 0.0
    vol_climax:      float = 0.0


class AntiChaseEngine:
    """
    Designed to run AFTER the final_score gate so it only filters
    entries that otherwise passed the quality bar.

    Parameters
    ----------
    chase_threshold : float
        composite score above which we block the entry (default 0.60)
    ema_ext_atr : float
        multiples of ATR from the nearest EMA that defines 'overextended'
    rsi_buy_max : float
        RSI above this level on a BUY signal = momentum exhaustion
    rsi_sell_min : float
        RSI below this level on a SELL signal = momentum exhaustion
    """

    def __init__(
        self,
        chase_threshold: float = 0.60,
        ema_ext_atr:     float = 3.0,
        rsi_buy_max:     float = 75.0,
        rsi_sell_min:    float = 25.0,
    ):
        self.threshold    = chase_threshold
        self.ema_ext_atr  = ema_ext_atr
        self.rsi_buy_max  = rsi_buy_max
        self.rsi_sell_min = rsi_sell_min

    def assess(self, df: pd.DataFrame, signal: str) -> ChaseAssessment:
        """
        Evaluate whether the current signal is chasing a move that
        is already exhausted.

        Returns ChaseAssessment with chase_score and is_chasing flag.
        """
        if signal == 'HOLD':
            return ChaseAssessment(0.0, False, [])

        reasons: List[str] = []
        over_s = rsi_s = mom_s = vol_s = 0.0

        if len(df) < 10:
            return ChaseAssessment(0.0, False, [])

        c   = df.iloc[-1]
        atr = float(c.get('atr', 0.0)) if 'atr' in df.columns else 0.0
        if atr <= 0:
            return ChaseAssessment(0.0, False, [])

        close  = float(c['close'])
        rsi    = float(c.get('rsi', 50.0)) if 'rsi' in df.columns else 50.0

        # ── 1. Overextension from EMA ─────────────────────────────────────────
        over_s = self._overextension_score(c, close, atr, signal, df, reasons)

        # ── 2. RSI extreme ────────────────────────────────────────────────────
        if signal == 'BUY' and rsi > self.rsi_buy_max:
            rsi_s = min(1.0, (rsi - self.rsi_buy_max) / 15.0)
            reasons.append(f"RSI overbought ({rsi:.0f}) — buying exhaustion")
        elif signal == 'SELL' and rsi < self.rsi_sell_min:
            rsi_s = min(1.0, (self.rsi_sell_min - rsi) / 15.0)
            reasons.append(f"RSI oversold ({rsi:.0f}) — selling exhaustion")

        # ── 3. Momentum stretch: consecutive large candles same direction ──────
        mom_s = self._momentum_stretch(df, signal, atr, reasons)

        # ── 4. Volatility climax ──────────────────────────────────────────────
        vol_s = self._vol_climax(df, atr, reasons)

        # ── Composite score ───────────────────────────────────────────────────
        score = (
            over_s * 0.35 +
            rsi_s  * 0.25 +
            mom_s  * 0.25 +
            vol_s  * 0.15
        )
        score = round(min(score, 1.0), 3)

        return ChaseAssessment(
            chase_score      = score,
            is_chasing       = score >= self.threshold,
            reasons          = reasons,
            overextension    = round(over_s, 3),
            rsi_extreme      = round(rsi_s,  3),
            momentum_stretch = round(mom_s,  3),
            vol_climax       = round(vol_s,  3),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _overextension_score(
        self,
        c:       pd.Series,
        close:   float,
        atr:     float,
        signal:  str,
        df:      pd.DataFrame,
        reasons: List[str],
    ) -> float:
        """Score 0–1 based on distance from nearest relevant EMA."""
        score = 0.0

        # EMA200: structural long-term overextension
        if 'ema200' in df.columns:
            ema200 = float(c.get('ema200', close))
            dist   = (close - ema200) / max(atr, 1e-8)
            if signal == 'BUY' and dist > self.ema_ext_atr:
                score = max(score, min(1.0, (dist - self.ema_ext_atr) / self.ema_ext_atr))
                reasons.append(
                    f"overextended above EMA200 ({dist:.1f}×ATR)"
                )
            elif signal == 'SELL' and dist < -self.ema_ext_atr:
                score = max(score, min(1.0, (-dist - self.ema_ext_atr) / self.ema_ext_atr))
                reasons.append(
                    f"overextended below EMA200 ({-dist:.1f}×ATR)"
                )

        # EMA50: medium-term overextension (softer threshold)
        if 'ema50' in df.columns:
            ema50 = float(c.get('ema50', close))
            dist50 = (close - ema50) / max(atr, 1e-8)
            ext_threshold = self.ema_ext_atr * 0.70
            if signal == 'BUY' and dist50 > ext_threshold * 1.5:
                score = max(score, min(0.6, (dist50 - ext_threshold) / ext_threshold))
            elif signal == 'SELL' and dist50 < -ext_threshold * 1.5:
                score = max(score, min(0.6, (-dist50 - ext_threshold) / ext_threshold))

        return score

    def _momentum_stretch(
        self,
        df:      pd.DataFrame,
        signal:  str,
        atr:     float,
        reasons: List[str],
    ) -> float:
        """
        Count consecutive candles (from most recent) that are large (>1.5×ATR)
        and in the same direction as the intended trade.
        These indicate momentum is already stretched.
        """
        tail = df.tail(5)
        count = 0
        for i in range(len(tail) - 1, -1, -1):
            row  = tail.iloc[i]
            body = float(row['close']) - float(row['open'])
            is_bull  = body > 0
            is_large = abs(body) >= 1.5 * atr

            if not is_large:
                break
            if signal == 'BUY' and is_bull:
                count += 1
            elif signal == 'SELL' and not is_bull:
                count += 1
            else:
                break

        if count >= 3:
            reasons.append(f"{count} consecutive large {signal} candles — stretched")
            return min(1.0, count * 0.30)
        elif count == 2:
            return 0.25
        return 0.0

    def _vol_climax(
        self, df: pd.DataFrame, current_atr: float, reasons: List[str]
    ) -> float:
        """
        Check if current ATR is in 'climax' territory (> mean + 2×std of recent window).
        Entering during a volatility spike means chasing the spike.
        """
        if 'atr' not in df.columns or len(df) < 20:
            return 0.0

        recent_atr = df['atr'].tail(20)
        mean_atr   = float(recent_atr.mean())
        std_atr    = float(recent_atr.std())
        if std_atr <= 0:
            return 0.0

        z = (current_atr - mean_atr) / std_atr
        if z > 2.5:
            reasons.append(f"volatility climax (ATR z-score={z:.1f})")
            return min(1.0, (z - 2.0) / 2.0)
        elif z > 2.0:
            return 0.3
        return 0.0
