"""
confidence_bootstrap.py — Synthetic confidence from technical signals.

Used during cold-start when AI hasn't accumulated enough training data.
Produces a "bootstrap confidence" score (0–100) from purely technical inputs
so the system can size trades sensibly without a trained model.

Formula:
    score = (
        htf_alignment    × 0.25 +
        trend_quality    × 0.20 +
        momentum_quality × 0.20 +
        liquidity_quality × 0.15 +
        volatility_quality × 0.10 +
        session_quality  × 0.10
    ) × 100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger('AI-Trade')


@dataclass
class BootstrapConfidence:
    score:             float = 50.0  # 0–100
    htf_alignment:     float = 0.0
    trend_quality:     float = 0.0
    momentum_quality:  float = 0.0
    liquidity_quality: float = 0.0
    volatility_quality:float = 0.0
    session_quality:   float = 1.0
    components:        dict  = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        """0–1 version for use in final_trade_score."""
        return self.score / 100.0

    def to_dict(self) -> dict:
        return {
            'score':              round(self.score, 1),
            'normalized':         round(self.normalized, 3),
            'htf_alignment':      round(self.htf_alignment, 3),
            'trend_quality':      round(self.trend_quality, 3),
            'momentum_quality':   round(self.momentum_quality, 3),
            'liquidity_quality':  round(self.liquidity_quality, 3),
            'volatility_quality': round(self.volatility_quality, 3),
            'session_quality':    round(self.session_quality, 3),
        }


class ConfidenceBootstrap:
    """
    Stateless — call compute() each signal cycle.
    All inputs are technical; no AI model is required.
    """

    def compute(
        self,
        *,
        signal:        str,           # 'BUY' | 'SELL'
        htf_bias:      str   = 'NEUTRAL',
        htf_strength:  float = 0.0,
        adx:           float = 20.0,
        rsi:           Optional[float] = 50.0,
        macd_hist:     Optional[float] = 0.0,
        stoch_k:       Optional[float] = 50.0,
        regime:        str   = '',
        mi_narrative          = None,   # MarketNarrative
        spread_ok:     bool  = True,
        session_ok:    bool  = True,
    ) -> BootstrapConfidence:

        result = BootstrapConfidence()
        is_buy = signal.upper() == 'BUY'

        # ── 1. HTF Alignment (25%) ────────────────────────────────────────────
        htf_agrees = (
            (is_buy  and htf_bias == 'BUY') or
            (not is_buy and htf_bias == 'SELL')
        )
        if htf_agrees:
            result.htf_alignment = float(np.clip(htf_strength, 0.0, 1.0))
        elif htf_bias == 'NEUTRAL':
            result.htf_alignment = 0.40   # neutral — cautious but not blocked
        else:
            result.htf_alignment = 0.05   # opposes HTF

        # ── 2. Trend Quality (20%) ────────────────────────────────────────────
        if adx >= 35:     result.trend_quality = 1.00
        elif adx >= 30:   result.trend_quality = 0.90
        elif adx >= 25:   result.trend_quality = 0.75
        elif adx >= 20:   result.trend_quality = 0.55
        elif adx >= 15:   result.trend_quality = 0.35
        else:             result.trend_quality = 0.20

        if regime in ('TREND_BULL', 'TREND_BEAR'):
            result.trend_quality = min(1.0, result.trend_quality + 0.10)
        elif regime in ('RANGE',):
            result.trend_quality *= 0.65
        elif regime in ('HIGH_VOL', 'EXHAUSTION'):
            result.trend_quality *= 0.50

        # ── 3. Momentum Quality (20%) ─────────────────────────────────────────
        mom = 0.0
        if is_buy:
            if rsi is not None:
                if   45 <= rsi <= 65:  mom += 0.38   # ideal RSI range for BUY
                elif 65 < rsi <= 72:   mom += 0.22   # extended but not overbought
                elif 38 <= rsi < 45:   mom += 0.15   # recovering
            if macd_hist is not None:
                if macd_hist > 0:      mom += 0.34
                elif macd_hist > -0.0001: mom += 0.15  # near zero = flat
            if stoch_k is not None:
                if stoch_k < 80:       mom += 0.28
                elif stoch_k < 90:     mom += 0.14
        else:  # SELL
            if rsi is not None:
                if   35 <= rsi <= 55:  mom += 0.38
                elif 28 <= rsi < 35:   mom += 0.22
                elif 55 < rsi <= 62:   mom += 0.15
            if macd_hist is not None:
                if macd_hist < 0:      mom += 0.34
                elif macd_hist < 0.0001: mom += 0.15
            if stoch_k is not None:
                if stoch_k > 20:       mom += 0.28
                elif stoch_k > 10:     mom += 0.14

        result.momentum_quality = float(np.clip(mom, 0.0, 1.0))

        # ── 4. Liquidity / MI Quality (15%) ──────────────────────────────────
        if mi_narrative is None:
            result.liquidity_quality = 0.50
        else:
            signals = set(getattr(mi_narrative, 'signals_active', []) or [])
            liq = 0.50  # neutral start

            if is_buy:
                if 'BOS_BULL'   in signals: liq += 0.25
                if 'CHOCH_BULL' in signals: liq += 0.25
                if 'LIQ_SWEEP_BULL' in signals: liq += 0.18
                if 'DISP_BULL'  in signals: liq += 0.15
                if 'REVERSAL_UP' in signals: liq += 0.20
                if 'RSI_DIV_BULL' in signals or 'MACD_DIV_BULL' in signals: liq += 0.10
                # Penalties
                if 'LIQ_SWEEP_BEAR' in signals: liq -= 0.25
                if 'REVERSAL_DOWN' in signals:  liq -= 0.40
                if 'DISP_BEAR' in signals:      liq -= 0.15
            else:  # SELL
                if 'BOS_BEAR'   in signals: liq += 0.25
                if 'CHOCH_BEAR' in signals: liq += 0.25
                if 'LIQ_SWEEP_BEAR' in signals: liq += 0.18
                if 'DISP_BEAR'  in signals: liq += 0.15
                if 'REVERSAL_DOWN' in signals: liq += 0.20
                if 'RSI_DIV_BEAR' in signals or 'MACD_DIV_BEAR' in signals: liq += 0.10
                # Penalties
                if 'LIQ_SWEEP_BULL' in signals: liq -= 0.25
                if 'REVERSAL_UP' in signals:    liq -= 0.40
                if 'DISP_BULL' in signals:      liq -= 0.15

            result.liquidity_quality = float(np.clip(liq, 0.0, 1.0))

        # ── 5. Volatility Quality (10%) ───────────────────────────────────────
        vol_climax = getattr(mi_narrative, 'volatility_climax', False) if mi_narrative else False
        if vol_climax:
            result.volatility_quality = 0.25
        elif regime == 'HIGH_VOL':
            result.volatility_quality = 0.45
        elif regime == 'EXPANSION':
            result.volatility_quality = 0.80
        else:
            result.volatility_quality = 1.00

        # ── 6. Session / Spread Quality (10%) ────────────────────────────────
        if session_ok and spread_ok:
            result.session_quality = 1.00
        elif session_ok or spread_ok:
            result.session_quality = 0.60
        else:
            result.session_quality = 0.25

        # ── Weighted composite ────────────────────────────────────────────────
        raw = (
            result.htf_alignment    * 0.25 +
            result.trend_quality    * 0.20 +
            result.momentum_quality * 0.20 +
            result.liquidity_quality * 0.15 +
            result.volatility_quality * 0.10 +
            result.session_quality  * 0.10
        )

        result.score = float(np.clip(raw * 100, 0, 100))
        result.components = result.to_dict()

        logger.debug(
            f"Bootstrap({signal}): score={result.score:.1f} "
            f"HTF={result.htf_alignment:.2f} trend={result.trend_quality:.2f} "
            f"mom={result.momentum_quality:.2f} liq={result.liquidity_quality:.2f}"
        )
        return result
