"""
uncertainty_engine.py — Standalone uncertainty quantification.

Scores four sources of uncertainty and aggregates them into a
UncertaintyAssessment that tells the Brain whether to HOLD.

Sources:
  1. Signal conflict    — agent votes disagree
  2. Regime ambiguity   — ADX in transition, regime unclear
  3. Volatility anomaly — ATR far from historical norm
  4. AI disagreement    — ML model opposes rule-based signal
  5. Narrative clarity  — few active MI signals, low setup quality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger('AI-Trade')


@dataclass
class UncertaintyAssessment:
    total_score:        float = 0.0   # 0.0–1.0  (higher = more uncertain)
    signal_conflict:    float = 0.0
    regime_ambiguity:   float = 0.0
    volatility_anomaly: float = 0.0
    ai_disagreement:    float = 0.0
    narrative_clarity:  float = 0.0
    should_hold:        bool  = False
    hold_reasons:       List[str] = field(default_factory=list)
    dominant_source:    str  = ''

    # threshold that triggered hold
    threshold_used:     float = 0.60

    def to_dict(self) -> dict:
        return {
            'total_score':        round(self.total_score, 3),
            'signal_conflict':    round(self.signal_conflict, 3),
            'regime_ambiguity':   round(self.regime_ambiguity, 3),
            'volatility_anomaly': round(self.volatility_anomaly, 3),
            'ai_disagreement':    round(self.ai_disagreement, 3),
            'narrative_clarity':  round(self.narrative_clarity, 3),
            'should_hold':        self.should_hold,
            'hold_reasons':       self.hold_reasons,
            'dominant_source':    self.dominant_source,
        }


class UncertaintyEngine:
    """
    Stateless scorer — call assess() each signal cycle.

    Parameters are read from config['market_intelligence'] but all have
    sensible defaults so it works without config.
    """

    # Default weights for the five components (must sum to 1.0)
    _DEFAULT_WEIGHTS = {
        'signal_conflict':    0.30,
        'regime_ambiguity':   0.20,
        'volatility_anomaly': 0.20,
        'ai_disagreement':    0.15,
        'narrative_clarity':  0.15,
    }

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get('market_intelligence', {})
        self._hold_threshold   = float(cfg.get('uncertainty_hold_threshold', 0.60))
        self._vol_lookback     = int(cfg.get('vol_uncertainty_lookback', 50))
        self._vol_z_threshold  = float(cfg.get('vol_z_threshold', 2.0))
        self._adx_ambig_low    = float(cfg.get('adx_ambig_low',  18.0))
        self._adx_ambig_high   = float(cfg.get('adx_ambig_high', 28.0))

    # ── Main entry ────────────────────────────────────────────────────────────

    def assess(
        self,
        *,
        agent_votes:     Optional[List[dict]] = None,   # [{name, vote, confidence}]
        df:              Optional[pd.DataFrame] = None,
        adx:             float = 0.0,
        regime:          str   = '',
        ai_bias:         str   = 'neutral',
        ai_confidence:   float = 50.0,
        rule_signal:     str   = 'HOLD',
        setup_quality:   float = 0.5,
        signals_active:  Optional[List[str]] = None,
        mi_narrative:    Optional[object] = None,       # MarketNarrative
    ) -> UncertaintyAssessment:

        result = UncertaintyAssessment(threshold_used=self._hold_threshold)

        result.signal_conflict    = self._score_signal_conflict(agent_votes, rule_signal)
        result.regime_ambiguity   = self._score_regime_ambiguity(adx, regime)
        result.volatility_anomaly = self._score_volatility_anomaly(df)
        result.ai_disagreement    = self._score_ai_disagreement(
            ai_bias, ai_confidence, rule_signal
        )
        result.narrative_clarity  = self._score_narrative_clarity(
            setup_quality, signals_active, mi_narrative
        )

        w = self._DEFAULT_WEIGHTS
        result.total_score = (
            w['signal_conflict']    * result.signal_conflict
            + w['regime_ambiguity']   * result.regime_ambiguity
            + w['volatility_anomaly'] * result.volatility_anomaly
            + w['ai_disagreement']    * result.ai_disagreement
            + w['narrative_clarity']  * result.narrative_clarity
        )

        # Dominant source
        scores = {
            'signal_conflict':    result.signal_conflict,
            'regime_ambiguity':   result.regime_ambiguity,
            'volatility_anomaly': result.volatility_anomaly,
            'ai_disagreement':    result.ai_disagreement,
            'narrative_clarity':  result.narrative_clarity,
        }
        result.dominant_source = max(scores, key=scores.get)

        # Build hold reasons
        if result.signal_conflict > 0.60:
            result.hold_reasons.append(
                f"agent conflict ({result.signal_conflict:.0%})"
            )
        if result.regime_ambiguity > 0.60:
            result.hold_reasons.append(
                f"regime unclear ADX={adx:.1f}"
            )
        if result.volatility_anomaly > 0.70:
            result.hold_reasons.append(
                f"volatility spike ({result.volatility_anomaly:.0%})"
            )
        if result.ai_disagreement > 0.65:
            result.hold_reasons.append(
                f"AI opposes signal ({ai_bias} vs {rule_signal})"
            )
        if result.narrative_clarity > 0.65:
            result.hold_reasons.append(
                f"low setup quality ({setup_quality:.2f})"
            )

        result.should_hold = result.total_score >= self._hold_threshold

        if result.should_hold:
            logger.debug(
                f"UncertaintyEngine: HOLD recommended "
                f"score={result.total_score:.2f} "
                f"dominant={result.dominant_source} "
                f"reasons={result.hold_reasons}"
            )

        return result

    # ── Component scorers ─────────────────────────────────────────────────────

    def _score_signal_conflict(
        self,
        agent_votes: Optional[List[dict]],
        rule_signal: str,
    ) -> float:
        """
        Measures disagreement among agent votes.
        If no votes are provided, bases score on rule_signal alone (HOLD = conflict).
        """
        if not agent_votes:
            # Without multi-agent votes we cannot measure conflict directly
            return 0.20 if rule_signal == 'HOLD' else 0.10

        buy_conf  = sum(v.get('confidence', 0.5) for v in agent_votes if v.get('vote') == 'BUY')
        sell_conf = sum(v.get('confidence', 0.5) for v in agent_votes if v.get('vote') == 'SELL')
        hold_conf = sum(v.get('confidence', 0.5) for v in agent_votes if v.get('vote') == 'HOLD')
        total     = buy_conf + sell_conf + hold_conf

        if total <= 0:
            return 0.5

        dominant = max(buy_conf, sell_conf, hold_conf)
        unanimity = dominant / total   # 1.0 = everyone agrees

        # Conflict is inverse of unanimity
        conflict = 1.0 - unanimity

        # Extra penalty if rule signal and dominant vote disagree
        dominant_vote = (
            'BUY' if buy_conf == dominant
            else ('SELL' if sell_conf == dominant else 'HOLD')
        )
        if rule_signal in ('BUY', 'SELL') and dominant_vote != rule_signal:
            conflict = min(1.0, conflict + 0.20)

        return float(np.clip(conflict, 0.0, 1.0))

    def _score_regime_ambiguity(self, adx: float, regime: str) -> float:
        """
        ADX in the grey zone between range and trend indicates regime uncertainty.
        """
        ambiguous_regimes = {'HIGH_VOL', 'ACCUMULATION', 'DISTRIBUTION'}

        base = 0.0

        # ADX score: 0 if clearly trending, 1 if clearly ranging, middle = ambiguous
        if adx <= 0:
            base = 0.40
        elif adx < self._adx_ambig_low:
            base = 0.30  # ranging — clear enough
        elif adx < self._adx_ambig_high:
            # transitional zone — maximum ambiguity
            frac = (adx - self._adx_ambig_low) / (self._adx_ambig_high - self._adx_ambig_low)
            base = 0.30 + frac * 0.50   # 0.30 → 0.80
        else:
            base = 0.10  # strong trend — low ambiguity

        if regime in ambiguous_regimes:
            base = min(1.0, base + 0.15)

        return float(base)

    def _score_volatility_anomaly(self, df: Optional[pd.DataFrame]) -> float:
        """
        Returns a score proportional to how far ATR is from its recent mean.
        Uses Z-score; clips at 1.0.
        """
        if df is None or 'atr' not in df.columns or len(df) < self._vol_lookback + 1:
            return 0.0

        atr_series = df['atr'].dropna()
        if len(atr_series) < self._vol_lookback:
            return 0.0

        recent  = atr_series.iloc[-self._vol_lookback:]
        current = float(atr_series.iloc[-1])
        mean    = float(recent.mean())
        std     = float(recent.std())

        if std < 1e-10:
            return 0.0

        z = abs(current - mean) / std

        # Normalise: z >= threshold → score 1.0
        score = min(1.0, z / self._vol_z_threshold)
        return float(score)

    def _score_ai_disagreement(
        self,
        ai_bias:       str,
        ai_confidence: float,
        rule_signal:   str,
    ) -> float:
        """
        High disagreement when AI opposes the rule signal with high confidence.
        """
        if rule_signal == 'HOLD':
            return 0.0   # nothing to disagree with

        ai_direction = ai_bias.lower()
        rule_dir     = rule_signal.lower()

        agrees = (
            (rule_dir == 'buy'  and ai_direction == 'bullish') or
            (rule_dir == 'sell' and ai_direction == 'bearish')
        )

        if agrees:
            # AI agrees — slight uncertainty only when confidence is low
            return float(np.clip((100 - ai_confidence) / 200, 0, 0.30))

        # AI neutral or opposing
        conf_norm = ai_confidence / 100.0
        if ai_direction == 'neutral':
            return float(np.clip(conf_norm * 0.40, 0, 0.40))

        # AI actively opposes
        return float(np.clip(conf_norm * 0.90, 0, 1.0))

    def _score_narrative_clarity(
        self,
        setup_quality:  float,
        signals_active: Optional[List[str]],
        mi_narrative:   Optional[object],
    ) -> float:
        """
        Low quality + few signals = high narrative uncertainty.
        """
        # Quality component: invert setup_quality
        quality_score = 1.0 - float(np.clip(setup_quality, 0.0, 1.0))

        # Signal density component
        n_signals = len(signals_active) if signals_active else 0
        if n_signals == 0:
            signal_score = 0.80
        elif n_signals == 1:
            signal_score = 0.50
        elif n_signals == 2:
            signal_score = 0.25
        else:
            signal_score = 0.05

        # MI narrative boost
        mi_boost = 0.0
        if mi_narrative is not None:
            if getattr(mi_narrative, 'reversal_detected', False):
                mi_boost = -0.15   # reversal is a clear signal → less uncertainty
            if getattr(mi_narrative, 'uncertainty', 0.0) > 0.7:
                mi_boost += 0.10

        score = 0.6 * quality_score + 0.4 * signal_score + mi_boost
        return float(np.clip(score, 0.0, 1.0))
