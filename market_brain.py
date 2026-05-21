"""
market_brain.py — Multi-agent central decision engine.

Architecture:
  Six specialist agents each cast a typed vote:
    TrendAgent       — EMA/ADX/HTF bias
    ReversalAgent    — RSI/MACD divergence, CHOCH, displacement
    LiquidityAgent   — sweeps, BOS, order flow
    VolatilityAgent  — ATR regime, expansion/contraction
    MomentumAgent    — MACD histogram, stoch, RSI slope
    DecisionAgent    — weighs all votes + memory + uncertainty

  DecisionAgent produces a BrainDecision that replaces the old
  'generate_signal → AI veto' flow entirely.

  Agent weights start equal but adapt based on per-regime
  performance stored in BrainMemory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from market_intelligence import MarketNarrative
from uncertainty_engine import UncertaintyAssessment, UncertaintyEngine

logger = logging.getLogger('AI-Trade')


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class BrainDecision:
    decision:            str   = 'HOLD'    # 'BUY'|'SELL'|'HOLD'
    confidence:          float = 0.0       # 0-1
    uncertainty:         float = 0.0       # 0-1 (from UncertaintyEngine)
    setup_quality:       float = 0.0       # 0-1
    market_regime:       str   = ''
    market_narrative:    str   = ''
    risk_state:          str   = 'normal'  # 'normal'|'elevated'|'high'|'extreme'
    reversal_probability:float = 0.0       # 0-1
    entry_quality:       float = 0.0       # 0-1 composite
    reasoning:           List[str] = field(default_factory=list)
    agent_votes:         List[dict] = field(default_factory=list)
    hold_reasons:        List[str] = field(default_factory=list)
    signals_active:      List[str] = field(default_factory=list)

    # Derived risk adjustment (used by RiskManager)
    risk_multiplier_adj: float = 1.0

    def to_dict(self) -> dict:
        return {
            'decision':             self.decision,
            'confidence':           round(self.confidence, 3),
            'uncertainty':          round(self.uncertainty, 3),
            'setup_quality':        round(self.setup_quality, 3),
            'market_regime':        self.market_regime,
            'market_narrative':     self.market_narrative,
            'risk_state':           self.risk_state,
            'reversal_probability': round(self.reversal_probability, 3),
            'entry_quality':        round(self.entry_quality, 3),
            'reasoning':            self.reasoning,
            'agent_votes':          self.agent_votes,
            'hold_reasons':         self.hold_reasons,
            'signals_active':       self.signals_active,
            'risk_multiplier_adj':  round(self.risk_multiplier_adj, 3),
        }


# ── Individual agents ─────────────────────────────────────────────────────────

class _Agent:
    name: str = 'base'
    base_weight: float = 1.0

    def vote(self, ctx: 'BrainContext') -> dict:
        """Return {'name', 'vote', 'confidence', 'reason'}."""
        raise NotImplementedError


class TrendAgent(_Agent):
    name = 'trend'
    base_weight = 1.0

    def vote(self, ctx: 'BrainContext') -> dict:
        htf   = ctx.htf_bias
        adx   = ctx.adx
        ema_ok = ctx.ema_trend  # 'BUY'|'SELL'|'NEUTRAL' from strategy layer

        if htf == 'BUY' and ema_ok == 'BUY' and adx >= 25:
            return {'name': self.name, 'vote': 'BUY',  'confidence': 0.80,
                    'reason': f'HTF+EMA aligned BUY ADX={adx:.1f}'}
        if htf == 'SELL' and ema_ok == 'SELL' and adx >= 25:
            return {'name': self.name, 'vote': 'SELL', 'confidence': 0.80,
                    'reason': f'HTF+EMA aligned SELL ADX={adx:.1f}'}
        if htf == 'BUY' and adx >= 20:
            return {'name': self.name, 'vote': 'BUY',  'confidence': 0.55,
                    'reason': 'HTF BUY bias, moderate ADX'}
        if htf == 'SELL' and adx >= 20:
            return {'name': self.name, 'vote': 'SELL', 'confidence': 0.55,
                    'reason': 'HTF SELL bias, moderate ADX'}
        return {'name': self.name, 'vote': 'HOLD', 'confidence': 0.40,
                'reason': 'No clear trend alignment'}


class ReversalAgent(_Agent):
    name = 'reversal'
    base_weight = 1.0

    def vote(self, ctx: 'BrainContext') -> dict:
        narr  = ctx.mi_narrative
        bull  = getattr(narr, 'rsi_divergence_bull', False) or getattr(narr, 'macd_divergence_bull', False)
        bear  = getattr(narr, 'rsi_divergence_bear', False) or getattr(narr, 'macd_divergence_bear', False)
        rev   = getattr(narr, 'reversal_detected', False)
        rev_d = getattr(narr, 'reversal_direction', '')

        if rev:
            if rev_d == 'UP':
                return {'name': self.name, 'vote': 'BUY',  'confidence': 0.78,
                        'reason': 'Reversal detected UP (CHOCH/divergence)'}
            if rev_d == 'DOWN':
                return {'name': self.name, 'vote': 'SELL', 'confidence': 0.78,
                        'reason': 'Reversal detected DOWN (CHOCH/divergence)'}

        if bull and not bear:
            return {'name': self.name, 'vote': 'BUY',  'confidence': 0.60,
                    'reason': 'Bullish RSI/MACD divergence'}
        if bear and not bull:
            return {'name': self.name, 'vote': 'SELL', 'confidence': 0.60,
                    'reason': 'Bearish RSI/MACD divergence'}
        if bull and bear:
            return {'name': self.name, 'vote': 'HOLD', 'confidence': 0.50,
                    'reason': 'Conflicting divergence signals'}

        return {'name': self.name, 'vote': 'HOLD', 'confidence': 0.35,
                'reason': 'No reversal signal'}


class LiquidityAgent(_Agent):
    name = 'liquidity'
    base_weight = 0.85

    def vote(self, ctx: 'BrainContext') -> dict:
        narr    = ctx.mi_narrative
        sweep_b = getattr(narr, 'liquidity_sweep_bull', False)
        sweep_s = getattr(narr, 'liquidity_sweep_bear', False)
        bos     = (getattr(narr, 'bos_choch', {}) or {})
        bos_b   = bos.get('bos_bull', False)
        bos_s   = bos.get('bos_bear', False)
        choch_b = bos.get('choch_bull', False)
        choch_s = bos.get('choch_bear', False)

        score_buy  = (0.4 if sweep_b else 0) + (0.3 if bos_b else 0) + (0.5 if choch_b else 0)
        score_sell = (0.4 if sweep_s else 0) + (0.3 if bos_s else 0) + (0.5 if choch_s else 0)

        if score_buy > 0.5 and score_buy > score_sell:
            return {'name': self.name, 'vote': 'BUY',  'confidence': min(0.80, 0.40 + score_buy),
                    'reason': 'Bullish sweep/BOS/CHOCH structure'}
        if score_sell > 0.5 and score_sell > score_buy:
            return {'name': self.name, 'vote': 'SELL', 'confidence': min(0.80, 0.40 + score_sell),
                    'reason': 'Bearish sweep/BOS/CHOCH structure'}
        return {'name': self.name, 'vote': 'HOLD', 'confidence': 0.35,
                'reason': 'No clear liquidity signal'}


class VolatilityAgent(_Agent):
    name = 'volatility'
    base_weight = 0.70

    def vote(self, ctx: 'BrainContext') -> dict:
        narr    = ctx.mi_narrative
        vol_ok  = not getattr(narr, 'volatility_climax', False)
        exp_b   = getattr(narr, 'displacement_bull', False)
        exp_s   = getattr(narr, 'displacement_bear', False)
        regime  = ctx.regime

        if not vol_ok:
            return {'name': self.name, 'vote': 'HOLD', 'confidence': 0.70,
                    'reason': 'Volatility climax — avoid new entries'}

        if exp_b and regime not in ('EXHAUSTION',):
            return {'name': self.name, 'vote': 'BUY',  'confidence': 0.60,
                    'reason': 'Bullish displacement (institutional BUY)'}
        if exp_s and regime not in ('EXHAUSTION',):
            return {'name': self.name, 'vote': 'SELL', 'confidence': 0.60,
                    'reason': 'Bearish displacement (institutional SELL)'}

        return {'name': self.name, 'vote': 'HOLD', 'confidence': 0.30,
                'reason': 'Normal volatility — no vol signal'}


class MomentumAgent(_Agent):
    name = 'momentum'
    base_weight = 0.90

    def vote(self, ctx: 'BrainContext') -> dict:
        rsi   = ctx.rsi
        macd  = ctx.macd_hist
        stoch = ctx.stoch_k

        bull_pts = 0
        bear_pts = 0

        if rsi is not None:
            if 45 <= rsi <= 75:  bull_pts += 1
            if 25 <= rsi <= 55:  bear_pts += 1

        if macd is not None:
            if macd > 0:  bull_pts += 1
            if macd < 0:  bear_pts += 1

        if stoch is not None:
            if stoch < 80:  bull_pts += 1
            if stoch > 20:  bear_pts += 1

        if bull_pts >= 2 and bull_pts > bear_pts:
            conf = 0.45 + bull_pts * 0.10
            return {'name': self.name, 'vote': 'BUY',  'confidence': min(0.75, conf),
                    'reason': f'Momentum BUY ({bull_pts}/3 indicators)'}
        if bear_pts >= 2 and bear_pts > bull_pts:
            conf = 0.45 + bear_pts * 0.10
            return {'name': self.name, 'vote': 'SELL', 'confidence': min(0.75, conf),
                    'reason': f'Momentum SELL ({bear_pts}/3 indicators)'}
        return {'name': self.name, 'vote': 'HOLD', 'confidence': 0.35,
                'reason': 'Mixed momentum'}


# ── Brain context ─────────────────────────────────────────────────────────────

@dataclass
class BrainContext:
    df:           Any               # pd.DataFrame
    mi_narrative: MarketNarrative
    htf_bias:     str   = 'NEUTRAL'
    htf_strength: float = 0.0
    adx:          float = 0.0
    rsi:          Optional[float] = None
    macd_hist:    Optional[float] = None
    stoch_k:      Optional[float] = None
    regime:       str   = ''
    ema_trend:    str   = 'NEUTRAL'  # pre-computed EMA trend from strategy
    ai_bias:      str   = 'neutral'
    ai_confidence:float = 50.0
    rule_signal:  str   = 'HOLD'    # what rule-based layer would have said
    symbol:       str   = 'XAUUSD'


# ── Market Brain ──────────────────────────────────────────────────────────────

class MarketBrain:
    """
    Central decision engine.  Call decide() each signal cycle.
    """

    _AGENTS = [
        TrendAgent(),
        ReversalAgent(),
        LiquidityAgent(),
        VolatilityAgent(),
        MomentumAgent(),
    ]

    # Minimum confidence to emit a directional signal
    _MIN_CONFIDENCE = 0.52

    # Minimum entry quality (composite) to skip HOLD
    _MIN_ENTRY_QUALITY = 0.38

    def __init__(
        self,
        config:      Optional[dict] = None,
        brain_memory = None,   # BrainMemory instance (optional)
    ):
        self._cfg    = config or {}
        self._memory = brain_memory
        self._uncertainty = UncertaintyEngine(config)

    # ── Main entry ────────────────────────────────────────────────────────────

    def decide(
        self,
        ctx:              BrainContext,
        open_trade_count: int = 0,
        max_trades:       int = 3,
    ) -> BrainDecision:

        result = BrainDecision(
            market_regime    = ctx.regime or getattr(ctx.mi_narrative, 'regime', ''),
            market_narrative = getattr(ctx.mi_narrative, 'narrative', ''),
            signals_active   = list(getattr(ctx.mi_narrative, 'signals_active', []) or []),
        )

        # ── Step 1: Collect agent votes ───────────────────────────────────────
        votes   = [agent.vote(ctx) for agent in self._AGENTS]
        weights = self._get_agent_weights(result.market_regime)
        result.agent_votes = votes

        # ── Step 2: Hard blocks from MI narrative ─────────────────────────────
        block_buy  = getattr(ctx.mi_narrative, 'block_buy',  False)
        block_sell = getattr(ctx.mi_narrative, 'block_sell', False)

        # ── Step 3: Tally weighted votes ──────────────────────────────────────
        buy_score  = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_w    = sum(weights.values())

        for v in votes:
            agent_name = v.get('name', '')
            w          = weights.get(agent_name, 1.0) / total_w
            conf       = float(v.get('confidence', 0.5))
            vote       = v.get('vote', 'HOLD')

            if vote == 'BUY':
                buy_score  += w * conf
            elif vote == 'SELL':
                sell_score += w * conf
            else:
                hold_score += w * conf

        # Apply hard blocks
        if block_buy:
            buy_score = 0.0
            result.hold_reasons.append(
                f"MI blocked BUY: {getattr(ctx.mi_narrative, 'block_reason', '')}"
            )
        if block_sell:
            sell_score = 0.0
            result.hold_reasons.append(
                f"MI blocked SELL: {getattr(ctx.mi_narrative, 'block_reason', '')}"
            )

        # ── Step 4: Determine preliminary direction ───────────────────────────
        if buy_score > sell_score and buy_score > hold_score:
            prelim = 'BUY'
            raw_conf = buy_score
        elif sell_score > buy_score and sell_score > hold_score:
            prelim = 'SELL'
            raw_conf = sell_score
        else:
            prelim = 'HOLD'
            raw_conf = hold_score

        # ── Step 5: Uncertainty assessment ───────────────────────────────────
        u_assess = self._uncertainty.assess(
            agent_votes    = votes,
            df             = ctx.df,
            adx            = ctx.adx,
            regime         = ctx.regime,
            ai_bias        = ctx.ai_bias,
            ai_confidence  = ctx.ai_confidence,
            rule_signal    = ctx.rule_signal,
            setup_quality  = float(getattr(ctx.mi_narrative, 'setup_quality', 0.5)),
            signals_active = list(getattr(ctx.mi_narrative, 'signals_active', []) or []),
            mi_narrative   = ctx.mi_narrative,
        )
        result.uncertainty = u_assess.total_score

        if u_assess.should_hold and prelim != 'HOLD':
            result.hold_reasons.extend(u_assess.hold_reasons)
            prelim   = 'HOLD'
            raw_conf = max(raw_conf * 0.5, 0.35)

        # ── Step 6: Memory-based confidence adjustment ────────────────────────
        mem_adj = self._memory_adjustment(result.market_regime, prelim)
        result.confidence = float(np.clip(raw_conf * mem_adj, 0.0, 1.0))

        # ── Step 7: Failure pattern penalty ──────────────────────────────────
        fail_sim = self._failure_similarity(
            result.signals_active, result.market_regime, prelim
        )
        if fail_sim > 0.65:
            result.hold_reasons.append(
                f"Similar setup failed before (similarity={fail_sim:.0%})"
            )
            result.confidence *= (1.0 - fail_sim * 0.40)
            if result.confidence < self._MIN_CONFIDENCE and prelim != 'HOLD':
                prelim = 'HOLD'

        # ── Step 8: Capacity check ────────────────────────────────────────────
        if open_trade_count >= max_trades and prelim != 'HOLD':
            prelim = 'HOLD'
            result.hold_reasons.append(
                f"Max concurrent trades ({open_trade_count}/{max_trades})"
            )

        # ── Step 9: Minimum confidence gate ──────────────────────────────────
        if prelim in ('BUY', 'SELL') and result.confidence < self._MIN_CONFIDENCE:
            result.hold_reasons.append(
                f"Confidence {result.confidence:.0%} < min {self._MIN_CONFIDENCE:.0%}"
            )
            prelim = 'HOLD'

        # ── Step 10: Setup quality & entry quality ────────────────────────────
        mi_quality = float(getattr(ctx.mi_narrative, 'setup_quality', 0.5))
        conf_adj   = float(getattr(ctx.mi_narrative, 'confidence_adjustment', 0.0))
        result.setup_quality = mi_quality

        result.entry_quality = float(np.clip(
            0.40 * result.confidence +
            0.35 * mi_quality +
            0.25 * (1.0 - result.uncertainty),
            0.0, 1.0,
        ))

        if prelim in ('BUY', 'SELL') and result.entry_quality < self._MIN_ENTRY_QUALITY:
            result.hold_reasons.append(
                f"Entry quality {result.entry_quality:.0%} too low"
            )
            prelim = 'HOLD'

        # ── Step 11: Reversal probability ────────────────────────────────────
        result.reversal_probability = self._estimate_reversal_prob(ctx)

        # ── Step 12: Risk state ───────────────────────────────────────────────
        result.risk_state, result.risk_multiplier_adj = self._assess_risk_state(
            result.uncertainty, result.reversal_probability, ctx.adx
        )

        if result.risk_state in ('extreme',) and prelim != 'HOLD':
            result.hold_reasons.append(f"Risk state: {result.risk_state}")
            prelim = 'HOLD'

        # ── Finalise ──────────────────────────────────────────────────────────
        result.decision = prelim
        result.reasoning = self._build_reasoning(result, votes, u_assess)

        logger.debug(
            f"MarketBrain: {result.decision} conf={result.confidence:.2f} "
            f"unc={result.uncertainty:.2f} quality={result.entry_quality:.2f} "
            f"regime={result.market_regime}"
        )
        return result

    # ── Agent weight adaptation ───────────────────────────────────────────────

    def _get_agent_weights(self, regime: str) -> Dict[str, float]:
        """
        Agents start with their base weights.
        If BrainMemory has enough data, adjust weights based on
        which agents historically performed well in this regime.
        (Simple heuristic — full adaptation requires more trade history.)
        """
        weights = {a.name: a.base_weight for a in self._AGENTS}

        if self._memory is None:
            return weights

        # Regime-specific boosts
        trend_regimes = {'TREND_BULL', 'TREND_BEAR', 'EXPANSION'}
        reversal_regimes = {'REVERSAL', 'EXHAUSTION', 'ACCUMULATION', 'DISTRIBUTION'}
        vol_regimes = {'HIGH_VOL', 'EXPANSION'}

        if regime in trend_regimes:
            weights['trend']    = 1.30
            weights['momentum'] = 1.10
            weights['reversal'] = 0.70
        elif regime in reversal_regimes:
            weights['reversal']   = 1.30
            weights['liquidity']  = 1.10
            weights['trend']      = 0.70
        elif regime in vol_regimes:
            weights['volatility'] = 1.30
            weights['trend']      = 0.80
        elif regime == 'RANGE':
            weights['momentum']   = 1.10
            weights['reversal']   = 1.10
            weights['trend']      = 0.60

        # Win-rate based tuning (scale all by relative success)
        try:
            stats = self._memory.get_pattern_stats(regime=regime, lookback_days=30)
            if stats.get('total', 0) >= 5:
                wr = stats.get('win_rate', 0.5)
                # If regime is losing, raise uncertainty and reduce all directional weights
                if wr < 0.40:
                    for k in weights:
                        weights[k] *= 0.85
        except Exception:
            pass

        return weights

    # ── Memory helpers ────────────────────────────────────────────────────────

    def _memory_adjustment(self, regime: str, direction: str) -> float:
        """
        Returns a multiplier [0.70, 1.30] based on how well this
        regime+direction has historically performed.
        """
        if self._memory is None or direction == 'HOLD':
            return 1.0
        try:
            wr = self._memory.get_regime_win_rate(regime)
            # Map win_rate → multiplier: 50% → 1.0, 70% → 1.20, 30% → 0.80
            adj = 0.6 + wr * 0.8
            return float(np.clip(adj, 0.70, 1.30))
        except Exception:
            return 1.0

    def _failure_similarity(
        self, signals: List[str], regime: str, direction: str
    ) -> float:
        if self._memory is None or direction == 'HOLD':
            return 0.0
        try:
            return self._memory.get_failure_similarity(signals, regime, direction)
        except Exception:
            return 0.0

    # ── Reversal probability ──────────────────────────────────────────────────

    def _estimate_reversal_prob(self, ctx: BrainContext) -> float:
        """
        Simple heuristic estimate of current reversal probability.
        """
        narr  = ctx.mi_narrative
        score = 0.0

        if getattr(narr, 'reversal_detected', False):        score += 0.35
        if getattr(narr, 'rsi_divergence_bull', False) or \
           getattr(narr, 'rsi_divergence_bear', False):      score += 0.15
        if getattr(narr, 'macd_divergence_bull', False) or \
           getattr(narr, 'macd_divergence_bear', False):     score += 0.15

        bos = getattr(narr, 'bos_choch', {}) or {}
        if bos.get('choch_bull') or bos.get('choch_bear'):  score += 0.20

        if getattr(narr, 'volatility_climax', False):        score += 0.10
        if getattr(narr, 'liquidity_sweep_bull', False) or \
           getattr(narr, 'liquidity_sweep_bear', False):     score += 0.10

        # Exhaustion RSI
        if ctx.rsi is not None:
            if ctx.rsi >= 75 or ctx.rsi <= 25:              score += 0.10

        return float(np.clip(score, 0.0, 1.0))

    # ── Risk state ────────────────────────────────────────────────────────────

    def _assess_risk_state(
        self,
        uncertainty: float,
        reversal_prob: float,
        adx: float,
    ):
        """
        Returns (risk_state, risk_multiplier_adj) tuple.
        """
        if uncertainty >= 0.80 or reversal_prob >= 0.70:
            return 'extreme', 0.0
        if uncertainty >= 0.65 or reversal_prob >= 0.55:
            return 'high', 0.50
        if uncertainty >= 0.50 or reversal_prob >= 0.40:
            return 'elevated', 0.75
        return 'normal', 1.0

    # ── Reasoning builder ─────────────────────────────────────────────────────

    def _build_reasoning(
        self,
        result: BrainDecision,
        votes:  List[dict],
        ua:     UncertaintyAssessment,
    ) -> List[str]:
        lines = []

        # Agent consensus summary
        dir_votes = [v for v in votes if v.get('vote') == result.decision]
        opp_votes = [v for v in votes if v.get('vote') not in (result.decision, 'HOLD')]
        if dir_votes:
            lines.append(
                f"{len(dir_votes)}/{len(votes)} agents agree: "
                + ', '.join(v['name'] for v in dir_votes)
            )
        if opp_votes:
            lines.append(
                f"Opposition: " + ', '.join(
                    f"{v['name']}({v.get('vote')})" for v in opp_votes
                )
            )

        # Regime
        lines.append(f"Regime: {result.market_regime}")

        # Quality metrics
        lines.append(
            f"Confidence={result.confidence:.0%} "
            f"Uncertainty={result.uncertainty:.0%} "
            f"Quality={result.setup_quality:.0%}"
        )

        # Reversal note
        if result.reversal_probability >= 0.40:
            lines.append(f"Reversal risk: {result.reversal_probability:.0%}")

        # Uncertainty breakdown
        if ua.dominant_source:
            lines.append(f"Uncertainty driver: {ua.dominant_source}")

        # Hold reasons
        for hr in result.hold_reasons:
            lines.append(f"HOLD reason: {hr}")

        # Top per-agent reasons
        for v in votes:
            if v.get('reason'):
                lines.append(f"  [{v['name']}] {v['reason']}")

        return lines
