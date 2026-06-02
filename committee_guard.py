"""
committee_guard.py - Institutional committee gate before live execution.

This layer borrows the useful shape of agentic trading desks without adding
cloud dependencies: a Director, Quant, Risk, and Execution review must agree
before an MT5 order is allowed through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


logger = logging.getLogger('AI-Trade')


@dataclass
class CommitteeVote:
    name: str
    verdict: str = 'APPROVE'  # APPROVE | CAUTION | VETO
    score: float = 0.0
    risk_multiplier: float = 1.0
    reasons: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'verdict': self.verdict,
            'score': round(self.score, 3),
            'risk_multiplier': round(self.risk_multiplier, 3),
            'reasons': self.reasons,
            'blockers': self.blockers,
        }


@dataclass
class CommitteeVerdict:
    approved: bool
    verdict: str
    score: float
    min_score: float
    risk_multiplier: float
    reasons: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    votes: List[CommitteeVote] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'approved': self.approved,
            'verdict': self.verdict,
            'score': round(self.score, 3),
            'min_score': round(self.min_score, 3),
            'risk_multiplier': round(self.risk_multiplier, 3),
            'reasons': self.reasons,
            'blockers': self.blockers,
            'votes': [v.to_dict() for v in self.votes],
        }


class InvestmentCommitteeGuard:
    """
    Final structured review before an order is sent to MT5.

    Earlier modules still do the heavy lifting. This guard exists to make the
    last step fail-closed when thesis, quant quality, risk, or execution sanity
    disagree.
    """

    DEFAULTS: Dict[str, Any] = {
        'enabled': True,
        'fail_closed': True,
        'min_score': 0.62,
        'min_score_safe': 0.74,
        'min_agent_approvals': 3,
        'min_adjusted_score': 0.45,
        'max_uncertainty': 0.72,
        'max_reversal_probability': 0.70,
        'max_noise_score': 0.58,
        'max_chase_score': 0.58,
        'max_spread_atr_ratio': 0.10,
        'warning_spread_atr_ratio': 0.07,
        'htf_conflict_strength': 0.55,
        'blocked_regimes': ['HIGH_VOL', 'EXHAUSTION'],
        'caution_regimes': [
            'RANGE',
            'REVERSAL',
            'LIQUIDITY_GRAB',
            'ACCUMULATION',
            'DISTRIBUTION',
        ],
    }

    _WEIGHTS = {
        'director': 0.25,
        'quant': 0.25,
        'risk': 0.30,
        'execution': 0.20,
    }

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self._root_cfg = config
        raw = config.get('committee_guard', {}) or {}
        self._cfg = {**self.DEFAULTS, **raw}

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.get('enabled', True))

    def review(
        self,
        *,
        symbol: str,
        signal: str,
        final_score: float,
        adjusted_score: float,
        grade: str,
        bootstrap_score: float,
        effective_ai: float,
        brain_decision: Any,
        mi_narrative: Any,
        htf_bias: str,
        htf_strength: float,
        adx: float,
        rsi: float,
        atr: float,
        close_price: float,
        spread_price: float,
        rr: float,
        lot_size: float,
        open_count: int,
        max_trades: int,
        noise_score: float,
        chase_score: float,
        drawdown: float,
        session: str,
        autonomy_level: int,
    ) -> CommitteeVerdict:
        if not self.enabled:
            return CommitteeVerdict(
                approved=True,
                verdict='BYPASS',
                score=1.0,
                min_score=0.0,
                risk_multiplier=1.0,
                reasons=['committee_guard disabled'],
            )

        try:
            votes = [
                self._director_vote(
                    signal, adjusted_score, grade, mi_narrative,
                    htf_bias, htf_strength,
                ),
                self._quant_vote(
                    final_score, adjusted_score, bootstrap_score,
                    effective_ai, mi_narrative, adx, rsi,
                    noise_score, chase_score,
                ),
                self._risk_vote(
                    brain_decision, drawdown, open_count, max_trades,
                    autonomy_level,
                ),
                self._execution_vote(
                    signal, rr, lot_size, atr, close_price, spread_price,
                ),
            ]
            return self._combine(votes, symbol, session)
        except Exception as exc:
            logger.warning(f"Committee review failed: {exc}")
            fail_closed = bool(self._cfg.get('fail_closed', True))
            return CommitteeVerdict(
                approved=not fail_closed,
                verdict='FAIL_CLOSED' if fail_closed else 'FAIL_OPEN',
                score=0.0 if fail_closed else 1.0,
                min_score=self._min_score(),
                risk_multiplier=0.0 if fail_closed else 1.0,
                blockers=[f"committee exception: {exc}"],
            )

    def _director_vote(
        self,
        signal: str,
        adjusted_score: float,
        grade: str,
        mi_narrative: Any,
        htf_bias: str,
        htf_strength: float,
    ) -> CommitteeVote:
        vote = CommitteeVote(name='director')
        reasons: List[str] = []
        blockers: List[str] = []

        if signal not in ('BUY', 'SELL'):
            blockers.append(f"invalid entry signal: {signal}")

        setup_quality = float(getattr(mi_narrative, 'setup_quality', 0.5))
        regime = str(getattr(mi_narrative, 'regime', '') or '').upper()
        blocked_regimes = {r.upper() for r in self._cfg.get('blocked_regimes', [])}
        caution_regimes = {r.upper() for r in self._cfg.get('caution_regimes', [])}

        if regime in blocked_regimes:
            blockers.append(f"blocked regime: {regime}")
        elif regime in caution_regimes:
            reasons.append(f"caution regime: {regime}")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.80)

        if signal == 'BUY' and bool(getattr(mi_narrative, 'block_buy', False)):
            blockers.append("MarketIntelligence blocks BUY")
        if signal == 'SELL' and bool(getattr(mi_narrative, 'block_sell', False)):
            blockers.append("MarketIntelligence blocks SELL")

        htf_score = 0.65
        conflict_strength = float(self._cfg.get('htf_conflict_strength', 0.55))
        if htf_bias == signal:
            htf_score = min(1.0, 0.65 + htf_strength * 0.35)
            reasons.append(f"HTF aligned {htf_bias} ({htf_strength:.2f})")
        elif htf_bias == 'NEUTRAL':
            htf_score = 0.55
            reasons.append("HTF neutral")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.90)
        elif htf_strength >= conflict_strength:
            blockers.append(
                f"HTF conflict: {htf_bias} vs {signal} ({htf_strength:.2f})"
            )
        else:
            htf_score = 0.45
            reasons.append(f"weak HTF conflict: {htf_bias} ({htf_strength:.2f})")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.75)

        min_adj = float(self._cfg.get('min_adjusted_score', 0.45))
        if adjusted_score < min_adj:
            blockers.append(f"adjusted_score {adjusted_score:.0%} < {min_adj:.0%}")

        vote.score = self._clip(
            adjusted_score * 0.50 + setup_quality * 0.30 + htf_score * 0.20
        )
        vote.reasons = reasons or [f"grade={grade} setup={setup_quality:.0%}"]
        vote.blockers = blockers
        vote.verdict = self._verdict_from(vote.score, blockers, reasons)
        return vote

    def _quant_vote(
        self,
        final_score: float,
        adjusted_score: float,
        bootstrap_score: float,
        effective_ai: float,
        mi_narrative: Any,
        adx: float,
        rsi: float,
        noise_score: float,
        chase_score: float,
    ) -> CommitteeVote:
        vote = CommitteeVote(name='quant')
        reasons: List[str] = []
        blockers: List[str] = []

        max_noise = float(self._cfg.get('max_noise_score', 0.58))
        max_chase = float(self._cfg.get('max_chase_score', 0.58))
        if noise_score >= max_noise:
            blockers.append(f"noise score {noise_score:.0%} >= {max_noise:.0%}")
        if chase_score >= max_chase:
            blockers.append(f"chase score {chase_score:.0%} >= {max_chase:.0%}")

        if rsi >= 75:
            reasons.append(f"RSI extended high ({rsi:.1f})")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.85)
        elif rsi <= 25:
            reasons.append(f"RSI extended low ({rsi:.1f})")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.85)

        if adx < 18:
            reasons.append(f"ADX weak ({adx:.1f})")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.85)
        elif adx >= 25:
            reasons.append(f"ADX supports trend ({adx:.1f})")

        if bool(getattr(mi_narrative, 'volatility_climax', False)):
            blockers.append("volatility climax active")

        adx_score = self._clip(adx / 35.0)
        clean_score = self._clip(1.0 - max(noise_score, chase_score))
        vote.score = self._clip(
            adjusted_score * 0.30 +
            final_score * 0.25 +
            clean_score * 0.25 +
            adx_score * 0.10 +
            self._clip((bootstrap_score / 100.0 + effective_ai) / 2.0) * 0.10
        )
        vote.reasons = reasons or [
            f"clean={clean_score:.0%} bootstrap={bootstrap_score:.0f}"
        ]
        vote.blockers = blockers
        vote.verdict = self._verdict_from(vote.score, blockers, reasons)
        return vote

    def _risk_vote(
        self,
        brain_decision: Any,
        drawdown: float,
        open_count: int,
        max_trades: int,
        autonomy_level: int,
    ) -> CommitteeVote:
        vote = CommitteeVote(name='risk')
        reasons: List[str] = []
        blockers: List[str] = []

        uncertainty = float(getattr(brain_decision, 'uncertainty', 0.0))
        reversal = float(getattr(brain_decision, 'reversal_probability', 0.0))
        risk_state = str(getattr(brain_decision, 'risk_state', 'normal') or 'normal')

        max_unc = float(self._cfg.get('max_uncertainty', 0.72))
        max_rev = float(self._cfg.get('max_reversal_probability', 0.70))
        if uncertainty >= max_unc:
            blockers.append(f"uncertainty {uncertainty:.0%} >= {max_unc:.0%}")
        if reversal >= max_rev:
            blockers.append(f"reversal probability {reversal:.0%} >= {max_rev:.0%}")
        if risk_state == 'extreme':
            blockers.append("brain risk_state=extreme")
        elif risk_state == 'high':
            reasons.append("brain risk_state=high")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.65)
        elif risk_state == 'elevated':
            reasons.append("brain risk_state=elevated")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.80)

        if open_count >= max_trades:
            blockers.append(f"capacity reached {open_count}/{max_trades}")

        risk_cfg = self._root_cfg.get('risk', {}) or {}
        max_dd = float(risk_cfg.get('max_drawdown', 0.08))
        if max_dd > 0 and drawdown >= max_dd:
            blockers.append(f"drawdown {drawdown:.0%} >= max {max_dd:.0%}")
        elif max_dd > 0 and drawdown >= max_dd * 0.70:
            reasons.append(f"drawdown pressure {drawdown:.0%}")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.70)

        if autonomy_level <= 2:
            reasons.append(f"AI advisory only L{autonomy_level}")

        safety_score = 1.0 - max(
            uncertainty * 0.45,
            reversal * 0.45,
            (drawdown / max(max_dd, 0.01)) * 0.35,
        )
        vote.score = self._clip(safety_score)
        vote.reasons = reasons or ["risk state normal"]
        vote.blockers = blockers
        vote.verdict = self._verdict_from(vote.score, blockers, reasons)
        return vote

    def _execution_vote(
        self,
        signal: str,
        rr: float,
        lot_size: float,
        atr: float,
        close_price: float,
        spread_price: float,
    ) -> CommitteeVote:
        vote = CommitteeVote(name='execution')
        reasons: List[str] = []
        blockers: List[str] = []

        risk_cfg = self._root_cfg.get('risk', {}) or {}
        min_rr = float(risk_cfg.get('min_rr_ratio', 1.8))
        if rr < min_rr:
            blockers.append(f"RR {rr:.2f} < {min_rr:.2f}")

        if lot_size <= 0:
            blockers.append(f"invalid lot size {lot_size}")

        spread_atr = spread_price / atr if atr > 0 else 0.0
        max_spread_atr = float(self._cfg.get('max_spread_atr_ratio', 0.10))
        warn_spread_atr = float(self._cfg.get('warning_spread_atr_ratio', 0.07))
        if spread_atr >= max_spread_atr:
            blockers.append(
                f"spread/ATR {spread_atr:.1%} >= {max_spread_atr:.1%}"
            )
        elif spread_atr >= warn_spread_atr:
            reasons.append(f"spread/ATR elevated {spread_atr:.1%}")
            vote.risk_multiplier = min(vote.risk_multiplier, 0.85)

        if close_price <= 0:
            blockers.append("invalid close price")
        if signal not in ('BUY', 'SELL'):
            blockers.append(f"cannot execute {signal}")

        rr_score = self._clip(rr / max(min_rr * 1.5, 0.01))
        spread_score = self._clip(1.0 - spread_atr / max(max_spread_atr, 0.01))
        lot_score = 1.0 if lot_size > 0 else 0.0
        vote.score = self._clip(rr_score * 0.45 + spread_score * 0.40 + lot_score * 0.15)
        vote.reasons = reasons or [f"RR={rr:.2f} spread/ATR={spread_atr:.1%}"]
        vote.blockers = blockers
        vote.verdict = self._verdict_from(vote.score, blockers, reasons)
        return vote

    def _combine(
        self,
        votes: List[CommitteeVote],
        symbol: str,
        session: str,
    ) -> CommitteeVerdict:
        weighted_score = 0.0
        weight_total = 0.0
        for vote in votes:
            weight = float(self._WEIGHTS.get(vote.name, 0.25))
            weighted_score += vote.score * weight
            weight_total += weight
        score = self._clip(weighted_score / max(weight_total, 0.01))

        blockers = [b for vote in votes for b in vote.blockers]
        reasons = [r for vote in votes for r in vote.reasons]
        approvals = sum(1 for vote in votes if vote.verdict == 'APPROVE')
        cautions = sum(1 for vote in votes if vote.verdict == 'CAUTION')

        min_score = self._min_score()
        min_approvals = int(self._cfg.get('min_agent_approvals', 3))

        if blockers:
            approved = False
            verdict = 'VETO'
        elif approvals < min_approvals:
            approved = False
            verdict = 'NO_CONSENSUS'
            blockers.append(f"approvals {approvals}/{len(votes)} < {min_approvals}")
        elif score < min_score:
            approved = False
            verdict = 'LOW_SCORE'
            blockers.append(f"committee score {score:.0%} < {min_score:.0%}")
        elif cautions:
            approved = True
            verdict = 'APPROVE_WITH_CAUTION'
        else:
            approved = True
            verdict = 'APPROVE'

        risk_multiplier = min([v.risk_multiplier for v in votes] + [1.0])
        if score < min_score + 0.06:
            risk_multiplier = min(risk_multiplier, 0.85)
        if cautions >= 2:
            risk_multiplier = min(risk_multiplier, 0.75)

        logger.debug(
            f"{symbol}: committee {verdict} score={score:.2f} "
            f"session={session} approvals={approvals}/{len(votes)}"
        )

        return CommitteeVerdict(
            approved=approved,
            verdict=verdict,
            score=score,
            min_score=min_score,
            risk_multiplier=self._clip(risk_multiplier),
            reasons=reasons[:8],
            blockers=blockers[:8],
            votes=votes,
        )

    def _min_score(self) -> float:
        risk_cfg = self._root_cfg.get('risk', {}) or {}
        if str(risk_cfg.get('mode', '')).lower() == 'safe':
            return float(self._cfg.get('min_score_safe', 0.74))
        return float(self._cfg.get('min_score', 0.62))

    @staticmethod
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _verdict_from(score: float, blockers: List[str], cautions: List[str]) -> str:
        if blockers:
            return 'VETO'
        caution_terms = (
            'caution',
            'weak',
            'extended',
            'elevated',
            'pressure',
            'neutral',
            'conflict',
            'high',
        )
        has_caution = any(
            any(term in reason.lower() for term in caution_terms)
            for reason in cautions
        )
        if score < 0.55 or has_caution:
            return 'CAUTION'
        return 'APPROVE'
