"""
exit_intelligence.py — Narrative-driven proactive exit system.

Called every 60 seconds on all open positions.  Evaluates whether the
market narrative has shifted enough to warrant early exit rather than
waiting for SL/TP or the mechanical trade_manager triggers.

ExitSignal actions:
  NONE          — do nothing, let trade manager handle
  CLOSE         — close position immediately
  REDUCE_50     — partial close 50% of remaining volume
  REDUCE_30     — partial close 30% of remaining volume
  TIGHTEN_SL    — move SL to tighter level (supplied as new_sl)
  PARTIAL_EXIT  — alias for REDUCE_30

Priority (highest wins):
  1. Emergency (reversal confirmed, dual divergence against, liquidity sweep against)
  2. Strong (CHOCH opposite, volatility climax in loss)
  3. Moderate (brain regime flip, uncertainty spike + losing)
  4. Light (narrative clarity drops, extended hold + no progress)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger('AI-Trade')


@dataclass
class ExitSignal:
    ticket:      int
    action:      str       # 'NONE'|'CLOSE'|'REDUCE_50'|'REDUCE_30'|'TIGHTEN_SL'
    reason:      str = ''
    urgency:     int = 0   # 0=none, 1=light, 2=moderate, 3=strong, 4=emergency
    new_sl:      float = 0.0   # only populated for TIGHTEN_SL
    confidence:  float = 1.0   # 0-1 confidence in the exit decision

    @property
    def should_act(self) -> bool:
        return self.action != 'NONE'


class ExitIntelligence:
    """
    Evaluates open positions against the current market narrative and
    returns an ExitSignal per position.

    Does NOT execute trades — that is the caller's responsibility.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get('trade_management', {})
        self._urgency_threshold = int(cfg.get('exit_urgency_threshold', 2))

    # ── Main entry ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        *,
        ticket:          int,
        direction:       str,          # 'BUY' | 'SELL'
        entry_price:     float,
        current_price:   float,
        sl_price:        float,
        tp_price:        float,
        atr:             float,
        mi_narrative,                  # MarketNarrative
        brain_decision   = None,       # BrainDecision (optional)
        uncertainty      = None,       # UncertaintyAssessment (optional)
        bars_held:       int = 0,
        profit_r:        float = 0.0,  # current unrealised R
    ) -> ExitSignal:

        candidates: List[ExitSignal] = []

        is_buy  = direction.upper() == 'BUY'
        narr    = mi_narrative
        against = 'SELL' if is_buy else 'BUY'

        # ── Priority 4 — Emergency exits ─────────────────────────────────────

        # Reversal confirmed against position direction
        if getattr(narr, 'reversal_detected', False):
            rev_dir = getattr(narr, 'reversal_direction', '')
            if (is_buy and rev_dir == 'DOWN') or (not is_buy and rev_dir == 'UP'):
                candidates.append(ExitSignal(
                    ticket   = ticket,
                    action   = 'CLOSE',
                    reason   = f"Reversal detected ({rev_dir}) against {direction}",
                    urgency  = 4,
                    confidence = 0.90,
                ))

        # Dual divergence against position
        rsi_div_against  = (
            (is_buy  and getattr(narr, 'rsi_divergence_bear', False)) or
            (not is_buy and getattr(narr, 'rsi_divergence_bull', False))
        )
        macd_div_against = (
            (is_buy  and getattr(narr, 'macd_divergence_bear', False)) or
            (not is_buy and getattr(narr, 'macd_divergence_bull', False))
        )
        if rsi_div_against and macd_div_against:
            candidates.append(ExitSignal(
                ticket   = ticket,
                action   = 'CLOSE',
                reason   = f"Dual divergence (RSI+MACD) against {direction}",
                urgency  = 4,
                confidence = 0.85,
            ))

        # Liquidity sweep against position AND losing
        liq_against = (
            (is_buy  and getattr(narr, 'liquidity_sweep_bear', False)) or
            (not is_buy and getattr(narr, 'liquidity_sweep_bull', False))
        )
        if liq_against and profit_r < -0.3:
            candidates.append(ExitSignal(
                ticket   = ticket,
                action   = 'CLOSE',
                reason   = f"Liquidity sweep against {direction} while losing",
                urgency  = 4,
                confidence = 0.80,
            ))

        # ── Priority 3 — Strong exits ─────────────────────────────────────────

        # CHOCH opposite to position
        bos_data = getattr(narr, 'bos_choch', {}) or {}
        choch_against = (
            (is_buy  and bos_data.get('choch_bear', False)) or
            (not is_buy and bos_data.get('choch_bull', False))
        )
        if choch_against:
            action = 'CLOSE' if profit_r < 0 else 'REDUCE_50'
            candidates.append(ExitSignal(
                ticket   = ticket,
                action   = action,
                reason   = f"CHOCH opposite to {direction}",
                urgency  = 3,
                confidence = 0.75,
            ))

        # Volatility climax while losing
        if getattr(narr, 'volatility_climax', False) and profit_r < -0.2:
            candidates.append(ExitSignal(
                ticket   = ticket,
                action   = 'REDUCE_50',
                reason   = "Volatility climax during loss — reduce exposure",
                urgency  = 3,
                confidence = 0.70,
            ))

        # Brain flipped regime against position
        if brain_decision is not None:
            brain_dir = getattr(brain_decision, 'decision', 'HOLD')
            if brain_dir == against:
                candidates.append(ExitSignal(
                    ticket   = ticket,
                    action   = 'CLOSE' if profit_r < 0 else 'REDUCE_50',
                    reason   = f"Brain now signals {brain_dir} — against open {direction}",
                    urgency  = 3,
                    confidence = min(0.90, getattr(brain_decision, 'confidence', 0.6)),
                ))

        # ── Priority 2 — Moderate exits ──────────────────────────────────────

        # High uncertainty while losing
        if uncertainty is not None:
            u_score = getattr(uncertainty, 'total_score', 0.0)
            if u_score >= 0.70 and profit_r < -0.15:
                candidates.append(ExitSignal(
                    ticket   = ticket,
                    action   = 'REDUCE_30',
                    reason   = f"High uncertainty ({u_score:.0%}) while losing",
                    urgency  = 2,
                    confidence = 0.65,
                ))

        # Single RSI divergence against while losing
        if rsi_div_against and profit_r < -0.3:
            candidates.append(ExitSignal(
                ticket   = ticket,
                action   = 'TIGHTEN_SL',
                reason   = f"RSI divergence against {direction}",
                urgency  = 2,
                new_sl   = self._tighter_sl(
                    is_buy, current_price, sl_price, atr, factor=0.5
                ),
                confidence = 0.60,
            ))

        # Displacement candle against position (institutional move)
        disp_against = (
            (is_buy  and getattr(narr, 'displacement_bear', False)) or
            (not is_buy and getattr(narr, 'displacement_bull', False))
        )
        if disp_against and profit_r < 0:
            candidates.append(ExitSignal(
                ticket   = ticket,
                action   = 'TIGHTEN_SL',
                reason   = f"Displacement candle against {direction}",
                urgency  = 2,
                new_sl   = self._tighter_sl(
                    is_buy, current_price, sl_price, atr, factor=0.6
                ),
                confidence = 0.60,
            ))

        # ── Priority 1 — Light exits ──────────────────────────────────────────

        # Extended hold with no progress
        if bars_held > 36 and abs(profit_r) < 0.20:
            candidates.append(ExitSignal(
                ticket   = ticket,
                action   = 'TIGHTEN_SL',
                reason   = f"Extended hold ({bars_held} bars) with no momentum",
                urgency  = 1,
                new_sl   = self._tighter_sl(
                    is_buy, current_price, sl_price, atr, factor=0.7
                ),
                confidence = 0.50,
            ))

        # ── Select highest urgency ────────────────────────────────────────────
        if not candidates:
            return ExitSignal(ticket=ticket, action='NONE', urgency=0)

        best = max(candidates, key=lambda s: (s.urgency, s.confidence))

        if best.urgency >= self._urgency_threshold:
            logger.info(
                f"ExitIntelligence #{ticket} {direction}: "
                f"{best.action} (urgency={best.urgency}) — {best.reason}"
            )
            return best

        return ExitSignal(ticket=ticket, action='NONE', urgency=0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tighter_sl(
        is_buy:  bool,
        price:   float,
        current_sl: float,
        atr:    float,
        factor: float = 0.5,
    ) -> float:
        """
        Proposes a tighter SL = current price ± (atr × factor).
        Returns 0.0 if it would be worse than current SL.
        """
        if atr <= 0:
            return 0.0

        if is_buy:
            proposed = price - atr * factor
            # Must be higher than current SL to be tighter
            if proposed > current_sl:
                return round(proposed, 5)
        else:
            proposed = price + atr * factor
            if proposed < current_sl:
                return round(proposed, 5)

        return 0.0

    def evaluate_all(
        self,
        positions: list,         # list of dicts with position fields
        mi_narrative,
        brain_decision = None,
        uncertainty    = None,
        atr: float = 0.0,
    ) -> List[ExitSignal]:
        """
        Batch evaluate all open positions.
        positions: list of dicts with keys:
            ticket, direction, entry_price, current_price,
            sl_price, tp_price, bars_held, profit_r
        """
        results = []
        for pos in positions:
            sig = self.evaluate(
                ticket        = pos.get('ticket', 0),
                direction     = pos.get('direction', 'BUY'),
                entry_price   = pos.get('entry_price', 0.0),
                current_price = pos.get('current_price', 0.0),
                sl_price      = pos.get('sl_price', 0.0),
                tp_price      = pos.get('tp_price', 0.0),
                atr           = pos.get('atr', atr),
                mi_narrative  = mi_narrative,
                brain_decision= brain_decision,
                uncertainty   = uncertainty,
                bars_held     = pos.get('bars_held', 0),
                profit_r      = pos.get('profit_r', 0.0),
            )
            results.append(sig)
        return results
