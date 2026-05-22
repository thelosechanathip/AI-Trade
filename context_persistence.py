"""
context_persistence.py — Context Persistence + Bias Stability Engine.

Solves the "flip-flopping AI" problem: the system used to change its
directional bias every few candles in response to short-term momentum.

How it works:
  - Maintains a slow-decaying EMA of the directional bias per symbol
  - Bias can only flip when the evidence score is >= flip_threshold
  - Tracks how many cycles the current bias has been held
  - Emits the stable_bias and bias_strength for consumption by main.py

The bias_stability_score feeds into the final_score calculation as a
modifier: stable long-held bias → small bonus; recent flip → small penalty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BiasState:
    """Per-symbol persistent bias state."""
    bias:          str   = 'NEUTRAL'  # current held bias
    strength:      float = 0.0        # 0–1, how strong the bias is
    flip_score:    float = 0.5        # rolling evidence for the opposite bias (EMA)
    cycles_held:   int   = 0          # how many cycles this bias has been maintained
    last_flip_ts:  float = field(default_factory=time.time)
    prev_bias:     str   = 'NEUTRAL'


@dataclass
class PersistenceResult:
    stable_bias:       str    # the bias we should act on (may differ from raw signal)
    bias_strength:     float  # 0–1
    cycles_held:       int    # cycles since last flip
    stability_bonus:   float  # -0.05 to +0.05 to add to final_score
    just_flipped:      bool   # True if bias changed this cycle
    signal_matches:    bool   # True if raw signal aligns with stable bias


class ContextPersistenceEngine:
    """
    Maintains slow-decaying directional bias per symbol.

    Parameters
    ----------
    ema_alpha : float
        Speed of evidence accumulation (0.20 = slow adaptation, 0.40 = faster).
        Lower = more stable, less reactive.
    flip_threshold : float
        Evidence score required to flip the bias.  Higher = harder to flip.
    min_cycles_before_flip : int
        Minimum cycles a bias must be held before it can be overridden.
        Prevents immediate reversal of a freshly-set bias.
    """

    def __init__(
        self,
        ema_alpha:              float = 0.20,
        flip_threshold:         float = 0.72,
        min_cycles_before_flip: int   = 3,
    ):
        self.alpha          = ema_alpha
        self.flip_threshold = flip_threshold
        self.min_cycles     = min_cycles_before_flip
        self._states: Dict[str, BiasState] = {}

    def update(
        self,
        symbol:      str,
        raw_signal:  str,     # current cycle's strategy signal ('BUY'/'SELL'/'HOLD')
        htf_bias:    str,     # H4 trend bias
        htf_strength: float,  # 0–1
        ai_confidence: float, # 0–1
        final_score: float,   # 0–1
    ) -> PersistenceResult:
        """
        Update the bias state for this symbol and return a stability assessment.

        Call this AFTER the final_score is computed but BEFORE execution.
        """
        if symbol not in self._states:
            self._states[symbol] = BiasState()

        state = self._states[symbol]

        # ── Build "flip evidence score" for the raw signal direction ──────────
        # Evidence = how strongly current data supports raw_signal
        # HTF agreement is the strongest signal
        htf_match = (raw_signal == htf_bias) if htf_bias != 'NEUTRAL' else False
        evidence  = (
            final_score * 0.40 +
            (htf_strength if htf_match else 0.0) * 0.35 +
            ai_confidence * 0.25
        )

        # Apply EMA smoothing to the flip score
        state.flip_score = (
            self.alpha * evidence + (1.0 - self.alpha) * state.flip_score
        )

        just_flipped = False

        # ── Decide whether to flip the bias ────────────────────────────────────
        if raw_signal in ('BUY', 'SELL') and raw_signal != state.bias:
            can_flip = state.cycles_held >= self.min_cycles
            score_ok = state.flip_score >= self.flip_threshold

            if can_flip and score_ok:
                state.prev_bias    = state.bias
                state.bias         = raw_signal
                state.strength     = evidence
                state.cycles_held  = 0
                state.last_flip_ts = time.time()
                just_flipped       = True
            # else: evidence not strong enough → hold current bias
        elif raw_signal in ('BUY', 'SELL') and raw_signal == state.bias:
            # Signal confirms current bias: strengthen and increment counter
            state.strength    = min(1.0, state.strength * 0.85 + evidence * 0.15)
            state.cycles_held += 1
        elif raw_signal == 'HOLD':
            # HOLD doesn't flip bias but slowly decays strength
            state.strength    = state.strength * 0.92
            state.cycles_held += 1

        # ── Stability bonus / penalty ──────────────────────────────────────────
        # Long-held aligned bias → small bonus (up to +0.05)
        # Recent flip → small penalty (−0.03) to allow settling
        if just_flipped:
            stability_bonus = -0.03
        elif state.cycles_held >= 5 and state.bias == raw_signal:
            stability_bonus = min(0.05, state.cycles_held * 0.008)
        else:
            stability_bonus = 0.0

        signal_matches = (raw_signal == state.bias) or (raw_signal == 'HOLD')

        return PersistenceResult(
            stable_bias     = state.bias,
            bias_strength   = round(state.strength, 3),
            cycles_held     = state.cycles_held,
            stability_bonus = round(stability_bonus, 4),
            just_flipped    = just_flipped,
            signal_matches  = signal_matches,
        )

    def get_state(self, symbol: str) -> BiasState | None:
        return self._states.get(symbol)

    def reset(self, symbol: str) -> None:
        """Reset bias state for a symbol (e.g. after a market regime change)."""
        self._states.pop(symbol, None)
