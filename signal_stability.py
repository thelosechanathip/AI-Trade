"""
signal_stability.py — Signal Persistence Filter.

Prevents the system from acting on a signal that appeared only once.
Requires N consecutive cycles of the same direction before allowing entry.

This kills "candle-reaction" trading where the strategy fires on one
aggressive candle and immediately reverses the next cycle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class StabilityResult:
    is_stable:    bool   # True = signal has been consistent enough to act on
    signal:       str    # The signal that is stable ('BUY', 'SELL', or 'HOLD')
    count:        int    # Consecutive cycles with this signal
    required:     int    # Cycles required to unlock entry
    flip_count:   int    # How many times signal changed in the window


class SignalStabilityTracker:
    """
    Tracks the last N signal readings per symbol and gatekeeps entry
    until the signal has appeared consistently for `required_cycles`.

    Design decisions:
    - 'HOLD' always passes (not blocking; just absence of signal)
    - A single flip-back resets the counter for the new direction
    - window_size controls how far back we remember
    """

    def __init__(self, required_cycles: int = 2, window_size: int = 5):
        self._required  = max(1, required_cycles)
        self._window    = max(self._required, window_size)
        # symbol -> deque of last N signals (newest at right)
        self._history: Dict[str, Deque[str]] = {}

    def record(self, symbol: str, signal: str) -> None:
        """Record the current signal for this symbol."""
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._window)
        self._history[symbol].append(signal)

    def check(self, symbol: str, signal: str) -> StabilityResult:
        """
        Check whether the current signal is stable enough to act on.

        HOLD is always considered stable (we never want to block a non-trade).
        A directional signal (BUY/SELL) requires `required_cycles` consecutive
        appearances in the history before it is considered stable.
        """
        if signal == 'HOLD':
            return StabilityResult(True, 'HOLD', 0, 0, 0)

        hist = list(self._history.get(symbol, []))
        if not hist:
            # No history yet — first appearance, not stable
            return StabilityResult(False, signal, 1, self._required, 0)

        # Count trailing consecutive matches (from the end of history)
        consecutive = 0
        for s in reversed(hist):
            if s == signal:
                consecutive += 1
            else:
                break

        # Count how many times the signal flipped within the window
        flips = 0
        for i in range(1, len(hist)):
            if hist[i] != hist[i - 1] and hist[i - 1] != 'HOLD':
                flips += 1

        is_stable = consecutive >= self._required

        return StabilityResult(
            is_stable = is_stable,
            signal    = signal,
            count     = consecutive,
            required  = self._required,
            flip_count= flips,
        )

    def reset(self, symbol: str) -> None:
        """Clear history for a symbol (e.g. after a trade is opened)."""
        self._history.pop(symbol, None)

    def get_history(self, symbol: str) -> list:
        return list(self._history.get(symbol, []))
