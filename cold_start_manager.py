"""
cold_start_manager.py — Progressive Autonomy Level Manager.

Tracks system readiness across 5 dimensions and automatically upgrades
(or downgrades) the AI autonomy level.

Autonomy Levels:
  0  RULE_BASED           — Strategy + MI only; Brain logs but never blocks
  1  RULE_PLUS_AI_FILTER  — Brain can reduce lot size; blocks only in emergencies
  2  AI_ASSISTED          — Brain influences sizing with bootstrap confidence
  3  SEMI_AUTONOMOUS      — Brain has significant entry/exit influence
  4  FULL_AUTONOMOUS      — Brain is primary decision maker

Upgrade Criteria (cumulative — each level adds to the previous):
  L0 → L1 : 10 trades
  L1 → L2 : 30 trades, AUC ≥ 0.48, win_rate ≥ 40%
  L2 → L3 : 75 trades, AUC ≥ 0.52, win_rate ≥ 45%, DD ≤ 10%
  L3 → L4 : 150 trades, AUC ≥ 0.55, win_rate ≥ 50%, DD ≤ 8%

Downgrade Triggers:
  5+ consecutive losses OR drawdown ≥ 8% → drop one level
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger('AI-Trade')

AUTONOMY_NAMES: Dict[int, str] = {
    0: 'RULE_BASED',
    1: 'RULE_PLUS_AI_FILTER',
    2: 'AI_ASSISTED',
    3: 'SEMI_AUTONOMOUS',
    4: 'FULL_AUTONOMOUS',
}

# Upgrade criteria: values the system must EXCEED to unlock the next level
_UPGRADE_CRITERIA: Dict[int, dict] = {
    1: {'min_trades': 10,  'min_auc': 0.00, 'min_winrate': 0.00, 'max_dd': 0.20},
    2: {'min_trades': 30,  'min_auc': 0.48, 'min_winrate': 0.40, 'max_dd': 0.12},
    3: {'min_trades': 75,  'min_auc': 0.52, 'min_winrate': 0.45, 'max_dd': 0.10},
    4: {'min_trades': 150, 'min_auc': 0.55, 'min_winrate': 0.50, 'max_dd': 0.08},
}

# Risk (lot) scale per level — conservative during early levels
_RISK_SCALES: Dict[int, float] = {0: 0.60, 1: 0.70, 2: 0.80, 3: 0.90, 4: 1.00}

# Bootstrap vs AI confidence blend weight (1.0 = 100% bootstrap, 0.0 = 100% AI)
_BOOTSTRAP_WEIGHTS: Dict[int, float] = {0: 1.00, 1: 0.80, 2: 0.55, 3: 0.30, 4: 0.00}

# Minimum simultaneous negative conditions needed for Brain to force HOLD
# Higher number = harder to trigger HOLD (more permissive)
_MIN_HOLD_CONDITIONS: Dict[int, int] = {0: 99, 1: 4, 2: 3, 3: 2, 4: 1}

_STATE_FILE = Path('data/autonomy_state.json')


class ColdStartManager:
    """
    Manages progressive autonomy levels.
    Call `check_upgrade()` periodically (e.g. each main loop cycle).
    """

    def __init__(self, config: dict = None):
        cfg = (config or {}).get('progressive_autonomy', {})
        self._level        = int(cfg.get('initial_level', 0))
        self._enabled      = bool(cfg.get('enabled', True))
        self._max_level    = int(cfg.get('max_level', 4))
        self._load_state()
        logger.info(
            f"ColdStartManager: initialized at level {self._level} "
            f"({self.level_name})"
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        if _STATE_FILE.exists():
            try:
                s = json.loads(_STATE_FILE.read_text())
                self._level = int(s.get('autonomy_level', self._level))
                logger.info(
                    f"ColdStartManager: restored level {self._level} "
                    f"({self.level_name}) from {_STATE_FILE}"
                )
            except Exception as exc:
                logger.warning(f"ColdStartManager: could not load state: {exc}")

    def _save_state(self) -> None:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps({'autonomy_level': self._level}))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_level(self) -> int:
        return self._level

    @property
    def level_name(self) -> str:
        return AUTONOMY_NAMES.get(self._level, 'UNKNOWN')

    @property
    def is_cold_start(self) -> bool:
        return self._level <= 1

    def get_risk_scale(self) -> float:
        """Lot-size multiplier for the current level."""
        return _RISK_SCALES.get(self._level, 1.0)

    def get_bootstrap_weight(self) -> float:
        """
        Weight of bootstrap confidence vs AI model confidence.
        1.0 = 100% bootstrap (AI cold), 0.0 = 100% AI (fully trained).
        """
        return _BOOTSTRAP_WEIGHTS.get(self._level, 0.50)

    def get_min_hold_conditions(self) -> int:
        """
        How many simultaneous negative conditions the Brain needs to force a HOLD.
        At level 0: practically impossible (99) — strategy always wins.
        At level 4: just 1 condition suffices.
        """
        return _MIN_HOLD_CONDITIONS.get(self._level, 3)

    def brain_override_allowed(self) -> bool:
        """Can the Brain replace the strategy signal entirely?"""
        return self._level >= 3

    def get_min_score_threshold(self) -> float:
        """
        Minimum final_trade_score required to proceed at this level.
        Lower levels are more permissive (trust strategy more).
        """
        thresholds = {0: 0.15, 1: 0.20, 2: 0.28, 3: 0.35, 4: 0.40}
        return thresholds.get(self._level, 0.25)

    # ── Upgrade / downgrade ───────────────────────────────────────────────────

    def check_upgrade(
        self,
        total_trades:      int,
        win_rate:          float,   # 0–1
        auc:               float,   # 0–1
        current_drawdown:  float,   # 0–1
    ) -> bool:
        """
        Check if the system should advance to the next autonomy level.
        Returns True if an upgrade happened.
        """
        if not self._enabled or self._level >= min(4, self._max_level):
            return False

        next_level = self._level + 1
        crit = _UPGRADE_CRITERIA.get(next_level, {})

        passes = (
            total_trades     >= crit.get('min_trades',   9999) and
            auc              >= crit.get('min_auc',       1.0) and
            win_rate         >= crit.get('min_winrate',   1.0) and
            current_drawdown <= crit.get('max_dd',        0.0)
        )

        if passes:
            old_name    = self.level_name
            self._level = next_level
            self._save_state()
            logger.info(
                f"ColdStartManager: AUTONOMY UPGRADE "
                f"{old_name} → {self.level_name} "
                f"(trades={total_trades}, win={win_rate:.0%}, "
                f"AUC={auc:.3f}, DD={current_drawdown:.2%})"
            )
            return True
        return False

    def check_downgrade(
        self,
        consecutive_losses: int,
        current_drawdown:   float,
    ) -> bool:
        """
        Automatically downgrade if risk metrics deteriorate.
        Returns True if a downgrade happened.
        """
        if not self._enabled or self._level <= 1:
            return False

        should_down = (
            consecutive_losses >= 5 or
            current_drawdown   >= 0.08
        )

        if should_down:
            old_name    = self.level_name
            self._level = max(1, self._level - 1)
            self._save_state()
            logger.warning(
                f"ColdStartManager: AUTONOMY DOWNGRADE "
                f"{old_name} → {self.level_name} "
                f"(losses={consecutive_losses}, DD={current_drawdown:.2%})"
            )
            return True
        return False

    def get_status(self) -> dict:
        return {
            'autonomy_level':         self._level,
            'level_name':             self.level_name,
            'is_cold_start':          self.is_cold_start,
            'risk_scale':             self.get_risk_scale(),
            'bootstrap_weight':       self.get_bootstrap_weight(),
            'min_hold_conditions':    self.get_min_hold_conditions(),
            'brain_override_allowed': self.brain_override_allowed(),
            'min_score_threshold':    self.get_min_score_threshold(),
        }
