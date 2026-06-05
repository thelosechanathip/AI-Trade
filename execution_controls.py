"""Persistent, risk-capped operator controls for live order execution."""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger("AI-Trade")

CONTROL_PATH = Path("data/execution_controls.json")
AUDIT_PATH = Path("data/execution_controls_audit.jsonl")

LOT_MODES = {"risk_split", "fixed_capped"}
_LOCK = threading.Lock()


class ExecutionControlError(ValueError):
    """Raised when an operator control request violates execution policy."""


class ExecutionControlConflict(ExecutionControlError):
    """Raised when an update uses an outdated control revision."""


class ExecutionControlStore:
    """Atomic JSON control store shared by the engine and dashboard API."""

    def __init__(
        self,
        config: dict | None = None,
        path: Path = CONTROL_PATH,
        audit_path: Path = AUDIT_PATH,
    ):
        self._path = Path(path)
        self._audit_path = Path(audit_path)
        self._policy = self._build_policy(config or {})

    @staticmethod
    def _build_policy(config: dict) -> dict:
        raw = config.get("execution_controls", {}) or {}
        max_count = max(1, int(raw.get("max_order_count", 4)))
        max_lot = max(0.01, float(raw.get("max_lot_per_order", 0.05)))
        max_total = max(max_lot, float(raw.get("max_total_lot_per_signal", 0.12)))
        default_count = min(max_count, max(1, int(raw.get("default_order_count", 1))))
        default_lot = min(max_lot, max(0.01, float(raw.get("default_lot_per_order", 0.01))))
        default_mode = str(raw.get("default_lot_mode", "risk_split"))
        if default_mode not in LOT_MODES:
            default_mode = "risk_split"
        return {
            "max_order_count": max_count,
            "max_lot_per_order": round(max_lot, 4),
            "max_total_lot_per_signal": round(max_total, 4),
            "default_order_count": default_count,
            "default_lot_per_order": round(default_lot, 4),
            "default_lot_mode": default_mode,
        }

    def _defaults(self) -> dict:
        return {
            "trading_enabled": True,
            "order_count": self._policy["default_order_count"],
            "lot_mode": self._policy["default_lot_mode"],
            "lot_per_order": self._policy["default_lot_per_order"],
            "revision": 0,
            "updated_at": "",
            "updated_by": "defaults",
        }

    def _validate(self, value: dict, *, strict: bool) -> dict:
        controls = self._defaults()
        controls.update({k: value[k] for k in controls if k in value})

        try:
            if strict and not isinstance(controls["trading_enabled"], bool):
                raise TypeError("trading_enabled must be boolean")
            controls["trading_enabled"] = bool(controls["trading_enabled"])
            controls["order_count"] = int(controls["order_count"])
            controls["lot_mode"] = str(controls["lot_mode"])
            controls["lot_per_order"] = round(float(controls["lot_per_order"]), 4)
            controls["revision"] = max(0, int(controls["revision"]))
            controls["updated_at"] = str(controls["updated_at"])
            controls["updated_by"] = str(controls["updated_by"])
        except (TypeError, ValueError) as exc:
            raise ExecutionControlError(f"invalid execution control type: {exc}") from exc

        violations = []
        if not 1 <= controls["order_count"] <= self._policy["max_order_count"]:
            violations.append(
                f"order_count must be 1..{self._policy['max_order_count']}"
            )
        if controls["lot_mode"] not in LOT_MODES:
            violations.append(f"lot_mode must be one of {sorted(LOT_MODES)}")
        if not 0.01 <= controls["lot_per_order"] <= self._policy["max_lot_per_order"]:
            violations.append(
                f"lot_per_order must be 0.01..{self._policy['max_lot_per_order']:.2f}"
            )

        if violations and strict:
            raise ExecutionControlError("; ".join(violations))
        if violations:
            safe = self._defaults()
            safe["trading_enabled"] = False
            safe["validation_error"] = "; ".join(violations)
            controls = safe
        elif value.get("validation_error"):
            controls["validation_error"] = str(value["validation_error"])

        controls["guardrails"] = dict(self._policy)
        return controls

    def get(self) -> dict:
        if not self._path.exists():
            return self._validate(self._defaults(), strict=True)
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ExecutionControlError("control file must contain an object")
            return self._validate(raw, strict=False)
        except Exception as exc:
            logger.error(f"Execution controls fail-closed: {exc}")
            safe = self._defaults()
            safe["trading_enabled"] = False
            safe["validation_error"] = f"control file unreadable: {exc}"
            return self._validate(safe, strict=False)

    def update(self, patch: dict, *, actor: str = "dashboard") -> dict:
        if not isinstance(patch, dict):
            raise ExecutionControlError("request body must be an object")

        allowed = {"trading_enabled", "order_count", "lot_mode", "lot_per_order"}
        unknown = sorted(set(patch) - allowed - {"expected_revision"})
        if unknown:
            raise ExecutionControlError(f"unknown controls: {', '.join(unknown)}")

        with _LOCK:
            current = self.get()
            expected = patch.get("expected_revision")
            if expected is not None and int(expected) != int(current["revision"]):
                raise ExecutionControlConflict(
                    f"revision changed: expected {expected}, current {current['revision']}"
                )

            candidate = {k: current[k] for k in self._defaults()}
            candidate.update({k: patch[k] for k in allowed if k in patch})
            candidate["revision"] = int(current["revision"]) + 1
            candidate["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            candidate["updated_by"] = actor[:80]
            validated = self._validate(candidate, strict=True)

            persisted = {k: validated[k] for k in self._defaults()}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(persisted, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp_path.replace(self._path)

            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit = {
                "ts": candidate["updated_at"],
                "actor": candidate["updated_by"],
                "revision": candidate["revision"],
                "before": {k: current.get(k) for k in allowed},
                "after": {k: validated.get(k) for k in allowed},
            }
            with self._audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(audit, sort_keys=True) + "\n")

            return validated


def _normalize_volume_down(volume: float, symbol_info) -> float:
    try:
        step = float(symbol_info.volume_step)
        vol_min = float(symbol_info.volume_min)
        vol_max = float(symbol_info.volume_max)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if step <= 0 or vol_min <= 0 or vol_max < vol_min or volume <= 0:
        return 0.0
    normalized = math.floor((min(float(volume), vol_max) + 1e-12) / step) * step
    if normalized + 1e-12 < vol_min:
        return 0.0
    return round(normalized, 8)


def build_order_plan(
    total_lot_budget: float,
    controls: dict,
    symbol_info: Any,
    capacity: int,
) -> list[float]:
    """
    Convert one aggregate risk-sized lot budget into a guarded batch plan.

    Fixed mode preserves the requested lot per order where possible. Risk-split
    mode divides the aggregate risk budget evenly across the requested orders.
    Both modes remain capped by broker constraints and execution policy.
    """
    if not controls.get("trading_enabled", False) or capacity <= 0:
        return []

    policy = controls.get("guardrails", {}) or {}
    requested_count = max(1, int(controls.get("order_count", 1)))
    count = min(
        requested_count,
        int(capacity),
        max(1, int(policy.get("max_order_count", requested_count))),
    )
    total_cap = min(
        max(0.0, float(total_lot_budget)),
        max(0.0, float(policy.get("max_total_lot_per_signal", total_lot_budget))),
    )
    max_each = max(0.0, float(policy.get("max_lot_per_order", total_cap)))
    if count <= 0 or total_cap <= 0 or max_each <= 0:
        return []

    if controls.get("lot_mode") == "fixed_capped":
        requested_each = _normalize_volume_down(
            min(float(controls.get("lot_per_order", 0.0)), max_each),
            symbol_info,
        )
        if requested_each <= 0:
            return []
        affordable = int((total_cap + 1e-12) // requested_each)
        planned_count = min(count, affordable)
        if planned_count > 0:
            return [requested_each] * planned_count

        capped_each = _normalize_volume_down(min(total_cap, max_each), symbol_info)
        return [capped_each] if capped_each > 0 else []

    while count > 0:
        each = _normalize_volume_down(min(total_cap / count, max_each), symbol_info)
        if each > 0 and each * count <= total_cap + 1e-9:
            return [each] * count
        count -= 1
    return []
