import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from execution_controls import (
    ExecutionControlConflict,
    ExecutionControlError,
    ExecutionControlStore,
    build_order_plan,
)


class ExecutionControlTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = ExecutionControlStore(
            {
                "execution_controls": {
                    "max_order_count": 4,
                    "max_lot_per_order": 0.05,
                    "max_total_lot_per_signal": 0.12,
                }
            },
            path=root / "controls.json",
            audit_path=root / "audit.jsonl",
        )
        self.symbol = SimpleNamespace(
            volume_step=0.01,
            volume_min=0.01,
            volume_max=100.0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_update_is_versioned_and_audited(self):
        first = self.store.update(
            {
                "order_count": 3,
                "lot_mode": "fixed_capped",
                "lot_per_order": 0.02,
                "expected_revision": 0,
            },
            actor="test",
        )
        self.assertEqual(first["revision"], 1)
        self.assertEqual(first["order_count"], 3)
        self.assertEqual(first["updated_by"], "test")

        with self.assertRaises(ExecutionControlConflict):
            self.store.update({"order_count": 2, "expected_revision": 0})

        audit = json.loads(self.store._audit_path.read_text(encoding="utf-8").strip())
        self.assertEqual(audit["revision"], 1)
        self.assertEqual(audit["after"]["order_count"], 3)

    def test_invalid_control_is_rejected(self):
        with self.assertRaises(ExecutionControlError):
            self.store.update({"order_count": 5})
        with self.assertRaises(ExecutionControlError):
            self.store.update({"lot_per_order": 0.10})

    def test_corrupt_file_disables_new_entries(self):
        self.store._path.parent.mkdir(parents=True, exist_ok=True)
        self.store._path.write_text("{bad json", encoding="utf-8")
        controls = self.store.get()
        self.assertFalse(controls["trading_enabled"])
        self.assertIn("validation_error", controls)

    def test_risk_split_never_exceeds_aggregate_budget(self):
        controls = self.store.get()
        controls.update({"order_count": 4, "lot_mode": "risk_split"})
        plan = build_order_plan(0.06, controls, self.symbol, capacity=4)
        self.assertEqual(plan, [0.01, 0.01, 0.01, 0.01])
        self.assertLessEqual(sum(plan), 0.06)

    def test_fixed_lot_reduces_count_before_exceeding_budget(self):
        controls = self.store.get()
        controls.update({
            "order_count": 4,
            "lot_mode": "fixed_capped",
            "lot_per_order": 0.02,
        })
        plan = build_order_plan(0.05, controls, self.symbol, capacity=4)
        self.assertEqual(plan, [0.02, 0.02])
        self.assertLessEqual(sum(plan), 0.05)

    def test_disabled_controls_create_no_orders(self):
        controls = self.store.get()
        controls["trading_enabled"] = False
        self.assertEqual(build_order_plan(0.10, controls, self.symbol, capacity=4), [])


if __name__ == "__main__":
    unittest.main()
