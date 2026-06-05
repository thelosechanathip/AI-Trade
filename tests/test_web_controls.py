import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import web_app
from execution_controls import ExecutionControlStore


class ExecutionControlApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.original_store = web_app.CONTROL_STORE
        web_app.CONTROL_STORE = ExecutionControlStore(
            web_app.ROOT_CONFIG,
            path=root / "controls.json",
            audit_path=root / "audit.jsonl",
        )
        self.client = TestClient(web_app.app)

    def tearDown(self):
        web_app.CONTROL_STORE = self.original_store
        self.temp_dir.cleanup()

    def test_get_and_update_controls(self):
        initial = self.client.get("/api/execution-controls")
        self.assertEqual(initial.status_code, 200)
        self.assertEqual(initial.json()["revision"], 0)

        updated = self.client.put(
            "/api/execution-controls",
            headers={"x-operator-id": "api-test"},
            json={
                "order_count": 2,
                "lot_mode": "fixed_capped",
                "lot_per_order": 0.02,
                "expected_revision": 0,
            },
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["revision"], 1)
        self.assertEqual(updated.json()["updated_by"], "api-test")

    def test_revision_conflict_returns_409(self):
        self.client.put(
            "/api/execution-controls",
            json={"order_count": 2, "expected_revision": 0},
        )
        conflict = self.client.put(
            "/api/execution-controls",
            json={"order_count": 3, "expected_revision": 0},
        )
        self.assertEqual(conflict.status_code, 409)

    def test_guardrail_violation_returns_422(self):
        response = self.client.put(
            "/api/execution-controls",
            json={"order_count": 999, "expected_revision": 0},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
