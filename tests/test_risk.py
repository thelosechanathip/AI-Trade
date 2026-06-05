import unittest
from types import SimpleNamespace

from risk import RiskManager


def make_risk_manager() -> RiskManager:
    manager = RiskManager.__new__(RiskManager)
    manager._cfg = {
        "risk_per_trade": 0.05,
        "risk_per_trade_hard_cap": 0.005,
        "use_kelly": True,
        "kelly_min_trades": 250,
        "mode": "normal",
        "max_drawdown": 0.10,
        "dd_scale_start": 0.03,
        "dd_scale_min": 0.25,
        "max_loss_streak": 3,
        "loss_streak_scale": 0.50,
    }
    manager.peak_balance = 10_000.0
    manager._loss_streak = 0
    return manager


class RiskManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = make_risk_manager()
        self.symbol = SimpleNamespace(
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            volume_step=0.01,
            volume_min=0.01,
            volume_max=100.0,
        )

    def test_hard_cap_limits_effective_risk(self):
        lot = self.manager.calculate_lot_size(
            balance=10_000.0,
            sl_distance=10.0,
            symbol_info=self.symbol,
            win_rate=80.0,
            avg_win=500.0,
            avg_loss=100.0,
            sample_count=500,
        )
        self.assertEqual(lot, 0.05)

    def test_invalid_symbol_metadata_fails_closed(self):
        lot = self.manager.calculate_lot_size(10_000.0, 10.0, None)
        self.assertEqual(lot, 0.0)

    def test_below_minimum_lot_fails_closed(self):
        tiny_account = self.manager.calculate_lot_size(
            balance=100.0,
            sl_distance=100.0,
            symbol_info=self.symbol,
        )
        self.assertEqual(tiny_account, 0.0)

    def test_estimate_risk_amount(self):
        amount = self.manager.estimate_risk_amount(0.05, 10.0, self.symbol)
        self.assertEqual(amount, 50.0)
        self.assertEqual(
            self.manager.estimate_risk_amount(0.05, 10.0, None),
            float("inf"),
        )


if __name__ == "__main__":
    unittest.main()
