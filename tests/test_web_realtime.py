import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import web_app


def ns(**values):
    return SimpleNamespace(**values)


class FakeMt5:
    POSITION_TYPE_BUY = 0
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2
    DEAL_ENTRY_OUT_BY = 3

    def __init__(self, account=True):
        self._account = ns(
            login=123,
            server="Demo",
            currency="USD",
            balance=1514.25,
            equity=1520.50,
            margin=10.0,
            margin_level=15205.0,
            margin_free=1510.50,
        ) if account else None
        self._positions = [
            ns(
                ticket=42,
                symbol="XAUUSD.v",
                type=0,
                volume=0.02,
                price_open=4440.0,
                price_current=4447.34,
                sl=4430.0,
                tp=4470.0,
                profit=6.25,
                swap=0.0,
                magic=20240101,
                time=0,
            )
        ]

    def account_info(self):
        return self._account

    def positions_get(self):
        return self._positions

    def history_deals_get(self, _start, _end):
        return [
            ns(
                type=0,
                entry=1,
                position_id=7,
                profit=-10.0,
                commission=-0.5,
                swap=0.0,
                fee=0.0,
            )
        ]

    def symbol_info(self, symbol):
        return ns(point=0.01) if symbol == "XAUUSD.v" else None

    def symbols_get(self, group=None):
        return [ns(name="XAUUSD.v")]

    def symbol_info_tick(self, symbol):
        if symbol != "XAUUSD.v":
            return None
        return ns(bid=4447.34, ask=4447.96, time_msc=1_780_000_000_000)

    def last_error(self):
        return (1, "not connected")


class LiveMt5OverlayTests(unittest.TestCase):
    def setUp(self):
        web_app._MT5_SYMBOL_CACHE.clear()
        web_app._LIVE_EQUITY_POINTS.clear()

    def test_live_mt5_overlays_account_tick_and_positions(self):
        fake = FakeMt5()
        state = {
            "timestamp": "old",
            "peak_balance": 1600.0,
            "terminal": {
                "XAUUSD": {
                    "price": 4000.0,
                    "spread_pips": 0.0,
                    "signal": "HOLD",
                    "atr": 10.0,
                }
            },
            "open_trades": [{"ticket": 42, "ai_confidence": 77}],
            "stats": {"weekly_pnl": 5.0},
        }

        with patch.dict(sys.modules, {"MetaTrader5": fake}):
            live = web_app._with_live_mt5(state)

        self.assertTrue(live["live"]["connected"])
        self.assertEqual(live["live"]["source"], "mt5")
        self.assertEqual(live["balance"], 1514.25)
        self.assertEqual(live["equity"], 1520.50)
        self.assertEqual(live["terminal"]["XAUUSD"]["price"], 4447.34)
        self.assertEqual(live["terminal"]["XAUUSD"]["ask"], 4447.96)
        self.assertEqual(live["terminal"]["XAUUSD"]["broker_symbol"], "XAUUSD.v")
        self.assertEqual(live["open_positions"][0]["profit"], 6.25)
        self.assertEqual(live["open_positions"][0]["ai_confidence"], 77)
        self.assertEqual(live["daily_pnl"], -4.25)
        self.assertEqual(live["equity_recent"][-1]["equity"], 1520.50)

    def test_live_mt5_falls_back_to_snapshot_when_account_unavailable(self):
        fake = FakeMt5(account=False)
        state = {"timestamp": "old", "balance": 100.0}

        with patch.dict(sys.modules, {"MetaTrader5": fake}):
            live = web_app._with_live_mt5(state)

        self.assertFalse(live["live"]["connected"])
        self.assertEqual(live["live"]["source"], "snapshot")
        self.assertEqual(live["balance"], 100.0)


if __name__ == "__main__":
    unittest.main()
