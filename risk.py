"""
risk.py — Position sizing, daily loss tracking, and drawdown enforcement.

Enhancements:
  • Kelly fraction sizing (quarter-Kelly, capped)
  • Drawdown scaling — reduce size as drawdown grows
  • Loss-streak protection — cut size after N consecutive losses
  • Safe / Aggressive mode support
  • risk_multiplier() consolidates all scaling into one call
  • Adaptive global cooldown: progressive pause after consecutive losses
    1 loss → 2h soft pause, 2 → 4h hard pause, 3 → 12h extended, 4+ → halt
"""

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional


def _iso_week(d: date) -> int:
    """Return ISO year*100+week to uniquely identify a calendar week."""
    y, w, _ = d.isocalendar()
    return y * 100 + w

logger = logging.getLogger('AI-Trade')

_STATE_FILE = Path('data/risk_state.json')


class RiskManager:
    def __init__(self, config: dict):
        self._cfg = config['risk']

        self.initial_balance:      float = 0.0
        self.peak_balance:         float = 0.0
        self.daily_start_balance:  float = 0.0
        self.weekly_start_balance: float = 0.0
        self._current_day:         date  = datetime.now().date()
        self._current_week:        int   = _iso_week(datetime.now().date())

        self._loss_streak:  int  = 0
        self._last_result:  str  = ''   # 'win' | 'loss' | ''

        # Adaptive global cooldown state
        self._global_cooldown_until: float = 0.0   # epoch seconds
        self._global_loss_streak:    int   = 0
        self._trading_halted:        bool  = False

        self._load_state()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        if not _STATE_FILE.exists():
            return
        try:
            s = json.loads(_STATE_FILE.read_text())
            self.initial_balance          = float(s.get('initial_balance',          0.0))
            self.peak_balance             = float(s.get('peak_balance',             0.0))
            self.daily_start_balance      = float(s.get('daily_start_balance',      0.0))
            self.weekly_start_balance     = float(s.get('weekly_start_balance',     0.0))
            self._loss_streak             = int(s.get('loss_streak',                0))
            self._global_loss_streak      = int(s.get('global_loss_streak',         0))
            self._global_cooldown_until   = float(s.get('global_cooldown_until',    0.0))
            self._trading_halted          = bool(s.get('trading_halted',            False))
            self._current_day             = date.fromisoformat(
                s.get('current_day', str(datetime.now().date()))
            )
            self._current_week            = int(s.get('current_week',
                                                _iso_week(datetime.now().date())))
        except Exception as exc:
            logger.warning(f"RiskManager: could not load state: {exc}")

    def _save_state(self) -> None:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps({
            'initial_balance':        self.initial_balance,
            'peak_balance':           self.peak_balance,
            'daily_start_balance':    self.daily_start_balance,
            'weekly_start_balance':   self.weekly_start_balance,
            'current_day':            str(self._current_day),
            'current_week':           self._current_week,
            'loss_streak':            self._loss_streak,
            'global_loss_streak':     self._global_loss_streak,
            'global_cooldown_until':  self._global_cooldown_until,
            'trading_halted':         self._trading_halted,
        }))

    # ── Balance update ────────────────────────────────────────────────────────

    def update_balance(self, balance: float) -> None:
        today      = datetime.now().date()
        this_week  = _iso_week(today)

        if self.initial_balance == 0.0:
            self.initial_balance      = balance
            self.daily_start_balance  = balance
            self.weekly_start_balance = balance
            self.peak_balance         = balance
            self._current_week        = this_week
            logger.info(f"RiskManager initialised | starting balance: {balance:.2f}")

        if today > self._current_day:
            self.daily_start_balance = balance
            self._current_day        = today
            logger.info(f"New trading day — daily baseline reset to {balance:.2f}")

        if this_week != self._current_week:
            self.weekly_start_balance = balance
            self._current_week        = this_week
            logger.info(f"New trading week — weekly baseline reset to {balance:.2f}")

        if balance > self.peak_balance:
            self.peak_balance = balance

        self._save_state()

    def record_trade_result(self, profit: float) -> None:
        """Call after each trade closes to update loss-streak and adaptive cooldown."""
        if profit > 0:
            self._loss_streak        = 0
            self._global_loss_streak = 0
            self._last_result        = 'win'
            # A win lifts both halts and cooldowns
            if self._trading_halted:
                self._trading_halted = False
                logger.info("Adaptive cooldown: trading halt LIFTED after winning trade")
            if self._global_cooldown_until > time.time():
                self._global_cooldown_until = 0.0
                logger.info("Adaptive cooldown: global cooldown cleared after win")
        else:
            self._loss_streak        += 1
            self._global_loss_streak += 1
            self._last_result         = 'loss'
            if self._loss_streak >= self._cfg.get('max_loss_streak', 3):
                logger.warning(
                    f"Loss streak: {self._loss_streak} consecutive losses — "
                    "switching to safe-mode sizing"
                )
            self._apply_adaptive_cooldown()
        self._save_state()

    def _apply_adaptive_cooldown(self) -> None:
        """
        Progressive global pause after consecutive losses (any direction/symbol).

        Streak → Cooldown:
          1 loss  → 2h soft pause
          2 losses → 4h hard pause
          3 losses → 12h extended pause
          4+ losses → trading halt (manual reset required)
        """
        cd_cfg = self._cfg.get('adaptive_cooldown', {})
        if not cd_cfg.get('enabled', True):
            return

        streak    = self._global_loss_streak
        hours_map = cd_cfg.get('hours_map', {1: 2, 2: 4, 3: 12})
        halt_at   = cd_cfg.get('halt_at_streak', 4)

        if streak >= halt_at:
            self._trading_halted = True
            logger.critical(
                f"ADAPTIVE COOLDOWN: Trading HALTED after {streak} consecutive losses. "
                "Manual reset required (delete data/risk_state.json or restart with reset)."
            )
            return

        cooldown_h = hours_map.get(streak, 0)
        if cooldown_h > 0:
            self._global_cooldown_until = time.time() + cooldown_h * 3600
            level = {1: 'soft', 2: 'hard', 3: 'extended'}.get(streak, 'hard')
            logger.warning(
                f"ADAPTIVE COOLDOWN: {level.upper()} pause {cooldown_h}h "
                f"after {streak} consecutive loss(es). "
                f"Resumes at {datetime.fromtimestamp(self._global_cooldown_until).strftime('%H:%M:%S')}"
            )

    def check_adaptive_cooldown(self) -> bool:
        """
        Returns True if trading is allowed, False if in adaptive cooldown/halt.
        Call this before attempting any new trade.
        """
        if self._trading_halted:
            logger.warning("ADAPTIVE COOLDOWN: Trading is HALTED — manual reset required")
            return False

        if self._global_cooldown_until > time.time():
            remaining_h = (self._global_cooldown_until - time.time()) / 3600
            logger.info(
                f"ADAPTIVE COOLDOWN: Global cooldown active — "
                f"{remaining_h:.1f}h remaining"
            )
            return False

        return True

    def reset_adaptive_halt(self) -> None:
        """Manually reset a trading halt (for admin use after review)."""
        self._trading_halted        = False
        self._global_loss_streak    = 0
        self._global_cooldown_until = 0.0
        self._save_state()
        logger.info("Adaptive cooldown reset — trading re-enabled")

    # ── Guard checks ──────────────────────────────────────────────────────────

    def check_daily_loss_limit(self, balance: float) -> bool:
        if self.daily_start_balance <= 0:
            return True
        loss_pct = (self.daily_start_balance - balance) / self.daily_start_balance
        if loss_pct >= self._cfg['max_daily_loss']:
            logger.warning(
                f"Daily loss limit reached: {loss_pct:.2%} "
                f"(threshold {self._cfg['max_daily_loss']:.2%}). "
                "No new trades until next session."
            )
            return False
        return True

    def check_weekly_loss_limit(self, balance: float) -> bool:
        """Block new trades when weekly loss exceeds max_weekly_loss (default 10%)."""
        limit = self._cfg.get('max_weekly_loss', 0.10)
        if self.weekly_start_balance <= 0:
            return True
        loss_pct = (self.weekly_start_balance - balance) / self.weekly_start_balance
        if loss_pct >= limit:
            logger.warning(
                f"WEEKLY loss limit reached: {loss_pct:.2%} "
                f"(threshold {limit:.2%}). No new trades until next week."
            )
            return False
        return True

    def weekly_pnl(self, balance: float) -> float:
        return balance - self.weekly_start_balance

    def check_drawdown_limit(self, balance: float) -> bool:
        if self.peak_balance <= 0:
            return True
        dd = (self.peak_balance - balance) / self.peak_balance
        if dd >= self._cfg['max_drawdown']:
            logger.critical(
                f"MAX DRAWDOWN EXCEEDED: {dd:.2%} "
                f"(threshold {self._cfg['max_drawdown']:.2%}). "
                "ENGINE HALTED."
            )
            return False
        return True

    def can_open_trade(self, open_count: int) -> bool:
        if open_count >= self._cfg['max_concurrent_trades']:
            logger.debug(
                f"Max concurrent trades reached ({open_count} / "
                f"{self._cfg['max_concurrent_trades']})"
            )
            return False
        return True

    # ── Risk scaling helpers ──────────────────────────────────────────────────

    def _drawdown_scale(self, balance: float) -> float:
        """
        Linear scale-down as drawdown grows from dd_scale_start to max_drawdown.
        Returns a multiplier in [dd_scale_min, 1.0].
        """
        dd_start = self._cfg.get('dd_scale_start', 0.05)
        dd_max   = self._cfg.get('max_drawdown',   0.10)
        dd_min_m = self._cfg.get('dd_scale_min',   0.30)

        dd = self.current_drawdown(balance)
        if dd <= dd_start:
            return 1.0
        if dd >= dd_max:
            return dd_min_m

        frac  = (dd - dd_start) / (dd_max - dd_start)
        scale = 1.0 - frac * (1.0 - dd_min_m)
        return max(dd_min_m, min(1.0, scale))

    def _kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Quarter-Kelly fraction based on historical win/loss stats.
        Caps at kelly_max_fraction (default 2.5%).
        Returns a fraction of balance (not a multiplier on risk_per_trade).
        """
        if avg_loss <= 0 or win_rate <= 0:
            return self._cfg.get('risk_per_trade', 0.01)

        b    = avg_win / avg_loss   # payoff ratio
        p    = win_rate / 100.0
        q    = 1.0 - p
        full_kelly = (b * p - q) / b if b > 0 else 0.0

        if full_kelly <= 0:
            return self._cfg.get('risk_per_trade', 0.01)

        quarter_kelly = full_kelly * 0.25
        cap = self._cfg.get('kelly_max_fraction', 0.025)
        return min(quarter_kelly, cap)

    def _mode_multiplier(self) -> float:
        """Mode-based multiplier: safe / normal / aggressive."""
        mode = self._cfg.get('mode', 'normal')
        if mode == 'safe':
            return self._cfg.get('safe_risk_multiplier', 0.50)
        if mode == 'aggressive':
            return self._cfg.get('aggressive_risk_multiplier', 1.20)
        return 1.0

    def risk_multiplier(self, balance: float) -> float:
        """
        Consolidated risk scaling: mode × drawdown scale × loss-streak.
        Returns a value in (0, max] that multiplies the base risk_per_trade.
        """
        mult = self._mode_multiplier()
        mult *= self._drawdown_scale(balance)

        max_streak = self._cfg.get('max_loss_streak', 3)
        if self._loss_streak >= max_streak:
            mult *= self._cfg.get('loss_streak_scale', 0.50)

        return max(0.10, mult)   # never go below 10% of normal

    # ── Position sizing ───────────────────────────────────────────────────────

    def calculate_lot_size(
        self,
        balance: float,
        sl_distance: float,
        symbol_info,
        win_rate: Optional[float] = None,
        avg_win:  Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> float:
        """
        Compute lot size with Kelly + drawdown + streak scaling.

        Uses Kelly fraction when win_rate / avg_win / avg_loss are provided,
        otherwise falls back to risk_per_trade × risk_multiplier.
        """
        use_kelly = self._cfg.get('use_kelly', True)

        if use_kelly and win_rate and avg_win and avg_loss:
            base_risk = self._kelly_fraction(win_rate, avg_win, avg_loss)
        else:
            base_risk = self._cfg['risk_per_trade']

        risk_amt = balance * base_risk * self.risk_multiplier(balance)

        tick_size  = float(symbol_info.trade_tick_size)
        tick_value = float(symbol_info.trade_tick_value)
        vol_step   = float(symbol_info.volume_step)
        vol_min    = float(symbol_info.volume_min)
        vol_max    = float(symbol_info.volume_max)

        if tick_size <= 0 or tick_value <= 0 or sl_distance <= 0:
            logger.warning(
                f"calculate_lot_size: invalid inputs "
                f"(sl={sl_distance}, tick_size={tick_size}, tick_value={tick_value}). "
                "Returning minimum lot."
            )
            return vol_min

        sl_ticks      = sl_distance / tick_size
        value_per_lot = sl_ticks * tick_value

        raw_lot = risk_amt / value_per_lot
        lot     = (raw_lot // vol_step) * vol_step
        lot     = max(vol_min, min(vol_max, lot))

        logger.info(
            f"Lot calc | balance={balance:.2f}  risk_amt={risk_amt:.2f}  "
            f"sl_dist={sl_distance:.5f}  val/lot={value_per_lot:.2f}  "
            f"mult={self.risk_multiplier(balance):.2f}  lot={lot:.2f}"
        )
        return round(lot, 2)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def current_drawdown(self, balance: float) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return max(0.0, (self.peak_balance - balance) / self.peak_balance)

    def daily_pnl(self, balance: float) -> float:
        return balance - self.daily_start_balance
