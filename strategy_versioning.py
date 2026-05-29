"""
strategy_versioning.py — Lightweight strategy version tracking.

Workflow:
  1. On startup, seed v1 from current config.yaml if no versions exist.
  2. Optimizer/user proposes candidate parameter set.
  3. propose_and_validate() runs a backtest and compares vs current version.
  4. If candidate passes acceptance criteria → save (but do NOT auto-deploy).
  5. Caller calls deploy(version_id) to make it active.
  6. rollback() reverts to the previous deployed version instantly.

Acceptance criteria (is_better_than_current):
  profit_factor  >= current × 0.90    (tolerates minor regression)
  max_drawdown   <= current + 2.0 %   (never materially worse DD)
  win_rate       >= current × 0.90    (tolerates minor regression)

Storage: data/strategy_versions.json  (human-readable JSON, append-only)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger('AI-Trade')

_VERSIONS_FILE = Path('data/strategy_versions.json')

# Parameters controlled by the versioning system.
# Format: (config_section, param_key)
VERSIONED_PARAMS: list = [
    ('strategy', 'ema_fast'),
    ('strategy', 'ema_slow'),
    ('strategy', 'rsi_period'),
    ('strategy', 'rsi_buy_min'),
    ('strategy', 'rsi_buy_max'),
    ('strategy', 'rsi_sell_min'),
    ('strategy', 'rsi_sell_max'),
    ('strategy', 'atr_period'),
    ('strategy', 'min_confluence'),
    ('strategy', 'adx_trend_threshold'),
    ('strategy', 'adx_range_threshold'),
    ('risk', 'atr_sl_multiplier'),
    ('risk', 'atr_tp_multiplier'),
    ('risk', 'min_rr_ratio'),
]


class StrategyVersionManager:
    def __init__(
        self,
        config: dict,
        versions_path: Optional[Path] = None,
    ):
        self._config    = config
        self._path      = versions_path or _VERSIONS_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._versions: List[dict] = self._load()

        if not self._versions:
            self._seed_initial()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> List[dict]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding='utf-8'))
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning(f"StrategyVersionManager: could not load versions: {exc}")
            return []

    def _save(self) -> None:
        tmp = self._path.with_suffix('.tmp')
        tmp.write_text(
            json.dumps(self._versions, indent=2, default=str),
            encoding='utf-8',
        )
        tmp.replace(self._path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec='seconds')

    def extract_params(self, config: dict) -> dict:
        """Pull versioned parameters out of a config dict."""
        params: dict = {}
        for section, key in VERSIONED_PARAMS:
            val = config.get(section, {}).get(key)
            if val is not None:
                params[f'{section}.{key}'] = val
        return params

    def apply_params(self, config: dict, params: dict) -> dict:
        """
        Apply a params dict (keys like 'strategy.ema_fast') to a config dict.
        Returns a modified copy; does NOT write config.yaml.
        """
        import copy
        cfg = copy.deepcopy(config)
        for dotkey, val in params.items():
            section, key = dotkey.split('.', 1)
            if section in cfg and key in cfg[section]:
                cfg[section][key] = val
        return cfg

    def _next_id(self) -> int:
        return max((v['version_id'] for v in self._versions), default=0) + 1

    def _seed_initial(self) -> None:
        params  = self.extract_params(self._config)
        version = {
            'version_id':        1,
            'timestamp':         self._now(),
            'source':            'initial',
            'params':            params,
            'backtest_metrics':  {},
            'live_metrics':      {},
            'deployed':          True,
            'notes':             'Seeded from config.yaml on first run',
        }
        self._versions.append(version)
        self._save()
        logger.info("StrategyVersionManager: seeded initial version v1 from config.yaml")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current_version(self) -> dict:
        """Return the most recently deployed version."""
        deployed = [v for v in self._versions if v.get('deployed')]
        return deployed[-1] if deployed else (self._versions[-1] if self._versions else {})

    def list_versions(self) -> List[dict]:
        return self._versions.copy()

    def save_version(
        self,
        params:           dict,
        backtest_metrics: dict,
        source:           str  = 'optimizer',
        notes:            str  = '',
        deploy:           bool = False,
    ) -> int:
        """
        Persist a new version. If deploy=True, marks all others as not deployed.
        Returns the new version_id.
        """
        version_id = self._next_id()
        if deploy:
            for v in self._versions:
                v['deployed'] = False
        self._versions.append({
            'version_id':       version_id,
            'timestamp':        self._now(),
            'source':           source,
            'params':           params,
            'backtest_metrics': backtest_metrics,
            'live_metrics':     {},
            'deployed':         deploy,
            'notes':            notes,
        })
        self._save()
        logger.info(
            f"StrategyVersionManager: saved v{version_id} "
            f"(source={source}, deployed={deploy})"
        )
        return version_id

    def update_live_metrics(self, version_id: int, live_metrics: dict) -> None:
        """Record real forward-trading performance for an already-deployed version."""
        for v in self._versions:
            if v['version_id'] == version_id:
                v['live_metrics'] = live_metrics
                self._save()
                return
        logger.warning(f"StrategyVersionManager: version {version_id} not found")

    def is_better_than_current(self, candidate: dict) -> tuple:
        """
        Acceptance check: compare candidate backtest metrics vs current version.

        Rules (all must pass):
          profit_factor  >= current_pf  × 0.90
          max_drawdown   <= current_dd  + 2.0 %
          win_rate       >= current_wr  × 0.90

        Returns (accepted: bool, reasons: list[str])
        """
        current_bt = self.get_current_version().get('backtest_metrics', {})
        reasons:   list = []

        if not current_bt:
            reasons.append("No baseline backtest — accepting candidate by default")
            return True, reasons

        curr_pf = float(current_bt.get('profit_factor',   1.0))
        curr_dd = float(current_bt.get('max_drawdown_pct', 10.0))
        curr_wr = float(current_bt.get('win_rate_pct',    50.0))

        cand_pf = float(candidate.get('profit_factor',    0.0))
        cand_dd = float(candidate.get('max_drawdown_pct', 100.0))
        cand_wr = float(candidate.get('win_rate_pct',     0.0))

        pf_ok = cand_pf >= curr_pf * 0.90
        dd_ok = cand_dd <= curr_dd + 2.0
        wr_ok = cand_wr >= curr_wr * 0.90

        if not pf_ok:
            reasons.append(
                f"FAIL profit_factor {cand_pf:.3f} < {curr_pf*0.90:.3f} "
                f"(current {curr_pf:.3f})"
            )
        if not dd_ok:
            reasons.append(
                f"FAIL max_drawdown {cand_dd:.1f}% > {curr_dd+2.0:.1f}% "
                f"(current {curr_dd:.1f}%)"
            )
        if not wr_ok:
            reasons.append(
                f"FAIL win_rate {cand_wr:.1f}% < {curr_wr*0.90:.1f}% "
                f"(current {curr_wr:.1f}%)"
            )

        accepted = pf_ok and dd_ok and wr_ok
        if accepted:
            reasons.append(
                f"PASS PF={cand_pf:.3f} DD={cand_dd:.1f}% WR={cand_wr:.1f}%"
            )
        return accepted, reasons

    def deploy(self, version_id: int) -> dict:
        """
        Mark version_id as deployed; un-deploy all others.
        Returns the params dict of the deployed version.
        Caller must apply params to the live config via apply_params().
        """
        target = None
        for v in self._versions:
            v['deployed'] = (v['version_id'] == version_id)
            if v['version_id'] == version_id:
                target = v
        if target is None:
            raise ValueError(f"StrategyVersionManager: version {version_id} not found")
        self._save()
        logger.info(f"StrategyVersionManager: deployed v{version_id}")
        return target['params']

    def rollback(self, version_id: Optional[int] = None) -> dict:
        """
        Roll back to a specific version (or the one before the current deployed version).
        Returns the restored params dict.
        """
        if version_id is None:
            all_ids = [v['version_id'] for v in self._versions]
            deployed_ids = [v['version_id'] for v in self._versions if v.get('deployed')]
            if not deployed_ids:
                raise ValueError("No deployed version found")
            curr_id  = deployed_ids[-1]
            curr_idx = all_ids.index(curr_id)
            if curr_idx == 0:
                raise ValueError("Already at the earliest version — cannot roll back further")
            version_id = all_ids[curr_idx - 1]
            logger.info(
                f"StrategyVersionManager: rolling back from v{curr_id} to v{version_id}"
            )
        return self.deploy(version_id)

    def propose_and_validate(
        self,
        candidate_params: dict,
        df,
        symbol:   str  = '',
        notes:    str  = '',
    ) -> dict:
        """
        Full validation workflow for a proposed parameter change:
          1. Apply candidate params to a config copy.
          2. Run a backtest.
          3. Compare vs current version.
          4. Save version if accepted (but do NOT auto-deploy).

        Returns:
          {
            accepted:         bool,
            version_id:       int | None,   # set when accepted
            backtest_metrics: dict,
            reasons:          list[str],
          }
        """
        from backtest import BacktestEngine
        import copy

        cfg = copy.deepcopy(self._config)
        cfg = self.apply_params(cfg, candidate_params)

        sym = symbol or cfg.get('trading', {}).get('symbols', ['XAUUSD'])[0]

        engine = BacktestEngine(cfg, sym)
        try:
            metrics = engine.run(df)
        except Exception as exc:
            logger.error(f"StrategyVersionManager.propose_and_validate: backtest error: {exc}")
            return {
                'accepted': False,
                'version_id': None,
                'backtest_metrics': {},
                'reasons': [f"Backtest failed: {exc}"],
            }

        if 'note' in metrics:
            return {
                'accepted': False,
                'version_id': None,
                'backtest_metrics': metrics,
                'reasons': [f"No trades executed: {metrics['note']}"],
            }

        accepted, reasons = self.is_better_than_current(metrics)
        for r in reasons:
            logger.info(f"StrategyVersioning: {r}")

        version_id = None
        if accepted:
            version_id = self.save_version(
                params           = candidate_params,
                backtest_metrics = metrics,
                source           = 'optimizer',
                notes            = notes or '; '.join(reasons),
                deploy           = False,
            )
            logger.info(
                f"StrategyVersionManager: candidate saved as v{version_id} "
                "(call deploy(version_id) to activate)"
            )
        else:
            logger.warning(
                "StrategyVersionManager: candidate REJECTED — "
                + '; '.join(reasons)
            )

        return {
            'accepted':         accepted,
            'version_id':       version_id,
            'backtest_metrics': metrics,
            'reasons':          reasons,
        }

    def get_version_summary(self) -> List[dict]:
        """Return a concise summary list suitable for display."""
        summary = []
        for v in self._versions:
            bt  = v.get('backtest_metrics', {})
            lv  = v.get('live_metrics', {})
            summary.append({
                'version_id':      v['version_id'],
                'timestamp':       v['timestamp'],
                'source':          v['source'],
                'deployed':        v['deployed'],
                'bt_pf':           bt.get('profit_factor',   '—'),
                'bt_dd':           bt.get('max_drawdown_pct','—'),
                'bt_wr':           bt.get('win_rate_pct',    '—'),
                'live_trades':     lv.get('total_trades',    0),
                'live_wr':         lv.get('win_rate',        '—'),
                'notes':           v.get('notes', ''),
            })
        return summary
