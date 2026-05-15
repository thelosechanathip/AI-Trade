"""
auto_optimizer.py — Background Optuna hyperparameter search + strategy evolution.

Runs as a daemon thread — never blocks the main trading loop.

Flow:
  1. ai_model.py calls submit_training_data(X, y) after every training cycle
  2. Every N cycles, a background thread starts an Optuna study
  3. Optuna finds better XGB + LGBM hyperparameters on a held-out validation split
  4. Best params saved to data/best_params.json
  5. ai_model.py loads them before the NEXT retrain cycle

Strategy param evolution (separate pass):
  - Grid-searches key thresholds (min_confluence, RSI/ADX ranges)
  - Evaluates by simulating signal generation on validation bars
  - If a config change improves expected trade win-rate, writes to data/best_strategy.json
  - main.py merges this into live config at startup

Graceful degradation:
  - If Optuna not installed → falls back to lightweight random search
  - All failures are logged and silently swallowed — never crashes the engine
"""

import json
import logging
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger('AI-Trade')

_BEST_PARAMS_PATH    = Path('data/best_params.json')
_BEST_STRATEGY_PATH  = Path('data/best_strategy.json')
_OPT_HISTORY_PATH    = Path('data/optimizer_history.json')


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


# ── AutoOptimizer ──────────────────────────────────────────────────────────────

class AutoOptimizer:
    """
    Background hyperparameter optimizer.

    Usage (from ai_model.py):
        self._optimizer = AutoOptimizer(config)
        ...
        self._optimizer.submit_training_data(X_tab, y_tab)
        best = self._optimizer.get_best_model_params()
    """

    def __init__(self, config: dict):
        self._cfg         = config
        self._opt_cfg     = config.get('optimizer', {})
        self._n_trials    = self._opt_cfg.get('n_trials', 40)
        self._interval    = self._opt_cfg.get('interval_cycles', 3)
        self._cycle       = 0

        self._lock        = threading.Lock()
        self._pending_X: Optional[np.ndarray] = None
        self._pending_y: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None

        self._best_model_params: dict  = _load_json(_BEST_PARAMS_PATH)
        self._best_strategy_params: dict = _load_json(_BEST_STRATEGY_PATH)
        self._history: List[dict]      = _load_json(_OPT_HISTORY_PATH) \
                                          if isinstance(_load_json(_OPT_HISTORY_PATH), list) \
                                          else []

        if self._best_model_params:
            logger.info(
                f"AutoOptimizer: loaded saved params "
                f"(prev AUC={self._best_model_params.get('auc', 0):.4f})"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def submit_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Called by ai_model.train() — may kick off background optimization."""
        self._cycle += 1
        if self._cycle % self._interval != 0:
            return
        if len(X) < 80:
            return

        with self._lock:
            self._pending_X = X.copy()
            self._pending_y = y.copy()

        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run_all, daemon=True, name='AutoOptimizer'
            )
            self._thread.start()
            logger.info(
                f"AutoOptimizer: cycle {self._cycle} — "
                f"background search started ({self._n_trials} trials)"
            )

    def get_best_model_params(self) -> dict:
        with self._lock:
            return dict(self._best_model_params)

    def get_best_strategy_params(self) -> dict:
        with self._lock:
            return dict(self._best_strategy_params)

    def clear_bad_params(self) -> None:
        """Discard Optuna params that caused worse-than-random CV performance."""
        with self._lock:
            self._best_model_params = {}
        if _BEST_PARAMS_PATH.exists():
            _BEST_PARAMS_PATH.unlink(missing_ok=True)
        logger.warning("AutoOptimizer: cleared stale/bad model params — will use defaults next cycle")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_all(self) -> None:
        with self._lock:
            if self._pending_X is None:
                return
            X, y = self._pending_X.copy(), self._pending_y.copy()
            self._pending_X = self._pending_y = None

        n = len(X)
        split = int(n * 0.75)
        X_tr_raw, X_val_raw = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        if len(np.unique(y_val)) < 2 or len(X_val_raw) < 10:
            logger.info("AutoOptimizer: validation set too small or single class — skipping")
            return

        # Scale using ONLY training data to avoid leakage into validation
        from sklearn.preprocessing import StandardScaler
        _scaler = StandardScaler()
        X_tr  = _scaler.fit_transform(X_tr_raw)
        X_val = _scaler.transform(X_val_raw)

        t0 = time.time()
        try:
            self._search_model_params(X_tr, y_tr, X_val, y_val)
        except Exception as exc:
            logger.error(f"AutoOptimizer model search failed: {exc}", exc_info=True)

        elapsed = time.time() - t0
        logger.info(f"AutoOptimizer: search finished in {elapsed:.1f}s")

    def _search_model_params(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> None:
        from sklearn.metrics import roc_auc_score

        best_params: dict = {}
        best_auc = self._best_model_params.get('auc', 0.0)

        # ── Try Optuna ─────────────────────────────────────────────────────────
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            use_optuna = True
        except ImportError:
            use_optuna = False
            logger.info("AutoOptimizer: Optuna not installed — using random search")

        # ── XGBoost ────────────────────────────────────────────────────────────
        xgb_params, xgb_auc = self._optimize_xgb(
            X_tr, y_tr, X_val, y_val, use_optuna
        )
        if xgb_auc > best_auc:
            best_auc = xgb_auc
            best_params['xgb'] = xgb_params
            logger.info(f"AutoOptimizer: XGB improved → AUC={xgb_auc:.4f}")

        # ── LightGBM ───────────────────────────────────────────────────────────
        lgbm_params, lgbm_auc = self._optimize_lgbm(
            X_tr, y_tr, X_val, y_val, use_optuna
        )
        if lgbm_auc > best_auc:
            best_params['lgbm'] = lgbm_params
            logger.info(f"AutoOptimizer: LGBM improved → AUC={lgbm_auc:.4f}")

        # ── Random Forest ──────────────────────────────────────────────────────
        rf_params, rf_auc = self._optimize_rf(
            X_tr, y_tr, X_val, y_val, use_optuna
        )
        if rf_auc > best_auc:
            best_params['rf'] = rf_params
            logger.info(f"AutoOptimizer: RF improved → AUC={rf_auc:.4f}")

        # ── Save if better ─────────────────────────────────────────────────────
        combined_auc = np.mean([a for a in [xgb_auc, lgbm_auc, rf_auc] if a > 0])
        if best_params or combined_auc > self._best_model_params.get('auc', 0.0):
            save_data = {
                'auc': float(combined_auc),
                'xgb':  xgb_params,
                'lgbm': lgbm_params,
                'rf':   rf_params,
                'n_samples': len(X_tr),
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            with self._lock:
                self._best_model_params = save_data
            _save_json(_BEST_PARAMS_PATH, save_data)

            # Append to history
            self._history.append({'auc': float(combined_auc), 'ts': save_data['timestamp']})
            _save_json(_OPT_HISTORY_PATH, self._history[-50:])

            logger.info(
                f"AutoOptimizer: new best params saved "
                f"(XGB={xgb_auc:.3f} LGBM={lgbm_auc:.3f} RF={rf_auc:.3f})"
            )

    def _optimize_xgb(
        self,
        X_tr, y_tr, X_val, y_val,
        use_optuna: bool,
    ) -> Tuple[dict, float]:
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import roc_auc_score

            def _eval(p: dict) -> float:
                m = XGBClassifier(**p, eval_metric='logloss', random_state=42, n_jobs=-1)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m.fit(X_tr, y_tr, verbose=False)
                return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

            if use_optuna:
                import optuna

                def objective(trial):
                    return _eval({
                        'n_estimators':     trial.suggest_int('n_estimators', 200, 900),
                        'max_depth':        trial.suggest_int('max_depth', 3, 8),
                        'learning_rate':    trial.suggest_float('lr', 0.01, 0.12, log=True),
                        'subsample':        trial.suggest_float('subsample', 0.55, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
                        'min_child_weight': trial.suggest_int('min_child', 3, 20),
                        'reg_lambda':       trial.suggest_float('lambda', 0.05, 8.0, log=True),
                        'reg_alpha':        trial.suggest_float('alpha', 0.0, 3.0),
                        'gamma':            trial.suggest_float('gamma', 0.0, 2.0),
                    })

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=self._n_trials,
                               show_progress_bar=False, n_jobs=1)
                return study.best_params, study.best_value

            else:
                # Random search fallback
                best_p, best_a = {}, 0.0
                rng = np.random.default_rng(42)
                for _ in range(max(8, self._n_trials // 4)):
                    p = {
                        'n_estimators':     int(rng.integers(200, 700)),
                        'max_depth':        int(rng.integers(3, 8)),
                        'learning_rate':    float(rng.uniform(0.01, 0.10)),
                        'subsample':        float(rng.uniform(0.6, 1.0)),
                        'colsample_bytree': float(rng.uniform(0.5, 1.0)),
                        'min_child_weight': int(rng.integers(3, 15)),
                        'reg_lambda':       float(rng.uniform(0.1, 5.0)),
                    }
                    a = _eval(p)
                    if a > best_a:
                        best_a, best_p = a, p
                return best_p, best_a

        except Exception as exc:
            logger.debug(f"AutoOptimizer XGB failed: {exc}")
            return {}, 0.0

    def _optimize_lgbm(
        self,
        X_tr, y_tr, X_val, y_val,
        use_optuna: bool,
    ) -> Tuple[dict, float]:
        try:
            from lightgbm import LGBMClassifier
            from sklearn.metrics import roc_auc_score

            def _eval(p: dict) -> float:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m = LGBMClassifier(**p, random_state=42, n_jobs=-1, verbose=-1)
                    m.fit(X_tr, y_tr)
                return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

            if use_optuna:
                import optuna

                def objective(trial):
                    return _eval({
                        'n_estimators':      trial.suggest_int('n_est', 200, 900),
                        'max_depth':         trial.suggest_int('max_depth', 3, 8),
                        'learning_rate':     trial.suggest_float('lr', 0.01, 0.12, log=True),
                        'subsample':         trial.suggest_float('subsample', 0.55, 1.0),
                        'colsample_bytree':  trial.suggest_float('colsample', 0.5, 1.0),
                        'min_child_samples': trial.suggest_int('min_child', 5, 40),
                        'reg_lambda':        trial.suggest_float('lambda', 0.05, 8.0, log=True),
                        'reg_alpha':         trial.suggest_float('alpha', 0.0, 3.0),
                        'num_leaves':        trial.suggest_int('leaves', 15, 63),
                    })

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=self._n_trials,
                               show_progress_bar=False, n_jobs=1)
                return study.best_params, study.best_value

            else:
                best_p, best_a = {}, 0.0
                rng = np.random.default_rng(43)
                for _ in range(max(8, self._n_trials // 4)):
                    p = {
                        'n_estimators':      int(rng.integers(200, 700)),
                        'max_depth':         int(rng.integers(3, 8)),
                        'learning_rate':     float(rng.uniform(0.01, 0.10)),
                        'subsample':         float(rng.uniform(0.6, 1.0)),
                        'colsample_bytree':  float(rng.uniform(0.5, 1.0)),
                        'min_child_samples': int(rng.integers(5, 30)),
                        'reg_lambda':        float(rng.uniform(0.1, 5.0)),
                    }
                    a = _eval(p)
                    if a > best_a:
                        best_a, best_p = a, p
                return best_p, best_a

        except Exception as exc:
            logger.debug(f"AutoOptimizer LGBM failed: {exc}")
            return {}, 0.0

    def _optimize_rf(
        self,
        X_tr, y_tr, X_val, y_val,
        use_optuna: bool,
    ) -> Tuple[dict, float]:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score

            def _eval(p: dict) -> float:
                m = RandomForestClassifier(**p, n_jobs=-1, random_state=42,
                                           class_weight='balanced')
                m.fit(X_tr, y_tr)
                return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

            if use_optuna:
                import optuna

                def objective(trial):
                    return _eval({
                        'n_estimators':    trial.suggest_int('n_est', 200, 800),
                        'max_depth':       trial.suggest_int('max_depth', 5, 15),
                        'min_samples_leaf': trial.suggest_int('min_leaf', 5, 25),
                        'max_features':    trial.suggest_categorical('max_feat', ['sqrt', 'log2']),
                        'max_samples':     trial.suggest_float('max_samples', 0.6, 1.0),
                    })

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=self._n_trials,
                               show_progress_bar=False, n_jobs=1)
                return study.best_params, study.best_value

            else:
                best_p, best_a = {}, 0.0
                rng = np.random.default_rng(44)
                for _ in range(max(8, self._n_trials // 4)):
                    p = {
                        'n_estimators':     int(rng.integers(200, 600)),
                        'max_depth':        int(rng.integers(5, 12)),
                        'min_samples_leaf': int(rng.integers(5, 20)),
                        'max_samples':      float(rng.uniform(0.65, 1.0)),
                    }
                    a = _eval(p)
                    if a > best_a:
                        best_a, best_p = a, p
                return best_p, best_a

        except Exception as exc:
            logger.debug(f"AutoOptimizer RF failed: {exc}")
            return {}, 0.0
