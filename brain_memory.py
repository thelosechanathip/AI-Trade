"""
brain_memory.py — SQLite-backed long-term memory for the Market Brain.

Tables:
  brain_trades      — every trade the Brain decided on (entry + exit)
  market_snapshots  — periodic market state snapshots
  narrative_memory  — regime/narrative patterns that preceded outcomes
  reversal_patterns — signals that appeared before reversals
  failure_patterns  — signal combos that preceded losses
  ai_decisions      — per-cycle Brain decisions (sampled)
  learning_feedback — aggregate stats per regime/setup used for adaptation

Usage:
  mem = BrainMemory()
  mem.record_decision(brain_decision, symbol, cycle_id)
  mem.record_trade_open(ticket, brain_decision, entry_price, symbol)
  mem.record_trade_close(ticket, exit_price, profit, outcome, bars_held)
  stats = mem.get_pattern_stats(regime='TREND_BULL', lookback_days=30)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger('AI-Trade')

_DB_PATH = Path('data/brain_memory.db')

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS brain_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket          INTEGER,
    symbol          TEXT,
    direction       TEXT,
    entry_price     REAL,
    exit_price      REAL,
    profit          REAL,
    outcome         TEXT,        -- 'win'|'loss'|'breakeven'|'open'
    bars_held       INTEGER,
    entry_time      TEXT,
    exit_time       TEXT,
    decision_conf   REAL,
    uncertainty     REAL,
    setup_quality   REAL,
    market_regime   TEXT,
    reversal_prob   REAL,
    reasoning       TEXT,        -- JSON list
    signals_active  TEXT,        -- JSON list
    agent_votes     TEXT         -- JSON list
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    symbol      TEXT,
    regime      TEXT,
    adx         REAL,
    rsi         REAL,
    atr         REAL,
    htf_bias    TEXT,
    narrative   TEXT,
    uncertainty REAL,
    signals     TEXT             -- JSON list
);

CREATE TABLE IF NOT EXISTS narrative_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT,
    regime          TEXT,
    narrative       TEXT,
    signals         TEXT,        -- JSON list
    outcome         TEXT,        -- 'win'|'loss'|'no_trade'
    profit_r        REAL,
    confidence      REAL,
    setup_quality   REAL
);

CREATE TABLE IF NOT EXISTS reversal_patterns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    symbol      TEXT,
    direction   TEXT,            -- 'UP'|'DOWN'  (expected reversal direction)
    signals     TEXT,            -- JSON list of signals that fired
    confirmed   INTEGER DEFAULT 0,
    bars_to_confirm INTEGER,
    price_move  REAL
);

CREATE TABLE IF NOT EXISTS failure_patterns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    symbol      TEXT,
    direction   TEXT,
    regime      TEXT,
    signals     TEXT,            -- JSON list
    loss_r      REAL,
    bars_held   INTEGER,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS ai_decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    symbol      TEXT,
    cycle_id    TEXT,
    decision    TEXT,
    confidence  REAL,
    uncertainty REAL,
    regime      TEXT,
    hold_reasons TEXT,           -- JSON list
    agent_votes  TEXT            -- JSON list
);

CREATE TABLE IF NOT EXISTS learning_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    regime          TEXT UNIQUE,
    win_count       INTEGER DEFAULT 0,
    loss_count      INTEGER DEFAULT 0,
    total_profit_r  REAL    DEFAULT 0.0,
    avg_conf_win    REAL    DEFAULT 0.0,
    avg_conf_loss   REAL    DEFAULT 0.0,
    avg_bars_win    REAL    DEFAULT 0.0,
    avg_bars_loss   REAL    DEFAULT 0.0,
    last_updated    TEXT
);

CREATE INDEX IF NOT EXISTS idx_bt_ticket   ON brain_trades(ticket);
CREATE INDEX IF NOT EXISTS idx_bt_symbol   ON brain_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_bt_outcome  ON brain_trades(outcome);
CREATE INDEX IF NOT EXISTS idx_nf_regime   ON narrative_memory(regime);
CREATE INDEX IF NOT EXISTS idx_lf_regime   ON learning_feedback(regime);
"""


class BrainMemory:
    def __init__(self, db_path: Optional[Path] = None):
        self._path = db_path or _DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat(timespec='seconds')

    @staticmethod
    def _j(obj) -> str:
        return json.dumps(obj) if obj is not None else '[]'

    # ── Write operations ──────────────────────────────────────────────────────

    def record_decision(
        self,
        brain_decision,          # BrainDecision dataclass
        symbol: str,
        cycle_id: str = '',
    ) -> None:
        """Persist a per-cycle Brain decision (sampled — not every cycle needed)."""
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO ai_decisions
                        (ts, symbol, cycle_id, decision, confidence, uncertainty,
                         regime, hold_reasons, agent_votes)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        self._now(),
                        symbol,
                        cycle_id,
                        brain_decision.decision,
                        brain_decision.confidence,
                        brain_decision.uncertainty,
                        brain_decision.market_regime,
                        self._j(brain_decision.hold_reasons),
                        self._j([
                            {'name': v.get('name'), 'vote': v.get('vote'),
                             'conf': v.get('confidence')}
                            for v in (brain_decision.agent_votes or [])
                        ]),
                    ),
                )
        except Exception as exc:
            logger.warning(f"BrainMemory.record_decision failed: {exc}")

    def record_trade_open(
        self,
        ticket:         int,
        brain_decision,
        entry_price:    float,
        symbol:         str,
    ) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO brain_trades
                        (ticket, symbol, direction, entry_price, outcome,
                         entry_time, decision_conf, uncertainty, setup_quality,
                         market_regime, reversal_prob, reasoning,
                         signals_active, agent_votes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        ticket,
                        symbol,
                        brain_decision.decision,
                        entry_price,
                        'open',
                        self._now(),
                        brain_decision.confidence,
                        brain_decision.uncertainty,
                        brain_decision.setup_quality,
                        brain_decision.market_regime,
                        brain_decision.reversal_probability,
                        self._j(brain_decision.reasoning),
                        self._j(getattr(brain_decision, 'signals_active', [])),
                        self._j(brain_decision.agent_votes or []),
                    ),
                )
        except Exception as exc:
            logger.warning(f"BrainMemory.record_trade_open failed: {exc}")

    def record_trade_close(
        self,
        ticket:     int,
        exit_price: float,
        profit:     float,
        outcome:    str,     # 'win'|'loss'|'breakeven'
        bars_held:  int = 0,
    ) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE brain_trades
                    SET exit_price=?, profit=?, outcome=?, exit_time=?, bars_held=?
                    WHERE ticket=?
                    """,
                    (exit_price, profit, outcome, self._now(), bars_held, ticket),
                )
        except Exception as exc:
            logger.warning(f"BrainMemory.record_trade_close failed: {exc}")

    def record_snapshot(
        self,
        symbol:    str,
        regime:    str,
        adx:       float,
        rsi:       float,
        atr:       float,
        htf_bias:  str,
        narrative: str,
        uncertainty: float,
        signals:   Optional[List[str]] = None,
    ) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO market_snapshots
                        (ts, symbol, regime, adx, rsi, atr,
                         htf_bias, narrative, uncertainty, signals)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        self._now(), symbol, regime, adx, rsi, atr,
                        htf_bias, narrative, uncertainty, self._j(signals),
                    ),
                )
        except Exception as exc:
            logger.warning(f"BrainMemory.record_snapshot failed: {exc}")

    def record_reversal_pattern(
        self,
        symbol:    str,
        direction: str,
        signals:   List[str],
    ) -> int:
        """Returns the row id for later confirmation."""
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO reversal_patterns (ts, symbol, direction, signals)
                    VALUES (?,?,?,?)
                    """,
                    (self._now(), symbol, direction, self._j(signals)),
                )
                return cur.lastrowid or 0
        except Exception as exc:
            logger.warning(f"BrainMemory.record_reversal_pattern failed: {exc}")
            return 0

    def confirm_reversal(
        self, pattern_id: int, bars_to_confirm: int, price_move: float
    ) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE reversal_patterns
                    SET confirmed=1, bars_to_confirm=?, price_move=?
                    WHERE id=?
                    """,
                    (bars_to_confirm, price_move, pattern_id),
                )
        except Exception as exc:
            logger.warning(f"BrainMemory.confirm_reversal failed: {exc}")

    def record_failure_pattern(
        self,
        symbol:    str,
        direction: str,
        regime:    str,
        signals:   List[str],
        loss_r:    float,
        bars_held: int,
        notes:     str = '',
    ) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO failure_patterns
                        (ts, symbol, direction, regime, signals, loss_r, bars_held, notes)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        self._now(), symbol, direction, regime,
                        self._j(signals), loss_r, bars_held, notes,
                    ),
                )
        except Exception as exc:
            logger.warning(f"BrainMemory.record_failure_pattern failed: {exc}")

    def update_learning_feedback(
        self,
        regime:     str,
        outcome:    str,   # 'win'|'loss'
        profit_r:   float,
        confidence: float,
        bars_held:  int,
    ) -> None:
        """Upsert aggregate stats per regime for adaptive weighting."""
        try:
            with self._conn() as conn:
                # Read current row
                row = conn.execute(
                    "SELECT * FROM learning_feedback WHERE regime=?", (regime,)
                ).fetchone()

                if row is None:
                    conn.execute(
                        """
                        INSERT INTO learning_feedback
                            (regime, win_count, loss_count, total_profit_r,
                             avg_conf_win, avg_conf_loss, avg_bars_win, avg_bars_loss,
                             last_updated)
                        VALUES (?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            regime,
                            1 if outcome == 'win' else 0,
                            1 if outcome == 'loss' else 0,
                            profit_r,
                            confidence if outcome == 'win' else 0.0,
                            confidence if outcome == 'loss' else 0.0,
                            bars_held  if outcome == 'win' else 0.0,
                            bars_held  if outcome == 'loss' else 0.0,
                            self._now(),
                        ),
                    )
                    return

                # Update with exponential moving average (alpha=0.15)
                alpha = 0.15
                win_count  = row['win_count']  + (1 if outcome == 'win'  else 0)
                loss_count = row['loss_count'] + (1 if outcome == 'loss' else 0)
                total_pr   = row['total_profit_r'] + profit_r

                def ema(old, new):
                    return old * (1 - alpha) + new * alpha if old else new

                if outcome == 'win':
                    avg_conf_win  = ema(row['avg_conf_win'],  confidence)
                    avg_conf_loss = row['avg_conf_loss']
                    avg_bars_win  = ema(row['avg_bars_win'],  bars_held)
                    avg_bars_loss = row['avg_bars_loss']
                else:
                    avg_conf_win  = row['avg_conf_win']
                    avg_conf_loss = ema(row['avg_conf_loss'], confidence)
                    avg_bars_win  = row['avg_bars_win']
                    avg_bars_loss = ema(row['avg_bars_loss'], bars_held)

                conn.execute(
                    """
                    UPDATE learning_feedback
                    SET win_count=?, loss_count=?, total_profit_r=?,
                        avg_conf_win=?, avg_conf_loss=?,
                        avg_bars_win=?, avg_bars_loss=?, last_updated=?
                    WHERE regime=?
                    """,
                    (
                        win_count, loss_count, total_pr,
                        avg_conf_win, avg_conf_loss,
                        avg_bars_win, avg_bars_loss,
                        self._now(), regime,
                    ),
                )
        except Exception as exc:
            logger.warning(f"BrainMemory.update_learning_feedback failed: {exc}")

    # ── Read / query operations ───────────────────────────────────────────────

    def get_pattern_stats(
        self,
        regime:        Optional[str] = None,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Returns win-rate, avg profit, and count for the given regime
        within the lookback window.
        """
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        try:
            with self._conn() as conn:
                q = """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN outcome='win'  THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses,
                        AVG(profit) as avg_profit,
                        AVG(decision_conf)  as avg_conf,
                        AVG(setup_quality) as avg_quality
                    FROM brain_trades
                    WHERE entry_time >= ?
                      AND outcome IN ('win','loss','breakeven')
                """
                params: list = [since]
                if regime:
                    q += " AND market_regime = ?"
                    params.append(regime)
                row = conn.execute(q, params).fetchone()

                if not row or row['total'] == 0:
                    return {
                        'total': 0, 'win_rate': 0.5,
                        'avg_profit': 0.0, 'avg_conf': 0.5,
                        'avg_quality': 0.5, 'regime': regime,
                    }

                total = row['total']
                return {
                    'total':      total,
                    'wins':       row['wins']   or 0,
                    'losses':     row['losses'] or 0,
                    'win_rate':   (row['wins'] or 0) / total,
                    'avg_profit': row['avg_profit'] or 0.0,
                    'avg_conf':   row['avg_conf']   or 0.5,
                    'avg_quality':row['avg_quality'] or 0.5,
                    'regime':     regime,
                }
        except Exception as exc:
            logger.warning(f"BrainMemory.get_pattern_stats failed: {exc}")
            return {'total': 0, 'win_rate': 0.5, 'avg_profit': 0.0,
                    'avg_conf': 0.5, 'avg_quality': 0.5, 'regime': regime}

    def get_regime_win_rate(self, regime: str) -> float:
        """Quick accessor: returns win rate for regime or 0.5 if unknown."""
        stats = self.get_pattern_stats(regime=regime, lookback_days=60)
        return stats['win_rate']

    def get_recent_losses(self, n: int = 5) -> List[Dict]:
        """Returns the N most recent losing trades as dicts."""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT ticket, symbol, direction, profit, market_regime,
                           signals_active, reasoning, entry_time
                    FROM brain_trades
                    WHERE outcome='loss'
                    ORDER BY exit_time DESC
                    LIMIT ?
                    """,
                    (n,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning(f"BrainMemory.get_recent_losses failed: {exc}")
            return []

    def self_review(self, ticket: int) -> Dict[str, Any]:
        """
        Post-trade self-review for a closed trade.
        Returns a dict with the original reasoning vs outcome for logging.
        """
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM brain_trades WHERE ticket=?", (ticket,)
                ).fetchone()
                if not row:
                    return {}

                trade = dict(row)
                regime = trade.get('market_regime', 'UNKNOWN')
                stats  = self.get_pattern_stats(regime=regime, lookback_days=90)

                review = {
                    'ticket':          ticket,
                    'direction':       trade.get('direction'),
                    'outcome':         trade.get('outcome'),
                    'profit':          trade.get('profit'),
                    'regime':          regime,
                    'regime_win_rate': stats.get('win_rate', 0.5),
                    'decision_conf':   trade.get('decision_conf'),
                    'setup_quality':   trade.get('setup_quality'),
                    'original_reasoning': json.loads(trade.get('reasoning') or '[]'),
                    'lesson': '',
                }

                outcome = trade.get('outcome', '')
                conf    = trade.get('decision_conf', 0.5) or 0.5
                quality = trade.get('setup_quality', 0.5) or 0.5

                if outcome == 'loss' and conf > 0.70:
                    review['lesson'] = (
                        f"High-confidence loss in {regime} — "
                        "consider raising quality threshold or reducing aggression"
                    )
                elif outcome == 'loss' and quality < 0.40:
                    review['lesson'] = (
                        "Low-quality setup taken — uncertainty filter may need "
                        "tighter quality gate"
                    )
                elif outcome == 'win' and quality > 0.65:
                    review['lesson'] = (
                        "High-quality setup paid off — confirm quality filter is working"
                    )
                else:
                    review['lesson'] = f"Standard {outcome} in {regime}"

                logger.info(
                    f"Self-review #{ticket}: {outcome} | conf={conf:.2f} | "
                    f"quality={quality:.2f} | lesson: {review['lesson']}"
                )
                return review
        except Exception as exc:
            logger.warning(f"BrainMemory.self_review failed: {exc}")
            return {}

    def get_failure_similarity(
        self,
        current_signals: List[str],
        regime: str,
        direction: str,
        lookback_days: int = 30,
    ) -> float:
        """
        Returns a similarity score (0–1) between current signals and
        known failure patterns in this regime+direction.
        High score → similar setups have lost before.
        """
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT signals FROM failure_patterns
                    WHERE ts >= ? AND regime=? AND direction=?
                    """,
                    (since, regime, direction),
                ).fetchall()

            if not rows or not current_signals:
                return 0.0

            current_set = set(current_signals)
            max_sim = 0.0
            for row in rows:
                past_signals = set(json.loads(row['signals'] or '[]'))
                if not past_signals:
                    continue
                inter = len(current_set & past_signals)
                union = len(current_set | past_signals)
                sim   = inter / union if union > 0 else 0.0
                max_sim = max(max_sim, sim)

            return float(max_sim)
        except Exception as exc:
            logger.warning(f"BrainMemory.get_failure_similarity failed: {exc}")
            return 0.0

    def prune_old_records(self, days: int = 90) -> None:
        """Remove records older than `days` to keep DB small."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        try:
            with self._conn() as conn:
                for table in (
                    'market_snapshots', 'ai_decisions',
                    'reversal_patterns', 'failure_patterns',
                ):
                    conn.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
            logger.debug(f"BrainMemory: pruned records older than {days} days")
        except Exception as exc:
            logger.warning(f"BrainMemory.prune_old_records failed: {exc}")
