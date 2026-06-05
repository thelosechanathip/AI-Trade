"""
utils.py — Shared utilities: logging, indicator math, market structure,
           database helpers, and state I/O.
"""

import json
import logging
import logging.handlers
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(config: dict) -> logging.Logger:
    log_cfg = config['logging']
    log_path = Path(log_cfg['file'])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('AI-Trade')
    logger.setLevel(getattr(logging, log_cfg['level'].upper(), logging.INFO))

    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=log_cfg['max_size_mb'] * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
    )
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ── Technical indicators ──────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(
    series: pd.Series, fast: int, slow: int, signal: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr = pd.concat(
        [high - low,
         (high - close.shift(1)).abs(),
         (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ADX, +DI, -DI) series."""
    high_arr  = high.values.astype(float)
    low_arr   = low.values.astype(float)
    close_arr = close.values.astype(float)
    n = len(close_arr)

    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    tr_arr   = np.zeros(n)

    for i in range(1, n):
        h_diff = high_arr[i] - high_arr[i - 1]
        l_diff = low_arr[i - 1] - low_arr[i]
        plus_dm[i]  = h_diff if h_diff > l_diff and h_diff > 0 else 0.0
        minus_dm[i] = l_diff if l_diff > h_diff and l_diff > 0 else 0.0
        tr_arr[i] = max(
            high_arr[i] - low_arr[i],
            abs(high_arr[i] - close_arr[i - 1]),
            abs(low_arr[i] - close_arr[i - 1]),
        )

    idx = close.index
    tr_s   = pd.Series(tr_arr, index=idx)
    pdm_s  = pd.Series(plus_dm, index=idx)
    mdm_s  = pd.Series(minus_dm, index=idx)

    atr14   = tr_s.ewm(com=period - 1, min_periods=period).mean()
    pdm14   = pdm_s.ewm(com=period - 1, min_periods=period).mean()
    mdm14   = mdm_s.ewm(com=period - 1, min_periods=period).mean()

    plus_di  = 100.0 * pdm14 / atr14.replace(0, np.nan)
    minus_di = 100.0 * mdm14 / atr14.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()

    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def compute_bollinger(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (middle, upper, lower, pct_b) where pct_b ∈ [0, 1]."""
    mid   = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    upper = mid + std_dev * sigma
    lower = mid - std_dev * sigma
    band_width = upper - lower
    pct_b = (close - lower) / band_width.replace(0, np.nan)
    return mid, upper, lower, pct_b.fillna(0.5)


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3, smooth: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Return (%K, %D) stochastic oscillator."""
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    raw_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(smooth).mean().fillna(50)
    d = k.rolling(d_period).mean().fillna(50)
    return k, d


def compute_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return a copy of df with all strategy indicators appended."""
    s  = config['strategy']
    df = df.copy()

    df['ema50']        = compute_ema(df['close'], s['ema_fast'])
    df['ema200']       = compute_ema(df['close'], s['ema_slow'])
    df['rsi']          = compute_rsi(df['close'], s['rsi_period'])
    df['macd_line'], df['macd_signal'], df['macd_hist'] = compute_macd(
        df['close'], s['macd_fast'], s['macd_slow'], s['macd_signal']
    )
    df['atr'] = compute_atr(df['high'], df['low'], df['close'], s['atr_period'])

    # ADX / DI
    adx_period = s.get('adx_period', 14)
    df['adx'], df['plus_di'], df['minus_di'] = compute_adx(
        df['high'], df['low'], df['close'], adx_period
    )

    # Bollinger Bands
    bb_period = s.get('bb_period', 20)
    bb_std    = s.get('bb_std', 2.0)
    df['bb_mid'], df['bb_upper'], df['bb_lower'], df['bb_pct'] = compute_bollinger(
        df['close'], bb_period, bb_std
    )

    # Stochastic
    df['stoch_k'], df['stoch_d'] = compute_stochastic(
        df['high'], df['low'], df['close'],
        s.get('stoch_k', 14), s.get('stoch_d', 3), s.get('stoch_smooth', 3)
    )

    return df.dropna()


# ── Market structure ──────────────────────────────────────────────────────────

def _find_swing_highs(highs: pd.Series, strength: int = 5) -> pd.Series:
    result = pd.Series(False, index=highs.index)
    arr = highs.values
    for i in range(strength, len(arr) - strength):
        window = arr[i - strength: i + strength + 1]
        if arr[i] == window.max():
            result.iloc[i] = True
    return result


def _find_swing_lows(lows: pd.Series, strength: int = 5) -> pd.Series:
    result = pd.Series(False, index=lows.index)
    arr = lows.values
    for i in range(strength, len(arr) - strength):
        window = arr[i - strength: i + strength + 1]
        if arr[i] == window.min():
            result.iloc[i] = True
    return result


def detect_market_structure(
    df: pd.DataFrame, lookback: int = 20, strength: int = 5
) -> Tuple[bool, bool, Optional[float], Optional[float]]:
    needed = (lookback * 3) + (strength * 2) + 5
    if len(df) < needed:
        return False, False, None, None

    recent = df.tail(needed)

    sh_mask = _find_swing_highs(recent['high'], strength)
    sl_mask = _find_swing_lows(recent['low'], strength)

    swing_highs = recent['high'][sh_mask].values
    swing_lows  = recent['low'][sl_mask].values

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, False, None, None

    last_sh, prev_sh = float(swing_highs[-1]), float(swing_highs[-2])
    last_sl, prev_sl = float(swing_lows[-1]),  float(swing_lows[-2])

    hh_hl = (last_sh > prev_sh) and (last_sl > prev_sl)
    lh_ll = (last_sh < prev_sh) and (last_sl < prev_sl)

    return hh_hl, lh_ll, last_sh, last_sl


# ── SQLite database ───────────────────────────────────────────────────────────

DB_PATH = Path('data/trades.db')


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket        INTEGER UNIQUE,
            symbol        TEXT,
            direction     TEXT,
            lot_size      REAL,
            entry_price   REAL,
            sl_price      REAL,
            tp_price      REAL,
            open_time     TEXT,
            close_time    TEXT,
            close_price   REAL,
            profit        REAL,
            status        TEXT DEFAULT 'open',
            ai_confidence REAL,
            notes         TEXT
        );
        CREATE TABLE IF NOT EXISTS equity_curve (
            ts      TEXT PRIMARY KEY,
            balance REAL,
            equity  REAL
        );
        CREATE TABLE IF NOT EXISTS activity_log (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      TEXT,
            symbol  TEXT NOT NULL,
            message TEXT NOT NULL,
            type    TEXT DEFAULT 'info'
        );
    ''')
    conn.commit()

    # Migration: add enriched journal columns if they don't exist yet.
    # SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we check
    # the column list first and add only what's missing.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    migrations = [
        ("session",      "TEXT DEFAULT ''"),    # LONDON / NEW_YORK / LONDON_NY / TOKYO / OFF_HOURS
        ("regime",       "TEXT DEFAULT ''"),    # TREND / RANGE / HIGH_VOL
        ("final_score",  "REAL DEFAULT 0.0"),   # composite quality score 0-1
        ("spread_pips",  "REAL DEFAULT 0.0"),   # spread at entry in pips
        ("close_reason", "TEXT DEFAULT ''"),    # SL / TP / EXIT_INTEL / MANUAL
        ("committee_verdict", "TEXT DEFAULT ''"),
        ("committee_score", "REAL DEFAULT 0.0"),
        ("committee_risk_multiplier", "REAL DEFAULT 1.0"),
    ]
    for col, col_def in migrations:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_def}")
    conn.commit()

    conn.execute("DELETE FROM activity_log WHERE LENGTH(ts) <= 8")
    conn.commit()
    conn.close()


def log_activity(symbol: str, message: str, msg_type: str = 'info') -> None:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn.execute(
            'INSERT INTO activity_log (ts, symbol, message, type) VALUES (?,?,?,?)',
            (ts, symbol, message, msg_type),
        )
        conn.execute(
            '''DELETE FROM activity_log
               WHERE symbol=? AND id NOT IN (
                   SELECT id FROM activity_log WHERE symbol=? ORDER BY id DESC LIMIT 100
               )''',
            (symbol, symbol),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_recent_activity(symbol: Optional[str] = None, limit: int = 20) -> list:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if symbol:
            rows = conn.execute(
                'SELECT ts, symbol, message, type FROM activity_log '
                'WHERE symbol=? ORDER BY id DESC LIMIT ?',
                (symbol.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT ts, symbol, message, type FROM activity_log '
                'ORDER BY id DESC LIMIT ?',
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def insert_trade(trade: dict) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    # Provide defaults for enriched fields so older callers don't break.
    row = {
        'ticket':       trade['ticket'],
        'symbol':       trade['symbol'],
        'direction':    trade['direction'],
        'lot_size':     trade['lot_size'],
        'entry_price':  trade['entry_price'],
        'sl_price':     trade['sl_price'],
        'tp_price':     trade['tp_price'],
        'open_time':    trade['open_time'],
        'ai_confidence':trade.get('ai_confidence', 0),
        'session':      trade.get('session', ''),
        'regime':       trade.get('regime', ''),
        'final_score':  trade.get('final_score', 0.0),
        'spread_pips':  trade.get('spread_pips', 0.0),
        'committee_verdict': trade.get('committee_verdict', ''),
        'committee_score': trade.get('committee_score', 0.0),
        'committee_risk_multiplier': trade.get('committee_risk_multiplier', 1.0),
    }
    conn.execute(
        '''INSERT OR REPLACE INTO trades
           (ticket, symbol, direction, lot_size, entry_price, sl_price, tp_price,
            open_time, ai_confidence, session, regime, final_score, spread_pips,
            committee_verdict, committee_score, committee_risk_multiplier, status)
           VALUES (:ticket, :symbol, :direction, :lot_size, :entry_price,
                   :sl_price, :tp_price, :open_time, :ai_confidence,
                   :session, :regime, :final_score, :spread_pips,
                   :committee_verdict, :committee_score,
                   :committee_risk_multiplier, 'open')''',
        row,
    )
    conn.commit()
    conn.close()


def close_trade_db(
    ticket: int,
    close_price: float,
    close_time: str,
    profit: float,
    close_reason: str = '',
) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        '''UPDATE trades
           SET close_price=?, close_time=?, profit=?, close_reason=?, status='closed'
           WHERE ticket=?''',
        (close_price, close_time, profit, close_reason, ticket),
    )
    conn.commit()
    conn.close()


def record_equity(balance: float, equity: float) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    ts = datetime.now().isoformat(timespec='seconds')
    conn.execute(
        'INSERT OR REPLACE INTO equity_curve (ts, balance, equity) VALUES (?,?,?)',
        (ts, balance, equity),
    )
    conn.commit()
    conn.close()


def get_open_trades_from_db() -> list:
    """Return all trades with status='open' — used to restore session state on restart."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT ticket, symbol, direction, ai_confidence, open_time "
            "FROM trades WHERE status='open'"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_trade_stats() -> dict:
    if not DB_PATH.exists():
        return {
            'win_rate': 0.0, 'profit_factor': 0.0, 'total_trades': 0,
            'total_profit': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
        }
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT profit FROM trades WHERE status='closed'"
    ).fetchall()
    conn.close()

    if not rows:
        return {
            'win_rate': 0.0, 'profit_factor': 0.0, 'total_trades': 0,
            'total_profit': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
        }

    profits = [r[0] for r in rows if r[0] is not None]
    wins    = [p for p in profits if p > 0]
    losses  = [p for p in profits if p < 0]

    win_rate      = len(wins) / len(profits) * 100 if profits else 0.0
    gross_profit  = sum(wins)
    gross_loss    = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        999.0 if gross_profit > 0 else 0.0
    )

    return {
        'win_rate':      round(win_rate, 2),
        'profit_factor': round(profit_factor, 3),
        'total_trades':  len(profits),
        'total_profit':  round(sum(profits), 2),
        'avg_win':       round(gross_profit / len(wins), 2) if wins else 0.0,
        'avg_loss':      round(gross_loss / len(losses), 2) if losses else 0.0,
    }


def _performance_from_profits(
    profits: list[float],
    total_trades: int | None = None,
    open_trades_today: int | None = None,
) -> dict:
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    closed_total = len(profits)
    total = closed_total if total_trades is None else int(total_trades)
    open_count = max(0, total - closed_total) if open_trades_today is None else int(open_trades_today)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        999.0 if gross_profit > 0 else 0.0
    )
    return {
        'total_trades': total,
        'closed_trades': closed_total,
        'open_trades_today': max(0, open_count),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round((len(wins) / closed_total) if closed_total else 0.0, 4),
        'today_pnl': round(sum(profits), 2),
        'total_profit': round(sum(profits), 2),
        'profit_factor': round(profit_factor, 3),
    }


def get_today_trade_stats() -> dict:
    """Trade stats for the local calendar day shown on the dashboard."""
    empty = _performance_from_profits([])
    if not DB_PATH.exists():
        return empty

    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    conn = sqlite3.connect(str(DB_PATH))
    opened_rows = conn.execute(
        """
        SELECT status FROM trades
        WHERE open_time >= ?
          AND open_time < ?
        """,
        (start.isoformat(timespec='seconds'), end.isoformat(timespec='seconds')),
    ).fetchall()
    closed_rows = conn.execute(
        """
        SELECT profit FROM trades
        WHERE status='closed'
          AND profit IS NOT NULL
          AND close_time >= ?
          AND close_time < ?
        """,
        (start.isoformat(timespec='seconds'), end.isoformat(timespec='seconds')),
    ).fetchall()
    conn.close()

    open_today = sum(1 for (status,) in opened_rows if str(status).lower() != 'closed')
    closed_profits = [float(profit) for (profit,) in closed_rows if profit is not None]
    return _performance_from_profits(
        closed_profits,
        total_trades=len(opened_rows),
        open_trades_today=open_today,
    )


# ── Learning analytics ────────────────────────────────────────────────────────

def get_context_performance(
    direction:    Optional[str] = None,
    session:      Optional[str] = None,
    regime:       Optional[str] = None,
    min_conf:     Optional[float] = None,
    max_conf:     Optional[float] = None,
    lookback_days: int = 60,
) -> dict:
    """
    Query closed trades filtered by context (session/regime/direction/confidence)
    and return performance statistics.

    Returns dict with keys: total, wins, losses, win_rate, profit_factor,
    avg_profit, enough_data (True when total >= 8).
    """
    empty = {
        'total': 0, 'wins': 0, 'losses': 0,
        'win_rate': 0.5, 'profit_factor': 1.0,
        'avg_profit': 0.0, 'enough_data': False,
    }
    if not DB_PATH.exists():
        return empty

    since = (datetime.now() - timedelta(days=lookback_days)).isoformat()

    conditions = ["status='closed'", "close_time >= ?", "profit IS NOT NULL"]
    params: list = [since]

    if direction:
        conditions.append("direction = ?")
        params.append(direction)
    if session:
        conditions.append("session = ?")
        params.append(session)
    if regime:
        conditions.append("regime = ?")
        params.append(regime)
    if min_conf is not None:
        conditions.append("final_score >= ?")
        params.append(min_conf)
    if max_conf is not None:
        conditions.append("final_score < ?")
        params.append(max_conf)

    sql = f"SELECT profit FROM trades WHERE {' AND '.join(conditions)}"

    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    except Exception:
        return empty

    profits = [r[0] for r in rows if r[0] is not None]
    if not profits:
        return empty

    wins_list   = [p for p in profits if p > 0]
    losses_list = [p for p in profits if p < 0]
    total       = len(profits)
    win_rate    = len(wins_list) / total
    gp          = sum(wins_list)
    gl          = abs(sum(losses_list)) if losses_list else 0.0
    pf          = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    return {
        'total':        total,
        'wins':         len(wins_list),
        'losses':       len(losses_list),
        'win_rate':     round(win_rate, 4),
        'profit_factor':round(pf, 3),
        'avg_profit':   round(sum(profits) / total, 2),
        'enough_data':  total >= 8,
    }


def compute_context_penalty(
    session:      str,
    regime:       str,
    direction:    str,
    final_score:  float,
    lookback_days: int = 60,
    min_trades:   int = 8,
    block_wr:     float = 0.20,
    penalty_wr:   float = 0.38,
) -> Tuple[float, bool, list]:
    """
    Compute a context-based score penalty from historical trade performance.

    Checks three contexts:
      1. Session  (e.g. LONDON)
      2. Regime   (e.g. RANGE)
      3. Direction (BUY / SELL)

    For each context with enough data (>= min_trades):
      - win_rate < block_wr (default 20%)  → block entirely
      - win_rate < penalty_wr (default 38%) → add penalty to score

    Penalty sizes:
      session   penalty: 0.08
      regime    penalty: 0.07
      direction penalty: 0.06

    Returns (penalty: float, block: bool, reasons: list[str])
    """
    penalty = 0.0
    block   = False
    reasons: list = []

    checks = [
        ('session',   {'session': session},   0.08),
        ('regime',    {'regime': regime},      0.07),
        ('direction', {'direction': direction}, 0.06),
    ]

    for label, filters, pen_size in checks:
        stats = get_context_performance(
            **filters, lookback_days=lookback_days
        )
        if not stats['enough_data'] or stats['total'] < min_trades:
            continue

        wr = stats['win_rate']
        n  = stats['total']

        if wr < block_wr:
            block = True
            reasons.append(
                f"BLOCK: {label}={list(filters.values())[0]} "
                f"win_rate={wr:.0%} ({n} trades) < {block_wr:.0%}"
            )
        elif wr < penalty_wr:
            penalty += pen_size
            reasons.append(
                f"penalty -{pen_size:.0%}: {label}={list(filters.values())[0]} "
                f"win_rate={wr:.0%} ({n} trades)"
            )

    return round(min(penalty, 0.20), 4), block, reasons


# ── Shared state JSON (live → dashboard) ─────────────────────────────────────

_STATE_PATH = Path('data/state.json')
_STATE_TMP  = Path('data/state.tmp.json')


def write_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_TMP.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')
    _STATE_TMP.replace(_STATE_PATH)


def read_state() -> dict:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}
