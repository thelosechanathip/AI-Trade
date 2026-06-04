"""
web_app.py -- FastAPI backend + real-time WebSocket.

Endpoints:
  GET  /                   -> serve static index.html
  GET  /api/state          -> current engine state JSON
  GET  /api/equity         -> equity curve (last 600 rows)
  GET  /api/trades         -> open trades
  GET  /api/trades/history -> all trades (open + closed)
  GET  /api/activity       -> activity log (last 30)
  GET  /api/activity/{symbol}
  GET  /api/ai_insights    -> AI prediction breakdown (RL, memory, ensemble)
  GET  /api/learning_stats -> online/RL/memory learning progress
  WS   /ws                 -> real-time push (state + activity + insights every second)

CORS is open so the Next.js dev server (port 3000) can connect.
"""

import asyncio
import json
import math
import sqlite3
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AI-Trade Dashboard", docs_url=None, redoc_url=None)
ASSETS_DIR = Path("static/assets")
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH        = Path("data/trades.db")
STATE_PATH     = Path("data/state.json")
INSIGHTS_PATH  = Path("data/ai_insights.json")
LEARN_PATH     = Path("data/learning_stats.json")
BRAIN_DB_PATH  = Path("data/brain_memory.db")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _json_safe(value):
    """Recursively remove NaN/Infinity values before strict JSON responses."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _json_response(value):
    return JSONResponse(_json_safe(value))


def _read_state() -> dict:
    if not STATE_PATH.exists():
        return _with_live_today_stats({})
    try:
        return _with_live_today_stats(json.loads(STATE_PATH.read_text(encoding="utf-8")))
    except Exception:
        return _with_live_today_stats({})


def _perf_from_profits(
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
        "total_trades": total,
        "closed_trades": closed_total,
        "open_trades_today": max(0, open_count),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round((len(wins) / closed_total) if closed_total else 0.0, 4),
        "today_pnl": round(sum(profits), 2),
        "total_profit": round(sum(profits), 2),
        "profit_factor": round(profit_factor, 3),
    }


def _today_trade_stats() -> dict:
    if not DB_PATH.exists():
        return _perf_from_profits([])
    from datetime import datetime, timedelta

    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    opened_rows = _query(
        """
        SELECT status FROM trades
        WHERE open_time >= ?
          AND open_time < ?
        """,
        (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
    )
    closed_rows = _query(
        """
        SELECT profit FROM trades
        WHERE status='closed'
          AND profit IS NOT NULL
          AND close_time >= ?
          AND close_time < ?
        """,
        (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
    )
    open_today = sum(1 for r in opened_rows if str(r.get("status", "")).lower() != "closed")
    closed_profits = [float(r.get("profit")) for r in closed_rows if r.get("profit") is not None]
    return _perf_from_profits(
        closed_profits,
        total_trades=len(opened_rows),
        open_trades_today=open_today,
    )


def _with_live_today_stats(state: dict) -> dict:
    state = dict(state or {})
    old_stats = state.get("stats") or {}
    today_stats = _today_trade_stats()
    today_stats["weekly_pnl"] = old_stats.get("weekly_pnl", state.get("weekly_pnl", 0.0))
    state["stats"] = today_stats
    state["runtime"] = {
        "cwd": str(Path.cwd()),
        "state_path": str(STATE_PATH.resolve()),
        "state_mtime": STATE_PATH.stat().st_mtime if STATE_PATH.exists() else None,
    }
    return state


def _query(sql: str, params: tuple = ()) -> list:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _brain_query(sql: str, params: tuple = ()) -> list:
    if not BRAIN_DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(BRAIN_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _decode_jsonish(value, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _trade_select_sql(limit: int = 100, history: bool = False) -> str:
    base_cols = [
        "ticket", "symbol", "direction", "lot_size", "entry_price",
        "sl_price", "tp_price", "open_time", "close_time",
        "close_price", "profit", "status", "ai_confidence",
    ]
    optional = [
        "committee_verdict", "committee_score", "committee_risk_multiplier",
    ]
    existing = set()
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        try:
            existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
        finally:
            conn.close()

    cols = base_cols + [c for c in optional if c in existing]
    where = "" if history else "WHERE status='open' OR close_time IS NULL"
    return (
        f"SELECT {', '.join(cols)} FROM trades "
        f"{where} ORDER BY open_time DESC LIMIT {int(limit)}"
    )


def _build_learning_journal() -> dict:
    """Summarise what the engine has recorded after losses and weak regimes."""
    if not BRAIN_DB_PATH.exists():
        return {
            "enabled": False,
            "summary": "ยังไม่พบ brain_memory.db",
            "action_items": [
                "ระบบมีโค้ดสำหรับจดจำบทเรียน แต่ยังไม่มีฐานข้อมูล memory ให้ dashboard อ่าน",
            ],
            "recent_failures": [],
            "weak_regimes": [],
            "recent_losses": [],
        }

    failures = _brain_query(
        """
        SELECT ts, symbol, direction, regime, signals, loss_r, bars_held, notes
        FROM failure_patterns
        ORDER BY ts DESC LIMIT 8
        """
    )
    feedback = _brain_query(
        """
        SELECT regime, win_count, loss_count, total_profit_r,
               avg_conf_win, avg_conf_loss, avg_bars_win, avg_bars_loss, last_updated
        FROM learning_feedback
        ORDER BY last_updated DESC LIMIT 10
        """
    )
    losses = _brain_query(
        """
        SELECT ticket, symbol, direction, profit, market_regime,
               signals_active, reasoning, entry_time, exit_time
        FROM brain_trades
        WHERE outcome='loss'
        ORDER BY exit_time DESC LIMIT 6
        """
    )

    for row in failures:
        row["signals"] = _decode_jsonish(row.get("signals"), [])
    for row in losses:
        row["signals_active"] = _decode_jsonish(row.get("signals_active"), [])
        row["reasoning"] = _decode_jsonish(row.get("reasoning"), [])

    weak_regimes = []
    for row in feedback:
        wins = int(row.get("win_count") or 0)
        loss_count = int(row.get("loss_count") or 0)
        total = wins + loss_count
        win_rate = wins / total if total else 0.5
        item = dict(row)
        item["total"] = total
        item["win_rate"] = round(win_rate, 4)
        if total >= 3 and win_rate < 0.45:
            weak_regimes.append(item)

    action_items = []
    if failures:
        last = failures[0]
        action_items.append(
            f"พบแพทเทิร์นแพ้ล่าสุด: {last.get('regime') or 'UNKNOWN'} / "
            f"{last.get('direction') or '-'} ระบบจะลดความมั่นใจเมื่อเจอ setup คล้ายเดิม"
        )
    if weak_regimes:
        worst = sorted(weak_regimes, key=lambda r: r["win_rate"])[0]
        action_items.append(
            f"Regime ที่ควรระวัง: {worst.get('regime')} "
            f"win rate {worst['win_rate']*100:.1f}% จาก {worst['total']} ไม้"
        )
    if losses:
        action_items.append(
            "มีบันทึก trade ที่ปิดขาดทุนแล้ว ระบบใช้ข้อมูลนี้ปรับ confidence, "
            "failure similarity และ cooldown"
        )
    if not action_items:
        action_items.append(
            "ระบบพร้อมจดบันทึก แต่ข้อมูลปิดไม้ยังน้อย จึงยังไม่มีข้อเสนอปรับกลยุทธ์ที่หนักพอ"
        )

    return {
        "enabled": True,
        "summary": (
            f"บันทึกแพทเทิร์นแพ้ {len(failures)} รายการ, "
            f"regime feedback {len(feedback)} รายการ, recent losses {len(losses)} รายการ"
        ),
        "action_items": action_items[:4],
        "recent_failures": failures,
        "weak_regimes": weak_regimes[:6],
        "recent_losses": losses,
    }


def _build_broadcast() -> dict:
    """Assemble the full state payload pushed over WebSocket every second."""
    state = _read_state()
    state['activity'] = _query(
        "SELECT ts, symbol, message, type FROM activity_log "
        "WHERE LENGTH(ts) > 8 ORDER BY id DESC LIMIT 40"
    )
    state['equity_recent'] = _query(
        "SELECT ts, balance, equity FROM equity_curve ORDER BY ts DESC LIMIT 60"
    )[::-1]
    # AI insights (prediction breakdown)
    state['ai_insights']    = _read_json(INSIGHTS_PATH)
    state['learning_stats'] = _read_json(LEARN_PATH)
    state['learning_journal'] = _build_learning_journal()
    return _json_safe(state)


# ── WebSocket connection manager ───────────────────────────────────────────────

class _WSManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, payload: dict):
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(_json_safe(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = _WSManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            payload = _build_broadcast()
            await websocket.send_json(_json_safe(payload))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/state")
def api_state():
    return _json_response(_read_state())


@app.get("/api/equity")
def api_equity():
    rows = _query(
        "SELECT ts, balance, equity FROM equity_curve ORDER BY ts DESC LIMIT 600"
    )
    rows.reverse()
    return _json_response(rows)


@app.get("/api/trades")
def api_trades():
    """Return only OPEN trades (status='open' or close_time IS NULL)"""
    return _json_response(_query(_trade_select_sql(limit=100, history=False)))


@app.get("/api/trades/history")
def api_trades_history():
    """Return ALL trades (open + closed) for historical analysis"""
    return _json_response(_query(_trade_select_sql(limit=200, history=True)))


@app.get("/api/activity")
def api_activity():
    return _json_response(_query(
        "SELECT ts, symbol, message, type FROM activity_log "
        "WHERE LENGTH(ts) > 8 ORDER BY id DESC LIMIT 30"
    ))


@app.get("/api/activity/{symbol}")
def api_activity_symbol(symbol: str):
    return _json_response(_query(
        "SELECT ts, symbol, message, type FROM activity_log "
        "WHERE symbol=? AND LENGTH(ts) > 8 ORDER BY id DESC LIMIT 15",
        (symbol.upper(),),
    ))


@app.get("/api/ai_insights")
def api_ai_insights():
    """AI prediction breakdown: tabular, regime, LSTM, online, RL, memory."""
    return _json_response(_read_json(INSIGHTS_PATH))


@app.get("/api/learning_stats")
def api_learning_stats():
    """Learning progress: RL agent, market memory, online model stats."""
    return _json_response(_read_json(LEARN_PATH))


@app.get("/api/learning_journal")
def api_learning_journal():
    """Readable post-loss memory and strategy-adjustment notes."""
    return _json_response(_build_learning_journal())


# ── Legacy HTML dashboard (static) ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    p = Path("static/index.html")
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AI-Trade API</h1><p>Open the Next.js dashboard on port 3000.</p>")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
