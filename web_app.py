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
import sqlite3
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AI-Trade Dashboard", docs_url=None, redoc_url=None)

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _query(sql: str, params: tuple = ()) -> list:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
    return state


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
                await ws.send_json(payload)
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
            await websocket.send_json(payload)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/state")
def api_state():
    return JSONResponse(_read_state())


@app.get("/api/equity")
def api_equity():
    rows = _query(
        "SELECT ts, balance, equity FROM equity_curve ORDER BY ts DESC LIMIT 600"
    )
    rows.reverse()
    return JSONResponse(rows)


@app.get("/api/trades")
def api_trades():
    """Return only OPEN trades (status='open' or close_time IS NULL)"""
    return JSONResponse(_query(
        """SELECT ticket, symbol, direction, lot_size, entry_price,
                  sl_price, tp_price, open_time, close_time,
                  close_price, profit, status, ai_confidence
           FROM trades 
           WHERE status='open' OR close_time IS NULL
           ORDER BY open_time DESC LIMIT 100"""
    ))


@app.get("/api/trades/history")
def api_trades_history():
    """Return ALL trades (open + closed) for historical analysis"""
    return JSONResponse(_query(
        """SELECT ticket, symbol, direction, lot_size, entry_price,
                  sl_price, tp_price, open_time, close_time,
                  close_price, profit, status, ai_confidence
           FROM trades 
           ORDER BY open_time DESC LIMIT 200"""
    ))


@app.get("/api/activity")
def api_activity():
    return JSONResponse(_query(
        "SELECT ts, symbol, message, type FROM activity_log "
        "WHERE LENGTH(ts) > 8 ORDER BY id DESC LIMIT 30"
    ))


@app.get("/api/activity/{symbol}")
def api_activity_symbol(symbol: str):
    return JSONResponse(_query(
        "SELECT ts, symbol, message, type FROM activity_log "
        "WHERE symbol=? AND LENGTH(ts) > 8 ORDER BY id DESC LIMIT 15",
        (symbol.upper(),),
    ))


@app.get("/api/ai_insights")
def api_ai_insights():
    """AI prediction breakdown: tabular, regime, LSTM, online, RL, memory."""
    return JSONResponse(_read_json(INSIGHTS_PATH))


@app.get("/api/learning_stats")
def api_learning_stats():
    """Learning progress: RL agent, market memory, online model stats."""
    return JSONResponse(_read_json(LEARN_PATH))


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
