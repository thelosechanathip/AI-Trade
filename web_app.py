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
import threading
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from execution_controls import (
    ExecutionControlConflict,
    ExecutionControlError,
    ExecutionControlStore,
)

app = FastAPI(title="AI-Trade Dashboard", docs_url=None, redoc_url=None)
ASSETS_DIR = Path("static/assets")
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH        = Path("data/trades.db")
STATE_PATH     = Path("data/state.json")
INSIGHTS_PATH  = Path("data/ai_insights.json")
LEARN_PATH     = Path("data/learning_stats.json")
BRAIN_DB_PATH  = Path("data/brain_memory.db")
_MT5_LOCK = threading.Lock()
_MT5_SYMBOL_CACHE: dict[str, str] = {}
_LIVE_EQUITY_LOCK = threading.Lock()
_LIVE_EQUITY_POINTS: list[dict] = []


def _load_root_config() -> dict:
    try:
        return yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


ROOT_CONFIG = _load_root_config()
CONTROL_STORE = ExecutionControlStore(ROOT_CONFIG)


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


def _float_attr(value, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(value, name, default) or default)
    except (TypeError, ValueError):
        return default


def _resolve_live_symbol(mt5, configured_symbol: str):
    actual = _MT5_SYMBOL_CACHE.get(configured_symbol)
    if actual:
        info = mt5.symbol_info(actual)
        if info is not None:
            return actual, info

    info = mt5.symbol_info(configured_symbol)
    if info is not None:
        _MT5_SYMBOL_CACHE[configured_symbol] = configured_symbol
        return configured_symbol, info

    matches = mt5.symbols_get(group=f"*{configured_symbol}*") or ()
    configured_upper = configured_symbol.upper()
    actual = next(
        (s.name for s in matches if s.name.upper().startswith(configured_upper)),
        None,
    ) or next(
        (s.name for s in matches if configured_upper in s.name.upper()),
        configured_symbol,
    )
    info = mt5.symbol_info(actual)
    if info is not None:
        _MT5_SYMBOL_CACHE[configured_symbol] = actual
    return actual, info


def _serialize_live_positions(mt5, positions, state: dict) -> list[dict]:
    old_positions = state.get("open_positions") or state.get("open_trades") or []
    old_by_ticket = {
        int(item.get("ticket", 0)): dict(item)
        for item in old_positions
        if item.get("ticket") is not None
    }
    buy_type = getattr(mt5, "POSITION_TYPE_BUY", 0)
    serialized = []

    for position in positions:
        ticket = int(getattr(position, "ticket", 0) or 0)
        item = old_by_ticket.get(ticket, {})
        item.update({
            "ticket": ticket,
            "symbol": str(getattr(position, "symbol", "")),
            "direction": "BUY" if getattr(position, "type", -1) == buy_type else "SELL",
            "lot": _float_attr(position, "volume"),
            "lot_size": _float_attr(position, "volume"),
            "open_price": _float_attr(position, "price_open"),
            "entry_price": _float_attr(position, "price_open"),
            "current_price": _float_attr(position, "price_current"),
            "sl": _float_attr(position, "sl"),
            "sl_price": _float_attr(position, "sl"),
            "tp": _float_attr(position, "tp"),
            "tp_price": _float_attr(position, "tp"),
            "profit": round(_float_attr(position, "profit"), 2),
            "swap": round(_float_attr(position, "swap"), 2),
            "magic": int(getattr(position, "magic", 0) or 0),
        })
        serialized.append(item)

    return serialized


def _mt5_today_stats(mt5, positions) -> tuple[dict, float]:
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    deals = mt5.history_deals_get(start, datetime.now()) or ()
    trade_types = {
        getattr(mt5, "DEAL_TYPE_BUY", 0),
        getattr(mt5, "DEAL_TYPE_SELL", 1),
    }
    closing_entries = {
        getattr(mt5, "DEAL_ENTRY_OUT", 1),
        getattr(mt5, "DEAL_ENTRY_INOUT", 2),
        getattr(mt5, "DEAL_ENTRY_OUT_BY", 3),
    }
    by_position: dict[int, dict] = {}
    realized_today = 0.0

    for deal in deals:
        if getattr(deal, "type", None) not in trade_types:
            continue
        amount = sum(
            _float_attr(deal, field)
            for field in ("profit", "commission", "swap", "fee")
        )
        realized_today += amount
        position_id = int(
            getattr(deal, "position_id", 0)
            or getattr(deal, "order", 0)
            or getattr(deal, "ticket", 0)
            or 0
        )
        group = by_position.setdefault(position_id, {"profit": 0.0, "closed": False})
        group["profit"] += amount
        if getattr(deal, "entry", None) in closing_entries:
            group["closed"] = True

    closed_profits = [
        group["profit"] for group in by_position.values() if group["closed"]
    ]
    start_ts = start.timestamp()
    open_today = sum(
        1 for position in positions
        if _float_attr(position, "time") >= start_ts
    )
    stats = _perf_from_profits(
        closed_profits,
        total_trades=len(closed_profits) + open_today,
        open_trades_today=open_today,
    )
    stats["today_pnl"] = round(realized_today, 2)
    stats["total_profit"] = round(realized_today, 2)
    return stats, realized_today


def _with_live_mt5(state: dict) -> dict:
    """Overlay account, tick, and position data directly from the connected MT5."""
    state = dict(state or {})
    analysis_timestamp = state.get("timestamp")
    now = datetime.now().astimezone()
    live = {
        "connected": False,
        "source": "snapshot",
        "timestamp": now.isoformat(timespec="milliseconds"),
        "analysis_timestamp": analysis_timestamp,
    }

    try:
        import MetaTrader5 as mt5
    except ImportError:
        live["error"] = "MetaTrader5 package is not installed"
        state["live"] = live
        return state

    try:
        with _MT5_LOCK:
            account = mt5.account_info()
            if account is None:
                last_error = mt5.last_error() if hasattr(mt5, "last_error") else None
                live["error"] = f"MT5 account unavailable: {last_error}"
                state["live"] = live
                return state

            positions = list(mt5.positions_get() or ())
            live_positions = _serialize_live_positions(mt5, positions, state)
            today_stats, realized_today = _mt5_today_stats(mt5, positions)

            terminal = {
                symbol: dict(values)
                for symbol, values in (state.get("terminal") or {}).items()
            }
            symbols = list(dict.fromkeys(
                list(ROOT_CONFIG.get("trading", {}).get("symbols", []))
                + list(terminal)
            ))
            symbols_summary = {
                symbol: dict(values)
                for symbol, values in (state.get("symbols") or {}).items()
            }

            for symbol in symbols:
                actual, symbol_info = _resolve_live_symbol(mt5, symbol)
                tick = mt5.symbol_info_tick(actual)
                if tick is None:
                    continue

                bid = _float_attr(tick, "bid")
                ask = _float_attr(tick, "ask")
                point = _float_attr(symbol_info, "point") if symbol_info else 0.0
                spread_points = (ask - bid) / point if point > 0 else 0.0
                spread_pips = spread_points / 10
                tick_msc = _float_attr(tick, "time_msc")
                tick_timestamp = (
                    datetime.fromtimestamp(tick_msc / 1000, timezone.utc).isoformat()
                    if tick_msc > 0
                    else now.astimezone(timezone.utc).isoformat()
                )

                terminal_item = terminal.setdefault(symbol, {})
                terminal_item.update({
                    "price": bid,
                    "bid": bid,
                    "ask": ask,
                    "spread_pips": round(spread_pips, 1),
                    "spread_points": round(spread_points, 1),
                    "broker_symbol": actual,
                    "live": True,
                    "tick_timestamp": tick_timestamp,
                    "updated": now.strftime("%H:%M:%S"),
                })
                summary_item = symbols_summary.setdefault(symbol, {})
                summary_item.update({
                    "close": bid,
                    "bid": bid,
                    "ask": ask,
                    "spread": round(spread_pips, 1),
                    "broker_symbol": actual,
                })

        floating_pnl = round(sum(item["profit"] for item in live_positions), 2)
        balance = _float_attr(account, "balance")
        equity = _float_attr(account, "equity")
        old_stats = state.get("stats") or {}
        today_stats["weekly_pnl"] = old_stats.get(
            "weekly_pnl", state.get("weekly_pnl", 0.0)
        )
        peak_balance = max(
            balance,
            float(state.get("peak_balance") or balance or 0.0),
        )

        state.update({
            "timestamp": now.isoformat(timespec="seconds"),
            "balance": balance,
            "equity": equity,
            "peak_balance": peak_balance,
            "margin": _float_attr(account, "margin"),
            "margin_level": _float_attr(account, "margin_level"),
            "free_margin": _float_attr(account, "margin_free"),
            "terminal": terminal,
            "symbols": symbols_summary,
            "open_positions": live_positions,
            "open_trades": live_positions,
            "floating_pnl": floating_pnl,
            "daily_pnl": round(realized_today + floating_pnl, 2),
            "drawdown_pct": round(
                max(0.0, (peak_balance - equity) / peak_balance * 100)
                if peak_balance > 0 else 0.0,
                2,
            ),
            "stats": today_stats,
        })
        live_equity_point = {
            "ts": now.isoformat(timespec="seconds"),
            "balance": balance,
            "equity": equity,
        }
        with _LIVE_EQUITY_LOCK:
            if (
                _LIVE_EQUITY_POINTS
                and _LIVE_EQUITY_POINTS[-1]["ts"] == live_equity_point["ts"]
            ):
                _LIVE_EQUITY_POINTS[-1] = live_equity_point
            else:
                _LIVE_EQUITY_POINTS.append(live_equity_point)
                del _LIVE_EQUITY_POINTS[:-60]
            live_equity_points = list(_LIVE_EQUITY_POINTS)

        historical_equity = list(state.get("equity_recent") or [])
        state["equity_recent"] = (historical_equity + live_equity_points)[-60:]
        live.update({
            "connected": True,
            "source": "mt5",
            "account_login": int(getattr(account, "login", 0) or 0),
            "server": str(getattr(account, "server", "") or ""),
            "currency": str(getattr(account, "currency", "") or ""),
            "position_count": len(live_positions),
        })
    except Exception as exc:
        live["error"] = f"MT5 live read failed: {exc}"

    state["live"] = live
    return state


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
    state['execution_controls'] = CONTROL_STORE.get()
    return _json_safe(_with_live_mt5(state))


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
    return _json_response(_with_live_mt5(_read_state()))


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


@app.get("/api/execution-controls")
def api_execution_controls():
    """Return live operator controls plus server-enforced guardrails."""
    return _json_response(CONTROL_STORE.get())


@app.put("/api/execution-controls")
def api_update_execution_controls(payload: dict, request: Request):
    """Update operator controls from localhost with optimistic revision checks."""
    host = request.client.host if request.client else ""
    local_hosts = {"127.0.0.1", "::1", "localhost", "testclient"}
    configured_key = str(
        ROOT_CONFIG.get("dashboard", {}).get("control_api_key", "") or ""
    )
    supplied_key = request.headers.get("x-control-key", "")
    if host not in local_hosts and (not configured_key or supplied_key != configured_key):
        raise HTTPException(status_code=403, detail="control writes require localhost or API key")

    actor = request.headers.get("x-operator-id") or host or "dashboard"
    try:
        return _json_response(CONTROL_STORE.update(payload, actor=actor))
    except ExecutionControlConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ExecutionControlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
