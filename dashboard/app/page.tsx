"use client";

import {
  useState, useEffect, useRef, useCallback, useMemo,
} from "react";
import dynamic from "next/dynamic";

// Recharts must be client-side only (no SSR)
const EquityAreaChart = dynamic(() => import("../components/EquityAreaChart"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full text-gray-600 text-sm">
      Loading chart…
    </div>
  ),
});

// ══════════════════════════════════════════════════════════════
// Types
// ══════════════════════════════════════════════════════════════

interface SymbolTerminal {
  price: number;
  spread_pips: number;
  signal: string;
  rsi: number;
  rsi_label: string;
  rsi_color: string;
  macd_status: string;
  macd_color: string;
  ema200_status: string;
  ema200_color: string;
  ai_confidence: number;
  ai_label: string;
  ai_bias: string;
  volatility_pct: number;
  volatility_ok: boolean;
  atr: number;
  updated: string;
}

interface Position {
  ticket: number;
  symbol: string;
  direction: string;
  lot?: number;
  lot_size?: number;
  open_price?: number;
  entry_price?: number;
  current_price: number;
  sl?: number;
  sl_price?: number;
  tp?: number;
  tp_price?: number;
  profit: number;
  committee_verdict?: string;
  committee_score?: number;
}

interface ActivityEntry {
  ts: string;
  symbol: string;
  message: string;
  type: string;
}

interface TradeRecord {
  ticket: number;
  symbol: string;
  direction: string;
  lot_size: number;
  entry_price: number;
  close_price: number;
  profit: number;
  open_time: string;
  close_time: string;
  status: string;
  ai_confidence: number;
}

interface Stats {
  total_trades: number;
  win_rate: number;
  total_profit: number;
  profit_factor: number;
}

interface EquityPoint {
  ts: string;
  balance: number;
  equity: number;
}

interface TradeState {
  timestamp: string;
  balance: number;
  equity: number;
  margin: number;
  free_margin: number;
  terminal: Record<string, SymbolTerminal>;
  open_positions?: Position[];
  open_trades?: Position[];
  drawdown_pct: number;
  daily_pnl: number;
  daily_start_balance: number;
  stats: Stats;
  activity: ActivityEntry[];
  equity_recent: EquityPoint[];
}

// ══════════════════════════════════════════════════════════════
// WebSocket hook
// ══════════════════════════════════════════════════════════════

function useTradeSocket(url: string) {
  const [data, setData]           = useState<TradeState | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef                     = useRef<WebSocket | null>(null);
  const retryRef                  = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen  = () => { setConnected(true); };
    ws.onclose = () => {
      setConnected(false);
      retryRef.current = setTimeout(connect, 3000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (e) => {
      try { setData(JSON.parse(e.data) as TradeState); } catch { /* ignore */ }
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { data, connected };
}

// ══════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════

function fmt(n: number | undefined, decimals = 2) {
  return n == null ? "—" : n.toFixed(decimals);
}

function signalColor(signal: string) {
  if (signal === "BUY"  || signal === "STRONG BUY")  return "text-emerald-400";
  if (signal === "SELL" || signal === "STRONG SELL") return "text-red-400";
  return "text-gray-400";
}

function signalBg(signal: string) {
  if (signal?.includes("BUY"))  return "signal-buy";
  if (signal?.includes("SELL")) return "signal-sell";
  return "signal-hold";
}

function logColor(type: string) {
  const map: Record<string, string> = {
    scan:      "text-yellow-400",
    indicator: "text-blue-400",
    ai:        "text-purple-400",
    signal:    "text-emerald-400",
    order:     "text-emerald-300",
    warning:   "text-red-400",
    info:      "text-gray-400",
  };
  return map[type] ?? "text-gray-500";
}

function logDot(type: string) {
  const map: Record<string, string> = {
    scan:      "bg-yellow-400",
    indicator: "bg-blue-400",
    ai:        "bg-purple-400",
    signal:    "bg-emerald-400",
    order:     "bg-emerald-300",
    warning:   "bg-red-400",
    info:      "bg-gray-500",
  };
  return map[type] ?? "bg-gray-600";
}

// ══════════════════════════════════════════════════════════════
// Sub-components
// ══════════════════════════════════════════════════════════════

// ── Live Clock ────────────────────────────────────────────────
function LiveClock() {
  const [time, setTime] = useState("");
  useEffect(() => {
    const update = () =>
      setTime(new Date().toLocaleTimeString("en-GB", { timeZone: "UTC", hour12: false }));
    update();
    const t = setInterval(update, 1000);
    return () => clearInterval(t);
  }, []);
  return <span className="font-mono text-sm text-gray-300">{time} UTC</span>;
}

// ── KPI Card ──────────────────────────────────────────────────
function KpiCard({
  label, value, sub, color = "text-white", icon,
}: { label: string; value: string; sub?: string; color?: string; icon?: string }) {
  return (
    <div className="card-sm flex flex-col gap-1 min-w-[110px]">
      <div className="flex items-center gap-1.5">
        {icon && <span className="text-base">{icon}</span>}
        <span className="text-[10px] uppercase tracking-widest text-gray-500">{label}</span>
      </div>
      <span className={`font-mono text-lg font-bold leading-tight ${color}`}>{value}</span>
      {sub && <span className="text-[10px] text-gray-600">{sub}</span>}
    </div>
  );
}

// ── Symbol Card ───────────────────────────────────────────────
function SymbolCard({ symbol, data }: { symbol: string; data: SymbolTerminal | undefined }) {
  const prevPrice = useRef<number>(0);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    if (!data) return;
    if (prevPrice.current && data.price !== prevPrice.current) {
      setFlash(data.price > prevPrice.current ? "up" : "down");
      const t = setTimeout(() => setFlash(null), 600);
      return () => clearTimeout(t);
    }
    prevPrice.current = data.price;
  }, [data?.price]);

  const isGold = symbol.includes("XAU");
  const accentColor = isGold ? "text-yellow-400" : "text-blue-400";
  const borderColor = isGold ? "border-yellow-500/20" : "border-blue-500/20";

  if (!data) {
    return (
      <div className={`card border ${borderColor} opacity-50`}>
        <div className={`section-title ${accentColor}`}>{symbol}</div>
        <div className="text-gray-600 text-sm font-mono">Waiting…</div>
      </div>
    );
  }

  const priceDecimals = isGold ? 2 : 5;
  const flashClass = flash === "up" ? "flash-up" : flash === "down" ? "flash-down" : "";

  return (
    <div className={`card border ${borderColor} glow-${isGold ? "green" : "blue"}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className={`text-xs font-bold uppercase tracking-widest ${accentColor}`}>
            {symbol}
          </span>
          {isGold && (
            <span className="ml-2 text-[10px] text-yellow-600">XAU/USD</span>
          )}
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold border ${signalBg(data.signal)}`}>
          {data.signal}
        </span>
      </div>

      {/* Price */}
      <div className={`font-mono text-3xl font-bold mb-1 transition-colors ${
        data.ai_bias === "bullish" ? "text-emerald-300" :
        data.ai_bias === "bearish" ? "text-red-300" : "text-white"
      } ${flashClass}`}>
        {data.price.toFixed(priceDecimals)}
      </div>

      {/* Spread */}
      <div className="flex items-center gap-3 mb-4 text-xs">
        <span className="text-gray-500">Spread</span>
        <span className={`font-mono font-semibold ${
          data.spread_pips > 5 ? "text-red-400" : "text-emerald-400"
        }`}>
          {data.spread_pips.toFixed(1)} pips
        </span>
        <span className="text-gray-600">ATR {data.atr.toFixed(isGold ? 2 : 5)}</span>
      </div>

      {/* Indicators grid */}
      <div className="space-y-2">
        {/* RSI */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 w-16">RSI</span>
          <div className="flex-1 mx-2">
            <div className="h-1 bg-[#1e2d40] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(data.rsi, 100)}%`,
                  background:
                    data.rsi > 70 ? "#ff4d6a" :
                    data.rsi < 30 ? "#ff4d6a" :
                    data.rsi > 50 ? "#00e896" : "#ffd166",
                }}
              />
            </div>
          </div>
          <span className={`text-xs font-mono font-semibold w-20 text-right ${
            data.rsi_color === "green" ? "text-emerald-400" :
            data.rsi_color === "red"   ? "text-red-400" : "text-yellow-400"
          }`}>
            {data.rsi.toFixed(1)} <span className="text-gray-600 font-normal">{data.rsi_label}</span>
          </span>
        </div>

        {/* MACD */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 w-16">MACD</span>
          <div className="flex-1" />
          <span className={`text-xs font-semibold ${
            data.macd_color === "green" ? "text-emerald-400" : "text-red-400"
          }`}>
            {data.macd_status}
          </span>
        </div>

        {/* EMA200 */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 w-16">EMA200</span>
          <div className="flex-1" />
          <span className={`text-xs font-semibold ${
            data.ema200_color === "white" ? "text-emerald-400" : "text-red-400"
          }`}>
            {data.ema200_status}
          </span>
        </div>

        {/* Volatility */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 w-16">Volatility</span>
          <div className="flex-1 mx-2">
            <div className="h-1 bg-[#1e2d40] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500 bg-blue-500"
                style={{ width: `${Math.min(data.volatility_pct, 100)}%` }}
              />
            </div>
          </div>
          <span className={`text-xs font-mono ${data.volatility_ok ? "text-blue-400" : "text-gray-600"}`}>
            {data.volatility_pct.toFixed(0)}%
          </span>
        </div>
      </div>

      <div className="mt-3 pt-2 border-t border-[#1e2d40] text-[10px] text-gray-600">
        Last update: {data.updated}
      </div>
    </div>
  );
}

// ── AI Signal Panel ───────────────────────────────────────────
function AISignalPanel({ terminal }: { terminal: Record<string, SymbolTerminal> }) {
  // Aggregate AI across all symbols — pick highest confidence with a direction
  const best = useMemo(() => {
    let top: SymbolTerminal & { symbol: string } | null = null;
    for (const [sym, t] of Object.entries(terminal)) {
      if (!top || t.ai_confidence > top.ai_confidence) {
        top = { ...t, symbol: sym };
      }
    }
    return top;
  }, [terminal]);

  if (!best) {
    return (
      <div className="card h-full flex items-center justify-center text-gray-600">
        Waiting for AI data…
      </div>
    );
  }

  const conf = best.ai_confidence;
  const isBuy    = best.ai_bias === "bullish";
  const isSell   = best.ai_bias === "bearish";
  const ringColor = isBuy ? "#00e896" : isSell ? "#ff4d6a" : "#4da6ff";
  const circumference = 2 * Math.PI * 54; // r=54

  return (
    <div className="card h-full">
      <div className="section-title">AI Signal Engine</div>

      <div className="flex items-center gap-6">
        {/* Circular progress */}
        <div className="relative flex-shrink-0">
          <svg width="128" height="128" className="-rotate-90">
            <circle
              cx="64" cy="64" r="54"
              fill="none" stroke="#1e2d40" strokeWidth="8"
            />
            <circle
              cx="64" cy="64" r="54"
              fill="none"
              stroke={ringColor}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${circumference}`}
              strokeDashoffset={`${circumference * (1 - conf / 100)}`}
              style={{ transition: "stroke-dashoffset 0.8s ease, stroke 0.4s ease" }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="font-mono text-3xl font-bold" style={{ color: ringColor }}>
              {conf}%
            </span>
            <span className="text-[10px] text-gray-500 uppercase tracking-wider">
              Confidence
            </span>
          </div>
        </div>

        {/* Right panel */}
        <div className="flex-1 space-y-3">
          <div>
            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1">
              Signal — {best.symbol}
            </div>
            <div className={`text-2xl font-bold ${signalColor(best.ai_label)}`}>
              {best.ai_label}
            </div>
          </div>

          <div>
            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1">
              Market Bias
            </div>
            <div className={`text-sm font-semibold capitalize ${
              isBuy ? "text-emerald-400" : isSell ? "text-red-400" : "text-gray-400"
            }`}>
              {best.ai_bias}
            </div>
          </div>

          {/* Gradient confidence bar */}
          <div>
            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1">
              Strength
            </div>
            <div className="h-2 bg-[#1e2d40] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${conf}%`,
                  background: `linear-gradient(90deg, #4da6ff 0%, ${ringColor} 100%)`,
                }}
              />
            </div>
            <div className="flex justify-between text-[9px] text-gray-600 mt-0.5">
              <span>0%</span>
              <span className="text-gray-500">
                {conf >= 80 ? "Strong" : conf >= 60 ? "Moderate" : conf >= 40 ? "Weak" : "Very Weak"}
              </span>
              <span>100%</span>
            </div>
          </div>

          {/* Per-symbol quick signals */}
          <div className="flex gap-2 flex-wrap">
            {Object.entries(terminal).map(([sym, t]) => (
              <div key={sym} className={`flex items-center gap-1.5 px-2 py-1 rounded text-[10px] border ${signalBg(t.signal)}`}>
                <span className="font-semibold">{sym}</span>
                <span>{t.ai_confidence}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Log Terminal ──────────────────────────────────────────────
function LogTerminal({ entries }: { entries: ActivityEntry[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [entries, autoScroll]);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    setAutoScroll(scrollHeight - scrollTop - clientHeight < 40);
  };

  const reversed = [...entries].reverse();

  return (
    <div className="card h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="section-title mb-0">System Log</div>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[10px] text-gray-500">LIVE</span>
          <button
            className={`text-[10px] px-2 py-0.5 rounded border ${
              autoScroll
                ? "border-emerald-500/40 text-emerald-400"
                : "border-gray-700 text-gray-500"
            }`}
            onClick={() => setAutoScroll((v) => !v)}
          >
            Auto-scroll
          </button>
        </div>
      </div>

      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto space-y-0.5 pr-1"
        style={{ minHeight: 0 }}
      >
        {reversed.map((entry, i) => (
          <div
            key={i}
            className="term-line items-start animate-slide-up"
          >
            <div className="flex items-center gap-1.5 flex-shrink-0 pt-0.5">
              <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 mt-0.5 ${logDot(entry.type)}`} />
              <span className="text-[10px] text-gray-600 w-16 flex-shrink-0 font-mono">
                {entry.ts?.slice(11, 19) ?? "—"}
              </span>
              <span className={`text-[10px] font-bold w-14 flex-shrink-0 ${
                entry.symbol === "XAUUSD" ? "text-yellow-600" : "text-blue-600"
              }`}>
                {entry.symbol}
              </span>
            </div>
            <span className={`text-xs leading-relaxed ${logColor(entry.type)}`}>
              {entry.message}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

// ── Open Positions ────────────────────────────────────────────
function OpenPositions({ positions }: { positions: Position[] }) {
  return (
    <div className="card">
      <div className="section-title">Open Positions ({positions.length})</div>
      {positions.length === 0 ? (
        <div className="text-gray-600 text-xs font-mono text-center py-4">
          No open positions
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-gray-600 border-b border-[#1e2d40]">
                <th className="text-left pb-2">Symbol</th>
                <th className="text-left pb-2">Dir</th>
                <th className="text-right pb-2">Lot</th>
                <th className="text-right pb-2">Entry</th>
                <th className="text-right pb-2">Price</th>
                <th className="text-right pb-2">P&L</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2d40]">
              {positions.map((p) => (
                <tr key={p.ticket} className="hover:bg-[#111d2e] transition-colors">
                  <td className="py-1.5 text-yellow-400 font-semibold">{p.symbol}</td>
                  <td className={`py-1.5 font-semibold ${
                    p.direction === "BUY" ? "text-emerald-400" : "text-red-400"
                  }`}>
                    {p.direction}
                  </td>
                  <td className="py-1.5 text-right text-gray-300">{(p.lot ?? p.lot_size ?? 0).toFixed(2)}</td>
                  <td className="py-1.5 text-right text-gray-400">{(p.open_price ?? p.entry_price ?? 0).toFixed(2)}</td>
                  <td className="py-1.5 text-right text-white">{p.current_price.toFixed(2)}</td>
                  <td className={`py-1.5 text-right font-bold ${
                    p.profit >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}>
                    {p.profit >= 0 ? "+" : ""}{p.profit.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Trade History ─────────────────────────────────────────────
function TradeHistory() {
  const [trades, setTrades] = useState<TradeRecord[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch("/api/trades");
        if (r.ok) setTrades(await r.json());
      } catch { /* ignore */ }
    };
    load();
    const t = setInterval(load, 30_000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="card h-full flex flex-col">
      <div className="section-title">Trade History ({trades.length})</div>
      <div className="flex-1 overflow-y-auto">
        {trades.length === 0 ? (
          <div className="text-gray-600 text-xs font-mono text-center py-8">
            No trades yet
          </div>
        ) : (
          <table className="w-full text-xs font-mono">
            <thead className="sticky top-0 bg-[#0d1424] z-10">
              <tr className="text-gray-600 border-b border-[#1e2d40]">
                <th className="text-left pb-2 pr-2">Symbol</th>
                <th className="text-left pb-2 pr-2">Dir</th>
                <th className="text-right pb-2 pr-2">Lot</th>
                <th className="text-right pb-2 pr-2">Entry</th>
                <th className="text-right pb-2 pr-2">Close</th>
                <th className="text-right pb-2">P&L</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2d40]">
              {trades.map((t) => (
                <tr key={t.ticket} className="hover:bg-[#111d2e] transition-colors">
                  <td className="py-1.5 pr-2 text-yellow-400">{t.symbol}</td>
                  <td className={`py-1.5 pr-2 font-semibold ${
                    t.direction === "BUY" ? "text-emerald-400" : "text-red-400"
                  }`}>
                    {t.direction}
                  </td>
                  <td className="py-1.5 pr-2 text-right text-gray-400">
                    {t.lot_size?.toFixed(2) ?? "—"}
                  </td>
                  <td className="py-1.5 pr-2 text-right text-gray-300">
                    {t.entry_price?.toFixed(2) ?? "—"}
                  </td>
                  <td className="py-1.5 pr-2 text-right text-gray-300">
                    {t.status === "OPEN"
                      ? <span className="text-yellow-600">OPEN</span>
                      : (t.close_price?.toFixed(2) ?? "—")}
                  </td>
                  <td className={`py-1.5 text-right font-bold ${
                    (t.profit ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}>
                    {t.profit != null
                      ? `${t.profit >= 0 ? "+" : ""}${t.profit.toFixed(2)}`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// Main Dashboard
// ══════════════════════════════════════════════════════════════

export default function Dashboard() {
  const { data, connected } = useTradeSocket("ws://localhost:8000/ws");

  const terminal   = data?.terminal   ?? {};
  const positions  = data?.open_positions ?? data?.open_trades ?? [];
  const activity   = data?.activity   ?? [];
  const stats      = data?.stats;
  const equityData = data?.equity_recent ?? [];

  const symbols = ["XAUUSD", "EURUSD"];

  return (
    <div className="min-h-screen bg-[#050a14] text-gray-100">

      {/* ── HEADER ─────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 bg-[#050a14]/95 backdrop-blur border-b border-[#1e2d40]">
        <div className="max-w-[1800px] mx-auto px-4 h-14 flex items-center justify-between">

          {/* Left: Brand */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-blue-600 flex items-center justify-center text-white text-sm font-bold">
              AI
            </div>
            <div>
              <div className="text-sm font-bold tracking-wider text-white">AI-TRADE</div>
              <div className="text-[10px] text-gray-500 uppercase tracking-widest">
                Autonomous System
              </div>
            </div>
          </div>

          {/* Center: Status */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                connected ? "bg-emerald-400 animate-pulse" : "bg-red-500 animate-blink"
              }`} />
              <span className={`text-xs font-semibold ${
                connected ? "text-emerald-400" : "text-red-400"
              }`}>
                {connected ? "ENGINE LIVE" : "RECONNECTING…"}
              </span>
            </div>

            <div className="hidden md:flex items-center gap-4 text-xs text-gray-500">
              <span>
                Balance{" "}
                <span className="text-white font-mono font-semibold">
                  ${fmt(data?.balance)}
                </span>
              </span>
              <span>
                Equity{" "}
                <span className={`font-mono font-semibold ${
                  (data?.equity ?? 0) >= (data?.balance ?? 0)
                    ? "text-emerald-400" : "text-red-400"
                }`}>
                  ${fmt(data?.equity)}
                </span>
              </span>
              <span>
                DD{" "}
                <span className={`font-mono font-semibold ${
                  (data?.drawdown_pct ?? 0) > 5 ? "text-red-400" : "text-gray-300"
                }`}>
                  {fmt(data?.drawdown_pct)}%
                </span>
              </span>
            </div>
          </div>

          {/* Right: Time */}
          <div className="flex items-center gap-3">
            <div className="text-[10px] uppercase tracking-wider text-gray-600 hidden sm:block">
              Server: Local
            </div>
            <LiveClock />
          </div>
        </div>
      </header>

      {/* ── KPI STRIP ──────────────────────────────────────────── */}
      <div className="max-w-[1800px] mx-auto px-4 py-3">
        <div className="flex gap-2 overflow-x-auto pb-1">
          <KpiCard
            label="Balance" icon="💰"
            value={`$${fmt(data?.balance)}`}
            color="text-white"
          />
          <KpiCard
            label="Equity" icon="📊"
            value={`$${fmt(data?.equity)}`}
            color={(data?.equity ?? 0) >= (data?.balance ?? 0) ? "text-emerald-400" : "text-red-400"}
          />
          <KpiCard
            label="Daily P&L" icon="📈"
            value={`${(data?.daily_pnl ?? 0) >= 0 ? "+" : ""}$${fmt(data?.daily_pnl)}`}
            color={(data?.daily_pnl ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}
          />
          <KpiCard
            label="Drawdown" icon="📉"
            value={`${fmt(data?.drawdown_pct)}%`}
            color={(data?.drawdown_pct ?? 0) > 5 ? "text-red-400" : "text-gray-300"}
          />
          <KpiCard
            label="Win Rate" icon="🎯"
            value={`${fmt(stats?.win_rate ?? 0, 1)}%`}
            color={(stats?.win_rate ?? 0) >= 50 ? "text-emerald-400" : "text-yellow-400"}
          />
          <KpiCard
            label="Profit Factor" icon="⚡"
            value={fmt(stats?.profit_factor ?? 0)}
            color={(stats?.profit_factor ?? 0) >= 1.5 ? "text-emerald-400" : "text-yellow-400"}
          />
          <KpiCard
            label="Total Trades" icon="📋"
            value={String(stats?.total_trades ?? 0)}
          />
          <KpiCard
            label="Open" icon="🔓"
            value={String(positions.length)}
            color={positions.length > 0 ? "text-yellow-400" : "text-gray-400"}
          />
          <KpiCard
            label="Free Margin" icon="🏦"
            value={`$${fmt(data?.free_margin)}`}
            color="text-blue-300"
          />
        </div>
      </div>

      {/* ── MAIN CONTENT GRID ──────────────────────────────────── */}
      <div className="max-w-[1800px] mx-auto px-4 pb-4">
        <div className="grid grid-cols-12 gap-3">

          {/* ── LEFT: Symbol Cards ───────────────────── col 1-3 */}
          <div className="col-span-12 lg:col-span-3 space-y-3">
            {symbols.map((sym) => (
              <SymbolCard key={sym} symbol={sym} data={terminal[sym]} />
            ))}
          </div>

          {/* ── CENTER ───────────────────────────────── col 4-9 */}
          <div className="col-span-12 lg:col-span-6 flex flex-col gap-3">

            {/* AI Signal */}
            <div style={{ height: "200px" }}>
              <AISignalPanel terminal={terminal} />
            </div>

            {/* Equity Chart */}
            <div className="card flex-1" style={{ minHeight: "220px" }}>
              <div className="section-title">Equity Curve</div>
              <div style={{ height: "180px" }}>
                <EquityAreaChart data={equityData} />
              </div>
            </div>

            {/* Open Positions */}
            <OpenPositions positions={positions} />
          </div>

          {/* ── RIGHT ────────────────────────────────── col 10-12 */}
          <div className="col-span-12 lg:col-span-3 flex flex-col gap-3">

            {/* Log Terminal */}
            <div style={{ height: "460px" }}>
              <LogTerminal entries={activity} />
            </div>

            {/* Trade History */}
            <div style={{ height: "340px" }}>
              <TradeHistory />
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}
