"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface Point {
  ts: string;
  balance: number;
  equity: number;
}

interface Props {
  data: Point[];
}

// Format time label from ISO timestamp
function fmtTime(ts: string) {
  if (!ts) return "";
  try {
    return new Date(ts).toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "UTC",
    });
  } catch {
    return ts.slice(11, 16);
  }
}

// Custom tooltip
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#111d2e] border border-[#1e2d40] rounded-lg px-3 py-2 text-xs font-mono shadow-xl">
      <div className="text-gray-400 mb-1">{fmtTime(label)}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} style={{ color: p.stroke }} className="flex gap-2">
          <span className="capitalize">{p.dataKey}:</span>
          <span className="font-bold">${Number(p.value).toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export default function EquityAreaChart({ data }: Props) {
  if (!data || data.length < 2) {
    return (
      <div className="flex items-center justify-center h-full text-gray-600 text-xs font-mono">
        Collecting equity data…
      </div>
    );
  }

  const firstBalance = data[0]?.balance ?? 0;
  const lastEquity   = data[data.length - 1]?.equity ?? 0;
  const isUp         = lastEquity >= firstBalance;
  const equityColor  = isUp ? "#00e896" : "#ff4d6a";
  const balColor     = "#4da6ff";

  const minVal = Math.min(...data.map((d) => Math.min(d.balance, d.equity))) * 0.9995;
  const maxVal = Math.max(...data.map((d) => Math.max(d.balance, d.equity))) * 1.0005;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={equityColor} stopOpacity={0.3} />
            <stop offset="95%" stopColor={equityColor} stopOpacity={0.0} />
          </linearGradient>
          <linearGradient id="balanceGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={balColor} stopOpacity={0.2} />
            <stop offset="95%" stopColor={balColor} stopOpacity={0.0} />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="2 4" stroke="#1e2d40" vertical={false} />

        <XAxis
          dataKey="ts"
          tickFormatter={fmtTime}
          tick={{ fill: "#4a5568", fontSize: 10, fontFamily: "monospace" }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
        />

        <YAxis
          domain={[minVal, maxVal]}
          tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
          tick={{ fill: "#4a5568", fontSize: 10, fontFamily: "monospace" }}
          axisLine={false}
          tickLine={false}
          width={60}
        />

        <Tooltip content={<CustomTooltip />} />

        <ReferenceLine
          y={firstBalance}
          stroke="#2e4060"
          strokeDasharray="3 3"
          strokeWidth={1}
        />

        <Area
          type="monotone"
          dataKey="balance"
          stroke={balColor}
          strokeWidth={1.5}
          strokeDasharray="4 2"
          fill="url(#balanceGrad)"
          dot={false}
          activeDot={{ r: 3, fill: balColor }}
        />

        <Area
          type="monotone"
          dataKey="equity"
          stroke={equityColor}
          strokeWidth={2}
          fill="url(#equityGrad)"
          dot={false}
          activeDot={{ r: 4, fill: equityColor }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
