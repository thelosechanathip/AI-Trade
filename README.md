# AI-Trade — Automated Gold Trading System

ระบบเทรดทองอัตโนมัติ (XAUUSD) บน MetaTrader 5 พร้อม AI และ Dashboard แบบ Real-time

---

## ภาพรวมระบบ

```
MetaTrader 5  ←→  Python Engine  ←→  FastAPI Backend  ←→  Dashboard (port 8000)
                       │
                  AI Model (scikit-learn)
                  Strategy Engine
                  Risk Manager
                  Trade Manager
```

- **เทรดอัตโนมัติ 100%** — เปิด/ปิดออเดอร์ผ่าน MT5 Python API
- **AI ยืนยันสัญญาณ** — RandomForest/GradientBoosting (local, ไม่ใช้ cloud)
- **Dashboard Real-time** — อัพเดทผ่าน WebSocket ทุก 1 วินาที
- **Risk Management เข้มงวด** — หยุดทันทีเมื่อ drawdown เกินกำหนด

---

## Requirements

| Software | Version |
|---|---|
| Python | 3.11+ |
| MetaTrader 5 | 5.0+ (เปิดค้างไว้และ login แล้ว) |
| Windows | 10/11 |

**Python packages:**
```
MetaTrader5, pandas, numpy, scikit-learn, PyYAML
fastapi, uvicorn[standard], joblib
```

---

## การติดตั้ง

```bash
# 1. Clone หรือ download โปรเจค
cd d:/Project/AI-Trade

# 2. สร้าง virtual environment
python -m venv venv
source venv/Scripts/activate   # Git Bash
# หรือ: venv\Scripts\activate  # CMD

# 3. ติดตั้ง dependencies
pip install -r requirements.txt
pip install "uvicorn[standard]"

# 4. เปิด MetaTrader 5 และ login ให้เรียบร้อย
```

---

## การใช้งาน

```bash
# activate venv ก่อนทุกครั้ง
source venv/Scripts/activate

# รันระบบ (เปิดทั้ง trading engine + web dashboard)
python run.py
```

เปิด browser ที่ **http://localhost:8000**

---

## โครงสร้างไฟล์

```
AI-Trade/
├── run.py               ← จุดเริ่มต้น — รันคำสั่งเดียวได้เลย
├── main.py              ← Trading engine หลัก (60s cycle loop)
├── strategy.py          ← สร้างสัญญาณซื้อ/ขาย (confluence scoring)
├── ai_model.py          ← AI model (RandomForest/GradientBoosting)
├── execution_mt5.py     ← MT5 wrapper — เปิด/ปิดออเดอร์
├── trade_manager.py     ← จัดการ position ที่เปิดอยู่ (BE/Trail/Partial)
├── risk.py              ← Risk management (lot size, drawdown, daily loss)
├── utils.py             ← Indicators, DB helpers, logging
├── web_app.py           ← FastAPI backend + WebSocket
├── backtest.py          ← Backtesting engine
├── config.yaml          ← การตั้งค่าทั้งหมด (แก้ที่นี่ที่เดียว)
├── static/
│   └── index.html       ← Dashboard UI
├── dashboard/           ← Next.js dashboard (optional)
├── data/
│   └── trades.db        ← SQLite database
└── logs/
    └── trading.log      ← Log file
```

---

## การตั้งค่า (config.yaml)

### สัญลักษณ์และ Timeframe
```yaml
trading:
  symbols: ["XAUUSD"]   # เทรดเฉพาะทอง
  timeframe: "M15"
  magic_number: 20240101
```

### Risk Management
```yaml
risk:
  risk_per_trade: 0.01      # เสี่ยง 1% ต่อไม้
  max_concurrent_trades: 2  # เปิดพร้อมกันสูงสุด 2 ออเดอร์
  max_daily_loss: 0.03      # หยุดถ้าขาดทุน 3% ต่อวัน
  max_drawdown: 0.10        # หยุดถ้า drawdown ถึง 10%
  min_rr_ratio: 2.0         # RR ขั้นต่ำ 1:2
  atr_sl_multiplier: 2.0    # SL = ATR × 2
  atr_tp_multiplier: 4.0    # TP = ATR × 4
```

### Trade Management
```yaml
trade_management:
  breakeven_r:       1.0    # ย้าย SL มา entry เมื่อกำไร 1R
  trailing_r:        1.5    # เริ่ม trailing stop เมื่อกำไร 1.5R
  trailing_step_r:   0.5    # ระยะ trail = 0.5 × SL เดิม
  partial_close:     true   # ปิดครึ่งหนึ่งเมื่อกำไร 1R
  partial_close_pct: 0.50
  cooldown_minutes:  90     # รอ 90 นาทีก่อนเปิดไม้ใหม่
  min_sl_points:     12.0   # SL ขั้นต่ำ 12 points เสมอ
  max_spread_gold:   15.0   # ข้ามถ้า spread > 15 pips
```

### AI Model
```yaml
ai:
  enabled: true
  min_confidence: 55        # ต้องมั่นใจ ≥ 55% ถึงเทรด
  model_type: "random_forest"
  retrain_interval: 1440    # re-train ทุก 24 ชั่วโมง
```

---

## กลยุทธ์การเทรด

### เงื่อนไขที่ REQUIRED (ต้องผ่านทั้งหมด)
1. **Session Filter** — เทรดเฉพาะ London (07:00–16:00 UTC) และ New York (12:00–21:00 UTC)
2. **Trend Filter** — ราคาต้องอยู่เหนือ EMA50 และ EMA200 (BUY) หรือต่ำกว่าทั้งคู่ (SELL)

### เงื่อนไขที่ SCORED (ต้องผ่าน ≥ 3 จาก 4)
| # | เงื่อนไข | รายละเอียด |
|---|---|---|
| 1 | MACD + RSI Aligned | MACD และ RSI ต้องชี้ทิศทางเดียวกัน |
| 2 | RSI Range | BUY: 45–75 / SELL: 25–55 |
| 3 | Market Structure | HH/HL (uptrend) หรือ LH/LL (downtrend) |
| 4 | Volatility | ATR ≥ 0.5× ค่าเฉลี่ย 20 แท่ง |

### AI Confirmation
- RandomForest predict bias (bullish/bearish) และ confidence %
- ต้อง bias ตรงกับ signal และ confidence ≥ 55%

---

## Active Trade Management

หลังเปิดออเดอร์แล้ว ระบบจัดการอัตโนมัติทุก 60 วินาที:

```
กำไร 1R  →  ปิด 50% ของ position (lock กำไร)
กำไร 1R  →  ย้าย SL มาที่ entry + buffer (break-even)
กำไร 1.5R →  เริ่ม trailing stop ระยะ 0.5× SL เดิม
```

---

## Dashboard

เปิดที่ **http://localhost:8000**

| ส่วน | รายละเอียด |
|---|---|
| Header | สถานะ Engine, เวลา local (UTC+7) |
| Terminal | ราคา live, spread, RSI/MACD/EMA200, AI confidence |
| Activity Log | log การสแกน, สัญญาณ, การเทรด (real-time WebSocket) |
| KPI Cards | Balance, Equity, Drawdown, Win Rate, Profit Factor |
| Equity Curve | กราฟ balance/equity ตามเวลา |
| Open Positions | position ที่เปิดอยู่พร้อม P&L |
| Trade History | ประวัติการเทรดทั้งหมด |

---

## Backtesting

```bash
python backtest.py
```

ผลลัพธ์แสดง: equity curve, win rate, drawdown, รายการเทรดทั้งหมด

---

## การแก้ปัญหาที่พบบ่อย

**`ModuleNotFoundError: No module named 'pandas'`**
```bash
source venv/Scripts/activate   # ต้อง activate venv ก่อนทุกครั้ง
```

**`No supported WebSocket library`**
```bash
pip install "uvicorn[standard]"
```

**`Terminal: Call failed` / MT5 ต่อไม่ติด**
- ตรวจสอบว่า MetaTrader 5 เปิดอยู่และ login แล้ว
- รอให้แถบสถานะใน MT5 แสดงว่าเชื่อมต่อ broker แล้วค่อย run

**ไม่มีสัญญาณเทรดทั้งวัน**
- ดู log: `HOLD: price between EMAs` = ราคาอยู่ระหว่าง EMA50 และ EMA200 รอ breakout
- ดู log: `score=X/4 need=3` = confluence ไม่ครบ ปกติในตลาด sideways
- ดู log: `cooldown active` = รอครบ 90 นาทีหลังไม้ก่อน

---

## Notes

- ระบบนี้ใช้เงินจริง ความเสี่ยงเป็นของผู้ใช้งาน
- ทดสอบด้วย demo account ก่อนเสมอ
- broker: InterStellarFinancial (symbol: XAUUSD.v)
- account currency: USC (US Cents)
