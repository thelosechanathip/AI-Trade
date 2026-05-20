# AI-Trade — ระบบเทรดทองอัตโนมัติ

ระบบเทรด XAUUSD (ทองคำ) อัตโนมัติ 100% บน MetaTrader 5 ขับเคลื่อนด้วย AI และ Machine Learning แบบ Local ไม่พึ่งพา Cloud

---

## ทำอะไรได้บ้าง

### เทรดอัตโนมัติ
- **เปิด/ปิดออเดอร์ผ่าน MT5 Python API** โดยไม่ต้องกดด้วยมือ
- รองรับ BUY และ SELL ทั้งสองทิศทาง ปรับตามสภาวะตลาดจริง
- วงจรสแกนทุก 60 วินาที ตลอด 24 ชั่วโมง (หรือเฉพาะ London/NY Session)
- **News Filter** — หยุดเทรดอัตโนมัติ 30 นาทีก่อน / 15 นาทีหลัง ข่าว High Impact (ใช้ปฏิทิน MT5 โดยตรง)
- **Cooldown** — เว้นระยะอย่างน้อย 60 นาทีระหว่างไม้บน symbol เดิม ป้องกันการเปิดถี่เกินไป

### วิเคราะห์สัญญาณหลายชั้น
- **Regime Detection** — แยกตลาดเป็น 3 สภาวะด้วย ADX
  - `TREND` (ADX ≥ 25) — ใช้ confluence scoring ตามเทรนด์
  - `RANGE` (ADX < 20) — ใช้ mean-reversion ด้วย Bollinger Bands + Stochastic + RSI
  - `HIGH_VOL` (ADX ≥ 42) — เทรดตามเทรนด์แต่ลดขนาด lot
- **HTF Filter** — กรอง 2 ชั้น: H4 EMA200 + D1 EMA50 ต้องสอดคล้องกันก่อนอนุญาตเทรด
- **Confluence Scoring** — ต้องผ่าน ≥ 3 จาก 5 เงื่อนไข:
  - MACD + RSI ชี้ทิศเดียวกัน
  - RSI อยู่ใน zone ที่เทรดได้
  - Market Structure (HH/HL หรือ LH/LL)
  - Volatility (ATR เหนือค่าเฉลี่ย)
  - MACD Histogram momentum กำลังแข็งขึ้น
- **Entry Momentum Gate** — บล็อก SELL ถ้า 2 แท่งล่าสุดเป็น bullish / บล็อก BUY ถ้า 2 แท่งเป็น bearish (ป้องกันเข้าสวนทาง bounce)

### AI / Machine Learning (Local ทั้งหมด)
- **Ensemble Model** — XGBoost + LightGBM + RandomForest calibrated พร้อม 46 features
- **Regime Sub-model** — แยก model สำหรับแต่ละ regime
- **LSTM** — จดจำ pattern ตามลำดับเวลา
- **Online SGD** — เรียนรู้เพิ่มเติมจากข้อมูลใหม่แบบ incremental ไม่ต้อง retrain ทั้งหมด
- **RL DQN Agent** — Reinforcement Learning ที่รับ reward จาก P&L จริงของทุก trade
- **Market Memory** — Case-based reasoning เปรียบเทียบ pattern ปัจจุบันกับ 500 trade ในอดีต
- **Self-Learning Loop** — ทุกครั้งที่ trade ปิด (SL/TP) ผลลัพธ์จะถูกส่งกลับไปปรับ model ทันที
- **Auto-Retrain** — train โมเดลซ้ำทุก 6 ชั่วโมงด้วยข้อมูล 2,000 แท่งล่าสุด
- **AI Filter Logic** — AI ไม่ได้ "ต้องเห็นด้วย" — AI จะบล็อกเฉพาะตอนที่มั่นใจว่าตลาดจะไปทิศตรงข้าม

### Risk Management
- **Kelly Criterion** — คำนวณ lot size จาก win rate และ avg win/loss จริง (capped 2%)
- **Drawdown Scaling** — ลด position size ตามสัดส่วนเมื่อ drawdown เพิ่มขึ้น
- **Daily Loss Limit** — หยุดเทรดทั้งวันถ้าขาดทุนเกิน 4% ของ balance (reset ตี 0)
- **Max Drawdown** — ปิด engine ถ้า drawdown แตะ 10%
- **Loss Streak Protection** — ลด lot 50% หลังแพ้ 3 ไม้ติดกัน
- **Direction Ban** — แบนทิศทางนั้น 4 ชั่วโมงหลังแพ้ซ้อน 2 ครั้งในทิศเดียวกัน
- **Per-Direction Cap** — เปิดได้สูงสุด 2 ไม้ต่อทิศทาง และจะไม่เพิ่มไม้ถ้าไม้ที่เปิดอยู่ขาดทุนทุกไม้
- **Spread Filter** — ข้ามถ้า spread > 15 pips (gold) หรือ 6 pips (forex)

### Active Trade Management
หลังเปิดออเดอร์แล้ว ระบบดูแลอัตโนมัติทุก 60 วินาที:

| เหตุการณ์ | การกระทำ |
|---|---|
| กำไรถึง 1R | ปิด 30% ของ position, ย้าย SL มา breakeven |
| กำไรถึง 2R | ปิด 30% เพิ่ม |
| กำไรถึง 3R | ปิด 40% ที่เหลือ |
| กำไรถึง 1.5R | เริ่ม trailing stop ตาม ATR |
| เปิดนาน 48 แท่ง (12 ชม.) และกำไร < -0.3R | ปิดออเดอร์ (time exit) |

### Dashboard Real-time
เปิดที่ **http://localhost:8001**

| ส่วน | รายละเอียด |
|---|---|
| KPI Cards | Balance, Equity, Today P&L, Drawdown %, Win Rate, Profit Factor |
| Terminal | ราคา live, regime, ADX, RSI, MACD, EMA200, AI bias + confidence |
| Equity Curve | กราฟ balance/equity ตามเวลา |
| AI Insights | breakdown prediction ของแต่ละโมเดล (Tabular/LSTM/RL/Memory) |
| RL Agent Panel | จำนวน trade outcomes, accuracy, reward สะสม |
| Pattern Memory | จำนวน pattern ที่จดจำ, win rate ใน memory |
| Online Learning | จำนวน update, accuracy ล่าสุด |
| Open Positions | ออเดอร์ที่เปิดอยู่พร้อม P&L real-time |
| Activity Log | log สแกน, สัญญาณ, การเทรด, AI decision แบบ real-time |

### Backtesting และ Optimization
```bash
# Backtest ปกติ
python backtest.py

# Walk-forward validation (6 folds)
python backtest.py --walk-forward

# Monte Carlo simulation (1000 runs)
python backtest.py --monte-carlo 1000
```

ผลลัพธ์: Sharpe ratio, Sortino ratio, Calmar ratio, Max Drawdown, Win Rate, Profit Factor

```bash
# Auto-optimize parameters (Optuna, 40 trials)
python auto_optimizer.py
```

---

## Requirements

| Software | Version |
|---|---|
| Python | 3.11+ |
| MetaTrader 5 | 5.0+ (เปิดและ login แล้ว) |
| Windows | 10 / 11 |

```
MetaTrader5, pandas, numpy, scikit-learn, PyYAML
fastapi, uvicorn[standard], joblib
xgboost>=2.0, lightgbm>=4.0
```

---

## การติดตั้ง

**วิธีที่ 1 — ใช้ installer (แนะนำ)**

ดาวน์โหลด `AI-Trade_Setup.exe` จากโฟลเดอร์ `releases/` แล้วรันได้เลย ติดตั้งใน `%LocalAppData%\Programs\AI-Trade`

**วิธีที่ 2 — Manual**

```bash
# 1. สร้าง virtual environment
python -m venv venv
venv\Scripts\activate

# 2. ติดตั้ง dependencies
pip install -r requirements.txt

# 3. เปิด MetaTrader 5 และ login
# 4. รันระบบ
python run.py
```

หรือดับเบิลคลิก **`start.bat`** เพื่อเปิดระบบพร้อม MT5 check อัตโนมัติ

---

## โครงสร้างไฟล์

```
AI-Trade/
├── run.py               ← จุดเริ่มต้น (รันคำสั่งเดียว)
├── start.bat            ← Launcher สำหรับ Windows
├── main.py              ← Trading engine หลัก (60s cycle)
├── strategy.py          ← Signal generation + regime detection
├── ai_model.py          ← Ensemble AI + RL + Memory + Online learning
├── execution_mt5.py     ← MT5 wrapper — place/close orders
├── trade_manager.py     ← Breakeven / Trailing / Partial close
├── risk.py              ← Kelly sizing + drawdown + loss streak
├── utils.py             ← Indicators, DB helpers, logging
├── web_app.py           ← FastAPI backend + WebSocket
├── backtest.py          ← Backtesting + walk-forward + Monte Carlo
├── auto_optimizer.py    ← Optuna parameter optimization
├── rl_agent.py          ← DQN Reinforcement Learning agent
├── market_memory.py     ← Case-based pattern memory
├── config.yaml          ← การตั้งค่าทั้งหมด (แก้ที่นี่)
├── config_safe.yaml     ← Profile สำหรับ conservative trading
├── config_aggressive.yaml ← Profile สำหรับ aggressive trading
├── static/
│   └── index.html       ← Dashboard UI (standalone HTML)
├── dashboard/           ← Next.js dashboard (optional, port 3000)
├── data/
│   ├── trades.db        ← SQLite (trades, equity curve, activity log)
│   ├── state.json       ← Engine state สำหรับ dashboard
│   ├── ai_insights.json ← AI prediction breakdown
│   └── learning_stats.json ← RL/memory/online learning stats
├── models/
│   ├── ai_ensemble.pkl  ← Trained ensemble model
│   ├── scaler.pkl       ← Feature scaler
│   ├── rl_dqn.pt        ← RL DQN weights
│   └── market_memory.json ← Pattern memory database
└── logs/
    └── trading.log      ← Rotating log (max 50 MB)
```

---

## การตั้งค่าสำคัญ (config.yaml)

```yaml
trading:
  symbols: ["XAUUSD"]       # symbol ที่เทรด
  timeframe: "M15"          # timeframe หลัก
  magic_number: 20240101    # ID ของระบบใน MT5

risk:
  risk_per_trade: 0.008     # เสี่ยง 0.8% ต่อไม้
  max_concurrent_trades: 3  # เปิดพร้อมกันสูงสุด 3 ออเดอร์ (รวมทุกทิศ)
  max_per_direction: 2      # สูงสุด 2 ไม้ต่อทิศทาง
  max_daily_loss: 0.04      # หยุดถ้าขาดทุน 4% ต่อวัน
  max_drawdown: 0.10        # ปิด engine ถ้า drawdown 10%

ai:
  enabled: true
  min_confidence: 55        # AI บล็อกสัญญาณเมื่อมั่นใจ ≥ 55% ว่าตลาดจะสวนทาง

trade_management:
  cooldown_minutes: 60      # รอ 60 นาทีก่อนเปิดไม้ใหม่บน symbol เดิม
  news_filter_enabled: true # หยุด 30 นาทีก่อน / 15 นาทีหลัง ข่าว High Impact

sessions:
  enabled: false            # false = เทรด 24 ชม., true = London+NY เท่านั้น

dashboard:
  port: 8001
```

---

## การแก้ปัญหาที่พบบ่อย

**MT5 ต่อไม่ติด**
- ตรวจสอบว่า MetaTrader 5 เปิดอยู่และ login แล้ว
- รอให้แถบสถานะใน MT5 แสดงว่าเชื่อมต่อ broker แล้วค่อยรัน

**Dashboard ไม่ขึ้น / "Not Found"**
- ตรวจสอบว่า port 8001 ไม่ถูก app อื่นใช้
- เปลี่ยน `dashboard.port` ใน config.yaml ถ้าจำเป็น

**ไม่มีสัญญาณเทรด**
- ดู log: `HOLD: price between EMAs` = รอ EMA crossover
- ดู log: `score=X/5 need=3` = confluence ไม่ครบ (ตลาด sideways)
- ดู log: `Direction ban` = แบนทิศทางนั้นอยู่ (หลังแพ้ 2 ครั้งติด)
- ดู log: `cooldown active` = รอครบ 60 นาที
- ดู log: `News blackout` = อยู่ใน window ของข่าว High Impact

**AI learning stats แสดง 0 ทั้งหมด**
- Restart engine — ระบบจะโหลด open trades จาก DB อัตโนมัติ และเริ่มเรียนรู้จาก trade ที่ปิดถัดไป

**`ModuleNotFoundError`**
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

---

## หมายเหตุ

- ระบบนี้เทรดด้วยเงินจริง ความเสี่ยงเป็นของผู้ใช้งาน
- ทดสอบบน demo account ก่อนเสมอ
- Broker: InterStellarFinancial — symbol: `XAUUSD.v`
- สกุลเงิน account: USC (US Cents)
