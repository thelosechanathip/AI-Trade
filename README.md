# AI-Trade — ระบบเทรดทองอัตโนมัติ

ระบบเทรด XAUUSD (ทองคำ) อัตโนมัติ 100% บน MetaTrader 5 ขับเคลื่อนด้วย AI และ Machine Learning แบบ Local ไม่พึ่งพา Cloud

---

## ทำอะไรได้บ้าง

### เทรดอัตโนมัติ
- **เปิด/ปิดออเดอร์ผ่าน MT5 Python API** โดยไม่ต้องกดด้วยมือ
- รองรับ BUY และ SELL ทั้งสองทิศทาง ปรับตามสภาวะตลาดจริง ไม่ bias ฝั่งใดฝั่งหนึ่ง
- วงจรสแกนทุก 60 วินาที ตลอด 24 ชั่วโมง (หรือเฉพาะ London/NY Session)
- **News Filter** — หยุดเทรดอัตโนมัติ 30 นาทีก่อน / 15 นาทีหลัง ข่าว High Impact (ใช้ปฏิทิน MT5 โดยตรง)
- **Cooldown** — เว้นระยะอย่างน้อย 60 นาทีระหว่างไม้บน symbol เดิม ป้องกันการเปิดถี่เกินไป

---

### วิเคราะห์สัญญาณ — Pipeline (v2.4)

สัญญาณต้องผ่าน **8 ชั้น** เรียงตามลำดับ ชั้นใดล้มเหลวระบบคืน HOLD ทันที:

| ชั้น | ชื่อ | รายละเอียด |
|:---:|---|---|
| 1 | Session Filter | เทรดเฉพาะช่วงเวลาที่กำหนด (London / NY หรือ 24h) |
| 2 | Market Context | สร้าง snapshot: regime, trend direction, strength, exhaustion, RSI state |
| 3 | **Trend Dominance Protection** | Hard block counter-trend ใน 3 กรณี (ดูด้านล่าง) |
| 4 | EMA Trend Filter | HTF-aware: ถ้า H4 ยืนยัน ใช้แค่ EMA50; ถ้าไม่ ต้องการทั้ง EMA50+EMA200 |
| 5 | **Entry Momentum Gate** | บล็อก SELL ถ้า 2 แท่งล่าสุด bullish / บล็อก BUY ถ้า 2 แท่งล่าสุด bearish |
| 6 | **RSI Recovery Gate** | บล็อก SELL ถ้า RSI เด้งจาก oversold ≥5pt / บล็อก BUY ถ้า RSI ตกจาก overbought ≥5pt |
| 7 | Confluence Scoring | ต้องผ่าน ≥ 3 จาก 5 เงื่อนไข (MACD+RSI, RSI zone, Structure, Volatility, MACD momentum) |
| 8 | Final HTF Guard | SELL ถูกบล็อกถ้า H4+D1+H1 บอก BUY และในทางกลับกัน |

#### Trend Dominance Protection — 3 กรณีที่บล็อกทันที
1. **HTF Conflict + Strong ADX** — H4 บอก SELL แต่จะเปิด BUY และ ADX ≥ 28 = บล็อก
2. **Short Exhaustion** — price อยู่ต่ำกว่า EMA200 เกิน 2.5% = shorts exhausted = บล็อก SELL
3. **RSI Recovery** — RSI เพิ่งเด้งขึ้นจาก oversold (<38) มากกว่า 5pt = บล็อก SELL

#### HTF Filter — 3 ชั้น (v2.4)
| ชั้น | Timeframe | Indicator | หน้าที่ |
|---|---|---|---|
| H4 | H4 | EMA200 | กรองเทรนด์กลาง (33 วันซื้อขาย) |
| D1 | D1 | EMA50 | ยืนยัน macro trend direction |
| H1 | H1 | EMA50 | bridge intraday momentum ระหว่าง M15 กับ H4 |

ทั้ง 3 ชั้นคืน `(bias, strength)` — strength 0–1 ส่งต่อไปให้ strategy ใช้ถ่วงน้ำหนักการตัดสินใจ

---

### AI / Machine Learning (Local ทั้งหมด)
- **Ensemble Model** — XGBoost + LightGBM + RandomForest calibrated พร้อม 46 features
- **Regime Sub-model** — แยก model สำหรับแต่ละ regime (TREND/RANGE/HIGH_VOL)
- **LSTM** — จดจำ pattern ตามลำดับเวลา
- **Online SGD** — เรียนรู้เพิ่มเติมแบบ incremental ไม่ต้อง retrain ทั้งหมด
- **RL DQN Agent** — Reinforcement Learning รับ reward จาก P&L จริงของทุก trade
- **Market Memory** — Case-based reasoning เปรียบเทียบกับ 500 trade ในอดีต
- **Self-Learning Loop** — ทุกครั้งที่ trade ปิด (SL/TP) ผลลัพธ์ส่งกลับปรับ model ทันที
- **Session Restore** — เมื่อ engine restart จะโหลด open trades จาก DB อัตโนมัติ ไม่สูญเสีย feedback
- **Auto-Retrain** — train ซ้ำทุก 6 ชั่วโมงด้วยข้อมูล 2,000 แท่งล่าสุด
- **AI Filter Logic** — AI บล็อกเฉพาะเมื่อมั่นใจ ≥55% ว่าตลาดจะไปทิศตรงข้ามสัญญาณ

---

### Risk Management
- **Kelly Criterion** — คำนวณ lot size จาก win rate และ avg win/loss จริง (capped 2%)
- **Drawdown Scaling** — ลด position size ตามสัดส่วนเมื่อ drawdown เพิ่มขึ้น
- **Daily Loss Limit** — หยุดเทรดทั้งวันถ้าขาดทุนเกิน 4% (reset ตี 0)
- **Max Drawdown** — ปิด engine ถ้า drawdown แตะ 10%
- **Loss Streak Protection** — ลด lot 50% หลังแพ้ 3 ไม้ติดกัน
- **Progressive Direction Ban** — ระบบใหม่ v2.4:
  - แพ้ 1 ครั้ง → soft ban 2h + ลด lot 50% ทิศนั้น
  - แพ้ 2 ครั้งติดกัน → hard ban 4h ทิศนั้น
  - ชนะ → ยกเลิก ban ทันที reset streak
- **Per-Direction Cap** — เปิดได้สูงสุด 2 ไม้ต่อทิศทาง
- **No-Stack-Into-Loss** — ไม่เพิ่มไม้ถ้าไม้ที่เปิดอยู่ทิศเดียวกันขาดทุนทุกไม้
- **Direction Lot Scale** — ลด lot อัตโนมัติตาม losing streak ของทิศนั้น
- **Spread Filter** — ข้ามถ้า spread > 15 pips (gold) หรือ 6 pips (forex)

---

### Active Trade Management
หลังเปิดออเดอร์ ระบบดูแลอัตโนมัติทุก 60 วินาที:

| เหตุการณ์ | การกระทำ |
|---|---|
| กำไรถึง 1R | ปิด 30%, ย้าย SL มา breakeven |
| กำไรถึง 2R | ปิด 30% เพิ่ม |
| กำไรถึง 3R | ปิด 40% ที่เหลือ |
| กำไรถึง 1.5R | เริ่ม trailing stop ตาม ATR |
| เปิดนาน 48 แท่ง และกำไร < -0.3R | Time exit — ปิดออเดอร์ |

---

### Dashboard Real-time
เปิดที่ **http://localhost:8001**

| ส่วน | รายละเอียด |
|---|---|
| KPI Cards | Balance, Equity, Today P&L, Drawdown %, Win Rate, Profit Factor |
| Terminal | ราคา live, regime, ADX, RSI, MACD, EMA200, HTF bias+strength, AI confidence |
| Equity Curve | กราฟ balance/equity ตามเวลา |
| AI Insights | breakdown prediction แต่ละโมเดล (Tabular/LSTM/RL/Memory) |
| RL Agent Panel | trade outcomes, accuracy, reward สะสม |
| Pattern Memory | pattern ที่จดจำ, win rate ใน memory |
| Online Learning | จำนวน update, accuracy ล่าสุด |
| Open Positions | ออเดอร์ที่เปิดอยู่พร้อม P&L real-time |
| Activity Log | log สแกน, สัญญาณ, การเทรด, AI decision, direction ban แบบ real-time |

---

### Backtesting และ Optimization
```bash
# Backtest ปกติ
python backtest.py

# Walk-forward validation (6 folds)
python backtest.py --walk-forward

# Monte Carlo simulation (1000 runs)
python backtest.py --monte-carlo 1000

# Auto-optimize parameters ด้วย Optuna (40 trials)
python auto_optimizer.py
```

ผลลัพธ์: Sharpe ratio, Sortino ratio, Calmar ratio, Max Drawdown, Win Rate, Profit Factor

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

ดาวน์โหลด `AI-Trade_Setup.exe` จากโฟลเดอร์ `releases/` แล้วรันได้เลย
ติดตั้งใน `%LocalAppData%\Programs\AI-Trade` (ไม่ต้อง admin)

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
├── run.py                  ← จุดเริ่มต้น (รันคำสั่งเดียว)
├── start.bat               ← Launcher สำหรับ Windows
├── main.py                 ← Trading engine หลัก (60s cycle)
│                              - _restore_open_trades()  ← restore session หลัง restart
│                              - _fetch_htf_bias()       ← H4+D1+H1 three-level filter
│                              - _update_direction_streak() ← progressive direction ban
├── strategy.py             ← Signal generation (v2.4 pipeline)
│                              - build_market_context()  ← MarketContext dataclass
│                              - _trend_dominance_blocked() ← 3-layer hard block
│                              - generate_signal()       ← 8-stage pipeline
├── ai_model.py             ← Ensemble AI + RL + Memory + Online learning
├── execution_mt5.py        ← MT5 wrapper — place/close orders
├── trade_manager.py        ← Breakeven / Trailing / Partial close / Time exit
├── risk.py                 ← Kelly sizing + drawdown + loss streak
├── utils.py                ← Indicators, DB helpers, logging
│                              - get_open_trades_from_db() ← session restore support
├── web_app.py              ← FastAPI backend + WebSocket
├── backtest.py             ← Backtesting + walk-forward + Monte Carlo
├── auto_optimizer.py       ← Optuna parameter optimization
├── rl_agent.py             ← DQN Reinforcement Learning agent
├── market_memory.py        ← Case-based pattern memory
├── config.yaml             ← การตั้งค่าทั้งหมด (แก้ที่นี่)
├── config_safe.yaml        ← Profile สำหรับ conservative trading
├── config_aggressive.yaml  ← Profile สำหรับ aggressive trading
├── static/
│   └── index.html          ← Dashboard UI (standalone HTML)
├── dashboard/              ← Next.js dashboard (optional, port 3000)
├── data/
│   ├── trades.db           ← SQLite (trades, equity curve, activity log)
│   ├── state.json          ← Engine state สำหรับ dashboard
│   ├── ai_insights.json    ← AI prediction breakdown
│   └── learning_stats.json ← RL/memory/online learning stats
├── models/
│   ├── ai_ensemble.pkl     ← Trained ensemble model
│   ├── scaler.pkl          ← Feature scaler
│   ├── rl_dqn.pt           ← RL DQN weights
│   └── market_memory.json  ← Pattern memory database
└── logs/
    └── trading.log         ← Rotating log (max 50 MB)
```

---

## การตั้งค่าสำคัญ (config.yaml)

```yaml
trading:
  symbols: ["XAUUSD"]
  timeframe: "M15"
  magic_number: 20240101

risk:
  risk_per_trade: 0.008     # เสี่ยง 0.8% ต่อไม้
  max_concurrent_trades: 3  # สูงสุด 3 ออเดอร์พร้อมกัน (รวมทุกทิศ)
  max_per_direction: 2      # สูงสุด 2 ไม้ต่อทิศทาง (BUY หรือ SELL)
  max_daily_loss: 0.04      # หยุดถ้าขาดทุน 4% ต่อวัน
  max_drawdown: 0.10        # ปิด engine ถ้า drawdown 10%

ai:
  enabled: true
  min_confidence: 55        # AI บล็อกเมื่อมั่นใจ ≥55% ว่าตลาดจะสวนทาง

trade_management:
  cooldown_minutes: 60
  news_filter_enabled: true

sessions:
  enabled: false            # false = 24h, true = London+NY เท่านั้น

htf_filter:
  enabled: true
  neutral_zone: 0.005       # buffer 0.5% รอบ H4 EMA200

direction_ban:
  enabled: true
  max_same_dir_losses: 2    # hard ban หลังแพ้ 2 ครั้งติดกัน
  ban_hours: 4              # hard ban 4h
  ban_hours_soft: 2         # soft ban 2h (หลังแพ้ 1 ครั้ง)

trend_dominance:
  enabled: true
  adx_strong_threshold: 28        # ADX ≥ 28 = เทรนด์แข็ง
  exhaustion_ema200_pct: 0.025    # ห่าง EMA200 > 2.5% = exhausted
  rsi_recovery_level: 38          # RSI ต่ำกว่า 38 แล้วเด้งขึ้น = บล็อก SELL
  rsi_rejection_level: 62         # RSI สูงกว่า 62 แล้วตกลง = บล็อก BUY

dashboard:
  port: 8001
```

---

## การอ่าน Log เพื่อทำความเข้าใจพฤติกรรมระบบ

| Log message | ความหมาย |
|---|---|
| `HOLD: price between EMAs` | ราคาอยู่ระหว่าง EMA50/200 รอ breakout |
| `Trend dominance blocked SELL: Short exhaustion` | price ลงไปไกลเกินไปแล้ว ระบบป้องกันไม่ให้เพิ่ม SELL |
| `Momentum gate: blocked SELL — 2 consecutive bullish bars` | gold กำลัง bounce ห้าม SELL |
| `RSI recovery gate: blocked SELL` | RSI เด้งจาก oversold สัญญาณ short exhaustion |
| `SOFT BAN SELL 2h` | แพ้ 1 ครั้ง → lot ลด 50% และรอ 2h |
| `HARD BAN SELL 4h` | แพ้ 2 ครั้งติด → ห้าม SELL 4h |
| `direction ban lifted` | ชนะ 1 ครั้ง → ยกเลิก ban ทันที |
| `HTF=BUY(0.85)` | H4+D1+H1 เห็นตรงกัน strength 85% |
| `score=3/5 need=3` | ผ่าน confluence พอดี |
| `News blackout` | อยู่ใน window ข่าว High Impact |
| `cooldown active — X min remaining` | รอครบ cooldown |

---

## การแก้ปัญหาที่พบบ่อย

**MT5 ต่อไม่ติด**
- ตรวจสอบว่า MetaTrader 5 เปิดอยู่และ login แล้ว
- รอให้แถบสถานะใน MT5 แสดงว่าเชื่อมต่อ broker แล้วค่อยรัน

**Dashboard ไม่ขึ้น / "Not Found"**
- ตรวจสอบว่า port 8001 ไม่ถูก app อื่นใช้
- เปลี่ยน `dashboard.port` ใน config.yaml ถ้าจำเป็น

**ไม่มีสัญญาณเทรดเลย**
- ดู log: `Trend dominance blocked` = market มีเทรนด์แรงแต่สัญญาณสวน — ถูกต้องแล้ว
- ดู log: `Momentum gate` = 2 แท่งล่าสุดสวนทิศ — รอจังหวะดีกว่า
- ดู log: `score=X/5 need=3` = confluence ไม่ครบ ตลาด sideways
- ดู log: `Direction ban` = แบนทิศนั้นอยู่ หลังแพ้ติดกัน
- ดู log: `cooldown active` = รอครบ 60 นาที
- ดู log: `News blackout` = อยู่ใน window ข่าว

**AI learning stats (RL/Memory) แสดง 0 หลัง restart**
- ปกติ — ระบบ restore open trades จาก DB แล้ว จะเริ่มนับเมื่อ trade ถัดไปปิด
- ถ้าแสดง 0 นานผิดปกติ: ดู log หา `Trade X labeled`

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
- Engine version: v2.4
