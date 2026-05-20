# CLAUDE.md — AI-Trade Project Guide

**โครงการ:** AI-Trade — ระบบเทรด XAUUSD อัตโนมัติ 100% ด้วย AI + ML (Local)

**ภาษาหลัก:** Python 3.11+ + MetaTrader 5 Python API  
**สไตล์การพัฒนา:** Clean, modular, production-ready, safety-first

---

## ภาพรวมโครงการ

ระบบเทรดทองคำ (XAUUSD) แบบอัตโนมัติเต็มรูปแบบบน MT5 โดยใช้ AI/ML แบบ Local ทั้งหมด ไม่พึ่ง Cloud

**จุดเด่นหลัก:**
- Multi-layer signal + Regime Detection (Trend/Range/HighVol)
- Ensemble AI (XGBoost + LightGBM + RF + LSTM + RL DQN)
- Self-learning loop + Online learning + Market Memory
- Risk Management ที่เข้มงวด (Kelly, Drawdown Scaling, Daily Limit ฯลฯ)
- Real-time Dashboard (FastAPI + WebSocket)
- Backtesting + Walk-forward + Monte Carlo + Optuna

---

## โครงสร้างสำคัญ (สำคัญที่สุดสำหรับ Claude)

```bash
AI-Trade/
├── main.py                 # ← Entry point ของ trading loop (60 วินาที)
├── strategy.py             # Signal generation + Regime + Confluence Scoring
├── ai_model.py             # Ensemble, LSTM, RL, Online SGD, AI Filter Logic
├── execution_mt5.py        # MT5 order placement / modification / close
├── trade_manager.py        # Breakeven, Trailing, Partial close, Time exit
├── risk.py                 # Kelly, Drawdown, Loss streak, Direction ban
├── utils.py                # Indicators, DB, Logging, Helpers
├── web_app.py              # FastAPI + WebSocket สำหรับ Dashboard
├── rl_agent.py
├── market_memory.py
├── backtest.py
├── auto_optimizer.py
├── config.yaml             # ← แก้ไขการตั้งค่าที่นี่เป็นหลัก
├── run.py                  # Launcher
└── start.bat