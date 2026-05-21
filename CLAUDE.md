# CLAUDE.md

# AI-Trade — ระบบเทรดทองอัตโนมัติ (XAUUSD)

ระบบเทรด XAUUSD (ทองคำ) อัตโนมัติ 100% บน MetaTrader 5  
ขับเคลื่อนด้วย AI และ Machine Learning แบบ Local ไม่พึ่งพา Cloud

---

## Overview

AI-Trade เป็นระบบ Algorithmic Trading สำหรับ MetaTrader 5 ที่ออกแบบมาเพื่อ:

- เทรดทองคำอัตโนมัติ
- วิเคราะห์สัญญาณหลายชั้น
- ใช้ AI + Machine Learning + Reinforcement Learning
- บริหารความเสี่ยงแบบมืออาชีพ
- มี Dashboard Real-time
- รองรับ Backtest / Optimization / Walk-forward

Engine Version: `v2.4`

---

# Core Features

## Automated Trading

- เปิด/ปิดออเดอร์ผ่าน MT5 Python API
- รองรับ BUY / SELL ทั้งสองทิศทาง
- Scan ตลาดทุก 60 วินาที
- รองรับ 24h หรือเฉพาะ London / NY Session
- News Filter:
  - หยุดเทรดก่อนข่าวแรง 30 นาที
  - กลับมาเทรดหลังข่าว 15 นาที
- Cooldown:
  - เว้นระยะ 60 นาทีต่อ symbol

---

# Signal Pipeline (v2.4)

ระบบใช้ Pipeline วิเคราะห์สัญญาณ 8 ชั้น

หากชั้นใดไม่ผ่าน → HOLD ทันที

| Layer | Name | Description |
|---|---|---|
| 1 | Session Filter | กรองช่วงเวลาเทรด |
| 2 | Market Context | วิเคราะห์ regime / trend / RSI / exhaustion |
| 3 | Trend Dominance Protection | ป้องกันสวนเทรนด์ |
| 4 | EMA Trend Filter | HTF-aware EMA filtering |
| 5 | Entry Momentum Gate | ตรวจ momentum ล่าสุด |
| 6 | RSI Recovery Gate | ป้องกัน exhaustion |
| 7 | Confluence Scoring | ต้องผ่าน ≥3/5 |
| 8 | Final HTF Guard | ตรวจ bias หลาย timeframe |

---

# Trend Dominance Protection

ระบบ block ทันทีใน 3 กรณี:

## 1. HTF Conflict + Strong ADX

- H4 บอก SELL แต่ signal BUY
- และ ADX ≥ 28

→ BLOCK

---

## 2. Short Exhaustion

- ราคาอยู่ต่ำกว่า EMA200 มากกว่า 2.5%

→ BLOCK SELL

---

## 3. RSI Recovery

- RSI เด้งจาก oversold มากกว่า 5 จุด

→ BLOCK SELL

---

# HTF Filter

| TF | Indicator | Purpose |
|---|---|---|
| H4 | EMA200 | Mid trend |
| D1 | EMA50 | Macro trend |
| H1 | EMA50 | Intraday bridge |

ทุก layer คืนค่า:

```python
(bias, strength)