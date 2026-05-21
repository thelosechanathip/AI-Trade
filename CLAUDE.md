# CLAUDE.md

# AI-Trade — ระบบเทรดทองอัตโนมัติ (XAUUSD)

ระบบเทรด XAUUSD (ทองคำ) อัตโนมัติ 100% บน MetaTrader 5
ขับเคลื่อนด้วย AI และ Machine Learning แบบ Local ไม่พึ่งพา Cloud

Engine Version: `v3.0` (Progressive Autonomous AI Trader)

---

## Overview

AI-Trade เป็นระบบ Algorithmic Trading สำหรับ MetaTrader 5 ที่ออกแบบมาเพื่อ:

- เทรดทองคำอัตโนมัติ 24h หรือเฉพาะ London/NY Session
- วิเคราะห์สัญญาณหลายชั้น (Rule Engine + Market Intelligence + AI Brain)
- ใช้ AI + ML + RL + Institutional Market Intelligence
- บริหารความเสี่ยงแบบมืออาชีพ (Kelly + Drawdown Scaling + Adaptive Cooldown)
- Progressive Autonomy: เริ่มจาก Rule-Based แล้วค่อยๆ evolve ไปเป็น Autonomous AI
- มี Dashboard Real-time (port 8001)
- รองรับ Backtest / Optimization / Walk-forward

---

## Core File Map

| File | Role |
|---|---|
| `main.py` | Engine + main loop + `_process_symbol()` |
| `strategy.py` | Rule-based signal generation (8-layer pipeline) |
| `market_intelligence.py` | Institutional MI engine (10 regimes, divergence, BOS/CHOCH) |
| `market_brain.py` | Multi-agent AI decision engine (6 specialist agents) |
| `confidence_bootstrap.py` | Synthetic confidence from technicals (cold-start) |
| `cold_start_manager.py` | Progressive autonomy level manager (L0–L4) |
| `exit_intelligence.py` | Narrative-driven proactive exit system |
| `brain_memory.py` | SQLite long-term memory (`brain_memory.db`) |
| `uncertainty_engine.py` | 5-component uncertainty quantifier |
| `risk.py` | Position sizing, drawdown, adaptive cooldown |
| `trade_manager.py` | Breakeven, trailing stop, partial close, time exit |
| `ai_model.py` | ML ensemble (RF+XGB+LGBM+LSTM+RL+Online SGD) |
| `rl_agent.py` | Double DQN reinforcement learning agent |
| `execution_mt5.py` | All MT5 API interactions |
| `config.yaml` | Master configuration (UTF-8 required) |

---

## Signal Pipeline (v3.0)

New architecture separates the **Rule Engine** (always active) from the **AI Brain** (context modifier):

```
Market Data
    ↓
Strategy Engine (Rule-Based — always primary)
    ↓
Market Intelligence (Regime + Divergence + BOS/CHOCH)
    ↓
AI Context Layer (Brain analyzes, doesn't block at L0-2)
    ↓
Progressive Autonomy Gate (Final Trade Score)
    ↓
Risk Intelligence (Kelly + drawdown + cold-start scaling)
    ↓
Execution Decision
```

---

## Progressive Autonomy System

**CRITICAL**: ระบบไม่ใช้ Full Autonomous AI ตั้งแต่เริ่ม เพราะ AI ยัง cold-start

| Level | Name | Brain Role | Risk Scale |
|---|---|---|---|
| 0 | RULE_BASED | Logs only, no blocking | 60% lot |
| 1 | RULE_PLUS_AI_FILTER | Reduces lot, emergency block only | 70% lot |
| 2 | AI_ASSISTED | Influences sizing via bootstrap conf | 80% lot |
| 3 | SEMI_AUTONOMOUS | Can override strategy signal | 90% lot |
| 4 | FULL_AUTONOMOUS | Primary decision maker | 100% lot |

**Auto-upgrade criteria:**
- L0→L1: 10 trades
- L1→L2: 30 trades, AUC ≥ 0.48, winrate ≥ 40%
- L2→L3: 75 trades, AUC ≥ 0.52, winrate ≥ 45%, DD ≤ 10%
- L3→L4: 150 trades, AUC ≥ 0.55, winrate ≥ 50%, DD ≤ 8%

**Auto-downgrade**: 5+ consecutive losses OR drawdown ≥ 8% → drop one level

---

## Cold Start Handling

During cold-start (L0-1):

- AI confidence is unreliable (model not yet trained on real outcomes)
- Brain NEVER blocks a valid strategy signal via confidence alone
- Only `emergency_block` fires (requires reversal_prob ≥ 0.72 AND uncertainty ≥ 0.78)
- Bootstrap confidence fills the gap (see below)
- Lot size is scaled down (60-70% of normal)

---

## AI Confidence Bootstrap

`confidence_bootstrap.py` computes synthetic confidence from technical signals:

```python
bootstrap_score = (
    htf_alignment     * 0.25 +   # HTF EMA bias
    trend_quality     * 0.20 +   # ADX strength
    momentum_quality  * 0.20 +   # RSI + MACD + Stoch
    liquidity_quality * 0.15 +   # BOS/CHOCH/sweeps from MI
    volatility_quality * 0.10 +  # No climax
    session_quality   * 0.10     # Session + spread OK
) * 100
```

**Final Trade Score** blends bootstrap + AI model:

```python
effective_ai = bootstrap * bootstrap_weight + brain.confidence * (1 - bootstrap_weight)
final_score  = setup_quality * 0.60 + effective_ai * 0.40
# bootstrap_weight: L0=1.0, L1=0.8, L2=0.55, L3=0.30, L4=0.0
```

---

## Market Intelligence Engine

`market_intelligence.py` — 10 regime types, always active regardless of autonomy level:

**Regimes**: TREND_BULL, TREND_BEAR, RANGE, EXPANSION, REVERSAL, ACCUMULATION,
DISTRIBUTION, LIQUIDITY_GRAB, EXHAUSTION, HIGH_VOL

**Signals detected**:
- RSI/MACD divergence (bullish + bearish)
- Displacement candles (body ≥ 2×ATR = institutional)
- Liquidity sweeps (spike + reversal)
- BOS / CHOCH (Break of Structure / Change of Character)
- Volatility climax (ATR > mean + 2σ)

**Output**: `MarketNarrative` dataclass with `block_buy`, `block_sell`, `setup_quality`,
`reversal_detected`, `signals_active`, `bos_choch` dict

---

## AI Brain (Market Brain)

`market_brain.py` — Multi-agent system, 6 specialist agents:

| Agent | Focus | Base Weight |
|---|---|---|
| TrendAgent | EMA/ADX/HTF alignment | 1.00 |
| ReversalAgent | RSI/MACD divergence, CHOCH | 1.00 |
| LiquidityAgent | Sweeps, BOS, order flow | 0.85 |
| VolatilityAgent | ATR regime, expansion | 0.70 |
| MomentumAgent | MACD hist, stoch, RSI slope | 0.90 |
| DecisionAgent | (implicit — weighted vote tally) | — |

Agent weights adapt by regime (TREND regime boosts TrendAgent, etc.)

**BrainDecision fields**: `decision`, `confidence`, `uncertainty`, `setup_quality`,
`market_regime`, `risk_state`, `reversal_probability`, `entry_quality`,
`reasoning[]`, `agent_votes[]`, `hold_reasons[]`, `emergency_block`, `advisory_only`

---

## HOLD Intelligence

System HOLDs when **MULTIPLE** conditions fire simultaneously — not just one:

```python
# At L0: strategy HOLD only (Brain never adds HOLD)
# At L1: Brain needs 4/5 negative conditions
# At L2: Brain needs 3/5 negative conditions
# At L3: Brain needs 2/5 negative conditions
# At L4: Brain needs 1 condition
```

Negative conditions checked:
1. Uncertainty > 0.75 (extreme)
2. Setup quality < 0.25 (terrible)
3. HTF conflict
4. Reversal probability > 0.70
5. Risk state = 'extreme'

---

## Autonomous Exit Intelligence

`exit_intelligence.py` — called every 60s on all open positions:

| Priority | Condition | Action |
|---|---|---|
| Emergency (4) | Reversal confirmed against position | CLOSE |
| Emergency (4) | Dual RSI+MACD divergence against | CLOSE |
| Strong (3) | CHOCH opposite, Brain flipped | CLOSE or REDUCE_50 |
| Moderate (2) | Uncertainty spike + losing | REDUCE_30 or TIGHTEN_SL |
| Light (1) | Extended hold + no progress | TIGHTEN_SL |

---

## SQLite Brain Memory

`brain_memory.py` creates `data/brain_memory.db`:

| Table | Contents |
|---|---|
| `brain_trades` | Per-trade: entry, exit, Brain decision at time of entry |
| `market_snapshots` | Periodic snapshots: regime, ADX, RSI, narrative |
| `narrative_memory` | Regime + outcome pairings |
| `reversal_patterns` | Reversal signal firings + confirmation |
| `failure_patterns` | Signal combos that preceded losses |
| `ai_decisions` | Per-cycle Brain decisions (sampled) |
| `learning_feedback` | Aggregate win/loss stats per regime |

Used by Brain for `memory_adjustment()` (boosts confidence in winning regimes)
and `failure_similarity()` (penalizes setups similar to past losers).

---

## AI Decision Lifecycle

```
Strategy signal (BUY/SELL/HOLD)
    ↓
[L0-2]: Strategy is primary. Brain provides context only.
[L3-4]: Brain can override signal.
    ↓
Emergency block check (ALL levels):
    - fires only if rev_prob ≥ 0.72 AND uncertainty ≥ 0.78
    ↓
Bootstrap + Final score gate:
    - final_score = setup_quality × 0.60 + effective_ai × 0.40
    - below minimum threshold → HOLD
    ↓
Risk scaling: cold_scale × conf_scale × dir_scale × brain_risk_adj
    ↓
Execute → record to brain_memory → self-review after close
```

---

## Dynamic Re-evaluation

Every 60s, `_re_evaluate_positions()` checks all open positions via `ExitIntelligence`.
The `_TerminalNarrative` proxy reconstructs MI signals from the terminal state dict.

---

## Configuration Notes

- `config.yaml` must be opened with `encoding='utf-8'` (Windows cp874 default fails on Thai comments)
- `progressive_autonomy.initial_level: 0` ensures system starts in RULE_BASED mode
- `progressive_autonomy.max_level: 4` (set to 2 for conservative deployment)
- Dashboard on port 8001 (not 8000, conflicts with AI Office app)
- `brain_memory.db` and `autonomy_state.json` are in `data/`

---

## Trend Dominance Protection

Block conditions (in strategy.py):

1. **HTF Conflict + Strong ADX**: H4=SELL but signal=BUY and ADX ≥ 28 → BLOCK
2. **Short Exhaustion**: Price > 2.5% below EMA200 → BLOCK SELL
3. **RSI Recovery**: RSI bounced > 5pts from oversold → BLOCK SELL

---

## HTF Filter (3-Level)

| TF | Indicator | Purpose |
|---|---|---|
| H4 | EMA200 | Mid trend (primary) |
| D1 | EMA50 | Macro trend |
| H1 | EMA50 | Intraday bridge |

Returns `(bias: str, strength: float)` — NEUTRAL if H4 in neutral zone or H4+D1 conflict.
