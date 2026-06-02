# AI-Trade v3.1 — Stable Autonomous XAUUSD Trading System

Fully automated gold (XAUUSD) trading on MetaTrader 5, powered by local AI/ML with no cloud dependency.

---

## Quick Start

```bash
# 1. Install dependencies
pip install MetaTrader5 pandas numpy scikit-learn xgboost lightgbm pyyaml flask

# 2. Open MetaTrader 5 and log in to your account

# 3. Start the engine
python main.py

# 4. Open the dashboard (separate terminal)
python dashboard.py
# Dashboard: http://localhost:8001
```

---

## Architecture Overview

```
Market Data (MT5 API)
        |
  Strategy Engine          <- Rule-based signal generation (always primary)
        |
  Market Intelligence      <- Regime, divergence, BOS/CHOCH, sweeps
        |
  AI Context Layer         <- Brain analyzes risk (context advisor at L0-2)
        |
  Progressive Autonomy     <- Final Trade Score gate (setup x60% + AI x40%)
        |
  Risk Intelligence        <- Kelly sizing + DD scaling + cold-start reduction
        |
  Committee Guard          <- Director/Quant/Risk/Execution final review
        |
     Execute               <- MT5 order placement
        |
  Exit Intelligence        <- Re-evaluates open positions every 60s
```

---

## AI Autonomy Levels

The system evolves from rule-based to fully autonomous as it accumulates data:

| Level | Name | Brain Role | Lot Scale |
|---|---|---|---|
| **0** | RULE_BASED | Logs + emergency block only | 60% |
| **1** | RULE_PLUS_AI_FILTER | Can reduce lot; blocks on extremes | 70% |
| **2** | AI_ASSISTED | Influences sizing via bootstrap confidence | 80% |
| **3** | SEMI_AUTONOMOUS | Can override strategy signal | 90% |
| **4** | FULL_AUTONOMOUS | Primary decision maker | 100% |

**Auto-upgrade** (checked every cycle):

| Transition | Criteria |
|---|---|
| L0 to L1 | 10 closed trades |
| L1 to L2 | 30 trades, AUC >= 0.48, win rate >= 40% |
| L2 to L3 | 75 trades, AUC >= 0.52, win rate >= 45%, DD <= 10% |
| L3 to L4 | 150 trades, AUC >= 0.55, win rate >= 50%, DD <= 8% |

**Auto-downgrade**: 5+ consecutive losses or drawdown >= 8% drops one level.

---

## Cold Start Behavior

**Why AI confidence is low at startup:**

The system uses an ensemble ML model (Random Forest + XGBoost + LightGBM + LSTM + RL DQN).
When first started there are no closed trade outcomes to learn from, so:
- ML models may predict at ~50% accuracy (insufficient data)
- RL agent contributes 0% until 50+ trade outcomes
- Online SGD contributes 0% until 30+ updates

**How the system handles this (no HOLD deadlock):**

1. Starts at **L0 (RULE_BASED)** — strategy signal is primary, AI is purely advisory
2. **Bootstrap Confidence** provides synthetic AI confidence from technical signals:
   - HTF alignment (25%), trend quality/ADX (20%), momentum (20%),
     liquidity/BOS/sweeps (15%), volatility (10%), session/spread (10%)
3. **Final Trade Score** = `setup_quality x 0.60 + bootstrap_conf x 0.40` at L0
4. Brain confidence threshold at L0 = **0.0** (never blocks by confidence alone)
5. Emergency block fires only when `reversal_prob >= 0.72 AND uncertainty >= 0.78`

**Result**: System trades conservatively (60% lot size) during cold start without any deadlock.

---

## AI Decision Flow

Each signal cycle (simplified):

```python
signal = generate_signal(df, config)       # Rule-based strategy (primary)

brain_decision = market_brain.decide(ctx, autonomy_level=level)

if brain_decision.emergency_block:         # Extreme reversal + uncertainty only
    HOLD

if level >= 3:                             # L3-4: Brain overrides strategy
    signal = brain_decision.decision

if signal == 'HOLD':                       # Strategy HOLD always respected
    return

# Bootstrap + Final Trade Score
final_score = setup_quality * 0.60 + effective_ai * 0.40
if final_score < min_threshold:            # Level-dependent minimum
    HOLD

# Lot sizing
lot_size = kelly_lot * cold_scale * conf_scale * brain_risk_adj * dir_scale
execute()
```

---

## Market Intelligence Engine

Runs every cycle and classifies market into 10 regimes:

| Regime | Description |
|---|---|
| TREND_BULL / TREND_BEAR | Clean trend, strong ADX, aligned EMAs |
| RANGE | Low ADX, oscillating price |
| EXPANSION | Volatility expansion, ADX rising |
| REVERSAL | BOS/CHOCH + divergence confirmed |
| ACCUMULATION | Sideways + bullish divergence |
| DISTRIBUTION | Sideways + bearish divergence |
| LIQUIDITY_GRAB | Stop hunt + sharp reversal |
| EXHAUSTION | Overextended from EMA200 |
| HIGH_VOL | ATR > mean + 2x standard deviation |

Detected signals: RSI divergence, MACD divergence, displacement candles,
liquidity sweeps, BOS (Break of Structure), CHOCH (Change of Character),
volatility climax.

---

## Multi-Agent Brain

6 specialist agents vote BUY/SELL/HOLD with weighted confidence:

- **TrendAgent** — EMA + ADX + HTF alignment
- **ReversalAgent** — Divergence + CHOCH
- **LiquidityAgent** — BOS + sweeps + structure
- **VolatilityAgent** — ATR regime + displacement
- **MomentumAgent** — MACD histogram + RSI + stoch

Weights adapt by regime (trending regime gives TrendAgent 1.30x weight).
Brain Memory adjusts confidence based on historical win rates per regime.

---

## AI Exit Intelligence

Every 60 seconds, exit_intelligence.py evaluates all open positions:

| Priority | Trigger | Action |
|---|---|---|
| Emergency | Reversal confirmed against position | CLOSE |
| Emergency | Dual RSI+MACD divergence against | CLOSE |
| Emergency | Liquidity sweep against + losing | CLOSE |
| Strong | CHOCH in opposite direction | CLOSE or REDUCE 50% |
| Strong | Brain flipped direction | CLOSE or REDUCE 50% |
| Moderate | High uncertainty + losing | REDUCE 30% |
| Moderate | RSI divergence against + losing | TIGHTEN SL |
| Light | 36+ bars held with no progress | TIGHTEN SL |

---

## SQLite Brain Memory

`data/brain_memory.db` stores long-term memory:

| Table | Purpose |
|---|---|
| brain_trades | Every trade: entry brain decision + outcome |
| market_snapshots | Per-cycle market state snapshots |
| narrative_memory | Regime + setup to outcome pairings |
| reversal_patterns | Reversal signals + confirmation tracking |
| failure_patterns | Signal combos that preceded losses |
| ai_decisions | Per-cycle Brain decisions (sampled) |
| learning_feedback | Win/loss stats per regime (confidence adj.) |

---

## Progressive Learning

| Component | Warm-up Required | Active After |
|---|---|---|
| Bootstrap Confidence | None (technical only) | Immediately |
| ML Ensemble (RF/XGB/LGBM) | First retrain (6h interval) | First retrain |
| Online SGD | 30+ updates | ~30 trades |
| RL DQN Agent | 50+ outcomes | ~50 trades |
| Brain Memory (regime adj.) | 5+ trades per regime | ~5 trades |
| Failure Pattern Penalty | Any recorded failure | First loss |

---

## Risk Management

- **Kelly Criterion** (quarter-Kelly, capped at 2%) for position sizing
- **Drawdown Scaling**: linear reduction as drawdown grows
- **Loss Streak Protection**: 50% size after consecutive losses
- **Adaptive Global Cooldown**: 1 loss=2h, 2=4h, 3=12h, 4+=halt
- **Direction Ban**: soft (50% size) after 1 same-dir loss, hard ban after 2
- **Cold-Start Scale**: 60-100% based on autonomy level
- **Confidence Scale**: 45-100% based on final trade score

---

## Configuration

Key settings in `config.yaml` (open with `encoding='utf-8'`):

```yaml
progressive_autonomy:
  enabled: true
  initial_level: 0     # Start RULE_BASED (recommended)
  max_level: 4         # Set 2 for conservative mode

risk:
  risk_per_trade: 0.008
  max_drawdown: 0.10
  adaptive_cooldown:
    enabled: true

strategy:
  min_confluence: 3    # Min 3/5 signals required
```

---

## Context Stability Engine (v3.1)

Four layers prevent "jittery" AI behavior — single-candle reactions, chasing exhausted moves, and rapid bias flips:

### Signal Persistence

Requires **2 consecutive cycles** of the same direction before entry is allowed. A single BUY signal followed by SELL resets the counter. Eliminates one-bar wonder trades.

### Noise Filter

Blocks entries on low-quality candles:

| Check | Trigger | Score |
|---|---|---|
| Doji | body/range < 18% | 1.0 |
| Weak body | body/range < 30% | 0.6 |
| Micro candle | body < 0.15×ATR | 1.0 |
| Parabolic | 4+ large same-dir candles | 1.0 |
| High spread | spread > 20% of ATR | proportional |

Composite score ≥ 0.60 → entry blocked.

### Anti-Chase Engine

Detects exhausted moves before entry:

| Check | Trigger | Weight |
|---|---|---|
| Overextension | price > 3×ATR from EMA200 | 35% |
| RSI extreme | BUY with RSI > 75 / SELL with RSI < 25 | 25% |
| Momentum stretch | 3+ large candles ≥ 1.5×ATR same direction | 25% |
| Vol climax | ATR > mean + 2.5×std | 15% |

Score ≥ 0.60 → entry blocked.

### Bias Stability (Context Persistence)

Maintains a slow-decaying directional bias per symbol:
- EMA alpha = 0.20 (very slow adaptation)
- Bias flip requires: held ≥ 3 cycles **AND** evidence score ≥ 0.72
- Signal opposing stable bias → entry skipped (bias not yet confirmed)
- Long-held aligned bias: up to +0.05 bonus to final_score

### Setup Grading

All checks passed → setup graded by adjusted_score:

| Grade | Score | Lot Scale |
|---|---|---|
| A+ | ≥ 70% | 100% |
| A  | 58–70% | 80% |
| B  | 45–58% | 55% |
| C  | < 45% | skip |

---

## File Structure

```
AI-Trade/
|-- main.py                    # Engine entry point
|-- strategy.py                # Rule-based signal pipeline
|-- market_intelligence.py     # MI engine (10 regimes)
|-- market_brain.py            # Multi-agent decision engine
|-- confidence_bootstrap.py    # Synthetic confidence (cold-start)
|-- cold_start_manager.py      # Progressive autonomy levels
|-- signal_stability.py        # Signal persistence filter (2+ cycles)
|-- noise_filter.py            # Noise rejection (doji/micro/parabolic)
|-- anti_chase.py              # Anti-chase engine (overextension guard)
|-- context_persistence.py     # Slow-decaying directional bias
|-- committee_guard.py         # Final Director/Quant/Risk/Execution gate
|-- exit_intelligence.py       # Proactive exit system
|-- brain_memory.py            # SQLite long-term memory
|-- uncertainty_engine.py      # 5-component uncertainty scorer
|-- ai_model.py                # ML ensemble
|-- rl_agent.py                # DQN reinforcement learning
|-- risk.py                    # Position sizing + risk checks
|-- trade_manager.py           # Breakeven/trailing/partial close
|-- execution_mt5.py           # MT5 API wrapper
|-- backtest.py                # Historical backtester
|-- config.yaml                # Master configuration
|-- data/
|   |-- trading.db             # Trade history (SQLite)
|   |-- brain_memory.db        # Brain long-term memory
|   |-- autonomy_state.json    # Current autonomy level
|   `-- risk_state.json        # Risk manager state
`-- logs/
    `-- trading.log
```
