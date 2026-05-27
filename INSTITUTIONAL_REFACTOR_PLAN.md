# Institutional Refactor Plan

This system must behave like a defensive execution engine, not a retail momentum bot. The priority stack is survival, stability, consistency, capital preservation, then profit.

## Root Cause Analysis

The live losses are consistent with a permissive decision stack:

- Risk is too elastic: Kelly sizing is enabled, drawdown scaling starts late, max concurrent trades was 3, and session filtering was disabled.
- Regime logic is not restrictive enough: `strategy.detect_regime()` is ADX-dominant and still permits range mean reversion. A survival-first XAUUSD system should block RANGE, HIGH_VOL, EXHAUSTION, MIXED_TREND, LIQUIDITY_GRAB, and REVERSAL_UNCONFIRMED for new entries.
- AI confidence can become fake confidence: early autonomy upgrades are allowed at 10/30 trades, AI can influence before confidence calibration is statistically meaningful, and RL/online/LSTM components can shape probability before enough live outcomes exist.
- Anti-chase is helpful but too soft: current overextension defaults use 3 ATR from EMA and score-based blocking. Institutional behavior should hard-block displacement entries, liquidity sweeps without retest, 3 impulse candles, RSI/MACD climax, and price more than 2 ATR from EMA200.
- Entry framework is confluence-based but not A+ only: it accepts enough partial conditions rather than requiring all critical conditions.
- Exit intelligence is reactive: it evaluates narrative shifts, but position lifecycle should include time stop, momentum decay, volatility contraction, regime flip, liquidity failure, partial TP, adaptive breakeven, and trailing stop rules.

## Immediate Disable List

Disable or cap immediately:

- Kelly sizing: disabled until 250+ closed trades and stable calibration.
- RL directional influence: disabled before 100 closed trades, Sharpe greater than 0.5, winrate greater than 48%, max DD less than 8%.
- SGD online directional influence: disabled before 100 closed trades.
- LSTM confidence: ignored before 150 closed trades and walk-forward validation.
- Brain override: no override before autonomy L3, and L3 should require 100+ trades plus calibration validation.
- Range mean-reversion entries: disabled for live XAUUSD until separately validated.
- Breakout chasing: disabled unless retest confirmation exists.

## Production Architecture

Decision flow:

Market Data
-> Feature Integrity Check
-> Hard Risk Lock
-> Session/Spread/News Gate
-> Regime Gate
-> Microstructure Gate
-> Anti-Chase Gate
-> A+ Setup Score
-> Confidence Calibration
-> Risk Sizing
-> Execution Guard
-> Exit Intelligence
-> Post-Trade Learning

No downstream module may override an upstream hard block.

## Hard Risk Lock

Required live gates:

- Daily kill switch: stop new trades at min(-3R, -4% equity from day start).
- Weekly SAFE_MODE: activate at 8% weekly drawdown.
- Monthly emergency: disable all AI influence at 12% monthly drawdown.
- Consecutive losses: 3 losses halt 12h, 5 losses halt 24h.
- Max open positions: 1.
- Cooldown: 90 minutes normal, 180 minutes SAFE_MODE.
- Max trades/day: 3 normal, 1 SAFE_MODE.
- Session: London open and New York open only.
- Spread: block if `spread / ATR > 0.08` normal, `> 0.05` SAFE_MODE.
- Equity curve: if 20-trade equity slope is negative, risk multiplier <= 0.50.

Risk formula:

```text
base_risk = 0.003 to 0.005
dd_mult = clip(1 - current_dd / 0.08, 0.20, 1.00)
loss_mult = {0:1.0, 1:0.65, 2:0.45, 3:0.0}
vol_mult = 0.50 if ATR percentile > 85 or < 20 else 1.00
uncertainty_mult = 1 - 0.70 * uncertainty
confidence_mult = clip((calibrated_conf - 0.55) / 0.25, 0.25, 1.00)
safe_mult = 0.50 if SAFE_MODE else 1.00
risk = base_risk * dd_mult * loss_mult * vol_mult * uncertainty_mult * confidence_mult * safe_mult
```

Hard cap: `risk <= 0.005`, cold-start cap `risk <= 0.003`.

## Confidence Engine

At autonomy L0-L2, AI may only reduce confidence. It cannot increase entry confidence.

Cold-start blend:

```text
if closed_trades < 100:
    ai_weight = min(0.10, closed_trades / 1000)
    rl_weight = 0
    sgd_weight = 0
    lstm_weight = 0
    ensemble_weight <= 0.10
else:
    ai_weight = gated by calibration, Sharpe, winrate, max DD

final_conf = min(bootstrap_conf, bootstrap_conf * (1 - ai_disagreement_penalty))
```

Calibration tracking:

```text
bucket predicted confidence into 50-60, 60-70, 70-80, 80-90, 90-100
expected = bucket midpoint
actual = realized winrate or positive-R rate
calibration_error = abs(expected - actual)
if rolling_error > 0.15 or Brier score deteriorates:
    ai_trust *= 0.50
if error persists 30 trades:
    ai_trust = 0
```

Upgrade requires all:

- 100+ closed trades.
- Sharpe > 0.5.
- Winrate > 48%.
- Max drawdown < 8%.
- Calibration error < 10%.
- Brier score improving over last 50 trades.

## Regime Filter

Trade only when all are true:

- ADX >= 28 and rising over 3 bars.
- EMA50 and EMA200 aligned with signal.
- EMA slope aligned on M15 and H1.
- H4 bias aligned or neutral with high H1 confirmation.
- ATR percentile between 25 and 80.
- CHOP index < 45.
- Trend efficiency ratio > 0.35.
- Structural integrity > 0.70.
- No volatility climax.
- No compression uncertainty.
- No exhaustion.
- No mixed bull/bear structure.

Block regimes:

- RANGE
- HIGH_VOL
- EXHAUSTION
- MIXED_TREND
- REVERSAL_UNCONFIRMED
- NEWS_VOLATILITY
- SESSION_TRANSITION
- LIQUIDITY_GRAB without retest

Regime score:

```text
regime_score =
  0.20 * adx_quality +
  0.20 * ema_slope_alignment +
  0.20 * htf_alignment +
  0.15 * atr_health +
  0.15 * trend_efficiency +
  0.10 * structural_integrity

trade only if regime_score >= 0.78 and no hard block
```

## Anti-Chase

Hard block if any:

- abs(price - EMA200) / ATR > 2.0.
- 3 same-direction impulse candles printed.
- Last candle body > 1.8 ATR.
- RSI > 70 for BUY or < 30 for SELL.
- MACD histogram at 90th percentile extension in trade direction.
- ATR z-score > 2.0.
- Entry immediately after displacement candle.
- Entry into prior swing liquidity within 0.5 ATR.
- Liquidity sweep occurred and no retest candle has closed.

Retest requirement:

```text
pullback_ok = price returns to EMA20/EMA50, broken structure, or fair-value area
retest_ok = retest holds for 1-3 candles without wick rejection against trade
confirmation_ok = close resumes in direction with body/range >= 0.45 and body <= 1.2 ATR
```

## A+ Entry Framework

Required entry conditions:

- Session quality high.
- Spread acceptable.
- HTF aligned.
- EMA alignment and slope aligned.
- ADX strong.
- Clean structure.
- Stable volatility.
- Retest confirmed.
- No divergence against trade.
- No exhaustion.
- No liquidity trap nearby.
- Candle quality high.
- Minimum RR >= 1.8.
- Regime confirmed.

If any critical condition is missing: HOLD.

Entry score:

```text
entry_score =
  0.16 * htf_alignment +
  0.14 * regime_score +
  0.12 * structure_quality +
  0.12 * retest_quality +
  0.10 * trend_quality +
  0.10 * volatility_quality +
  0.08 * candle_quality +
  0.08 * liquidity_safety +
  0.06 * spread_quality +
  0.04 * rr_quality

normal threshold: >= 0.82
SAFE_MODE threshold: >= 0.90
```

## Microstructure Filter

Compute:

- Wick dominance: `max(upper_wick, lower_wick) / range`.
- Candle efficiency: `abs(close - open) / (high - low)`.
- Rejection state: wick against direction > 0.45 range.
- Fake breakout: breaks swing then closes back inside range.
- Liquidity grab: pierces swing by > 0.1 ATR and rejects.
- Stop hunt: sweep plus opposite close plus elevated volume proxy.
- Compression quality: BB width percentile and post-compression expansion score.
- Absorption: high wick, low close progress, elevated tick volume.

Block if candle efficiency < 0.35, wick against trade > 0.45, fake breakout true, or absorption against trade true.

## Exit Framework

Exit score:

```text
exit_score =
  0.25 * regime_flip +
  0.20 * momentum_decay +
  0.15 * volatility_anomaly +
  0.15 * liquidity_failure +
  0.10 * divergence_against +
  0.10 * structure_break +
  0.05 * time_decay

close if exit_score >= 0.65
tighten SL if exit_score >= 0.45
```

Immediate exit:

- Regime flips against position.
- Volatility anomaly appears.
- Strong divergence against position.
- Liquidity sweep against position confirmed.
- Market structure breaks.

Position lifecycle:

- Partial TP: 30% at 1R, 30% at 1.8R, let 40% trail.
- Breakeven: after 1R only if spread stable and no wick failure.
- Time stop: exit after 12 M15 bars if profit < 0.3R and momentum decays.
- Max hold: 48 M15 bars unless trend quality remains high.
- Smart trailing: trail below/above structure or 1.2 ATR, never closer than noise band.

## SAFE_MODE

Activate when:

- Drawdown rising.
- Weekly DD > 8%.
- Winrate over last 20 trades < 40%.
- Confidence calibration unstable.
- Regime accuracy degrading.
- 2 consecutive losses in same direction.

Rules:

- A+ only.
- Half size.
- Max 1 trade/day.
- No RL, no AI override.
- Cooldown 180 minutes.
- HTF confirmation mandatory.
- Entry score >= 0.90.
- Spread/ATR <= 0.05.

## Code-Level Implementation Plan

Add modules:

- `hard_risk_lock.py`: daily/weekly/monthly DD, R tracking, cooldowns, session lock, equity slope.
- `institutional_regime.py`: CHOP, ATR percentile, trend efficiency, compression, expansion quality, structural integrity.
- `microstructure.py`: wick, fake breakout, liquidity grab, stop hunt, absorption, candle efficiency.
- `confidence_calibration.py`: Brier score, bucket calibration, AI trust coefficient.
- `safe_mode.py`: state machine for defensive mode.
- `setup_scorer.py`: A+ gate and entry scoring.
- `trade_journal.py`: post-trade feature persistence and failure clustering.

Refactor existing modules:

- `risk.py`: replace permissive scaling with hard locks and institutional sizing.
- `strategy.py`: remove live RANGE branch; generate candidates only, not final trade permission.
- `main.py`: enforce hard gates before AI; make AI a reducer during L0-L2.
- `ai_model.py`: add cold-start gates around RL, SGD, LSTM, memory boost, and ensemble weight.
- `anti_chase.py`: convert critical chase conditions to hard blocks.
- `noise_filter.py`: merge with microstructure quality and lower thresholds in SAFE_MODE.
- `exit_intelligence.py`: add proactive exit score and position lifecycle.
- `brain_memory.py`: store regime, volatility, trend, confidence components, spread, ATR percentile, liquidity state, entry/exit reason, and failure cluster.

## Pseudocode

```python
def evaluate_symbol(symbol):
    account = executor.get_account_info()
    if not hard_risk_lock.allow_new_trade(account):
        return HOLD

    df = market_data(symbol)
    features = feature_builder.compute(df)

    if not session_gate.ok(now) or news_gate.blackout() or spread_gate.block(features):
        return HOLD

    regime = institutional_regime.classify(features)
    if regime.hard_block or regime.score < 0.78:
        return HOLD

    candidate = strategy.candidate(df, regime)
    if candidate.signal == HOLD:
        return HOLD

    micro = microstructure.assess(df, candidate.signal)
    if micro.hard_block:
        return HOLD

    chase = anti_chase.assess(df, candidate.signal)
    if chase.hard_block:
        return HOLD

    setup = setup_scorer.score(candidate, regime, micro, chase)
    if setup.score < threshold_for_mode():
        return HOLD

    bootstrap = confidence_bootstrap.compute(...)
    ai = ai_model.predict(df)
    calibrated = confidence_calibrator.apply(ai)

    if autonomy_level <= 2 or closed_trades < 100:
        final_conf = min(bootstrap.score, bootstrap.score - ai.disagreement_penalty)
    else:
        final_conf = blend(bootstrap, calibrated, trust=calibrator.ai_trust)

    if final_conf < required_confidence():
        return HOLD

    risk = risk_sizer.compute(account, final_conf, regime, setup)
    if risk <= 0:
        return HOLD

    execute(candidate, risk)
```

## Roadmap

1. Emergency hardening: keep current config safe, max one position, no Kelly, safe mode, higher AI confidence, session-only trading.
2. Add `HardRiskLock` and route it before `_process_symbol()`.
3. Add `ConfidenceCalibration` and cold-start AI reducer logic.
4. Add institutional regime filter and disable live RANGE entries.
5. Upgrade anti-chase and microstructure hard blocks.
6. Replace confluence entry with A+ setup scorer.
7. Rewrite exit lifecycle and proactive exit scoring.
8. Expand trade journal and failure clustering.
9. Add backtest/walk-forward reports for DD, trade frequency, calibration, and regime-specific expectancy.
10. Only then allow AI trust to rise above 15%.

