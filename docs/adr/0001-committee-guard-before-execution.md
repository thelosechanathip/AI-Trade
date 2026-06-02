# ADR 0001: Committee Guard Before MT5 Execution

Status: Accepted

## Context

AI-Trade already has a rule engine, Market Brain, memory, stability filters,
noise filtering, anti-chase logic, and risk sizing. The missing architectural
piece was a final structured review that combines thesis quality, quant quality,
risk state, and execution sanity after SL/TP and lot size are known but before
the MT5 order is sent.

AutoHedge uses a Director -> Quant -> Risk -> Execution pipeline. Vibe-Trading
uses preset research and risk workflows that validate outputs before delivery.
AI-Trade should keep its local deterministic design, but adopt that review shape
as a fail-closed execution guard.

## Decision

Add `InvestmentCommitteeGuard` as a local deterministic committee gate. It runs
after final score, stability, noise, anti-chase, SL/TP, RR, and lot sizing are
computed, and before `MT5Executor.place_market_order()`.

The committee has four votes:

- Director: signal, regime, MarketIntelligence block flags, HTF alignment.
- Quant: final score, adjusted score, bootstrap confidence, ADX, noise, chase.
- Risk: Brain uncertainty, reversal probability, drawdown, capacity.
- Execution: RR, spread/ATR, lot size, close price, executable signal.

The order is blocked on veto, insufficient consensus, or low committee score.
Approved but cautious setups can scale lot size down before execution.

## Consequences

- Live orders now carry one additional fail-closed safety layer.
- The dashboard state includes the latest committee verdict for each symbol.
- The trade journal stores committee verdict, score, and risk multiplier.
- Thresholds are configurable in `committee_guard` across master, safe, and
  aggressive configs.
- The system remains local and does not import external repo dependencies.
