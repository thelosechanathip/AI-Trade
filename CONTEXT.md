# AI-Trade Context

AI-Trade is a defensive autonomous XAUUSD execution engine. The domain language
prioritizes survival, stability, consistency, capital preservation, then profit.

## Glossary

- Rule Engine: the primary signal generator at autonomy levels L0-L2.
- Market Brain: advisory multi-agent context analysis that can reduce risk and
  only becomes primary at L3-L4.
- Hard Gate: a check that blocks execution and cannot be overridden downstream.
- Committee Guard: the final Director, Quant, Risk, and Execution review before
  an MT5 order is sent.
- Committee Verdict: structured approval, caution, or veto result from the
  Committee Guard.
- Risk-First Execution: order flow where sizing and execution are allowed only
  after risk state, setup quality, spread, RR, and context agree.
- Execution Control Plane: audited operator settings for enabling new entries,
  choosing requested orders per signal, and selecting a lot mode.
- Order Plan: the final risk-capped list of child orders derived from one
  approved aggregate lot budget.
- Fixed-Capped Lot: an operator-requested lot per child order that may only be
  reduced by risk budget, portfolio exposure, capacity, or broker constraints.
- Portfolio Open Risk: estimated cash loss at stop-loss across all strategy
  positions. New entries fail closed when this exposure cannot be verified.
