# ADR 0002: Operator Controls Cannot Override Risk Budget

Status: Accepted

## Context

Operators need to choose how many child orders are opened for an approved
signal and, when desired, request a fixed lot per child order. Treating those
values as direct MT5 commands would let the dashboard bypass position sizing,
portfolio exposure limits, Committee Guard decisions, and broker constraints.

The engine also needs an immediate control for disabling new entries without
stopping position management or exit intelligence.

## Decision

Add a persistent Execution Control Plane with audited revisions. It controls:

- Whether new entries are enabled.
- Requested child orders per approved signal.
- Risk-split or fixed-capped lot mode.
- Requested lot per child order in fixed-capped mode.

The engine continues to calculate one aggregate risk-sized lot budget. An Order
Plan can only divide or reduce that budget. It cannot increase it. The plan is
also capped by portfolio open risk, remaining global and directional capacity,
configured execution guardrails, and broker volume rules.

Control updates use optimistic revisions and are written atomically. Invalid or
unreadable controls fail closed by disabling new entries. Open-position
management continues independently.

## Consequences

- Multiple child orders share one approved risk budget rather than multiplying
  risk by the requested order count.
- Fixed lot is operator intent, not a risk override.
- A partially failed batch stops immediately and preserves successful child
  orders in the trade journal.
- Dashboard control writes are local or require the configured control key.
- Every accepted control change is appended to an audit log.
