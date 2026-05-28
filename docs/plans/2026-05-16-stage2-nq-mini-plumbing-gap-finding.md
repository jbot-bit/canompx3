# Stage 2 NQ-Mini Plumbing Gap — Design Note (Parked 2026-05-16)

**Status:** STAGE 2 DORMANT WIRING IMPLEMENTED 2026-05-29. Real profile
activation remains parked until Josh explicitly names the profile/divisor and
rollout conditions.

This note originally parked the work. It is retained because the plumbing gap
and activation cautions remain the right context for Stage 3.

## Context

The action-queue item `nq_mini_stage2_wiring_2026_05_15` (P2, owner=joshd) describes Stage 2
as "wire `resolve_execution_symbol(profile, strategy_symbol)` into
`trading_app/live/session_orchestrator.py` and `webhook_server.py` at the order-build
sites." Reading the action-queue note in isolation suggests a ~30-line, 1-file change.

This note records what an honest design pass found that the action-queue spec did NOT
mention.

## The plumbing gap

`resolve_execution_symbol()` at `trading_app/prop_profiles.py:157-171` takes an
`AccountProfile` as its first argument:

```python
def resolve_execution_symbol(profile: AccountProfile, strategy_symbol: str) -> tuple[str, int]:
    sym_map = profile.execution_symbol_map
    ...
```

`SessionOrchestrator.__init__` (`trading_app/live/session_orchestrator.py:298-336`) does
NOT receive an `AccountProfile`. It receives a `Portfolio` (line 306-307).

`Portfolio` (`trading_app/portfolio.py:90-103`) does NOT carry a reference to its source
`AccountProfile`. It encodes the profile only indirectly via `Portfolio.name` (set to
`f"profile_{profile_id}"` at `portfolio.py:948`).

Therefore the orchestrator has no clean access to the
`execution_symbol_map`/`execution_qty_divisor` fields needed by Stage 2.

## What Stage 2 actually requires

To wire `resolve_execution_symbol()` into the orchestrator ENTRY branch, one of three
paths must be taken:

### Option A (Recommended) — Plumb AccountProfile through Portfolio

Add `account_profile: AccountProfile | None = None` to `Portfolio` dataclass at
`trading_app/portfolio.py:90`. Set it in `build_portfolio_for_profile()` at
`trading_app/portfolio.py:947-955`. Orchestrator reads `self.portfolio.account_profile`.
Backwards-compatible: 22 existing `Portfolio(...)` construction sites (7 in
`portfolio.py`, 2 in `live_config.py`, 13 in tests) all remain valid with the new field
defaulting to `None`. Cleanest architecturally — Portfolio already mediates Profile→Strategies.

**Circular-import caveat:** `trading_app/portfolio.py` lazily imports
`trading_app/prop_profiles` at function bodies (lines 728, 733, 910), suggesting the
module has historical circular-import sensitivity. A top-level `from
trading_app.prop_profiles import AccountProfile` may or may not cause a regression —
likely safe (no reverse dependency), but the conservative implementation is to use
`TYPE_CHECKING` + string-quoted annotation:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trading_app.prop_profiles import AccountProfile

@dataclass
class Portfolio:
    ...
    account_profile: "AccountProfile | None" = None
```

### Option B — Pass AccountProfile into orchestrator __init__

Add `account_profile: AccountProfile | None = None` to
`SessionOrchestrator.__init__()`. Three production construction sites to update
(`trading_app/live/multi_runner.py:64`, `scripts/e2e_sim_test.py:38`,
`scripts/run_live_session.py:700`). More boilerplate at each call site, but no Portfolio
dataclass change. Less canonically-linked than Option A.

### Option C — Parse profile_id from Portfolio.name

`Portfolio.name = f"profile_{profile_id}"`. The orchestrator could strip the prefix and
look up `ACCOUNT_PROFILES[profile_id]`. **REJECTED by integrity-guardian.md § 2** —
re-encodes the convention; string-parsing is fragile; the linkage should be a typed
reference, not a parsed name.

## Why this is parked

Stage 2 ships **dormant infrastructure**. No current `ACCOUNT_PROFILES` row populates
`execution_symbol_map`. Drift check `check_nq_mini_substitution_wired_or_unused`
(`pipeline/check_drift.py:7536-7641`) already enforces invariant 1 (no profile populated
→ PASS regardless), so no silent-mis-route hazard exists today. The check's invariant 2
(populated AND callsite-present) becomes load-bearing only when a profile opts in.

Until a profile-activation decision is made, the ROI of doing Stage 2 is:
- 0 R/day P&L impact (dormant code).
- 0 reduction in current capital risk (drift check already protects).
- +1 production file (orchestrator), +1 production file (portfolio dataclass), +1 test
  file extended, possible 22-site construction verification, adversarial audit gate
  (optional MEDIUM-severity per `.claude/rules/adversarial-audit-gate.md`).

The user's current mission (2026-05-16) is "ship the live MNQ/MES/MGC app, fix only live
blockers". Stage 2 closes a future hole, not a live blocker. Parked.

## Reopen criteria

Reopen Stage 2 when ANY of the following is true:
1. A profile-activation decision is made (i.e., the user wants to actually route MNQ→NQ
   on a specific profile for a specific firm).
2. The Stage 2b webhook wiring is being designed (TradingView path also needs
   `AccountProfile` access; Option A's Portfolio field would have to be paired with a
   separate webhook-side profile-resolution mechanism).
3. Another feature requires the AccountProfile→Orchestrator linkage (e.g., per-profile
   kill rules, per-profile sizing overrides, per-profile session limits). At that point
   Option A becomes load-bearing for more than Stage 2 alone.

## What was NOT parked

`docs/runtime/action-queue.yaml::nq_mini_stage2_wiring_2026_05_15` is now `status:
parked` for Stage 3 profile activation. Stage 2 wiring exists, but no real profile row
has been activated.

## Lessons

- Action-queue `next_action` blocks are summaries, not designs. A summary that says
  "wire X at site Y" can hide an entire layer of plumbing if the caller does not
  currently have access to the dependency.
- The first design pass on this work assumed `self.account_profile` existed on the
  orchestrator. A 30-second grep showed it does not. Plans should validate every
  load-bearing path reference before claiming "1-file change."
- Dormant infrastructure protected by a drift check is institutionally fine. There is no
  obligation to ship the wiring just because the contract landed. The trigger to ship is
  the moment the wiring stops being dormant.
