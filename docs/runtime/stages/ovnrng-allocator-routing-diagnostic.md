---
slug: ovnrng-allocator-routing-diagnostic
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Does ovn/atr predict which session is best to trade today? Read-only diagnostic.
---

# Stage: OVNRNG allocator-routing diagnostic

## Plan

### Question

PR #61's cross-session replication showed ovn/atr has session-specific
effects: Q5 (high overnight vol) is BEST on LONDON_METALS, US_DATA_1000,
NYSE_CLOSE and WORST/flat on NYSE_OPEN. This suggests an ALLOCATOR
signal (route to session conditional on ovn/atr), not a FILTER signal.

This diagnostic answers: **does a router rule "trade top-K sessions
per ovn/atr bin" beat "trade all DEPLOY lanes equally" on IS?**

If yes → worth writing a Pathway-B pre-reg for a router.
If no → ovn/atr is noise at the allocator level too; close the line.

### Task decomposition

1. Load 8 lookahead-clean sessions × MNQ E2 RR=1.5 CB=1 orb_minutes=5,
   IS only.
2. Compute ovn/atr quintile per trading day (using all-MNQ daily
   distribution — this is the ALLOCATOR's decision variable,
   session-invariant).
3. Build conditional `ExpR(session | ovn/atr bin)` table — 8 sessions
   × 5 bins = 40 cells.
4. Check: does the RANKING of sessions by ExpR CHANGE with bin? If
   the best session is fixed across all bins, no router signal. If
   it changes, router signal is present.
5. Simulate:
   - **Uniform policy**: equal weight across all 8 sessions — take
     every eligible trade.
   - **Router policy**: for each day (given its ovn/atr bin),
     TRADE ONLY the top-K sessions ranked by conditional ExpR,
     where K ∈ {1, 2, 3}.
6. Report both policies' aggregate ExpR, N, annualized Sharpe,
   and delta. Note power caveat: router policy has fewer trades per
   day, so annualized R may be lower even if per-trade ExpR is
   higher.
7. Sanity check: confirm the router's signal is not a look-ahead
   artifact. `ovn/atr` at Brisbane 17:00 is known for all 8 eligible
   sessions (they start ≥17:00). Good.
8. Tunnel-vision check: **test both top-K rankings BY ovn/atr bin AND
   alternate framings** (e.g., "top-K by all-time session ExpR,
   ignore bin" as a baseline router that doesn't use the signal —
   this controls for whether the router's signal value is from
   ovn/atr bin awareness or just from session concentration).

### Scope Lock

- `research/audit_ovnrng_allocator_routing.py` (new)
- `docs/audit/results/2026-04-21-ovnrng-allocator-routing.md` (new)

### Blast Radius

- Read-only. Zero production-code touch.
- Canonical data: `orb_outcomes`, `daily_features`.
- No new filters. No pre-reg. No deployment change.
- NOT writing any allocator-router code; that's a separate turn if
  this diagnostic passes.

### Acceptance criteria

1. Script runs without exceptions on current `gold.db`.
2. MD contains conditional ExpR(session | bin) 8×5 table.
3. MD contains "best session per bin" column + per-bin argmax sessions.
4. MD contains simulation: uniform vs router-top-K (K=1, 2, 3) ExpR,
   N, annualized Sharpe, delta.
5. MD contains control: router using bin-agnostic ranking (same
   top-K but fixed across bins) — isolates the BIN awareness's
   contribution.
6. MD classifies verdict: ROUTER_WORTHWHILE / MARGINAL /
   NO_SIGNAL.
7. MD discusses tunnel-vision / alternative framings / blockers.
8. `python pipeline/check_drift.py` passes.
9. No production code touched.

## Non-goals

- Not writing the router pre-reg (separate turn if PASS).
- Not proposing deployment change.
- Not testing OOS (peeking avoided).
- Not adjusting for multiple testing across different router K values
  (descriptive diagnostic, not confirmatory).
