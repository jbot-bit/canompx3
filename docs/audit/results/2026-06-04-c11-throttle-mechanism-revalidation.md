# C11 Throttle — Live-Mechanism Re-Validation Result (Stage 1)

**Date:** 2026-06-04
**Profile:** `topstep_50k_mnq_auto` (MNQ, 3-lane book)
**Design:** `docs/plans/2026-06-04-c11-throttle-live-mechanism-design.md`
**Stage:** `docs/runtime/stages/2026-06-04-c11-throttle-mechanism-revalidation.md`
**Harness:** `C:\Users\joshd\c11_matrix\throttle_mechanism_revalidate.py` (read-only, outside repo)
**Predecessor:** `docs/audit/results/2026-06-04-c11-throttle-validation.md` (validated the
non-deployable 0.5× *size*-scale form; this result tests the deployable *integer
participation* forms).

---

## Scope

The validated equity-drawdown throttle halves daily participation by scaling pnl
**0.5×**. The live book emits a **fixed 1 micro per signal** and the canonical resolver
`prop_profiles.resolve_execution_order` **fails closed on non-integer quantity** — so a
0.5× size scale is forbidden at 1 micro (the gate-vs-live divergence class C11 exists to
kill). The naive integer alternative — `factor=0.0`, skip ALL entries while in drawdown —
was separately validated (`throttle_validate.py`, factor=0.0 primary) and **FAILS the 85%
edge floor catastrophically** because it latches forever (skipped days add $0 → running
peak never advances → never recovers; ~17% full / 0% holdout edge retained).

This Stage-1 exercise tests three **integer participation** mechanisms — ones that cut
participation by ~half using only take=1 / skip=0 per-signal decisions, WITHOUT a
half-micro and WITHOUT permanent latching — across the 9 design factors, to determine
whether any is a viable live C11 fix. **Validation only.** No production-code edit, no
mechanism committed, no live arming.

Mechanisms:
- **A — deterministic alternating skip** ("sit out every other signal while engaged"):
  take every other eligible signal. ~50% participation, fully causal, parity-safe with
  NO shared RNG.
- **B — Bernoulli skip (seeded, p=0.5):** take each engaged signal w.p. 0.5 from a seeded
  integer stream. Unbiased over which trades drop; needs a shared seeded+logged draw for
  gate↔live parity.
- **C — daily participation budget / stop-after-first-loss:** an **intraday** rule.

## Decision

**VERDICT: NO live-runnable integer participation mechanism clears C11 robustly.** The
throttle is **NOT a viable live C11 fix** on a 1-micro book at this trigger grid.
`topstep_50k_mnq_auto` C11 remains **NO-GO, live NOT armed.**

### 9-factor matrix (binding gates: full DD ≤ $1,600, holdout DD ≤ $1,600, edge_full ≥ 85%, breach = 0, WF ≥ 3/5; band ≥ 2 adjacent)

| Mechanism | trig | full 90d DD | edge_full | edge_ho | WF | drops winners | gate |
|---|---|---|---|---|---|---|---|
| A alt-skip | 600 | **$2,298** (WORSE than baseline) | 76.9% | 33.9% | 5/5 | **YES** | FAIL (DD + edge) |
| A alt-skip | 800 | **$2,298** (WORSE) | 78.8% | 35.4% | 5/5 | **YES** | FAIL (DD + edge) |
| A alt-skip | 1000 | $1,651 | 90.9% | 98.1% | 4/5 | no | FAIL (DD) |
| A alt-skip | 1200 | $1,440 ✅ | 88.9% ✅ | 100.0% | 5/5 | no | **clears, but band width = 1 (knife-edge)** |
| B Bernoulli | 600 | $1,548 ✅ | **67.6%** ❌ | 50.5% | 5/5 | YES | FAIL (edge) |
| B Bernoulli | 800 | $1,667 | 81.6% | 44.1% | 4/5 | no | FAIL (DD + edge) |
| B Bernoulli | 1000 | $1,925 | 91.6% ✅ | 87.5% | 4/5 | no | FAIL (DD) |
| B Bernoulli | 1200 | $1,772 | 64.6% | 100.0% | 4/5 | YES | FAIL (DD + edge) |
| **C daily budget** | — | — | — | — | — | — | **NOT EVALUABLE** |

- **A** clears all gates at exactly ONE trigger (1200) → **band width 1 = knife-edge / overfit**, fails the ≥2-adjacent robustness requirement. At triggers 600/800 it makes DD **worse than baseline** ($2,298 vs $2,039) by dropping net winners during recovery.
- **B** clears DD at trigger 600 but fails the 85% edge floor (67.6%); no trigger clears both DD and edge. Coin-flip variance on a 1-micro book is too high.
- **C** is an intraday rule (cap engaged-day trades / stop after the first intraday loss). The `DailyScenario` series is one aggregate pnl per day with NO intraday trade sequence, so C is **structurally not evaluable** on this data. The harness refuses to fake it (institutional-rigor §6, no silent failures) and logs it as NOT-EVALUABLE per the stage file's "a skipped factor must be logged as such, not omitted."

### Root cause (why half-participation ≠ half-size)

Halving *size* (the 0.5× validated form) shaves every trade's contribution **uniformly**,
smoothing the drawdown. Halving *participation* by integer skipping is **lumpy**: it cannot
control *which* trades it drops relative to the drawdown path. When alternating-skip
happens to drop the winners that drive a recovery, DD gets worse, not better (A 600/800 =
$2,298). So the only 1-micro-legal way to approximate "half participation" does not inherit
the size-scale form's drawdown-smoothing property. This is structural, not a tuning miss.

## Files

- `C:\Users\joshd\c11_matrix\throttle_mechanism_revalidate.py` — A/B/C × 9-factor harness
  (read-only, outside repo; baseline $2,038.84 parity anchor; daily-loss belt; anchored WF
  2020–24; frozen 2025 holdout; no-lookahead replay).
- `C:\Users\joshd\c11_matrix\throttle_validate.py` — predecessor (size-scale 0.5× validated,
  0.0 skip-all primary = edge-floor FAIL).
- Canonical sources consulted (read-only, never mutated): `trading_app/account_survival.py`
  (`_load_profile_daily_scenarios`, `_max_observed_rolling_drawdown`,
  `_historical_daily_loss_breach_days`, `simulate_survival`), `trading_app/prop_profiles.py`
  (`get_profile`, `daily_loss_dollars`, `resolve_execution_order`).
- Literature grounding: `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
  — Kelly "adjust risk to current capital" (p.149); "use HALF-KELLY" (p.146); prop-account
  trailing-DD forces low effective vol target (p.83). Carver's native mechanism is
  continuous *size* scaling (p.127 notes our fixed 1-contract sizing), which a 1-micro book
  cannot run — these integer participation mechanisms are the discretization onto 1 micro,
  and this result shows that discretization does not preserve the drawdown-control property.

## Verification

- **Parity anchor:** baseline max-90d-DD reproduces canonical **$2,038.84 exactly** (`True`),
  total $23,412.29; and at a never-engaging trigger A reproduces baseline DD + total exactly
  (mechanism machinery is clean against the canonical baseline).
- **Mechanism confirmation:** at trigger 600, A skips 422 days netting **+$5,418** of pnl
  (120 winners / 139 losers, winners larger) — the harness empirically confirms it drops net
  winners while throttled, which is why DD worsens. The worse-DD result is a real mechanism
  property, not a harness bug.
- **No-lookahead:** one-day-at-a-time replay for A and B → leaks = 0 (day-t skip decision
  uses peak/bal through t-1 only; today's pnl never informs today's skip).
- **Daily-loss belt:** modeled canonically (`min(min_balance_delta, total_pnl) ≤ −$450`);
  0 baseline breaches, 0 throttled breaches across all evaluable cells.

## Limitations

This is a Stage-1 validation verdict, **not** an approval to arm, and several items are
open or out of scope — stated honestly:

- **Factor C is UNSUPPORTED on this data.** The daily-aggregate `DailyScenario` series has
  no intraday trade sequence, so "stop after first intraday loss" / per-day trade caps
  cannot be evaluated. C may yet help, but proving it requires intraday trade-level data
  this harness does not have. It is logged NOT-EVALUABLE, not dismissed.
- **Trigger grid is bounded (600–1200, recover = 0.5× trigger).** A wider or finer grid was
  not swept; A clears at a single trigger (1200), so a narrow band near 1200 might widen
  with a finer grid — but a knife-edge single cell is, per the design's robustness gate,
  exactly the overfit signature this validation is built to reject. Reporting the bound
  rather than silently truncating (no-silent-caps).
- **Edge-floor discipline held, not relaxed.** The 85% full-history floor is the YAML's
  binding criterion; no goalpost was moved to manufacture a pass. The softer 70% holdout
  floor is reported, not used as a hard kill.
- **MC survival factor (#4) not run for the integer mechanisms.** The predecessor ran a
  parity-proven throttle-aware MC for the 0.5× form; this Stage-1 stops at the DD/edge/WF
  gates because no mechanism clears them — an MC on a failing candidate adds no decision
  value. If a future grid produces a clearing band, the MC factor must be run before any
  arming.
- **Holdout touch discipline.** Holdout DD is printed for all grid cells (multi-cell read),
  so treat it as corroboration, not a single-touch test — same caveat as the predecessor.
- **Implementation is Tier B (capital path) and NOT done; arming requires a separate
  operator GO + the still-OPEN bracket-parity adversarial-audit gate (`9b3fc530`).** This
  result arms nothing.

## Next (operator decision — do NOT proceed without it)

The participation-throttle path is closed at this grid. The genuine remaining levers:
1. **Multi-micro base size** — if the book trades ≥2 micros per signal normally, a true
   0.5× *size* scale becomes a legal integer (2 → 1), recovering the validated $1,459 DD
   form on a deployable basis. This is the most direct fix and matches the operator's
   sizing instinct. Requires re-validating the 0.5× form on the multi-micro book.
2. **Intraday stop-after-first-loss (mechanism C)** — requires intraday trade-level data;
   build that data path, then evaluate C.
3. **ORB-cap / bracket-parity path** — separate track; audit `9b3fc530` still open.
4. **Larger account** — the 3-lane book fits at Topstep $100k / Bulenox $50k per prior C11
   analysis.

**Live status: STILL BLOCKED. NO-GO.**
