# Four-Gate REGIME Standalone-Eligibility Monitor — Design

**Date:** 2026-06-03
**Author:** Claude Code
**Status:** DESIGN APPROVED (Stage 1 paper-shadow only). Capital-path; live wiring is a separate, re-approved stage.
**Operator directive:** trade REGIME-tier (N 30–99) standalone where the fat ExpR lives — gated by live monitoring, auto-pause on any gate flip. "Onto the regime, onto the hotness — proper monitoring."

---

## Purpose

REGIME-tier strategies (sample 30–99) carry the highest ExpR in the book
(e.g. MGC LONDON_METALS ~0.73R; MNQ COMEX_SETTLE ~0.62R rolling) but are
fenced out of standalone trading by RESEARCH_RULES.md doctrine
("Never standalone", lines 55 + 291). The earnings bottleneck is allocation,
not discovery (668 promotable MNQ lanes, 3 allocated). This monitor lets a
REGIME lane trade standalone ONLY while four independent gates all hold,
auto-pausing the instant any flips — turning fenced EV into disciplined,
monitored capacity instead of overfitting-to-noise.

## Grounding (surfaces verified read-only 2026-06-03)

- **Allocator gate chain** (`trading_app/lane_allocator.py`): a clean pipeline
  of `apply_chordia_gate` -> `apply_c8_gate` -> `apply_live_tradeability_gate`
  -> `build_allocation`, each taking and returning `list[LaneScore]`, each able
  to flip a lane to PAUSE with a reason. The new predicate is another gate in
  this chain. CONFIRMED.
- **`LaneScore`** carries `trailing_expr`, `trailing_n`, `recent_3mo_expr`,
  `session_regime_expr`, `sr_status` (defaults "UNKNOWN"), `status`,
  `status_reason`. It does NOT carry a fitness field — Gate B needs a join to
  fitness, not a field read. CONFIRMED.
- **Fitness** (`trading_app/strategy_fitness.py`): `classify_fitness(...)`
  returns FIT/WATCH/DECAY/STALE. FIT already requires
  `rolling_exp_r > 0 AND recent_sharpe_30 > -0.1 AND rolling_sample >= 15`.
  CONFIRMED.
- **Account DD** (`trading_app/account_survival.py`): `SurvivalSummary` exposes
  `trailing_dd_breach_probability`, `p95_max_dd`,
  `historical_max_observed_90d_dd_dollars`, profile-scoped via
  `read_survival_report_state`. CONFIRMED — but this file is currently DIRTY
  with live peer edits, so Gate D treats DD as an INJECTED value, not a hard
  import of the churning API.
- **SR-UNKNOWN circular gate**: `sr_status` defaults "UNKNOWN"; 782 lanes
  paused on it. OUT OF SCOPE for this work (separated by operator decision).
  The monitor is INERT on live allocation until SR-UNKNOWN is also fixed —
  stated loudly, not hidden.

## The four gates (de-biased — no double-counting)

A REGIME lane is standalone-eligible IFF ALL hold; any flip -> PAUSE:

- **Gate A — structural (primary):** session-regime healthy
  (`session_regime_expr` via the existing proven mechanism). This is the
  dominant gate. Rationale: the allocator's own history shows session-regime
  gating earned +630R while per-strategy month-streak pausing LOST -799R.
  Per-strategy signals must never override a healthy/unhealthy session call.
- **Gate B — strategy fitness:** `classify_fitness == FIT`. Covers recent
  expectancy + sharpe + min-sample-15. (NOT re-checked elsewhere.)
- **Gate C — REGIME-specific adequacy (additive to B), LITERATURE-GROUNDED:**
  - rolling-N >= 100 (NOT my earlier guess of 30). Grounded in
    Pepelyshev-Polunchenko 2015 (CUSUM/SR): the deployed SR drift detector
    estimates pre-anomaly mu/sigma from the **first 100 live trades** per
    strategy. Below 100, the detector is uncalibrated (`sr_status == NO_DATA`)
    so the strategy CANNOT be safely monitored standalone. Also consistent with
    Bailey-Lopez de Prado 2014 selection-bias framing (thin samples inflate
    measured ExpR).
  - last-trade within 60 TRADING days (NOT 45 calendar days). Grounded in
    Pepelyshev-Polunchenko: SR is calibrated to ARL-to-false-alarm ~= 60
    trading days (~one quarter). Recency aligns to the detector's own clock.
  - SR-calibrated: `sr_status != "NO_DATA"` (the detector has enough history).
- **Gate D — account safety:** account DD/MLL headroom present (injected from
  `SurvivalSummary`; decoupled from the peer-dirty API). Grounded in Carver 2015
  ch12 (speed-and-size): match lane variance/size to the account's DD profile.
- **Auto-pause = existing SR detector (NOT a new circuit-breaker).** GROUNDING
  CORRECTION: my earlier "hand-rolled DD-spike cooldown" reinvented
  Pepelyshev-Polunchenko CUSUM/SR, which is ALREADY IMPLEMENTED in
  `trading_app/sr_monitor.py` (A~=60, h~=0.3, ARL~=60d) and surfaced as
  `LaneScore.sr_status` (CONTINUE/ALARM/NO_DATA). A REGIME lane auto-pauses when
  `sr_status == "ALARM"`. No parallel cooldown is built — consume the canonical
  detector. This removes a redundant gate and a re-encoded-canonical-logic risk
  (institutional-rigor.md violation avoided).

Layering invariant: A is structural and dominant; B/C/D/cooldown are vetoes
that can only TIGHTEN eligibility, never loosen it, and never apply to CORE
lanes. This directly encodes the -799R lesson.

## Approaches considered

- **(A) Bolt onto the existing allocator gate chain as a new `apply_*_gate`** —
  CHOSEN. Lowest blast radius; reuses the proven +630R session-regime mechanism;
  mirrors the verified `apply_live_tradeability_gate` template exactly.
- (B) New standalone monitor service — REJECTED (YAGNI; duplicates fitness/DD
  infra; new failure surface).
- (C) Overlay/size-boost only — REJECTED (operator chose gated-standalone).

## Stage plan

- **Stage 1 (THIS, approved): paper-shadow + predicate ONLY.**
  Pure eligibility predicate + a shadow ledger that RECORDS what it WOULD gate,
  computed against live lane scores, changing ZERO live allocation. Proves the
  monitor would have gated REGIME lanes correctly before any capital.
- **Stage 2 (separate sign-off): live wiring.** Insert the gate into the chain
  so REGIME lanes can actually trade standalone. Requires SR-UNKNOWN resolved
  to have live effect.
- **Stage 3 (separate): SR-UNKNOWN circular-gate fix.**

## Failure modes & the tests that catch them (Stage 1)

- Recreating the -799R per-strategy-noise trap -> test: FIT strat in UNHEALTHY
  session is shadow-PAUSED (Gate A dominates).
- Stale hotness -> test: high rolling-ExpR but last trade old / rolling-N below
  REGIME floor -> shadow-PAUSED.
- CORE regression -> test: predicate is NEVER applied to CORE lanes (N>=100);
  CORE shadow verdict is always "n/a — not REGIME".
- Account blind spot -> test: DD headroom exhausted -> shadow-PAUSED even when
  A/B/C pass.
- Cooldown enabling -> test: cooldown state can only flip active->paused.
- Shadow leaks to live -> test: running the shadow ledger changes ZERO live
  allocation rows (the central Stage-1 safety property).

## Rollback

Stage 1 is purely additive (new module + shadow ledger + tests). It imports
nothing into the live allocation path. Removing the new files restores exact
prior behavior. Built in an isolated worktree off clean origin/main; main
untouched until merge + sign-off.

## Out of scope (explicit, no silent scope creep)

- No live allocation wiring. No capital-path edits. No SR-UNKNOWN fix.
- No profile / live_config / deployment-sizing edits.
- No R-to-dollar sizing (requires contract/account rules — approval-gated).
- No change to any deployment threshold.
