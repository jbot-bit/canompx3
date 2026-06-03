# SR-Watchlist Part C — Report-Only Enrollment Diagnostic (DESIGN, not yet built)

**Status:** DESIGN APPROVED + 2nd-pass-audited (scope: Part C only, then
reassess). No production code written. Tier B capital-adjacent → IMPLEMENTATION
requires its own stage file + adversarial-audit gate before merge, even though
Part C itself writes no capital-affecting state. Second-pass corrected 3 errors
(gate location, NO_DATA==UNKNOWN gating, paper_trades tier dependency) — see § 9.

**Authored:** 2026-06-03, branch `session/joshd-maximise-ops-fix`.
**Grounds:** `memory/project_allocation_bottleneck_is_sr_unknown_circular_gate_2026_06_03.md`
(live numbers re-verified this session, below).

---

## 1. The problem (live, fresh from DB this session)

Profile `topstep_50k_mnq_auto`, rebalance 2026-05-30 (staleness OK, 4 days):

- **active_count = 3, paused_count = 845, all_scores = 848.**
- Paused-reason histogram (this session, via `get_lane_allocation_summary`):
  - **782** — `strict live gate: SR status UNKNOWN is not CONTINUE for fresh live allocation` (93% of paused)
  - 42 — `Session regime COLD` (LEGIT — negative recent regime; KEEP)
  - 19 — `live tradeability gate: EN deployment-unsafe: filter_type 'PD_*' selects by close-confirmed break_dir` (LEGIT E2 look-ahead/fill-safety; DO NOT TOUCH)
  - 2 — `strict live gate: SR status ALARM` (LEGIT — real drift)
- `list_promotable_candidates`: **total_promotable = 719, currently_allocated = 3, validated = 848.**

## 2. Root cause — circular SR-maturity gate (traced in code; CORRECTED in 2nd pass)

- **The PAUSE decision lives in `scripts/tools/rebalance_lanes.py:128-138`**, NOT
  the allocator. It is **flag-gated by `--strict-live-clean`** (line 128): when
  set, every lane with `sr_status != "CONTINUE"` is forced to `status="PAUSE"`
  with reason `"strict live gate: SR status {X} is not CONTINUE for fresh live
  allocation"`. (My first-pass attribution to
  `lane_allocator.enrich_scores_with_liveness` was WRONG — `enrich` only
  *populates* `sr_status` from `sr_state.json`; it does not pause.)
- `enrich_scores_with_liveness()` (`lane_allocator.py:625`) sets
  `s.sr_status = sr_state.get(strategy_id, "UNKNOWN")`.
- `sr_state.json` only holds lanes the SR monitor SCORED.
- `sr_monitor._build_lanes()` (`trading_app/sr_monitor.py:106-128`) iterates
  `get_profile_lane_definitions(profile_id)` → resolves to `profile.daily_lanes`
  or `load_allocation_lanes()` (`trading_app/prop_profiles.py:1191-1234`) =
  **the current allocation only.**
- **Loop:** allocated ⇒ scored ⇒ CONTINUE-eligible; unallocated ⇒ UNKNOWN ⇒ paused under `--strict-live-clean` ⇒ never allocated.

**CRITICAL 2nd-pass correction — the gate pauses `!= "CONTINUE"`, so `NO_DATA`
is paused IDENTICALLY to `UNKNOWN`** (`rebalance_lanes.py:135`). Enrolling a
candidate only unlocks it if it reaches **CONTINUE** (≥1 non-alarming forward
trade). A scored-but-zero-forward-trade candidate is `NO_DATA` → still paused.
**So the realistic unlock is bounded by candidates with ≥1 forward trade, NOT by
the full 719.** This is precisely why "honest NO_DATA" is not just ethics — it is
what the gate already enforces. Part C's NO_DATA-vs-CONTINUE count IS the unlock
estimate.

**Enabling fact (with the dependency my first pass silenced):**
`prepare_monitor_inputs()` (`sr_monitor.py:157-208`) is a **3-tier preference**,
NOT an unconditional canonical-forward read:
  1. ≥`BASELINE_WINDOW` (=50) `paper_trades` rows → baseline from paper, monitor remainder;
  2. some (<50) paper rows → monitor all paper;
  3. **zero paper rows → `_load_canonical_forward_trades()`** (canonical `orb_outcomes` since `HOLDOUT_SACRED_FROM`).
Watchlist candidates (never live/paper) MUST land in tier 3. **Part C must
ASSERT `paper_trades` is empty for each enrolled candidate** (query
`SELECT count(*) FROM paper_trades WHERE strategy_id=?`); if any stray paper rows
exist the baseline source silently diverges and the verdict is not
canonical-forward-derived. Surface any non-empty case explicitly.

## 3. SR verdict logic (read this session — reshapes the diagnostic)

`run_monitor` (`sr_monitor.py:289-302`):
- `status = "NO_DATA"` iff zero forward trades.
- `status = "CONTINUE"` iff ≥1 forward trade AND the SR monitor never alarmed.
- `status = "ALARM"` iff the SR statistic crosses threshold.

**There is NO minimum-N floor for CONTINUE** — a single non-alarming forward
trade yields CONTINUE today. So the Part C diagnostic split is:
**NO_DATA (zero post-holdout forward trades) vs scorable (≥1).**
This is the load-bearing honesty point (see § 6).

## 4. Part C scope — report-only, ZERO capital risk

1. Add a **watchlist universe** to `sr_monitor`: alongside allocated lanes,
   enroll all promotable-FIT candidates (the 719) as monitor-only lanes.
   - New helper (likely `get_promotable_watchlist_lanes()` in `prop_profiles.py`)
     returns canonical lane defs for FIT-but-unallocated strategies, reusing
     `parse_strategy_id` + the same lane-def shape as
     `get_profile_lane_definitions` (institutional-rigor §4 — delegate, don't
     re-encode).
2. Score each via the EXISTING `_load_canonical_forward_trades` path.
3. Write verdicts to a SEPARATE **`sr_watchlist_state.json`** using the
   canonical derived-state envelope (drift #124/#125 enforce the contract).
4. **The allocator never reads this file.** No lane becomes live. Pure
   observability.
5. New drift check: `sr_watchlist_state.json` is report-only — assert no
   allocation-consuming source reads it (mirrors the lane_allocation authority
   inversion checks #163/#164).

## 5. Anticipated blast radius (NOT edited in this design phase)

| File | Part C role |
|---|---|
| `trading_app/sr_monitor.py` | add watchlist universe + `_build_watchlist_lanes()`; write `sr_watchlist_state.json`. |
| `trading_app/prop_profiles.py` | `get_promotable_watchlist_lanes()` helper (canonical lane defs for FIT-unallocated). |
| `trading_app/lane_allocator.py` | **NOT touched in Part C.** (Part E only.) |
| `scripts/tools/rebalance_lanes.py` | **NOT touched in Part C.** This is where the `--strict-live-clean` gate (lines 128-138) actually pauses. Part E target, not C. |
| new `sr_watchlist_state.json` | derived-state envelope; report-only. |
| `pipeline/check_drift.py` | +1 check: watchlist file is report-only / unread by allocator AND rebalance_lanes. |

## 6. Self-check (happy / edge / failure)

- **Happy:** 719 enrolled → each scored from canonical forward N → watchlist file
  shows CONTINUE/ALARM/NO_DATA distribution. Allocator unchanged. ✓
- **Edge (THE critical one):** post-`HOLDOUT_SACRED_FROM` (2026-01-01) is a short
  window. Many candidates will be **NO_DATA** (zero forward trades) — they can
  never reach CONTINUE no matter the gate. Part C's PRIMARY DELIVERABLE is
  quantifying this: "X of 719 are scorable, Y are sample-starved NO_DATA." This
  tells us whether the SR gate is the only bottleneck or whether forward-sample
  is the real ceiling. **Decision (this session): honest NO_DATA — no capital
  moves on NO_DATA; surface the gap.**
- **Failure (envelope drift):** watchlist file MUST reuse the canonical
  derived-state envelope writer/reader or checks #124/#125 fail closed.

## 7. Part E (NOT in scope now — recorded for lineage)

The actual capital unlock: allocator liveness gate accepts a **matured CONTINUE**
from watchlist monitoring (defined maturity rule: min forward N, CONTINUE not
ALARM, holdout-clean) → watchlist→allocation promotion. Full adversarial-audit
gate + Monte-Carlo survival (Criterion 11) + per-account DD matching
(self-funded sizing doctrine). Build only after Part C's numbers justify it.

## 8. Reassess trigger

After Part C ships and is run: read the NO_DATA/CONTINUE/ALARM split. If the
scorable set is large enough to matter, design Part E. If most candidates are
NO_DATA, the real bottleneck is forward-sample, not the gate — and the next
lever is time/data accrual, not allocator surgery.

## 9. Second pass — bias, gaps, silences (self-audit before any build)

Ran a targeted adversarial pass on this design (2026-06-03). Fixes already folded
into §§ 2, 5 above. Residual items the IMPLEMENTATION session must carry:

- **Confirmation bias — "the gate is the bottleneck" is a HYPOTHESIS, not a
  given.** The `--strict-live-clean` gate may be a *deliberate* conservative live
  floor the operator wants. Part C is report-only precisely so it can DISPROVE
  the unlock thesis (most candidates NO_DATA ⇒ gate isn't the real ceiling)
  without committing to Part E. Do not frame Part C as "the fix" — it is the
  measurement that decides whether a fix is even warranted.
- **Conditional-on-flag silence.** The 782 pauses exist because the 2026-05-30
  rebalance ran WITH `--strict-live-clean` (proven: that exact reason string only
  emits under the flag, `rebalance_lanes.py:128`). The watchlist's relevance
  holds only while that flag stays on. If a future rebalance drops the flag, the
  bottleneck changes shape — re-read the histogram before assuming Part C still
  applies.
- **paper_trades dependency (folded into § 2).** Watchlist verdicts are only
  canonical-forward-derived if `paper_trades` is empty for the candidate. ASSERT
  + surface, never assume.
- **Half-size / shadow / `_S075` lane identity.** `get_profile_lane_definitions`
  carries `is_half_size`, `stop_multiplier`, `shadow_only`. The watchlist helper
  must decide how to treat these for FIT-unallocated candidates (default: enroll
  at full-size identity, mark provenance) — do NOT silently inherit the active
  profile's stop_multiplier onto a candidate that was validated at a different
  one. Open question for the IMPLEMENTATION stage.
- **Duplicate-identity collisions.** 848 scored vs 719 promotable vs 3 active —
  confirm the watchlist universe = promotable-FIT MINUS already-allocated, with
  no double-count across apertures (the triple-aperture trap;
  `.claude/rules/daily-features-joins.md`).
- **Cost-of-scoring.** Scoring ~719 lanes pulls forward outcomes per lane.
  `compute_pairwise_correlation` already SIGSEGV'd at 762 lanes once
  (`lane_allocator.py:667`, 2026-05-21). Part C does NOT correlate (report-only,
  no selection) so it should be safe, but bulk-load forward outcomes once per
  (instrument, orb_minutes) rather than 719 separate connects.

## 10. Part C acceptance criteria (done = all true)

1. `sr_watchlist_state.json` exists, derived-state-envelope-valid (drift #124/#125 pass).
2. New drift check proves NO allocation/rebalance consumer reads the watchlist file.
3. Output reports, for the 719: counts of CONTINUE / ALARM / NO_DATA, plus the
   forward-N distribution and the paper_trades-non-empty exceptions.
4. Allocator + rebalance behavior byte-identical to pre-change (report-only proof).
5. `check_drift.py` green; watchlist scoring runs without SIGSEGV on the full set.
6. Self-review + (capital-adjacent module touched) evidence-auditor pass before merge.
