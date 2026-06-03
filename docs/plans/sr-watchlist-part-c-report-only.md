# SR-Watchlist Part C — Report-Only Enrollment Diagnostic (DESIGN, not yet built)

**Status:** DESIGN APPROVED (scope: Part C only, then reassess). No production code
written. Tier B capital-adjacent → IMPLEMENTATION requires its own stage file +
adversarial-audit gate before merge, even though Part C itself writes no
capital-affecting state.

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

## 2. Root cause — circular SR-maturity gate (traced in code)

- `lane_allocator.enrich_scores_with_liveness()` → `sr_status = sr_state.get(id, "UNKNOWN")`; UNKNOWN fails the strict live gate.
- `sr_state.json` only holds lanes the SR monitor SCORED.
- `sr_monitor._build_lanes()` (`trading_app/sr_monitor.py:106-128`) iterates
  `get_profile_lane_definitions(profile_id)` → resolves to `profile.daily_lanes`
  or `load_allocation_lanes()` (`trading_app/prop_profiles.py:1191-1234`) =
  **the current allocation only.**
- **Loop:** allocated ⇒ scored ⇒ CONTINUE-eligible; unallocated ⇒ UNKNOWN ⇒ never allocated.

**Enabling fact:** `_load_canonical_forward_trades()` (`sr_monitor.py:139-154`)
already pulls the forward stream from canonical `orb_outcomes` since
`HOLDOUT_SACRED_FROM` — NOT live paper fills. The monitor can already score any
candidate; it just isn't enrolled.

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
| new `sr_watchlist_state.json` | derived-state envelope; report-only. |
| `pipeline/check_drift.py` | +1 check: watchlist file is report-only / unread by allocator. |

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
