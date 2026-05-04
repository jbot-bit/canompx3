# MES mirror of MNQ US_DATA_1000 VWAP_MID_ALIGNED long — K=2 scan

**Generated:** 2026-04-18T16:00:11+00:00
**Pre-reg:** `docs/audit/hypotheses/2026-04-19-mes-mnq-mirror-vwap-mid-aligned-us-data-1000-v1.yaml` (LOCKED, commit_sha=4fd08031)
**Script:** `research/mes_mnq_mirror_v1_scan.py`
**IS window:** `trading_day < 2026-01-01` (Mode A, from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`)

## Summary

Cells: 2 | CONTINUE: 0 | KILL: 2

**K2 baseline sanity smoke-test:** PASS (same-path reproducibility only — not a genuine implementation-drift cross-check; see Post-scan corrections addendum below).

## Per-cell IS results

| Cell | N_on | ExpR_on | WR_on | ExpR_base | Δ_IS | t | raw_p | boot_p | q_family | years+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1_RR1.5 | 492 | 0.0617 | 0.465 | -0.0012 | 0.0629 | 1.197 | 0.2317 | 0.2012 | 0.4634 | 5/7 |
| H2_RR2.0 | 443 | 0.0217 | 0.375 | -0.0603 | 0.0820 | 0.344 | 0.7311 | 0.7041 | 0.7311 | 5/7 |

## Gate breakdown

| Cell | bh_pass_family | abs_t_IS_ge_3 | N_IS_on_ge_100 | years_positive_ge_4_of_7 | bootstrap_p_lt_0.10 | ExpR_on_IS_gt_0 | not_tautology | not_extreme_fire | not_arithmetic_only | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H1_RR1.5 | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | KILL |
| H2_RR2.0 | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | KILL |

## Flags

| Cell | fire_rate | T0 |corr| | tautology | extreme_fire | arithmetic_only |
|---|---:|---:|---|---|---|
| H1_RR1.5 | 0.589 | 0.067 | N | N | N |
| H2_RR2.0 | 0.584 | 0.066 | N | N | N |

## OOS descriptive (NOT used to select or tune)

| Cell | N_OOS_on | ExpR_on_OOS | Δ_OOS | dir_match |
|---|---:|---:|---:|---|
| H1_RR1.5 | 11 | 0.3113 | 0.1103 | Y |
| H2_RR2.0 | 10 | 0.1592 | 0.1924 | Y |

## Per-year IS breakdown

| Cell | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1_RR1.5 | +0.330 (N=41) | --0.007 (N=94) | --0.081 (N=83) | +0.030 (N=69) | +0.092 (N=64) | +0.190 (N=79) | +0.021 (N=62) |
| H2_RR2.0 | +0.214 (N=37) | +0.063 (N=90) | --0.086 (N=76) | --0.151 (N=60) | +0.011 (N=58) | +0.150 (N=67) | +0.017 (N=55) |

## K2 harness cross-check detail

| Cell | baseline_prereg | baseline_run | match | on_prereg | on_run | match |
|---|---:|---:|---|---:|---:|---|
| H1_RR1.5 | -0.0012 | -0.0012 | Y | 0.0617 | 0.0617 | Y |
| H2_RR2.0 | -0.0603 | -0.0603 | Y | 0.0217 | 0.0217 | Y |

## Decision

**Verdict: KILL per K1.** Zero of 2 cells pass all gate clauses. Honest negative evidence on MES cross-instrument portability of the MNQ VWAP_MID_ALIGNED US_DATA_1000 long signal. No re-runs with different thresholds (forbidden by pre-reg § execution_gate).

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mes_mnq_mirror_v1_scan.py
```

No writes to validated_setups or experimental_strategies. No randomness in IS statistics. Bootstrap p is seeded (seed=42, B=10_000).

---

## Post-scan corrections addendum — 2026-04-19

This scan's implementation had three MEDIUM findings during the 2026-04-19 code review. The primary KILL verdict is UNCHANGED — both cells still fail on t-gate and q-gate — but the implementation is clarified:

### Finding AI #1 — `years_positive_ge_4_of_7` gate semantics

**Original implementation** (prior commit state):
```python
gates["years_positive_ge_4_of_7"] = bool(
    c.per_year_total > 0 and c.per_year_positive / c.per_year_total >= 4.0 / 7.0
)
```
This is a RATIO (4/7) over only-counted years (N ≥ 10 per year). Pre-reg spec is ABSOLUTE: "years_positive_IS >= 4 out of 7 full years".

**Fix** (current commit):
```python
gates["years_positive_ge_4_of_7"] = bool(c.per_year_positive >= 4)
```
Strict absolute count. Years with N < 10 still don't contribute to the positive count (power floor preserved); denominator no longer affects the gate.

**Impact on this scan:** Both cells show `years_positive = 5` → gate passes under BOTH implementations. Not a verdict-change here, but documented for future pre-regs that may have thin-year cells where ratio vs absolute diverges.

### Finding AI #3 — K2 label honest framing

The pre-reg's K2 kill criterion says "Harness canonical-baseline cross-check". The implementation compares baseline values computed by `filter_signal` at pre-reg time vs `filter_signal` at scan time — same code path. That's a sanity smoke-test (does the scan run at all, do numbers reproduce) not a genuine implementation-drift cross-check (which would require an independent second computation method, e.g., direct SQL predicate vs. `filter_signal`).

The original summary label "K2 harness cross-check: PASS" overstated what K2 actually protected against. This addendum and the Summary section have been updated to "K2 baseline sanity smoke-test: PASS". For future pre-regs, see the pre-reg writer template update (Phase 5 of the 2026-04-19 overnight session plan).

**Impact on this scan:** None — K2 still PASS, and the baseline numbers remain as reported. The label change is honest framing only.

### Finding AI #4 — Pathway A theory citation for VWAP_MID_ALIGNED

The pre-reg cites Fitschen Ch 3 as Pathway A grounding. Fitschen grounds intraday trend-follow on equity indices generically. VWAP_MID_ALIGNED specifies a particular ORB-VWAP alignment mechanism not covered in Fitschen Ch 3 specifically. For this KILL outcome the consequence is moot; but future mirror pre-regs should either cite a filter-mechanism-specific source, or explicitly acknowledge the within-class-refinement framing ("Pathway A grounds the ORB entry class; the VWAP filter is a within-class refinement tested on an already-grounded strategy type").

**Impact on this scan:** None (verdict unchanged on KILL).

### Closure status

All three findings addressed. Primary KILL verdict unchanged. For the full findings provenance see commit d285e319 followed by the 2026-04-19 /code-review output; for the pre-reg writer template update addressing AI #3 and AI #4 as institutional policy, see `docs/prompts/prereg-writer-prompt.md` (Phase 5 overnight session commit).

