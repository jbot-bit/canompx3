# Pipeline Correctness Audit — 2026-02-27

**Scope:** Read-only audit of the full chain from outcome computation → strategy discovery → strategy validation → edge families. Goal: verify that validated setups are actually what they claim to be.

**Auditor:** Claude (automated code audit)
**Files inspected:** outcome_builder.py, entry_rules.py, config.py, cost_model.py, strategy_discovery.py, strategy_validator.py, execution_engine.py, build_edge_families.py, rolling_portfolio.py, plus cross-codebase JOIN/filter safety grep.

---

## Conviction Report

### CRITICAL Findings

#### C1. Backtest/Live Feature Parity — IB-Conditional Exit Logic Missing from Backtest

| Field | Detail |
|-------|--------|
| **File** | `trading_app/execution_engine.py:588-611` |
| **Category** | Feature parity divergence |
| **What backtest does** | outcome_builder.py computes outcomes with fixed target/stop — no IB (Initial Balance) conditioning |
| **What live does** | execution_engine.py checks `SESSION_EXIT_MODE` (line 589). If `"ib_conditional"`, trade behavior changes drastically: aligned IB → `hold_7h` mode (target removed), opposed IB → immediate REJECT, pending IB → deferred decision |
| **Impact** | Any strategy validated on a session with `ib_conditional` exit mode has backtest outcomes that DO NOT reflect live behavior. Validated edge was measured under fixed-target assumptions; live execution uses a completely different exit mechanism. |
| **Fix type** | CODE — either implement IB logic in outcome_builder.py or remove IB conditioning from execution_engine.py until backtest parity is achieved |

#### C2. Entry Logic Duplication — execution_engine.py Reimplements Instead of Sharing

| Field | Detail |
|-------|--------|
| **File** | `trading_app/execution_engine.py` (E1: ~line 629-660, E2: ~line 570-627, E3: scattered) |
| **Category** | Feature parity risk |
| **What backtest does** | outcome_builder.py → entry_rules.py shared functions (`detect_confirm`, `_resolve_e1`, `_resolve_e2`, `_resolve_e3`) |
| **What live does** | execution_engine.py has its OWN implementation of entry detection and confirmation logic, NOT calling entry_rules.py |
| **Impact** | Any future change to entry_rules.py (e.g., fixing a confirm bar edge case) will NOT propagate to live execution. Two implementations WILL inevitably diverge. Current logic appears functionally equivalent on inspection, but this is a ticking time bomb. |
| **Fix type** | CODE — refactor execution_engine.py to call entry_rules.py functions |

#### C3. build_edge_families.py — CV Fails Silently on Negative Expectancy

| Field | Detail |
|-------|--------|
| **File** | `scripts/tools/build_edge_families.py:255` |
| **Code** | `cv_expr = std_expr / mean_expr if mean_expr > 0 else None` |
| **Category** | Silent data loss |
| **What happens** | Families with zero or negative mean expectancy get `cv_expr = None`. The `classify_family()` function then cannot use CV for robustness classification, effectively giving these families a free pass on the CV check. |
| **Impact** | A family of strategies where most members have negative expectancy (but the median head is positive) bypasses the CV-based robustness gate. This is the exact scenario CV should flag — high variance including negative members. |
| **Fix type** | CODE — use `abs(mean_expr)` as denominator, or assign `cv_expr = float('inf')` when mean ≤ 0 to ensure the family is flagged as high-variance |

#### C4. No FK Constraint on edge_families.head_strategy_id

| Field | Detail |
|-------|--------|
| **File** | `scripts/tools/build_edge_families.py:164` |
| **Schema** | `head_strategy_id TEXT NOT NULL` — no REFERENCES clause |
| **Category** | Referential integrity gap |
| **What happens** | When validated_setups are purged (e.g., by strategy_validator re-run), edge_families rows can point to non-existent strategies. |
| **Impact** | Downstream consumers (live_config, dashboard, reports) may reference ghost strategies. Queries joining edge_families → validated_setups will silently drop families with orphaned heads. |
| **Fix type** | CODE + SCHEMA — add FK constraint or add a drift check that detects orphaned head_strategy_id values |

---

### HIGH Findings

#### H1. build_edge_families.py — Sample vs Population Stdev for CV

| Field | Detail |
|-------|--------|
| **File** | `scripts/tools/build_edge_families.py:253` |
| **Code** | `std_expr = statistics.stdev(exprs)` |
| **Category** | Statistical bias |
| **What happens** | Uses `statistics.stdev()` which divides by (N-1). For small families (2-4 members), this inflates CV by 20-40% compared to `statistics.pstdev()` (divides by N). |
| **Impact** | Small families are penalized disproportionately in robustness classification. A 2-member family's CV is inflated by 41% (√2 vs √1). This could cause legitimate small families to fail robustness checks. |
| **Fix type** | CODE — use `statistics.pstdev(exprs)` since we're measuring the actual population of family members, not sampling from a larger population |

#### H2. strategy_validator.py — Missing Guard for wf_result=None + error=None

| Field | Detail |
|-------|--------|
| **File** | `trading_app/strategy_validator.py:771-776` |
| **Code** | See logic at lines 771-776 — checks `wr.get("error")` then `wr.get("wf_result")` |
| **Category** | Validation bypass |
| **What happens** | If a walkforward worker returns `{"error": None, "wf_result": None}` (e.g., worker crashed silently, returned empty dict, or timeout produced no result), NEITHER the error branch NOR the wf_result branch triggers. The strategy retains its pre-walkforward status (PASSED from serial phases). |
| **Impact** | A strategy could pass validation without actually completing walkforward testing. The gap exists because the code assumes `error` and `wf_result` are mutually exclusive and exhaustive — but a third state (both None) is possible. |
| **Fix type** | CODE — add `else` clause: if neither error nor wf_result, set status="REJECTED" with notes="Phase 4b: No walkforward result received" |

---

### MEDIUM Findings

#### M1. rolling_portfolio.py — DOW Stats Query Missing orb_minutes Filter

| Field | Detail |
|-------|--------|
| **File** | `trading_app/rolling_portfolio.py:314-324` |
| **Category** | Filter leakage (reporting only) |
| **What happens** | `compute_day_of_week_stats()` queries orb_outcomes filtering by symbol, orb_label, and entry_model — but NOT orb_minutes. Meanwhile, daily_features eligibility is loaded only for `orb_minutes = 5` (line 281). Outcomes for 15m and 30m ORB are pulled and averaged together, with eligibility checked only against 5m ORB size. |
| **Impact** | DOW stats in rolling portfolio reports mix outcomes across different ORB apertures and apply the wrong filter eligibility. This affects reporting/analysis only — does NOT affect validated_setups or live trading. |
| **Fix type** | CODE — either filter orb_outcomes by orb_minutes per family member, or load daily_features for all orb_minutes and check eligibility per-aperture |

#### M2. strategy_validator.py — Ambiguous Notes on Phase A + Phase 4b Rejection

| Field | Detail |
|-------|--------|
| **File** | `trading_app/strategy_validator.py:769,784` |
| **Category** | Observability |
| **What happens** | When a strategy is rejected in Phase A (serial) and then also fails Phase 4b (walkforward), the notes field gets overwritten with the Phase 4b reason. The Phase A rejection reason is lost. |
| **Impact** | Audit trail is degraded — cannot determine the original reason for rejection without re-running. Low operational risk since the strategy is rejected either way. |
| **Fix type** | CODE — append Phase 4b notes instead of overwriting: `notes = f"{notes}; {wf_notes}"` |

#### M3. Execution Engine Confirm Bar Logic — Functionally Equivalent but Reimplemented

| Field | Detail |
|-------|--------|
| **File** | `trading_app/execution_engine.py` |
| **Category** | Code duplication |
| **What happens** | Confirm bar counting in execution_engine.py reimplements the consecutive-closes-outside-ORB logic from entry_rules.py. On inspection, the logic appears functionally equivalent. |
| **Impact** | Low risk today, but any future change to confirmation logic must be made in TWO places. See C2 — this is a symptom of the broader duplication issue. |
| **Fix type** | CODE — addressed by fixing C2 (shared entry_rules.py) |

---

### LOW Findings

#### L1. strategy_validator.py — Redundant orb_minutes in JOIN Clause

| Field | Detail |
|-------|--------|
| **File** | `trading_app/strategy_validator.py:~231` |
| **Category** | Code clarity |
| **What happens** | A JOIN clause includes orb_minutes matching that is already guaranteed by the WHERE filter on the outer query. Not harmful, but adds confusion. |
| **Impact** | None — query is correct, just redundant. |
| **Fix type** | Optional CODE cleanup |

---

## Clean Categories (Verified Correct)

These areas were audited and found to be functioning correctly:

| Component | What Was Checked | Verdict |
|-----------|-----------------|---------|
| **outcome_builder.py grid iteration** | E1 × CB1-5, E2 × CB1 only, E3 × CB1 only | CORRECT — matches TRADING_RULES.md |
| **outcome_builder.py ambiguous bar handling** | Fill-bar where both target AND stop are hit | CORRECT — conservatively scored as loss |
| **outcome_builder.py time-stop** | C8 early exit, C3 session-end exit | CORRECT — per-session EARLY_EXIT_MINUTES from config |
| **strategy_discovery.py triple-join** | orb_minutes in WHERE clause on daily_features | CORRECT — no cross-contamination |
| **strategy_discovery.py metric computation** | ExpR, Sharpe, WR, total_r formulas | CORRECT |
| **strategy_discovery.py filter application** | matching_day_set precomputed, intersected with break days | CORRECT — no filter leakage |
| **strategy_discovery.py write scope** | DELETE/INSERT on experimental_strategies | CORRECT — scoped to instrument + orb_minutes being processed |
| **strategy_validator.py FDR** | Benjamini-Hochberg implementation | CORRECT |
| **strategy_validator.py DELETE scope** | Scoped to processed_orb_minutes | CORRECT — fixed after Feb 2026 incident |
| **strategy_validator.py Phases 1-6** | All serial validation phases | CORRECT |
| **cost_model.py instrument costs** | MGC=$8.40/RT, MNQ=$2.74/RT, MES=$3.74/RT, M2K=$3.24/RT | CORRECT — matches TRADING_RULES.md |
| **cost_model.py R-multiple formulas** | `to_r_multiple()` deducts friction; `pnl_points_to_r()` doesn't | CORRECT — intentional design |
| **build_edge_families.py source query** | Uses validated_setups (not experimental_strategies) | CORRECT |
| **build_edge_families.py write scope** | DELETE/INSERT on edge_families | CORRECT — scoped to instrument |
| **Codebase-wide triple-join audit** | 11 files checked for daily_features joins | CLEAN — all include orb_minutes |

---

## Summary

| Severity | Count | Core Validation Impact |
|----------|-------|----------------------|
| CRITICAL | 4 | C1 and C2 affect live execution parity; C3 and C4 affect edge family integrity |
| HIGH | 2 | H1 biases small family classification; H2 is a rare but exploitable validation bypass |
| MEDIUM | 3 | M1 affects reporting only; M2-M3 are observability/duplication issues |
| LOW | 1 | Cosmetic |

**Bottom line on "are validated setups what they say they are?":**

The core validation chain (outcome_builder → strategy_discovery → strategy_validator) is **fundamentally sound**. Grid iteration, filter application, triple-join safety, FDR correction, and the 6-phase serial validation are all correct. The validated setups themselves accurately reflect the backtest outcomes they were derived from.

The **two critical concerns** are:

1. **Live execution will NOT match backtest** for IB-conditional sessions (C1+C2). The edge was measured under one set of rules but will be traded under another. This doesn't mean validated setups are wrong — it means the live execution path diverges from what was validated.

2. **Edge family integrity has gaps** (C3+C4). Families with negative-mean members can bypass CV checks, and orphaned head strategies can persist after purges. This affects portfolio construction downstream of validation, not the validation itself.
