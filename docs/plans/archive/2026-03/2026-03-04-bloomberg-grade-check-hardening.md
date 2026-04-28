---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Bloomberg-Grade Check Hardening

**Date:** 2026-03-04
**Status:** Design approved, implementation in progress
**Goal:** Every check proves it works. No misleading output. No untested guardrails.

---

## Current State (Verified)

| System | Total Checks | Tested | Coverage |
|--------|-------------|--------|----------|
| `check_drift.py` | 57 | 12 | 21% |
| `audit_behavioral.py` | 5 | 0 | 0% |
| `audit_integrity.py` | 17 | 0 | 0% |
| **TOTAL** | **79** | **12** | **15%** |

### Misleading Output Issues
- audit_integrity checks 7/8/9/10/13/14/17: `return []` always — print data via `_print_informational()` but check functions do nothing
- check_drift checks 40/41: print warnings but return `[]` = "PASSED [OK]"
- DB-dependent checks: silently skip when DB unavailable, show "PASSED [OK]"
- Behavioral audit check 3: only scans `health_check.py` and `audit_*.py`, misses `trading_app/` and `pipeline/`

---

## Work Stream 1: Fix Misleading Output

### 1A. Remove fake checks from audit_integrity registry
Checks 7/8/9/10/13/14/17 always `return []`. They are informational displays, not checks.
- Remove from `CHECKS` registry
- Keep data display in `_print_informational()`
- Update final count: "all {N} checks clean" reflects REAL checks only

### 1B. Fix advisory checks in check_drift
Checks 40 (WF coverage) and 41 (data years) always return `[]`.
- Add `is_advisory` flag to CHECKS registry tuples
- Print "ADVISORY" instead of "PASSED [OK]" for advisory checks
- Track advisory count separately in summary

### 1C. Track and report DB skips
DB-dependent checks silently skip when DB is locked/unavailable.
- Add skip tracking: when a check can't connect, increment skip_count
- Print "SKIPPED (DB unavailable)" instead of "PASSED [OK]"
- Summary: "N checks passed, M skipped, K failed"

### 1D. Broaden behavioral audit scope
Check 3 (broad except) `EXCEPT_SCAN_GLOBS` only covers 2 patterns.
- Add `trading_app/*.py`, `pipeline/*.py` to scan globs
- Self-exclude `audit_behavioral.py` (it uses broad except legitimately in CLI arg check)

---

## Work Stream 2: Test Every Check

### Priority order (highest impact first)

**Tier 1 — Data integrity checks (DB-dependent)**
Each test creates a temp DuckDB, injects data, verifies check catches violations.
- check_no_e0_in_db (35)
- check_doc_stats_consistency (36)
- check_orphaned_validated_strategies (42)
- check_uncovered_fdr_strategies (43)
- check_validated_filters_registered (29)
- check_audit_columns_populated (50)
- check_live_config_spec_validity (54)
- check_cost_model_field_ranges (55)
- check_session_resolver_sanity (56)
- check_daily_features_row_integrity (57)

**Tier 2 — Static code analysis checks**
Each test creates temp files with violations, verifies detection.
- check_schema_query_consistency (4)
- check_import_cycles (5)
- check_hardcoded_paths (6)
- check_connection_leaks (7)
- check_dashboard_readonly (8)
- check_entry_models_sync (13)
- check_entry_price_sanity (14)
- check_nested_isolation (15)
- check_all_imports_resolve (16)
- check_nested_production_writes (17)
- check_schema_query_consistency_trading_app (18)
- check_timezone_hygiene (19)
- check_market_state_readonly (20)
- check_sharpe_ann_presence (21)
- check_ingest_authority_notice (22)
- check_validation_gate_existence (24)
- check_naive_datetime (25)
- check_dst_session_coverage (26)
- check_db_config_usage (27)
- check_e2_e3_cb1_only (30)
- check_orb_minutes_in_strategy_id (31)
- check_orb_labels_session_catalog_sync (32)
- check_stale_session_names_in_code (33)
- check_sql_adapter_validation_sync (34)
- check_old_session_names (38)
- check_variant_selection_metric (44)
- check_research_provenance_annotations (45)
- check_cost_model_completeness (46)
- check_trading_rules_authority (47)
- check_stale_scratch_db (37)
- check_ml_config_canonical_sources (48)
- check_ml_lookahead_blacklist (49)
- check_ml_model_files_exist (51)
- check_ml_config_hash_match (52)
- check_ml_model_freshness (53)

**Tier 3 — Behavioral audit checks**
- check_hardcoded_check_counts
- check_hardcoded_instrument_lists
- check_broad_except_success
- check_cli_arg_drift
- check_triple_join_guard

**Tier 4 — Integrity audit checks (10 real checks)**
- check_outcome_coverage
- check_validated_session_integrity
- check_edge_family_integrity
- check_e0_contamination
- check_old_session_names (integrity version)
- check_e0_cb2_contamination
- check_dead_instrument_contamination
- check_duplicate_strategy_ids
- check_win_rate_sanity
- check_negative_expectancy

---

## Work Stream 3: Expand Coverage

### 3A. double_break look-ahead scanner
New behavioral audit check: scan all Python files in `pipeline/`, `trading_app/`, `scripts/tools/`, `research/` for `double_break` used as a filter/predictor (not just a column reference).
- Flag: `WHERE double_break`, `if.*double_break`, `df[.*double_break.*]` used in filter context
- Allowlist: `audit_behavioral.py` (self), test files, docs

### 3B. Data continuity check (new drift check)
New check: verify no unexpected gaps in trading days per instrument.
- Query daily_features for each active instrument
- Flag gaps > 5 business days (holidays cause 1-3 day gaps normally)
- Informational only (don't block — market closures are legitimate)

### 3C. Cross-reference triple-join in Python (not just SQL)
Current check 5 in audit_behavioral only catches SQL `JOIN daily_features`.
- Also catch DataFrame merges: `pd.merge(.*daily_features` or `.merge(.*daily_features`
- Verify `orb_minutes` appears in the merge call

---

## Implementation Sequence

1. Work Stream 1 first (fixes misleading output — highest impact per line of code)
2. Work Stream 3 next (new checks expand coverage)
3. Work Stream 2 last (tests — most code to write, but checks already work)

Each stream: implement → run all checks → verify no regressions → commit.
