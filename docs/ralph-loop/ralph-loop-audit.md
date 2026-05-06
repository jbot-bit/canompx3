# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 184

## RALPH AUDIT — Iteration 184
## Date: 2026-05-07
## Infrastructure Gates: 122 drift checks PASS; behavioral audit 7/7 PASS; ruff PASS; 73 tests passed, 1 skipped
## Scope: trading_app/prop_profiles.py (critical, 37 importers, Priority 1)

---

## Iteration 184 — trading_app/prop_profiles.py

### Auto-Targeting
- Priority 1: `trading_app/prop_profiles.py` — critical tier, 37 importers, never scanned
- Priority 0 check: No unresolved CRIT/HIGH in deferred-findings.md or HANDOFF.md at time of targeting

### Infrastructure Gates
- `check_drift.py`: 122 PASS — NO DRIFT DETECTED
- `audit_behavioral.py`: 7/7 PASS
- `ruff`: PASS
- Tests: 73 passed, 1 skipped (test_prop_profiles.py)

---

## File: trading_app/prop_profiles.py

Full scan: dataclasses, canonical sourcing, `_LANE_NAMES`, `get_profile_lane_definitions`, `parse_strategy_id`, `get_lane_registry`, `effective_daily_lanes`.

### Finding PROP-184 — MEDIUM — FIXED

**PREMISE:** `_LANE_NAMES` is a static session-keyed dict whose values (e.g. `"NYSE_CLOSE_VOL"`, `"SING_G8"`) no longer match the `lane_name` format written to `paper_trades.lane_name` by `paper_trade_logger.py`, which uses the dynamic format `"{orb_label}_{filter_type[:12]}"`. Any downstream code consuming `lane_name` from `get_profile_lane_definitions` will get a value that doesn't join to DB records.

**TRACE:** `trading_app/prop_profiles.py:1030` → `lane["lane_name"] = _LANE_NAMES.get(lane.orb_label, lane.orb_label)` → uses stale dict. `trading_app/live/paper_trade_logger.py:77` writes `lane_name = f"{orb_label}_{filter_type[:12]}"` — producer format diverged from consumer.

**EVIDENCE:** Grep confirmed `paper_trade_logger.py` uses dynamic format. `_LANE_NAMES` dict has 8 static entries keyed only on `orb_label`, ignoring `filter_type`. DB records for `NYSE_OPEN` lane with `COST_LT12` filter would have `lane_name="NYSE_OPEN_COST_LT12"` but the dict returned `"NYSE_CLOSE_VOL"` for `NYSE_CLOSE` or `lane.orb_label` fallback for unlisted labels.

**DOCTRINE:** integrity-guardian.md § 4 (Impact awareness — producer/consumer lane_name format divergence); integrity-guardian.md § 6 (No silent failures — stale lookup silently produces wrong value).

**FIX:** Replace `_LANE_NAMES.get(lane.orb_label, lane.orb_label)` with `f"{lane.orb_label}_{parsed['filter_type'][:12]}"`. Mark `_LANE_NAMES` DEPRECATED with backward-compat note for `research/garch_profile_production_replay.py`.

**Diff lines:** 4 production lines changed (well under 20-line cap).

**Test added:** `test_lane_name_is_dynamic_not_static` in `tests/test_trading_app/test_prop_profiles.py` — asserts every lane from `topstep_50k_mnq_auto` profile uses `{orb_label}_{filter_type[:12]}` format.

**Status: FIXED** — Commit `74a8ed63`

---

### Other Patterns Scanned — All Clean

- **Canonical sources:** `ENTRY_MODELS` from `trading_app.config`, `ACTIVE_ORB_INSTRUMENTS` indirectly via `asset_configs`. No hardcoded instrument lists in logic.
- **parse_strategy_id:** Pure string parsing, no canonical violations.
- **Holdout dates:** None present. No `date(2026,...)` patterns.
- **Broad exceptions:** `except Exception` at line ~940 in `load_lane_allocation` logs via `logger.exception` — correct.
- **Silent failures:** `effective_daily_lanes` falls back to `daily_lanes` if JSON missing — intentional with logger.warning. Correct.
- **No-touch zones:** `trading_app/config.py` import only — not modified.

---

## Iteration 184 — Overall Summary

1 file scanned. 1 MEDIUM finding (FIXED). 0 LOW findings.

**Consecutive LOW-only iterations: 0** (reset — MEDIUM finding fixed this iteration)

### Infrastructure Gate Results
- check_drift.py: 122 PASS — NO DRIFT DETECTED
- audit_behavioral.py: 7/7 PASS
- ruff: PASS (auto-formatted test file)
- Tests: 73 passed, 1 skipped

### Action: fix
### Classification: [judgment]
### Commit: 74a8ed63

---

## Files Fully Scanned
- trading_app/live/session_orchestrator.py (iters 173, 174, 175, 176, 177, 178, 182)
- trading_app/live/session_safety_state.py (iters 176, 178)
- tests/test_trading_app/test_session_orchestrator.py (iters 173, 174, 175, 176, 177, 178)
- scripts/infra/telegram_feed.py (iter 173)
- pipeline/db_config.py (iter 179)
- trading_app/holdout_policy.py (iter 179)
- trading_app/hypothesis_loader.py (iter 179)
- pipeline/build_daily_features.py (iter 180)
- trading_app/db_manager.py (iter 180)
- trading_app/lifecycle_state.py (iter 180)
- trading_app/live/projectx/auth.py (iter 180)
- trading_app/live/multi_runner.py (iter 180)
- pipeline/log.py (iter 181)
- pipeline/system_context.py (iter 181)
- pipeline/asset_configs.py (iter 182)
- pipeline/cost_model.py (iter 182, no-touch audit)
- pipeline/dst.py (iter 182, no-touch audit)
- pipeline/paths.py (iter 183)
- trading_app/validated_shelf.py (iter 183)
- trading_app/strategy_fitness.py (iter 183)
- trading_app/prop_profiles.py (iter 184)

## Next Iteration Targets

Priority 1 (unscanned critical, by importer count):
1. `trading_app/outcome_builder.py` (11 importers, critical — SQL no-touch zone for discovery SQL, but non-SQL logic auditable)
2. `trading_app/strategy_discovery.py` (10 importers, critical — SQL no-touch zone)
3. `trading_app/strategy_validator.py` (critical — SQL no-touch zone for SQL logic)
