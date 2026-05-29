---
task: Live capital evidence gate - make missing execution evidence impossible to ignore
mode: CLOSED
slug: 2026-05-23-live-capital-evidence-gate
risk_tier: high
---

## Scope Lock

- `scripts/tools/project_pulse.py`
- `scripts/run_live_session.py`
- `tests/test_tools/test_project_pulse.py`
- `tests/test_scripts/test_run_live_session_preflight.py`
- this stage file

## Purpose

Harden the live capital operating loop without adding strategy discovery. Current deployed-lane truth must be read from profile/allocation code and current `gold.db`; execution attribution must be read from `paper_trades` plus readable `live_journal.db/live_trades`; Criterion 11 and Criterion 12 state must be read through existing lifecycle/control-state readers.

## Required Audit First

1. Current deployed/active lane source: `trading_app.prop_profiles.get_profile_lane_definitions()` backed by profile `daily_lanes` or allocation JSON.
2. Trade attribution sources: `gold.db.paper_trades` and, when readable, `live_journal.db/live_trades`.
3. Project pulse capital checks: `scripts/tools/project_pulse.py::collect_deployment_state()` and `collect_lifecycle_control()`.
4. Account survival source: `trading_app.account_survival` via `trading_app.lifecycle_state`.
5. SR state source: `data/state/sr_state.json` via `trading_app.lifecycle_state`.
6. Exact missing strategy IDs must be computed live from deployed IDs minus observed `paper_trades` or `live_trades` groups.

## Implementation

- Add a read-only execution-evidence summary to `project_pulse`.
- Emit a `broken/high` item when any deployed lane has zero execution rows across `paper_trades` and readable `live_trades`.
- Surface live-journal read status so journal evidence gaps are not hidden.
- Promote missing/invalid Criterion 12 state to a `broken/high` pulse item.
- Add live preflight checks for Criterion 11 survival state and Criterion 12 SR state.
- Keep `--signal-only` usable; it is the path that accumulates monitoring evidence.
- Harden capital recommendation boundaries so unknown/report-only capital packets cannot produce `PROMOTE`.
- Treat unreadable/malformed capital packets as broken evidence instead of silently absent.
- Compute execution staleness from the combined latest paper/live evidence row.

## Boundaries

- No strategy discovery.
- No allocator/profile mutation.
- No DB schema change.
- No DB writes from `project_pulse` or live preflight.
- No automatic `paper_trade_logger` backfill from pulse.
- Stop if a source of truth is unclear.

## Verification

- `python -m pytest tests/test_tools/test_project_pulse.py -q` - PASS, 82 passed after the audit improvement and review hardening.
- `python -m pytest tests/test_scripts/test_run_live_session_preflight.py tests/test_scripts/test_run_live_session_telemetry_maturity.py -q` - PASS, 38 passed.
- `uv run python -m pytest tests/test_tools/test_project_pulse.py tests/test_scripts/test_run_live_session_preflight.py tests/test_scripts/test_run_live_session_telemetry_maturity.py -q` - PASS, 120 passed on Python 3.13.9 after the live-journal source improvement and review hardening.
- `python -m py_compile scripts/tools/project_pulse.py scripts/run_live_session.py` - PASS.
- `ruff check scripts/tools/project_pulse.py scripts/run_live_session.py tests/test_tools/test_project_pulse.py tests/test_scripts/test_run_live_session_preflight.py` - PASS.
- `ruff format --check pipeline/ trading_app/ scripts/` - PASS after formatting `scripts/tools/project_pulse.py`.
- `python pipeline/check_drift.py` - PASS, no drift detected.
- `python scripts/tools/audit_behavioral.py` - PASS, all 7 checks clean.
- `$env:PYTHONPATH=(Get-Location).Path; python scripts/tools/audit_integrity.py` - PASS, all integrity checks clean.
- `python scripts/tools/project_pulse.py --fast --format json` - expected exit 1 because new capital evidence gates identify broken live-state evidence.
- `python scripts/tools/project_pulse.py --fast` - expected exit 1; cockpit shows execution evidence, C11 block, C12 block, packet boundary, and NO_CHANGE recommendation.

Fresh rerun note after final formatting/review hardening:

- `python -m pytest tests/test_tools/test_project_pulse.py -q` - PASS, 82 passed.
- `python -m pytest tests/test_scripts/test_run_live_session_preflight.py tests/test_scripts/test_run_live_session_telemetry_maturity.py -q` - PASS, 38 passed.
- `python -m py_compile scripts/tools/project_pulse.py scripts/run_live_session.py` - PASS.
- `ruff check scripts/tools/project_pulse.py scripts/run_live_session.py tests/test_tools/test_project_pulse.py tests/test_scripts/test_run_live_session_preflight.py` - PASS.
- `ruff format --check pipeline/ trading_app/ scripts/ tests/test_tools/test_project_pulse.py tests/test_scripts/test_run_live_session_preflight.py` - PASS, 441 files already formatted.
- `git diff --check` - PASS.
- `python scripts/tools/audit_behavioral.py` - PASS.
- `python pipeline/check_drift.py` - BLOCKED by active `outcome_builder.py --instrument MNQ --force ...` process holding `C:\Users\joshd\canompx3\gold.db` (PID 47848); non-DB checks passed, DB-dependent checks skipped/failed on lock.
- `python scripts/tools/audit_integrity.py` - BLOCKED by the same DB lock on `gold.db`.
- `python scripts/tools/project_pulse.py --fast --format json` and `python scripts/tools/project_pulse.py --fast` - expected exit 1; current run additionally reports DB-locked deployment/execution checks as paused while still blocking on missing Criterion 11 and Criterion 12 state.

Full-suite note:

- `python -m pytest tests/ -x -q` under Python 3.11.9 stops during collection on existing Python 3.12+ `type` alias syntax in `scripts/tools/fast_lane_research_review.py`.
- `py -3.13 -m pytest tests/ -x -q` gets past that syntax but stops on missing `exchange_calendars` in the launcher install.
- `uv run python -m pytest tests/ -x -q` built the locked environment, then exited 137 before pytest output. Targeted impacted tests pass in that same `uv` environment.

## Result

- Current deployed profile: `topstep_50k_mnq_auto`.
- Deployed lanes: 4.
- Execution attribution coverage from `paper_trades` plus readable `live_journal.db/live_trades`: 1/4 lanes.
- Live journal source status: `ok`.
- Missing execution rows:
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
  - `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
- Stale execution attribution:
  - `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
- Criterion 11 survival report: missing/blocking.
- Criterion 12 SR state: missing/blocking.
- Fast-lane packet: report-only boundary, no add/remove.
- Capital recommendation: `NO_CHANGE`.

## Remaining Gaps

- Refresh or backfill execution attribution evidence through the existing supported command paths before trusting deployed-lane performance.
- Generate/refresh the Criterion 11 survival report through the existing survival command path.
- Refresh Criterion 12 SR state through the existing control-state path.
- Do not promote or remove capital until the cockpit recommendation is no longer blocked by execution, survival, or SR evidence.
