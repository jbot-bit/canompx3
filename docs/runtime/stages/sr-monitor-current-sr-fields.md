---
task: SR-monitor + lane-overrides hygiene fixes (Stages 4, 5, 6, 7)
mode: IMPLEMENTATION
scope_lock:
  - data/state/lane_overrides_topstep_50k_mnq_auto.json
  - trading_app/sr_monitor.py
  - tests/test_trading_app/test_sr_monitor.py
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift.py
  - docs/runtime/sr_monitor_workflow.md
  - memory/feedback_sr_monitor_peak_vs_current_misread.md
  - C:/Users/joshd/.claude/projects/C--Users-joshd-canompx3/memory/MEMORY.md
  - .claude/settings.local.json
  - docs/runtime/stages/sr-monitor-current-sr-fields.md
---

## Blast Radius

- `data/state/lane_overrides_topstep_50k_mnq_auto.json` — remove 3 false-positive pause entries on deployed lanes; registry's pre-existing `watch` outcomes will activate via `lifecycle_state.py:236`.
- `trading_app/sr_monitor.py` — additive: emit `current_sr_stat`, `trades_since_alarm`, `recent_10_mean_r` per lane in the results dict. Existing `sr_stat` peak semantic preserved (locked by `test_run_monitor_reports_sr_at_alarm_not_at_stream_end`). Module docstring updated.
- `tests/test_trading_app/test_sr_monitor.py` — new test asserting `current_sr_stat <= sr_stat` on a recovered stream + new fields present.
- `pipeline/check_drift.py` — new advisory check `check_sr_pauses_have_recent_evidence` (conjunction: recovered SR AND trades_since_alarm >= 10 AND recent_10_mean_r > 0).
- `tests/test_pipeline/test_check_drift.py` — fixture-based test for the new drift check.
- `docs/runtime/sr_monitor_workflow.md` — NEW operator-workflow doc (peak vs current SR, registry workflow, --apply-pauses gotcha).
- `memory/feedback_sr_monitor_peak_vs_current_misread.md` — NEW memory entry; MEMORY.md index addition.
- `.claude/settings.local.json` — ~6 read-only allow rules for sr_monitor/lane_ctl/sr_review_registry.

Reads: `gold.db` (read-only via existing sr_monitor path), `sr_review_registry.py` (in-memory dict), `lane_overrides_*.json`. Writes: `sr_state.json` (additive fields), `lane_overrides_topstep_50k_mnq_auto.json` (delete 3 keys).

Non-impact (verified Stage 4 prep):
- `trading_app/lifecycle_state.py:read_criterion12_state` reads via `payload.results[]` and ignores unknown fields.
- `trading_app/lane_allocator.py:load_sr_state` consumes only `status`.
- No `schema_version` bump (additive field only).
