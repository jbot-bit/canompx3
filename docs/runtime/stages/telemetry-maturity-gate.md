---
task: Build automatic live-telemetry maturity gate (30-trading-day floor)
mode: IMPLEMENTATION
slug: telemetry-maturity-gate
created: 2026-05-17
scope_lock:
  - trading_app/live/telemetry_maturity.py
  - tests/test_trading_app/test_telemetry_maturity.py
---

## Task

Create a fail-closed maturity gate on the live-telemetry side. Until at least
30 distinct trading_days of signal-only (or live) bot uptime are recorded for a
given instrument in the canonical signal logs, any live-throughput triage MUST
return verdict UNVERIFIED_INSUFFICIENT_TELEMETRY for that instrument's lanes,
regardless of how many gate fires or zero-fire windows are observed.

## Mode

IMPLEMENTATION (single stage). New module + test file only.

## Grounding (canonical sources)

- docs/institutional/pre_registered_criteria.md Criterion 8 + Amendment 2.7
  (Mode A holdout): OOS gates require N>=30 with power-floor enforcement; below
  threshold, verdict is UNVERIFIED, never DEAD.
- .claude/rules/backtesting-methodology.md RULE 3.2 + 3.3: N_on_OOS<30 floor
  for any binary OOS gate; below floor, statistical power is insufficient to
  distinguish signal from noise.
- trading_app/holdout_policy.HOLDOUT_SACRED_FROM = 2026-01-01 is the policy
  precedent for bright-line fail-closed enforcement (no soft warnings).
- memory/feedback_n_unique_trading_days_floor_clustered_se.md: clustered-SE
  estimator degrades below 30 cluster floor; cell-level UNVERIFIED kill is
  doctrine.

The 30-trading-day floor on telemetry is the operational analog of the C8
power floor: until enough distinct trading_days of uptime exist, the
diagnostic has no statistical basis for any verdict other than
UNVERIFIED_INSUFFICIENT_TELEMETRY.

## Blast Radius

- trading_app/live/telemetry_maturity.py is a NEW module. No callers exist
  yet on the production side. Future caller: a re-run of the live-throughput
  triage diagnostic in docs/runtime/diagnostics/ will import and call
  evaluate_telemetry_maturity() before assigning per-lane verdicts.
- tests/test_trading_app/test_telemetry_maturity.py is a NEW test file
  exercising the gate with synthetic signal log files in tmp_path.
- Reads: signal logs at repo_root/live_signals_YYYY-MM-DD.jsonl per
  trading_app.live.session_orchestrator SIGNALS_DIR convention.
- Writes: none. Pure read+compute module.
- Does NOT touch lane_allocation.json, validated_setups, gate thresholds,
  filter logic, or any allocator field.

## Acceptance

1. python -m pytest tests/test_trading_app/test_telemetry_maturity.py -v
   passes (5 tests).
2. Mutation probe: synthesize 29 distinct trading_day SESSION_START records
   for MNQ -> verdict UNVERIFIED_INSUFFICIENT_TELEMETRY; add one more day
   -> verdict TELEMETRY_MATURE.
3. python pipeline/check_drift.py exit 0.
4. Module exports a single public function and a single public constant.
   No production callers are wired in this stage (caller wiring is a
   separate scope) -- the goal is the canonical primitive only.
