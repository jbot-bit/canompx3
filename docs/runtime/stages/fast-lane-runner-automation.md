---
task: FAST_LANE v5.1 runner automation — template-aware verdict branch in chordia_strict_unlock_v1
mode: IMPLEMENTATION
slug: fast-lane-runner-automation
started: 2026-05-18T00:00:00+10:00
---

## Task

Make `research/chordia_strict_unlock_v1.py` template-aware. When prereg declares
`metadata.template_version: fast_lane_v5.1`, the runner additionally computes
and appends a FAST_LANE verdict (PROMOTE / NEEDS-MORE / KILL) per the v5.1 gate
table at `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml`. Heavyweight
Chordia verdict is emitted unchanged. Fail-closed on unknown template_version.

## Mode

IMPLEMENTATION (single stage, no sub-stages).

## Scope Lock

- research/chordia_strict_unlock_v1.py
- tests/test_research/test_chordia_strict_unlock_v1_fast_lane.py
- pipeline/check_drift.py

## Blast Radius

- `research/chordia_strict_unlock_v1.py` — extend `Cell` dataclass with `template_version`; add `_fast_lane_verdict_v5_1()` pure function; extend `_write_markdown()` with optional FAST_LANE block; extend stdout to print both verdicts. ZERO Python callers in `trading_app/` or `pipeline/` (subprocess-invoked only). Heavyweight `_verdict()` UNCHANGED. Existing emissions tests in `test_chordia_strict_unlock_v1_emissions.py` UNCHANGED.
- `tests/test_research/test_chordia_strict_unlock_v1_fast_lane.py` — NEW file. Covers 6 v5.1 gates × verdict outcomes + 4 edge cases (holdout fail, missing per-direction, fire-rate violation, single-direction bypass).
- `pipeline/check_drift.py` — add `check_fast_lane_runner_template_routing()` and register it in the check tuple. Sentinel-date gate: only applies to v5.1 result MDs dated >= 2026-05-20 (grandfathers the existing manual `2026-05-18-mnq-usdata1000-...` instance). Reads: `docs/audit/hypotheses/` + `docs/audit/results/`. Writes: none.
- DATA LAYER: ZERO writes. Read-only `gold.db` via `_load_universe` (unchanged). No writes to `experimental_strategies`, `validated_setups`, `chordia_audit_log.yaml`, `lane_allocation.json`.
- DOCTRINE INVARIANT: PROMOTE means "worth heavyweight Chordia review", never deploy. No auto-deployment path. Per `TEMPLATE-fast-lane-v5.1.yaml` lines 8-9, 191-194.

## Done criteria

1. New fast-lane tests pass (≥18 gate-cases + 4 edge cases).
2. Existing emissions tests still pass.
3. `python pipeline/check_drift.py` passes.
4. New drift check lights up on injection (manually corrupt the new v5.1 result MD → check fails).
5. Re-run existing `2026-05-18-mnq-usdata1000-...` cell produces FAST_LANE block matching the operator-written PROMOTE (grandfathered file gets the automated block on next run; drift check sentinel skips it pre-2026-05-20 anyway).

## Design answers (locked)

- Unknown `template_version` → FAIL-CLOSED (SystemExit with explicit message).
- Fire-rate gate failure → result MD carries a 1-line "degenerate filter, not t-stat failure" diagnostic note.
- Drift check sentinel = 2026-05-20 (grandfathers existing manual v5.1 result).
