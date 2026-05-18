---
task: build deterministic FAST_LANE v5.1 PROMOTE queue scanner + revoke #2 + queue #1 as UNVERIFIED_OOS_POWER
mode: IMPLEMENTATION
scope_lock:
  - scripts/research/fast_lane_promote_queue.py
  - tests/test_scripts/test_fast_lane_promote_queue.py
  - docs/runtime/promote_queue.yaml
  - docs/audit/results/2026-05-18-mnq-comexsettle-e2-rr20-cb1-orbvol16k-pooled-o5-fast-lane-v1.revocation.md
  - docs/audit/results/2026-05-18-mnq-comexsettle-e2-rr20-cb1-orbvol16k-pooled-o5-fast-lane-v1.md
  - docs/audit/results/2026-05-18-heavyweight-candidate-pack.md
  - pipeline/check_drift.py
  - docs/runtime/decision-ledger.md
  - HANDOFF.md
---

## Blast Radius

- `scripts/research/fast_lane_promote_queue.py` — NEW. Read-only scanner. Reads `docs/audit/results/*fast-lane*.md`, `docs/audit/hypotheses/*.yaml`, `docs/runtime/action-queue.yaml`. Zero writes outside `docs/runtime/promote_queue.yaml` (and only on `--write`).
- `tests/test_scripts/test_fast_lane_promote_queue.py` — NEW. 9 tests with synthetic-MD fixtures via `tmp_path`. No DB access.
- `docs/runtime/promote_queue.yaml` — NEW. Derived state file (NOT canonical). Drift check #157 reconstructs and diffs.
- `docs/audit/results/...orbvol16k...revocation.md` — NEW sidecar. Cites per-direction numbers already in original MD (no new research, no new K spent).
- `docs/audit/results/...orbvol16k...fast-lane-v1.md` — APPEND-ONLY pointer to sidecar at bottom of existing FAST_LANE verdict block. Preserves audit trail.
- `docs/audit/results/2026-05-18-heavyweight-candidate-pack.md` — NEW. Markdown evidence pack for Lane #1. theory_citation BLANK. OOS power pre-computed via `research.oos_power.one_sample_power`. Status label: `UNVERIFIED_OOS_POWER`.
- `pipeline/check_drift.py` — ADD `check_fast_lane_promote_orphans` (Check 157). Walks the scanner output; fails if any entry is ERROR, if `promote_queue.yaml` is stale/hand-edited, or if any PROMOTE MD has no row in the rebuilt queue.
- `docs/runtime/decision-ledger.md` — ONE entry for Lane #2 revocation.
- `HANDOFF.md` — ONE line under "Next Session" pointing at queue + revocation.

Reads (read-only): `docs/runtime/action-queue.yaml`, all `*.yaml` under `docs/audit/hypotheses/`, all `*fast-lane*.md` under `docs/audit/results/`. Zero writes to allocator/live/state.

## Why this stage

The FAST_LANE v5.1 runner shipped (stage commit `019889a5`, "fast-lane-runner-automation"). The missing infra: a deterministic queue scanner that prevents orphan PROMOTEs (PROMOTE result MDs with no follow-up). Inline analysis this session also caught one PROMOTE that's a pooling artifact (lane #2 ORB_VOL_16K — both per-direction sub-stats fail v5.1 gates as standalone). The scanner makes that catch deterministic and the revocation auditable.

Per Code Guardian two-pass: PASS 1 audit complete; PASS 2 implementation under this scope.

## Done criteria

1. `pytest tests/test_scripts/test_fast_lane_promote_queue.py -v` green (≥9 tests).
2. `python scripts/research/fast_lane_promote_queue.py` (default `--dry-run`) classifies lane #1 → QUEUED, lane #2 → ERROR (pooling artifact, no revocation sidecar yet).
3. Revocation sidecar lands; rerun scanner; lane #2 → REVOKED.
4. Heavyweight candidate pack for lane #1 written; status header `UNVERIFIED_OOS_POWER` with measured OOS power from `research.oos_power.one_sample_power`.
5. `python pipeline/check_drift.py` passes (modulo the pre-existing MGC_CME_REOPEN_ORB_G4 trade-window carry-over per HANDOFF lines 14-15).
6. Drift check #157 injection test confirms it catches a hand-edited `promote_queue.yaml` (status flip from REVOKED back to QUEUED → drift fail).
7. Single scoped feature commit when all of (1)-(6) green.
