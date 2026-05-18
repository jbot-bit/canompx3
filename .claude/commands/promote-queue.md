---
description: FAST_LANE v5.1 PROMOTE-queue scanner — reconstructs queue state from result MDs + revocation sidecars + heavyweight preregs + action-queue. Read-only by default.
allowed-tools: Bash
---

# /promote-queue — FAST_LANE v5.1 PROMOTE queue scanner

Reconstruct the current PROMOTE-queue state from on-disk artifacts. The queue is **derived state** — never hand-edit `docs/runtime/promote_queue.yaml`; the drift check (#157) reconstructs from source and fails on cache disagreement.

## Usage

- `/promote-queue` — dry-run (default). Prints the would-be queue and diff vs cache. Does not write.
- `/promote-queue --write` — refresh `docs/runtime/promote_queue.yaml` from current on-disk state.
- `/promote-queue --json` — emit JSON to stdout (for piping / drift-check integration).
- `/promote-queue --write --json` — refresh cache and emit JSON.

Path overrides (rare): `--results-dir`, `--cache-path`, `--hypotheses-dir`, `--action-queue`.

## Status enum (exhaustive — no UNKNOWN)

- **QUEUED** — PROMOTE + no revocation sidecar + no heavyweight prereg + no park entry + per-direction sanity gate PASS. Awaits heavyweight authoring.
- **ESCALATED** — PROMOTE + matching heavyweight prereg under `docs/audit/hypotheses/`. Already in the institutional pipeline.
- **REVOKED** — PROMOTE + `<base>.revocation.md` sidecar present. Pooled finding refuted (typically sample-doubling artifact caught by per-direction sanity gate).
- **PARKED** — PROMOTE + `docs/runtime/action-queue.yaml` entry naming this strategy_id with park.
- **ERROR** — PROMOTE + (missing/unparseable directional breakdown OR per-direction sanity gate fires `REVOKE_RECOMMENDED` with no revocation sidecar yet). Forces operator attention.

REVOKED beats ESCALATED in precedence.

## Per-direction sanity gate

Re-applies v5.1 thresholds (t≥2.5, N≥50, fire-rate ∈ [0.05, 0.95]) to the `## Directional breakdown` table every v5.1 result MD carries (PR #300 emissions). When a pooled PROMOTE has BOTH per-direction sub-stats failing standalone, the pooled PROMOTE is a sample-doubling artifact → `REVOKE_RECOMMENDED`. Zero new K spent — re-inspection of stats already emitted.

## Exit codes

- `0` — clean state (no ERROR entries).
- `2` — at least one ERROR entry. Triggers drift check #157 if cache is stale or hand-edited.

## Implementation

```bash
.venv/Scripts/python.exe scripts/research/fast_lane_promote_queue.py $ARGUMENTS
```

## Workflow

1. **After a FAST_LANE v5.1 run emits new PROMOTE result MDs:** run `/promote-queue` to see what landed; if all entries are QUEUED/ESCALATED/REVOKED/PARKED, follow with `/promote-queue --write` to refresh the cache.
2. **ERROR entry?** Inspect the result MD's `## Directional breakdown`. If the sanity gate fired `REVOKE_RECOMMENDED`, author a `.revocation.md` sidecar next to the result MD; next scan moves it to REVOKED. If the breakdown is unparseable, fix the result MD source.
3. **QUEUED entry ready for heavyweight prereg?** Operator authors the prereg manually (no auto-generation, no fabricated theory citation). Once the prereg lands under `docs/audit/hypotheses/` with matching `scope.strategy_id`, next scan moves the entry to ESCALATED.

## Related

- `scripts/research/fast_lane_promote_queue.py` — the scanner itself (full source, full CLI surface via `--help`).
- `pipeline/check_drift.py::check_fast_lane_promote_orphans` — drift check #157; pre-commit + CI enforce.
- `docs/runtime/promote_queue.yaml` — DERIVED cache (never hand-edit).
- `docs/audit/results/2026-05-18-heavyweight-candidate-pack.md` — evidence pack pattern for QUEUED entries awaiting heavyweight authoring.
- `docs/audit/results/*.revocation.md` — sidecar pattern for REVOKED entries.
