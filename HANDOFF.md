# HANDOFF.md - Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done - update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## This Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Summary:** Cleared the `topstep_50k_mnq_auto` live-validity blocker in both this Codex worktree and canonical `C:\Users\joshd\canompx3`. Root cause was strict live allocation using SR state for the current book as if it covered all candidates, allowing `UNKNOWN` SR candidates and old SR-alarm lanes to rotate back in. `rebalance_lanes.py --strict-live-clean` now requires current SR `CONTINUE` evidence, computes correlation only after hard gates, and the allocator caches feature rows so rebalance stays bounded. Canonical allocation is now 3 lanes: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`, `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`. Canonical C11/C12 refreshed green, `live_readiness_report --strict-zero-warn` green with only telemetry maturity advisory, and canonical signal-only preflight passed 13/13.
- **⚠ Truth note (Claude, 2026-05-30, verified):** the `--strict-live-clean` flag and the lane_allocator `feature_cache` opt described above are **NOT committed on any branch** (`git log --all -S` = 0 hits). They live only in `stash@{0}` + `docs/runtime/rescued/2026-05-30-*` (see RESCUE-MANIFEST). Capital-path, Codex-owned — must be committed by its owner, not dropped. The regenerated `lane_allocation*.json` is the OUTPUT of that un-landed code, so canonical `docs/runtime/lane_allocation.json` on HEAD does NOT reflect it.
- **✅ Drift CLEAN — RETRACTION of an earlier false claim (Claude, 2026-05-30, execution-verified):** an earlier version of this note (commit `82721bcc`) claimed `check_active_native_trade_windows_match_provenance` was FAILING on lane `MNQ_COMEX_SETTLE_...OVNRNG_100`. **That was wrong** — I wrote it from a stale memory note without executing, violating Rule 11 (never trust metadata). Direct call returns `VIOLATIONS: 0`; full `check_drift.py --skip-crg-advisory` = **NO DRIFT DETECTED, 170 passed, 0 failed** (incl. Check 191 cold-recheck PASSED). The COMEX_SETTLE lane IS present in canonical `lane_allocation.json` (one of the 3 active lanes). `backfill_validated_trade_windows.py` (live write) = `inspected=848 drifted=0 updated=0`. Trade-window provenance is canonical. No action owed.

## Current Codex Follow-up
- **Tool:** Codex
- **Date:** 2026-06-01
- **Summary:** Implemented the peer-parity EV proof-pack work on isolated branch `codex/ev-proof-pack-harness`: EV-1 bootstrap health artifact CLI (`scripts/tools/bootstrap_health_proof.py`), EV-2 `live_readiness_report.py` profile proof-pack schema/markdown section, and EV-3 bounded benchmark harness (`scripts/tools/bounded_benchmark_harness.py`). Durable artifacts are under `docs/audit/results/2026-06-01-*`.
- **Runtime truth:** Current worktree preflight is not green: `codex-wsl` expected interpreter `.venv-wsl/bin/python` is missing, the tree is dirty before commit, and fast pulse reports 3 broken items. `live_readiness_report --strict-zero-warn` exits nonzero because current Criterion 11/12 evidence is not green. These are recorded in the committed proof JSONs instead of hidden.
- **Verification:** `python -m pytest tests/test_tools/test_session_preflight.py tests/test_tools/test_project_pulse.py tests/test_tools/test_bootstrap_health_proof.py tests/test_tools/test_bounded_benchmark_harness.py tests/test_tools/test_live_readiness_report.py -q` passed 150 tests. `ruff check ... --quiet` and `git diff --check` passed.
- **CI follow-up:** PR #350 first CI run timed out in `Tests with coverage (tools and research)` while `tests/test_tools/test_fast_lane_status.py` was running. Fixed `fast_lane_status` to extract `scope.strategy_id` and `metadata.template_version` with narrow scalar scans instead of full PyYAML loads; live status build dropped from ~24s to ~0.5s locally. Targeted EV/status tests passed 56 tests.
- **CI follow-up 2:** Second PR #350 CI run completed the tools/research shard but failed `tests/test_tools/test_git_hooks_env.py::test_pre_commit_prefers_wsl_venv_before_windows_venv_on_posix_shells` and ended at the 10-minute shard cap. Fixed pre-commit commit-lock Python selection to prefer `.venv-wsl` on POSIX and raised only the tools/research shard timeout to 15 minutes. `python -m pytest tests/test_tools/test_git_hooks_env.py tests/test_tools/test_fast_lane_status.py -q` passed 24 tests.
- **CI follow-up 3:** Third PR #350 CI run passed tools/research and failed fast-lane drift shard on `test_drift_check_fails_on_unrevoked_pooling_artifact`. Fixed `check_fast_lane_promote_orphans()` to flag any `pooling_artifact` with no revocation sidecar regardless of scanner terminal status. Exact failing test now passes locally.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-31
- **Commit:** 2810fead — @ feat(prop): encode MFFU Builder + Flex specs/tiers/payouts; fix Rapid sim-cap leak
- **Files changed:** 6 files
  - `HANDOFF.md`
  - `docs/audit/2026-05-31-mffu-forced-progression-live-cap-memo.md`
  - `docs/runtime/stages/2026-05-31-mffu-builder-prop-rules-stage2.md`
  - `tests/test_prop_profiles_mffu.py`
  - `trading_app/prop_firm_policies.py`

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
