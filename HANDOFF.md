# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-26
- **Commit:** a41dcb50 — Merge remote-tracking branch 'origin/main' into chore/hardening-three-ghost-deployments
- **Files changed:** 16 files
  - `.claude/hooks/session-start.py`
  - `HANDOFF.md`
  - `docs/runtime/stages/debt-burndown-import-side-effects.md`
  - `docs/runtime/stages/hwm-stage1-gap1-none-reason-contract-guard.md`
  - `docs/runtime/stages/hwm-stage2-tracker-integrity.md`
  - `docs/runtime/stages/hwm-stage3-orchestrator-pre-session.md`
  - `docs/runtime/stages/hwm-stage4-inactivity-monitor.md`
  - `docs/runtime/stages/hwm-warning-tier-notify-dispatch.md`
  - `docs/runtime/stages/live-overnight-resilience-hardening.md`
  - `docs/runtime/stages/osr-debt-frame-audit.md`
  - `docs/runtime/stages/ralph-iter-174-f4-bracket-naked.md`
  - `docs/runtime/stages/ralph-iter-179-pass-one-hardening.md`
  - `docs/runtime/stages/ralph-iter-181-r4-signals-rotation.md`
  - `docs/runtime/stages/ralph-iter-185-f7-fill-poller-timeout.md`
  - `docs/runtime/stages/ruff-format-iter179-fixup.md`
  - ... and 1 more

## Stages closed this session (work landed, file deleted)

| Stage file | Marker commit |
|---|---|
| hwm-stage1-gap1-none-reason-contract-guard | `8a40a142` (#129) |
| hwm-stage2-tracker-integrity | `8a40a142` (#129) |
| hwm-stage3-orchestrator-pre-session | `8a40a142` (#129) |
| hwm-stage4-inactivity-monitor | `8a40a142` (#129) |
| hwm-warning-tier-notify-dispatch | `8a40a142` (#129) |
| ralph-iter-174-f4-bracket-naked | `87dffa38` |
| osr-debt-frame-audit | `eaa4e055` (#106) |
| debt-burndown-import-side-effects | `2e9f4145` (#128) |
| live-overnight-resilience-hardening | `e02c529d` (audit-verified `2f45a3e8`) |
| ralph-iter-179-pass-one-hardening | `0d54d52e` (#107) |
| ralph-iter-181-r4-signals-rotation | `36276381` (#110) |
| ralph-iter-185-f7-fill-poller-timeout | `36276381` (#110) |

## Active stages (4 remaining)

1. **`open-pr-review-debt.md`** (TRIVIAL) — 5 deferred PRs (#72, #74, #59, #8, #12) need `evidence-auditor` before merge. PR #36 from this list is now CLOSED → re-extracted as PR #135.
2. **`pass-three-magic-number-drift-check.md`** (IMPLEMENTATION) — **PARALLEL TERMINAL `canompx3-pt`** (PR #124). Do not touch.
3. **`pr48-mgc-shadow-only-overlay-contract.md`** (IMPLEMENTATION) — operator observation pending; do not action without observation result.
4. **`ruff-format-iter179-fixup.md`** (TRIVIAL) — 4 fixes; some may already be obsolete after PR #107/#110. Pre-check each before doing the work.

## Next Steps — Active

1. **PR #135** (DRAFT, mine, `canompx3-ghost` worktree) — re-extract of closed #36. Explicit 5-item TODO in PR body: `pipeline/build_daily_features.py` hook, `pipeline/ingest_statistics.py` docstring, 3 new drift checks in `pipeline/check_drift.py` (~292 LOC), 8 regression tests in `tests/test_pipeline/test_check_drift_db.py` (~156 LOC). Capital-at-risk-adjacent → full IMPLEMENTATION discipline; no `cherry-pick` from closed #36's commits.

2. **PR #133** (DRAFT, mine, `recover/l1-europe-flow-prereg`) — L1 EUROPE_FLOW raw break-quality prereg recovered from `archive/stash-2026-04-26-l1-europe-flow-pr1a-v2`. CI green. Triage: keep-as-DRAFT pending review, or close + retag stash archive.

3. **PR #134** (DRAFT, mine, `recover/l6-us-data-2026-diagnostic`) — L6 MNQ_US_DATA_1000 2026 OOS diagnostic from `archive/stash-2026-04-26-l6-wip-pre-correction`. CI green. Same triage choice as #133.

4. **PR #99** (DRAFT, Codex's `wt-codex-mnq-hiroi-scan`) — research(mnq) MNQ COMEX geometry hardening + COMEX family stamp. Recommended: comment + leave for Codex.

5. **Open PR review debt** (5 PRs) — `#72`, `#74`, `#59`, `#8`, `#12 (stacked on #8)`. Dispatch `evidence-auditor` in parallel; merge PASS, close FAIL with citation. Separate session.

6. **Action queue P1 close-first** — `cross_asset_session_chronology_spec` (write the chronology spec before any cross-asset timing scan/execution).

## Parallel session activity (DO NOT TOUCH)

- **`canompx3-pt`** — `chore/pass-three-drift` (PR #124 OPEN, CI failed; magic-number rationale audit + 9 trading_app/live constants; 69 dirty files). Owner: parallel Claude session.
- **`canompx3-ralph-burndown`** — `ralph/crit-high-burndown-v5.2` (branch marked `[gone]` on remote — likely force-pushed; 19 dirty files; HEAD `6740d938` predates merged Ralph PRs #107/#110). Stale; do not write to it.
- **MGC shadow observation** — operator action; observation result unblocks `mes_q45_exec_bridge` action-queue item.

## Blockers / Warnings

- Memory entries calling F-1 "dormant" are FACTUALLY WRONG — F-1 is **active** and fail-closing in signal-only. Day-1 XFA seeds `_topstep_xfa_eod_balance = $0.0` (canonical per `topstep_scaling_plan.py:51-53`); 6 deployed MNQ lanes ALLOW through `risk_mgr.can_enter()`. Verified empirically (E2E probe in PR #100/B6 work).
- `build_live_portfolio()` is **DEPRECATED** — `--all --signal-only` without `--profile` hard-fails all 3 orchestrators. Only `--profile` path works.
- Drift checks pre-existing: #4 (`work_queue.py` schema parser false-positive on `table 'the'`) and #59 (MGC — 1 day with `!= 3 daily_features` rows). Neither blocking.
- Working-tree debt (not in this commit): `requirements.txt` regenerated by `uv export`, dirty in main worktree. Separate concern; needs its own commit + uv-lock-vs-export reconciliation.
- 57 zombie local branches with no worktree; some have merged PRs (e.g. `chore/orphan-stage-cleanup` = PR #104 merged). Cleanup pass deferred — would benefit from `git branch -vv | rg gone` audit + `clean_gone` command.

## Durable References

- `docs/runtime/action-queue.yaml` — P1/P2 action queue
- `docs/runtime/decision-ledger.md` — durable decisions
- `docs/runtime/debt-ledger.md` — known debt
- `docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md` — HWM Stages 1-4 design (now landed in #129)
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` — O-SR grounding (still pending: cusum_monitor → SR Eq 10)
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` — ORB premise
