# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-27
- **Commit:** 9dd4e7b5 — chore(stages): close main-ci-preflight stage (PR #143 merged at 574912e2)
- **Files changed:** 2 files
  - `HANDOFF.md`
  - `docs/runtime/stages/main-ci-preflight.md`

## Session decisions (2026-04-26 evening + late-evening sweep)

- **Env hardening landed** (PR #136, merged) — venv pinned to `uv.lock`, `_env_drift_lines()` reports drift at session-start, `requirements.txt` regenerated from lock.
- **Same-worktree mutex landed** (PR #138, merged `e306284b`) — atomic `O_EXCL` lock at `<git-dir>/.claude.pid` prevents two Claudes in one worktree (closes the `cannot lock ref 'HEAD'` race). 4 scenarios tested.
- **Architecture decision: per-session clones REJECTED.** Researched against 2026 industry consensus (Anthropic, Augment Code, Penligent, Upsun) — worktrees + runtime isolation is the recommended pattern. `_discover_git_common_root()` in `pipeline/paths.py:33-65` already implements canonical worktree-shared-DB.
- **Phase 1B / 1C DEFERRED** — cross-worktree push lock and runtime isolation audit, both low-priority/advisory. Not blocking.
- **Hook-output trim DEFERRED** — needs production-code edits to `scripts/tools/task_route_packet.py`; properly stage-gated.
- **Recovery PRs #133/#134 CLOSED** — stash content already preserved at `archive/stash-2026-04-26-l1-europe-flow-pr1a-v2` and `archive/stash-2026-04-26-l6-wip-pre-correction`. PRs closed with archive-tag citations; `recover/l1-europe-flow-prereg` and `recover/l6-us-data-2026-diagnostic` deleted local + remote.
- **Evidence-auditor obligation discharged** — the stage `open-pr-review-debt.md` listed 7 deferred PRs. As of 2026-04-26: #72/#74/#59/#8 CLOSED, #12 MERGED, #36 CLOSED (re-extracted as PR #135 which MERGED). #99 remains DRAFT (Codex). All 6 non-DRAFT entries resolved historically without auditor pass — recorded as discharged, not pending.
- **Conservative zombie sweep** — deleted 4 `[gone]`-flagged local branches (`chore/handover-2026-04-26-pr125-126`, `ralph/burndown-v5.3-rebased`, `ralph/crit-high-burndown-v5.3-high-tier`, `research/pr48-sizer-rule-oos-backtest`). `ralph/crit-high-burndown-v5.2` retained because `canompx3-ralph-burndown` worktree still uses it.

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
| hardening-three-fixes | `290006e5` (#137) |
| open-pr-review-debt | (this PR — obligation discharged historically) |

## Active stages (2 remaining)

1. **`pass-three-magic-number-drift-check.md`** (IMPLEMENTATION) — **PARALLEL TERMINAL `canompx3-pt`** (PR #124). Do not touch.
2. **`pr48-mgc-shadow-only-overlay-contract.md`** (IMPLEMENTATION) — operator observation pending; do not action without observation result.

## Next Steps — Active

1. **PR #99** (DRAFT, Codex's `wt-codex-mnq-hiroi-scan`) — research(mnq) MNQ COMEX geometry hardening + COMEX family stamp. Standing instruction: comment + leave for Codex.

2. **Action queue** — fully closed as of 2026-04-26. `cross_asset_session_chronology_spec` marked CLOSED (spec frozen at `docs/plans/2026-04-25-cross-asset-session-chronology-spec.md` via PR #101 / `87ebc885`); re-opens when a future cross-asset prereg explicitly cites and gates on it. No P1 ready items remain.

3. **Operator action** — MGC shadow observation; result unblocks `mes_q45_exec_bridge` action-queue item.

## Parallel session activity (DO NOT TOUCH)

- **`canompx3-pt`** — `chore/pass-three-drift` (PR #124 OPEN, CI failed; magic-number rationale audit + 9 trading_app/live constants; 69 dirty files). Owner: parallel Claude session.
- **`canompx3-ralph-burndown`** — `ralph/crit-high-burndown-v5.2` (branch `[gone]` on remote; HEAD `6740d938` predates merged Ralph PRs #107/#110). Stale; do not write to it. Local branch retained because of this worktree.
- **`canompx3-ghost`** — checked out on `main` (PR #135 work landed; worktree may be idle or retained for next ghost task).
- **MGC shadow observation** — operator action; observation result unblocks `mes_q45_exec_bridge` action-queue item.

## Blockers / Warnings

- Memory entries calling F-1 "dormant" are FACTUALLY WRONG — F-1 is **active** and fail-closing in signal-only. Day-1 XFA seeds `_topstep_xfa_eod_balance = $0.0` (canonical per `topstep_scaling_plan.py:51-53`); 6 deployed MNQ lanes ALLOW through `risk_mgr.can_enter()`. Verified empirically (E2E probe in PR #100/B6 work).
- `build_live_portfolio()` is **DEPRECATED** — `--all --signal-only` without `--profile` hard-fails all 3 orchestrators. Only `--profile` path works.
- Drift checks pre-existing: #4 (`work_queue.py` schema parser false-positive on `table 'the'`) and #59 (MGC — 1 day with `!= 3 daily_features` rows). Neither blocking.
- Working-tree debt: `requirements.txt` regenerated by `uv export` may be dirty in `canompx3-ghost` (main) worktree. Separate concern; needs its own commit + uv-lock-vs-export reconciliation.
- ~53 zombie local branches remain after 2026-04-26 conservative sweep (4 deleted). Most have merged PRs but no `[gone]` flag because origin still hosts the branch. Aggressive sweep deferred — would benefit from `clean_gone` command + cross-reference of `git branch --list` against `gh pr list --state all`.

## Durable References

- `docs/runtime/action-queue.yaml` — P1/P2 action queue
- `docs/runtime/decision-ledger.md` — durable decisions
- `docs/runtime/debt-ledger.md` — known debt
- `docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md` — HWM Stages 1-4 design (landed in #129)
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` — O-SR grounding (still pending: cusum_monitor → SR Eq 10)
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` — ORB premise
