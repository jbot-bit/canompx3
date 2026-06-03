# Maximise / No-Tunnel-Vision / Exploit-Map Sprint — 2026-06-03

Verdict: `CONTINUE`

## Phase 0 Preflight

- cwd: `/workspace/canompx3`
- starting branch: `work`; action branch: `session/joshd-maximise-no-tunnel-vision`
- `git status --short` before branch switch: clean.
- `git worktree list --porcelain`: one worktree, `/workspace/canompx3`, branch `refs/heads/work` before switching.
- worktree guard status: `lease_present=false`, lease path `/workspace/canompx3/.git/.claude.worktree.lease.json`.
- main peer lease: no separate main checkout was present under `/workspace`; no live peer lease was found for this worktree.
- preflight blockers observed before local correction: `core.hooksPath=<unset>` and missing `.venv-wsl/bin/python`. `core.hooksPath` was restored with `git config core.hooksPath .githooks`; `.venv-wsl` remains an environment blocker for repo-managed mutating preflight, but targeted Python checks can run with the available interpreter.

## Source-of-truth extracts only

Files read in required order or closest canonical location:

- `chatgpt_bundle/00_INDEX.md`
- `chatgpt_bundle/01_OPERATING_RULES.md`
- `chatgpt_bundle/02_USER_PROFILE.md`
- `chatgpt_bundle/07_PLAYBOOKS.md`
- `chatgpt_bundle/04_DECISION_LOG.md`
- `chatgpt_bundle/06_RD_GRAVEYARD.md`
- `docs/STRATEGY_BLUEPRINT.md`
- `chatgpt_bundle/CANONICAL_VALUES.md`
- `TRADING_RULES.md`
- `RESEARCH_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/mechanism_priors.md`
- `chatgpt_bundle/backtesting-methodology.md`
- `TODO.md`
- `HANDOFF.md`
- Relevant current result ledgers: `docs/audit/results/2026-06-01-live-readiness-proof-pack.{md,json}`, `docs/audit/results/2026-06-01-live-readiness-strict-zero-warn.json`, `docs/audit/results/2026-06-02-mnq-single-leg-account-fit-replacement-v1.md`.

Extracted no-go traps and source-of-truth chains:

- Research truth remains `bars_1m` -> `daily_features` -> `orb_outcomes`; derived `validated_setups`, `edge_families`, and `live_config` are deployment/allocation surfaces only.
- 2026 holdout is monitoring-only for selection/tuning/ranking.
- E0, dead ORB instruments, arbitrary 24h grids, broad morning re-mining, post-hoc threshold rescue, and deployment from ASX-open discovery are no-go.
- Live deployment requires profile-scoped live readiness, Criterion 11 account-survival evidence, Criterion 12 validity, telemetry maturity, and strict live gates.
- The operational deadlock trap is real but the direct hook bug has partly been fixed already: the worktree guard now resolves hook payload `cwd`, blocks only index-mutating repo operations, and emits a live-peer message with `scripts/tools/new_session.sh <descriptor>` as the safe path. The remaining measured gap was `workflow_doctor` still recommending only `worktree_guard.py --status --json` for a `peer_lease` block, which inspects the holder but does not provide the clean escape hatch.

## Fresh checks / canonical query attempts

- `python - <<'PY' ... pipeline.paths.GOLD_DB_PATH ... PY`: canonical DB path resolved to `/workspace/canompx3/gold.db`; `exists=False`.
- `duckdb.connect('gold.db', read_only=True)`: failed with `IO Error: Cannot open database "/workspace/canompx3/gold.db" in read-only mode: database does not exist`.
- `python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --format json --strict-zero-warn`: failed for the same missing canonical DB before loading `validated_setups`.
- `python - <<'PY' ... trading_app.live_config.LIVE_PORTFOLIO ... PY`: current derived live portfolio contains 8 family specs, but this is not a research/allocation truth count.
- `python scripts/tools/workflow_doctor.py json`: after the code change, reported branch `session/joshd-maximise-no-tunnel-vision`, hooks OK, `lease_present=false`, DB missing with read-only open warning, dashboard port not listening, launchers present, and current `next` as dirty-tree inspection while edits were uncommitted.

## Opportunity map

| rank | lane | claim | evidence source | expected EV | risk | effort | source-of-truth chain | smallest safe diff | action_now? |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Worktree / hook / git / drift friction | The prior deadlock class has a remaining operator-guidance gap: diagnostics knew launchers existed but a peer-lease block still recommended holder inspection instead of opening an isolated worktree. | `workflow_doctor.choose_next`, existing worktree-guard block message, launcher inventory from `workflow_doctor.py json`. | High: prevents repeated Claude/operator dead time and avoids temptation to force-release live peers. | Low if safety remains fail-closed for live peers. | Small. | `.claude/hooks/worktree_guard.py` -> `scripts/tools/worktree_guard.py` lease; `scripts/tools/workflow_doctor.py` diagnostics; `START_WORKTREE.bat` / `scripts/tools/new_session.sh` launchers. | Make `workflow_doctor` recommend `START_WORKTREE.bat <descriptor>` or `scripts/tools/new_session.sh <descriptor>` for peer leases; add focused tests. | yes |
| 2 | Allocation / live-readiness | Cannot re-verify promotable/FIT vs allocated counts in this WSL checkout because canonical `gold.db` is absent. Stale proof pack shows live-readiness blockers: missing Criterion 11 survival report and invalid Criterion 12. | Fresh DB open attempt failed; `2026-06-01-live-readiness-proof-pack.json`; `LIVE_PORTFOLIO` import. | Potentially high, but current volatile counts are unavailable. | High if acting on stale docs. | Medium once DB exists. | `validated_setups`/allocator/live_config for status only; `account_survival` and profile gates for deployment safety. | NEED EVIDENCE: mount/provide canonical DB, then run live readiness and allocator dry-run. | no |
| 3 | Dashboard/live-trading readiness | Dashboard is not currently listening in this WSL checkout; recent handoff says CTA visibility was fixed and tested. No fresh runtime smoke can verify buttons without launching app and DB/runtime context. | `workflow_doctor.py json` dashboard block; `HANDOFF.md` recent dashboard note. | Medium if runtime blocks live launch. | Medium if UI smoke conflates no server with broken UI. | Medium. | dashboard runtime -> API `/api/action/start` -> live preflight -> `scripts.run_live_session`. | No UI diff without a reproduced break. | no |
| 4 | Drift check speed | Drift remains a reported residual gap: prior fast/full drift timed out; `workflow_doctor` exposes profiling commands but does not run drift by default. | `HANDOFF.md`; `workflow_doctor.collect_drift`. | Medium; important but less direct than peer-lease escape. | Medium if caching hides drift. | Medium. | `pipeline/check_drift.py`; `scripts/tools/profile_check_drift.py`. | Profile before changing; no removal. | no |
| 5 | ASX cash open research gap | Structurally plausible Sydney-time event, but no DB and no pre-reg scan can be done here. It does not beat the operational blocker. | Canonical rules plus missing `gold.db`; no ASX scan run. | Unknown. | High if tested/ranked without prereg or 2026 discipline. | Medium. | `pipeline/dst.py` session catalog -> `orb_outcomes` scoped rebuild -> pre-2026 scan. | Create prereg only if chosen; not chosen. | no |
| 6 | Research integrity / stale docs | Stale allocation/live-readiness claims exist because DB-backed counts cannot be refreshed locally. | Missing DB, stale result ledgers. | Medium. | Low for appending corrections; high if rewriting history. | Small/medium. | Repo docs are claims; DB/code are truth. | Append this report with NEED EVIDENCE rather than changing historic results. | partial |

## Action taken

Chosen action: lane 1, operational blocker.

Before behavior:

- `workflow_doctor.choose_next()` handled a `peer_lease` block by returning `python scripts/tools/worktree_guard.py --status --json` with reason `inspect holder`.
- That preserved diagnostics but did not tell an anchored operator how to escape into an isolated worktree, despite `START_WORKTREE.bat` and `scripts/tools/new_session.sh` being discoverable.

After behavior:

- For `peer_lease`, `workflow_doctor.choose_next()` now returns `START_WORKTREE.bat <descriptor>` when the Windows launcher is visible.
- If the Windows launcher is absent but the bash launcher exists, it returns `scripts/tools/new_session.sh <descriptor> && cd ../canompx3-<descriptor>`.
- If no launcher is visible, it falls back to the read-only status command.
- The reason explicitly says to open an isolated worktree while the live peer lease remains intact.

Changed files:

- `scripts/tools/workflow_doctor.py`
- `tests/test_tools/test_workflow_doctor.py`
- `docs/audit/results/2026-06-03-maximise-no-tunnel-vision-sprint.md`
- `HANDOFF.md`

## Still blocked

- Fresh allocation/live-readiness counts are `NEED EVIDENCE`: canonical `/workspace/canompx3/gold.db` is absent in this WSL checkout, so promotable/FIT candidate counts, live allocation status, account-safe portfolio EV, and current Criterion 11/12 state cannot be truthfully refreshed here.
- `.venv-wsl/bin/python` is absent, so repo-managed mutating preflight remains blocked even though targeted checks can run with `/root/.pyenv/versions/3.13.13/bin/python`.

## Next action

`python -m trading_app.account_survival --profile topstep_50k_mnq_auto`
