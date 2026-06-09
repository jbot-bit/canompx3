# CLI Entrypoints Integration Plan — 2026-06-09

## Status / Why

Active. The audit found no installed console commands. Operator confirmed this is **not intentional** and is ADHD-hostile: too many workflows require remembering `python -m ...`, file paths, and flags.

Goal: a small, memorable `canompx3` command surface that delegates to existing canonical code and does **not** change trading/live behavior.

## Research grounding

Official Python packaging path:

- PyPA: `[project.scripts]` is the standard `pyproject.toml` table for console scripts; table key = command name, value = object reference. Do **not** use `[project.entry-points.console_scripts]` alongside it because that is ambiguous. <https://packaging.python.org/en/latest/specifications/pyproject-toml/#entry-points>
- Setuptools: installed console scripts let users call a command directly instead of `python -m`; installers create wrapper scripts that effectively import the target and `sys.exit(target())`. Target functions should accept no arguments and parse user input inside the function body. <https://setuptools.pypa.io/en/latest/userguide/entry_point.html#console-scripts>
- Python stdlib: `argparse` auto-generates help/usage and errors; it supports subcommands, aliases, and dispatch via `set_defaults(func=...)`, so no new Click/Typer dependency is needed for pass 1. <https://docs.python.org/3.13/library/argparse.html>

Repo grounding from static probe on 2026-06-09:

- `pyproject.toml` packages `pipeline*`, `trading_app*`, `scripts*`, `research*`; no `[project.scripts]` exists yet.
- Good direct `main()` targets already exist: `pipeline.health_check`, `pipeline.check_drift`, `scripts.audits.run_all`, `scripts.tools.project_pulse`, `scripts.tools.context_resolver`, `scripts.tools.workflow_doctor`, `scripts.tools.refresh_control_state`, `scripts.tools.live_readiness_report`, `scripts.tools.stage_reaper_audit`, `scripts.tools.worktree_guard`, `scripts.run_live_session`, `scripts.run_webhook_server`.
- `trading_app.live.bot_dashboard` has `run_dashboard(...)` and a `__main__` parser, but no importable `main()`; dashboard wrapper needs a small adapter or module refactor.

## Design decision

Use a **two-layer UX**:

1. `canompx3` umbrella command with subcommands for discoverability: `canompx3 doctor`, `canompx3 health`, `canompx3 drift`, etc.
2. A few direct aliases for muscle memory and shell search: `canompx3-doctor`, `canompx3-health`, `canompx3-drift`, etc.

Why this beats “many commands only”:

- ADHD-friendly: one root command is enough to discover all subcommands with `canompx3 --help`.
- Compatible with official packaging: both root and aliases are normal `[project.scripts]` entries.
- Low dependency risk: stdlib `argparse`, no new CLI framework.
- Safer rollout: aliases thinly delegate; no business logic fork.

## Non-negotiable invariants

- Wrappers only delegate; they do not reimplement audit, drift, live, dashboard, or broker logic.
- Exit codes must match the canonical target.
- `--help` must not require `gold.db`, broker credentials, network, dashboard port, or live imports when avoidable.
- Live/capital wrappers must not bypass preflight, confirmation, account routing, kill switches, or existing guardrails.
- No public command for every file in `scripts/`; curate the operator surface.
- Docs and `[project.scripts]` must stay in sync by test or drift check.

## Target command surface

### Tier 0 — land first (safe discovery/workflow)

| Command / alias | Delegates to | Notes |
| --- | --- | --- |
| `canompx3 health`, `canompx3-health` | `pipeline.health_check:main` | Local health. May fail if DB/hooks absent; preserve that. |
| `canompx3 drift`, `canompx3-drift` | `pipeline.check_drift:main` | Drift detection. Preserve existing slow/fast flags. |
| `canompx3 audit`, `canompx3-audit` | `scripts.audits.run_all:main` | System audit. |
| `canompx3 pulse`, `canompx3-pulse` | `scripts.tools.project_pulse:main` | Orientation/status. |
| `canompx3 context`, `canompx3-context` | `scripts.tools.context_resolver:main` | Deterministic context routing. |
| `canompx3 doctor`, `canompx3-doctor` | `scripts.tools.workflow_doctor:main` | Workflow/worktree/dashboard status. |

### Tier 1 — second (side-effecting but non-broker)

| Command / alias | Delegates to | Notes |
| --- | --- | --- |
| `canompx3 refresh-control`, `canompx3-refresh-control` | `scripts.tools.refresh_control_state:main` | Writes control state; label in help/docs. |
| `canompx3 live-readiness`, `canompx3-live-readiness` | `scripts.tools.live_readiness_report:main` | Report/proof pack path. |
| `canompx3 stage-reaper-audit`, `canompx3-stage-reaper-audit` | `scripts.tools.stage_reaper_audit:main` | Audit/report only unless target already mutates. |
| `canompx3 worktree-guard`, `canompx3-worktree-guard` | `scripts.tools.worktree_guard:main` | Lease/status/release helper; preserve existing flags. |

### Tier 2 — last (live/app surfaces)

| Command / alias | Delegates to | Notes |
| --- | --- | --- |
| `canompx3 dashboard`, `canompx3-dashboard` | adapter around `trading_app.live.bot_dashboard.run_dashboard` or new `main()` | Needs wrapper because module lacks importable `main()`. |
| `canompx3 live-session`, `canompx3-live-session` | `scripts.run_live_session:main` | Must preserve all preflight/confirm/account gates. |
| `canompx3 webhook-server`, `canompx3-webhook-server` | `scripts.run_webhook_server:main` | Server launcher; document port/env behavior. |

## Implementation slices for separate terminals

Use isolated worktrees/terminals if parallel. Do not overlap owner files.

### Terminal A — Tier 0 root + aliases

Owner files: `pyproject.toml`, new `scripts/cli/`, `tests/test_cli/`.

Tasks:

1. Add `scripts/cli/__init__.py` and `scripts/cli/main.py`.
2. Implement `canompx3` umbrella with `argparse` subcommands dispatching to canonical `main()` functions.
3. Add Tier 0 direct alias functions (`health()`, `drift()`, etc.) that call the same dispatch targets.
4. Add `[project.scripts]` for `canompx3` + Tier 0 aliases only.
5. Tests: command registry parses; each alias delegates to expected import path; `canompx3 --help` and each Tier 0 alias `--help` exit 0 without DB/broker/network.

Exit checks:

```bash
python -m pytest tests/test_cli -q
python -m pytest tests/test_tools/test_context_resolver.py tests/test_tools/test_workflow_doctor.py -q
ruff check pyproject.toml scripts/cli tests/test_cli
git diff --check
```

### Terminal B — docs parity / operator inventory

Owner files: `docs/workflows/cli-entrypoints.md`, optional short links in `CLAUDE.md`/`CODEX.md`, docs tests.

Tasks:

1. Write a compact command inventory: command, use when, side effects, canonical target, common example.
2. Add a docs test that every `[project.scripts]` entry appears in the inventory and every documented `canompx3*` command exists in `[project.scripts]`.
3. Add one short startup-doc pointer, not a duplicated table.

Exit checks:

```bash
python -m pytest tests/test_docs/test_cli_entrypoints_docs.py -q
python scripts/tools/context_resolver.py --task "CLI entrypoints command inventory docs" || true
git diff --check
```

### Terminal C — Tier 1 wrappers

Owner files: `pyproject.toml`, `scripts/cli/`, `tests/test_cli/`, command inventory rows.

Tasks:

1. Add Tier 1 subcommands + aliases after Terminal A lands.
2. Preserve exit codes and stdout/stderr behavior.
3. Make help text explicit about side effects.
4. Add no-accidental-live-import tests for `--help`.

Exit checks:

```bash
python -m pytest tests/test_cli -q
python -m pytest tests/test_tools/test_live_readiness_report.py tests/test_tools/test_worktree_guard.py -q
ruff check pyproject.toml scripts/cli tests/test_cli
git diff --check
```

### Terminal D — Tier 2 live/app wrappers

Owner files: `pyproject.toml`, `scripts/cli/`, `tests/test_cli/`, command inventory rows.

Tasks:

1. Add only after A/B/C are green.
2. For dashboard, add an importable `main()` adapter if needed; do not duplicate server logic.
3. For live-session, delegate exactly to `scripts.run_live_session:main`; do not special-case flags.
4. Add tests proving no bypass of preflight/account/confirm paths.

Exit checks:

```bash
python -m pytest tests/test_scripts/test_run_live_session_preflight.py tests/test_scripts/test_run_live_session_account_id_sentinel.py tests/test_trading_app/test_bot_dashboard.py -q
python -m pytest tests/test_cli -q
ruff check pyproject.toml scripts/cli tests/test_cli
git diff --check
```

## Serial fallback

If one terminal only: A → B → C → D, committing after each green slice. Do not start D until A/B/C are committed and clean.

## Ready Claude Code prompt — first pass only

```text
Read AGENTS.md, CLAUDE.md, CODEX.md, HANDOFF.md, then docs/plans/active/2026-06/2026-06-09-cli-entrypoints-integration-plan.md.

Implement Terminal A only. Add the stdlib-argparse `canompx3` umbrella command plus Tier 0 aliases through `[project.scripts]`. Use thin dispatch to existing canonical main() functions; do not touch Tier 1/Tier 2 live/app commands. Add tests proving registry sync, --help behavior, and delegation targets. Run Terminal A exit checks plus git diff --check. Commit on the current branch.
```

## Acceptance criteria

- `uv sync --frozen` / editable install exposes `canompx3` and Tier 0 aliases on PATH.
- `canompx3 --help` is the single discoverability surface.
- Tier 0 aliases work and are documented.
- Docs and `[project.scripts]` cannot drift silently.
- Later Tier 1/Tier 2 wrappers preserve canonical safety behavior and exit codes.
