# Context Source Catalog

Generated catalog of canonical routing sources and published read models.

## Domains

- `repo_governance`
- `research_methodology`
- `trading_runtime`

## Concepts

- `sacred_holdout_policy` — `trading_app/holdout_policy.py`
- `runtime_control_plane` — `pipeline/system_context.py`
- `shared_root_mutation_override` — `pipeline/system_context.py`
- `task_packet_contract` — `pipeline/work_capsule.py`

## Live Views

- `gold_db_mcp` — Canonical live trading/research query surface. (`trading_app/mcp_server.py`)
- `research_context` — Generated research view with strict truth boundaries. (`scripts/tools/context_views.py`)
- `recent_performance_context` — Generated recent-performance view for fit weakness questions. (`scripts/tools/context_views.py`)
- `trading_context` — Generated trading/runtime context view. (`scripts/tools/context_views.py`)
- `verification_context` — Generated verification-focused context view. (`scripts/tools/context_views.py`)
- `system_brief` — Derived minimal startup read-model for the current task. (`pipeline/system_brief.py`)

## Verification Steps

- `project_pulse_fast` — `./.venv-wsl/bin/python scripts/tools/project_pulse.py --fast --format json`
- `system_context_text` — `./.venv-wsl/bin/python scripts/tools/system_context.py --context codex-wsl --action orientation`
- `system_brief_json` — `./.venv-wsl/bin/python scripts/tools/system_brief.py --format json`
- `pytest_full` — `./.venv-wsl/bin/python -m pytest -q`
- `drift_check` — `./.venv-wsl/bin/python pipeline/check_drift.py`
- `render_context_catalog` — `./.venv-wsl/bin/python scripts/tools/render_context_catalog.py`

## Understanding Packs

- `coding_runtime_pack` — Repo shell, git, interpreter, and verification control surfaces.
- `trading_runtime_pack` — Live/trading runtime owners, deployment state, and operator truth surfaces.
- `research_methodology_pack` — Research doctrine, holdout policy, and institutional thresholds.
- `project_orientation_pack` — Minimal project-wide startup truth for cold-start repo work.

## Variables

- `orb_utc_window` — `pipeline/dst.py`
- `holdout_policy_var` — `trading_app/holdout_policy.py`
- `deployable_validated_relation` — `pipeline/db_contracts.py`