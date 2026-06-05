+++
capsule_id = "institutional-audit-2026-06-03"
title = "institutional-audit-2026-06-03"
status = "active"
branch = "wt-codex-institutional-audit-2026-06-03"
worktree_name = "institutional-audit-2026-06-03"
tool = "codex"
task_id = "institutional_full_audit_next_phase"
route_id = "live_capital_runtime_audit"
briefing_level = "mutating"
purpose = "Run the next phase of the institutional audit/action plan for topstep_50k_mnq_auto without live allocation or broker launch changes."
summary = "Refreshing live-readiness evidence, attribution state, audit gates, and the narrow code/docs repairs needed by the institutional audit."
objective = "Leave the worktree with green targeted gates or explicit blockers, durable evidence, and no unrecorded live-risk changes."
next_step = "Run remaining verification gates and update the audit report and handoff."
created_at = "2026-06-03T00:00:00+00:00"
updated_at = "2026-06-03T00:00:00+00:00"
stage_path = "docs/runtime/stages/institutional-audit-2026-06-03.md"
task_domains = ["live_readiness", "capital_review", "repo_audit", "runtime_governance"]
authorities = ["AGENTS.md", "HANDOFF.md", "CLAUDE.md", "CODEX.md", "docs/governance/system_authority_map.md", "TRADING_RULES.md", "RESEARCH_RULES.md"]
scope_paths = ["HANDOFF.md", "REPO_MAP.md", "docs/audit/results/2026-06-03-institutional-full-audit-tasked-plan.md", "docs/runtime/capsules/institutional-audit-2026-06-03.md", "docs/runtime/stages/institutional-audit-2026-06-03.md", "docs/runtime/lane_allocation/topstep_50k_mnq_auto.next-phase-dry-run.json", "scripts/audits/phase_2_infra_config.py", "scripts/audits/phase_3_docs.py", "tests/test_trading_app/test_strategy_discovery_architecture.py", "trading_app/config.py", "trading_app/fdr.py", "trading_app/strategy_discovery.py", "trading_app/strategy_validator.py"]
out_of_scope = ["live allocation writes", "broker/live launch", "schema changes", "broad research scans", "ASX cash-open exploration", "force-killing peer processes"]
verification_commands = ["python scripts/tools/work_capsule.py", "python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync", "python scripts/tools/project_pulse.py --fast --format json", "python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn", "python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto --strict-live-clean --output docs/runtime/lane_allocation/topstep_50k_mnq_auto.next-phase-dry-run.json", "python scripts/audits/run_all.py --phase 2", "python scripts/audits/run_all.py --phase 3", "python scripts/audits/run_all.py --phase 4", "python scripts/audits/run_all.py --phase 7", "python scripts/audits/run_all.py --phase 9", "python -m pytest tests/test_trading_app/test_strategy_discovery_architecture.py tests/test_trading_app/test_strategy_validator.py::TestBenjaminiHochberg -q", "python -m py_compile trading_app/fdr.py trading_app/strategy_discovery.py trading_app/strategy_validator.py trading_app/config.py scripts/audits/phase_2_infra_config.py scripts/audits/phase_3_docs.py", "ruff check trading_app/fdr.py trading_app/strategy_discovery.py trading_app/strategy_validator.py trading_app/config.py scripts/audits/phase_2_infra_config.py scripts/audits/phase_3_docs.py tests/test_trading_app/test_strategy_discovery_architecture.py", "ruff format --check trading_app/fdr.py trading_app/strategy_discovery.py trading_app/strategy_validator.py trading_app/config.py scripts/audits/phase_2_infra_config.py scripts/audits/phase_3_docs.py tests/test_trading_app/test_strategy_discovery_architecture.py", "git diff --check"]
acceptance_criteria = ["Current work capsule is recognized by scripts/tools/work_capsule.py.", "Criterion 11 and Criterion 12 are valid against canonical gold.db.", "Strict live readiness remains green except explicitly classified non-blocking telemetry maturity.", "Allocation dry-run reports no live lane changes.", "Changed Python files compile and targeted tests pass.", "Audit report and HANDOFF.md record exact evidence and blockers."]
risks = ["gold.db write contention can block state refresh commands.", "Full outcome rebuilds can mutate large research truth tables and are not part of this capsule without a separate stage.", "Windows pytest cleanup can emit PermissionError after tests finish."]
references = ["docs/audit/results/2026-06-03-institutional-full-audit-tasked-plan.md", "docs/runtime/lane_allocation/topstep_50k_mnq_auto.next-phase-dry-run.json"]
decision_refs = ["docs/runtime/decision-ledger.md#current"]
debt_refs = ["docs/runtime/debt-ledger.md#open-debt"]
history_window = "current"
+++

# Context

The next phase focuses on turning the prior institutional audit from plan into measured runtime evidence and narrow repairs. The live-risk boundary is explicit: no broker launch and no live allocation writes.

# Decision Ledger

- Use canonical `C:\Users\joshd\canompx3\gold.db` for live-readiness refreshes so project-pulse DB identity gates match the active profile.
- Keep allocation changes report-only through the strict-live-clean rebalance output.

# Verification Ledger

- Commands and results are recorded in `docs/audit/results/2026-06-03-institutional-full-audit-tasked-plan.md`.

# Open Questions

- Scoped MES/MGC/MNQ outcome refresh remains a separate mutating data task unless the operator explicitly launches that stage.
