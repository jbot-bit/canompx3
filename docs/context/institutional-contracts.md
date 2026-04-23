# Institutional Routing Contracts

Generated registry of concepts, protocols, answer contracts, and briefing rules.

## Concepts

- `sacred_holdout_policy` — Research and validation work must honor the holdout boundary and grandfather rules. (`trading_app/holdout_policy.py`)
- `runtime_control_plane` — Repo startup, interpreter, git, and claim state must be read from canonical control surfaces. (`pipeline/system_context.py`)
- `shared_root_mutation_override` — Mutating work in the shared root is exceptional and must require an explicit override. (`pipeline/system_context.py`)
- `task_packet_contract` — Active scoped work should carry one explicit capsule with scope, authorities, and verification. (`pipeline/work_capsule.py`)

## Decision Protocols

- `research_investigation_protocol` — Resolve the live question, confirm the canonical research constraints, and gather evidence before explaining.
- `completion_protocol` — Do not claim done until the declared verification profile is green or the exact blocker is named.
- `implementation_protocol` — Read the owning rules first, stay inside the scoped surface, and verify behavior after edits.
- `live_status_protocol` — For runtime questions, prefer control state and deployment summaries over research doctrine.
- `system_orientation_protocol` — Return the smallest complete startup model for the current task or repo surface.

## Answer Contracts

- `research_investigation_answer` — A research answer should report the question, evidence surface, and constrained conclusion.
- `completion_answer` — A completion answer should say whether the target is done, verified, or blocked.
- `implementation_answer` — An implementation answer should explain the intended change and bounded verification.
- `live_status_answer` — A runtime answer should foreground readiness, blockers, and near-term operator state.
- `orientation_answer` — An orientation answer should compress the startup model to the minimum sufficient surface.

## Understanding Packs

- `coding_runtime_pack` — Repo shell, git, interpreter, and verification control surfaces.
- `trading_runtime_pack` — Live/trading runtime owners, deployment state, and operator truth surfaces.
- `research_methodology_pack` — Research doctrine, holdout policy, and institutional thresholds.
- `project_orientation_pack` — Minimal project-wide startup truth for cold-start repo work.

## Drilldown Playbooks

- `research_recent_performance_drilldown` — Move from broad fit weakness to lane-level recent-performance evidence using approved query templates.

## Briefing Contracts

- `orientation_briefing` — Minimal complete repo orientation for cold-start work.
- `investigation_briefing` — Compact but complete research/truth-finding startup brief.
- `mutating_briefing` — Fail-closed mutating startup brief with scoped ownership and verification.
