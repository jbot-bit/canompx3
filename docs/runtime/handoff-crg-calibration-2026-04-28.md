---
status: parked
owner: claude (CRG calibration session 2026-04-28)
created: 2026-04-28
parent_task: nexus-crg-calibration + anti-fuckup hooks
resume_with: read this file, then resume PASS 1 of the user's CRG calibration prompt
---

# Handoff — CRG/Nexus calibration (parked at PASS 1)

User prompt that initiated this work is the "Mode: CRG/Nexus calibration + anti-fuckup workflow build" directive. This handoff captures state at context-clear so the next session resumes exactly where we stopped instead of re-deriving.

## Filesystem changes (uncommitted, untracked-deletion only)

- Deleted `.gitnexus/` — 32K orphan dir from failed GitNexus eval (LadybugDB scratch). No git footprint (was untracked).
- Deleted `docs/superpowers/specs/2026-04-28-edge-discovery-enforcement-gaps-design.md` — dead design doc marked `needs-revision`, three claimed gaps disproven during self-audit. No git footprint.

Other untracked files left in place (other-session work, not ours to touch):
- `.claude/hooks/.completion-notify-last`
- `.claude/hooks/bias-grounding-guard.py`
- `live_session.stop`
- `live_signals_2026-04-27.jsonl`

One tracked modification not from this session: `docs/audit/results/2026-04-28-d4-aistudio-claims-audit.md` (other session).

## Branch state

- Current branch: `research/2026-04-28-phase-d-mnq-comex-settle-pathway-b`
- 5 ahead, 1 behind main
- main is at `b03b19a4` (PR #171: CRG calibration + Phase D D4 prereg)
- Branch delta is `[judgment]` Phase D research commits — no `pipeline/` or `trading_app/` changes — so CRG built-on-main is fully accurate for canonical APIs

## CRG state (verified, not memory)

- Package: `code-review-graph` v2.3.2, location `.venv/Lib/site-packages/code_review_graph/`
- Homepage: `https://code-review-graph.com`
- MCP server registered as `code-review-graph` (stdio transport)
- Graph DB: `.code-review-graph/graph.db` (local working tree)
- Last build: 2026-04-28T21:11:50, on `main` at `b03b19a42fa4`
- Stats: 13,738 nodes, 149,477 edges, 1,035 files, 12,693 embeddings
- Edge kinds: CALLS 108718, CONTAINS 12726, IMPORTS_FROM 7480, INHERITS 121, REFERENCES 48, TESTED_BY 20384
- Languages: python, bash, powershell

## What's working vs blocked

| Tool | Status | Notes |
|---|---|---|
| `list_graph_stats_tool` | working | <1s response |
| `query_graph_tool` | working | importers_of pipeline/cost_model.py returned 152 results |
| `get_minimal_context_tool` | BLOCKED | harness permission rejection ("user doesn't want to proceed") — needs pre-approval next session |
| `get_impact_radius_tool` | UNTESTED | not yet probed |
| `semantic_search_nodes_tool` | UNTESTED | embeddings exist (12693), worth probing |
| `get_review_context_tool` | UNTESTED | likely highest-value tool for PR review workflow |

One transient `MCP error -32000: Connection closed` on `get_minimal_context_tool`. Resolved by `/mcp` reconnect. Suggests the heavier tools may crash the MCP server under some conditions — re-probe carefully next session.

## PASS plan progress

- PASS 0 verify state — COMPLETE
- PASS 1 cheap probes — 1 of 5 done (importers_of cost_model)
- PASS 2 design `docs/plans/active/2026-04/nexus-crg-operating-model.md` — NOT STARTED
- PASS 3 implement low-blast-radius hooks/commands — NOT STARTED
- PASS 4 verification — NOT STARTED

## Resume order (next session)

1. Pre-approve MCP permissions for `mcp__code-review-graph__get_minimal_context_tool`, `get_impact_radius_tool`, `semantic_search_nodes_tool`, `get_review_context_tool` (or accept on first prompt).
2. Pull official docs — start with package's installed `__init__.py`, then `code-review-graph.com`, then any README in the venv site-packages dir.
3. Finish PASS 1 probes on `pipeline/cost_model.py`, `pipeline/dst.py`, `trading_app/holdout_policy.py`. Use `detail_level: "minimal"` to keep responses small.
4. PASS 2 design doc at `docs/plans/active/2026-04/nexus-crg-operating-model.md` per user's required structure (5 workflows + failure modes + escalation ladder).
5. PASS 3 hooks: branch-context warning is the highest-leverage one (incident 2026-04-28: agent edited research file thinking it was on main, mis-diagnosed missing PR; this session). Stay in `.claude/hooks/`, `.claude/commands/`, `.code-review-graphignore`, `docs/plans/`.
6. PASS 4: drift, targeted tests, hook syntax checks, exact pass/fail output.

## Stop conditions still active

- If CRG/MCP unavailable, stop and diagnose only.
- If branch/main mismatch widens, stop and report exact branch state before editing.
- If a change touches trading logic or DB schema, stop.

## Parked work (do not pursue without explicit go)

- Anti-fuckup hooks broader brainstorm (Thread A from brainstorming session)
- D5 Conditional Sizing (AIStudio's separate directive — different task)
- Scratch-policy + E2-lookahead advisory backlog (132 + 2 unannotated; not blocking)

## Known mistakes from this session (don't repeat)

1. Edited a `research/` file on a research branch while believing it was on main. Caught and reverted. Hardening: pre-edit branch context hook (designed but not shipped — too speculative for the session's scope).
2. Mis-diagnosed PR #171 as orphaned. Cause: I checked HEAD on the research branch, saw `b03b19a4` not in ancestor chain, jumped to "lost". Truth: just on a research branch cut before the merge. Lesson: `git branch --show-current` first, always.
3. Drifted between three scope-cuts (enforcement-gap doc → CRG calibration → AIStudio D5). User had to repeatedly redirect. Lesson: parking a task is fine; switching every 3 messages isn't.

## Verification commands for next-session quick check

```
git branch --show-current
git status --short
code-review-graph status
ls .gitnexus/ docs/superpowers/specs/2026-04-28-edge-discovery-enforcement-gaps-design.md 2>&1   # both should be missing
```
