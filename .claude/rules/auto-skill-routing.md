# Auto-Skill Routing

The user does not type `/skill` commands. Route by intent.

For non-trivial repo work, scope first with:
`python scripts/tools/context_resolver.py --task "<user request>" --format markdown`

Use skills only after the route narrows what should be read.

## Intent Map

- Session start, status, catch-up, "where are we" -> `/orient`
- Continue, next, what now, keep going -> `/next`
- Git ops like commit/push/merge -> execute directly
- Done / complete / finished -> `/verify done`
- Trading book / what's live / tonight / playbook -> `/trade-book`
- Health / decay / fitness / regime -> `/regime-check`
- Something wrong / weird / doesn't add up / failing -> `/quant-debug`
- Plan / design / brainstorm / approach / 4t -> `/design`
- Past findings / history / NO-GO / remind me -> `/pinecone-assistant`
- Kill-verdict lookup / has X been ruled out / graveyard / NO-GO on X -> `/nogo` (single-call research-catalog filter; faster than `/pinecone-assistant` for verdict-only)
- Test a hypothesis / research / investigate edge -> `/research` (and call `/nogo` FIRST so we don't re-litigate a buried verdict)
- Real-capital scrutiny / bias check / before deploy -> `/capital-review`
- Review / check my work / before I commit -> `/code-review`
- Editing schema or canonical config -> full stage-gate
- Improve a skill -> `/skill-improve`

## Code-graph (CRG) — auto-route, do not wait for `/crg-*`

CRG is a non-truth navigation surface (`feedback_crg_no_graph_storm.md`: one tool per turn, `detail_level=minimal`, pass `repo_root`). Auto-invoke on these intents:

- "where is X / what calls X / find Y / who imports Z" -> `/crg-search` (semantic + FTS)
- "blast radius / before editing / impact / what will this break" -> `/crg-blast` (preceded by `/crg-context`)
- "predicate lineage / contamination / what consumes feature X" -> `/crg-lineage`
- "tests for X / what tests cover Y" -> `/crg-tests`
- "dead code / unused functions" -> `/crg-deadcode`
- Any non-trivial review or pre-edit on `pipeline/` or `trading_app/` -> `/crg-context` first

Always pass `repo_root="C:/Users/joshd/canompx3"` (the canonical root) — worktrees share the parent's graph per `feedback_crg_worktree_repo_root_resolution.md`.

Match intent, not exact words. Destructive skills still require explicit user intent.
Hooks already handle routine post-work verification, so do not restate that ceremony here.
