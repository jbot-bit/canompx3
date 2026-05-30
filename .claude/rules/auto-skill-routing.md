# Auto-Skill Routing

> **Hook-fired (2026-05-13):** `.claude/hooks/intent-router.py` matches the
> user prompt against the Intent Map + CRG section below on every
> UserPromptSubmit and injects a single-line `additionalContext` cue naming
> the recommended skill. Editing the rules here changes the documented
> routing surface; the hook keeps its own compiled regex table in sync.
> If you change the bullets, also update `INTENT_RULES` in the hook.
>
> **Load policy (2026-05-17):** this file does NOT auto-load on edits — the
> hook injects the routing cue at prompt-time, so the rule never needs to
> ship as per-edit context. Read on demand when modifying `INTENT_RULES`.
> Parity with the hook is enforced by
> `check_intent_router_routing_parity` in `pipeline/check_drift.py`.

The user does not type `/skill` commands. Route by intent.
The user also should not need to remember tool/plugin names or to request a
second-pass check. For check/improve/implement/fix/review/plan-style prompts,
`.claude/hooks/targeted-grounding-router.py` injects a compact cue to do a
targeted truth check and second-pass critique before acting. Details:
`.claude/rules/targeted-grounding.md`.
`/resource` and `/lit` are explicit local-literature grounding triggers: run
`python scripts/tools/check_pdf_tooling.py` and
`python scripts/tools/check_literature_coverage.py`, open `resources/INDEX.md`,
and use mapped `docs/institutional/literature/` extracts only when covered.
Missing extract means read the resource directly if it exists on this PC. Raw
PDFs are local-PC assets, not guaranteed remote state; if absent, say so and do
not imply raw-PDF verification. Do not skim/guess PDF content.

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
