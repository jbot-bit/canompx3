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
- Test a hypothesis / research / investigate edge -> `/research`
- Real-capital scrutiny / bias check / before deploy -> `/capital-review`
- Review / check my work / before I commit -> `/code-review`
- Editing schema or canonical config -> full stage-gate
- Improve a skill -> `/skill-improve`

Match intent, not exact words. Destructive skills still require explicit user intent.
Hooks already handle routine post-work verification, so do not restate that ceremony here.
