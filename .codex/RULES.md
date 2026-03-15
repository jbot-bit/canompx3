# Codex Rule Index

Codex should reach into the shared `.claude/rules/` set directly.

## Default Rules For Most Code Changes

- `.claude/rules/workflow-preferences.md`
- `.claude/rules/validation-workflow.md`

## Specialized Rules

- Pipeline structure: `.claude/rules/pipeline-patterns.md`
- Feature joins and feature-table safety: `.claude/rules/daily-features-joins.md`
- Research and audit framing: `.claude/skills/audit-prompts/SKILL.md`
- Trading and quant identity: `.claude/rules/quant-agent-identity.md`
- MCP usage: `.claude/rules/mcp-usage.md`
- Integrity work: `.claude/rules/integrity-guardian.md`
- NotebookLM integration: `.claude/rules/notebooklm.md`
- Pinecone helper flow: `.claude/skills/pinecone-assistant/SKILL.md`
- M25-specific audit flow: `.claude/rules/m25-audit.md`

## Load Guidance

- Load the smallest relevant subset.
- For pipeline or trading-app edits, start with workflow and validation rules.
- For statistical claims or strategy conclusions, load quant identity and research rules before writing conclusions.
- For feature work, always check `docs/specs/` first.
- For methodology work, respect that NotebookLM is retired here; use local PDFs via `.claude/rules/notebooklm.md`.
