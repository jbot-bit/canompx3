# Codex Rule Index

Codex should reach into the shared `.claude/rules/` set directly.

## Default Rules For Most Code Changes

- `.claude/rules/workflow-preferences.md`
- `.claude/rules/validation-workflow.md`

## Specialized Rules

- Full Claude rule parity source:
  - `.claude/rules/adversarial-audit-gate.md`
  - `.claude/rules/auto-memory-capture.md`
  - `.claude/rules/auto-skill-routing.md`
  - `.claude/rules/autonomy-contract.md`
  - `.claude/rules/backtesting-methodology-failure-log.md`
  - `.claude/rules/backtesting-methodology.md`
  - `.claude/rules/branch-discipline.md`
  - `.claude/rules/branch-flip-protection.md`
  - `.claude/rules/condition-based-waiting.md`
  - `.claude/rules/daily-features-joins.md`
  - `.claude/rules/hypothesis-prereg-discipline.md`
  - `.claude/rules/institutional-rigor.md`
  - `.claude/rules/integrity-guardian.md`
  - `.claude/rules/large-file-reads.md`
  - `.claude/rules/m25-audit.md`
  - `.claude/rules/mcp-usage.md`
  - `.claude/rules/multi-terminal-shared-file-hygiene.md`
  - `.claude/rules/parallel-session-isolation.md`
  - `.claude/rules/pipeline-patterns.md`
  - `.claude/rules/plugin-routing.md`
  - `.claude/rules/pooled-finding-rule.md`
  - `.claude/rules/quant-agent-identity.md`
  - `.claude/rules/quant-audit-failure-patterns.md`
  - `.claude/rules/quant-audit-protocol.md`
  - `.claude/rules/research-truth-protocol.md`
  - `.claude/rules/self-funded-sizing-doctrine.md`
  - `.claude/rules/shell-canon.md`
  - `.claude/rules/stage-gate-protocol.md`
  - `.claude/rules/strategy-awareness.md`
  - `.claude/rules/subagent-budget.md`
  - `.claude/rules/targeted-grounding.md`
  - `.claude/rules/telemetry-maturity-waiver.md`
  - `.claude/rules/validation-workflow.md`
  - `.claude/rules/workflow-preferences.md`
  - `.claude/rules/worktree-venv-isolation.md`
- Pipeline structure: `.claude/rules/pipeline-patterns.md`
- Feature joins and feature-table safety: `.claude/rules/daily-features-joins.md`
- Research and audit framing: `.claude/skills/audit/SKILL.md` (mode: prompts)
- Trading and quant identity: `.claude/rules/quant-agent-identity.md`
- MCP usage: `.claude/rules/mcp-usage.md`
- Integrity work: `.claude/rules/integrity-guardian.md`
- Pinecone + PDF routing: `.claude/skills/pinecone-assistant/SKILL.md`
- M25 audit triage rules: `.claude/rules/m25-audit.md` (script: `python scripts/tools/m25_auto_audit.py`)

## Load Guidance

- Load the smallest relevant subset.
- For pipeline or trading-app edits, start with workflow and validation rules.
- For statistical claims or strategy conclusions, load quant identity and research rules before writing conclusions.
- For feature work, always check `docs/specs/` first.
- For methodology work, use local PDFs in `resources/` (BH FDR, walk-forward, deflated Sharpe).
- If a rule applies but is not summarized here, use `canompx3-claude-parity`
  and read the exact `.claude/rules/*.md` file instead of relying on memory.
