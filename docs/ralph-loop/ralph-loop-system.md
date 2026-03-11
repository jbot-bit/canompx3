# Ralph Loop — Autonomous Audit + Fix System

## Overview

The Ralph Loop is a multi-agent continuous improvement system that audits, fixes,
and verifies this codebase in autonomous cycles. Each iteration:

1. **Audit** — Find real bugs with institutional-grade rigor
2. **Understand** — Map blast radius, trace callers, read authority docs
3. **Plan** — Select highest-priority fix, document approach
4. **Implement** — Apply minimal targeted fix (2-pass method)
5. **Validate** — Run 6-gate verification (drift, tests, lint, behavioral, blast, regression)
6. **Review** — Accept or reject, append to history
7. **Repeat**

## Architecture

```
scripts/ralph_loop_runner.sh     <- Runner (bash loop)
  |
  +-- .claude/agents/ralph-architect.md    <- Coordinator
  +-- .claude/agents/ralph-auditor.md      <- Finds bugs (never writes code)
  +-- .claude/agents/ralph-implementer.md  <- Applies minimal fixes
  +-- .claude/agents/ralph-verifier.md     <- 6-gate verification
  |
  +-- docs/ralph-loop/ralph-loop-audit.md   <- Current findings (overwritten)
  +-- docs/ralph-loop/ralph-loop-plan.md    <- Current plan (overwritten)
  +-- docs/ralph-loop/ralph-loop-history.md <- All iterations (append-only)
  +-- docs/ralph-loop/deferred-findings.md  <- Open debt ledger (structured)
  +-- docs/ralph-loop/logs/                 <- Per-iteration logs
```

## Agent Roles

| Agent | Reads | Writes | Runs Code |
|-------|-------|--------|-----------|
| Architect | audit, history, CLAUDE.md | plan, history | No |
| Auditor | all production code, tests | audit report | Yes (read-only: drift, tests, lint) |
| Implementer | plan, blast radius files | production code, tests | Yes (tests, drift) |
| Verifier | plan, audit, changed files | history, audit | Yes (all 6 gates) |

## Running

```bash
# Full loop (runs until stopped)
bash scripts/ralph_loop_runner.sh

# Single iteration
bash scripts/ralph_loop_runner.sh --once

# Audit only (no fixes)
bash scripts/ralph_loop_runner.sh --audit-only

# Stop gracefully
touch ralph_loop.stop
```

## Safety Boundaries

The system will NOT autonomously:
- Change database schema
- Modify entry models or trading logic
- Push to remote
- Delete files or branches
- Touch more than 5 files in one iteration
- Override CLAUDE.md, TRADING_RULES.md, or RESEARCH_RULES.md

These require human approval and are flagged in the plan file.

## Severity Levels

| Level | Definition | Action |
|-------|-----------|--------|
| CRITICAL | Data corruption, crashes, security holes, capital loss | Fix immediately |
| HIGH | Incorrect logic, broken workflows, silent failures | Fix this iteration |
| MEDIUM | Architecture risks, missing tests, diagnostic gaps | Plan for next iteration |
| LOW | Cleanup, style, documentation | Track, fix when convenient |

## Integration with Existing Tools

The Ralph Loop leverages the repository's existing infrastructure:

| Tool | Used By | Purpose |
|------|---------|---------|
| `pipeline/check_drift.py` | Auditor, Verifier | 71+ automated drift checks |
| `scripts/tools/audit_behavioral.py` | Auditor, Verifier | 7-gate behavioral rules |
| `pytest tests/` | Auditor, Implementer, Verifier | Full test suite |
| `ruff check` | Verifier | Lint enforcement |
| `docs/prompts/ENTRY_MODEL_GUARDIAN.md` | Architect | Schema change safety |
| `docs/prompts/PIPELINE_DATA_GUARDIAN.md` | Architect | Pipeline change safety |

## Grounding in Literature

The agents' auditing methodology is informed by:
- **Lopez de Prado** — Seven sins of quantitative investing, deflated Sharpe, PBO
- **Aronson** — Evidence-based technical analysis, data snooping bias
- **Harvey/Liu/Zhu** — Multiple testing framework, BH FDR for strategy selection
- **Chan** — Algorithmic trading infrastructure, backtesting pitfalls
- **Carver** — Systematic trading, position sizing, portfolio construction
- **Bailey et al.** — Walk-forward validation, probability of backtest overfitting

These are available in `resources/` for reference.
