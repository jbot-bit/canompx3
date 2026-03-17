# Ralph Loop — Autonomous Audit + Fix System

## Overview

The Ralph Loop is a single-agent continuous improvement system that audits, fixes,
and verifies this codebase in autonomous cycles. Each iteration:

1. **Read state** — Check audit file, history, ledger, centrality
2. **Audit** — Run infrastructure gates, Seven Sins scan on target file
3. **Select + blast radius** — Pick highest-priority finding, check callers
4. **Implement + verify** — Apply minimal fix, run tests + drift check
5. **Update files** — Audit state, history, ledger, deferred findings
6. **Report** — Structured output block for batch tracking

## Architecture

```
scripts/tools/ralph.sh              <- CLI dispatcher (once/batch/loop/review/audit/doctor)
  |
  +-- scripts/tools/ralph_headless.sh   <- Batch runner (N iterations via claude -p)
  +-- scripts/tools/ralph_review.sh     <- Post-batch Opus quality gate
  |
  +-- .claude/agents/ralph-loop.md      <- Single agent prompt (Sonnet)
  +-- .claude/skills/ralph/SKILL.md     <- Interactive skill (/ralph)
  |
  +-- docs/ralph-loop/ralph-loop-audit.md   <- Current findings (overwritten)
  +-- docs/ralph-loop/ralph-loop-plan.md    <- Current plan (overwritten)
  +-- docs/ralph-loop/ralph-loop-history.md <- All iterations (append-only)
  +-- docs/ralph-loop/deferred-findings.md  <- Open debt ledger (structured)
  +-- docs/ralph-loop/ralph-ledger.json     <- Cross-iteration intelligence
  +-- docs/ralph-loop/import_centrality.json <- Production-path weighting
  +-- docs/ralph-loop/logs/                 <- Per-iteration logs
```

## Running

```bash
# Single interactive iteration (uses /ralph skill)
bash scripts/tools/ralph.sh once

# Batch of 5 iterations (headless, no interaction needed)
bash scripts/tools/ralph.sh batch

# Batch with custom count and scope
bash scripts/tools/ralph.sh batch --iterations 10 --scope "live_config.py"

# Continuous loop (5-iteration batches with 30s pause)
bash scripts/tools/ralph.sh loop

# Post-batch Opus review of judgment commits
bash scripts/tools/ralph.sh review --last 10

# Quick audit only (drift + behavioral + ruff, no fixes)
bash scripts/tools/ralph.sh audit

# Preflight health check
bash scripts/tools/ralph.sh doctor

# Stop a running loop gracefully
touch ralph_loop.stop
```

### Headless Mode Details (v3)

The headless runner (`ralph_headless.sh`) uses:
- `claude -p` with `--output-format json` for structured output + cost tracking
- `--dangerously-skip-permissions` to prevent blocked permission prompts
- `--max-turns 25` to prevent runaway iterations
- `--no-session-persistence` to keep session list clean
- `--append-system-prompt` for headless-specific instructions
- Retry once on empty output before counting as error
- Git safety checks (stash uncommitted changes between iterations)
- Separate stderr from stdout (prevents JSON corruption)

Per-iteration logs: `docs/ralph-loop/logs/headless-YYYYMMDD_HHMM-iterN.json` (raw)
and `*.txt` (extracted result text).

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

| Tool | Purpose |
|------|---------|
| `pipeline/check_drift.py` | 71+ automated drift checks |
| `scripts/tools/audit_behavioral.py` | 7-gate behavioral rules |
| `pytest tests/test_<module>.py` | Targeted test suite (never full `pytest tests/` — OOMs) |
| `ruff check` | Lint enforcement |
| `docs/prompts/ENTRY_MODEL_GUARDIAN.md` | Schema change safety |
| `docs/prompts/PIPELINE_DATA_GUARDIAN.md` | Pipeline change safety |

## Grounding in Literature

- **Lopez de Prado** — Seven sins of quantitative investing, deflated Sharpe, PBO
- **Aronson** — Evidence-based technical analysis, data snooping bias
- **Harvey/Liu/Zhu** — Multiple testing framework, BH FDR for strategy selection
- **Chan** — Algorithmic trading infrastructure, backtesting pitfalls
- **Carver** — Systematic trading, position sizing, portfolio construction
- **Bailey et al.** — Walk-forward validation, probability of backtest overfitting

These are available in `resources/` for reference.
