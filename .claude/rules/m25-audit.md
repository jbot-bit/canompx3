---
paths:
  - "scripts/tools/m25*"
---
# M2.5 Second-Opinion Audit Protocol

## Authority Chain
**CLAUDE.md > Claude Code > M2.5 suggestions.** M2.5 is a second-opinion scanner, not an authority. Its findings are unverified suggestions.

## False Positive Rates (Empirical, Mar 2026)
| Prompt Style | FP Rate | Notes |
|-------------|---------|-------|
| Bug-hunt mode (old) | ~70-79% | Just "find bugs" |
| Structured graded mode | ~41% | "What's good + what's bad + grade" |
| Institutional persona | ~41% | Bloomberg/head-of-quant framing |

Architecture context preamble (auto-prepended since Mar 4 2026) reduces FP by preventing M2.5's top false positive patterns.

## Known M2.5 Blind Spots (Validated Mar 2026)
These are patterns M2.5 consistently gets wrong. The architecture context preamble addresses them, but verify during triage:

1. **Cross-file architecture** — M2.5 flags outcome_builder grid search as "data snooping" because it can't see BH FDR in strategy_validator.py. The preamble now explains this.
2. **DuckDB replacement scans** — M2.5 doesn't know DuckDB can reference in-scope pandas DataFrames in SQL. The preamble now explains this.
3. **4-gate ML system** — M2.5 evaluates individual gates (delta_r >= 0) in isolation. The preamble now explains the combined system.
4. **atexit exception handling** — M2.5 flags `except Exception: pass` in atexit handlers as "silent failure." The preamble now explains this is correct for shutdown cleanup.
5. **-999.0 NaN sentinel** — M2.5 flags fillna(-999.0) as a bug. The preamble explains it's an intentional domain sentinel for proximity features.
6. **Cost model coverage** — M2.5 flags "missing cost handling" because it can't see cost_model.py. The preamble confirms costs are handled.
7. **Dead instruments** — M2.5 flags "survivorship bias" because it sees only active instruments. The preamble explains dead instruments were tested.

## Verification Protocol (MANDATORY)
When M2.5 flags a finding, Claude Code MUST:
1. Read the actual code at the cited line numbers (often off by 5-20 lines)
2. Trace the execution path to confirm/deny the claim
3. Check if existing guards (try/except, if-checks, type constraints) already handle it
4. Cross-reference against CLAUDE.md rules, `.claude/rules/`, and project invariants
5. Only implement fixes for findings verified as TRUE

## What M2.5 Is Good At
- File-level bug hunts (`--mode bugs`) on single files
- Bias detection (`--mode bias`) on research/strategy code
- JOIN correctness (`--mode joins`) on SQL-heavy pipeline code
- **Institutional improvement suggestions** (`--mode improvements`) with structured prompts
- Spotting silent `except: pass` and similar anti-patterns
- Identifying dead code and unreachable branches

## What M2.5 Is Bad At
- System-level reasoning (can't trace cross-file call chains)
- Technology-specific knowledge (DuckDB, databento SDK)
- Understanding fail-open design patterns (flags them as bugs)
- Recognizing when a guard exists in a file it wasn't given
- Line number accuracy (typically off by 5-20 lines)

## Available Modes
| Mode | When to Use | Prompt Style |
|------|------------|--------------|
| `general` | Default catch-all review | Structured: good + findings + recs |
| `bias` | Strategy/research code, ML training | Statistical bias focus |
| `joins` | SQL-heavy pipeline code | daily_features JOIN focus |
| `bugs` | Python implementation review | Type/None/timezone/resource focus |
| `improvements` | Institutional improvement suggestions | Head-of-quant graded sections |

## Integration Points
- **`/m25-audit` skill**: Smart auto-detect slash command. Auto-triages all findings. Primary entry point.
- **Pre-commit hook [5/5]**: Runs on staged files, advisory only, never blocks
- **Health check**: Runs on last commit's changes, advisory only
- **On-demand scripts** (used by the skill internally):
  - `python scripts/tools/m25_auto_audit.py` — changed-file scanner
  - `python scripts/tools/m25_ml_audit.py` — full ML integration audit
  - `python scripts/tools/m25_audit.py <file> --mode <mode>` — single-file audit

## Auto-Triage Classification
| Verdict | Meaning | Action |
|---------|---------|--------|
| **TRUE** | Real issue confirmed by code reading | Fix or flag |
| **PARTIALLY TRUE** | Real concern but existing guards mitigate | Note residual risk |
| **FALSE POSITIVE** | M2.5 is wrong (existing guard, wrong line, cross-file blindness) | None |
| **WORTH EXPLORING** | Not a bug but genuine improvement suggestion | Add to research queue |

## NEVER Do This
- Implement an M2.5 suggestion without reading the actual code first
- Trust M2.5's line number citations without verifying
- Let M2.5 override a CLAUDE.md rule or project invariant
- Run M2.5 findings as automated fixes (no auto-apply)
- Present raw M2.5 output without auto-triage (always triage first)
- Evaluate a single quality gate in isolation (always check the full gate system)
