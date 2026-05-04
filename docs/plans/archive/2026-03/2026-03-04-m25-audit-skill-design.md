---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# M2.5 Audit Skill Design

**Date:** 2026-03-04
**Status:** Approved
**Command:** `/m25-audit`

## Overview

Unified slash command for MiniMax M2.5 second-opinion audits with automatic triage. Smart auto-detect routes to the right mode without requiring explicit arguments. Every mode ends with Claude verifying each finding against actual code before presenting results.

## Mode Routing

```
/m25-audit                  -> auto-detect from git diff + conversation context
/m25-audit <custom prompt>  -> institutional verdict mode
```

### Auto-Detection Rules (priority order)
1. `$ARGUMENTS` contains a custom prompt (>20 chars, not a filepath) -> **verdict** mode
2. `$ARGUMENTS` is a filepath -> **single-file** audit with MODE_MAP
3. `git diff --name-only HEAD` includes `trading_app/ml/` files -> **ml** mode
4. `git diff --name-only HEAD` includes `pipeline/` or `trading_app/` files -> **quick** mode
5. No changes detected -> inform user, suggest explicit mode

### Internal Modes

| Mode | Script | Files Sent | Prompt | Output |
|------|--------|-----------|--------|--------|
| quick | `m25_auto_audit.py` | Changed `.py` files | Per-file mode via MODE_MAP | Per-file CLEAN/FINDINGS |
| ml | `m25_ml_audit.py` | All `trading_app/ml/*.py` + 5 context docs | Domain-expert ML audit prompt | Full ML integration audit |
| verdict | `m25_audit.py` (audit function) | Context docs + relevant code | User-provided custom prompt | Institutional-grade review |
| single-file | `m25_audit.py` | Specified file | Best mode via MODE_MAP | Single file audit |

## Auto-Triage Loop

After every M2.5 run, Claude performs verification:

1. Parse each finding by severity (CRITICAL/HIGH/MEDIUM/INFO)
2. For each finding:
   - Read the actual code at cited line numbers (M2.5 is often off by 5-20 lines)
   - Trace execution path to confirm/deny the claim
   - Check if existing guards (try/except, if-checks, type constraints) handle it
   - Cross-reference CLAUDE.md rules and project invariants
3. Build triage table with columns: #, Finding, M2.5 Severity, Claude Verdict, Action
4. Save to `research/output/m25_triage_<YYYYMMDD_HHMM>.md`
5. Present table to user

### Verdict Categories
- **TRUE** - Real issue, action needed
- **PARTIALLY TRUE** - Real concern but mitigated by existing guards
- **FALSE POSITIVE** - M2.5 wrong (expected ~70% of the time)

## Downstream Effects

| Trigger | Action |
|---------|--------|
| TRUE finding affects ML memory | Update relevant file in `.claude/projects/.../memory/` |
| TRUE finding suggests new drift pattern | Flag for `check_drift.py` addition |
| TRUE finding affects config | Note for manual review (never auto-edit config) |
| Finding contradicts CLAUDE.md | CLAUDE.md wins, mark FALSE POSITIVE |
| Triage changes research conclusions | Flag for `sync_pinecone.py` |

## Files

| File | Action |
|------|--------|
| `.claude/commands/m25-audit.md` | **NEW** - The skill |
| `.claude/rules/m25-audit.md` | **UPDATE** - Add skill reference |
| `scripts/tools/m25_audit.py` | No changes |
| `scripts/tools/m25_auto_audit.py` | No changes |
| `scripts/tools/m25_ml_audit.py` | No changes |

## Constraints

- M2.5 is advisory only (~70% false positive rate)
- Authority: CLAUDE.md > Claude Code > M2.5 suggestions
- Never auto-apply fixes without verification
- Never let M2.5 override project invariants
- All M2.5 scripts require `MINIMAX_API_KEY` in `.env`
