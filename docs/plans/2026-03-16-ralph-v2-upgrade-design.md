# Ralph V2 — Autonomous Audit Upgrade Design

**Status:** IN PROGRESS
**Date:** 2026-03-16
**Purpose:** Make Ralph smarter after 111 iterations of diminishing returns

---

## Problem Statement

Ralph has scanned 165 files across 111 iterations. All CRITICAL/HIGH findings in production code were fixed by iteration ~72. The last 40 iterations have been LOW/mechanical fixes (f-strings, imports) in research scripts. Ralph has no awareness it's doing busywork, no cross-iteration learning, and no production-path prioritization.

## What Changes

Three tiers, increasing complexity. No schema changes. No trading logic changes. No pipeline data flow changes.

### Tier 1: Diminishing Returns Detector

**What:** After Step 1 audit, if all findings are LOW and `consecutive_low_only >= 3` in the ledger, Ralph emits a stop signal instead of fixing another f-string.

**Files touched:**
- MODIFY: `.claude/agents/ralph-loop.md` — add early exit logic after Step 1
- READ: `docs/ralph-loop/ralph-ledger.json` — check consecutive_low_only

**Blast radius:** Zero. Only affects Ralph's own behavior. No production code.

### Tier 2: Production-Path Weighting

**What:** A static analysis script that ranks files by import centrality. Ralph reads this at Step 0 and uses it as a tiebreaker among same-severity findings.

**Files touched:**
- CREATE: `scripts/tools/import_graph.py` — builds centrality JSON
- CREATE: `docs/ralph-loop/import_centrality.json` — output (gitignored, regenerated on demand)
- MODIFY: `.claude/agents/ralph-loop.md` — read centrality at Step 0, use in Step 2

**Blast radius:** Zero. New read-only tooling. Ralph uses it as advisory info.

### Tier 3: Cross-Iteration Intelligence

**What:** Replace append-only markdown history with structured JSON ledger. Ralph reads this at Step 0 for pattern awareness.

**Files touched:**
- CREATE: `scripts/tools/ralph_build_ledger.py` — migration script (one-time, also used for rebuilds)
- CREATE: `docs/ralph-loop/ralph-ledger.json` — structured iteration data
- MODIFY: `.claude/agents/ralph-loop.md` — read ledger at Step 0, update ledger at Step 4
- KEEP: `docs/ralph-loop/ralph-loop-history.md` — still append-only for human readability

**Blast radius:** Low. Only Ralph's own docs. History.md preserved as-is.

---

## Implementation Steps

### Step 1: Build import_graph.py (Tier 2)
- Scan pipeline/, trading_app/, scripts/ for internal imports
- Count in-degree per module
- Tier: critical (10+), high (5-9), medium (2-4), low (0-1)
- Output JSON to docs/ralph-loop/import_centrality.json
- Run and verify output makes sense

### Step 2: Build ralph_build_ledger.py (Tier 3)
- Parse ralph-loop-history.md into structured JSON
- Extract: iteration count, finding types, fix rates, consecutive LOW count
- Output to docs/ralph-loop/ralph-ledger.json
- Run and verify counts match history

### Step 3: Update ralph-loop.md (Tiers 1-3)
- Step 0: add reads for ledger JSON + centrality JSON
- Step 1: add diminishing returns check after audit
- Step 2: add centrality as tiebreaker
- Step 4: add ledger update (append iteration to JSON)

### Step 4: Verify
- Run Ralph once with `/ralph` to confirm it reads the new files
- Confirm diminishing returns detector fires (current state is 10+ consecutive LOW)
- Confirm centrality tiebreaker works (production files rank higher)

### Step 5: Commit
- One commit with all changes: `docs: Ralph V2 — diminishing returns + centrality + JSON ledger`

---

## Test Strategy

- No new tests needed (Ralph is an agent prompt, not production code)
- Verification: run one Ralph iteration and confirm:
  1. It reads the ledger and centrality files without error
  2. It reports diminishing returns (current consecutive_low_only >= 10)
  3. History.md still gets appended normally

## Risks

- **JSON ledger parse errors**: history.md is semi-structured markdown. Parser might miss some iterations. Mitigation: log unparsed lines, don't fail on partial data.
- **Diminishing returns false positive**: what if a HIGH finding appears in a file Ralph hasn't scanned yet? Mitigation: only trigger if `consecutive_low_only >= 3 AND all known unscanned files are research/`.
- **Token budget**: reading two extra JSON files costs turns. Mitigation: both files are small (<5KB). One extra read at Step 0.

## Future (not in this PR)

- Tier 4: Specialist panel (3-4 focused Haiku subagents)
- Tier 5: SICA meta-improvement loop (self-modifying prompt)
- Integration with pr-review-toolkit specialist agents
