# Conditional Edge Framework Rollout

**Date:** 2026-04-22
**Branch:** `wt-codex-conditional-edge-framework`
**Purpose:** turn the repo's emerging "most edges are conditional" insight into project doctrine, prereg structure, and one concrete bounded implementation study.

## Problem

The repo has strong statistical controls, but it still too often asks the wrong first question:

- "is this standalone?"
- "does this one filter rescue the parent?"
- "is the top bucket deployable already?"

That creates false kills, weak promotions, and repeated tunnel vision.

## Design

Ship three layers together:

1. **Authoritative doctrine**
   Add `docs/institutional/conditional-edge-framework.md` and route other research docs to it.

2. **Prereg semantics**
   Extend the hypothesis template so conditional studies declare:
   - `research_question_type`
   - role block per hypothesis
   - parent
   - comparator
   - primary metric
   - promotion target

3. **Light enforcement**
   Extend `trading_app/hypothesis_loader.py` so role-aware preregs fail loud when those fields are missing or malformed.

4. **Concrete proof path**
   Pre-register and implement the bounded `PR48` role study:
   - `MNQ`, `MES`, `MGC`
   - `O5`, `E2`, `CB1`, `RR1.5`
   - compare `parent`, `Q4+Q5`, `Q5`, `continuous quintile sizer`
   - sacred OOS `2026-01-01` onward untouched

## Non-goals

- no broad new discovery scan
- no changes to trading runtime
- no promotion to live config
- no reinterpretation of old result docs without fresh canonical replay

## Acceptance criteria

- new framework doc exists and is linked from project research docs
- prereg template supports role-aware conditional studies
- loader validates new role-aware preregs fail-closed
- one bounded `PR48` implementation study is pre-registered and runnable from repo code
- tests for loader enforcement pass

## Verification plan

- targeted loader tests
- targeted script compile / execution for the bounded study
- manual markdown sanity check on new docs
