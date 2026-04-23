---
name: discover
description: Research edge discovery and idea triage — routes through the institutional discovery protocol before any scan or build work
allowed-tools: Read, Grep, Glob, Bash
---
Research edge discovery and hypothesis triage: $ARGUMENTS

Use when: "discover", "scan for edges", "research [idea]", "find strategies",
"edge discovery", "new idea", "chart read", "hypothesis triage"

## Step 0: Discovery Front Door (MANDATORY)

Before ANY scan, prereg, or implementation talk, route the task through the
canonical discovery doctrine:

- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/STRATEGY_BLUEPRINT.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/mechanism_priors.md`
- `docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md`

The front-door question is NOT "what can I scan quickly?"

The front-door questions are:
- what is the correct object?
- what is knowable at decision time?
- what role is this best suited for?
- is this a current-stack test, architecture problem, new-data problem, or dead
  path?

## Step 1: Lock the Object Before Any Scan

Before proposing tests, explicitly classify:
- unit of analysis
- information horizon
- role mapping
- path type

If the idea is really a:
- conditioner
- allocator input
- confluence candidate
- execution translation problem

do NOT force it through a standalone framing first.

## Step 2: Use Canonical Truth Only

Discovery truth comes from:
- `bars_1m`
- `daily_features`
- `orb_outcomes`

Derived layers and docs may orient but are NOT proof.

Do not mix:
- post-trigger information into pre-trade E2 logic
- runtime limitations into signal invalidation
- shelf presence into deployment proof

## Step 3: Smallest Honest Next Move

If the path survives triage, the next move is usually:
- a narrow pre-registered hypothesis scope
- not a broad exploratory scan
- not implementation
- not prompt improvisation

## Step 4: Output Contract

Report using the structure in
`docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md`:
- correct object
- discovery map
- honest next tests
- park / kill list
- final recommendation

## Rules

- NEVER jump from idea to scan without locking the object first
- NEVER use derived layers as discovery proof
- NEVER collapse "not standalone" into "dead"
- NEVER collapse "not yet proven" into "alive"
- Preserve the sacred holdout from `2026-01-01`
- No scan recommendation is valid until it can be expressed as a
  pre-registered hypothesis scope
