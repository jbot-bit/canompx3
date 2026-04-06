Split an oversized task into minimal safe stages: $ARGUMENTS

Use when: stage-gate classifies TOO BROAD, or user says "this is too much", "break this down"
Triggers: "split", "too broad", "break down", "decompose"

## STEP 1: WHY IS IT TOO BROAD?

State which conditions triggered (from stage-gate Step 4):
- [ ] Non-local cross-domain dependency
- [ ] Unclear acceptance criteria
- [ ] Unresolved canon/policy decision
- [ ] Downstream contamination of upstream work
- [ ] Multiple independently testable deliverables

## STEP 2: FIND THE SEAMS

Split priority (upstream before downstream):
1. Schema/config before code that consumes it
2. One instrument before all instruments
3. Pipeline truth before trading_app logic
4. Implementation before validation/rebuild chains
5. Each independently testable deliverable = its own stage

## STEP 3: OUTPUT STAGES

For each stage (max 4 — if more, split again):
```
Stage N: [one line]
  Domain: [code|DB|config|artifacts|docs]
  Files: [exact paths, ≤5 per stage]
  Depends: [Stage N-1 output, or "none"]
  Blocker: [what must be true first]
  Done when: [exact verification command + expected result]
  NOT this stage: [explicitly deferred items]
```

## STEP 4: PRESENT STAGE 1 ONLY

"Stage 1 is safe to execute. Stages 2-N deferred until Stage 1 completes and verifies."

If user approves → write Stage 1 as approved stage in `docs/runtime/stages/<slug>.md` → proceed via /stage-gate Step 5.

## REFUSE:
- Executing all stages at once
- Stages with >5 files each (split further)
- Downstream stages before upstream truth is verified
