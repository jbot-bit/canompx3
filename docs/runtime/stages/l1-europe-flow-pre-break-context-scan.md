---
slug: l1-europe-flow-pre-break-context-scan
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-22
updated: 2026-04-23
task: Execute the next stage after PR #67's prereg: run the bounded L1 EUROPE_FLOW pre-break-context scan exactly as frozen in the prereg, without re-opening banned break-bar or ATR-normalized variants.
---

# Stage: L1 EUROPE_FLOW pre-break-context scan

## Question

PR #67 already landed the doctrine fix and the prereg for the L1 EUROPE_FLOW pre-break-context path. The next stage is now execution, not more framing:

> when the frozen admissible pre-break-context features are scanned exactly as preregistered, does any small-K L1 EUROPE_FLOW family survive honestly?

## Scope Lock

- Use the frozen prereg from the merged L1 docs
- Session in scope: `EUROPE_FLOW`
- Lane framing in scope: L1 only
- Admissible features limited to pre-ORB-end / pre-break-context features
- Explicitly exclude:
  - `break_*` columns
  - ATR-normalized ratio variants
  - pooled ML reframing

## Blast Radius

- Research-only:
  - one scan runner under `research/`
  - one result doc under `docs/audit/results/`
- No doctrine changes in this stage
- No production-code edits unless a separate bug is discovered

## Approach

1. Read the merged prereg and treat it as binding.
2. Implement only the scan required to execute that prereg.
3. Report:
   - K actually tested
   - fire-rate sanity
   - honest OOS treatment
   - whether anything survives or the path closes cleanly
4. If all cells die, close the path without reframing.

## Suggested Branch / PR

- Branch: `research/l1-europe-flow-pre-break-scan`
- PR title: `research(l1): execute EUROPE_FLOW pre-break-context prereg scan`

## Acceptance Criteria

1. The scan matches the merged prereg exactly.
2. No banned break-bar columns enter the predictor set.
3. Result doc says `CONTINUE`, `PARK`, or `KILL` for the prereg path.
4. Any surviving result is still treated as research output, not live deployment.

## Non-goals

- Not a rewrite of PR #67
- Not another doctrine correction
- Not a broad L1 portfolio rewrite

## Execution Outcome

- Executed on 2026-04-23 against the restored frozen prereg at `docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml`
- Runner: `research/l1_europe_flow_pre_break_context_scan.py`
- Result: `docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg.md`
- Outcome: `KILL`

Why:

- `pre_velocity_HIGH_Q3` did not reach raw significance (`delta_IS=+0.034`, `p=0.558`)
- `rel_vol_HIGH_Q3` showed IS lift (`delta_IS=+0.127`) but failed the frozen `K=2` family BH-FDR gate (`q=0.0537`) and flipped OOS sign (`delta_OOS=-0.396`)
- No admissible feature survived honestly, so the prereg path closes without reopening banned `break_*` or ATR-normalized replacement variants
