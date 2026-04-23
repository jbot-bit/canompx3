---
slug: mnq-nyse-close-rr10-followup
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-22
updated: 2026-04-23
task: Run the narrow RR1.0 governance/failure-mode follow-up for MNQ NYSE_CLOSE that the blocker audit called for. Treat the family as alive-but-blocked, not fresh void and not dead.
---

# Stage: MNQ NYSE_CLOSE RR1.0 follow-up

## Question

`docs/audit/results/2026-04-19-mnq-nyse-close-failure-mode-audit.md` already narrowed the issue:

- broad RR1.0 baseline is alive
- prior attempted filters were narrow and unstable
- the real issue is a pathway / blocker problem

The next stage is:

> run a narrow RR1.0 native governance/failure-mode follow-up on the broad session family, instead of another random filter sweep.

## Scope Lock

- Instrument/session in scope: `MNQ NYSE_CLOSE`
- Primary target: `RR1.0`
- Focus on blocker/failure-mode and native governance framing
- No broad filter family sweep
- No multi-session adjacent fishing

## Blast Radius

- Research-only:
  - one runner under `research/`
  - one result doc under `docs/audit/results/`
- Comparison to current blocker surfaces is allowed
- No production portfolio changes in this stage

## Approach

1. Start from the blocker audit as upstream truth.
2. Audit the broad RR1.0 family with native governance framing:
   - what broad baseline is actually alive
   - why prior filter attempts failed
   - whether the blocker is methodological, policy, or merely historical narrowness
3. End with one honest recommendation:
   - `CONTINUE with narrow prereg`
   - `PARK pending blocker removal`
   - `KILL broad-family follow-up`

## Suggested Branch / PR

- Branch: `research/mnq-nyse-close-rr10-followup`
- PR title: `research(nyse-close): narrow rr1.0 failure-mode follow-up`

## Acceptance Criteria

1. RR1.0 broad-family status is re-stated from canonical truth.
2. Prior failed filter attempts are treated as blocker evidence, not forgotten.
3. Result doc ends in one narrow next move, not a new exploration void.
4. No random filter sweep is run in this stage.

## Non-goals

- Not a fresh NYSE_CLOSE rediscovery campaign
- Not a deployment recommendation
- Not a broad session-family rescan across all RRs and apertures

## Execution Outcome

- Executed on 2026-04-23 via `research/mnq_nyse_close_rr10_followup.py`
- Result: `docs/audit/results/2026-04-23-mnq-nyse-close-rr10-followup.md`
- Outcome: `CONTINUE with narrow prereg`

Why:

- Broad RR1.0 remains positive on `O5`, `O15`, and `O30`, so the family is not dead.
- The actual RR1.0 experimental surface is still only three narrow `O5` rows: `GAP_R015`, `OVNRNG_100`, and `ORB_G5_NOFRI`.
- The repo already contains two independent locked `ORB_G8` RR1.0 NYSE_CLOSE hypotheses, but no matching RR1.0 experimental row or durable result.
- The next honest step is now explicit: `docs/runtime/stages/mnq-nyse-close-orbg8-rr10-prereg.md`
