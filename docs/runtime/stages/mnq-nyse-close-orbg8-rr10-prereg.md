---
slug: mnq-nyse-close-orbg8-rr10-prereg
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-23
updated: 2026-04-23
task: Execute the exact native NYSE_CLOSE follow-through identified by the RR1.0 governance audit: run the locked MNQ NYSE_CLOSE ORB_G8 RR1.0 prereg and close the path honestly.
---

# Stage: MNQ NYSE_CLOSE ORB_G8 RR1.0 prereg

## Question

The 2026-04-23 follow-up closed the blocker framing and narrowed the honest next
move:

> does the already-locked native `ORB_G8` gate survive on the exact `MNQ NYSE_CLOSE O5 E2 CB1 RR1.0` lane, or does the family stay alive-but-unpromoted?

## Scope Lock

- Instrument/session in scope: `MNQ NYSE_CLOSE`
- Exact lane: `O5`, `E2`, `CB1`, `RR1.0`
- Exact filter: `ORB_G8`
- Use canonical filter delegation only
- No sibling filter sweep (`COST_LT12`, `X_MES_ATR60`, `NO_FILTER`, wider apertures)

## Blast Radius

- Research-only:
  - one execution runner under `research/`
  - one result doc under `docs/audit/results/`
- No portfolio/live/profile changes in this stage
- No broad NYSE_CLOSE family rewrite

## Approach

1. Treat `docs/audit/hypotheses/2026-04-23-mnq-nyse-close-orbg8-rr10-prereg.yaml` as binding.
2. Replay the exact lane with canonical `ORB_G8` delegation.
3. Report IS, OOS, and era-stability outcomes clearly.
4. End with one honest verdict: `CONTINUE`, `PARK`, or `KILL`.

## Suggested Branch / PR

- Branch: `research/mnq-nyse-close-orbg8-rr10`
- PR title: `research(nyse-close): execute ORB_G8 rr1.0 prereg`

## Acceptance Criteria

1. Exact lane identity is preserved.
2. No new hypothesis shopping occurs after metrics inspection.
3. Result doc closes the `ORB_G8` path honestly.
4. Any survival remains research-only until a separate promotion step.

## Non-goals

- Not a direct raw-baseline promotion
- Not a broad RR1.0 rescan
- Not a portfolio policy change

## Execution Outcome

- Executed on 2026-04-23 via `research/mnq_nyse_close_orbg8_rr10_prereg.py`
- Result: `docs/audit/results/2026-04-23-mnq-nyse-close-orbg8-rr10-prereg.md`
- Outcome: `KILL`

Why:

- The exact `ORB_G8` path is statistically strong in-sample (`ExpR_on=+0.1107`, `t=3.285`, `p=0.0005`) but fails the prereg's era-stability gate.
- `2025` on-signal performance is negative at meaningful size (`N_on=132`, `ExpR=-0.0697`), triggering Criterion 9-style kill logic.
- 2026 OOS does not offer a real selector test because `ORB_G8` fires on every observed row (`42/42`).
- This closes the native `ORB_G8` filter path. It does not kill the broad RR1.0 NYSE_CLOSE family; the remaining honest branch is role audit, not filter rescue.
