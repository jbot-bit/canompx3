# MNQ NYSE_CLOSE RR1.0 Follow-up

Date: 2026-04-23

## Scope

Close the staged RR1.0 failure-mode / governance follow-up for `MNQ NYSE_CLOSE` without reopening a broad filter sweep.

Inputs used:

- `gold.db::orb_outcomes`
- `gold.db::experimental_strategies`
- `docs/audit/results/2026-04-19-mnq-nyse-close-failure-mode-audit.md`
- locked repo hypotheses under `docs/audit/hypotheses/`
- `trading_app/portfolio.py` raw-baseline exclusion surface

## Broad RR1.0 Baseline

| Aperture | N IS | Avg IS | p IS | Pos Years | N OOS | Avg OOS |
|---|---:|---:|---:|---:|---:|---:|
| O5 | 805 | +0.0838 | 0.0083 | 5/7 | 42 | +0.4832 |
| O15 | 327 | +0.1169 | 0.0217 | 6/7 | 17 | +0.3490 |
| O30 | 197 | +0.1384 | 0.0334 | 5/7 | 6 | +0.9241 |

## RR1.0 Surface Actually Tested

| Strategy | Filter | Aperture | N | ExpR | p | Status |
|---|---|---:|---:|---:|---:|---|
| MNQ_NYSE_CLOSE_E2_RR1.0_CB1_GAP_R015 | GAP_R015 | 5 | 92 | +0.1744 | 0.0682 | REJECTED |
| MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G5_NOFRI | ORB_G5_NOFRI | 5 | 640 | +0.0734 | 0.0415 | REJECTED |
| MNQ_NYSE_CLOSE_E2_RR1.0_CB1_OVNRNG_100 | OVNRNG_100 | 5 | 267 | +0.1510 | 0.0070 | REJECTED |

## Locked Native Candidates Present In Repo

| Source | Hypothesis | Filter | Aperture | Already in RR1.0 experimental table? |
|---|---|---|---:|---|
| docs/audit/hypotheses/2026-04-09-mnq-comprehensive.yaml | MNQ NYSE_CLOSE G8 RR1.0 | ORB_G8 | 5 | False |
| docs/audit/hypotheses/2026-04-13-wave4-session-rr-expansion.yaml | NYSE_CLOSE ORB size gate | ORB_G8 | 5 | False |
| docs/audit/hypotheses/2026-04-13-wave4-session-rr-expansion.yaml | NYSE_CLOSE cost gate | COST_LT12 | 5 | False |
| docs/audit/hypotheses/2026-04-13-wave4-session-rr-expansion.yaml | NYSE_CLOSE cross-asset vol | X_MES_ATR60 | 5 | False |

## Alternate Framings Checked

| Frame | Support | Verdict |
|---|---|---|
| Null edge | Broad RR1.0 is positive on O5, O15, and O30 pre-2026. | REJECT |
| Pure policy blocker | Portfolio builders exclude NYSE_CLOSE, but the native RR1.0 candidate path was also never executed. | PARTIAL |
| Historical narrowness / missed execution | RR1.0 experiments are limited to apertures [5], NO_FILTER tested=False, ORB_G8 executed=False. | PRIMARY |

## Decision

**Outcome:** `CONTINUE with narrow prereg`

Freeze and execute the exact MNQ NYSE_CLOSE ORB_G8 RR1.0 prereg before any new sweep or policy change.

Why:

- Broad RR1.0 is alive on all three audited apertures, so a broad-family kill would be false.
- RR1.0 experimentation is still only three O5 filters (`GAP_R015`, `OVNRNG_100`, `ORB_G5_NOFRI`).
- `NO_FILTER`, `O15`, and `O30` were never taken through the RR1.0 experimental path.
- The repo already contains two independent locked `ORB_G8` NYSE_CLOSE RR1.0 hypotheses, but no corresponding RR1.0 experimental row or durable result doc.
- `trading_app/portfolio.py` still excludes `NYSE_CLOSE` from both raw-baseline builders, so direct promotion without a narrow exact prereg would overstep the evidence.

## EV-Based Next Move

Highest-EV path: execute one exact native prereg, `MNQ NYSE_CLOSE ORB_G8 RR1.0`.

Not recommended now:

- no direct portfolio unblock of raw `NYSE_CLOSE` baselines
- no new broad RR1.0 sweep
- no COST / cross-asset follow-up before the native ORB-size path is closed
