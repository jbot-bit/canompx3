# US_DATA_1000 Long NOT_F6 Candidate Lane — v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-usdata1000-long-not-f6-candidate-lane-v1.yaml`

**Script:** `research/usdata1000_long_not_f6_candidate_lane_v1.py`

**Family K:** 2

**Gates:** H1 (t≥+3.0, BH-FDR q<0.05), C6 (WFE≥0.50), C8 (OOS/IS≥0.40 and N_OOS≥50), C9 (era stability)

## Summary counts

- CANDIDATE_READY: **0**
- RESEARCH_SURVIVOR: 2
- KILL_IS: 0

## Full result table

| RR | N_IS | Net ExpR | t | q(BH) | WFE | N_OOS | Net OOS | OOS/IS | Worst Yr | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0 | 368 | +0.1583 | +3.277 | 0.0006 | 1.260 | 13 | -0.0982 | -0.62 | -0.0050 | **RESEARCH_SURVIVOR** |
| 1.5 | 363 | +0.2261 | +3.631 | 0.0003 | 1.173 | 13 | -0.2509 | -1.11 | -0.0161 | **RESEARCH_SURVIVOR** |

## Interpretation

- `NOT_F6_INSIDE_PDR` is a **research-provisional candidate lane**, not a live-ready lane: both RR cells clear H1/C6/C9, but both fail C8 because 2026 OOS remains too thin and does not yet support the required OOS/IS gate.
- This confirms the role-design conclusion: `NOT_F6` is the right primary lane route, but it is still awaiting enough forward OOS.

## Not done by this result

- No writes to `validated_setups`, `edge_families`, or `lane_allocation`.
- No deployment or capital action.
- No shadow execution contract yet; that is the next bounded step if continuing.