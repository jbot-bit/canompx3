# MNQ 15m E2 CB1 RR=1.0 unfiltered lane discovery — v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-mnq-15m-rr10-unfiltered-lane-discovery-v1.yaml`

**Script:** `research/mnq_15m_rr10_unfiltered_lane_discovery_v1.py`

**Cost model:** MNQ $2.74 RT (from `pipeline.cost_model.COST_SPECS`)

## Per-cell verdicts

| Session | N_IS | Raw ExpR IS | Net ExpR IS | t | p(1t) | q(BH) | WFE | N_OOS | Net ExpR OOS | OOS/IS | Worst Era | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CME_PRECLOSE | 289 | +0.2192 | +0.2192 | +4.063 | 0.0000 | 0.0000 | 1.034 | 15 | +0.1640 | +0.75 | +0.0989 | **CME_PRECLOSE_RESEARCH_SURVIVOR** |
| NYSE_OPEN | 1545 | +0.0974 | +0.0974 | +3.958 | 0.0000 | 0.0000 | 1.340 | 55 | +0.2574 | +2.64 | -0.0433 | **NYSE_OPEN_CANDIDATE_READY** |
| US_DATA_1000 | 1594 | +0.0966 | +0.0966 | +4.037 | 0.0000 | 0.0000 | 0.859 | 60 | +0.1799 | +1.86 | +0.0096 | **US_DATA_1000_CANDIDATE_READY** |

## Gate summary

H1 = Net ExpR > 0 AND HC3 t ≥ +3.0 AND p < 0.05 (Chordia with-theory).
C6 (WFE ≥ 0.50), C8 (Net_OOS / Net_IS ≥ 0.40, N_OOS ≥ 50), C9 (no era N≥50 with ExpR < -0.05).
BH-FDR q-value reported at K=4 (pre-reg K framing).

- CANDIDATE_READY (2): ['NYSE_OPEN', 'US_DATA_1000']
- RESEARCH_SURVIVOR (1): ['CME_PRECLOSE']
- KILL_IS (0): []
- SCAN_ABORT (0): []

## Follow-on actions (NOT taken by this pre-reg)

- `CANDIDATE_READY` cells → Phase 0 validated_setups promotion flow (C5 DSR / C11 account-death MC / C12 SR-monitor). Requires its own pre-reg and runner.
- `RESEARCH_SURVIVOR` cells → document deploy-gate failure mode for future research; do NOT deploy.
- `KILL_IS` cells → closed under this pre-reg.
- Monotonic-rank sizing overlay on top of any passing cell → SEPARATE pre-reg after candidate promotion.

## Not done by this result

- No write to validated_setups / edge_families / lane_allocation / live_config.
- Does NOT modify Q4-band MNQ contract (PR #43).
- Does NOT test other RRs, apertures, or filtered overlays.
- Does NOT apply monotonic-rank sizing (separate follow-on).
