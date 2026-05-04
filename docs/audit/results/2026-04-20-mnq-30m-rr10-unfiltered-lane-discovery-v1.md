# MNQ 30m E2 CB1 RR=1.0 unfiltered lane discovery — v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-mnq-30m-rr10-unfiltered-lane-discovery-v1.yaml`

**Script:** `research/mnq_30m_rr10_unfiltered_lane_discovery_v1.py`

**Cost model:** MNQ $2.74 RT (from `pipeline.cost_model.COST_SPECS`)

## Per-cell verdicts

| Session | N_IS | Raw ExpR IS | Net ExpR IS | t | p(1t) | q(BH) | WFE | N_OOS | Net ExpR OOS | OOS/IS | Worst Era | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| US_DATA_1000 | 1388 | +0.1038 | +0.1038 | +4.009 | 0.0000 | 0.0001 | 1.056 | 44 | +0.1652 | +1.59 | +0.0509 | **US_DATA_1000_RESEARCH_SURVIVOR** |
| NYSE_OPEN | 1270 | +0.1071 | +0.1071 | +3.931 | 0.0000 | 0.0001 | 1.363 | 42 | +0.3682 | +3.44 | -0.0129 | **NYSE_OPEN_RESEARCH_SURVIVOR** |
| COMEX_SETTLE | 1384 | +0.0490 | +0.0490 | +1.921 | 0.0275 | 0.0275 | 2.215 | 50 | +0.0936 | +1.91 | -0.1188 | **COMEX_SETTLE_KILL_IS** |
| NYSE_CLOSE | 197 | +0.1384 | +0.1384 | +2.142 | 0.0167 | 0.0223 | 0.746 | 6 | +0.9241 | +6.68 | - | **NYSE_CLOSE_KILL_IS** |

## Gate summary

H1 = Net ExpR > 0 AND HC3 t ≥ +3.0 AND p < 0.05 (Chordia with-theory).
C6 (WFE ≥ 0.50), C8 (Net_OOS / Net_IS ≥ 0.40, N_OOS ≥ 50), C9 (no era N≥50 with ExpR < -0.05).
BH-FDR q-value reported at K=4 (pre-reg K framing).

- CANDIDATE_READY (0): []
- RESEARCH_SURVIVOR (2): ['US_DATA_1000', 'NYSE_OPEN']
- KILL_IS (2): ['COMEX_SETTLE', 'NYSE_CLOSE']
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
