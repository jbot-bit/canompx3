---
pooled_finding: false
audit_target: "MNQ 4th uncorrelated lane portfolio addition under C11"
auditor_context: opus-4-8 research-track (charter yes-the-baton-vectorized-valiant.md)
canonical_layers: [orb_outcomes, daily_features]
db_freshness: "orb_outcomes via canonical loader, as_of 2026-06-04"
holdout_date: "2026-01-01"
base_commit_sha: "9429c540"
pre_registration: docs/audit/hypotheses/2026-06-05-mnq-fourth-lane-portfolio-addition.yaml
verdict: "CONTINUE (one survivor) — research-only, NO promotion/arming"
---

# MNQ 4th-lane portfolio addition under C11 — Result

**Mission (charter):** does ADDING a 4th uncorrelated lane improve portfolio EV
under C11 BETTER than sizing up? Sizing-up is PROVEN dead (2-micro op_pass 29.5%,
fails C11 operational gate). **Research-only. No `prop_profiles.py` / live_config /
schema / DB writes. Nothing promoted, nothing armed.**

## VERDICT: CONTINUE — one survivor: `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED`

Adding this one CME_PRECLOSE lane to the 3-lane cap_x0.80 book **lowers** the
book's strict max-90d-DD ($1,594 → $1,129, **−$465**) while **adding** return
(+$345 marginal p50 PnL) and holding op_pass at 100%. This is the sub-linearity
thesis in its strongest form — **negative marginal drawdown** from a
negatively-correlated leg. Sizing-up (linear DD, op_pass → 29.5%) cannot do this.

The charter's hypothesis "a 4th lane beats sizing-up" is **confirmed by canonical
measurement**. This is a CONTINUE to the next stage (independent adversarial
audit + the bracket-parity gate `9b3fc530` that blocks ALL C11 deployment), NOT
a deploy decision.

## Source-traced setup (no inference)

| Fact | Value | Source |
|---|---|---|
| C11 budget | **$1,800** | `effective_strict_dd_budget` = 0.90 × $2,000 MLL (express fraction) |
| Baseline book | 3-lane cap_x0.80 | COMEX_SETTLE(49.8×0.8), US_DATA_1000(143.2×0.8), TOKYO_OPEN(44.2×0.8) |
| Baseline DD / op_pass | $1,594.03 / 99.9% | canonical `evaluate_profile_survival(write_state=False)` |
| Contracts | 1 micro/lane (hardcoded) | `account_survival._load_lane_trade_paths` `contracts_per_trade=1` |
| Stop | 0.75 | profile-level |

**Note:** the harness `c11_matrix/c11_unified_levers.py` carried a STALE `$1,600`
budget. Source-trace shows the express profile uses **$1,800** (0.90 fraction).
All gates below use $1,800.

## Candidate universe (Step 0 grounding)

- Graveyard check (research-catalog MCP, NO-GO/KILL/PARK): `EUROPE_FLOW_OVNRNG_100`
  surveyed **DEAD** (6 blockers); MNQ per-session edge attempts heavily KILLed.
- `2026-06-03-brisbane-morning-orb-coverage-edge-audit.md`: "668 promotable FIT MNQ
  candidates exist; only 3 allocated. **Bottleneck = ALLOCATION, not discovery.**"
- `2026-04-19-validated-shelf-vs-live-deployment-audit.md`: allocator already gates
  rho=1.0 same-session substitutes; the deployable shelf's "extras" are mostly
  same-session dupes.
- Candidates = validated_setups active MNQ lanes at sessions DISTANT from the 3
  deployed (CME_PRECLOSE 06:00, EUROPE_FLOW 18:00, NYSE_CLOSE 07:00, NYSE_OPEN
  00:30, SINGAPORE_OPEN 11:00), ExpR>0.12, N>=100. K=9 ≤ MinBTL bound (PASS).

## Step 1 — Correlation screen (decisive pre-filter)

All 9 candidates measured near-zero / negative daily correlation vs the deployed
book (canonical per-day $ series):

| Candidate | corr_all | first_day | note |
|---|---|---|---|
| CME_PRECLOSE ATR_P70 | −0.070 | 2019-05 | |
| CME_PRECLOSE COST_LT08 | −0.045 | 2019-05 | |
| **CME_PRECLOSE VWAP_BP_ALIGNED** | **−0.026** | **2019-05** | survivor |
| CME_PRECLOSE RR2.0 ATR_P70 | −0.054 | 2019-05 | |
| EUROPE_FLOW PDR_R105 | −0.037 | 2019-05 | |
| EUROPE_FLOW COST_LT08 | −0.024 | 2019-08 | window-trunc |
| NYSE_CLOSE X_MES_ATR70 | −0.024 | 2019-08 | window-trunc |
| NYSE_OPEN X_MGC_ATR70 | +0.007 | **2022-09** | window-trunc (kills 2022 DD) |
| SINGAPORE_OPEN ATR_P70 | +0.082 | 2019-05 | TOKYO-adjacent |

**Window-truncation trap:** `_load_profile_daily_scenarios` sets
`common_start = max(lane first days)`. A candidate starting after 2019-05 slices
the book's window and HIDES the 2022 binding-year DD. Three candidates
disqualified on this alone.

## Step 2 — Dual-gate (canonical, BOTH C11 terms)

`evaluate_profile_survival(write_state=False, n_paths=10000, seed=20260605)` —
computes strict DD term AND operational MC term exactly as the live gate:

| Candidate | DD$ | ≤$1800 | op% | ≥70 | breach | GATE | mDD$ | mPnL$ | window |
|---|---|---|---|---|---|---|---|---|---|
| **CME_PRECLOSE VWAP_BP_ALIGNED** | **1129** | Y | **100.0** | Y | 0 | **PASS** | **−465** | **+345** | full |
| EUROPE_FLOW PDR_R105 | 1571 | Y | 99.9 | Y | 0 | PASS | −23 | +290 | full |
| CME_PRECLOSE ATR_P70 | 1594 | Y | 95.5 | Y | 1 | FAIL | +0 | +292 | full |
| CME_PRECLOSE COST_LT08 | 1341 | Y | 91.2 | Y | 2 | FAIL | −253 | +405 | full |
| CME_PRECLOSE RR2.0 ATR_P70 | 1594 | Y | 95.5 | Y | 1 | FAIL | +0 | +213 | full |
| SINGAPORE_OPEN ATR_P70 | 1594 | Y | 91.9 | Y | 2 | FAIL | +0 | +304 | full |
| EUROPE_FLOW COST_LT08 | 1457 | Y | 99.9 | Y | 0 | (PASS) | −137 | +321 | **TRUNC** |
| NYSE_CLOSE X_MES_ATR70 | 1364 | Y | 91.3 | Y | 2 | FAIL | −230 | +160 | TRUNC |
| NYSE_OPEN X_MGC_ATR70 | 1352 | Y | 83.5 | Y | 2 | FAIL | −242 | +977 | TRUNC |

**The op-gate is the killer** (same mechanism that killed sizing-up): most
candidates inject daily-loss-limit breaches (`breach_days>0` → strict term fails)
even though the DD-magnitude term passes. Only 2 full-window candidates clear BOTH.

## Step 3 — Adversarial verification (the 2 PASS candidates)

| Test | CME_PRECLOSE VWAP_BP_ALIGNED | EUROPE_FLOW PDR_R105 |
|---|---|---|
| 2022 binding-year DD (baseline $1,782) | **$1,082** (−$700) ✓ | $1,578 (≈flat) |
| Holdout 2025+ DD | **$975** ✓ | **$1,606** (doubled vs $802 baseline) ✗ |
| Cost +20% (net ×0.80) DD / breach | **$899 / 0** ✓ | $1,285 / **3 breaches** ✗ |
| Drop-1-year jackknife (DD range) | **$1,082–$1,124** stable ✓ | n/a |

- **CME_PRECLOSE VWAP_BP_ALIGNED survives all four.** The DD reduction holds in
  the actual 2022 binding year (not a window relocation), is stable to dropping
  any single year (not luck-driven), and the breach-free strict term holds under
  +20% cost stress.
- **EUROPE_FLOW PDR_R105 FAILS verification:** doubles 2025/holdout DD, and cost
  stress introduces 3 daily-loss breaches → strict term would fail. **PARK/reject.**

## Honest caveats (not papered over)

1. **All candidates carry `n_trials_at_discovery=36372`** — the same MinBTL-at-
   discovery concern (blocker B4) that contributed to the EUROPE_FLOW_OVNRNG_100
   kill. This is a property of the WHOLE shelf, including the 3 already-deployed
   lanes. The test here is **marginal portfolio DD-efficiency**, not fresh
   standalone re-validation. The survivor is NOT independently re-validated as a
   new edge; it is a shelf lane that improves the book's DD-per-dollar.
2. **`validated_setups` is a derived layer** (banned for truth-finding) — used
   ONLY to enumerate candidates. All DD/EV/op_pass truth came from the canonical
   `account_survival` loader on `orb_outcomes`+`daily_features`.
3. **The adversarial harness's absolute baseline ($1,782, union-of-fire-days
   calendar) differs from the canonical $1,594** (common_start + daily_features
   calendar). The canonical dual-gate ($1,129 / op 100%) is authoritative for
   absolute numbers; the adversarial harness is valid for the relative/per-window
   deltas, which all agree in direction.
4. **No deploy claim.** C11 deployment is independently blocked by the open
   bracket-parity audit `9b3fc530` and `prop_profiles.py` is peer-owned.

## What this means for "max return" (charter answer)

- **Sizing-up: dead** (re-confirmed — op-gate collapses). 
- **A 4th uncorrelated lane: ALIVE** — and structurally superior. The recommended
  candidate doesn't just add sub-linear DD; it nets DD DOWN ($1,594→$1,129) while
  adding return. That is the diversification dividend the charter predicted.
- The bottleneck the 04-19 audit named (allocation, not discovery) is exactly
  what this resolves: a validated shelf lane, allocated as a 4th leg.

## Next stage (NOT done here — Tier B, gated)

1. Independent adversarial audit of this result (separate context).
2. Bracket-parity audit `9b3fc530` must close (blocks all C11 deployment).
3. Then — and only then — operator GO + `prop_profiles.py` 4-lane book (peer-owned)
   + re-run canonical survival + live-readiness. **None of that is authorized here.**

## Reproduction (read-only)

```
c11_matrix/fourth_lane_screen.py        # correlation screen
c11_matrix/fourth_lane_dual_gate.py     # canonical dual-gate (both C11 terms)
c11_matrix/fourth_lane_adversarial.py   # per-year / holdout / cost / jackknife
```
Pre-reg: `docs/audit/hypotheses/2026-06-05-mnq-fourth-lane-portfolio-addition.yaml`
(MinBTL gate PASS, all_passed:true).
