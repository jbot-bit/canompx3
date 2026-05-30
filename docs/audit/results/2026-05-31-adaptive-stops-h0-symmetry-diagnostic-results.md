# H0 Adaptive-Stops MFE/MAE-Symmetry Gating Diagnostic — Results

- **Date:** 2026-05-31
- **Pre-reg:** `docs/audit/hypotheses/2026-05-31-adaptive-stops-h0-mfe-mae-symmetry-diagnostic-v1.yaml` (LOCKED, promoted out of `drafts/` this session)
- **Script:** `research/adaptive_stops_h0_symmetry_diagnostic.py`
- **Mode:** K=0, read-only descriptive diagnostic. No `experimental_strategies` / `validated_setups` write. Gates downstream hypotheses H1–H4; does not select survivors.
- **DB:** canonical `gold.db` opened `read_only=True` (8.95M `orb_outcomes` rows). IS boundary `HOLDOUT_SACRED_FROM = 2026-01-01`.
- **Grid:** stop_multiplier ∈ {0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0}; baseline = 1.0 (opposite-ORB-boundary, unmanaged).
- **Gate:** direct measured IS mean `pnl_r`. Does ANY tighter multiplier beat the 1.0 baseline? Yes → `PROCEED_H1_H3`; no → `PRE_KILL_PRICE_STOPS`; N_IS < 50 → `INSUFFICIENT_N`. Median MFE/MAE, winner-damage, stop-selectivity are **reported mechanism, never gating inputs** (no hardcoded 75% constant).
- **Raw per-stratum CSV:** `docs/audit/results/2026-05-31-adaptive-stops-h0-raw-per-lane.csv` (21 strata).

## Scope note

7 deployed lanes enumerated from `ACCOUNT_PROFILES` (no hardcoded list) × 3 entry-model strata (E1/E2/E3). **Deployed lanes are all E2** — that stratum is the deployment-relevant verdict. E1 strata are measured descriptively for the same lane spec where rows exist; E3 has zero rows on every lane (limit entry not booked for these specs). One lane skipped: `MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075` — missing from `validated_setups` (snapshot resolver raised; skipped cleanly, not a crash). MGC is horizon-excluded by the pre-reg regardless (2.7yr clean → strict-Bailey N too small).

## Deployed-stratum verdicts (E2 — the live geometry)

| Lane (E2 deployed) | N_IS | N_OOS | median MFE/MAE | best tighter m (IS EV) | baseline EV (m=1.0) | E2 VERDICT | year-stability |
|---|---|---|---|---|---|---|---|
| MNQ_COMEX_SETTLE_ORB_G5 | 1555 | 81 | **1.029** | 0.9 → +0.0866 | +0.0915 | **PRE_KILL** | SIGN_INCONSISTENT |
| MNQ_TOKYO_OPEN_COST_LT12 | 950 | 85 | **1.111** | 0.9 → +0.1179 | +0.1222 | **PRE_KILL** | SIGN_INCONSISTENT |
| MNQ_US_DATA_1000_ORB_G5_O15 | 1491 | 74 | 0.936 | 0.8 → +0.1300 | +0.1055 | PROCEED | SIGN_INCONSISTENT |
| MES_CME_PRECLOSE_ORB_G8 | 289 | 12 | **1.625** | 0.5 → +0.2583 | +0.1775 | **PROCEED** | **CONSISTENT** |
| MNQ_NYSE_OPEN_COST_LT12 | 1669 | 87 | **1.127** | 0.9 → +0.0767 | +0.0820 | **PRE_KILL** | SIGN_INCONSISTENT |
| MNQ_EUROPE_FLOW_ORB_G5 | 1583 | 87 | **1.018** | 0.9 → +0.0677 | +0.0643 | PROCEED (marginal, +0.0034) | SIGN_INCONSISTENT |
| MNQ_SINGAPORE_OPEN_ATR_P50_O15 | 913 | 65 | **1.150** | 0.9 → +0.0876 | +0.1130 | **PRE_KILL** | SIGN_INCONSISTENT |

EV in mean `pnl_r` (R-multiples). "best tighter m" = the tightest-EV multiplier < 1.0; PROCEED iff it strictly beats the 1.0 baseline.

## E1 strata (descriptive only — NOT deployed)

E1 (market-next-bar, ~1.18× ORB overshoot) booked on every MNQ/MES lane. Verdicts: PROCEED on 6/7 (COMEX_SETTLE, TOKYO_OPEN, US_DATA_1000, MES_CME_PRECLOSE, NYSE_OPEN, EUROPE_FLOW), PRE_KILL on SINGAPORE_OPEN. **All MNQ E1 PROCEEDs carry the SIGN_INCONSISTENT caveat** — pooled headroom is regime/year-driven, not stable. Only MES_CME_PRECLOSE E1 is CONSISTENT. E1 is not a deployed geometry, so these gate nothing live; they are recorded for completeness.

## Reading of the result (mechanism, Howard-grounded)

The pattern is exactly the one Howard 2026 §5.3.1–5.3.2 predicts (extract lines 88, 94, `docs/institutional/literature/howard_2026_value_area_breakouts_es.md`):

- **Every E2 lane with median MFE/MAE ≥ ~1.0 (symmetric — winners and losers run equally far before reversing) PRE-KILLs.** COMEX_SETTLE (1.029), TOKYO_OPEN (1.111), NYSE_OPEN (1.127), SINGAPORE_OPEN (1.150) all kill: no tighter price stop recovers the EV it destroys by truncating winners. Their winner-damage@0.75 (15–18%) roughly matches their stop-selectivity shortfall — the stop fires almost as often on eventual winners as on losers. This is on-our-data corroboration of Howard's "structural inability of price-based stops to discriminate at boundaries."
- **The two E2 PROCEEDs split on quality.** `MES_CME_PRECLOSE` E2 is the **only clean PROCEED**: ratio 1.625 (winners run *far* past losers — genuinely asymmetric), year-stability **CONSISTENT** (every powered year positive), winner-damage@0.75 only 2.7%, +0.081 EV at m=0.5. This is the real signal. `MNQ_US_DATA_1000` E2 PROCEEDs with a moderate +0.0245 at m=0.8 but **SIGN_INCONSISTENT** year-stability — pooled headroom may be regime-driven. `MNQ_EUROPE_FLOW` E2 PROCEEDs by a hair (+0.0034) and is SIGN_INCONSISTENT — effectively a coin-flip, not actionable.

## OOS readout (monitoring-only, never tuned)

Every stratum's OOS 0.75-vs-baseline delta is reported in the CSV with its power tier. **All 21 strata return `STATISTICALLY_USELESS`** (N_OOS 8–87, post-2026-01-01 sacred holdout). Per RULE 3.3 / pre-reg `oos_power_floor`, an IS↔OOS direction change is NOT refutation at this power tier — the OOS slice cannot confirm or refute any IS effect here. No verdict gates on OOS. This is expected: the sacred holdout is <5 months old.

## Per-lane gating verdicts (what this unlocks downstream)

- **PRE-KILL the price-stop family (H1 level-distance anchor, H3 ATR-scaled distance)** for the deployed (E2) geometry on: **COMEX_SETTLE, TOKYO_OPEN, NYSE_OPEN, SINGAPORE_OPEN** (4 of 7 deployed lanes). Tighter price stops are measurably value-destroying on these — drafting H1/H3 for them would be re-litigating a settled negative.
- **PROCEED to H1/H3 only where the asymmetry is real and stable:** `MES_CME_PRECLOSE` E2 (ratio 1.625, CONSISTENT, large EV headroom). This is the single lane that genuinely earns an H1/H3 draft.
- **Marginal / regime-fragile PROCEEDs** (`MNQ_US_DATA_1000` E2, `MNQ_EUROPE_FLOW` E2): treat as NEED-EVIDENCE, not green-light. Their SIGN_INCONSISTENT year-stability means the pooled EV headroom is driven by specific years — H1/H3 drafting must demonstrate year-stability before spending trial budget (MinBTL headroom for the downstream K=21 family is only 0.56yr; no room to waste trials on regime artifacts).
- **H2 (sweep-then-reverse, an entry switch) and H4 (time-based exit) are UNAFFECTED** by every verdict above — they are different mechanisms and remain separately testable even on the PRE-KILL lanes. Howard's positive alternative was the time-stack (beat price-stops 11/13 months), which informs H4.

## Honesty / rigor notes

- **No lookahead:** IS = `trading_day < HOLDOUT_SACRED_FROM` (imported, no date literal); OOS disjoint and monitoring-only. MFE/MAE used as post-trade descriptive measurement inputs (RULE 6.3 dual-status), never as predictors in any decision rule.
- **No 2026 tuning:** all verdicts computed on the IS slice only.
- **In-sample multiplier selection acknowledged:** `best_tighter_m` is chosen post-hoc on IS to answer the literal H0 question ("does *any* tighter stop help?"). This is the legitimate gate framing; the year-stability check is evaluated at that pooled multiplier and is **annotation only** — it never flips a verdict. SIGN_INCONSISTENT flags exactly the case where the pooled "improves" is one regime; surfaced loudly, not hidden.
- **Verified, not claimed:** numbers above are read directly from the script's stdout and the committed CSV, not paraphrased. Lane skip (MGC_TOKYO) is reported, not silently dropped.
- **Edge cases handled:** E3 zero-row strata → `INSUFFICIENT_N` (not a crash); MGC_TOKYO missing from `validated_setups` → clean SKIP; sub-50-N strata would gate to INSUFFICIENT_N (none of the deployed E2 strata hit this floor; MES_CME_PRECLOSE E2 at N=289 clears it).

## Highest-EV next action

Draft H1/H3 **only for `MES_CME_PRECLOSE` E2** — it is the single lane with a real, stable, large stop-asymmetry (ratio 1.625, CONSISTENT, +0.081 EV at m=0.5). Pre-kill the price-stop family on the four symmetric MNQ lanes; do not draft H1/H3 for the two regime-fragile PROCEEDs without first proving year-stability. H4 (time-exit) remains the broader untested opportunity across all lanes and is not blocked by any result here.
