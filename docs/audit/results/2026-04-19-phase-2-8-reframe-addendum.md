# Phase 2.8 — reframe addendum (second-brain / institutional skeptical audit)

**Date:** 2026-04-19
**Supersedes framing of:** `2026-04-19-multi-year-regime-stratification.md`
**Does NOT supersede:** the retirement verdicts for the 2 SGP PURE_DRAG lanes (those are Phase 2.4/2.7/2.8-triple-confirmed).
**Source data:** `research/output/phase_2_8_multi_year_regime_stratification.csv` (38 lanes × 2020/2022/2024 columns, generated 22:47 UTC).

---

## 1. Restate the original claim

> "2024 regime break = recurring high-vol-year regime effect." — **Refuted.** 34/38 lanes VOL_NEUTRAL, only 3 SINGLE_YEAR_DRAG cells (2 SGP 2024, 1 X_MES_ATR60 2020), 0 RECURRING.

This is the headline written into the result doc and the session handoff.

## 2. Verdict on the audit itself — **PARTIAL**

The headline is not wrong — it answers the exact question that was asked. It is, however, a **narrow, self-confirming** test of one framing, presented as broadly refutatory of a wider hypothesis it did not actually test. Five issues qualify the claim.

### 2.1 The run didn't match the handoff's promise

- Handoff said the background job was `--comprehensive` (7 years: 2019-2025).
- CSV on disk has only 3 year columns (2020/2022/2024) — the default `STRATIFY_YEARS` run, not `COMPREHENSIVE_YEARS`.
- Either the comprehensive job failed, was overwritten, or the flag wasn't used.
- The 7 Tier-4 retirement candidates that were part of the Phase 2.5 book (14 lanes) — **none** are in the Phase 2.8 CSV. The 38 rows are the 38 ACTIVE validated_setups, not a 52-row union.
- **"No recurring regime" was established only on 3 high-vol years out of a possible 7, and only on lanes that are still ACTIVE.** Low-vol control years (2019, 2021, 2023) and retired lanes were never in the sample.

### 2.2 The PATTERN classification is asymmetric and uses a bare threshold, not a subset t-test

- `SINGLE_YEAR_DRAG` fires when `|delta_year| > 0.03` AND `y_expr < some_threshold` AND `n_year >= 30`.
- No bootstrap, no t-test, no BH on 38 lanes × 3 years = **114 implicit year-level tests**.
- At K=114, BH q=0.10 requires raw p ≤ ~0.001 per survivor. The 2 SGP 2024 cells have N=89 subset, t ~ 2.8-3.5 on the single year — not survivor-grade under honest per-year multiple testing.
- The classification *only flags negative* deltas (DRAG). Tailwinds of equal magnitude get labelled VOL_NEUTRAL. This is narrative-asymmetry — the kind the institutional-rigor rule 8 "Never trust metadata" specifically calls out.

### 2.3 There is a strong signal the headline classification buries

Cross-section of per-session mean `delta2024` (negative = 2024 was a tailwind for that session):

| Session | n lanes | min | mean | max |
|---|--:|--:|--:|--:|
| CME_PRECLOSE | 3 | +0.029 | **+0.032** | +0.038 |
| EUROPE_FLOW | 10 | -0.002 | **+0.017** | +0.049 |
| COMEX_SETTLE | 8 | -0.024 | -0.003 | +0.014 |
| US_DATA_1000 | 5 | -0.055 | -0.018 | +0.001 |
| NYSE_OPEN | 6 | -0.020 | **-0.010** | -0.002 |
| TOKYO_OPEN | 4 | -0.023 | **-0.015** | -0.010 |
| SINGAPORE_OPEN | 2 | -0.044 | **-0.029** | -0.014 |

**2024 was directionally asymmetric by session, not noise:** late/macro-flow sessions (CME_PRECLOSE, EUROPE_FLOW) took drag; Asia-open and US-morning sessions got a tailwind. This is a session-structural finding that "34/38 VOL_NEUTRAL" smothers. Same phenomenon, opposite sign on opposite session groups — consistent with a 2024 intraday-flow regime (Asia momentum works, Europe range-bound, US-morning works, CME close range-bound).

### 2.4 The SGP-momentum 2024 failure has a mechanism-signature the existing doc misses

RR dose-response on MNQ EUROPE_FLOW CROSS_SGP_MOMENTUM:

| RR | full_expr | y2024_expr | delta2024 |
|---|--:|--:|--:|
| 1.0 | +0.050 | -0.048 | +0.020 |
| 1.5 | +0.081 | -0.125 | +0.041 |
| 2.0 | +0.112 | -0.132 | +0.049 |

**Damage scales with RR, not with N (N is 89 at all 3 RRs).** Losers are stopping at 1R while winners no longer reach 2R — the classic signature of intraday range compression / truncated-tail regime. The existing doc's "we can't characterize this from our data" is premature — the mechanism signature IS in the data: it's range compression, not momentum-decay and not volatility.

Operational implication: the correct rescue is NOT regime-conditional gating (which the existing doc already ruled out). It is **exit-shape**: test a tighter profit-lock or trailing-stop variant on the same (session, filter) cell in 2024+ data. That is a new pre-reg, not a rescue of the retired lane.

### 2.5 Hidden per-year weaknesses on Phase 2.5 Tier-1 / "GOLD" lanes

Phase 2.8 only surfaces the top `delta` cell per lane. Sampling reveals more:

- **MNQ COMEX_SETTLE X_MES_ATR60 RR1.5** (Phase 2.5 Tier-1): y2020 = -0.019, ex2020 = +0.255, delta2020 = **+0.057**. The PATTERN logic flagged this UNEVALUABLE / VOL_NEUTRAL because 2020 sample is modest. A genuine 0.057-R gap between IS-with-2020 and IS-without-2020 on a flagship lane should at least be logged.
- **MNQ US_DATA_1000 VWAP_MID_ALIGNED RR2.0 O15** (Phase 2.5 ARITHMETIC_LIFT): y2022 = -0.043 (full = +0.148). 2022 rate-hike regime is negative on this lane — a session × macro interaction the pattern label hides.

Neither is a retire-now finding. Both belong in a watch-list.

## 3. Institutional checklist

| Check | Status | Note |
|---|---|---|
| Look-ahead | PASS | All years pre-2026-01-01 Mode A. Canonical `compute_mode_a` used. No leakage risk. |
| Execution realism | PASS | No new execution logic — inherits Phase 2.4/2.5 cost model and canonical filter delegation. |
| Sample correctness | PASS | 38 active-validated rows enumerated; SGP N=89 per year noted. |
| Eligibility scope | PARTIAL | Test doesn't include the 14 Phase 2.5 Tier-4 retirement candidates, or the 7 years 2019/2021/2023/2025. |
| Cost / risk math | PASS | No lane-P&L recompute; relies on orb_outcomes pre-cost. |
| Multiple testing | **FAIL** | 114 implicit year-level tests with no BH. 0.03 threshold is a bare cutoff. |
| Narrative symmetry | **FAIL** | PATTERN flags DRAG but not equal-magnitude TAILWIND. |
| Session-structural framing | **FAIL** | Uniform VOL_NEUTRAL label hides a clean session-level directional asymmetry. |
| Mechanism read | **PARTIAL** | SGP failure dismissed as "can't characterize" when RR dose-response is a legible range-compression signature. |

## 4. Alternative interpretations (honest, not post-hoc rescue)

### I1. 2024 is a session-level regime asymmetry, not a vol-regime event

Supported by §2.3 cross-section. Mechanism plausible: in 2024, Asian/US-morning sessions trended, Europe-flow and US-close ranges compressed. This is consistent with the Chan p120 "vol regime ≠ directional regime" result but goes further — it says the directional regime was SESSION-selective in 2024, not universal.

**Honest test:** compute per-session correlation of `y2024_lane_expr` with `session_time_of_day_block` across more lanes (including retired ones) to verify the directional split isn't a sampling artifact of the 38-row active set.

### I2. The SGP 2024 break is an exit-shape problem, not a filter-decay problem

Supported by §2.4 RR dose-response. The existing doc treats SGP as "retired, something unknowable in 2024." This is incomplete — the data signature IS interpretable.

**Honest test:** pre-reg a trailing-stop / ATR-lock variant on MNQ EUROPE_FLOW SGP 2024-2025 data alone, with pre-committed kill criteria. Sample thin (N≈89 per year) so expect low power; but it's the narrow honest next question.

### I3. CME_PRECLOSE is a 2020-specific tailwind lane

Per-session mean delta2020 = **-0.047** on CME_PRECLOSE (the strongest negative delta in the whole sweep). 2020-COVID was a BIG tailwind for this session/filter cluster. If this is mean-reverting (i.e., the lane's validated `expectancy_r` is inflated by the 2020 contribution), the Phase 2.5 Tier-1 MES CME_PRECLOSE cells (COST_LT08, ORB_G8) may be 2020-dependent.

**Honest test:** recompute each CME_PRECLOSE Tier-1 lane's ExpR with 2020 excluded. If `ex2020_expr` subset-t drops below 1.96 on any Tier-1 CME_PRECLOSE lane, downgrade Tier.

### I4. VWAP_MID_ALIGNED on US-data session has a rate-hike-regime weakness

Per §2.5 — 2022 turned two of three VWAP_MID_ALIGNED variants on US_DATA_1000 negative on the year. This is consistent with VWAP-based entries failing in a strong-trending macro regime (rate-hike = one-sided vol with less mean-reversion around VWAP).

**Honest test:** compute per-year VWAP-fire-rate and VWAP-adherence metric during 2022. If 2022 had distinctly different VWAP dispersion, the weakness is regime-driven, not noise — and the lane should be qualified as non-robust in strong-trending macro years.

### I5. "No recurring regime" statement is only valid on the tested 3 high-vol years

The Phase 2.8 v1 verdict should be narrowed to: "No recurrence on 2020 + 2022 + 2024, tested on 38 currently-active lanes." The broader claim "vol regime is not the explanation" is still supported, but "no recurring regime of any kind" is NOT supported — 2019, 2021, 2023, 2025 were never in the sample.

## 5. Missed opportunities

1. **Session × year heat-map.** Simple 7×7 (session × year) mean-ExpR table would reveal the asymmetry in §2.3 in 30 seconds. Should be a standard Phase 2.8 artifact.
2. **Per-year t-test + BH on the 114 cells.** Would surface honest year-level outliers without arbitrary 0.03 cutoffs.
3. **RR-dose-response scan as a diagnostic column.** For any lane with 2+ RR variants, report `∂delta_year / ∂RR`. Range-compression detector.
4. **"Tier-1 regime fragility" cross-check.** Every Phase 2.5 Tier-1 / Phase 2.7 GOLD candidate should have an ex-each-year subset-t alongside its full subset-t. Would have caught the COMEX_SETTLE X_MES_ATR60 2020 drag pre-GOLD-labelling.

## 6. What to actually change

Non-retraction (the v1 doc's retirement verdicts stand):

- Keep the 2 SGP RR1.5/RR2.0 PURE_DRAG retirements (already Phase 2.4/2.7 confirmed, Phase 2.8 agnostic on framing).
- Keep the 2 ARITHMETIC_LIFT retirements from Phase 2.5 (unrelated to this reframe).
- Keep the GOLD-pool framing but add a FRAGILITY CHECK column (§ Missed opportunity 4).

Doctrine-level additions:

- Phase 2.8-style year stratification must include (a) a cross-session heat map, (b) per-cell subset-t, (c) symmetric DRAG/BOOST labels, (d) BH correction at K = n_lanes × n_years.
- When a pattern-labeller uses a bare absolute threshold for classification, it must be accompanied by a t-test on the subset that drives the label.
- "Comprehensive" 7-year sweeps must be verified by output file columns, not by handoff text. Always check the CSV header before claiming comprehensive.

## 7. Honest next test

Single most-productive follow-up: **re-run Phase 2.8 with `--comprehensive` (2019-2025) AND with the 14 Phase 2.5 Tier-4 retirement candidates included** (total K = 52 lanes × 7 years = 364 cells). Apply per-cell subset-t + BH q=0.10 at K=364. Publish the cross-session × year heat map.

This answers both "is there a recurring regime?" (the stated question) and "does 2024 show session-directional asymmetry?" (the question Phase 2.8 v1 didn't ask).

If user accepts, this becomes Phase 2.8-v2 or Phase 2.9 — research-only, no production code.

## 8. Audit trail

- Raw data: `research/output/phase_2_8_multi_year_regime_stratification.csv`
- Cross-section script used for this reframe: inline Python, see commit body.
- Canonical references: `.claude/rules/quant-audit-protocol.md` § STEP 2, `.claude/rules/backtesting-methodology.md` RULE 4 (K framing) + RULE 12 (red flags).
- Institutional-rigor rules invoked: 1 (self-review before claim-of-done), 6 (no silent failures — unverified comprehensive run), 8 (verify before claiming).

## 9. Verdict (per skeptical-audit prompt)

**Partial.** The v1 doc's narrow claim — "ex-2020 / ex-2022 / ex-2024 on active lanes shows no RECURRING drag pattern" — is true as stated. The v1 doc's broader implications — "2024 is just year-specific idiosyncrasy, 34/38 are regime-robust, the recurring-vol-regime hypothesis is refuted" — are **not adequately supported** by the run as executed. The execution was missing 4 of 7 years, missing 14 of 52 relevant lanes, missing per-cell significance testing, and missing the symmetric directional framing. Retirement verdicts stand; GOLD-pool confidence should be tempered pending a proper 7-year × 52-lane rerun.
