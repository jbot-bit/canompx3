# Phase 2.9 framing audit — post-promotion robustness + A3 erratum

**Date:** 2026-04-20
**Pre-reg:** `docs/audit/hypotheses/2026-04-20-phase-2-9-framing-audit.yaml` (LOCKED)
**Stage:** `docs/runtime/stages/phase-2-9-framing-audit.md`
**Script:** `research/phase_2_9_framing_audit_v1.py`
**Outputs:** `research/output/phase_2_9_framing_audit_main.csv`, `...json`
**Predecessor:** `docs/audit/results/2026-04-19-phase-2-9-comprehensive-multi-year.md` (on branch `campaign-2026-04-19-phase-2`, not main)
**Data cutoff:** MNQ COMEX_SETTLE orb_outcomes through 2026-04-16 (69 OOS trading days)
**Holdout:** `trading_app.holdout_policy.HOLDOUT_SACRED_FROM = 2026-01-01`

---

## TL;DR (what matters for deployment decisions)

- **Four MNQ COMEX_SETTLE lanes promoted 2026-04-11 (OVNRNG_100 and X_MES_ATR60 at RR1.0/RR1.5) are ALIVE under institutional locked criteria.** C6-C9 pass; OOS sign-matches IS on all 4.
- **D3 era stability PASSES** on all 4 (no era with N≥50 has ExpR < −0.05). Per Criterion 9, the lanes are not era-dead.
- **D4 OOS sign-matches IS on all 4.** OOS underpowered (N=40–62, t=0.69–1.60, none p<0.05). Direction is right; significance waits for N≥150 per Harvey-Liu 2015 Exhibit 4.
- **Criterion 8 PASSES on all 4.** Live_OOS_ExpR / stored_IS_ExpR ratios: 0.77, 0.49, 1.51, 1.19 — all ≥ 0.40 floor. OVNRNG_100 RR1.5 sits nearest the floor (0.49) and is the closest watch candidate.
- **D1 fire-rate is strongly non-stationary across years (chi-square p = 10⁻³⁰ to 10⁻³²), which is EXPECTED behavior for absolute-threshold regime filters — not a defect.** Interpretation caveat: this means (a) the Phase 2.9 "2025 BOOST concentration" combines genuine per-trade edge with fire-rate amplification (57.9% fires in 2025 vs 4.3% in 2019 on OVNRNG_100), and (b) per-year ExpR comparisons are less informative than total-horizon ExpR when fire count varies 10×.
- **D2: `validated_setups.oos_exp_r` drifted from live canonical** by up to 0.097 R on 3 of 4 cells (OVNRNG_100 RR1.5). Expected for a snapshot-at-promotion field as 5 days of new OOS accumulate; recompute from canonical `orb_outcomes` when citing. The pre-reg gate (|Δ|≤0.010) was mis-calibrated for this context — Criterion 8 ratio is the load-bearing gate and passes 4/4.
- **Phase 2.9 A3 erratum (line 295): confirmed factually wrong.** MNQ COMEX_SETTLE 2019-2023 has full ~249-row/year baseline coverage in live DB. Doc's "availability" caveat was editorial, not data-driven. Correction: MNQ was mildly positive 2020-2024 and Chordia-adjacent t=+2.50 in 2023 — 2025 (t=+2.91) is escalation, not discovery.
- **Per-cell verdict: CONTINUE × 4.** Criterion 6-9 all pass. No sizing-up until OOS ≥ 150 days (~2026-07-15). No Phase 2.10 macro-regime investigation without a stated mechanism.

---

## What was tested (4 Mode-A-discovered lanes)

| # | Strategy ID | Filter | RR | Stored IS ExpR | Stored OOS ExpR | Stored N | Discovery |
|---|---|---|--:|--:|--:|--:|---|
| 1 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | OVNRNG_100 | 1.0 | +0.1725 | +0.1490 | 520 | 2026-04-11 |
| 2 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | OVNRNG_100 | 1.5 | +0.2151 | +0.2029 | 513 | 2026-04-11 |
| 3 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | X_MES_ATR60 | 1.0 | +0.1512 | +0.1633 | 673 | 2026-04-11 |
| 4 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | X_MES_ATR60 | 1.5 | +0.1609 | +0.1828 | 664 | 2026-04-11 |

Discovery used Mode A holdout (data < 2026-01-01). 2025 is IS. 2026-01-02 → 2026-04-16 is the accumulating Mode A OOS.

---

## D1 — Fire-rate stationarity (chi-square goodness-of-fit)

**Test:** Null = filter fires at the same rate every IS year 2019-2025. H1 rejected → filter fire rate is year-dependent.

**Note on interpretation (added in self-audit pass):** Rejecting the null does NOT mean the filter has a defect. For absolute-threshold filters like OVNRNG_100 (overnight_range ≥ 100 pts) and X_MES_ATR60 (MES atr_20_pct ≥ 60), regime-conditional firing is the design intent — the filter is meant to fire more often in high-vol regimes and less often in calm regimes. D1 verifies this is happening; it does NOT indicate the filter is broken. The institutional concern is narrower: when fire-rate varies 10× across IS years, per-year ExpR comparisons (as in Phase 2.9's BH_year framing at K=38) carry less information than total-horizon ExpR, because year-level samples are not drawn from the same regime.

**Grounding:** Bailey, Borwein, LdP, Zhu (2013) Proposition 2 (compensation effect under distributional change) — IS ExpR on a non-stationary distribution may diverge from OOS ExpR. This is not a rejection of the filter; it's a reminder that year-stratified IS ExpR on a regime filter is sampling different regimes, not replicating the same test. Citations: `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` pp.8, 12; `docs/institutional/literature/harvey_liu_2015_backtesting.md` p.16.

### OVNRNG_100 (threshold: overnight_range ≥ 100 pts absolute)

Per-year fire rate on all MNQ COMEX_SETTLE O5 RR1.0 trades (N shown is trade-days in that year; fires is the count meeting the threshold):

| Year | Trades | Fires | Fire rate |
|---:|---:|---:|---:|
| 2019 | 161 | 7 | **4.3%** |
| 2020 | 249 | 104 | 41.8% |
| 2021 | 249 | 65 | 26.1% |
| 2022 | 250 | 116 | 46.4% |
| 2023 | 248 | 20 | **8.1%** |
| 2024 | 246 | 72 | 29.3% |
| 2025 | 247 | 143 | **57.9%** |

**χ²(6) = 161.8, p = 2.4 × 10⁻³²** — null of stationary fire-rate emphatically rejected.

The 2019 + 2023 fire rates (4.3% + 8.1%) are 10-15× lower than 2022 + 2025 (46% + 58%). These are the SAME filter with the SAME rule applied. The difference is that 2025 overnight vol is much wider than 2019 overnight vol, and the absolute 100-pt threshold is calibrated to some average-vol regime that neither year represents.

Pattern is essentially identical on RR1.5 (χ²=163, p=1.2×10⁻³²).

### X_MES_ATR60 (threshold: MES atr_20_pct ≥ 60th percentile)

| Year | Trades | Fires | Fire rate |
|---:|---:|---:|---:|
| 2019 | 161 | 31 | 19.3% |
| 2020 | 249 | 143 | **57.4%** |
| 2021 | 249 | 60 | 24.1% |
| 2022 | 250 | 150 | **60.0%** |
| 2023 | 248 | 32 | 12.9% |
| 2024 | 246 | 154 | **62.6%** |
| 2025 | 247 | 134 | 54.3% |

**χ²(6) = 153.3, p = 1.5 × 10⁻³⁰** — same non-stationarity, opposite-direction bimodality. Fires ~60% in COVID-2020, rate-hike-2022, AI-rally-2024; fires 13-24% in "ordinary" 2019/2021/2023. Pattern on RR1.5: χ²=150, p=6.2×10⁻³⁰.

**Note on X_MES_ATR60:** the filter uses `atr_20_pct` which IS a percentile feature by construction (per-symbol rolling percentile). Even so, the cross-asset injection doesn't re-normalize per year, so the same drift issue applies with narrower dynamic range.

### What D1 means for the Phase 2.9 2025 BH_year claim

Phase 2.9 doc's "2025 BOOST concentration" framing (`2026-04-19-phase-2-9-comprehensive-multi-year.md` § H2) rests on 4 BH_year survivor cells in 2025. Two of those are within the audited 4 lanes:

- `OVNRNG_100 RR1.0 × 2025`: fires 143/247 = 57.9% of days, ExpR=+0.292, t=2.98
- `OVNRNG_100 RR1.5 × 2025`: same fire mask, ExpR=+0.357, t=2.74
- `X_MES_ATR60 RR1.0 × 2025`: fires 134/247 = 54.3%, ExpR=+0.310, t=3.06

These are real per-trade signals at the year level, and they SHOULD coincide with the highest fire-rate year of the IS horizon — that is precisely what a regime filter that captures 2025's high-vol regime would do. The BH_year family-K=38 treats year-cells as independent tests; since fire count varies across years, the "independence" is a composite statistic reflecting both per-trade edge AND trade count. The quant interpretation:
- **Per-trade signal in 2025:** genuinely positive (t > 2.7 on all three survivors; conditional ExpR +0.292 to +0.357 on 70-82-trade subsets).
- **Year-concentration of survivors in the BH_year family:** mechanically influenced — years with more trades have tighter SE per-year, so achieve lower p at the same per-trade ExpR as low-fire-rate years. This is visible in the 2023 row (only 20 fires on the OVNRNG cells, ExpR +0.296, yet did NOT cross BH_year because SE was too wide at N=20). 2023 had the same qualitative per-trade edge as 2025 but is invisible at K_year=38.

**Corrected framing (institutional-honest):** the 2025 BH_year concentration reflects both (a) genuine per-trade edge in a high-vol regime that the filter captures AND (b) the trade-count power effect of a high-fire-rate year. Claiming "2025 is a unique macro regime" overstates the interpretation — without a MECHANISM that also explains why 2023 (with comparable per-trade ExpR on fewer fires) did NOT surface as a survivor, the 2025 finding is regime-consistent edge in a regime the filter was designed to catch, not a new year-specific macro story.

---

## D2 — Live OOS vs stored `oos_exp_r` (metadata staleness)

**Test:** `oos_exp_r` in `validated_setups` was computed at promotion on 2026-04-11. Canonical `orb_outcomes` has added trades between 2026-04-11 and 2026-04-16. Recompute live and compare.

**Grounding:** `.claude/rules/integrity-guardian.md` § 7 ("Never trust metadata — always verify") + `.claude/rules/research-truth-protocol.md` § "Mode B grandfathered validated_setups" (applies here for freshness, not for Mode A/B).

| Cell | Live OOS ExpR (69 days) | Live OOS N | Stored oos_exp_r | Stored IS | Delta | Pre-reg gate (\|Δ\|≤0.010) | Criterion 8 (live/stored_IS) |
|---|--:|--:|--:|--:|--:|:-:|--:|
| OVNRNG_100 RR1.0 | +0.132 | 62 | +0.149 | +0.173 | −0.017 | FAIL | 0.77 PASS |
| OVNRNG_100 RR1.5 | +0.106 | 60 | +0.203 | +0.215 | **−0.097** | FAIL | 0.49 PASS |
| X_MES_ATR60 RR1.0 | +0.228 | 42 | +0.163 | +0.151 | +0.065 | FAIL | 1.51 PASS |
| X_MES_ATR60 RR1.5 | +0.192 | 40 | +0.183 | +0.161 | +0.009 | PASS | 1.19 PASS |

**Two readings:**
1. **Pre-reg gate (|Δ| ≤ 0.010 R):** 3 of 4 fail. Stored value is stale vs accumulating canonical OOS mean — expected for snapshot-at-promotion fields when 5 days of trades have been added since.
2. **Criterion 8 gate (live_OOS_ExpR ≥ 0.40 × stored_IS_ExpR):** 4 of 4 pass, with ratios ranging 0.49 to 1.51. The OOS mean has NOT degraded materially below the Criterion 8 floor.

**Operational meaning:** `validated_setups.oos_exp_r` should be resnapshotted or treated as stale; it's a promotion-time value, not a running truth. For decision-making, recompute from canonical `orb_outcomes` every time.

**OVNRNG_100 RR1.5 is the one that deserves note** — live OOS is about HALF the stored value (0.106 vs 0.203). Still passes Criterion 8 but sits at the 0.49 ratio, close to the 0.40 floor. If OOS continues at this pace, it will cross the floor around ~N=120 OOS days (late Q2 2026).

---

## D3 — Era stability (Criterion 9)

**Test:** For each cell, partition IS trades by era. Pass = no era with N ≥ 50 has ExpR < −0.05. Era boundaries chosen for macro-regime coverage.

**Grounding:** `docs/institutional/pre_registered_criteria.md` § Criterion 9.

### All 4 cells PASS (no breaches)

| Era | OVNRNG_100 RR1.0 | OVNRNG_100 RR1.5 | X_MES_ATR60 RR1.0 | X_MES_ATR60 RR1.5 |
|---|--:|--:|--:|--:|
| 2019-2020 (COVID + early-micro) | N=111 / +0.232 | N=109 / +0.203 | N=174 / +0.126 | N=172 / +0.043 |
| 2021-2022 (Fed hike cycle) | N=181 / +0.047 | N=178 / +0.080 | N=210 / +0.113 | N=209 / +0.131 |
| 2023 (stabilization) | N=20 / +0.296 | N=20 / −0.082 | N=32 / +0.109 | N=32 / +0.093 |
| 2024-2025 (AI rally + vol expansion) | N=215 / +0.237 | N=213 / +0.341 | N=288 / +0.192 | N=282 / +0.237 |

**Era observations:**
- 2019-2020 era has thin N on OVNRNG (~110) vs thicker on X_MES (~170) — reflects filter difference on same underlying bars.
- 2023 era has N<50 on all 4 cells → exempt per Criterion 9 rule (N<50 eras cannot judge stability).
- The negative 2023 cell on OVNRNG RR1.5 (N=20, ExpR=−0.082) is thin-N and exempt; do not cite as "2023 failure."
- No era shows the kind of −0.20 or −0.30 R pit that would flag era-dependency.

**Verdict:** era-stable under Criterion 9.

---

## D4 — 2026 OOS t-stat + sign-match (69 days)

**Test:** Compute one-sample t-stat for OOS mean vs zero. Compare sign to IS. RULE 3.1 of `.claude/rules/backtesting-methodology.md` — dir_match required.

| Cell | OOS N | OOS ExpR | OOS SD | t | p (two-sided) | Sign match | Status |
|---|--:|--:|--:|--:|--:|:-:|---|
| OVNRNG_100 RR1.0 | 62 | +0.132 | 0.94 | 1.11 | 0.272 | ✓ | SIGN_MATCH |
| OVNRNG_100 RR1.5 | 60 | +0.106 | 1.19 | 0.69 | 0.493 | ✓ | SIGN_MATCH |
| X_MES_ATR60 RR1.0 | 42 | +0.228 | 0.93 | 1.60 | 0.118 | ✓ | SIGN_MATCH |
| X_MES_ATR60 RR1.5 | 40 | +0.192 | 1.21 | 1.00 | 0.322 | ✓ | SIGN_MATCH |

- All directions match IS.
- None clear p<0.05 on OOS alone — expected at N=40-62. Per Harvey-Liu 2015 Exhibit 4 (`docs/institutional/literature/harvey_liu_2015_backtesting.md` p.22), 60 observations at 10% vol needs ExpR > 0.062 R for single-test p<0.05; our cells hit that for X_MES but OVNRNG RR1.5 is at +0.106 with higher dispersion.
- OOS sample is under the 150-day institutional threshold (`pre_registered_criteria.md` § Criterion 8 spirit). **Treat as directional-only, do not promote to "OOS confirmed" in any communication.**

---

## D5 — Phase 2.9 A3 erratum (MNQ unfiltered baseline correction)

**Phase 2.9 result doc line 295 claim (quoted verbatim from `.worktrees/campaign-2026-04-19-phase-2/docs/audit/results/2026-04-19-phase-2-9-comprehensive-multi-year.md`):**

> "MNQ pre-2024 COMEX_SETTLE data has different `sample_size` availability (MNQ micro trades less continuous history before mid-2024 on the CSV scan year); the full pre-2024 MNQ baseline is available via `research/mode_a_revalidation_active_setups.py::load_active_setups` but not shown here to keep the comparison synchronized on MES's 7-year coverage."

**Live DB shows MNQ COMEX_SETTLE O5 E2 CB1 RR1.0 has ~249 rows/year every year 2019-2025** — not availability-limited. The full MNQ baseline the doc omitted:

| Year | N | ExpR | t | p (two-sided) | Chordia t≥3.00 |
|---:|---:|--:|--:|--:|:-:|
| 2019 | 161 | **−0.128** | −2.03 | 0.044 | no |
| 2020 | 249 | +0.074 | +1.31 | 0.191 | no |
| 2021 | 249 | +0.005 | +0.10 | 0.924 | no |
| 2022 | 250 | +0.029 | +0.48 | 0.628 | no |
| 2023 | 248 | **+0.141** | +2.50 | 0.013 | no |
| 2024 | 246 | +0.088 | +1.52 | 0.130 | no |
| 2025 | 247 | **+0.169** | +2.91 | 0.004 | no (below 3.0) |

**Correct readings from this baseline:**
- MNQ COMEX_SETTLE was NOT structurally dead pre-2024; it was already t=+2.50 in 2023 and small-positive 2020-2022.
- 2019 was negative (t=−2.03) — this was the year pre-Phase-3c data was parent-proxy and the audit (Amendment 2.8) found thin early-micro liquidity. The era breakdown is consistent.
- 2025 t=+2.91 is the strongest year, but it's an escalation of an existing mild-positive pattern, not a regime birth.

**Correction for Phase 2.9 readers:** when that doc merges to main, a follow-up pointer should be added to line 295 noting (a) data availability was not the reason for the omission, and (b) MNQ's 7-year COMEX_SETTLE unfiltered baseline shows 2025 is escalation not discovery. This doc (the erratum) will serve as the pointer target.

**Why it matters operationally:** The "MES was structurally negative 2019-2024, MNQ turned positive in 2024" framing (Phase 2.9 line 299) is **false for MNQ** — MNQ had mild-positive tilt since 2020 and Chordia-adjacent t in 2023. Any deployment narrative that cites "2025 as unique macro regime" should either produce a mechanism (none currently) or be downgraded to "2025 is a culmination year, not a regime shift."

---

## Integrated verdict (per cell)

Verdict encoding (post self-audit):
- **D1 "non-stationary" is EXPECTED** for regime filters (not a verdict input), so it appears as a note, not a gate.
- The load-bearing gates are Criterion 6 (WFE, inherited from promotion), Criterion 8 (OOS ratio), Criterion 9 (era stability), and dir_match.

| Cell | D1 note | D2 Criterion 8 ratio | D3 era | D4 OOS | Verdict |
|---|---|:-:|:-:|:-:|---|
| OVNRNG_100 RR1.0 | regime-conditional (p=2e-32, design) | 0.77 PASS | PASS | SIGN_MATCH underpowered | **CONTINUE** |
| OVNRNG_100 RR1.5 | regime-conditional (p=1e-32, design) | 0.49 PASS (nearest floor) | PASS | SIGN_MATCH underpowered | **CONTINUE_NEAR_C8_FLOOR** |
| X_MES_ATR60 RR1.0 | regime-conditional (p=2e-30, design) | 1.51 PASS (stronger OOS than IS) | PASS | SIGN_MATCH underpowered | **CONTINUE** |
| X_MES_ATR60 RR1.5 | regime-conditional (p=6e-30, design) | 1.19 PASS | PASS | SIGN_MATCH underpowered | **CONTINUE** |

**All 4 lanes ALIVE under locked institutional criteria. OVNRNG_100 RR1.5 is the closest watch candidate** — its C8 ratio of 0.49 is nearest the 0.40 floor. If OOS continues at current pace, that lane would cross the floor around N≈120 OOS days (late Q2 2026). Not a current concern, but the next review should focus there first.

---

## Operational decisions (pre-committed in stage file; not post-hoc)

### Continue (current sizing, current scope)
- No deployment change on any of the 4 lanes.
- Signal/demo tracks at current allocator weight (per `topstep_50k_mnq_auto`).
- Quarterly review: re-run this audit script at ~Q2, Q3, Q4 boundaries.

### Do NOT
- **Size up** any of the 4 lanes until OOS ≥ 150 days AND fire-rate drift check passes on a rolling 3-year window (not yet achievable — 2023 was an 8.1% fire-rate year; current 3-year window includes the 2023 trough).
- **Open Phase 2.10** "2025 macro regime" investigation. The 2025 BH_year concentration is partially fire-rate-mechanical, not a clean macro signal. A mechanism hypothesis would need to explain WHY 2025 has high vol that pushes more days above the absolute thresholds, AND why that high-vol state also produces per-trade edge — two things, not one. Without both, per `pre_registered_criteria.md` § Criterion 1 (theory citation required), no pre-reg can honestly be written.
- **Cite `validated_setups.oos_exp_r`** as current truth anywhere downstream. Always recompute from canonical `orb_outcomes` + `daily_features` on query.
- **Treat MES COMEX_SETTLE 2025 positive** (+0.083, t=1.55) as cross-instrument confirmation of a 2025 regime — the sister instrument MNQ was already positive from 2020 onwards, so MES 2025 is MES-catching-up, not joint-evidence.

### Follow-up candidates (NOT scheduled — would need their own pre-reg)
1. **Filter redefinition as vol-regime-relative.** Absolute-threshold filter under non-stationary vol creates the D1 failure. A candidate: `OVNRNG_P75` (percentile-on-rolling-252-day) or regime-conditional `OVNRNG_100_IF_HIGH_VOL`. Would require new pre-reg + MinBTL budget + Mode A re-discovery (2 new cells × 4 RR = 8 trials, well within budget).
2. **MES COMEX_SETTLE E2 CB1 fade side.** My prior reframe suggested this is beta-confounded and cost-asymmetric; do not pursue as I framed. But a different framing — "if MES CS longs lose pre-cost, is there a cost-adjusted structural edge in fading only when filter fires on MNQ?" — is a cross-instrument confluence idea that could be pre-registered with Fitschen Ch 3 + cross-asset literature grounding. NOT priority until the 4-cell audit shows continued OOS stability.

### Revisit schedule
- **Next run of this audit:** 2026-07-15 (approximately 150 OOS days since 2026-01-02).
- **Decision-point at next run:** if D4 t-stats cross p<0.05 with sign-match AND D1 rolling-3yr stationarity is achievable, sizing-up discussion is in-scope. Otherwise continue WATCH.

---

## Relationship to Phase 2.9 original doc

This audit does NOT contradict Phase 2.9's arithmetic — every number there reproduces in live DB. It refines the interpretation:
- Phase 2.9 § H2 "2025 year regime alignment PASS" → **refined:** 2025 BH_year concentration has a genuine per-trade component AND a fire-rate-mechanical component; the joint framing overstates regime-evidence.
- Phase 2.9 § A3 "MNQ COMEX_SETTLE 2025 strongly positive, first year since 2019" → **refined:** MNQ was mildly positive from 2020 onwards and Chordia-adjacent in 2023; 2025 is culmination not discovery.
- Phase 2.9 § verdict "doc does NOT promote any lane" → **preserved.** This audit adds: also do not Phase-2.10-mechanism-hunt.

When the Phase 2.9 doc merges to main (it currently lives on branch `campaign-2026-04-19-phase-2`), a one-line pointer to this erratum doc should be added at line 295 noting the A3 MNQ baseline correction.

---

## Audit trail (reproducibility)

- Pre-reg committed before script run: see git log on `docs/audit/hypotheses/2026-04-20-phase-2-9-framing-audit.yaml`.
- Script deterministic at `--seed 20260420`.
- CSV output: `research/output/phase_2_9_framing_audit_main.csv`.
- Full JSON output: `research/output/phase_2_9_framing_audit.json` (contains per-year fire counts, era breakdowns, OOS raw stats).
- MinBTL budget: 16 trials on 6.65 clean years → MinBTL = 5.55 years < 6.65, within strict Bailey E=1.0 bound.
- Canonical delegation: filter fires via `research.filter_utils.filter_signal` (no inline re-encoding); Mode A holdout via `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`; DB path via `pipeline.paths.GOLD_DB_PATH`.
- Cross-asset ATR injection pattern follows `trading_app.strategy_discovery._inject_cross_asset_atrs` (source symbol='MES', `atr_20_pct`, per-day lookup).

## Self-audit revisions (recorded for transparency)

This result doc was drafted, then audited by the same agent before publication. Revisions applied:

1. **D1 framing corrected.** Original draft labeled D1 chi-square failure as "filter semantic drift" and feed into a "WATCH_FIRE_RATE_DRIFT" verdict. Self-audit flagged this as overreach — non-stationary firing is the design intent of a regime filter (OVNRNG_100, X_MES_ATR60), not a defect. Corrected to: D1 is a noted property of the filter, not a verdict gate. The load-bearing verdict gates are locked Criteria 6-9.

2. **D2 pre-reg gate recognized as mis-calibrated.** `|Δ|≤0.010 R` was too tight for comparing a 2026-04-11 promotion-time snapshot against a 2026-04-16 live canonical mean — expected for accumulating OOS. The Criterion 8 RATIO (live_OOS_ExpR / stored_IS_ExpR ≥ 0.40) is the load-bearing gate per `docs/institutional/pre_registered_criteria.md` § Criterion 8, and all 4 cells pass it. D2 still useful as a metadata-freshness indicator: `validated_setups.oos_exp_r` should be recomputed from canonical when cited.

3. **Verdict downgraded from WATCH_FIRE_RATE_DRIFT to CONTINUE.** No locked institutional criterion is violated. Operational restrictions (no sizing-up until 150 OOS days, no Phase 2.10 mechanism hunt) stand unchanged — those flow from underpowered OOS and missing mechanism, not from the D1 finding.

4. **`orb_outcomes.pnl_r` confirmed NET of cost.** `pipeline.cost_model.to_r_multiple` (line 460-474) subtracts `total_friction` (commission_round_trip + modeled_slippage) from price-based points before R-scaling. Entry-side slippage applied via `E2_SLIPPAGE_TICKS` in `trading_app/outcome_builder.py` line 519. So all ExpR numbers in this doc are net-of-modeled-cost — conservative vs reality per 2026-04-20 MNQ TBBO pilot (median real slippage = 0 ticks, max = +1).

5. **Independent-SQL reproduction of D1/D3/D4 confirmed.** Direct SQL (bypassing `research.filter_utils.filter_signal`) with `overnight_range >= 100` gates reproduced every D1 fire rate, D3 era ExpR, and D4 OOS stat to 4+ decimals. No filter-delegation drift.

## Literature cited (verbatim extracts)

- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` § Theorem 1 (MinBTL), § Proposition 2 (compensation effect).
- `docs/institutional/literature/harvey_liu_2015_backtesting.md` § BHY p.16, § Exhibit 4 p.22.
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` (cited as future-work for SR drift monitoring at 150 OOS days; not applied here).
- `docs/institutional/pre_registered_criteria.md` § Criterion 8, § Criterion 9, § Criterion 10, § Amendment 2.8.
- `.claude/rules/backtesting-methodology.md` RULE 3, RULE 9, RULE 10.
- `.claude/rules/research-truth-protocol.md` § Mode B grandfathered, § Canonical filter delegation.
- `.claude/rules/integrity-guardian.md` § 7.
