# Howard 2026 — Stop Distance, Exit Methodology, and Signal Preservation in Intraday Value Area Breakouts (E-mini S&P 500)

**Source:** `resources/Howard_2026_Value_Area_Breakouts_ES.pdf` (23 pages, SSRN id 6350238, March 2026)
**Author:** Theo Johann Howard, Independent Researcher, ICL Academy (TheoJHoward1@gmail.com).
**Publication:** Working paper, self-published via SSRN (`ssrn-6350238.pdf`), March 2026. **NOT peer-reviewed.** Author acknowledges using Anthropic's Claude for code development and statistical analysis assistance (PDF p. 1 footnote).
**Period analyzed:** 2025-01 → 2026-01 (13 months, 6,284 events, 277 trading days).
**Data:** E-mini S&P 500 futures (ES), one-second trade data, CME source unspecified. Forward-fill applied during periods of no trading activity (limitation flagged by author, § 6.4).
**Pages read:** Full text, 23 pages (PDF p. 1-23).
**Extracted:** 2026-05-12
**Criticality:** 🔴 **HIGH for cross-asset triangulation, ⚠ MEDIUM for direct ORB applicability** — first systematic backtest of a value-area-breakout-continuation rule with full algorithmic formalization. Independent ES replication of the Yordanov 2026 NQ work (different instrument, different methodology, different period). The pooled-finding spirit on value-area mechanism now has 3 sources: Tolušić (FX, formal model), Yordanov (NQ empirical), Howard (ES empirical). Also: load-bearing **negative** result on price-based stops at structural boundaries — directly relevant to project ORB stop placement.

**Plan reference:** `~/.claude/plans/image-1-analyse-jolly-kite.md` (cross-asset triangulation of value-area mechanism).

**Mode A/B framing:** This extract makes no Mode B claims. It does not cite any `research/output/` artifact and is not used as discovery evidence for any project-side strategy. Cited for mechanism + stop-loss methodology only.

**Mechanism-priors anchor:** `docs/institutional/mechanism_priors.md` § "Auction theory literature" + § "Stop-loss / exit doctrine".

---

## What this paper is

Howard formalizes the "Setup 2" value-area-breakout-continuation pattern from Steidlmayer / Dalton tradition into a fully reproducible algorithm — session volume profile construction → breakout detection → acceptance verification → pullback measurement → entry → exit management. He then tests the unfiltered signal across 6,284 events on ES futures, and runs a 12-parameter stop-distance sweep plus a time-based exit-stack comparison.

The three load-bearing findings are (verbatim from abstract, PDF p. 1):

1. **The unfiltered continuation signal is not economically significant.** Pooled mean return is −2.20 ticks; day-clustered t = −1.37, p = 0.17. Remove single outlier month (April 2025, broad-tariff regime) and mean is statistically indistinguishable from zero (−0.19 ticks, p = 0.74).
2. **Price-based stop-loss rules are value-destroying at all 12 tested distances.** Stop selectivity peaks at 69%, below the approximately 75% threshold required to offset winner truncation. The mechanism is a selectivity failure tied to nearly symmetric MFE/MAE excursion distribution (median ratio ≈ 1.0).
3. **A time-based exit stack significantly outperforms the price-based baseline in 11 of 13 months (sign test p = 0.02).** Stack components: trailing stops, break-even tightening, no-progress timeouts, fail-reentry exits.

Two supporting findings:
- **Pullback depth** (normalized distance price retraces into the former value area before re-entry) is the strongest cross-sectional predictor: shallow pullbacks outperform deep pullbacks by +4.01 ticks (t = 3.40, p < 0.001). Survives Bonferroni correction.
- **Time-of-day effect:** events in the first 30 minutes (09:30-10:00 ET) produce −4.04 ticks (p < 0.001), significantly worse than later windows. Consistent with value area boundaries being unreliable during initial auction formation.

---

## § 5.1 Base Signal — the null result (verbatim, PDF p. 12)

> "The pooled horizon return across all 6,284 events is −2.20 ticks. Using naive event-level standard errors (SE = 0.68), this is statistically significant (t = −3.22, p = 0.001). However, day-clustered standard errors (SEclust = 1.83), which account for within-day correlation among events, yield t = −1.37 (p = 0.17) — not significant at any conventional level. The effect size is negligible regardless (Cohen's d = 0.04). The pooled win rate is 50.9% (z = 1.39, p = 0.17), not significantly different from 50%.
>
> [...] April 2025 is a statistical outlier by Grubbs' test (G = 2.75, critical value G0.05,13 = 2.46). [...] Removing April shifts the pooled mean from −2.20 to −0.19 ticks (p = 0.74). No other single month's removal changes the qualitative conclusion."

**Plain reading and project mapping:** the headline pooled value of −2.20 ticks is misleading three ways: (1) the naive SE is wrong (events within a day are not independent); the day-clustered SE wipes out significance, (2) April 2025 (broad-tariff regime) is a Grubbs outlier that drives the negative mean, (3) without April the result is zero. This is a class lesson in pooled-finding-rule territory: a pooled p that looks significant under naive clustering may evaporate under day-clustered or month-cluster standard errors. Our `pipeline/check_drift.py::check_pooled_finding_annotations` enforces per-cell breakdowns precisely against this pattern.

**Project-relevant takeaway:** the value-area-breakout-continuation hypothesis on ES (a different index) returns approximately zero edge, NOT a confirmation of Yordanov's 71.1% baseline on NQ at 0.5× deviation. The Yordanov claim is at a much shorter horizon (0.5× filter range, no fixed-time horizon) and on a different instrument; Howard's claim is at fixed 600-second horizon. The mechanism may be similar but the two papers measure different operational targets. Pair them carefully.

---

## § 5.2.1 Pullback Depth — the strongest cross-sectional predictor (verbatim, PDF p. 12-13)

> "Pullback depth is the strongest cross-sectional predictor. [...] A median-split comparison yields a difference of +4.01 ticks between shallow and deep pullbacks (t = 3.40, p < 0.001). This result survives Bonferroni correction for the five primary filter tests (αadj = 0.01)."

**Table 4: Horizon Returns by Pullback Depth (verbatim, PDF p. 13):**

| Pullback bin | n | Mean | WR (%) | t-stat | p-value |
|--------------|------|-------|--------|--------|---------|
| ≤0.25 (shallow) | 1,711 | +1.49 | 54.4 | +1.78 | 0.076 |
| 0.25–0.50 | 1,617 | −2.45 | 54.5 | −2.71 | 0.007 |
| 0.50–0.75 | 820 | −3.50 | 46.8 | −3.06 | 0.002 |
| 0.75–1.00 (deep) | 395 | −2.52 | 51.4 | −1.24 | 0.216 |
| > 1.00 (overshoot) | 1,741 | −4.89 | 45.9 | −3.04 | 0.002 |

> "The mechanism is intuitive: a shallow pullback indicates that the breakout boundary is being 'respected' — buying or selling pressure defends the level, consistent with the boundary representing a genuine transition in market consensus. A deep pullback, by contrast, suggests that the initial breakout may have been a false signal or that the value area boundary no longer represents a meaningful support/resistance level."

**Plain reading and project mapping:** the shallow-pullback bin (≤0.25, n=1,711, +1.49 ticks, WR 54.4%) is the only positive bin. This directly parallels Yordanov 2026 Filter Case A: shallow pullback / inside-VA close ≈ structural integrity intact → continuation likely. The empirical mapping between the two papers' classifications is strong even though the operational measures (Filter Case A in Yordanov; pullback depth ≤0.25 in Howard) are computationally different. Author flags this as "exploratory and requires confirmation on independent data" (§ 5.2.5 + § 6.4). For our project, the project-side analog would be a "structural-integrity gate" on the ORB breakout — does price retrace back near the ORB extreme deeply or shallowly after the break? Worth a pre-reg on `daily_features.first_pullback_depth_pct` (if such a column exists; if not, the trade-bar reconstruction is feasible per § 3.1 of the source paper).

---

## § 5.3 Stop-Distance Sweep — the negative result on price stops (verbatim, PDF p. 14)

> "We test twelve stop parameterizations: six boundary-relative (±2, 4, 8, 16, 32, 64 ticks inside the old value area) and six entry-relative (±4, 8, 16, 32, 64, 128 ticks from entry). [...] Every tested stop distance produces a worse mean realized return than the unmanaged horizon. The relationship is monotonic: wider stops yield better (less negative) mean returns, converging toward the horizon benchmark as the stop becomes less likely to fire."

**Table 5: Boundary-Relative Stops (verbatim, PDF p. 14):**

| Stop | n | Mean | Stop % | Select. |
|------|------|--------|--------|---------|
| Horizon (no stop) | 6,284 | −3.69 | 0% | — |
| Boundary ±2 | 6,284 | −25.28 | 77% | 62% |
| Boundary ±4 | 6,284 | −16.93 | 66% | 64% |
| Boundary ±8 | 6,284 | −11.80 | 55% | 66% |
| Boundary ±16 | 6,284 | −9.07 | 42% | 69% |
| Boundary ±32 | 6,284 | −8.29 | 30% | 67% |
| Boundary ±64 | 6,284 | −7.72 | 17% | 64% |

> "Entry-relative stops exhibit the same pattern but with even worse outcomes [...] Every entry-relative stop produces a worse mean than the horizon baseline."

### § 5.3.1 The Selectivity Mechanism (verbatim, PDF p. 15-16)

> "For a stop to improve expected value relative to the unmanaged horizon, selectivity must exceed a threshold that depends on the average cost of truncating winners versus the average benefit of early-exiting losers. [...] With our data's nearly symmetric MFE/MAE distribution (median ratio ≈ 1.0) and typical stopped-winner horizon returns in the range of 20–30 ticks, [the threshold] yields approximately 75%.
>
> The maximum observed selectivity across all tested stop distances is 69% (at boundary ±16 ticks). This shortfall is not marginal — it represents a structural inability of price-based stops to discriminate between winners and losers at value area boundaries."

### § 5.3.2 Winner Damage (verbatim, PDF p. 16)

> "Among horizon winners (events with rH > 0), fully 90.2% experience MAE ≥ 2 ticks — meaning that a boundary ±2 stop would have terminated 90% of eventual winners. Even at the widest tested boundary stop (±64 ticks), 30.5% of winners would have been stopped. Winners and losers are statistically indistinguishable in their early-trade adverse excursion patterns; the paths diverge only later, after the stop would have already fired."

**Plain reading and project mapping:** this is the most directly project-actionable finding in the paper. ORB strategies in our system use ORB-size-derived stops (E2 stop-market, ATR-relative). Howard demonstrates that price-based stops at structural boundaries (value area / VWAP / VAH/VAL) destroy value monotonically because MFE/MAE early dynamics are statistically indistinguishable between winners and losers. The 75% breakeven selectivity threshold is asset-class general (derived from MFE/MAE symmetry, not ES-specific). If our ORB lanes have a similar near-symmetric early MFE/MAE distribution, **any tightening of the price-stop at the ORB structural boundary should be expected to destroy value**. This is testable on our existing `daily_features` + `orb_outcomes` data via the standard MFE/MAE columns. **Direct pre-reg target.**

---

## § 5.4 Exit Stack — the time-based alternative (verbatim, PDF p. 17-19)

> "The exit stack outperforms the baseline in 11 of 13 months. The sign test rejects the null hypothesis of equal performance (p = 0.023). The largest improvements occur in the two most volatile months (April: +32.20 ticks; August: +18.20 ticks), where the stack's time-based exits divert trades away from the structural stop that fires on nearly every event."

**Table 7: Exit Layer Frequency Under the Exit Stack (verbatim, PDF p. 19):**

| Exit layer | Share (%) | Role |
|------------|-----------|------|
| Structural stop | 37–50 | Hard risk limit |
| Trailing stop | 20–25 | Profit protection |
| No-progress exit | 10–15 | Remove stalled trades |
| Break-even stop | 8–12 | Lock entry after 1R MFE |
| Fail-reentry exit | 3–6 | Remove value area re-entries |
| Target | 2–5 | Full profit capture |
| Horizon timeout | 5–10 | Residual time exit |

> "The stack's advantage derives from conditioning exit decisions on temporal evolution rather than instantaneous price, providing the discrimination that price-based stops lack." (§ 6.2 + § 7 conclusion)

**Plain reading and project mapping:** the time-stack reduces the structural-stop's exit share from ~77-100% (baseline) to 37-50% (stack), diverting the rest to trailing / no-progress / break-even / fail-reentry exits. The mechanism is *temporal conditioning* — instead of "exit if price crosses X", the rule is "exit if price has not made progress within 120s". The two most-volatile months (April +32.20 ticks, August +18.20 ticks) drive most of the stack's edge — meaning the stack's value is concentrated in high-volatility tail events. This is consistent with Tolušić 2026 § 3.2 (Crisis regime has different strategy edge than balance regime).

**Project-relevance:** our existing E2 stop-market and E3 break-bar exits are predominantly *price-based*. Considering a time-stack overlay (no-progress timeout at e.g. 120s post-entry, fail-reentry timeout if price re-crosses the ORB extreme) is a Stage-2+ Phase-D candidate. Worth a pre-reg on a single deployed lane.

⚠ **Methodological warning:** Howard's exit-stack comparison is on the SAME 6,284 events for both arms (counter-factual returns under each stop rule, computed offline from the same tick data). This is correct methodology for an apples-to-apples comparison but does NOT establish that the stack would generate positive realized P&L if deployed alone — both arms have negative pooled mean. The stack is "less bad", not "good". The base signal must be re-tested on different data with the stack in place before any deployment claim.

---

## § 5.2.3 Time-of-Day — verbatim, PDF p. 14

> "Events occurring in the first 30 minutes of the session (09:30–10:00) produce a mean horizon return of −4.04 ticks (p < 0.001), significantly worse than events after 10:00 (near zero). This is consistent with the hypothesis that value area boundaries from the prior session become more meaningful as the current session develops: early in the session, the market is still establishing its own value area, and the prior session's boundaries compete with the emerging current-session structure."

**Plain reading and project mapping:** value-area boundaries from the prior session are unreliable for the first 30 minutes of the current session. Project mapping: our `NYSE_OPEN` ORB starts at 00:30 Brisbane (09:30 ET). Howard's result implies any prior-VA-derived overlay on `NYSE_OPEN` ORB is structurally weakest at that lane, strengthening only on `US_DATA_1000` (01:00 Brisbane / 10:00 ET) and later. Worth conditioning any future VA-overlay pre-reg on time-of-day.

---

## § 5.6 Regime Dependence — verbatim, PDF p. 20

> "The 13-month sample exhibits substantial regime dependence. Three months (April 2025, August 2025, January 2026) show anomalous behavior: under the baseline exit system, the structural stop fires on 97–100% of events, producing deeply negative realized returns. In the remaining ten months, the stop fire rate ranges from 37–50%.
>
> This bimodal stop behavior suggests that the value area boundary's validity as a structural reference varies with market conditions. In high-volatility regimes, the value area itself may be poorly defined (wide, rapidly shifting), and its boundaries represent noise rather than meaningful support/resistance levels."

**Plain reading and project mapping:** the same volatility regime dependency identified in Tolušić 2026 Table 2 (Quiet → Crisis swap) appears here as a ~bimodal stop-fire-rate across months. This is independent corroboration of the Tolušić balance/imbalance phase transition on ES futures. Worth a separate audit-style note: regime classification should be part of any value-area-derived pre-reg, not an afterthought.

---

## § 4.4 Overfitting safeguards — the honest disclosure (verbatim, PDF p. 11)

> "All parameters in the v2.2 canonical specification (Table 1) were fixed prior to running the 13-month backtest. No walk-forward optimization, parameter grid search, or in-sample/out-of-sample splitting was performed. The stop-distance sweep and filter analysis are explicitly labeled as exploratory post-hoc analyses.
>
> We note that the absence of in-sample/out-of-sample splitting is a limitation; however, given that the primary base signal result is null (mean indistinguishable from zero), the risk of overfitting to a non-existent signal is low. **The filter finding (pullback depth, p < 0.001) should be treated as hypothesis-generating and requires confirmation on independent data.**"

**Plain reading and project mapping:** the author is explicit that the pullback-depth and stop-sweep findings are **post-hoc exploratory** and not validated on holdout. This is the Aronson 2007 / Bailey 2013 data-mining-bias regime. Even with p < 0.001 surviving Bonferroni, the finding is hypothesis-generating, not confirmatory. **Any project use of the pullback-depth filter must do its own Mode A IS verify + Mode A OOS test on a different period or instrument** before considering it ground-truth. This is exactly the role of `lopez_de_prado_2018_afml_ch_3_7_8.md` § 7.4 (CPCV) — the right OOS gate for a short-data hypothesis-generating finding.

---

## § 6.4 Limitations (verbatim, PDF p. 21-22)

> "Several limitations warrant acknowledgment. First, the 13-month sample, while covering over 6,000 events and multiple volatility regimes, represents a single macroeconomic era. [...]
>
> Second, the 120-second acceptance requirement, while ensuring genuine breakout conviction, guarantees that a substantial portion of any continuation move has already occurred before entry is triggered. Shorter acceptance thresholds or alternative confirmation mechanisms could yield different results.
>
> Third, no walk-forward or out-of-sample validation was performed. [...]
>
> Fourth, the analysis uses a single instrument (ES) in a single market (CME). Generalizability to other futures contracts, other exchanges, or other asset classes is unknown.
>
> Fifth, forward-filling the one-second price series during periods of no trading activity introduces potential biases in measured returns and excursions (Phillips and Yu, 2023). Events occurring during low-liquidity periods may have measured MFE/MAE values that reflect stale prices rather than executable levels."

---

## How this paper SHOULD and SHOULD NOT be used in our project

### LEGITIMATE uses

1. **Cross-asset triangulation of the value-area mechanism.** With Yordanov (NQ) + Howard (ES) + Tolušić (FX) we have 3 independent sources on the value-area / Setup-2 mechanism family. Pooled-finding spirit (per `.claude/rules/pooled-finding-rule.md`) requires per-cell breakdowns — these three papers, *as a set*, satisfy the multi-source requirement at the *mechanism* level even though their operational measures differ.
2. **Stop-loss methodology citation.** § 5.3 + § 6.1 are the canonical empirical demonstration that price-based stops at structural boundaries destroy expected value, with a quantitative selectivity framework. Cite when proposing any new stop-rule pre-reg.
3. **Selectivity framework (Equation 2).** Selectivity* = r̄ᴴ_W / (r̄ᴴ_W + S). This is the project's missing analytical tool for evaluating whether a tighter stop *can* improve EV given the MFE/MAE distribution. If MFE/MAE is symmetric, no tighter stop can improve EV. Worth coding into `research/` as a reusable analysis utility.
4. **Time-based exit-stack template.** § 5.4 + Table 7 is a complete recipe for a 7-layer exit system (structural / trailing / no-progress / break-even / fail-reentry / target / horizon). Project-applicable as a Stage 2+ overlay candidate, especially on lanes with bimodal stop-fire patterns under our existing E2.
5. **Time-of-day caveat for VA overlays.** § 5.2.3 says prior-session VA boundaries are unreliable in the first 30 minutes of the new session. Constrains where on the ORB session catalog a VA-derived overlay can apply.

### ILLEGITIMATE uses

1. **As confirmation of Yordanov 2026.** Howard finds zero base signal on ES; Yordanov finds 71.1% baseline at 0.5× on NQ. These are NOT confirming each other at the operational level (different horizons, different operational measures, different instruments). They confirm each other only at the *mechanism* level: shallow-pullback / Filter-Case-A / inside-VA-close → continuation more likely. Direct numerical cross-reference is misleading.
2. **As justification for the time-based exit stack as a positive-EV system.** The stack is "less bad" than the baseline on this data — both arms have negative pooled mean. Deployment would require independent positive-EV validation.
3. **As basis for an MGC value-area pre-reg.** Single-instrument (ES) → MGC is a large stretch. ES and NQ are equity indices; MGC is metals. Mechanism is asset-class general (Steidlmayer + Tolušić); empirical numbers are not. Multi-instrument cross-validation required before MGC deployment.
4. **As OOS evidence for any of our deployed lanes.** Howard's 2025-01 → 2026-01 period overlaps our `HOLDOUT_SACRED_FROM=2026-01-01` cutoff. Even if we replicated Howard's methodology on our data, we'd be running on a Mode B grandfathered window. Pure Mode A would require post-2026-04 forward data.

### Pre-registration notes if pursuing Howard-grounded work

Per `pre_registered_criteria.md` and `.claude/rules/backtesting-methodology.md`:
- **Stop-rule pre-reg:** Pathway-B K=1 paired-ΔR comparing price-based stop (current E2 stop-market) vs time-based no-progress exit (e.g., exit if no 0.5R MFE within 120s after entry). Per `pooled-finding-rule.md`, report per-lane breakdown not just pooled.
- **Pullback-depth filter pre-reg:** Pathway-A K=1 single-cell test on `daily_features.first_pullback_depth_pct ≤ 0.25` overlay applied to one deployed lane. Survives Bonferroni in Howard's data but is post-hoc exploratory; treat as hypothesis-generating, not confirmatory.
- **Mode A discipline:** all backtests against `HOLDOUT_SACRED_FROM = 2026-01-01`; Howard's data overlap is informational only.
- **OOS power floor** per `research/oos_power.py`.
- **Selectivity decomposition** required for any stop-rule comparison: report stopped-event MFE/MAE ratio + selectivity vs the breakeven threshold from Equation 2.

---

## Related literature in `docs/institutional/literature/`

| File | Connection |
|------|------------|
| `yordanov_2026_nq_orb_value_area_breakouts.md` | Companion NQ-empirical paper. Yordanov's Filter Case A (close inside VA) ↔ Howard's shallow pullback (≤0.25). Yordanov's "Cross + Miss" warning ↔ Howard's deep pullback / overshoot (≥1.0). Triangulation at mechanism level, NOT direct numerical confirmation. |
| `tolusic_2026_amt_inventory_dynamics.md` | Formal AMT model (FX, theoretical). Howard's bimodal stop-fire rate across months is consistent with Tolušić's balance/imbalance phase transition at \|I*\| = σ/α. |
| `chan_2013_ch1_backtesting_lookahead.md` | Howard's "forward-fill during no-trading periods" caveat is a microstructure-grade look-ahead concern (Phillips-Yu 2023 cited). Chan Ch 1 already grounds this class of bias. |
| `harvey_liu_2015_backtesting.md` | Howard's day-clustered SE correction (naive p=0.001 → clustered p=0.17) is the empirical analog of Harvey-Liu's Sharpe haircut: naive inference systematically overstates significance. |
| `bailey_lopezdeprado_2014_dsr_sample_selection.md` | Howard's post-hoc filter analysis is exactly the regime where DSR matters. Pullback-depth survives Bonferroni at α_adj = 0.01 but Bonferroni is FWER, not the deflated-Sharpe framework. |
| `lopez_de_prado_2018_afml_ch_3_7_8.md` | LdP 2018 § 7 CPCV is the right OOS gate for Howard's hypothesis-generating findings (short data, no IS/OOS split done). |
| `topstep_2026_auction_market_theory_intro.md` | Practitioner orientation. Howard formalizes one specific pattern (Setup 2) from the broader Dalton/Steidlmayer practitioner corpus. |
| `aronson_2007_ebta_data_snooping.md` | Howard acknowledges his filter analysis is post-hoc exploratory and hypothesis-generating only. This is exactly Aronson's "data miner's mistake" warning applied honestly. |

---

## Verification

Per `institutional-rigor.md` § 5 (Evidence Over Assertion), every load-bearing numeric claim in this extract was verified by direct extraction from the PDF via `pymupdf` (run 2026-05-12, output saved to `/tmp/howard_full.txt`):

| Claim | Source | Verified |
|-------|--------|----------|
| 6,284 events / 13 months (2025-01 → 2026-01) | Abstract + § 4.1 | ✓ |
| Pooled mean −2.20 ticks / naive t = −3.22 p=0.001 / day-clustered t = −1.37 p=0.17 | Table 2 + § 5.1 | ✓ |
| April-removed mean −0.19 ticks p=0.74 | Table 3 + § 5.1 | ✓ |
| Pullback ≤0.25 shallow: n=1,711, mean +1.49, WR 54.4%, t=+1.78, p=0.076 | Table 4 | ✓ |
| Shallow vs deep difference +4.01 ticks (t=3.40, p<0.001), Bonferroni-corrected | § 5.2.1 | ✓ |
| Boundary ±16 max selectivity 69%, threshold ~75% | Table 5 + § 5.3.1 | ✓ |
| MFE/MAE median ratio ≈ 1.0 (nearly symmetric) | § 5.3.1 + § 6.2 | ✓ |
| Horizon winners with MAE ≥ 2 ticks: 90.2% | § 5.3.2 | ✓ |
| Exit stack better in 11/13 months, sign-test p = 0.02 | Table 6 + § 5.4 | ✓ |
| First-30-min mean −4.04 ticks p<0.001 | § 5.2.3 + § 5.5 | ✓ |
| Three regime-anomaly months (Apr/Aug 2025, Jan 2026) | § 5.6 | ✓ |
| 277 trading days; day-clustered SE=1.834 | § 5.7.3 | ✓ |
| 120-second acceptance requirement | § 3.2 + § 6.4 limitation | ✓ |

If the source PDF is revised, this extract becomes the historical record at hash time 2026-05-12. Re-verify before extending.

---

## Provenance

- Source PDF: `resources/Howard_2026_Value_Area_Breakouts_ES.pdf`, downloaded by user from SSRN 2026-05-12.
- File: 375,337 bytes, 23 pages, embedded text fully extractable (no OCR required).
- Extraction tool: `pymupdf` (`fitz`) called from `/tmp/extract_howard.py`.
- Output capture: `/tmp/howard_full.txt` (local-only).
- Method: full-text read of all 23 pages, then verbatim quote selection on load-bearing sections (Abstract, § 5.1 Base Signal, § 5.2.1 Pullback Depth, § 5.3 Stop-Distance Sweep, § 5.4 Exit Stack, § 5.6 Regime Dependence, § 6.4 Limitations).
- Cross-references in the "Related literature" table verified by `ls docs/institutional/literature/` on 2026-05-12.
