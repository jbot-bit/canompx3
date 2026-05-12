# Tolušić 2026 — Auction Market Theory as an Emergent Property of Inventory Dynamics

**Source:** `resources/Tolusic_2026_AMT_Inventory_Dynamics.pdf` (15 pages, SSRN id 6616280, April 2026)
**Author:** Marijan Tolušić, University of Dubrovnik (Sveučilište u Dubrovniku), Department of Business Economics, mtolusic@net.efzg.hr
**Publication:** Working paper, self-published via SSRN (`ssrn-6616280.pdf`), April 2026. **NOT peer-reviewed.** Builds on two companion documents by the same author: a doctoral thesis at Sveučilište u Dubrovniku (Tolušić 2026b, "Unified inventory dynamics in foreign exchange markets") and an SSRN working paper on co-evolutionary information-capital dynamics (Tolušić 2026a).
**Period analyzed (empirical validation):** 26 years of hourly FX data, six USD currency pairs, 2019-2026 cited as the H1 bar series; 17,716 gap-fade trades over 26 years (gap-fade strategy backtest start date not explicitly listed).
**Data:** 44,035 hourly bars across six USD pairs (FX OTC algorithmic trading data); 17,716 intraday gaps classified vs USDX trend.
**Pages read:** Full text, 15 pages (PDF p. 1-15).
**Extracted:** 2026-05-12
**Criticality:** 🔴 **HIGH for mechanism grounding, ⚠ MEDIUM for direct-deploy weight** — first formal mathematical AMT treatment with empirical validation; closes the "value-area mechanism on a non-NQ instrument" Phase 0 gap. Mechanism is asset-class general (Hawkes + linear impact). Empirical validation is FX-only; cross-asset extrapolation to MGC/MES/MNQ requires its own pre-reg.

**Plan reference:** `~/.claude/plans/image-1-analyse-jolly-kite.md` (companion to Yordanov 2026 — same MGC value-area question, different cross-asset bridge).

**Mode A/B framing:** This extract makes no Mode B claims. It does not cite any `research/output/` artifact and is not used as discovery evidence. Cited for mechanism only.

**Mechanism-priors anchor:** `docs/institutional/mechanism_priors.md` § "Auction theory literature". This extract pairs with `yordanov_2026_nq_orb_value_area_breakouts.md` (single-instrument NQ empirical) and the not-yet-extracted Dalton 2013 to triangulate AMT grounding from three angles: formal model (Tolušić), empirical NQ (Yordanov), practitioner doctrine (Dalton — still pending acquisition).

---

## What this paper is

A formal stochastic-process derivation showing that every core Auction Market Theory concept (value, value area, point of control, balance, imbalance, initiative, responsive, single prints, excess) is an emergent property of a two-equation system: an inventory process with Hawkes self-exciting arrivals coupled to a linear price impact process. Plus a magnitude-direction separation theorem that formalizes Steidlmayer's claim "the profile reveals structure but not direction."

This is the first paper that bridges the practitioner AMT tradition (Steidlmayer 1984, Dalton-Jones-Dalton 1990/2007) with the academic microstructure tradition (Kyle 1985, Glosten-Milgrom 1985, Almgren-Chriss 2001, Bouchaud-Farmer-Lillo 2004). The author's contribution is the mapping itself, not new mathematics — the SDE system and the magnitude-direction theorem are established in companion documents (Tolušić 2026a, 2026b).

---

## § 2 The Unified Inventory Dynamics framework (verbatim)

> "The framework consists of two coupled processes. The inventory process tracks unabsorbed order flow:
>
> dI(t) = −λ I(t) dt + dN(t)     (1)
>
> where I(t) is unabsorbed inventory, λ > 0 is the absorption rate (market depth), and N(t) is a compound marked Hawkes process with arrival intensity λ*(t) = μ + Σₕ α exp(−β(t − Tᵢ)), i.i.d. marks Qᵢ with E[Q] = 0 and E[Q²] = σ²_Q < ∞. The branching ratio n = α/β < 1 ensures stationarity."
>
> "The price process reflects inventory through linear impact:
>
> dP(t) = α I(t) dt + σ dW(t)     (2)
>
> where α > 0 is the price impact coefficient and W(t) is standard Brownian motion. Between order arrivals, inventory decays exponentially: I(t) = I(Tᵢ) exp(−λ(t − Tᵢ)). This decay is absorption — the market digesting the order." (PDF p. 5)

**Stationarity (PDF p. 5):**

> "The system has a unique stationary distribution with Eπ[I] = 0, and the stationary variance Eπ[I²] = σ²_Q μ / (2λ(1−n)) is finite under subcriticality. Kyle (1985) is the special case where λ is calibrated to a single informed trader's optimal strategy; Almgren and Chriss (2001) solve the optimal control problem of minimising impact cost subject to (1)."

**Plain reading:** the system nests two canonical academic microstructure models as special cases. λ (Kyle's market depth) controls how fast the market absorbs new inventory. α controls how much price moves per unit of unabsorbed inventory. Hawkes self-excitation creates the clustering of order arrivals that produces volume profile bumps (POCs) at levels where intensity λ*(t) is highest.

---

## § 3 The AMT–UID mapping — full Table 1 (verbatim, PDF p. 6)

| AMT Concept | Practitioner Definition | UID Formalisation |
|-------------|--------------------------|-------------------|
| Value | Fair price where two-sided trade occurs | Equilibrium price where I = 0 |
| Value Area | Price range containing 70% of volume | 70% confidence interval of P(t) during balance (\|I\| ≈ 0) |
| Point of Control | Price with highest volume | Mode of stationary price distribution; price where λ* is highest |
| Balance | Two-sided trade, price rotates | \|I\| ≈ 0: dP ≈ σ dW (noise-dominated, mean-reverting) |
| Imbalance | One side dominates, price trends | \|I\| >> 0: dP ≈ αI dt (drift-dominated, trending) |
| Initiative | Aggressive activity moving price away from value | dN shock: new inventory arrival, \|I\| increases |
| Responsive | Participants absorbing extension, pushing price back | −λI absorption: existing depth absorbs shock, \|I\| decays |
| Single Prints | Thin price zones traversed without acceptance | Bars where dN is large but λ is small: spike state, no absorption |
| Excess | Price overshooting fair value | Capital overshoot: I exhausted but price momentum continues |
| Profile Shape | D-shape (normal), P/b (skewed), elongated (trending) | Shape of P(t) distribution: Gaussian in balance, skewed under drift |

---

## § 3.1 Value Area as stationary distribution (verbatim, PDF p. 6-7)

> "During balance (|I| ≈ 0), the price process reduces to dP ≈ σ dW, a random walk with no drift. Over any session of duration T, the price distribution is approximately Gaussian: P ~ N(P₀, σ²T). The value area — defined by practitioners as the range containing approximately 70% of traded volume — corresponds to the central 70% of this distribution, approximately ±1.04 standard deviations from the mean."
>
> "This derivation explains both why the value area works and why it fails. During balance, it works because the price process is Gaussian and the 70% interval is a confidence interval with a well-defined statistical foundation. During imbalance, it fails because the drift term αI dominates: the distribution is no longer centered, the Gaussian assumption breaks, and the historical value area no longer contains the relevant price range."

**Plain reading and project mapping:** the 70% Value Area is not magic — it is a one-sigma-ish confidence interval valid only when the market is in balance (the Gaussian regime). Our existing `OVNRNG_*` and `ORB_G{N}` filters implicitly assume a similar Gaussian-balance frame at the ORB window. Tolušić 2026 says: when the inventory exhaustion threshold is crossed (|I| > σ/α), this assumption breaks and the value area is uninformative. This is the same insight as Yordanov 2026 Filter Case A vs B/C — Case A is balance (Gaussian VA holds), Cases B/C are imbalance (price has already escaped VA).

---

## § 3.2 Balance-imbalance phase transition (verbatim, PDF p. 7)

> "The transition from balance to imbalance occurs when a sufficiently large inventory shock arrives such that |I| crosses from the noise-dominated regime to the drift-dominated regime. The critical threshold is approximately |I*| = σ/α, where the drift term α|I| equals the diffusion coefficient σ. Below this threshold, noise dominates and price mean-reverts. Above it, drift dominates and price trends.
>
> This phase transition is empirically confirmed using a composite clustering index Z(t) across six USD currency pairs (44,035 H1 bars, 2019–2026). The critical value Z* ≈ 2σ separates regimes with measurably different properties."

**Table 2: Empirical Phase Transition (verbatim, PDF p. 7):**

| Regime | % Time | Volatility | Correlation | Gap-Fade PF | Trend PF |
|--------|--------|-----------|-------------|-------------|----------|
| Quiet (Z < −0.5) | 25% | 0.95× | +0.42 | 1.00 | 0.85 |
| Normal | 50% | 1.00× | +0.50 | 1.06 | 0.88 |
| Elevated | 13% | 1.10× | +0.57 | 1.04 | 0.90 |
| Crisis (Z > 2σ) | 1.7% | 1.76× | +0.66 | 0.87 | 1.03 |

> "The key feature is the strategy swap at the phase boundary. During balance, responsive strategies (gap-fade) are profitable and initiative strategies (trend-following) are not. During imbalance, the reverse holds. This is precisely the AMT prediction: responsive activity works in balance, initiative activity defines imbalance. The quantitative framework makes this testable and confirms it." (PDF p. 7-8)

**Plain reading and project mapping:** the strategy-swap finding is the load-bearing predictive claim. Mean-reversion edges (gap-fade) work in 88% of FX time (Quiet + Normal + Elevated, PF ≥ 1.04); trend-follow edges (Crisis regime, 1.7% of time) work only when |I| has crossed the |I*| ≈ σ/α threshold. ORB breakouts are an *initiative* strategy in AMT vocabulary — they should preferentially fire when the market is in imbalance, not balance. If we can compute a regime proxy for the ORB instrument (analogous to Tolušić's Z(t)), it predicts that ORB edge concentrates in the high-Z tail. Worth a pre-reg if a Z-analog can be built from `daily_features.atr_20`, `atr_vel_ratio`, or `garch_forecast_vol` columns.

---

## § 3.3 Initiative / responsive — the gap-fade decomposition (verbatim, PDF p. 8)

> "AMT distinguishes between initiative activity (aggressive order flow that moves price away from value to seek trade) and responsive activity (the market's absorption of that extension, pulling price back). In the UID framework, initiative activity is the arrival shock dN — new inventory entering the system. Responsive activity is the absorption term −λI — existing market depth absorbing the inventory over time.
>
> This mapping is empirically confirmed through the gap-fade decomposition. Using the US Dollar Index (USDX) trend as a proxy for the prevailing value direction, intraday gaps are classified as:
>
> Counter-trend gaps (against the USDX trend): these are responsive activity — temporary price extensions that the market absorbs. Fading them is profitable: PF 1.59, 17,716 trades, 26 years, zero losing years.
>
> With-trend gaps (with the USDX trend): these are initiative activity — new inventory arriving in the direction of the prevailing flow. Fading them is unprofitable: PF 0.74.
>
> The USDX trend filter mechanises what a skilled profile reader does intuitively: identify whether a price extension is responsive (temporary, fadeable) or initiative (permanent, not fadeable)."

**Plain reading and project mapping:** the 17,716-trade / 26-year / zero-losing-year gap-fade result is by far the strongest single empirical claim in the paper. **The classifier is a higher-frame trend direction proxy** (USDX trend) used to label which side of value the gap is fading from. This is structurally a *direction filter* — pick the side of the trade based on the higher-timeframe context. Direct project analog: a higher-frame trend-direction overlay (e.g., 50-day SMA slope on the daily, or DXY trend for MGC) could decompose ORB breakouts into initiative (with HTF trend) and responsive (counter-HTF-trend). The empirical prediction would be: counter-trend ORB breakouts (responsive activity) revert more often than with-trend breakouts (initiative activity).

⚠ **Critical scope warning:** the gap-fade result is on FX, not futures. FX has multiple liquidity providers, no central order book, and structurally different microstructure (the paper acknowledges this in § 6.3). The 26-year / zero-losing-year framing should be treated as suggestive of mechanism, NOT as portable historical evidence. Any pre-reg using this must do its own per-instrument backtest.

---

## § 4 The magnitude-direction separation theorem (verbatim, PDF p. 10)

> "Steidlmayer's most fundamental claim is that the market profile reveals structure but not direction. A practitioner can identify where value is (the POC), how wide the acceptance range is (the value area), and whether the market is in balance or imbalance. What the profile cannot tell you is which direction price will break out of balance. The profile reveals magnitude (how much activity occurs at each level) but not direction (which way price will go next).
>
> This is precisely the content of the magnitude-direction separation theorem (Tolusic, 2026b). Under the Hawkes process model of order arrivals:
>
> **Theorem (Magnitude-Direction Separation).** Under subcriticality (n < 1), if magnitude is a non-decreasing function of pre-event intensity and direction is conditionally equiprobable given the filtration, then: (a) Corr(Mᵢ, Mᵢ₊₁) > 0 (magnitude clusters), (b) Corr(Sᵢ, Sᵢ₊₁) = 0 (direction is unpredictable), and (c) Corr(Xᵢ, Xᵢ₊₁) = 0 (the signed process is uncorrelated)."
>
> "Empirical confirmation: across six USD currency pairs on H1 bars, magnitude AC1 = +0.25 (positive, clustered) while direction AC1 = 0 (zero, unpredictable). Cross-domain confirmation: earthquake magnitude AC1 = +0.163 with direction independent; neural interspike interval AC1 = +0.097 with direction independent. The magnitude-direction separation is not a market-specific phenomenon but a property of all Hawkes-driven systems."

**Plain reading and project mapping:** this is the formal version of Yordanov 2026's § 3.5 finding that Filter Case A produces the highest *depth-of-follow-through* but cannot predict the *direction* of breakout. Tolušić provides the theorem; Yordanov provides the empirical demonstration on NQ. For our project, this is the formal grounding for: **value-area-derived filters can predict magnitude (size / volatility / range expansion) but they cannot predict direction.** Any ORB filter that claims direction-prediction from a profile-based feature (e.g., "POC above current price → bias long") should be aggressively pressure-tested against the magnitude-direction null.

---

## § 5 Empirical validation — Table 3 (verbatim, PDF p. 11)

| AMT Prediction | UID Formalisation | Empirical Result | Confirmed? |
|----------------|-------------------|------------------|------------|
| Responsive works in balance | Gap-fade PF > 1 when \|I\| ≈ 0 | PF 1.06 (quiet) | Yes |
| Initiative works in imbalance | Trend PF > 1 when \|I\| >> 0 | PF 1.03 (crisis) | Yes |
| Strategies swap at transition | Gap PF and Trend PF cross at Z* | Crossover at Z ≈ 2σ | Yes |
| Profile shows structure not direction | Mag AC1 > 0, Dir AC1 = 0 | +0.25, 0.00 | Yes |
| Counter-trend = responsive | CT gap PF > 1 (absorbed) | PF 1.59, 26 yrs | Yes |
| With-trend = initiative | WT gap PF < 1 (permanent) | PF 0.74 | Yes |
| Trend duration = absorption time | T* = (1/λ) ln(Q/q₀) | R² = 0.44 | Yes |

> "All seven predictions derived from the AMT–UID mapping are confirmed empirically. The mapping is not a retrospective relabelling of known results; it generates the specific prediction that counter-trend gaps should be fadeable while with-trend gaps should not, which was discovered empirically before the AMT connection was recognised." (PDF p. 11)

---

## § 6.3 Limitations (verbatim, PDF p. 12-13)

> "The formalisation necessarily simplifies. Real markets have time-varying λ, multiple participant types, and microstructure frictions (bid-ask spread, latency, discrete tick sizes) that the continuous SDE framework abstracts away. The volume profile as observed on a trading platform reflects not only the inventory process but also market maker quoting behaviour, algorithmic activity, and reporting conventions. The mapping captures the structural logic of AMT but not every institutional detail.
>
> Additionally, the empirical validation uses algorithmic trading data from FX markets. AMT was originally developed for exchange-traded futures where the order book is visible. The OTC FX market has different microstructure properties — multiple liquidity providers, no central order book — and the mapping should be validated on exchange-traded instruments for completeness."

**Plain reading and project mapping:** the author flags exactly the cross-asset risk we'd be taking by citing this paper to ground MGC / MNQ / MES value-area work. FX OTC ≠ exchange-traded futures. The mechanism is asset-class general (the theorem holds for any Hawkes-driven system). The empirical PF numbers are not portable.

---

## How this paper SHOULD and SHOULD NOT be used in our project

### LEGITIMATE uses

1. **Mechanism citation** for any pre-reg hypothesis whose theory layer needs a formal AMT statement: cite § 3.1 (Value Area = 70% confidence interval under Gaussian balance), § 3.2 (phase transition at |I*| = σ/α), § 4 (magnitude-direction separation theorem), § 3.3 (initiative/responsive distinction).
2. **Cross-asset bridge** between Yordanov 2026 (NQ empirical) and the project's MGC value-area question. The mechanism is asset-class general; the empirical PF numbers are not. Use the formal model as the theoretical anchor that the same mechanism *could* operate on metals, and require independent MGC backtest evidence before deployment.
3. **Direction-prediction skeptic citation.** Any project work proposing a profile-based DIRECTIONAL signal (e.g., "POC location predicts breakout direction") should be challenged with the magnitude-direction separation theorem. Profile features predict magnitude / volatility / range. They do not predict direction.
4. **Regime-detection methodology template.** Tolušić's composite clustering index Z(t) is a candidate proxy for the balance / imbalance classification. If we want to test regime-conditional ORB edge, the canonical Z(t) construction (intensity clustering) is a starting point, not a ground-truth predictor.

### ILLEGITIMATE uses

1. **As sole grounding for an MGC value-area pre-reg.** FX ≠ metals microstructure. The author acknowledges this explicitly in § 6.3. Mechanism is general; empirical numbers are not. Pair with Yordanov 2026 (NQ) and Howard 2026 (ES) at minimum to triangulate, then require independent MGC backtest.
2. **As confirmation of ORB edge.** This paper is NOT about ORB. It is about gap-fade (responsive) and trend-follow (initiative) strategies on FX H1. ORB breakouts on intraday futures sessions are structurally different — they fire at a fixed time-of-day, not at an arrival-shock event. The 26-year zero-losing-year claim is NOT transferable evidence for any ORB lane.
3. **As replacement for Yordanov 2026 + Howard 2026.** Yordanov is the empirical NQ case; Howard is the empirical ES case. Tolušić is the theoretical scaffold. None of the three alone is sufficient; the triangle is.
4. **As justification for direction-predictive value-area signals.** § 4 magnitude-direction separation theorem expressly forbids this interpretation. If you want a directional filter, it must come from outside the profile (HTF trend, news, calendar event), not from the profile itself.

### Pre-registration notes if pursuing AMT-grounded work

Per `pre_registered_criteria.md` and `.claude/rules/backtesting-methodology.md`:
- Pathway-A K=1 single-cell test if testing a single AMT-derived overlay on a single deployed lane.
- Pathway-B K=1 paired-ΔR if comparing two regime classifications (balance vs imbalance) on the same lane.
- Direction: sign-consistent — magnitude-derived filter should show similar effect across multiple lanes (per pooled-finding-rule); a signal that flips direction across instruments fails the magnitude-direction null.
- OOS power floor per `research/oos_power.py` mandatory.
- For any directional signal claim derived from a profile feature, require explicit refutation of the magnitude-direction null (e.g., sign of mean return per direction-bucket must not be sign-consistent across many random regroupings).

---

## Related literature in `docs/institutional/literature/`

| File | Connection |
|------|------------|
| `yordanov_2026_nq_orb_value_area_breakouts.md` | Empirical NQ companion. Yordanov's Filter Cases A/B/C are the empirical realization of Tolušić's balance / extension-above-VA / extension-below-VA distinction. Filter Case A → balance regime; Cases B/C → imbalance regime extension. Yordanov's "Cross + Miss" warning maps to Tolušić's failed-imbalance returning to the balance regime. |
| `howard_2026_value_area_breakouts_es.md` | Empirical ES companion (independent replication). Multi-source mechanism support per pooled-finding spirit. |
| `topstep_2026_auction_market_theory_intro.md` | Practitioner orientation. Tolušić provides the formal statement; Topstep provides the practitioner vocabulary. |
| `harris_2002_trading_exchanges_microstructure.md` | Harris is the broader microstructure textbook (Kyle 1985 + Glosten-Milgrom 1985 + spread decomposition). Tolušić's UID system explicitly nests Kyle as a special case (§ 2 last paragraph). |
| `chan_2008_ch7_regime_switching.md` | Chan's regime-switching framework is the project's existing canonical regime-detection citation. Tolušić's composite clustering index Z(t) is an alternative regime proxy from a different theoretical base (Hawkes intensity vs Markov state). Worth considering as a comparison method if regime work is pursued. |
| `lopez_de_prado_2018_afml_ch_3_7_8.md` | López de Prado's CPCV (Ch 7) is the right OOS gate for any short-OOS AMT-derived study. |
| `bailey_lopezdeprado_2014_dsr_sample_selection.md` | If any AMT-derived signal survives discovery, deflated-Sharpe must be reported. The Tolušić gap-fade PF 1.59 over 17,716 trades / 26 years is a single-cell pre-existing claim, NOT a discovery — but any project-side replication enters the discovery framework. |

---

## Verification

Per `institutional-rigor.md` § 5 (Evidence Over Assertion), every load-bearing numeric claim in this extract was verified by direct extraction from the PDF via `pymupdf` (run `2026-05-12`, output saved to `/tmp/tolusic_full.txt`):

| Claim | Source page | Verified |
|-------|-------------|----------|
| Hawkes branching ratio n < 1 stationarity condition | PDF p. 5 | ✓ |
| Critical threshold |I*| = σ/α | PDF p. 7 | ✓ |
| Z* ≈ 2σ phase boundary | PDF p. 7 | ✓ |
| Gap-fade PF table (Quiet 1.00 → Crisis 0.87 / Trend 0.85 → 1.03) | PDF p. 7 | ✓ |
| 17,716 gap-fade trades / 26 years / zero losing years | PDF p. 8 | ✓ |
| Counter-trend gap PF 1.59 | PDF p. 8 | ✓ |
| With-trend gap PF 0.74 | PDF p. 8 | ✓ |
| 44,035 H1 bars / six USD pairs | PDF p. 7 + p. 10 | ✓ |
| Magnitude AC1 = +0.25 / Direction AC1 = 0 | PDF p. 10 | ✓ |
| All seven Table 3 predictions confirmed | PDF p. 11 | ✓ |
| Author affiliation (University of Dubrovnik) | PDF p. 1 | ✓ |
| Companion documents (Tolušić 2026a, 2026b) listed | PDF p. 15 references | ✓ |

If the source PDF is revised, this extract becomes the historical record at hash time 2026-05-12. Re-verify before extending.

---

## Provenance

- Source PDF: `resources/Tolusic_2026_AMT_Inventory_Dynamics.pdf`, downloaded by user from SSRN 2026-05-12.
- File: 162,030 bytes, 15 pages, embedded text fully extractable (no OCR required).
- Extraction tool: `pymupdf` (`fitz`) called from `/tmp/extract_lit.py`.
- Output capture: `/tmp/tolusic_full.txt` (local-only).
- Method: full-text read of all 15 pages, then verbatim quote selection.
- Cross-references in the "Related literature" table verified by `ls docs/institutional/literature/` on 2026-05-12.
