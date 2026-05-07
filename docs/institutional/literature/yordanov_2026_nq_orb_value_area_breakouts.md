# Yordanov 2026 — Opening Range Breakout Structure in E-Mini NASDAQ-100 Futures

**Source:** https://researchpaperfilteropen.vercel.app/
**Provenance:** Web-published working paper (no DOI). HTML retrieved via `curl` 2026-05-07; HTTP 200, 87141 bytes; UTF-8 decoded; 17/17 load-bearing claims in this extract independently re-verified against the source text (see "Verification" section below). Source page is dynamic (revisable) — extracted state is fixed at 2026-05-07.
**Author:** Georgi Yordanov (LinkedIn search returned multiple homonyms; specific author identity not independently verified — bibliographic risk noted in "Limitations" below).
**Publication:** 2026 working paper, self-published. **NOT peer-reviewed.**
**Period analyzed:** 2025-09-01 → 2026-04-24 (159 trading days)
**Data:** Databento CME Globex MBO (Market-By-Order) feed, dataset `GLBX.MDP3`, instrument `NQ.c.0` (E-mini NASDAQ-100 continuous front-month)
**Extracted:** 2026-05-07
**Plan reference:** `~/.claude/plans/image-1-analyse-jolly-kite.md` (post-thrust horizontal-range chart question)
**Mechanism-priors anchor:** `docs/institutional/mechanism_priors.md` § "Auction theory literature" (this extract closes a Phase-0 gap explicitly noted there since 2026-04-15).

**Criticality for our project:** 🟢 **HIGH for mechanism grounding, ⚠ MEDIUM for citation weight.**
- HIGH because: (a) instrument family overlap (NQ → MNQ already deployed), (b) data feed identical (Databento MBO), (c) full algorithmic methodology disclosed, (d) directly tests the value-area / balance / breakout-continuation mechanism that the project's `OVNRNG_*` family already partially captures.
- MEDIUM citation weight because: not peer-reviewed; single instrument; 8-month period; author identity not verified.

**How this maps to your chart question** (`~/.claude/plans/image-1-analyse-jolly-kite.md`): the horizontal box you drew on MGC 5m is structurally equivalent to "Filter Case A" in this paper — price contained inside the value area at a reference timestamp. The paper provides empirical hit rates for what happens *next* after such a containment, which is exactly the predictive overlay the project's NO-GO registry has not yet tested in this framing.

---

## § 1 The mechanism (verbatim, § 1 Introduction)

> "The Opening Range Breakout (ORB) is one of the oldest intraday frameworks in active use. The premise is straightforward: markets establish a reference range during the early session, and when price escapes that range, it tends to follow through in the breakout direction for a meaningful distance."

> "Rather than defining the opening range as a simple high/low window, this study uses the Volume Profile Value Area — the price range containing 70% of opening volume. This provides a structurally grounded reference zone that reflects where genuine two-sided price acceptance occurred during the opening session."

**Project-relevance:** Our existing ORB definition uses raw high/low of the ORB window (`pipeline/build_daily_features.py` ORB columns). This paper proposes substituting volume-weighted Value Area as the reference range. That is a structurally different predicate, NOT the same as our existing `ORB_G{N}` size filters or `OVNRNG_*` overnight-range filters.

---

## § 2.2 Value Area definition (verbatim, § 2.2)

> "All trades in this window construct a Volume Profile — a histogram of traded volume at each 0.25-point (one tick) price level. From this profile the Value Area is computed: the price range containing 70% of opening volume, expanded outward from the Point of Control (POC) using the standard single-tick expansion algorithm."

Definitions (verbatim, § 2.2):
- **VAH** — Value Area High (upper bound of the 70% volume zone)
- **VAL** — Value Area Low (lower bound of the 70% volume zone)
- **VA_Mid** — `(VAH + VAL) / 2` (geometric midpoint of the Value Area)
- **POC** — Point of Control (price with highest traded volume)

---

## § 2.3 Filter cases — the key structural classifier (verbatim)

The 5-minute candle closing at the end of the opening window determines the **Filter Case**:

| Case | Condition (close vs VA) | Filter High | Filter Low | Interpretation |
|------|--------------------------|-------------|------------|----------------|
| **A** | Inside VA (`VAL ≤ close ≤ VAH`) | VAH | VAL | Price accepted within opening value. Clean structural boundaries. |
| **B** | Above VAH | 10:00 candle high | VAL | Buyers extended above VA. Upper boundary set by opening session high. |
| **C** | Below VAL | VAH | 10:00 candle low | Sellers extended below VA. Lower boundary set by opening session low. |

**Project-relevance:** Case A is the empirical realization of the chart you drew (MGC 5m 2026-05-06/07). The horizontal box represents value-area containment.

---

## § 2.7 Two-Episode Framework (verbatim selection)

> "After Episode 1 (initial breakout), the session is monitored for a Filter Mid cross — a 5-minute close back through the Filter Mid in the direction opposite to the breakout. This signals the end of Episode 1."

> "If a cross occurs, the remaining session is scanned for Episode 2 (bo2) — a fresh breakout through either filter boundary. A key classification is whether Episode 1 reached the 0.5× deviation target before the cross (`bo1_hit_dev_before_cross`)."

---

## § 3.2 Headline base rate (verbatim, § 3.2)

> "The 71.1% hit rate at 0.5× establishes a strong baseline: after a Filter Range breakout, nearly three in four sessions extend at least half the filter range beyond the breakout level. The rate decays to 51.6% at 1×, 36.5% at 1.5×, and 25.2% at 2×."

| Deviation target | Hit rate (n=159) |
|------------------|------------------|
| 0.5× filter range | 71.1% |
| 1.0× | 51.6% |
| 1.5× | 36.5% |
| 2.0× | 25.2% |

---

## § 3.5 Filter Case is the key conditional predictor (verbatim)

> "At 0.5× deviation, all three cases perform broadly similarly (63–81%). The divergence widens sharply at deeper targets: at 1.5×, Case A reaches 54.1% while Case C reaches only 10.5%. This is the most structurally informative dimension for setting follow-through expectations."

| Case | Condition | Avg Filter Range | n (days) | 1.5× Target |
|------|-----------|------------------|----------|-------------|
| A | Close inside VA | 97.8 pts | 38 | 54.1% |
| B | Close above VAH | 140.6 pts | 21 | 23.8% |
| C | Close below VAL | 186.2 pts | 19 | 10.5% |

**Project-relevance:** Case A — the value-area-containment shape — produces the **highest** depth-of-follow-through. This **directly contradicts** the naive intuition that a "post-thrust horizontal range" is exhaustion. In NQ over 159 days, Case A is the *cleanest* structural breakout context.

---

## § 3.8 The strongest single signal — "Cross + Miss" warning (verbatim)

> "Three behaviorally distinct groups emerge:
> ① No Cross — Episode 1 ran to end of session (n=80): 85.0% at 0.5×, 66.2% at 1.0×. The strongest group.
> ② Cross + Dev Hit (n=35): This group by definition hit the 0.5× target (100%). At deeper targets, however, follow-through weakens — only 60.0% at 1.0×.
> ③ Cross + Dev Miss (n=44): Only 22.7% at 0.5× — a 48.4 percentage-point drop from the baseline 71.1%."

> "This is the clearest warning signal in the model: a breakout that retreats through the Filter Mid without having extended even 0.5× of the filter range has a high probability of full failure. With 159 days of data, this gap strengthens (from 42.7pp in the initial study), confirming the pattern is not a small-sample artifact."

**Project-relevance:** This is a **veto signal**, not an entry signal. Operationally that's far cheaper to test under our institutional rigor (fewer multiple-comparison degrees of freedom; tests "should we NOT trade this" rather than "should we trade this") and it's a directionally distinct hypothesis from anything in the NO-GO registry.

---

## § 3.6 Counter-intuitive finding: high-volume breakout candle is NOT a conviction signal (verbatim § 3.6)

> "The conventional intuition — high-volume breakout candles signal conviction — is not supported by the data. High-volume breakout candles show the worst follow-through at every target level: 65.4% at 0.5× vs. 80.8% for low-volume candles. The inverse relationship is consistent across all four deviation levels."

(Author later notes this signal **faded** with more data — at n=159 the volume effect largely disappears: Low=69.8%, Mid=75.5%, High=67.9% at 0.5×. § 4 Finding 4: ✗ No Edge.)

---

## § 4 Edge classification (author's own grading)

| Finding | Verdict |
|---------|---------|
| 71.1% baseline follow-through | ✓ Confirmed Structural Edge |
| Opening Δ% predicts continuation | ~ Mild Edge — gap narrows with more data |
| Filter Case predicts depth | ~ Conditional Edge — deeper targets only |
| Higher BO candle volume → conviction | ✗ NO EDGE — initial signal was small-sample artifact |
| DOWN BO candle delta filter | ✗ Signal Faded — initial n=6 bucket was noise |
| Cross + Miss warning | ✓ STRONGEST SIGNAL — 48.4pp below baseline |
| Episode 2 breakouts beat baseline | ✓ Confirmed |
| Directional asymmetry at deep targets | ✓ NEW — 23.7pp gap (DOWN reaches 2× more often than UP) |

---

## § 5 Limitations (verbatim, all five reproduced)

> "Sample size: 159 trading days provides substantially improved statistical grounding vs. the initial 78-day study. Key signals (Episode warning, base rate) are now reliable. Sub-group analyses at n<25 (e.g., DOWN × Positive delta, n=24; extreme volume buckets) still require caution."

> "Single instrument: All data is NQ continuous front-month futures only. Findings may not generalize to other equity indices (ES, YM, RTY), commodities, currencies, or different contract specifications."

> "Single time period: January–April 2026 represents one specific market regime. ORB behavior may differ materially in higher/lower volatility periods, strongly trending environments, or different macro regimes."

> "No transaction costs: Deviation hit rates measure raw price distance. Bid-ask spread, slippage, and commissions are not incorporated. Realized results would be lower, potentially materially so for tight targets."

> "No entry logic: This is a structural analysis of follow-through probability, not a complete strategy. Entry price (market vs. limit), stop placement, and position sizing are unmodeled."

---

## How this paper SHOULD and SHOULD NOT be used in our project

### LEGITIMATE uses
1. **Mechanism citation** for any pre-reg hypothesis testing value-area containment (Case A) as a pre-condition for ORB breakout strategies on **MNQ** (instrument-class-aligned).
2. **Veto-signal pre-reg** — testing the "Cross + Miss" pattern as a *don't-trade* gate on existing deployed lanes (low multiple-comparison cost, directionally distinct from the NO-GO registry).
3. **Methodology template** — the paper's MBO-based Value Area construction (§ 2.2) is fully reproducible against our existing Databento ingest. Any in-house replication study can lift the `pipeline/build_daily_features.py` ORB extraction logic and add a Volume Profile / VA computation alongside.
4. **Falsification material** — the paper's "high-volume = conviction" null result is consistent with the project's NR7/Crabel kills. Use as cross-evidence that volume-weighted intraday signals are noise-prone.

### ILLEGITIMATE uses
1. **As sole grounding for cross-instrument deployment.** Single-instrument (NQ) → MNQ extrapolation is the smallest stretch and still requires its own pre-reg. NQ → MGC is a much larger stretch (different asset class, different microstructure). Do NOT cite this paper to ground an MGC pre-reg on its own.
2. **As replacement for Dalton/Steidlmayer.** This is empirical work, not theoretical. The original AMT/Market Profile theory (Dalton, Steidlmayer) remains needed for any deeper mechanism-priors update to `docs/institutional/mechanism_priors.md`.
3. **As confirmation of a deployed-lane edge.** The 8-month period overlaps our `HOLDOUT_SACRED_FROM = 2026-01-01` cutoff. Treat the paper's results as IS-OOS-mixed; cannot import its hit rates as "OOS confirmation" of anything we deploy.

### Pre-registration notes if pursuing veto-signal
Per `pre_registered_criteria.md` and `backtesting-methodology.md`:
- Pathway-A K=1 single-cell test if testing veto on a single deployed lane.
- Direction: signs must align — Cross+Miss → reduce trades, baseline cross-no-cross hit-rate gap should reproduce on MNQ within OOS power tier.
- Cost: this is a **non-execution** signal (skip days), so transaction-cost objection (paper § 5) is irrelevant for our use.
- OOS power floor (`research/oos_power.py`) required before any binary kill.

---

## Prior project work on adjacent ground (must read before any new pre-reg)

**The project has already attempted Dalton-style value-area work in 2026-Q1 without literature backbone.** Honest extract reader must consider this archive before claiming Yordanov-grounded findings as novel:

**Archive scripts (`research/archive/`):**
- `research_dalton_80_rule.py`, `research_dalton_80_rule_deepdive.py` — Dalton 80% Rule (open outside prior VA → re-enter → traverse to opposite VA boundary). Tested across MGC/MNQ/MES, anchors 0900/1000/1100, three variants (touch_A_B, close_A_B, first_touch_1h).
- `analyze_value_area.py` — full VA reversion + breakout strategy, 72-cell grid (3 stop multipliers × 3 RR × 2 bin × 2 time × 2 modes), walk-forward.
- `research_dalton_filter_uplift.py`, `research_dalton_filter_anchor_uplift.py` — Dalton signal as an overlay/uplift filter on existing lanes.
- `research_dalton_mid_monetization.py` — VA midpoint as target.
- `research_dalton_mnq0900_oos.py` — MNQ 0900 anchor OOS-specific replay.

**Output artifacts (`research/output/`):**
- `dalton_80_rule_notes.md` — STRICT mode: hit rates 3–50%, mostly 20–30% (well below the 80% rule label). LOOSE mode: 60–100% but those are touch counts not directional outcomes.
- `dalton_filter_anchor_uplift.md` — Best lane: MNQ 0900 aggregate ON-trades N=90 (across 2024+2025+2026, 5.7% fire rate), Δ(on-off)=+0.7357R, aggregate WR ON/OFF 62.2%/32.3%, ΔDD favorable. MES 0900 also positive. MGC anchors all near-zero or negative. **Sub-30 fire rate raises Bailey-LdP DSR concern; small-N inflation per `feedback_oos_power_floor.md`.**
- `dalton_mnq0900_oos.md` — train year 2024 (ON N=48, WR=79.2%, +1.11R) → test 2025-2026 (ON N=42, WR=42.9%, +0.53R; OFF N=864, WR=28.4%, +0.078R; net uplift +0.4559R). **Two correctness caveats:** (1) the WR collapse 79%→43% is a regime shift the headline +0.46R hides — would fail C9 era stability under any reasonable single-split test; (2) the "test" block aggregates 2025 (Mode A IS under `HOLDOUT_SACRED_FROM=2026-01-01`) with 2026 (Mode A sacred OOS), so per `research-truth-protocol.md` § "Mode B grandfathered baselines" this CANNOT be cited as Mode A OOS evidence — it is a Mode B grandfathered split. KEEP/WATCH zone per the doc's own verdict guide. Not committed to `validated_setups`.
- `dalton_mid_monetization_notes.md` — VA midpoint as target: predominantly negative ExpR across MGC/MES/MNQ at 0900/1000/1100. KILL-zone.

**Status reading:**
- None of this work appears in `validated_setups`, `live_config`, `prop_profiles`, or `lane_allocation.json`.
- None of it is in the NO-GO registry (`STRATEGY_BLUEPRINT.md` § 5) — so it's "not killed, not validated."
- The work pre-dates Phase 0 literature grounding (2026-04-07) and the current Mode A holdout (`HOLDOUT_SACRED_FROM=2026-01-01`). Any IS/OOS conclusion in those output files used the Mode B grandfathered window and **cannot be cited as Mode A evidence** per `research-truth-protocol.md` § "Mode B grandfathered baselines".

**What this extract DOES change vs the prior archive:**
1. The archive tested *entry signals* (Dalton 80% Rule reversion, VA breakout entry); Yordanov 2026 + the Cross+Miss veto framing here is a **non-execution** gate on existing lanes — structurally different test.
2. The archive ran without a citable mechanism source; this extract IS that source for the empirical pattern (Filter Case A → highest follow-through depth).
3. The archive used *self-built* value-area definitions with bin-size hyperparameters (0.5 in `analyze_value_area.py`); Yordanov 2026 specifies the canonical 70%-volume single-tick-expansion algorithm (§ 2.2). Any new pre-reg should adopt the canonical definition, not invent a new one.

**What this extract DOES NOT change:**
1. Hit rates from the archive STRICT mode (3–50%) are honest data. They falsify any "Dalton 80% Rule actually hits 80%" claim on MGC/MNQ/MES.
2. The archive's MNQ 0900 +0.46R OOS uplift is a candidate-keeper in its own right, **independent** of Yordanov 2026. If pursued, requires fresh Mode A verify per `research-truth-protocol.md`.
3. The MGC anchor results (uniformly poor Dalton 80% performance on gold) raise an instrument-specific caution: NQ-derived value-area mechanism should **not** be assumed to transfer to MGC without independent MGC-side replication.

---

## Verification (independent re-check 2026-05-07)

Per `integrity-guardian.md` § 5 (Evidence Over Assertion), every load-bearing numeric claim in this extract was re-verified against the source HTML by `curl` re-fetch + grep. Result: **17/17 verified** including: 71.1% baseline, 48.4pp Cross+Miss gap, 22.7% Cross+Miss probability, 54.1% Case A 1.5×, 10.5% Case C 1.5×, 159 trading days, "Cross + Miss" warning name, "Filter Case A" label, VAH/VAL/POC vocabulary, "Databento" source attribution, "GLBX.MDP3" dataset code, "NQ.c.0" instrument code, "70% of opening volume" VA definition, "Single instrument" + "No transaction costs" limitations.

If the source URL changes or the page is revised, this extract becomes the historical record — not stale to its own citations, but verify before extending.
