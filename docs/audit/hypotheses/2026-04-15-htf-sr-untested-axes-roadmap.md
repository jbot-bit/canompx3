# HTF/S-R Untested Axes — Roadmap (not a pre-registration)

**Date:** 2026-04-15
**Session:** HTF
**Purpose:** honestly catalog what the 2026-04-15-prior-day-zone-positional-features-orb pre-registration did NOT test, so future sessions don't over-read the Phase 1 NO-GO as closure of the whole HTF/S-R question.

---

## What Phase 1 DID test

`docs/audit/hypotheses/2026-04-15-prior-day-zone-positional-features-orb.md` v4:
- 8 binary daily-level features (F1 NEAR_PDH, F2 NEAR_PDL, F3 NEAR_PIVOT, F4 ABOVE_PDH, F5 BELOW_PDL, F6 INSIDE_PDR, F7 GAP_UP, F8 GAP_DOWN)
- 2 instruments (MNQ, MES)
- 3 sessions (CME_PRECLOSE, US_DATA_1000, NYSE_OPEN)
- 1 aperture (O5)
- 2 RR targets ({1.0, 1.5})
- 1 direction mode (SYMMETRIC)
- 1 entry model (E2)
- 1 θ primary (0.30)
- 1 filter role (binary pass/fail)
- Protocol B (Chordia t ≥ 3.79, no literature support)

**Verdict:** NO-GO on all 96 cells. F6_INSIDE_PDR MNQ NYSE_OPEN reaches t=3.40 (BH-FDR local PASS) but fails Chordia + cross-instrument coherence + holdout direction.

## What Phase 1 DID NOT test (ranked by expected information gain)

### High-value un-tested axes

1. **LONG/SHORT direction split** — pre-reg §4.1 disclosed Type II risk. Balanced-asymmetric effects invisible under SYMMETRIC. Our live strategies are direction-split.
   - *Reopen:* new pre-reg with direction-split scope; K doubles to ~192, pushes Bailey past E=1.2 relaxed bound, needs operational ceiling appeal OR drop to 2 features × 2 directions.
   - *Literature needed:* Fitschen Ch 3 (intraday trend-follow grounded) + direction-specific mechanism for one feature.

2. **EUROPE_FLOW and COMEX_SETTLE sessions** — both deployed edges (ExpR +0.097 and +0.121 per Blueprint). PDR magnitude filters already work on EUROPE_FLOW. Zone/positional filters on these sessions are untested.
   - *Reopen:* new pre-reg with session scope = {EUROPE_FLOW, COMEX_SETTLE}; K stays within E=1.2 bound.
   - *Literature needed:* same Protocol B approach; no new mechanism required since we're just broadening scope.

3. **MGC** — excluded by Bailey data-horizon. Gold traders use level psychology heavily.
   - *Reopen:* single-feature Protocol A on MGC with N ≤ 7 strict Bailey; requires level-theory literature (Dalton or Murphy, neither in `resources/`).
   - *Blocker:* literature acquisition first.

4. **O15 / O30 apertures** — levels may resolve over more time; prior-day range interactions at slower apertures untested.
   - *Reopen:* add O15/O30 to existing scope; K grows 3×, pushes past E=1.2.
   - *Literature needed:* Fitschen Ch 3 supports intraday trend-follow at multiple timeframes.

5. **Higher RR {2.0, 2.5, 3.0}** — our deployed strategies include up to RR2.5 (NYSE_CLOSE). Level-based filters may show stronger effects at higher RR because PDH/PDL act as natural targets.
   - *Reopen:* RR 2.0 add to existing features; K += 48.
   - *Literature needed:* none new; RR is operational parameter.

### Medium-value un-tested axes

6. **Weekly and monthly HTF levels** — our `daily_features` only stores daily-level features. Weekly_high, monthly_high, 5-day-rolling-high would require new pipeline work.
   - *Reopen:* add weekly/monthly features to `pipeline/build_daily_features.py`; separate from this study.
   - *Literature needed:* ideally Dalton or Murphy on HTF levels.

7. **Market profile features (VAH/VAL/POC)** — classical S/R framework. Not computed anywhere.
   - *Reopen:* major pipeline feature addition; Phase 3 work.
   - *Literature needed:* Dalton "Mind Over Markets" (not in `resources/`).

8. **Filter role variations:**
   - **CONDITIONING role:** does ABOVE_PDH change ExpR distribution (scale, skew, tail)? Different statistical test.
   - **DIRECTIONAL role:** does ABOVE_PDH predict LONG > SHORT? Separate from filter role.
   - **STOP-TARGET role:** does ABOVE_PDH change optimal stop distance (0.5x vs 1.0x)? Different hypothesis.
   - *Reopen:* role-specific pre-registrations; one per role; K manageable per role.
   - *Literature needed:* depends on role.

### Low-value / structurally-hard axes

9. **Continuous distance feature** (not binary zone). More statistical power but harder interpretation. Trade direction on prop lanes needs binary decisions.

10. **Confluence filters** (e.g., ABOVE_PDH ∩ weekly_high ∩ round_number). Multi-feature AND combinations. K-explosion risk; see companion NO-GO `2026-04-15-multi-timeframe-chain-full-scope-nogo.md`.

11. **θ sensitivity at {0.15, 0.50}** — pre-reg §6 mandatory θ-monotonicity check; deferred in Phase 1 run. All current cells are primary θ=0.30 only.
    - *Reopen:* same script, re-run at additional θ; no new pre-reg needed.

## What's closed (do not revisit without new data)

From Phase 1 direct results + prior project work:

- **8 specific binary filters on MNQ/MES × top-3 sessions × O5 × RR{1,1.5} × SYMMETRIC × Protocol B** — NO-GO (this Phase 1).
- **F6 INSIDE_PDR as SKIP signal on MNQ NYSE_OPEN** — t=3.40 but fails Chordia 3.79 under Protocol B. Interesting lead but not deployable as-is.
- **Multi-timeframe indicator-on-indicator chaining at full scope** — structural MinBTL-dead per `2026-04-15-multi-timeframe-chain-full-scope-nogo.md`.
- **`prev_day_range / atr_20` on NYSE_OPEN × MNQ** — KILLED per `presession-volatility-filters.md:78`.
- **`took_pdh_before_1000` × US_DATA_1000 × {MES, MNQ}** — QUARANTINED (WFE>1.89 suspect).
- **Crabel contraction/expansion session-level** — NO-GO per `RESEARCH_ARCHIVE.md:567`.

## Recommended next moves (priority order)

### Option α (cheapest, highest info-density)
Re-run the existing script at θ=0.15 and θ=0.50 to complete θ-monotonicity gate on F1/F2/F3. Cost: ~1 hour compute. Info: confirm or rule out knife-edge on the 3 proximity features. Does NOT reopen K budget.

### Option β (medium cost, direct relevance)
New pre-reg: direction-split on 3 strongest features (F6, F5, F1) × MNQ+MES × top-3 sessions × O5 × RR{1,1.5} × LONG/SHORT split. K = 3 × 2 × 3 × 1 × 2 × 2 = 72. Fits E=1.2. Addresses Type II risk disclosed in Phase 1. ~2 hours compute + reviewer pass.

### Option γ (higher cost, broader scope)
New pre-reg: existing 8 features × MNQ+MES × {EUROPE_FLOW, COMEX_SETTLE, CME_PRECLOSE} × O5 × RR{1,1.5} × SYMMETRIC. Addresses session-coverage gap. K=96, same as Phase 1 but different sessions. Reveals whether level signals work on EUROPE_FLOW where PDR magnitude works.

### Option δ (highest cost, literature-gated)
Acquire Dalton + Murphy + Crabel PDFs. Extract passages. Write Protocol A pre-reg with literature-grounded mechanism for one specific signal (e.g., "INSIDE_PDR SKIP on MNQ intraday based on market-profile balance-zone theory"). K ≤ 28 strict Bailey. Chordia t ≥ 3.00.

## Companion documents

- `docs/audit/hypotheses/2026-04-15-prior-day-zone-positional-features-orb.md` — the Phase 1 pre-reg (v4, grade A−)
- `docs/audit/results/2026-04-15-prior-day-zone-positional-features-orb-results.md` — Phase 1 results (NO-GO scoped)
- `docs/audit/hypotheses/2026-04-15-multi-timeframe-chain-full-scope-nogo.md` — structural NO-GO for unbounded chain
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` — literature extract grounding CORE strategy (not level filters)
- `research/prior_day_features_orb.py` — Phase 1 research script

## Do not close this question based on Phase 1 alone

Phase 1 tested ONE sliver. The HTF/S-R question has ≥11 un-tested axes. The honest claim is "8 specific binary prior-day-level filters do not beat Protocol-B null on 96 cells under MNQ+MES top-3 sessions O5 RR{1,1.5} SYMMETRIC" — not "HTF/S-R doesn't help ORB."
