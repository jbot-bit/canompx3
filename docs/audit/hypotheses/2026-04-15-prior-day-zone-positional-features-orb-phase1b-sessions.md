# Pre-Registration Phase 1B — Prior-Day Zone / Positional / Gap-Categorical Features on Untested Sessions

**Date registered:** 2026-04-15
**Session:** HTF
**Protocol:** B (same as parent)
**Parent pre-reg:** `2026-04-15-prior-day-zone-positional-features-orb.md` v4
**Status:** PRE-REGISTERED (sibling; no execution yet)

---

## 1. Why a sibling, not an extension

Phase 1A tested 3 sessions (CME_PRECLOSE, US_DATA_1000, NYSE_OPEN). **Sessions are statistically independent markets** — a NO-GO on NYSE_OPEN does not close EUROPE_FLOW, COMEX_SETTLE, or LONDON_METALS. Running the remaining high-edge sessions requires its own pre-registration with its own K budget and its own survivors.

This sibling inherits all gates, features, controls, and methodology from v4 parent. Only scope changes are declared below.

## 2. Scope changes from v4 parent

| Axis | v4 parent | v1B this sibling |
|------|-----------|-------------------|
| Sessions | CME_PRECLOSE, US_DATA_1000, NYSE_OPEN | **EUROPE_FLOW, COMEX_SETTLE, LONDON_METALS** |
| Instruments | MNQ, MES | MNQ, MES (same) |
| Apertures | O5 | O5 (same) |
| RR | {1.0, 1.5} | {1.0, 1.5} (same) |
| Features | F1-F8 | F1-F8 (same) |
| Direction | SYMMETRIC | SYMMETRIC (same) |
| Entry model | E2 | E2 (same) |
| Holdout | IS < 2026-01-01, OOS [2026-01-01, 2026-04-07] | same |

## 3. K accounting

Identical to parent: N_MinBTL=48, K_local=96 (this sibling's 96 p-values), K_global=639+96=735 (adds this sibling's 96 to the global universe; parent's 96 already in it).

Under per-feature family (reviewer-raised alternative to strict K=96): F6 at K=12 has Bonferroni = 0.05/12 = 0.00417.

Bailey MinBTL same disclosure: N=48 passes relaxed E=1.2 (5.38 yr needed vs 6.65 available).

## 4. Reporting enhancement (addresses reviewer-flagged bug in v4 run)

v4 script computed `partial_regression_mean_p` (cluster-robust-SE via statsmodels cluster-by-trading_day) but did NOT display in the results table. This sibling REQUIRES both columns in the output:
- `p_raw` (Welch's t, no clustering) — for comparison with v4
- `p_cluster` (OLS cluster-SE by trading_day) — the binding gate per parent §6

Binding p-value is `p_cluster`. Welch's `p_raw` is descriptive only.

## 5. Acceptance criteria (strict)

Identical to v4 parent §6 Table. Principal thresholds:
- Bonferroni-local: `p_cluster < 0.05 / 96 = 5.21e-4`
- BH-FDR local: `q < 0.05` at K=96 on `p_cluster`
- Chordia: `|t| ≥ 3.79` (cluster-robust t)
- Era stability: 5 eras, `ExpR ≥ -0.05` at N ≥ 50
- Holdout direction match + `OOS_ExpR ≥ 0.40 × IS_ExpR`
- Jaccard redundancy `< 0.40` vs deployed BASE_GRID_FILTERS + v4 survivors (none, so trivially satisfied)
- Cross-instrument coherence per v4 FM#1

## 6. Known risk specific to this sibling

**LONDON_METALS historically a gold-dominated session** — MNQ and MES on LONDON_METALS likely have thinner trade counts than on NYSE_OPEN. Expect cells where N_IS may not clear 100 (exploratory-only floor per v4 §6). Declared expected.

**EUROPE_FLOW has deployed `PrevDayRangeNormFilter` on MNQ** per `docs/specs/presession-volatility-filters.md:106`. Our 8 features are positional/zone, not magnitude — different signals. Jaccard gate will quantify overlap.

## 7. Pre-declared expected outcome

Given session independence and the one-session F6 finding on NYSE_OPEN (t=-3.40), base-rate expectation is 0-2 cells pass binding gates. DEFER is the likely verdict for cells with `N_OOS < 30`.

**Why test anyway:** a survivor on a different session would be a genuine NEW EDGE not currently exploited by our deployed portfolio. Asymmetric upside (find new lane) vs low cost (~1-2 hrs compute).

## 8. Banned actions (inherited from v4)

- Protocol switch after seeing result
- Direction flip
- Scope expansion to MGC or other sessions without new pre-reg
- Omitting CTE dedup
- Peeking before 2026-10-15 evaluation lock

## 9. Commit trail

- This pre-reg → committed
- Script: `research/prior_day_features_orb_phase1b.py` — SHA frozen post-first-run
- Results: `docs/audit/results/2026-04-15-prior-day-zone-positional-features-orb-phase1b-sessions-results.md`
