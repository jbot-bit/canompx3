# R&D Graveyard — Consolidated NO-GO Registry

**Purpose:** Before proposing any new research direction, check this file. If the user's idea matches a killed hypothesis, flag it and cite the postmortem. Reopen only with a pre-registered critique of the original failure mode.

**Scope:** Hypotheses, filters, strategies, and architectural approaches that have been tested or analyzed and are now CLOSED, DEAD, or PAUSED. This is wider than `STRATEGY_BLUEPRINT.md`'s NO-GO — it includes ML attempts, portfolio reorganizations, and architecture decisions.

**Last consolidated:** 2026-04-18.

**Provenance note:** The entries below were compiled from the project's memory index (`memory/MEMORY.md` as surfaced in the session that built this file), not by re-reading every primary postmortem or audit file. Specific numbers (e.g., "CPCV AUC 0.50", "holdout dollar lift -$18K", "corr=0.069") are quoted from the memory-index summary lines — if you need them for a decision, open the cited postmortem path and re-verify. Do not cite these numbers in new research docs without primary-file re-verification.

---

## Rule for re-opening

Any entry here can be reopened ONLY with:
1. A pre-registered hypothesis file at `docs/audit/hypotheses/YYYY-MM-DD-<slug>.md`
2. An explicit critique of the original failure mode (what's different this time?)
3. A new dataset OR a fundamentally different feature class OR a corrected methodology

"Let's try again with slightly different parameters" is NOT valid reopening.

---

## ML — all three attempts dead

### ML V3 DEAD + DELETED (2026-04-11)
**Hypothesis:** pooled-confluence random forest over volume + volatility + timing features across deployed lanes.

**Result:**
- CPCV AUC = 0.50 (no predictive power)
- Holdout dollar lift = -$18K
- Per-strategy local losses on every lane

**Structural finding (the key one):** filters ≠ multivariate ML features. The extreme-value signal that makes volume features produce 48 BH-FDR survivors AS FILTERS is smoothed away in a multivariate RF. Tree-based methods average over the tail; the tail IS the edge.

**Status:** `trading_app/ml/` entirely DELETED from repo. Blueprint §5 NO-GO + §6 rewritten PERMANENTLY DEAD.

**Reopen criteria:** new feature class + pre-registration + new dataset + explicit V3 postmortem critique.

**Postmortem:** `docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md`.

### ML V1 + V2 DEAD (prior)
Superseded by V3 attempt which was also dead. Same structural reason — multivariate smoothing kills extreme-value signals.

---

## Portfolio Dedup Sprint — PREMISE WRONG (2026-04-14)

**Hypothesis:** consolidate live book by deduplicating strategies with `is_family_head=False` (retire the non-heads).

**Result:** Sprint premise was wrong. `is_family_head` is a LABEL, not a retirement flag. Allocator picks non-heads when trailing metrics favor them (3 of 6 live lanes were non-heads at audit time). 6 live lanes = 6 distinct families = 6 sessions → architecturally clean, no dedup needed.

**Status:** sprint artifacts + prompt deleted. Real unlocks (Half-Kelly sizing, narrow cull) remain valid but gated by F-1 XFA dormant.

**Memory:** `portfolio_dedup_nogo.md`.

---

## H2 book closed — Path C (2026-04-15)

**Hypothesis path:** volume × volatility composite on MNQ COMEX_SETTLE (rel_vol_HIGH_Q3 × garch_vol_pct≥70).

**Result — nuanced, not full kill:**
- T5 family PASS — `garch_vol_pct≥70` universal across 527 combos (68.5% positive, every instrument ≥62%)
- DSR at empirical var_sr=0.0174 ambiguous — passes K=5 marginal, fails K=12/36/527/14261
- **Composite rel_vol × garch: NO SYNERGY.** corr=0.069 (orthogonal) but BOTH ExpR +0.220 < garch-alone +0.263. rel_vol is SUBSUMED by garch on this cell.
- IS 4-cell dollars: both higher per-trade because joint-fire days are bigger-risk ORBs; per-R edge still favors garch-alone.

**Decision:** NO CAPITAL. Signal-only shadow H2 + top-3 universality. Path C book closed.

**Path A (HTF level features) DEFERRED** — needs `prev_week_*` / `prev_month_*` pipeline features built first.

**Artifacts:** `docs/audit/results/2026-04-15-path-c-h2-closure.md`, `research/close_h2_book_path_c.py`.

---

## GC → MGC cross-validation — PROXY DEPLOYMENT DEAD (2026-04-11)

**Hypothesis:** strategies validated on GC (16yr) transfer to MGC (real micro 3.8yr).

**Result:** 9/10 FAIL. Edge does NOT transfer GC → MGC. MGC proxy deployment path is dead.

**Structural reason:** parent-vs-micro spread, different liquidity regime, different tick structure.

**Status:** GC is research-only, not deployable. MGC kept `deployable_expected=False` until real-micro horizon reaches ~5yr (~2027-06).

**Memory:** `gc_mgc_cross_validation_results.md`.

---

## IBS + NR7 filters — NO-GO (2026-04-13)

**Hypothesis:** external-IBS (internal bar strength from prior day) + NR7 (narrowest range of last 7) as ORB filters.

**Result:**
- IBS Q1 t=3.89 N=74 IS on CME_PRECLOSE O15 RR1.0 (initial hit). Cross-confirmed MES t=3.45 N=44.
- **Holdout REVERSES both instruments.**
- 34/192 BH-FDR IS survivors → 0 after holdout validation.
- NR7: 0/85 BH-FDR, direction flips ALL 9 combos. Session-range NR7 also tested: 0/96.

**Status:** no reopen path. IBS and NR7 remain DEAD for ORB filtering.

**Memory:** `ibs_nogo_corrected.md`, `nr7_nogo.md`.

---

## Gap-fill / vol-spike / narrow IB — HYPOTHESES DEAD (2026-04-13)

**Four hypotheses audited in 425-test exhaustive scan:**

| Hypothesis | Result |
|------------|--------|
| Gap-fill signal (gap at open → mean-revert to fill) | DEAD |
| Volatility-spike (high ATR vel → ORB breakout) | REVERSED (inverse signal weakly better) |
| Narrow inside-bar (NR7-like ORB) | REVERSED (wider IBs do better) |
| VWAP gate (price > VWAP at ORB end) | **DEPLOYED** (MNQ US_DATA_1000 O15 RR1.0/1.5/2.0) |

**VWAP gate is the SURVIVOR.** Grade A- per code review; caveats: OOS peeked, DSR<0.95.

**Commits:** `f301b887` + `28b377aa`. Memory: `filter_hypothesis_audit_apr13.md`.

---

## Phase D Garch allocator — PAUSED (2026-04-18)

**Three versions tested (A4a, A4b, A4c):**
- **A4b:** found "working" metric — turned out to be mis-metric bug
- **A4c:** fixed the bug, ran clean. Harness gate PASS. Candidate FAILED primary on both surfaces. Destruction-shuffle clean. OOS direction flipped. **Real null.**

**Decision:** garch-family allocator path PAUSED. No reopening until meaningfully different mechanism pre-registered.

**Next:** fresh EV queue, not more garch allocator work.

**Commits:** framing `1a721e92`, verdict `a88505cd` (local, push TBD).

**Memory:** `allocator_a4c_null_parked.md`.

---

## Adversarial fade — DEAD (2026-04-15)

**Hypothesis:** 23 SKIP cells (filter-passes-but-ExpR-negative) from comprehensive scan, re-tested with direction flipped to SHORT.

**Result:** 13 FADE_FAILS + 9 FADE_WEAK + 1 CONDITIONAL (p=0.047 marginal). Level-near is NOISE regime, not a directional-inverted edge. Both directions underperform.

**Caveat:** E2 stop-market only — does NOT rule out E_RETEST (limit-on-retest after failed break).

**Status:** DEAD for E2. Deferred reopen for E_RETEST when Phase C infra ready.

**Memory:** `fade_audit_apr15.md`.

---

## Tier 2 "framework-killed near-misses" — ALL DECAYING (2026-04-09)

**Hypothesis:** 5 HANDOFF "framework-killed near-misses" may have been prematurely closed.

**Result:** All 5 decaying or noise. CANCELLED.

**Memory:** `tier2_truth_audit_apr9.md`.

---

## Batch T0-T8 — 27 HOT/WARM cells (2026-04-15)

**Audit scope:** 27 cells from comprehensive scan surviving promising-tier.

**Results:**
- 2 VALIDATED (0 FAIL): MNQ BRISBANE_1025 O15 RR2.0 short F3_NEAR_PIVOT_30 (N=607, ExpR=-0.124 SKIP); MES US_DATA_830 O30 RR1.0 short F2_NEAR_PDL_30 (N=253, ExpR=-0.246 SKIP)
- 17 CONDITIONAL
- 8 KILL

**KILL reasons:** T7 per-year instability (3× MES CME_REOPEN), T8 cross-instrument divergence (2× MES NYSE_CLOSE short F1), T4 sensitivity, T7+T3 combos.

**Memory:** `batch_t0t8_apr15.md`.

---

## Horizon T0-T8 non-volume — 5 candidates (2026-04-15)

**Results:**
- **H2 VALIDATED (8P/0F):** MNQ COMEX_SETTLE O5 RR1.0 long `garch_forecast_vol_pct≥70`. Signal-only shadow pre-registered. NOT DEPLOYED (same DSR-caveat doctrine as rel_vol).
- H1 CONDITIONAL (7P/1F): MES LONDON_METALS O30 RR1.5 long `overnight_range_pct≥80`. Fail = T3 WFE=1.33 LEAKAGE_SUSPECT driven by N_OOS_on=11.
- H3/H4/H5 CONDITIONAL: `is_monday` / `dow_thu` / `ovn_took_pdh` all fail T3 thin-OOS only.

**Zero KILLs, zero DUPLICATE_FILTERs, zero ARITHMETIC_ONLY.**

**Memory:** `t0_t8_audit_horizon_non_volume.md`.

---

## REL_VOL finding — EDGE_WITH_CAVEAT (2026-04-15, v2 stress tested)

**Hypothesis:** `rel_vol_HIGH_Q3` (ORB break_bar_volume / 20-day same-minute median) passes BH-global at K=14,261 on 5 lanes.

**Result:**
- CORE hard gates 4/4 (bootstrap p=0.0005, temporal both halves, exceeds K=14K noise max, per-day t sig)
- DSR informational/ambiguous — fails at K≥36, passes partially at K=5
- Per-trade Sharpe: 0.05-0.17 — MODEST (expected robust edge ≥0.25)
- Verified clean via `build_daily_features.py:1326-1380` (explicit "No look-ahead" comment)

**Decision:** DO NOT deploy capital yet. Signal-only shadow 6-12mo + Shiryaev-Roberts monitor OR composite with orthogonal signals (Phase D) before live.

**Key lesson:** v1 stress test over-punished using `dsr.py` default `var_sr` (wrong population) + K=14261 (inflated N_eff). v2 corrected: empirical var_sr=0.012, DSR reported at K=5/12/36/72/300/900/14261, treated informational-only.

**Memory:** `rel_vol_stress_test_apr15.md`.

---

## Pre-Apr-15 kills (from STRATEGY_BLUEPRINT NO-GO registry)

See `STRATEGY_BLUEPRINT.md` § NO-GO REGISTRY for the canonical project-specific list. Includes (non-exhaustive):
- `double_break` filter — look-ahead bias, banned forever
- NODBL filter — removed Feb 2026 after 6 strategies proved artifacts
- E0 entry model — purged as survivorship contributor
- Arbitrary 30-min clock-grid sessions — all negative

---

## Architecture-level kills

### Scratch DB at `C:\db\gold.db` — DEPRECATED (2026-03)
Caused stale-data bugs across sessions. Canonical is `<project>/gold.db`. Drift check #37 verifies; #62 blocks hardcoded scratch defaults in code.

### Parallel cost/session/instrument encodings — BANNED
Every re-implementation drifts. See E2 canonical-window fix (2026-04-07) for the proof — divergence between backtest and live was a look-ahead risk per Chan 2013 p4. Single-source-of-truth list at `CANONICAL_VALUES.md` §6.

### Mode B holdout — RESCINDED (2026-04-08)
Brief usage 2026-04-03 to 2026-04-08. Amendment 2.7 restored Mode A (sacred 2026-01-01 boundary). 124 validated_setups rows grandfathered as research-provisional. See `04_DECISION_LOG.md` §1 for full reasoning.

---

## What to DO when user proposes something similar

1. **Match the proposal against this file.** Search for the feature name, strategy class, or architectural pattern.
2. **If match found:** cite the dead entry + original postmortem. Ask user: "How is this different from [dead attempt]?"
3. **If no match but conceptually adjacent** (e.g., "new composite of volume + volatility" → H2 Path C):
   - Flag the adjacency
   - Point to the nearest postmortem
   - Suggest pre-registration BEFORE scanning
4. **If clearly new:** proceed with pre-registration template from `docs/institutional/hypothesis_registry_template.md`.

The goal is NOT to veto new work. It's to save the user from rediscovering their own dead ends.
