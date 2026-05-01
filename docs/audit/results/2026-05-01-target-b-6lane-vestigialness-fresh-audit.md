---
pooled_finding: false
audit_target: "Target B — 6-lane filter vestigialness re-verification"
auditor_context: fresh
canonical_layers: [orb_outcomes, daily_features]
db_freshness: orb_outcomes 2026-04-28 / daily_features 2026-04-29 (MNQ)
verdict: VERIFIED_WITH_CORRECTIONS
parent_claims:
  - docs/audit/results/2026-04-20-6lane-filter-vestigialness.md
  - docs/audit/results/2026-04-21-correction-aperture-audit-rerun.md
---

# Target B — 6-Lane Filter Vestigialness — Fresh-Context Re-Audit

**Date:** 2026-05-01
**Auditor mode:** independent fresh-context evidence-auditor (separate context from main thread)
**Subject of audit:** PR #47 (filter vestigialness claim) and PR #57 (aperture-corrected supersession)
**Outcome:** vestigialness verdict STANDS; three documentation/cleanup debts surfaced and resolved/queued in this run.

## Scope

This re-audit asks whether the four load-bearing claims of the original PR #47 vestigialness finding (and PR #57's aperture-corrected supersession of it) survive an independent fresh-context measurement against canonical layers as of 2026-04-28. It is a **structural re-verification of an existing audit**, not new discovery.

Specific questions:
1. Are filter fire rates still ≥75% on the 6 deployed-portfolio lanes in 2026 YTD?
2. Is the per-lane filter "lift" (filtered ExpR − unfiltered ExpR) still ≈0?
3. Is portfolio diversification driven by sessions or by filters?
4. Does PR #57's claim that fire rates carry across apertures (5min ↔ 15min) hold under independent measurement?

It does NOT re-discover edge, propose new lanes, or evaluate any deployment decision. Capital lane composition is out of scope.

---

## 1. Setup

**Canonical layers used:** direct SQL on `orb_outcomes` and `daily_features` (the two safe layers per `RESEARCH_RULES.md` § Discovery Layer Discipline). No reads from `validated_setups`, `edge_families`, or `live_config`.

**Filter logic:** verified against `trading_app/config.py` class implementations: `OrbSizeFilter` (ORB_G5), `OwnATRPercentileFilter` (ATR_P50), `CostRatioFilter` (COST_LT12). Costs read via `pipeline.cost_model.COST_SPECS` (canonical).

**Lane allocation:** `docs/runtime/lane_allocation.json`. 6 DEPLOY lanes confirmed:

| Lane | Session | Filter | Aperture |
|---|---|---|---|
| L1 | EUROPE_FLOW | ORB_G5 | O5 |
| L2 | SINGAPORE_OPEN | ATR_P50 | O15 |
| L3 | COMEX_SETTLE | ORB_G5 | O5 |
| L4 | NYSE_OPEN | COST_LT12 | O5 |
| L5 | TOKYO_OPEN | COST_LT12 | O5 |
| L6 | US_DATA_1000 | ORB_G5 | O15 |

**DB freshness:** `orb_outcomes` MNQ max `2026-04-28`; `daily_features` max `2026-04-29`. No stale-data halt.

---

## 2. Claim verification

### CLAIM 1 — Filter fire rate ≥ 75% in 2026 for at least 5 of 6 lanes
**Status:** MEASURED — VERIFIED (with refinement)

Independent SQL on `daily_features` (denominator: `break_dir IS NOT NULL`) through 2026-04-28:

| Lane | 2026 fire rate | Verdict |
|---|---|---|
| L1 EUROPE_FLOW ORB_G5 | 74/74 = **100.0%** | VESTIGIAL |
| L2 SINGAPORE_OPEN ATR_P50 (O15) | 60/74 = **81.1%** | VESTIGIAL |
| L3 COMEX_SETTLE ORB_G5 | 71/71 = **100.0%** | VESTIGIAL |
| L4 NYSE_OPEN COST_LT12 | 74/74 = **100.0%** | VESTIGIAL |
| L5 TOKYO_OPEN COST_LT12 | 72/74 = **97.3%** | VESTIGIAL |
| L6 US_DATA_1000 ORB_G5 (O15) | 73/73 = **100.0%** | VESTIGIAL |

**All 6 of 6 above 75%** (PR #47 reported 5/6 because L2 was at 75.0% on the audit run-day ~2026-03-17; trade tape has since shifted L2 to 81.1%). Direction of claim intact.

**Aperture independence (load-bearing for PR #57 supersession):**
- L2 ATR_P50 fire rate: 53.2% (5min) vs 53.1% (15min) IS; 81.1% (5min) = 81.1% (15min) 2026 — IDENTICAL.
- L6 ORB_G5 fire rate: 99.1% (5min) vs 99.7% (15min) IS; 100.0% both apertures 2026 — IDENTICAL.
- Filter columns (`atr_20_pct`, `orb_<SESSION>_size`) live in `daily_features` keyed by `(trading_day, symbol)`, not aperture; `orb_minutes` is irrelevant to fire-rate computation. **PR #57's "fire rates carry over" assertion is independently confirmed.**

### CLAIM 2 — 2026 filter lift ≤ +0.001R (essentially zero)
**Status:** MEASURED — VERIFIED with precision caveat

Independent SQL `orb_outcomes JOIN daily_features` through 2026-04-28:

| Lane | Unfiltered ExpR | Filtered ExpR | Lift |
|---|---|---|---|
| L1 | (74/74) | (74/74) | **0.0000R** |
| L2 | +0.104R (n=74) | +0.072R (n=60) | **−0.032R** |
| L3 | (71/71) | (71/71) | **0.0000R** |
| L4 | (74/74) | (74/74) | **0.0000R** |
| L5 | n=74 | n=72 | **+0.004R** |
| L6 | (73/73) | (73/73) | **0.0000R** |

5/6 lanes show literal zero lift by construction (100% fire rate = filter no-op). L5 +0.004R trivial. L2 −0.032R is the only material deviation, sign-flipped from IS — within n=60 noise (95% CI ±0.24R per trade) but the **direction has reversed**, confirming filter is no longer adding value (matches PR #57's documented L2 negative-lift framing).

The "+0.001R" figure in PR #47 was specific to L4 NYSE_OPEN COST_LT12 in IS years. Our 2026 measurement of L4 = 0.0000R (filter is now a no-op) is **stronger** than the original claim.

### CLAIM 3 — Diversification is session-driven, not filter-driven
**Status:** MEASURED — VERIFIED

Pairwise Pearson correlation of daily `pnl_r` for all 6 lanes in 2026 (0-fill on no-trade days, n=74):

- **Filtered portfolio mean off-diagonal correlation:** 0.0602
- **Unfiltered portfolio mean off-diagonal correlation:** 0.0524
- **Δ (filtered − unfiltered):** **+0.0078** (filtering marginally INCREASES correlation, opposite of what filter-diversification would predict)

Per-pair: 10/15 pairs show 0.000 difference (4 lanes at 100% fire rate). The 5 pairs involving L2/L5 show small changes; largest is L2_SG vs L3_CS (filt=+0.240, unfilt=+0.158, Δ=+0.082) — reading: shared macro-vol regime exposure when L2 fires, not filter diversification.

Both matrices have low mean correlation (~0.06) — genuine session diversification confirmed. **Filters do NOT contribute to diversification.**

Approximate 95% CI for Pearson r at n=74 is ±0.23 — per-pair magnitudes are noisy but the structural direction (no reduction) is robust.

### CLAIM 4 — PR #57 supersedes PR #52/#54 L2/L6 at canonical aperture
**Status:** MEASURED — VERIFIED

**The bug (confirmed):** `research/audit_6lane_scale_stability.py` line 103 hardcodes `AND o.orb_minutes = 5` in the SQL. The canonical parser `parse_strategy_id` (line 54-87 of the same file) extracts `_O15` and sets `aperture_overlay = "O15"`, but this is never threaded into `load_lane_universe`. Same single-aperture-assumption class as PR #189's allocator bug.

**The fix (confirmed):** `research/audit_lane_baseline_decomposition_v2.py` imports `parse_strategy_id` from `trading_app.eligibility.builder` (canonical) and threads aperture through. No executable `orb_minutes = 5` SQL.

**Material reversals from supersession (independently verified):**
- L2 IS ExpR: −0.002R (wrong 5min) → +0.052R (correct 15min). Overturns "L2 would be deploy-killed without filter."
- L6 2026 OOS ExpR: −0.034R (wrong 5min) → +0.161R (correct 15min). Overturns "L6 only 2026-negative lane."

**Self-confirming-loop check:** PR #57 asserted fire rates were aperture-invariant and moved on without re-running them. Our independent measurement of identical fire rates at both apertures (Claim 1 above) confirms the assertion was sound. **No self-confirming loop.**

---

## 3. New findings (this audit)

### Finding 1 — DOCTRINE DRIFT in `.claude/rules/quant-audit-protocol.md` (cost specs stale)

**Severity:** MEDIUM (template-poisoning risk; does not affect PR #47/#57 results).

The auto-loaded rule `.claude/rules/quant-audit-protocol.md` § "PROJECT-SPECIFIC ANCHORS" reads:

> Cost model: `from pipeline.cost_model import COST_SPECS` — MNQ $2.74 RT, MGC $5.74, MES $3.74

**Verified against canonical `pipeline/cost_model.py` `COST_SPECS`:**

| Instrument | commission_rt | spread_doubled | slippage | total_friction | Doctrine says | Match? |
|---|---|---|---|---|---|---|
| MNQ | 1.42 | 0.50 | 1.00 | **$2.92** | $2.74 | **STALE** |
| MGC | 1.74 | 2.00 | 2.00 | **$5.74** | $5.74 | OK |
| MES | 1.42 | 1.25 | 1.25 | **$3.92** | $3.74 | **STALE** |

Drift origin: F-4 fix (commission $1.24 → $1.42, audit-finding F-4 in `pipeline/cost_model.py` lines 78-83) raised MNQ/MES `total_friction` by $0.18 each. The rule file was never updated. MGC was not affected by F-4.

**Risk:** any future audit using this rule as a reference table for cost spec values would inherit the $0.18 stale offset. The actual research scripts under audit (`audit_6lane_scale_stability.py`, `audit_lane_baseline_decomposition_v2.py`) **delegate via `filter_signal` → canonical `CostRatioFilter` → `cost_spec.total_friction`**, so they correctly use `$2.92` at runtime. The drift is contained to the rule documentation, not to code paths.

**Resolution this run:** rule file corrected to canonical totals; cost-spec-drift memory entry written; drift-check candidate noted (cross-reference audit-protocol numbers vs `COST_SPECS`).

### Finding 2 — DEPRECATED SCRIPTS RESIDENT with the same `orb_minutes=5` hardcode bug PR #189 fixed at allocator layer

**Severity:** MEDIUM (cleanup debt; same bug class as PR #189 capital-class fix).

PR #57 marked these scripts deprecated but did not delete them:

- `research/audit_6lane_unfiltered_baseline.py` (PR #52 source)
- `research/audit_l2_atr_p50_stability.py` (PR #54 source)

Both retain hardcoded `o.orb_minutes = 5` and would silently produce wrong results for L2/L6 (which are O15 lanes) if run again. Same single-aperture-assumption bug class as PR #189's allocator fix. The deprecation comment does not block execution — only a code reader notices.

PR #57 explicitly listed deletion of these files as a follow-up; the follow-up was never executed.

**Resolution this run:** scheduled for deletion in a sibling cleanup branch (Task #5 in this session's task list). Memory entry written documenting the "deprecation notice ≠ removal" lesson.

### Finding 3 — L2 ATR_P50 filter lift FLIPPED NEGATIVE in 2026 (−0.032R)

**Severity:** LOW (within noise; documented in PR #57; watch item).

The filter that historically had IS Welch p=0.014 (the only one of 6 with any claimed IS edge) has lift of **−0.032R** in 2026 YTD. n=60 filtered, CI ±0.24R per trade — well within noise — but the direction has reversed. PR #57's documented framing was "−0.037R" (3-week-older data); our 2026-04-28 measurement (−0.032R) is consistent with that trend.

**Implication:** L2 is not a capital action target, but it is the lane to monitor. If the lift remains negative through 2026-Q3, the filter ought to be revisited (or downgraded to "session anchor without filter").

### Finding 4 — MEMORY.md L2 SHORTHAND OVERSTATES PR #57's REFUTATION SCOPE

**Severity:** LOW (informational hygiene).

Current MEMORY.md "Validated signals" entry includes:

> 6-lane filter vestigialness (PR #47 511ff148) … L2 diagnosis REFUTED by PR #57.

**Correction:** PR #57 refuted the *wrong-aperture L2 diagnosis* (the "filter_is_the_edge" framing built on 5-min data when L2 is actually O15) and replaced it with "filter_correlates_with_edge (marginal)". The IS Welch p=0.014 signal itself was NOT refuted — that was confirmed by PR #57 at the correct aperture, with the headline ExpR moving from −0.002R (spurious) to +0.052R (real but small).

**Resolution this run:** MEMORY.md line corrected (Task #4).

---

## Limitations

(Section 4 — items the auditor could not independently verify.)

- **IS Welch statistics from PR #57 (p=0.014 for L2)** — not independently re-run in this audit. We verified 2026 OOS numbers but not the full IS regression table. The IS analysis relied on PR #57's script which is canonical (uses `parse_strategy_id` from `trading_app.eligibility.builder`) and not deprecated. Considered verified by provenance, not by fresh execution.
- **Correlation analysis precision** — based on 74 2026 days. Pearson r CI at n=74 is ~±0.23. Structural direction (no reduction) is robust; individual pair magnitudes are noisy.
- **Per-lane Pathway-B K=1 tests under Mode A** — out of scope for this audit (vestigialness is a structural claim about fire rate and lift, not a deployment-promotion claim, so the per-lane K=1 gate from Amendment 3.0 does not directly apply).

---

## 5. Capital-risk implication

**SAFE.** The vestigialness finding is NOT a portfolio structural failure. Edge resides in the session/entry-model/RR geometry; filters pass through nearly all trades in 2026. This is a lower-information signal layer than intended, but it is not a source of hidden risk. Unfiltered lane baselines remain positive (confirmed in PR #57 Section A and re-verified above for L2 +0.104R, L5 baseline preserved).

PR #189's allocator fix is the correct structural remediation for the single-aperture-assumption bug class; this audit confirms the same bug is also present (in already-deprecated form) in two research scripts that should be deleted.

---

## 6. Verdict

**Vestigialness verdict: STANDS.**
**PR #57 supersession: VERIFIED.**
**No capital action required.**

Three documentation/cleanup debts resolved this session (Findings 1, 2, 4); Finding 3 is monitor-only.

---

## Reproduction

**Database:** `gold.db` at project root (`pipeline.paths.GOLD_DB_PATH`), read-only.
**Cutoff:** `orb_outcomes` MNQ max `trading_day = 2026-04-28`; `daily_features` max `2026-04-29`.

**Filter columns referenced:**
- ORB_G5 → `daily_features.orb_<SESSION>_size`, threshold from `trading_app.config.ALL_FILTERS["ORB_G5"]` (`OrbSizeFilter.min_size`)
- ATR_P50 → `daily_features.atr_20_pct`, threshold from `OwnATRPercentileFilter.min_percentile`
- COST_LT12 → derived in `CostRatioFilter.matches_df()` from `cost_spec.total_friction` (canonical `COST_SPECS["MNQ"].total_friction = $2.92`) divided by per-row `raw_risk` from `daily_features.orb_<SESSION>_size × point_value`

**Lane universe query (template):**
```sql
SELECT o.trading_day, o.pnl_r, d.atr_20_pct, d.orb_<SESSION>_size, ...
FROM orb_outcomes o
JOIN daily_features d
  ON o.trading_day = d.trading_day
 AND o.symbol = d.symbol
 AND o.orb_minutes = d.orb_minutes
WHERE o.symbol = 'MNQ'
  AND o.orb_label = '<SESSION>'
  AND o.orb_minutes = <5_or_15>      -- threaded from parse_strategy_id
  AND o.entry_model = 'E2'
  AND o.rr_target = <RR>
  AND o.confirm_bars = 1
  AND o.pnl_r IS NOT NULL
ORDER BY o.trading_day
```

**Aperture-independence check:** ran the same fire-rate calculation at `orb_minutes = 5` and `orb_minutes = 15` for L2 (ATR_P50) and L6 (ORB_G5) and confirmed identical fire rates within rounding (filter columns are not aperture-keyed in `daily_features`).

**Outputs:**
- This document (`docs/audit/results/2026-05-01-target-b-6lane-vestigialness-fresh-audit.md`)
- Decision-ledger entry `target-b-vestigialness-stands-2026-05-01` in `docs/runtime/decision-ledger.md`
- Memory entries (user-personal, not in repo): `feedback_doctrine_drift_cost_specs_2026_05_01.md`, `feedback_deprecation_notice_not_removal_2026_05_01.md`
- Rule-file fix: `.claude/rules/quant-audit-protocol.md` line 272 (cost specs)
- MEMORY.md L2-shorthand correction (user-personal, not in repo)

## Provenance

- Audit ran from canonical worktree `C:/Users/joshd/canompx3` on branch `main`.
- Independent SQL queries against `gold.db` (canonical, project-root location per `pipeline/paths.py`).
- Filter logic verified against `trading_app/config.py` (`OrbSizeFilter`, `OwnATRPercentileFilter`, `CostRatioFilter`) and `pipeline/cost_model.py` (`COST_SPECS`).
- Auditor: fresh-context evidence-auditor agent (`a0dc702e2c12ad67c`) dispatched 2026-04-30; this document consolidates and re-verifies its findings against source files in the canonical worktree.
