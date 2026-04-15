# Pre-Registration — Prior-Day Zone / Positional / Gap-Categorical Features on ORB Outcomes

**Date registered:** 2026-04-15
**Author:** Claude Code (session HTF)
**Protocol:** B — feature-space enumeration (no pre-existing literature extract for the novel slice)
**Status:** PRE-REGISTERED v4 (two reviewer passes applied; grade A−; no execution yet)

---

## 1. Context & prior-art disposition

`docs/specs/presession-volatility-filters.md` shows prior-day **magnitude** features are extensively tested:
- `prev_day_range/atr_20` (PDR) — DEPLOYED on LONDON_METALS, EUROPE_FLOW × {MGC, MNQ}; KILLED on NYSE_OPEN × MNQ (OOS sign flip, BH FAIL).
- `abs(gap_open_points)/atr_20` — DEPLOYED on CME_REOPEN × MGC only.
- `took_pdh/took_pdl × NYSE_CLOSE × MES` — KILLED (T6 FAIL).
- `took_pdh × US_DATA_1000 × {MES, MNQ}` — QUARANTINED (WFE>1.89 suspect-leakage).
- `overnight_range/atr × NYSE_CLOSE × MES` — KILLED.

`docs/RESEARCH_ARCHIVE.md:567` registers Crabel contraction/expansion session-level as NO-GO (ORB-size proxy confound, r=0.35-0.86).

Classification verdict: **PARTIAL**. Re-testing any of the above is prohibited; the uncovered slice is the zone / positional / gap-categorical family defined in §3.

## 2. Mechanism category

**FILTER** — the signal changes the *selection probability* of taking an ORB breakout given its geometric relationship to prior-day levels.

No pre-existing `docs/institutional/literature/` extract supports the specific mechanism. Under Protocol B, Chordia threshold = **t ≥ 3.79** (`pre_registered_criteria.md:112`). Writing a literature extract after 2026-04-15 does **not** reopen the t ≥ 3.00 pathway (banned, §13).

## 3. Features (8 binary signals)

Scaling denominator is `atr_20` (verified prior-day-only at `pipeline/build_daily_features.py:1151-1156` — `true_ranges[max(0, i-20) : i]` slice excludes today). Session ORB midpoint = `(orb_high + orb_low) / 2` of the session's ORB window — fully computed at ORB window close, pre-entry-signal.

| # | Name | Definition | Repaints? | Pre-trade? |
|---|------|------------|-----------|-----------|
| F1 | `NEAR_PDH` | `abs(orb_mid - prev_day_high) / atr_20 < 0.30` | No | Yes |
| F2 | `NEAR_PDL` | `abs(orb_mid - prev_day_low) / atr_20 < 0.30` | No | Yes |
| F3 | `NEAR_PIVOT` | `abs(orb_mid - (prev_day_high + prev_day_low + prev_day_close)/3) / atr_20 < 0.30` | No | Yes |
| F4 | `ABOVE_PDH` | `orb_mid > prev_day_high` | No | Yes |
| F5 | `BELOW_PDL` | `orb_mid < prev_day_low` | No | Yes |
| F6 | `INSIDE_PDR` | `prev_day_low < orb_mid < prev_day_high` | No | Yes |
| F7 | `GAP_UP` | `gap_type == 'gap_up'` | No | Yes (today_open at 09:00 Brisbane precedes all active-session ORBs) |
| F8 | `GAP_DOWN` | `gap_type == 'gap_down'` | No | Yes |

### 3.1 Removed feature (v2 → v3)

`ALIGNED_PREV_DIR` (was F7 in v2) **removed** as look-ahead: it used `break_dir` which is unknown at E2 stop-market signal-generation time (direction learned at fill). Per reviewer finding #9. Not reintroducible as a `prev_day_direction`-only filter within this pre-reg — that requires a new hypothesis.

### 3.2 θ choice disclosure

θ=0.30 is an **arbitrary prior** (no literature extract supports a specific value). Sensitivity at θ ∈ {0.15, 0.50} is **mandatory** as knife-edge defense (§6 θ-monotonicity gate).

### 3.3 Correlation disclosure

- F1 NEAR_PDH vs F2 NEAR_PDL: inherently negatively correlated (price near one level is far from the other). Jaccard reported pairwise; do not aggregate signals.
- F7/F8 vs deployed `GapNormFilter`: different denominators (`prev_day_range` vs `atr_20`) and semantic (directional vs magnitude). Jaccard gate (§6) will quantify.

## 4. Scope

### 4.1 Phase 1 (this run)

- **Instruments:** `MNQ`, `MES` only. MGC excluded by **data-horizon constraint** (3.9 yr backfilled clean data; Bailey relaxed E=1.2 bound N ≤ 17 insufficient). MGC mechanism is not excluded; only data is insufficient.
- **Sessions per instrument:** `CME_PRECLOSE`, `US_DATA_1000`, `NYSE_OPEN` (top-3 by unfiltered baseline ExpR per `STRATEGY_BLUEPRINT.md:210-213`).
  - **Selection disclosure:** session selection is prior-data-informed (baseline ranking). Filter-EFFECT test and absolute-ExpR test are *designed* to be separable — session fixed effect `C(session)` absorbs baseline level, and the `feature × C(session)` interaction tests whether the filter effect varies by session. Separability is not asserted a priori; the interaction term is the empirical test. If selection induced bias (e.g., high-ExpR sessions happen to be higher-vol where filters work better), the interaction term will flag it.
- **Apertures:** O5 only (primary per `STRATEGY_BLUEPRINT.md:203`).
- **RR targets:** `{1.0, 1.5}` (RR2.0 weakening per L144-146).
- **Direction:** **SYMMETRIC** (on-signal vs off-signal ExpR delta). **Disclosed Type II risk:** balanced asymmetric effects (e.g., +0.2R long / -0.2R short) net to 0 mean-delta and will be missed. Phase 2 direction split gated on Phase 1 survivor.
- **Entry model:** E2 (stop-market per `STRATEGY_BLUEPRINT.md:130`). E0 banned.
- **Holdout:**
  - IS = `trading_day < HOLDOUT_SACRED_FROM` (2026-01-01) — imported from `trading_app.holdout_policy`.
  - OOS = `2026-01-01 ≤ trading_day < HOLDOUT_GRANDFATHER_CUTOFF` (2026-04-08) — ~68 trading days per instrument. Data on/after 2026-04-08 is forward-sacred and **excluded entirely** from this study.

### 4.2 Phase 2 (pre-registered; contingent on Phase 1 survivor)

If any Phase 1 cell passes all binding gates, Phase 2 runs:
- The **same surviving (feature × session × RR)** on MGC using all clean MGC data (backfilled to 2022-06-13 where available).
- Budget: **N ≤ 7** strict Bailey E=1.0 for MGC 3.9 yr (`criteria.md:86`).
- Direction split (LONG / SHORT) on that surviving cell on MNQ/MES + MGC → at most 2 additional cells per survivor.

Phase 2 gates: same as Phase 1 §6, with MinBTL recomputed at the Phase 2 N.

### 4.3 Phase 1 expected-outcome disclosure

~68 OOS trading days × 3 sessions × 2 RR = 408 OOS observations per instrument, spread across 16 (feature × RR × session × instrument) cells per instrument. After fire-rate filtering, expected N_OOS per cell = 20–40. **Most cells are expected to land in DEFER** (N_OOS < 30), not GO or NO-GO. DEFER is resolved at a single locked evaluation date (§10).

## 5. K accounting (three distinct counts, three distinct roles)

Grounded in `bailey_et_al_2013_pseudo_mathematics.md:43` (Eq.6) and line 49 verbatim (*"no more than 45 independent model configurations should be tried"*): MinBTL binds on **independent configurations**, not total tests. Cross-instrument replication is coherence (§9 FM#1), not search-budget doubling.

| Name | Definition | Value | Role | Authority |
|------|------------|-------|------|-----------|
| N_MinBTL | Independent strategy definitions = features × sessions × apertures × RR × direction | **48** (8×3×1×2×1) | Bailey Eq.6 data-sufficiency | `bailey_et_al_2013_pseudo_mathematics.md:43` |
| K_local | p-values computed = N_MinBTL × instruments | **96** (48×2) | BH-FDR local correction | `pre_registered_criteria.md:100` |
| K_global | K_local + prior project-search load | **639** (96 + 61 validated_setups + 57 prior hypothesis files + 425 filter-family audit per `STRATEGY_BLUEPRINT.md:14`) | Cross-family BH-FDR | — |

Prior-search counts (61, 57, 425) are live-queried at pre-registration commit time (Volatile Data Rule, `CLAUDE.md:70-71`) and then **frozen** in §5 and §16 as the exact values used for this run's Bonferroni/BH arithmetic. If a future re-audit (§11) re-queries and finds different counts, that triggers a new pre-registration — the current run's gates remain bound to the frozen values shown here.

**Bailey Eq.6** (N=48 against 6.65 clean years MNQ/MES, calendar-shared):

| E[max_N] | MinBTL required (yr) | Pass at 6.65 yr |
|----------|----------------------|----------------|
| 1.0 (strict)                        | 7.74 | **FAIL** |
| 1.18 (break-even)                   | 6.65 | marginal |
| 1.2 (relaxed per criteria.md:84)    | 5.38 | **PASS** |
| 1.5 (professional)                  | 3.44 | PASS |

**Operational ceiling** (`pre_registered_criteria.md:90`): 48 ≪ 300 → PASS.

**Disclosure:** Study operates at relaxed Bailey E[max_N] ≥ 1.2, NOT strict E=1.0. Margin is `6.65 - 5.38 = 1.27 yr` against 6.65 calendar years available — tight, not comfortable. MNQ and MES share calendar time (same 1,951 pre-holdout days), so the cross-instrument correlation reduces effective degrees of freedom further. Allowed by criteria.md §2 line 84 with downstream gates (§6) as compensating defenses. Chordia t ≥ 3.79 (not 3.00), 5-era stability, holdout `≥ 0.4 × IS`, and partial-regression interaction are the four defenses specifically hardened for this thin margin.

Framework-integrity controls (§7) are tested **outside** the K=96 budget. Honest total tests = K_local + 3 controls = 99; controls do not consume Bonferroni/BH-FDR budget because they're not candidates for adoption (reviewer finding #11 addressed).

## 6. Gates (all binding unless marked diagnostic)

| Gate | Threshold | Source |
|------|-----------|--------|
| N per cell | ≥ 30 exploratory; cells with N < 100 flagged non-deployable | `pre_registered_criteria.md:152` |
| Bonferroni-local | `p < 0.05 / 96 = 5.21e-4` | standard |
| **BH-FDR local** | **`q < 0.05`** at K=96 | `pre_registered_criteria.md:100, 226, 415` (locked 0.05, NOT 0.10) |
| **BH-FDR global** | **`q < 0.05`** at K=639 | same |
| Chordia t | `t ≥ 3.79` | `pre_registered_criteria.md:112` (no literature credit under Protocol B) |
| Era stability | `ExpR ≥ -0.05` in each of 5 eras [2015-19, 2020-22, 2023, 2024-25, 2026] with N ≥ 50 | `pre_registered_criteria.md:170` |
| Holdout direction | OOS direction matches IS direction | `pre_registered_criteria.md:162` |
| Holdout effect | `OOS_ExpR ≥ 0.40 × IS_ExpR` | `pre_registered_criteria.md:162` |
| Holdout N | `N_OOS ≥ 30`; else DEFER | `pre_registered_criteria.md:152` |
| Effect floor (absolute) | NET-of-cost ExpR uplift vs control `≥ 0.10 R/trade` | conservative micro-edge floor; no criteria.md lock |
| Jaccard ≤ 0.30 | daily fire-pattern vs every `BASE_GRID_FILTERS` element, every deployed lane filter, AND the 8 features pairwise in this study | MEMORY.md wave4 precedent 0.19-0.40 |
| Jaccard 0.30-0.40 | weakly correlated → partial-regression pass required | same |
| Jaccard > 0.40 | redundant → auto NO-GO | same |
| Partial regression (mean) | `outcome_R ~ feature + atr_20 + orb_size + C(session) + C(instrument) + feature:C(session)` with OLS + cluster-robust SE by `trading_day`; feature coefficient AND feature:C(session) interaction jointly significant at Bonferroni-local | reviewer findings #5, #10 |
| Partial regression (binary) | logistic regression on win/loss, same covariates; feature coefficient significant at Bonferroni-local | bounded-outcome bimodality defense |
| **θ monotonicity** | All of: (a) primary θ=0.30 delta lies between the adjacent deltas inclusive (`min(θ=0.15, θ=0.50) ≤ θ=0.30 ≤ max(θ=0.15, θ=0.50)`) — equivalent to "primary is not the strict extremum" = "the three-point sequence is monotone"; (b) all three deltas share the same sign; (c) adjacent-θ deltas have absolute magnitude ≥ 25% of primary. Any violation → auto-reject. | knife-edge defense |
| DSR | > 0.95 — **diagnostic only** | `pre_registered_criteria.md:136` implementation gap |
| Power | N_required at effect-floor with power ≥ 0.80 — **diagnostic only** | Harvey-Liu 2015 |

## 7. Framework-integrity controls (outside K=96 budget; three independent checks)

- **C1 — Destruction control (negative):** within-era shuffle of F1's feature column. Preserves year effects; DOES NOT preserve weekday structure (reviewer finding #5 disclosed). Must fail every binding gate.
- **C2 — Known-null feature:** `seeded_rng(20260415).binomial(1, 0.5, len(df))` — pure noise keyed per `(symbol, trading_day, session)` to prevent within-day clustering from masking framework leakage. Expected: zero effect. Must fail every binding gate. A pass indicates framework bug (t-inflation via missing CTE dedup, triple-join, or cost-not-applied).
- **C3 — Known-positive sanity:** `VWAPBreakDirectionFilter` (deployed, in `validated_setups`) re-tested on the same scope. Must PASS binding gates. A fail indicates framework bug (signal suppression).

Framework is validated only if **C1 fails AND C2 fails AND C3 passes**. Any deviation → abandon run, file pipeline bug.

## 8. Unit of analysis & data hygiene

- Per-trade observations (one per session × aperture per trading day with valid ORB break).
- Clustered SE by `trading_day` (multi-session intra-day correlation).
- Block bootstrap B=10,000 with block size `⌈T^{1/3}⌉` (Politis-Romano stationary bootstrap), T = trading days in cell.

### 8.1 CTE dedup guard (reviewer finding #2 — MANDATORY)

Every SQL CTE or subquery reading `daily_features` for a non-ORB-specific column (prev_day_*, gap_type, atr_20, all features in §3) MUST contain `WHERE d.orb_minutes = 5`. Omission triples row count and inflates t-stats by √3 — difference between pass and fail at t ≥ 3.79. Reference: `.claude/rules/daily-features-joins.md` § CTE Guard. Precedent bug: commit `94546ccf` (IBS t=6.78 → 3.89 after fix).

**§13 ban:** failing to apply `orb_minutes = 5` CTE dedup is an auto-reject of the run.

### 8.2 Cost-net semantic assertion (reviewer finding #6)

Script must explicitly verify that the outcome R-value is cost-net, not print a warning. Assertion:
```python
# Pick one trade; recompute cost_$ from COST_SPECS[symbol];
# convert to cost_R using stop_distance; assert field ≈ expected_net.
sample = df.iloc[0]
cost_R = compute_cost_R(sample['symbol'], sample['stop_distance'])
expected_net = sample['gross_R'] - cost_R
assert abs(sample['outcome_R'] - expected_net) < 0.01, \
    f"outcome_R not cost-net: {sample['outcome_R']} vs expected {expected_net}"
```
Fail-closed, no soft-warn. Integrity-guardian rule 7: "Never trust metadata."

## 9. Failure modes (enumerated, any ≥ 1 ⇒ NO-GO for that cell)

1. Direction of ExpR-delta between MNQ and MES for same feature × session **fails cross-instrument coherence**, defined as EITHER (a) opposite signs AND `|delta_MNQ - delta_MES| ≥ 0.05 R` (flip), OR (b) `min(|delta_MNQ|, |delta_MES|) < 0.05 R` (= half the effect floor — insufficient coherence: one instrument shows ~null while the other shows effect).
2. Era failure — any era with N ≥ 50 shows `ExpR < -0.05`.
3. θ-monotonicity violation per §6 rule.
4. Jaccard > 0.40 with any deployed filter or BASE_GRID element.
5. C1 destruction control passes any binding gate → framework broken.
6. C2 known-null feature passes any binding gate → framework broken.
7. C3 known-positive fails any binding gate → framework broken.
8. Partial regression (mean) feature coefficient OR feature×session interaction not jointly significant at Bonferroni-local.
9. Partial regression (binary) feature coefficient not significant at Bonferroni-local.
10. Holdout direction flip OR `OOS_ExpR < 0.40 × IS_ExpR` (given sufficient N_OOS).
11. N_OOS < 30 at evaluation date → DEFER resolves to NO-GO per §10.

## 10. Evaluation schedule (reviewer finding #7 — optional stopping defense)

**Single locked evaluation date: 2026-10-15.** No peeking before this date. No interim adjudication.

On 2026-10-15:
- For each cell with `N_OOS ≥ 30`: evaluate all binding gates. GO if all pass; NO-GO otherwise.
- For each cell with `N_OOS < 30`: **auto-NO-GO** (insufficient forward power).

No re-tuning on OOS. No re-running with different seeds. No cell-by-cell early peeks. Pocock-boundary correction not required because there is no interim look.

## 11. Re-audit triggers

- Pipeline rebuild (Phase 3c-style).
- Schema change on any cited `daily_features` column.
- Entry-model swap (E2 → other).
- `COST_SPECS` change for MNQ or MES.
- New instrument activated in `ACTIVE_ORB_INSTRUMENTS`.
- Any addition to `BASE_GRID_FILTERS` or `VALIDATED_SETUPS` that overlaps the fire-pattern of a survivor → redundancy re-check.

## 12. Reproducibility

- Seed: `numpy.random.default_rng(20260415)`.
- DB: `duckdb.connect(str(pipeline.paths.GOLD_DB_PATH), read_only=True)`.
- Script: `research/prior_day_features_orb.py` — **SHA frozen on first run** (reviewer finding #14); any post-run edit re-opens the registration.
- Costs: pipeline's cost-net outcome field, assertion-verified per §8.2.
- Era boundaries literal from §6 gate.

## 13. Banned actions (any violation auto-rejects the run)

- Direction flip after seeing result.
- θ re-tuning after seeing result.
- Cell addition after seeing result (including Phase 2 outside exact surviving cell).
- Protocol switch (A↔B) after seeing any result.
- **Retroactive Protocol reclassification** — writing a `literature/` extract post-registration does NOT reopen the t ≥ 3.00 pathway (reviewer finding #13).
- Instrument scope expansion to MGC without a Phase 1 survivor.
- Session scope expansion beyond the 3 declared sessions.
- Re-running with different seed and picking best.
- DSR or Power as binding gates (diagnostic only).
- Declaring GO with `N_OOS < 30`.
- **Failing to apply `orb_minutes = 5` CTE dedup on any `daily_features` read** (reviewer finding #2, §8.1).
- **Accepting outcome_R without the cost-net assertion in §8.2** passing.
- Peeking at OOS results before 2026-10-15.
- Editing `research/prior_day_features_orb.py` after the first run's SHA commit without a new pre-registration.
- Using `double_break`, `took_pdh_before_1000`, `took_pdl_before_1000`, `overnight_range_pct`, `break_dir`, `break_ts`, `break_delay_min` as feature inputs (look-ahead / previously-killed).

## 14. Commit trail

On completion:
1. Commit this pre-registration file before any test runs.
2. Commit research script (`research/prior_day_features_orb.py`); SHA frozen post-first-run.
3. Commit results table as `docs/audit/results/2026-04-15-prior-day-zone-positional-features-orb-results.md`.
4. Commit independent code-reviewer log (v2 → v3 audit + v3 → results audit).
5. If GO on any cell: write Protocol-A Phase 2 hypothesis with literature extract requirement + direction split + MGC arm.
6. If NO-GO: add `docs/STRATEGY_BLUEPRINT.md` § 5 entry.
7. Regardless of verdict: companion NO-GO for multi-timeframe indicator-on-indicator chaining at full scope (Option 3 from session HTF brainstorm) — K-explosion ≥ 5,832 trials vs Bailey N ≤ 120 at E=1.2. Reopen: pre-registered single chain with literature mechanism AND K ≤ 120.

## 15. Git SHA at registration

Populated post-commit by drift check #94.

## 16. Self-audit history

- **v1 (2026-04-15 first draft):** initial write.
- **v2 (2026-04-15 self-audit):** K tri-split, θ sensitivity, MGC Phase 2, direction Type II, cost-net rough note, positive control, DEFER soft.
- **v3 (2026-04-15 independent code-reviewer applied, grade C+ → A−):** BH-FDR corrected to q<0.05 (was 0.10 — threshold relaxation); CTE dedup hard-banned in §13; K_global citation fixed to STRATEGY_BLUEPRINT.md:14; C2 known-null added (framework-integrity not circular); C3 VWAP demoted to sanity; cost-net asserted not just printed; DEFER locked to single eval date 2026-10-15; θ-monotonicity tightened; F7 ALIGNED_PREV_DIR dropped (look-ahead via break_dir); partial regression adds `feature:C(session)` interaction; FM#1 flip threshold defined; retroactive-protocol-reclassification banned; script SHA frozen on first run.
- **v4 (2026-04-15 second reviewer pass, grade A−):** θ-monotonicity rephrased for single-rule clarity (primary lies between adjacents inclusive, same sign, ≥25% magnitude); FM#1 extended with coherence-floor clause (`min(|delta_MNQ|, |delta_MES|) < 0.05 R` auto-fails); C2 seed expanded to `(symbol, trading_day, session)` to defeat within-day clustering masking; §4.1 "separable in expectation" language downgraded (interaction term is the test, not an assertion); §5 K_global freeze semantics clarified (live-queried at commit, then frozen for gate arithmetic); §5 Bailey margin disclosure tightened (1.27 yr margin named; compensating gates enumerated).
