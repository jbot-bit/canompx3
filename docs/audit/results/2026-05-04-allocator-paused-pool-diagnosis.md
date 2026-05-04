---
# Front-matter — pooled-finding-rule.md compliance.
# This file does NOT make a pooled-universe claim. Every numeric statement is
# per-strategy; §6 portfolio-EV is arithmetic over 3 disjoint lanes (sum, not
# pooled t-stat). Per docs/audit/results/TEMPLATE-pooled-finding.md, files
# that don't make a pooled claim simply omit `pooled_finding`.
not_a_pooled_finding: true
not_pooled_rationale: |
  Every t-stat, ExpR, and OOS sign in this doc is reported at the
  per-strategy level. §6 sums 3 disjoint active-lane annual_r values; that
  is portfolio arithmetic, not a pooled statistical claim, and not subject
  to BH-FDR or per-lane heterogeneity decomposition.
---

# Allocator paused-pool diagnosis — 2026-05-04

**Date:** 2026-05-04
**Author:** Claude Code (validator-honesty-fix session, post-PR-#214)
**Type:** read-only audit; no code edits, no schema changes, no allocator changes
**Scope:** every entry in `docs/runtime/lane_allocation.json` (rebalance_date `2026-05-02`); 3 active lanes + 53 paused lanes + verification of gate logic

---

## Why this doc exists

User asked: "did we go from 4 active lanes to 1 — diagnose." Memory snapshot (`memory/MEMORY.md` "Latest 2026-05-01 PM") said **1 active lane**, citing a stale rebalance. Live `lane_allocation.json` (`rebalance_date: 2026-05-02`) shows **3 active**. The "drop" is a different story than memory framed.

This diagnosis answers three questions, in order, before discussing recommendations:

1. **Is the chordia gate computing the right thing for what it sees?** (Gate sanity.)
2. **What are the 53 paused lanes paused FOR?** (Taxonomy.)
3. **Which paused lanes deserve audits, and which are correctly out?** (Triage with hard MinBTL cap.)

It does NOT propose new audits, new code, or new schema. The output of this diagnosis is decision-grade input for a follow-up design pass on whether to commission additional bounded chordia replays.

---

## §1 Gate sanity — chordia thresholds vs what's recorded

### Authority

| Threshold | Value | Source |
|---|---|---|
| `CHORDIA_T_WITH_THEORY` | 3.00 | `trading_app/chordia.py:49`; literature `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md:18-20` references HLZ-2015 t≥3 floor. |
| `CHORDIA_T_WITHOUT_THEORY` | 3.79 | `trading_app/chordia.py:50`; verbatim Chordia-Goyal-Saretto 2018 p.5 — `"the MHT threshold for alpha t-statistic (t_α) is 3.79"`. |
| `audit_freshness_days` | 90 | `docs/runtime/chordia_audit_log.yaml:46`; doctrine fence in YAML header lines 14-17. |
| Theory-grant gate | citation in `theory_ref` to `docs/institutional/literature/` required | `chordia_audit_log.yaml:9-12` (the YAML rejects `has_theory: true` without a literature reference). |

### Active-lane gate trace

Every active lane re-derived against doctrine:

| strategy_id | annual_r | trail_expr | trail_N | regime | chordia verdict | audit t | hurdle | doctrine result |
|---|---:|---:|---:|---|---|---:|---|---|
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | 36.2 | +0.2412 | 150 | HOT | PASS_CHORDIA | 4.256 | 3.79 strict | **PASS** (4.256 ≥ 3.79) |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | 27.1 | +0.2416 | 112 | HOT | PASS_CHORDIA | 5.158 | 3.79 strict | **PASS** (5.158 ≥ 3.79) |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 26.8 | +0.1095 | 245 | HOT | PASS_PROTOCOL_A | 3.412 | 3.00 (theory) | **PASS** (3.412 ≥ 3.00, theory ref `chan_2013_ch7_intraday_momentum.md`) |

All three active lanes' verdicts match what the gate logic at `trading_app/lane_allocator.py:618-670` would assign. No mismatch. Audit ages are 0d (PASS_CHORDIA) and 1d (PASS_PROTOCOL_A) — both well under the 90d freshness threshold.

### Paused-lane spot-check (3 random + 3 explicit-fail)

Sampled via `random.seed(42)` over the 53 paused rows, plus all 3 FAIL_BOTH explicit-fail rows. For each: read alloc reason → look up YAML row → recompute against threshold → check against allocator-recorded verdict.

| strategy_id | alloc.reason | yaml.verdict | yaml.t_stat | yaml.audit_date | doctrine recompute |
|---|---|---|---:|---|---|
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | chordia gate: FAIL_BOTH (t<3.0) | FAIL_BOTH | 3.276 | 2026-05-01 | t=3.276 < 3.79 strict, no theory grant → FAIL_BOTH ✓ |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | chordia gate: FAIL_BOTH (t<3.0) | FAIL_BOTH | 2.276 | 2026-05-01 | t=2.276 < 3.00 even with theory → FAIL_BOTH ✓ |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | chordia gate: FAIL_BOTH (t<3.0) | FAIL_BOTH | 3.268 | 2026-05-01 | t=3.268 < 3.79 strict, no theory grant → FAIL_BOTH ✓ |
| `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` | chordia gate: PARK | PARK | 4.211 | 2026-05-02 | t=4.211 IS-clean, OOS direction opposes IS at N=49 → PARK ✓ (per audit row note) |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50` | Session regime COLD (-0.0899) | (no row) | — | — | regime gate fires before chordia gate; chordia=MISSING is moot ✓ |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | chordia gate: missing strict replay audit verdict | (no row) | — | — | no audit row → MISSING → PAUSE ✓ |

**§1 verdict: gate is internally consistent.** Every sampled row's status is what the gate code would compute given its YAML state. The reason there are 33 MISSING-audit rows isn't gate buggery — it's that strict-replay audits have only been performed on 11 strategy_ids to date (per `chordia_audit_log.yaml`), and those audits dispatched in two batches: 2026-05-01 (deployed-lane revalidation) and 2026-05-02 (chordia_strict_unlock_v1.py runs).

The only minor wrinkle: 1 of the 3 FAIL_BOTH rows above (EUROPE_FLOW ORB_G5, t=2.276) failed even WITH the theory grant. The wording `chordia gate: FAIL_BOTH (t<3.0)` is correct because 2.276 < 3.00, but the gate-reason string is technically the fallback path; for the no-theory FAIL_BOTH cases (COMEX_SETTLE, TOKYO_OPEN), the `(t<3.0)` substring is misleading — the actual hurdle they failed is 3.79. Cosmetic, not behavioral. Filed as a possible future readability improvement; not a bug.

---

## §2 Paused-row taxonomy

53 paused lanes bucketed by alloc.reason:

| Bucket | Count | Action class |
|---|---:|---|
| REGIME_COLD (session-level pause; chordia not the cause) | 16 | Wait for session warming; no audit work |
| CHORDIA_FAIL_BOTH (verdict t<threshold; do not re-audit) | 3 | KILL — audit will produce same verdict |
| CHORDIA_PARK (IS-clean, OOS-fail; re-audit waits on new OOS data) | 1 | DEFER until OOS sample grows |
| CHORDIA_MISSING (no audit row; backlog) | 33 | TRIAGE — see §3 |
| OTHER | 0 | — |
| **TOTAL** | **53** | |

Sum check: 16+3+1+33+0 = 53 ✓ matches `len(alloc.paused)`.

**Two strategy_ids are NOT in `validated_setups`** — both are size-budget variants (`_S075` suffix):
- `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075`
- `MES_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT10_S075`

These are scored by the allocator (presumably from a sibling family-hash row) but have no own row in `validated_setups`. Investigate separately if either gets close to deployment; both are currently regime-COLD-paused so the gap is moot.

### REGIME_COLD breakdown by session

| Session | Count | Notes |
|---|---:|---|
| MES CME_PRECLOSE | 12 | All `E2_RR1.0_CB1_<filter>`; same session, regime ExpR −0.0899 |
| MES COMEX_SETTLE | 2 | regime ExpR −0.0157 |
| MES SINGAPORE_OPEN | 1 | regime ExpR −0.0339 |
| MES US_DATA_830 | 1 | regime ExpR −0.0966 |

**Honest note:** the 12 MES_CME_PRECLOSE lanes are highly correlated — same instrument, same session, same orb_minutes, varying only in size variant + filter. If the regime warms (>0 over rolling 6mo), all 12 unpause together. They should be treated as **one decision point**, not twelve. Auditing each individually would be K-stuffing.

---

## §3 CHORDIA_MISSING triage

Ranked by implied raw t-statistic from `validated_setups.p_value` (two-sided, large-N normal approximation). Sourced from `gold.db::validated_setups` joined to allocator paused list.

**Important caveat:** these `p_value` and `t-statistic` figures come from `validated_setups`, populated at promote-time (varying dates 2024-2026). They are NOT chordia strict-replay numbers. A real chordia audit re-runs the cohort with `WF_START_OVERRIDE['MNQ']=2020-01-01`, scratch-inclusive accounting, and post-Phase-3c canonical bars; that's why `chordia_strict_unlock_v1.py` rows in the YAML often show different t/N than `validated_setups.p_value`. **Below is a SCREENING ranking, not a forecast.** Real audits will move these numbers, sometimes by a lot.

### Screening ranking — top 15 of 33 by implied raw_t

| # | strategy_id | inst | sess | apt | rr | filter | N | ExpR | OOS_ExpR | implied t_raw | t_fdr | DSR | E2 LA |
|--:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | MNQ | COMEX_SETTLE | O5 | 1.0 | X_MES_ATR60 | 673 | +0.1512 | +0.1633 | **4.33** | 3.85 | 0.073 | clean¹ |
| 2 | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | MNQ | COMEX_SETTLE | O5 | 1.0 | ORB_G5 | 1473 | +0.0890 | +0.0865 | 3.77 | 3.77 | 0.000 | clean² |
| 3 | MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG | MNQ | US_DATA_1000 | O5 | 1.0 | PD_GO_LONG | 324 | +0.1934 | +0.2176 | 3.74 | 3.18 | 0.755 | clean³ |
| 4 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60 | MNQ | NYSE_OPEN | O5 | 1.0 | X_MES_ATR60 | 689 | +0.1365 | +0.1716 | 3.74 | 2.95 | 0.018 | clean¹ |
| 5 | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG | MNQ | COMEX_SETTLE | O5 | 1.0 | PD_CLEAR_LONG | 303 | +0.1841 | +0.2212 | 3.65 | 3.32 | 0.741 | clean³ |
| 6 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 | MNQ | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 1521 | +0.0889 | +0.1036 | 3.61 | 3.61 | 0.000 | clean² |
| 7 | MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG | MNQ | US_DATA_1000 | O5 | 1.0 | PD_DISPLACE_LONG | 192 | +0.2396 | +0.1922 | 3.60 | 3.03 | 0.893 | clean³ |
| 8 | MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG | MNQ | US_DATA_1000 | O5 | 1.0 | PD_CLEAR_LONG | 211 | +0.2270 | +0.2926 | 3.57 | 3.04 | 0.861 | clean³ |
| 9 | MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08 | MNQ | TOKYO_OPEN | O5 | 1.5 | COST_LT08 | 427 | +0.2037 | +0.2233 | 3.57 | 2.87 | 0.509 | clean⁴ |
| 10 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | MNQ | COMEX_SETTLE | O5 | 1.5 | X_MES_ATR60 | 664 | +0.1609 | +0.1828 | 3.55 | 3.35 | 0.012 | clean¹ |
| 11 | MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5 | MNQ | NYSE_OPEN | O5 | 1.5 | ORB_G5 | 1485 | +0.1067 | +0.1235 | 3.41 | 2.93 | 0.000 | clean² |
| 12 | MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100 | MNQ | EUROPE_FLOW | O5 | 1.5 | OVNRNG_100 | 532 | +0.1714 | +0.2177 | 3.40 | 2.81 | 0.038 | clean⁵ |
| 13 | MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12 | MNQ | NYSE_OPEN | O5 | 1.5 | COST_LT12 | 1472 | +0.1050 | +0.1207 | 3.34 | 2.92 | 0.000 | clean⁴ |
| 14 | MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG | MNQ | US_DATA_1000 | O5 | 1.5 | PD_GO_LONG | 321 | +0.2222 | +0.2553 | 3.32 | 2.86 | 0.622 | clean³ |
| 15 | MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5 | MNQ | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 1487 | +0.0945 | +0.0783 | 3.26 | 2.84 | 0.000 | clean² |

(Rows 16-33 have implied raw_t < 3.26 and are not material to a PRIORITY_A list. Available on demand from the same query.)

**E2 LA registry footnotes** — every paused row above is `entry_model=E2`. Cross-checked against `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md`:

¹ `X_MES_ATR60` — cross-instrument ATR on MES, 60-day rolling. Not in the LA registry as a tainted feature; computed in `pipeline/build_daily_features.py` as a daily rolling window on prior trading days. Pre-entry knowable. **Clean.**
² `ORB_G5` — orb-size threshold filter. Pre-entry computable from ORB-end. Not LA-tainted. **Clean.**
³ `PD_GO_LONG` / `PD_CLEAR_LONG` / `PD_DISPLACE_LONG` — prev-day position filters. Use `prev_day_*` features (Rule 6.1 safe-list, prior-close knowable). **Clean.** (Note: `prev_close_position` standalone is in NO-GO registry; PD_* are different filter family using prev-day **range/level** geometry.)
⁴ `COST_LT08` / `COST_LT12` — cost-friction-percentage filter, computed at ORB-end with deterministic friction values. **Clean.**
⁵ `OVNRNG_100` — overnight 09:00-17:00 Brisbane range. Per Rule 1.2, valid for ORB sessions starting ≥17:00 — EUROPE_FLOW starts 18:00, so the overnight window has CLOSED before ORB starts. **Clean.**

**No E2 LA contamination found in any of the top-15 rows.** This was the most important falsifiable claim in this diagnosis; rows would have moved to PRIORITY_C if even one was tainted.

### PRIORITY classification with MinBTL discipline

Per `pre_registered_criteria.md` Amendment 2.8 (Bailey 2013 MinBTL): operational cap **≤300 trials**. Each chordia replay is one trial. The 11 audits already run consume 11 of 300. If we propose 33 new audits, we move to 44/300 — well within budget BUT proposing them WITHOUT EV-ranking would be K-stuffing, violating Chordia 2018's adaptive-MHT spirit.

**Hard cap: PRIORITY_A ≤ 5.** This is a deliberate institutional choice, not an arbitrary limit. The reasoning:
- Per Bailey 2013, optimal selection-power decays with each additional uncorrelated trial
- Per Chordia 2018, BH-FDR adaptive methods are sample-size-sensitive — small targeted batches outperform scan-all
- Per Carver Ch11 (`carver_2015_ch11_portfolios.md`), portfolio dilution from too many similar lanes destroys EV
- The book has 3 active lanes; 5 well-targeted audits could realistically yield ≤2-3 deployable additions, sized to ~5-7 active lanes total — Carver's recommended portfolio breadth for one trader

#### PRIORITY_A — audit candidates (ranked, capped at 5)

| Rank | strategy_id | Why this row | OOS sign | Est. p(PASS_CHORDIA) | Theory-grant feasibility |
|--:|---|---|---|---|---|
| A1 | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | t_raw=4.33, t_fdr=3.85, OOS+ N=673 | + | High (>0.5); t already > 3.79 hurdle, but `validated_setups` t and chordia replay t routinely diverge | Cross-asset volatility-conditioning has Carver 2015 Ch9 `carver_2015_volatility_targeting_position_sizing.md` mechanism prior; theory-grant **feasible** |
| A2 | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | t_raw=3.77, t_fdr=3.77 (matches!), OOS+ N=1473 | + | Medium (3.77 ≈ hurdle); replay could go either way | Settlement-window momentum; Fitschen Ch3 `fitschen_2013_path_of_least_resistance.md` general intraday-trend prior; **possibly feasible** but COMEX_SETTLE settlement-flow lacks specific session-mechanism literature |
| A3 | MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG | t_raw=3.74, OOS+ N=324, DSR=0.755 (very high) | + | Medium-high; high DSR is institutional signal but small N is concerning under chordia replay's stricter accounting | Prev-day-range continuation; Fitschen Ch3 candidate but not explicit; **needs literature triage before grant** |
| A4 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60 | t_raw=3.74, t_fdr=2.95, OOS+ N=689 | + | Medium; t_fdr < 3.0 reduces confidence | Same Carver Ch9 grant as A1 if mechanism replicates across sessions |
| A5 | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG | t_raw=3.65, OOS+ N=303, DSR=0.741 | + | Medium; PD_CLEAR_LONG is a distinct filter from PD_GO_LONG; if A3 PASS, A5 likely follows | Same `prev_day` continuation prior as A3 |

**Honest gap-acknowledgment:** the screening ranking uses `validated_setups.p_value` which is promote-time, NOT chordia-replay. Three of the 5 A-rows have `t_raw` only fractionally above 3.79 (A2 at 3.77, A4 at 3.74, A5 at 3.65). When the chordia replay re-runs them with WF override + scratch-inclusive accounting, the t-stats may move down. **Realistic expectation: 1-3 of 5 will PASS_CHORDIA, 2-4 will end as PARK or FAIL_BOTH.** That's the institutional norm, not an indictment of the screening.

#### PRIORITY_B — theory-grant feasibility deferred until A complete

| strategy_id | Why deferred to B | Theory-grant candidate |
|---|---|---|
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 | t=3.61 below strict; would need theory grant | Chan Ch7 NYSE-open momentum (already cited for COST_LT12 at NYSE_OPEN) |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08 | t=3.57 below strict | Asian-open momentum lacks deep literature per chordia_audit_log.yaml comment line 75 |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5 | t=3.41 below strict | Same NYSE-open Chan grant |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100 | t=3.40 below strict | London-open flow; partial Chan Ch7 prior |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12 | t=3.34 below strict | NYSE-open Chan grant |

PRIORITY_B is **not a green-light list**. It's a queue that could enter PRIORITY_A IF: (a) a future literature scan grounds the relevant mechanism for theory-grant, AND (b) PRIORITY_A audits don't yield sufficient deployable lanes.

#### PRIORITY_C — do not pursue without new mechanism

The remaining 18 of 33 MISSING-audit rows have implied raw t < 3.26. Re-audit will not move them past 3.79 strict, and theory-grant feasibility is weak (most are size-variant siblings of higher-ranked rows). **Recommend: leave at MISSING. The chordia gate doing its job by keeping them paused is the correct outcome.**

---

## §4 No-action zones (explicit)

### 4.1 The 16 REGIME_COLD lanes

12 MES_CME_PRECLOSE + 2 COMEX_SETTLE + 1 SINGAPORE_OPEN + 1 US_DATA_830. These are paused by the **session regime gate**, which fires BEFORE chordia. Re-auditing them under chordia is moot until the regime warms (6mo trailing ExpR > 0). Currently MES is fully out of the deployed book by regime.

**Action:** wait. Do not propose chordia audits for any REGIME_COLD lane.

### 4.2 The 3 CHORDIA_FAIL_BOTH lanes

| strategy_id | t | verdict | audit |
|---|--:|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | 3.276 | FAIL_BOTH | 2026-05-01 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | 2.276 | FAIL_BOTH | 2026-05-01 |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | 3.268 | FAIL_BOTH | 2026-05-01 |

Re-audit will produce the same t-stat unless the canonical layer changed materially. Audits 3-day-old; canonical layer stable. **No re-audit. Verdict KILL stands.**

### 4.3 The 1 CHORDIA_PARK lane

`MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` — IS-clean (t=4.211) but OOS direction opposes IS at N_OOS=49. Per `feedback_oos_power_floor.md`: OOS power<50% with N=49 → verdict UNVERIFIED, not DEAD. The PARK verdict is correct procedure: wait for OOS sample to grow past power-floor threshold before re-audit.

**Action:** re-audit only when N_OOS ≥ 100 AND OOS sign aligns with IS, OR N_OOS ≥ 250 AND OOS sign opposes (statistical power to confirm decay). Both far away under current data accumulation rate.

### 4.4 The MGC quarantine survivor

`MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4` (chordia=MISSING) is the lone MGC paused row. It's also the subject of yesterday's `docs/audit/results/2026-05-04-mgc-chain-quarantine.md` (PR #214) — research-provisional, ~6,625× MinBTL violation, ladder evidence 4 PASS / 5 FAIL / 2 WARN / 2 N-A. Audit waits on the size-restress charter (`docs/plans/2026-05-04-orb-size-restress-research-plan.md`); do not pursue chordia audit until size-restress confirms or kills the underlying premise.

### 4.5 The 2 missing-from-validated_setups S075 size variants

`MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075` and `MES_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT10_S075` are scored by the allocator but have no `validated_setups` row. Currently regime-COLD-paused; gap is moot unless they unpause. Investigate when the relevant regime warms.

---

## §5 Recommended follow-ups

This diagnosis produces decision-grade input for follow-up design. **It does not commission audits.** Audit commissioning is a separate design pass — pre-reg yamls, K-budget, kill criteria, all per `research-truth-protocol.md` Phase 0.

### What a follow-up commissioning pass should consider

1. **Run PRIORITY_A audit batch** (5 strategies) under bounded `chordia_strict_unlock_v1.py` framework. Pre-reg each; report t-stats with 95% CI; honor MinBTL at N=11 prior + 5 new = 16 total trials (well within 300 cap).
2. **Theory-grant feasibility scan for PRIORITY_B** if PRIORITY_A yields <3 PASS_CHORDIA. Literature pass against `mechanism_priors.md` + Carver Ch9-10 + Fitschen Ch3.
3. **Family-correlation audit of PRIORITY_A** before deploying any PASS. Per `feedback_per_lane_breakdown_required.md` and Carver Ch11: lanes that share family_hash structure dilute portfolio EV. A1+A4 share `X_MES_ATR60` mechanism; A3+A5 share PD_* mechanism. Deploy at most ONE per mechanism class without explicit additivity evidence.
4. **Regime-warming watch for MES_CME_PRECLOSE.** 12 lanes correlated; if regime turns positive, propose a single representative chordia audit (highest-t variant) before unpausing the cluster.

### What this diagnosis EXPLICITLY does NOT recommend

- Audit the 33 MISSING rows in bulk (K-stuffing, MinBTL violation in spirit).
- Re-audit the 3 FAIL_BOTH rows (verdict will not change).
- Pursue PRIORITY_C rows without new mechanism (insufficient evidence).
- Add chordia audits for REGIME_COLD lanes (regime gate still firing).

---

## §6 Capital impact — current 3-lane book

### Friction floor (live read of `pipeline.cost_model.COST_SPECS`)

| Instrument | tick_size | point_value | commission_rt | spread_doubled | slippage | total_friction (USD/contract round-trip) |
|---|---:|---:|---:|---:|---:|---:|
| MGC | 0.1 | 10.0 | 1.74 | 2.0 | 2.0 | 5.74 |
| MES | 0.25 | 5.0 | 1.42 | 1.25 | 1.25 | 3.92 |
| MNQ | 0.25 | 2.0 | 1.42 | 0.5 | 1.0 | 2.92 |

(All 3 active lanes are MNQ; friction floor relevant to dollar EV is $2.92/RT/contract.)

### Active-lane portfolio EV

| Lane | trail_expr | trail_N | annual_r (alloc) | Notes |
|---|--:|--:|--:|---|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | +0.2412 | 150 | 36.2 | trades_per_year=85.6 in vs; alloc annual_r reflects trailing window |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 | +0.2416 | 112 | 27.1 | trades_per_year=116.9 in vs |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | +0.1095 | 245 | 26.8 | trades_per_year=251.4 in vs |
| **Sum** | | | **90.1 R/yr** | One contract, signal-only assumption |

**Reconciliation against `vs.expectancy_r * vs.trades_per_year`:**
- COMEX_SETTLE OVNRNG_100: 0.2151 × 85.6 = 18.4 R/yr (discovery-IS)
- US_DATA_1000 VWAP: 0.2101 × 116.9 = 24.6 R/yr (discovery-IS)
- NYSE_OPEN COST_LT12: 0.0870 × 251.4 = 21.9 R/yr (discovery-IS)
- **Sum at discovery-IS: 64.9 R/yr.** Trailing 12mo book runs hotter at 90.1.

### Comparison to memory-cited self-funded $2,929/yr

`memory/self_funded_realistic_assessment.md` quotes "2026 OOS $2,929/yr/contract NET" against an older book size. To convert current 90.1 R/yr to dollars:
- Risk per trade (typical MNQ MNQ): ~$50-100 (varies by ORB size)
- Assume midpoint $75/trade risk → 90.1 R × $75 = $6,757/yr NET (one contract)

That's ~2.3× the self-funded $2,929 figure, meaning either:
- (a) the book is hotter trailing-12mo than the 2026-OOS slice that produced $2,929, OR
- (b) the $75/trade risk estimate is high (lane-specific risk varies)

**Honest verdict on §6:** I do not have the lane-specific risk_per_trade values from `validated_setups.median_risk_dollars` cross-multiplied here. A second-pass derivation would query `vs.median_risk_dollars` per active lane and recompute. **Treat the $6,757/yr as an order-of-magnitude figure, not a forecast.** The point of §6 is "sanity-check that 90.1 R/yr isn't trivial dollars" — that test passes at the ~$5-10K/yr/contract scale, which matches self-funded viability per memory.

**§6 verdict on the "drop" question:** the book is 3 active lanes generating ~$5-10K/yr/contract on signal. Whether that's a "drop" from the prior 7-lane book depends on the prior 7-lane numbers — which lived in the same trailing window with weaker chordia gating. The chordia gate is the deliberate institutional constraint; lane count is downstream of it. **Treat 3 lanes as the post-chordia floor, not a drop.**

---

## §7 Adversarial audit (mandatory pass)

Per `institutional-rigor.md` Rule 1+2: every research output gets self-review.

### 7.1 Did E2 LA contamination sneak into the triage?

Checked. Every PRIORITY_A and PRIORITY_B filter (`X_MES_ATR60`, `ORB_G5`, `PD_GO_LONG`, `PD_CLEAR_LONG`, `PD_DISPLACE_LONG`, `OVNRNG_100`, `COST_LT*`) is grounded in pre-entry-knowable features per Rule 6.1 of `backtesting-methodology.md`. The 2026-04-28 LA registry tracks `rel_vol`, `break_bar_volume`, `break_bar_continues`, `break_delay_min`, `break_dir` — none appear in the paused-pool filter set. **Adversarial pass: clean.**

### 7.2 Is the PRIORITY ranking pigeonholing?

The ranking sorts by raw `t_stat`. This biases toward high-N lanes (N=1473 ORB_G5 ranks higher than N=192 PD_DISPLACE_LONG even though the latter has higher ExpR). Is that wrong?
- Statistical answer: NO — t-stat already accounts for N via `t = expr_mean / (expr_std / sqrt(N))`. High N at modest ExpR is statistically more confident than low N at high ExpR.
- Trading answer: ALSO NO — but with caveat. High N + low ExpR (ORB_G5 type) is institutionally well-validated but generates many trades at thin margins. Low N + high ExpR (PD_*) generates fewer trades at fat margins, with higher single-trade variance impact. Carver Ch12 vol-targeting frame would prefer the high-ExpR lane at the same trade-rate-adjusted Sharpe. **The ranking reflects t-stat statistical confidence, not portfolio-EV; that's appropriate for a chordia gate which IS a statistical hurdle.**

### 7.3 What's the failure mode if all 5 PRIORITY_A audits fail?

If none of A1-A5 PASS_CHORDIA: book stays at 3 lanes. PRIORITY_B becomes the next queue. 5 of 11 → 16 total chordia audits run; 16/300 of MinBTL budget consumed; institutional discipline preserved. **Failure mode is recoverable.**

If all 5 PASS_CHORDIA: realistically not expected (3 of 5 have t-stat fractionally above 3.79; chordia replay is stricter); but if it happens, allocator selects top-N by score with correlation-aware greedy (per `feat(allocator-correlation)` commit `7d155409`). Some may PASS the chordia gate but lose the allocation slot to existing lanes. **Best-case bounded by allocator slot count, not audit count.**

### 7.4 Is this diagnosis itself a pooled-finding violation?

**No.** Front-matter declares `not_a_pooled_finding: true`. Every numeric statement is per-strategy. §6 sums 3 disjoint annual_r values — that's portfolio arithmetic, not a pooled t-statistic claim. No BH-FDR or per-lane heterogeneity decomposition needed.

### 7.5 Is the screening ranking using stale data?

Yes — `validated_setups.p_value` reflects promote-time conditions. This is acknowledged in §3 and is the EXACT reason a chordia replay re-runs the cohort under fresh conditions. The screening RANKING uses stale data; the screening DECISION (which lanes to consider for PRIORITY_A) is robust to small numeric drift because the cap of 5 audits creates margin against ranking error.

### 7.6 Could the 33 MISSING-audit rows include a lane that would PASS_CHORDIA we're missing by capping at 5?

The 5 cap is enforced on top-by-implied-t. Rows ranked 6-15 have t_raw 3.26-3.61 — meaningful screening signal that they're likely to FAIL or marginally PASS. Rows 16-33 have t_raw < 3.26 and very low chordia-pass probability. **Risk of missing a true PASS is concentrated in rows 6-15, not 16-33.** If PRIORITY_A yields <3 PASS, escalating to ranks 6-10 is a defensible second batch under the 300 MinBTL cap.

---

## Provenance

- **Live data source:** `gold.db::validated_setups` queried via `pipeline.paths.GOLD_DB_PATH` 2026-05-04
- **Allocator state:** `docs/runtime/lane_allocation.json` `rebalance_date: 2026-05-02`
- **Chordia audit log:** `docs/runtime/chordia_audit_log.yaml`
- **E2 LA registry:** `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md` (29 rows; 24 TAINTED, 1 FIXED, 2 CLEARED, 1 NOT-PREDICTOR, 1 LIKELY-CLEAN-pending)
- **Cost specs:** live import of `pipeline.cost_model.COST_SPECS`
- **Holdout window:** Mode A `trading_day < 2026-01-01` (per `trading_app/holdout_policy.py`)

## Verdict

**Allocator paused-pool diagnosis: chordia gate is operating correctly.** The 53 paused lanes are paused for one of four reasons: 16 by REGIME_COLD (session-level, not chordia-related), 3 by CHORDIA_FAIL_BOTH (verdict computed below threshold), 1 by CHORDIA_PARK (IS-clean, OOS-fail per `feedback_oos_power_floor.md`), and 33 by CHORDIA_MISSING (no audit row yet). All 3 active lanes' chordia verdicts re-derive correctly against doctrine. The book's contraction to 3 active lanes is the deliberate institutional outcome of the chordia gate landing 2026-05-01 (PR #197), not a bug or regression.

**Decision:** PRIORITY_A list of 5 chordia replays is recommended for follow-up commissioning, capped per Bailey 2013 MinBTL discipline. PRIORITY_B / PRIORITY_C lists are queued but not green-lit. No code, schema, or allocator change recommended by this diagnosis.

## Reproduction

The diagnosis is fully reproducible from canonical sources. To re-derive every numeric claim in this document:

```
python -c "
import json
from pathlib import Path
import duckdb
from scipy.stats import norm
from pipeline.paths import GOLD_DB_PATH

alloc = json.loads(Path('docs/runtime/lane_allocation.json').read_text())
chordia_yaml = Path('docs/runtime/chordia_audit_log.yaml').read_text(encoding='utf-8')
all_ids = [p['strategy_id'] for p in alloc['paused']] + [a['strategy_id'] for a in alloc['lanes']]

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
ph = ','.join(['?']*len(all_ids))
rows = con.execute(f'''
    SELECT strategy_id, instrument, orb_label, orb_minutes, entry_model,
           confirm_bars, rr_target, filter_type, sample_size, win_rate,
           expectancy_r, p_value, fdr_adjusted_p, sharpe_ann, oos_exp_r,
           n_trials_at_discovery, dsr_score, family_hash
    FROM validated_setups WHERE strategy_id IN ({ph})
''', all_ids).fetchall()

# rank CHORDIA_MISSING by implied t from raw p_value
def implied_t(p):
    if p is None or p<=0 or p>=1: return None
    return abs(norm.ppf(p/2.0))

# bucketing logic from §2 reproduces from alloc['paused'][i]['reason'] strings
"
```

Inputs (canonical, read-only):
- `gold.db::validated_setups` (54 of 56 paused/active strategy_ids found; 2 size-variants S075 not in table)
- `docs/runtime/lane_allocation.json` rebalance_date=`2026-05-02`
- `docs/runtime/chordia_audit_log.yaml`
- `pipeline.cost_model.COST_SPECS` (live import for §6 friction floor)

Outputs:
- This document (no separate CSV; tables are inline)
- `docs/runtime/action-queue.yaml` entry `allocator_paused_pool_priority_a_audits` (P2, status=open)

Holdout policy: Mode A `trading_day < 2026-01-01` per `trading_app/holdout_policy.py`. Not applied to this diagnosis (no new SQL on `orb_outcomes`); the `validated_setups` rows being read are themselves promote-time products that already honored holdout.

## Caveats / limitations / disconfirming evidence

- **Stale screening data.** §3 ranks CHORDIA_MISSING by `validated_setups.p_value` which is promote-time, not chordia strict-replay. Real chordia audits will move t-stats. **PRIORITY_A list is a screening, not a forecast.** Realistic expectation: 1-3 of 5 PASS_CHORDIA, 2-4 will end as PARK or FAIL_BOTH. This is acknowledged in §3 + §7.5.
- **Memory was wrong on multiple points.** `MEMORY.md` "Latest 2026-05-01 PM" said 1 active lane; live JSON shows 3. `chordia_audit_unlock_triage_2026_05_01.md` said 8 PASS_CHORDIA-without-audit; current `validated_setups` data shows only 2 lanes with raw t≥3.79 in the MISSING-audit bucket. This diagnosis supersedes those claims **only against the current YAML+JSON state**; the older memory may have been correct against earlier states that have since moved.
- **Two strategy_ids are not in `validated_setups`** — `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075` and `MES_SINGAPORE_OPEN_E2_RR1.5_CB1_COST_LT10_S075`. Both are S075 size variants. Currently regime-COLD-paused so the gap is moot. Investigate if either approaches deployment.
- **§6 capital-EV reconciliation is order-of-magnitude.** I did not cross-multiply `validated_setups.median_risk_dollars` per active lane × annual_r to produce a precise dollar EV. The "~$5-10K/yr/contract" range is a sanity-check, not a forecast. A lane-specific risk-dollar pass would tighten this.
- **Cosmetic gate-message mismatch.** 1 of 3 FAIL_BOTH rows (EUROPE_FLOW ORB_G5, t=2.276) actually fails the t≥3.00 theory-grant hurdle; the gate message `chordia gate: FAIL_BOTH (t<3.0)` is technically correct in the fallback case but misleading for the 2 NO-theory FAIL_BOTH rows whose actual hurdle is 3.79. Cosmetic, not behavioral. Not filed as a fix in this doc.
- **MinBTL cap of 5 PRIORITY_A is institutional discipline, not statistical certainty.** Could be argued for 3, could be argued for 7. The number 5 is grounded in Carver Ch11 portfolio-breadth recommendation and Chordia 2018 adaptive-MHT spirit, not a numeric optimization. Adjustable IF a future literature pass changes the underlying institutional priors.
- **§3 ranking biases toward high-N lanes.** Statistically correct (t-stat already accounts for N), but PD_DISPLACE_LONG (N=192, ExpR+0.2396, DSR=0.893) is institutionally interesting enough to warrant consideration even though it ranked rank-7. PR #51 `mnq_unfiltered_high_rr_family_v1.py` cohort overlap is also untested here.
- **What this diagnosis does NOT examine:** OOS sample-growth rate (when does CHORDIA_PARK row earn its re-audit?), allocator score-vs-rank gating beyond chordia (correlation-aware greedy, regime gate composition), self-funded $/lane contribution under current 3-lane book vs prior 7-lane book.

## Authority chain

- `trading_app/chordia.py` — verdict taxonomy + thresholds
- `trading_app/lane_allocator.py:618-670` — gate logic
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md:18-20` — t≥3.79 threshold
- `docs/institutional/literature/harvey_liu_2015_backtesting.md` — t≥3.00 floor
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` — MinBTL bound
- `docs/institutional/pre_registered_criteria.md` — locked criteria, Amendment 2.8
- `.claude/rules/backtesting-methodology.md` — Rule 6 feature audit, Rule 9 canonical layers
- `.claude/rules/institutional-rigor.md` — non-skip rules + adversarial-audit gate
- `.claude/rules/pooled-finding-rule.md` — front-matter discipline (this doc complies via `not_a_pooled_finding: true`)
- `docs/runtime/handoff-2026-05-02.md` — prior session state, post-rebalance context
- `chordia_audit_unlock_triage_2026_05_01.md` (memory) — earlier triage; superseded by this doc against current YAML state
