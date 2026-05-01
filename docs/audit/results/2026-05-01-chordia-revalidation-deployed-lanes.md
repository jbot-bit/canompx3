---
pooled_finding: false
pre_reg: docs/audit/hypotheses/2026-05-01-chordia-revalidation-deployed-lanes.yaml
runner: research/chordia_revalidation_deployed_2026_05_01.py
date: 2026-05-01
version: 3-steel-man-honest-grounding-2026-05-01
supersedes_v1_v2_inline: yes  # see § Audit correction history below
---

# Chordia revalidation of currently-deployed lanes — 2026-05-01 (v3 steel-man honest grounding)

## Scope

Defensive honesty gate per `pre_registered_criteria.md` Criterion 4. K=4 Pathway-B
individual hypothesis tests on the 4 lanes currently DEPLOY-flagged in
`lane_allocation.json` rebalance_date 2026-04-18 (the lane set at the time the
pre-reg was locked, BEFORE the 2026-05-01 monthly rebalance recommendation).

The audit asks the most basic honesty question: **what is each deployed lane's
Chordia t-statistic on its IS slice (`trading_day < HOLDOUT_SACRED_FROM`), and
where does it land vs the locked thresholds, given honest literature grounding?**

Pre-reg targets: `MNQ_EUROPE_FLOW`, `MNQ_COMEX_SETTLE`, `MNQ_NYSE_OPEN`,
`MNQ_TOKYO_OPEN` — all with E2 entry, CB1, mixed RR1.0/1.5, mixed ORB_G5/COST_LT12
filters.

**Type:** defensive honesty gate. K=4 Pathway-B individual hypothesis tests.
**Mode:** Mode A IS-only (`trading_day < 2026-01-01` per
`trading_app.holdout_policy.HOLDOUT_SACRED_FROM`).
**Lanes audited:** 4 deployed lanes per `lane_allocation.json` rebalance_date
2026-04-18 — BEFORE the 2026-05-01 monthly rebalance recommendation that
surfaces O15/O30 lanes from PR #189.
**Pooled finding:** false. Per-lane verdicts only; no aggregate p across lanes.

## Verdict — 3-of-4 lanes FAIL the defensive Chordia honesty gate

| Hyp | Lane | t | Theory | Threshold | Verdict |
|---|---|---:|:---:|---:|---|
| H1 | MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | **2.276** | True | 3.00 | **FAIL_BOTH** |
| H2 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | **3.276** | False | 3.79 | **FAIL_BOTH** |
| H3 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 3.412 | True | 3.00 | **PASS_PROTOCOL_A** |
| H4 | MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | **3.268** | False | 3.79 | **FAIL_BOTH** |

The audit went through three honesty passes:
- **v1 (initial run):** 1 FAIL / 3 PASS — assumed all 4 lanes had pre-registered
  theory grounding per "Fitschen Ch 5-7 + Chan" citations.
- **v2 (post-Fitschen-PDF audit):** 4 FAIL / 0 PASS — Fitschen Ch 5-7 cite confirmed
  phantom by direct PDF reading; conservative blanket `has_theory=False`.
- **v3 (this version, steel-man pass):** 3 FAIL / 1 PASS — granted `has_theory=True`
  to lanes whose entry mechanism + instrument + session match Chan Ch 7's verbatim
  "entry at the market open" + FSTX equity-index European-session-open case study
  (p.155-157). H1 EUROPE_FLOW and H3 NYSE_OPEN qualify; H2 COMEX_SETTLE (metals
  close, not equity-index open) and H4 TOKYO_OPEN (Asian-session, not in Chan FSTX
  example) do not.

**Real-money exposure unchanged.** Live deployment is signal-only (F-1 hard gate
verified active per `trading_app/live/session_orchestrator.py:237-266` —
`_apply_signal_only_f1_seed` enforces position-cap even in signal-only). Doctrine
action below is ledger-only — no allocator code, no `prop_profiles.py`, no
`lane_allocation.json` edits required by this gate.

**Decision on whether to escalate beyond ledger-only doctrine action belongs to
user.**

## Per-lane verdict table — full

| Hyp | Lane | Theo | Thr | N_uni | N_fire | Fire% | IS ExpR | Sharpe | t | Long t / N | Short t / N | Verdict |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H1 | MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | True | 3.00 | 1,718 | 1,583 | 92.1% | 0.0643 | 0.0572 | 2.276 | 1.902 / 773 | 1.322 / 810 | FAIL_BOTH |
| H2 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | False | 3.79 | 1,658 | 1,577 | 95.1% | 0.0941 | 0.0825 | 3.276 | 2.413 / 839 | 2.215 / 738 | FAIL_BOTH |
| H3 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | True | 3.00 | 1,719 | 1,695 | 98.6% | 0.0790 | 0.0829 | 3.412 | 2.064 / 862 | 2.765 / 833 | PASS_PROTOCOL_A |
| H4 | MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | False | 3.79 | 1,722 |   950 | 55.2% | 0.1222 | 0.1060 | 3.268 | 1.948 / 469 | 2.666 / 481 | FAIL_BOTH |

Threshold via `trading_app.chordia.chordia_threshold(has_theory)`:
- `True` → 3.00 (`CHORDIA_T_WITH_THEORY`)
- `False` → 3.79 (`CHORDIA_T_WITHOUT_THEORY`)

**Observations:**
- **H3 NYSE_OPEN is the single survivor.** Cleanest mechanism-to-literature mapping
  of the four lanes (Chan Ch 7 p.155 stop-cascade-at-market-open + equity-index
  intraday momentum FSTX case study). t = 3.412 clears the theory-grounded threshold
  by 14% — comfortable but not stellar.
- **H1 EUROPE_FLOW also gets `has_theory=True` per consistency** (Chan FSTX is
  literally a European-session equity-index future) but its raw t = 2.276 fails even
  the relaxed 3.00 threshold. Theory grounding doesn't save a lane whose IS
  evidence is genuinely weak.
- **H2 COMEX_SETTLE and H4 TOKYO_OPEN sit at t = 3.27** — they would PASS the
  theory-grounded threshold (3.00) but the audit could not find honest literature
  support. H2 because COMEX_SETTLE is a metals settlement window (not equity-index
  session-open). H4 because Chan's FSTX example is European, not Asian, and
  generalizing by region is a stretch Chan doesn't make.
- **No direction sign-flip on any lane.** Long and short legs agree direction-wise.
  Pooled t is honest summary per `feedback_per_lane_breakdown_required.md`.
- **Fire rates 92-99% on 3 of 4 lanes** trigger RULE 8.1 `extreme_fire` smell-test
  but mechanistic (size and cost gates fire on most days).

## Theory-grounding audit — what each lane has and doesn't have

Required by `pre_registered_criteria.md` Criterion 4: Chordia threshold relaxation
from 3.79 to 3.00 ONLY allowed for strategies with "strong pre-registered economic
theory support." Each lane was audited against direct PDF reads of cited literature
+ existing local extracts.

### H1 — MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 — `has_theory=True` (verdict: FAIL_BOTH on raw t)

- **Original v1 cite:** Fitschen Ch 5-7 (institutional-flow concentration on European
  morning ORB; ORB_G5 captures size-gate that filters compression-day false breaks)
- **Audit:** Fitschen Ch 5/6/7 cite phantom (verified by direct PDF read of
  `resources/Building_Reliable_Trading_Systems.pdf` pp.65-117). Ch 5=Exits,
  Ch 6=Filters with OPPOSITE polarity to ORB_G{N}, Ch 7=Money-Mgmt.
- **Steel-man (v3):** Chan 2013 Ch 7 p.156 FSTX (Dow Jones STOXX 50) case study IS a
  European-session equity-index future trading session-open momentum at Sharpe 1.4.
  EUROPE_FLOW MNQ E2 is structurally identical (European-session equity-index future,
  stop-cascade breakout at session open). This grounds entry mechanism.
- **`has_theory: True`** by consistency with H3 (same Chan Ch 7 grounding).
- **Verdict:** raw t = 2.276 fails even the relaxed 3.00 threshold. Theory grounding
  cannot save a lane whose multi-year IS evidence is genuinely weak. **FAIL_BOTH.**

### H2 — MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 — `has_theory=False` (verdict: FAIL_BOTH)

- **Original v1 cite:** Fitschen Ch 5-7 (late-session institutional repositioning
  around COMEX close; ORB_G5 again gates compression)
- **Audit:** same Fitschen Ch 5-7 phantom as H1.
- **Steel-man check:** Chan 2013 Ch 7 grounds equity-index intraday momentum and
  stop-cascade-at-market-open. COMEX_SETTLE is a metals-market settlement window
  (close-driven, not open-driven) — Chan's mechanism doesn't apply. The instrument
  is MNQ (equity-index) but the SESSION is metals-settlement. No literature support
  in local extracts for "equity-index trading the metals-settlement window."
- **`has_theory: False`** — strict 3.79 threshold applies.
- **Verdict:** t = 3.276, threshold = 3.79. **FAIL_BOTH** (would pass at theory
  threshold; doesn't clear strict).

### H3 — MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 — `has_theory=True` (verdict: PASS_PROTOCOL_A)

- **Original v1 cite:** Fitschen Ch 5-7 + Chan 2013 Ch 7 (NYSE-open volatility burst
  with COST_LT12 cost-gate)
- **Audit:** Fitschen Ch 5-7 cite phantom (same as H1/H2). Chan Ch 7 grounding
  verified — verbatim from `chan_2013_ch7_intraday_momentum.md`:
  - p.155: "the triggering of stops. Such triggers often lead to the so-called
    breakout strategies. We'll see one example that involves an entry at the
    market open..."
  - p.157: "The execution of these stop orders often leads to momentum because a
    cascading effect may trigger stop orders placed further away from the open
    price as well."
  - p.156: FSTX (Dow Jones STOXX 50 futures) equity-index session-open momentum
    APR 13%, Sharpe 1.4 over 8 years.
- **Steel-man:** NYSE_OPEN is the canonical equity-index session open. MNQ E2 is
  a stop-market-on-first-range-cross entry. The Chan Ch 7 mapping is direct:
  equity-index intraday momentum on session-open via stop-cascade. COST_LT12
  filter overlay attenuates but does not reverse the entry mechanism. Per the
  project's binary `has_theory` model in `trading_app/chordia.py`, the lane-as-a-whole
  inherits theory from the entry.
- **`has_theory: True`** — relaxed 3.00 threshold applies.
- **Verdict:** t = 3.412, threshold = 3.00. **PASS_PROTOCOL_A.** Cleanest
  mechanism-to-literature mapping of the four lanes; clears by 14%.

### H4 — MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 — `has_theory=False` (verdict: FAIL_BOTH)

- **Original v1 cite:** Fitschen Ch 5-7 + Chan 2009 Ch 1 §1.4 (TOKYO_OPEN as Asia
  liquidity transition)
- **Audit:** Fitschen Ch 5-7 phantom. Chan 2009 Ch 1 §1.4 confirmed PHANTOM in
  earlier audit (`docs/institutional/literature/chan_2009_ch1_intraday_session_handling.md`,
  2026-04-27): the Chan 2009 book's Ch 1 has three sub-sections, none §1.4. The
  whole H4 cite is fabricated.
- **Steel-man check:** TOKYO_OPEN is an equity-index session-open BUT Chan FSTX
  example is European, not Asian. Generalizing the FSTX result to "all
  equity-index session-opens regardless of region" is a stretch Chan doesn't make.
  Asian-session momentum dynamics may differ from European/US per microstructure
  literature (not in `resources/`).
- **`has_theory: False`** — strict 3.79 threshold applies (conservative reading
  per `institutional-rigor.md` Rule 7).
- **Verdict:** t = 3.268, threshold = 3.79. **FAIL_BOTH.**

## Doctrine action

### H1, H2, H4 (3 lanes FAIL_BOTH) → research-provisional + signal-only continuation

Per pre-reg `failure_policy.on_FAIL_BOTH` (verbatim from yaml line 230-237):

> Lane downgrades to research-provisional + signal-only per Amendment 2.7. Lane stays
> in allocator; live OOS accrual continues; NO real-money exposure change until clean
> re-derivation lands.

**Specifically:**
1. No edit to `prop_profiles.py daily_lanes`.
2. No edit to `lane_allocation.json` (which IS the live deployment for
   `topstep_50k_mnq_auto` per `prop_profiles.py:1097-1157` — the rebalance from
   2026-04-18 to 2026-05-01 was discovered to be a deployment change, not a
   recommendation; reversal pending separate user decision).
3. No edit to `live_config`.
4. The 3 FAIL_BOTH lanes continue to fire signals to `live_signals.jsonl` for OOS
   evidence accumulation. F-1 enforces position cap even in signal-only.
5. If lanes return clean OOS dir_match in 2026-2026 (still pending power per
   `feedback_oos_power_floor.md`), a re-derivation pre-reg under stricter discovery
   may rehabilitate.

### H3 NYSE_OPEN (1 lane PASS_PROTOCOL_A) → retained as deployed

Per `failure_policy.on_PASS_PROTOCOL_A`: lane retained as deployed; no action.
Sits at t = 3.412 — comfortably above the theory-grounded threshold but well below
strict CHORDIA_STRICT (3.79). Carver Stage-2 sizing pre-reg
(`docs/audit/hypotheses/2026-05-01-carver-stage2-vol-targeted-sizing.yaml`) requires
PASS_CHORDIA strict for sizing-up consideration; H3 does not qualify.

## Limitations / what this gate does NOT do

- **Does NOT promote new lanes.** The 2026-05-01 rebalance recommendation (7 new
  candidate DEPLOY lanes) is unaffected by this gate.
- **Does NOT relax thresholds.** Both PROTOCOL_A and CHORDIA_STRICT thresholds stay
  locked per `pre_registered_criteria.md` Criterion 4 and `trading_app/chordia.py`.
  v3 verdict shifts came from per-lane `has_theory` flag corrections, not threshold
  relaxation.
- **Does NOT pool across lanes.** Per `pooled-finding-rule.md`, this file declares
  `pooled_finding: false`. Each lane stands or falls on its own t-statistic.
- **Does NOT replace OOS gates.** Chordia is necessary but not sufficient. PASS lanes
  can still fail OOS dir_match (Criterion 8) or Shiryaev-Roberts monitoring
  (Criterion 12). Existing OOS power-floor veto per `feedback_oos_power_floor.md`
  continues to apply.
- **Does NOT touch allocator code.** No edit to `prop_profiles.py`,
  `lane_allocation.json`, `live_config`. All "doctrine action" entries are
  ledger-only.
- **Does NOT decide the 2026-05-01 rebalance question.** That's a separate decision
  documented in `docs/runtime/chordia-revalidation-decision-audit-2026-05-01.md`.

## Reproduction

To reproduce this audit:

```bash
PYTHONIOENCODING=utf-8 python research/chordia_revalidation_deployed_2026_05_01.py
```

Inputs:
- `gold.db` (read-only via `pipeline.paths.GOLD_DB_PATH`)
- Canonical helpers: `trading_app.chordia.compute_chordia_t`, `trading_app.chordia.chordia_threshold`,
  `research.filter_utils.filter_signal` (delegates to `trading_app.config.ALL_FILTERS[key].matches_df`)
- `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (= 2026-01-01)

Outputs:
- stdout: per-lane verdict table (reproduced in § Verdict above)
- This file: result doc

Runtime: < 60 seconds (4 lanes, ~1700 candidate days IS-only per lane).

## Methodology audit

- **Triple-join** on `(trading_day, symbol, orb_minutes)` per
  `daily-features-joins.md`. Verified in
  `research/chordia_revalidation_deployed_2026_05_01.py:55-72`.
- **Canonical filter delegation**: `research.filter_utils.filter_signal` →
  `trading_app.config.ALL_FILTERS[key].matches_df` per `research-truth-protocol.md`.
- **Canonical threshold selection**: `trading_app.chordia.chordia_threshold(has_theory)`.
- **IS-only**: `WHERE trading_day < trading_app.holdout_policy.HOLDOUT_SACRED_FROM`
  (= 2026-01-01).
- **Scratch policy**: `pnl_r.fillna(0.0)` per `feedback_scratch_pnl_null_class_bug.md`.
  Not load-bearing on these 4 lanes.
- **Independent SQL cross-check** (H1): independent query reading
  `orb_EUROPE_FLOW_size > 5` directly returned N_fire=1566, ExpR=0.0658. Runner via
  canonical `OrbSizeFilter.matches_df` returned N_fire=1583, ExpR=0.0643. The
  17-trade and 0.0015R gap reflects the canonical filter's NaN-handling — the
  correct (canonical) answer.
- **Direction split** derived from `target_price > stop_price` (no `direction` column
  in `orb_outcomes`).
- **MinBTL**: K=4 trials at expected per-lane Sharpe 0.05-0.11 → MinBTL ≈ 230 trades.
  All 4 lanes have N_fire >> 230. Satisfied.

## Audit correction history

### v1 (drafted 2026-05-01 ~mid-session, never committed)
Original verdicts: 1 FAIL_BOTH (H1), 3 PASS_PROTOCOL_A. All 4 lanes assumed
`has_theory=True` per "Fitschen Ch 5-7 + Chan" citations.

### v2 (drafted 2026-05-01 post-Fitschen-PDF-audit, never committed)
User-requested grounding audit caught the v1 cites. Direct PDF extraction of
`Building_Reliable_Trading_Systems.pdf` Ch 5/6/7 confirmed phantom citations.
Conservative blanket `has_theory=False` → 4 FAIL_BOTH / 0 PASS.

### v3 (this version, 2026-05-01 final)
Steel-man pass on Chan 2013 Ch 7 grounding. Chan p.155-157 verbatim grounds
stop-cascade-at-market-open mechanism + FSTX European equity-index session-open
case study (Sharpe 1.4). Lanes whose (entry + instrument-class + session-class)
match this pattern get `has_theory=True`:
- H1 EUROPE_FLOW: True (direct FSTX class match)
- H2 COMEX_SETTLE: False (metals settlement, not equity-index session-open)
- H3 NYSE_OPEN: True (canonical equity-index session-open)
- H4 TOKYO_OPEN: False (Asian-session, generalization Chan doesn't make)

Result: 1 PASS_PROTOCOL_A (H3), 3 FAIL_BOTH (H1/H2/H4).

Also caught and fixed in same iteration: runner had a verdict-logic bug — was using
hardcoded `CHORDIA_T_WITH_THEORY` constant and ignoring per-lane `has_theory` flag.
Fixed to call `trading_app.chordia.chordia_threshold(has_theory)`.

## Honest-grounding statements

- The grounding audit was triggered by user pushback, not by the runner finding
  something suspicious. Without that pushback, v1 verdicts would have stood as
  doctrine. This is the institutional value of user grounding-skepticism.
- The pre-reg's "Fitschen Ch 5-7" citation was a fabricated reference to chapters
  that exist but don't cover what was claimed. Direct PDF extraction surfaced the
  gap. Going forward, every literature citation MUST specify chapter AND page-range
  AND verbatim mechanism per `fitschen_2013_path_of_least_resistance.md` Usage Rule 1.
- The H3 PASS_PROTOCOL_A verdict should NOT be read as "H3 is validated." Chordia
  is necessary but not sufficient; OOS gates still apply.
- The 3 FAIL_BOTH lanes are NOT being removed from signal-only. Live OOS evidence
  continues to accumulate. The doctrine action is a ledger marker, not a kill.

## Cross-reference

- Pre-reg (with audit-correction block): `docs/audit/hypotheses/2026-05-01-chordia-revalidation-deployed-lanes.yaml`
- Runner: `research/chordia_revalidation_deployed_2026_05_01.py`
- Design note: `docs/runtime/chordia-revalidation-honest-grounding-design-2026-05-01.md`
- Three-decision audit: `docs/runtime/chordia-revalidation-decision-audit-2026-05-01.md`
- Upgraded Fitschen extract: `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md`
- Chan Ch 7 extract (steel-man source): `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`
- Phantom-citation precedent: `docs/institutional/literature/chan_2009_ch1_intraday_session_handling.md`
- F-1 protective-state verification: `trading_app/live/session_orchestrator.py:237-266`
- Carver Stage-2 (gated downstream — still MOOT, no PASS_CHORDIA): `docs/audit/hypotheses/2026-05-01-carver-stage2-vol-targeted-sizing.yaml`
- GARCH cross-session (gating reconsidered, H2 home-lane FAIL): `docs/audit/hypotheses/2026-05-01-mnq-garch-p70-cross-session-companion.yaml`
- Aperture extension (already self-corrects on Fitschen): `docs/audit/hypotheses/2026-05-01-aperture-extension-o15-o30-london-usdata.yaml`
