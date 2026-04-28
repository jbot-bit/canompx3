# Parked Pathway-B Additivity Triage — 2026-04-29

**Date:** 2026-04-29
**Profile under audit:** `topstep_50k_mnq_auto`
**Canonical gate:** `trading_app.lane_correlation.check_candidate_correlation`
**Thresholds:** `RHO_REJECT_THRESHOLD = 0.70`, `SUBSET_REJECT_THRESHOLD = 0.80` (same-session only)
**Runner:** `research/2026-04-29-parked-pathway-b-additivity-audit.py`

---

## Question Answered

For the two PARK_PENDING_OOS_POWER Pathway-B candidates that landed
2026-04-28, can each one be deployed as **additive** capacity to the
existing 6-lane book if/when its OOS power floor (`N_OOS_on >= 50`)
clears (~Q3-2026), or does the canonical lane-correlation gate already
block deployment regardless of OOS verification?

This audit converts a "wait 5 months for N" question into either
"wait, then deploy" or "wait, then still cannot deploy" and reduces
optionality risk on the 2026-05-15 timeline.

## Scope (locked)

- D2 **B-MES-EUR** (pre-reg `docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml`, verdict PARK_PENDING_OOS_POWER)
- D4 **B-MNQ-COX** (pre-reg `docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-pathway-b-v1.yaml`, verdict PARK_PENDING_OOS_POWER)
- B-MNQ-EUR — does **not** exist as pre-reg or result (verified 2026-04-29 via `ls docs/audit/hypotheses/`); no audit input exists.
- D-0 v2 — **excluded**: PARK_ABSOLUTE_FLOOR_FAIL is a different failure class (statistical underpowering of the IS effect); per-scope rule "don't mix failure classes".

## Method

1. Each candidate cell predicate registered as a **hypothesis-scoped** canonical filter in `trading_app/config.py::_HYPOTHESIS_SCOPED_FILTERS`:
   - `OVNRNG_PCT_GT80` — `OvernightRangeFilter(min_pct=80.0, strict_gt=True)`
   - `GARCH_VOL_PCT_GT70` — `GARCHForecastVolPctFilter(direction="high", pct_threshold=70.0, strict_gt=True)`
2. Both classes extended with `strict_gt: bool = False` (default preserves all existing `>=` semantics; opt-in `>` matches the strict pre-reg cell predicate).
3. Filters NOT routed into `BASE_GRID_FILTERS` or `get_filters_for_grid()`. They are reachable ONLY via `ALL_FILTERS` lookup (canonical-call path) or Phase-4 hypothesis-injection. Verified by per-instrument × per-session test sweep across all 12 active sessions and (MNQ, MES, MGC) — see `tests/test_trading_app/test_config.py::TestStrictGtVariants::test_*_not_in_legacy_grid_for_any_session`.
4. Deployed lanes pulled via canonical `trading_app.prop_profiles.get_profile_lane_definitions("topstep_50k_mnq_auto")` — single source of truth.
5. Canonical `check_candidate_correlation` called for each candidate against every deployed lane. Daily-pnl correlation computed on shared trading days; subset-coverage computed against the smaller of {candidate days, deployed days}.

## Verdict Matrix

| Candidate | filter_type | N_days | worst_rho | worst_subset | gate_pass | disposition |
|---|---|---:|---:|---:|---|---|
| **D2 B-MES-EUR** | `OVNRNG_PCT_GT80` | 387 | **+0.3207** | 99.7% | TRUE | **PASS_ADDITIVITY** |
| **D4 B-MNQ-COX** | `GARCH_VOL_PCT_GT70` | 407 | **+0.8071** | 100.0% | FALSE | **FAIL_ADDITIVITY** |

### D2 B-MES-EUR — PASS_ADDITIVITY

| Deployed lane | shared | rho | subset_cov | reject |
|---|---:|---:|---:|---|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 380 | +0.3207 | 98.2% | NO (different instrument; same session — but rho < 0.70) |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 273 | +0.0004 | 70.5% | NO |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | 368 | -0.0674 | 95.1% | NO |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 386 | -0.0494 | 99.7% | NO |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 295 | -0.0094 | 76.2% | NO |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 386 | -0.0350 | 99.7% | NO |

Worst rho is the +0.32 vs MNQ same-session lane — under the 0.70 threshold. Subset-coverage gate only fires for **same-session same-instrument** lanes (canonical: `lane_correlation.py:125-133`); D2 is MES vs an MNQ book, so the high subset_cov is not a reject signal. **No same-instrument deployed lane exists for MES on EUROPE_FLOW** — D2 is genuinely diversifying capacity.

### D4 B-MNQ-COX — FAIL_ADDITIVITY

| Deployed lane | shared | rho | subset_cov | reject |
|---|---:|---:|---:|---|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 406 | +0.0515 | 99.8% | NO |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 391 | +0.0101 | 96.1% | NO |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | **407** | **+0.8071** | **100.0%** | **YES — rho>0.70; subset>80% (same session, same instrument)** |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 407 | +0.0654 | 100.0% | NO |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 342 | +0.0792 | 84.0% | NO |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 407 | -0.0092 | 100.0% | NO |

D4 trades the same `(MNQ, COMEX_SETTLE, E2, CB1)` cell as the deployed `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` lane. The candidate fires on a **superset** of the deployed lane's days (subset_cov = 100% — every D4 day is also a deployed-lane day) and pnl-correlates at +0.81 — well above the 0.70 RULE 7 reject threshold. The economic interpretation: the GARCH high-vol regime on COMEX_SETTLE day-aligns with the existing ORB_G5 size filter so tightly that adding D4 to the book contributes ~0 incremental risk-adjusted capacity. **Even if OOS power clears in Q3-2026, D4 cannot deploy as currently specified.**

This corroborates the +0.7733 RULE 7 flag recorded in the D4 pre-reg at run-time and tightens it from "Phase E gate required" to "Phase E gate **fails** at the additivity step under the canonical threshold".

## Decision Implications

- **D2 B-MES-EUR** — gate PASSES at the additivity layer. Phase E admission still requires C6/C8 OOS power (~Q3-2026) and routine Pathway B downstream gates. **Optionality preserved.**
- **D4 B-MNQ-COX** — gate FAILS at the additivity layer. Two paths if the user wants to recover D4 optionality:
  1. **Re-spec the candidate** as a *replacement* for the deployed `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` (same-cell swap), not as additive capacity. Requires its own pre-reg + Pathway-B head-to-head.
  2. **Park D4 indefinitely** until either the deployed COMEX_SETTLE ORB_G5 lane decays out of the book (RULE 7 loses its anchor) or a different cell predicate is proposed that day-aligns less with ORB_G5.
  - **Brute-force OOS accrual on the existing predicate is wasted.** Even a clean OOS PASS will not unlock deployment under the canonical RULE 7 threshold.

## Caveats

- Daily-pnl correlation, not trade-level correlation. Two lanes that take different trades on the same day collapse to one daily P&L number; this is the convention `lane_correlation.py` uses (and what `check_candidate_correlation` exposes). For per-trade additivity questions, a separate trade-level audit is required.
- Subset-coverage gate is asymmetric: it only triggers when the candidate is in the same `(orb_label, instrument)` as the deployed lane. D2 (MES) escapes this by virtue of being a different instrument; if a future MES-on-EUROPE_FLOW lane is added to the book, D2 must be re-audited.
- `RHO_REJECT_THRESHOLD = 0.70` is an institutional canonical constant (`lane_correlation.py:24`). Tightening or loosening it is a policy decision, not a research finding.

## Verification

- All `test_trading_app/test_config.py` tests pass post-edit (203/203, 17 new strict-gt regression tests added).
- `pipeline/check_drift.py`: 20 violations pre-existed at HEAD before this session's edits (`certifi` / `annotated_types` / `click` import-resolution gaps in local venv). Post-edit count unchanged — no new violations introduced.
- D2 candidate N: 387 days. D4 candidate N: 407 days. Both > 100 (above noise floor for daily-pnl correlation).
- Audit script is read-only on canonical layers; no write paths to `validated_setups`, `lane_allocation.json`, or live config.

## Addendum (2026-04-29) — D4 reframing as sizing-overlay classifier

The "FAIL_ADDITIVITY → respec as replacement / park indefinitely" framing
above is structurally correct under RULE 7 but **economically incomplete**.
Direct execution evidence on already-loaded data (no new backtest) shows
the D4 predicate behaves as a **regime classifier on the already-deployed
`MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` lane**, not as a separate trade
candidate.

### Evidence (no new run; numbers are this session's)

- D4-as-candidate IS-window rho vs deployed: **+0.8063** (full-window +0.8071) — verdict holds on the lock-window, not OOS-data-dependent.
- D4 day-set is a strict subset of deployed-lane day-set: **407/1647 days IS+OOS, 370/1389 IS** (subset_cov 100% on smaller side).
- On the **deployed** `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` lane, IS-only (`trading_day < 2026-01-01`), partitioned by the D4 predicate:
  - `garch_forecast_vol_pct > 70` (D4-on): **N=370, ExpR=+0.2477**
  - `garch_forecast_vol_pct <= 70`: **N=1019, ExpR=+0.0530**
  - Delta ExpR (high − rest) = **+0.1947 R/trade**

### Mechanism reframing

D4's high correlation against the deployed lane is what disqualifies it as
a **new lane** under RULE 7. That same property — same-day signal alignment
on a strict day-subset — is what would qualify it as a **binary regime gate
on the existing lane**. Two filters are not stacked; one filter (`ORB_G5`)
already runs in production, and the GARCH>70 predicate would modulate
sizing or eligibility on that lane's existing trade-set. There is no new
position to add to the book.

This is the Carver Ch 9–10 vol-targeting / forecast-combination pattern
documented under `docs/institutional/mechanism_priors.md` — a classifier
overlaying an existing rule, not a competing rule.

### Third recovery path (in addition to the two listed earlier)

3. **Reframe D4 as a sizing-overlay classifier** on the deployed
   `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` lane. Promotion would require
   its own pre-reg with:
   - Carver Ch 9–10 forecast-combination citation (literature already
     extracted under `docs/institutional/literature/`).
   - OOS power floor evaluated on the **gate's lift**, not the lane's
     N (different power calculation; the gate sees ~24.7% of lane days).
   - Era stability of the **+0.1947 R lift**, not the lane ExpR.
   - Implementation as a binary eligibility/half-size modulator inside
     the existing lane's execution path; no new lane in
     `lane_allocation.json`, no new entry in the additivity gate.

   Under this path D4 never crosses RULE 7, because there is no new
   lane to gate.

### Disposition update

- D4 disposition (additivity gate as new lane): **FAIL_ADDITIVITY** — unchanged.
- D4 forward optionality:
  - Path 1 (replacement-lane same-cell pre-reg): still open.
  - Path 2 (park indefinitely): still open.
  - **Path 3 (sizing-overlay classifier pre-reg): newly identified;
    higher-EV than Paths 1–2 conditional on a fresh pre-reg passing
    its own gates.**

### Scope discipline (what this addendum is NOT)

- Not a new backtest. Numbers above were already produced this session
  during the audit pre/post-flight checks.
- Not a threshold tune. The `> 70` predicate is the locked pre-reg
  predicate; no sweep was run.
- Not a D5 implementation, not a pre-reg, not an allocator change.
  Path 3 requires a separate pre-reg cycle starting from
  `docs/prompts/prereg-writer-prompt.md`.

---

## Addendum 2 (2026-04-29) — OOS power-floor accrual ETA correction

The original D2/D4 pre-regs (2026-04-28) and HANDOFF both estimated
"~Q3-2026" as the OOS power-floor (`N_OOS_on >= 50`) clearance window
uniformly. Read-only check against canonical `orb_outcomes` (DB max
`trading_day = 2026-04-26`) shows the two candidates accrue at very
different rates:

| Candidate | OOS N as of 2026-04-26 | OOS density (trades/cal-day) | Projected ETA to N=50 |
|---|---:|---:|---|
| D2 B-MES-EUR  (`OVNRNG_PCT_GT80`)  | 20 / 113 cal days | 0.177 | **2026-10-09** (early Q4-2026, ~163 cal days out) |
| D4 B-MNQ-COX  (`GARCH_VOL_PCT_GT70`) | 37 / 113 cal days | 0.327 | **2026-06-01** (early Q2-2026, ~36 cal days out) |

Notes:
- Linear projection assumes the trailing 113-day fire rate holds. Real
  rate is regime-dependent; this is a planning estimate, not a guarantee.
- D4's earlier ETA does **not** unblock deployment — RULE 7 additivity
  failure (this doc, original verdict) still applies. The ETA only
  changes when the *gate-as-overlay* (Path 3) pre-reg can run with
  enough OOS data; gate-level OOS power calculation is different from
  lane-level and would need its own pre-reg-time computation.
- D2's ETA pushes into Q4-2026, ~6 months further than the original
  HANDOFF wording suggested. Plan accordingly: D2 cannot satisfy its
  C8 gate before late Q4-2026 at current accrual rate.

Evidence: read-only invocation of canonical `_load_strategy_outcomes`
on each candidate's locked filter_type, partitioned at
`HOLDOUT_SACRED_FROM = 2026-01-01`. No backtest, no re-spec, no
threshold change.

---

## Files

- `trading_app/config.py` — `OvernightRangeFilter` + `GARCHForecastVolPctFilter` extended with `strict_gt: bool = False`; `OVNRNG_PCT_GT80` + `GARCH_VOL_PCT_GT70` registered in `_HYPOTHESIS_SCOPED_FILTERS`.
- `tests/test_trading_app/test_config.py` — `TestStrictGtVariants` class added (17 tests). `test_total_count` updated 95 → 97 with rationale comment.
- `research/2026-04-29-parked-pathway-b-additivity-audit.py` — read-only audit runner.
- `docs/audit/results/2026-04-29-parked-pathway-b-additivity-triage.md` — this document.
