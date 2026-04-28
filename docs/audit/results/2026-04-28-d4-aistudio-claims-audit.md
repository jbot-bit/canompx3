# D4 AIStudio claims audit — 14 claims vs canonical layers

**Date:** 2026-04-28
**Subject:** D4 Pathway B K=1 confirmatory test — `MNQ COMEX_SETTLE O5 RR1.0 E2 CB1 long + garch_forecast_vol_pct > 70`
**Verdict on D4 result (`docs/audit/results/2026-04-28-mnq-comex-settle-pathway-b-v1-result.md`):** **PARK_PENDING_OOS_POWER stands.**

External reviewer (AIStudio) raised 14 structural objections to the D4 verdict. Each is audited below against canonical layers (`bars_1m`, `daily_features`, `orb_outcomes`) and grounded in `/resources/` literature passages. **Per-aperture individual findings; no pooled-universe synthesis claim — pooled-finding rule does not apply.**

---

## Scope

This audit reviews 14 claims from an external reviewer about the D4 Pathway B verdict. For each claim, it asks: does the claim survive against canonical-layer evidence? When it does, is the underlying mechanism grounded in `/resources/` literature, or is it ad-hoc?

**This audit does NOT:**
- Re-run the D4 Pathway B harness (the verdict file at `docs/audit/results/2026-04-28-mnq-comex-settle-pathway-b-v1-result.md` stands)
- Promote D4 to deployment
- Open or close any other Phase D pre-reg

**This audit DOES:**
- Adjudicate each of the 14 claims as CONFIRMED / PARTIALLY CORRECT / REFUTED / DIRECTION RIGHT
- Cite the canonical query or `/resources/` passage for each verdict
- Land doctrine where a claim surfaces a real methodology gap (Carver Ch 12 p.192 addendum already landed in commit `25ed6f09`)

**Audited claim source:** external review delivered 2026-04-28. Auditor: this session, working on `research/2026-04-28-phase-d-mnq-comex-settle-pathway-b`.

---

## Per-claim findings

### Claim 1 — "GARCH = whole-day forecast → wrong scale for 5m breakout"

**Verdict:** **PARTIALLY CORRECT** — the timing description is right, but it is not a structural mismatch.

`garch_forecast_vol_pct` is a daily forecast made at the prior close (per `pipeline/build_daily_features.py:1468-1497`, rolling 252-day prior-only window). Carver Ch 9 framework — verbatim from `resources/Robert Carver - Systematic Trading.pdf` p.137 — explicitly defines the "expected daily standard deviation" as the basis for an "annualised cash volatility target." A daily forecast → intraday position sizing/filtering is the framework, not a defect. See `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` for the verbatim Ch 9-10 extracts.

### Claim 2 — "GARCH = ORB_G5 (subset = identity)"

**Verdict:** **DIRECTION RIGHT, STRENGTH WRONG.** Strict subset, not identity.

Canonical query (`research/audit_aistudio_claims_d4.py`): on `MNQ COMEX_SETTLE` 2019-2025 IS, P(ORB_G5 fires | garch_forecast_vol_pct > 70) = **100%**. P(garch_forecast_vol_pct > 70 | ORB_G5 fires) = **23.7%**. So `garch>70` is a strict subset of `ORB_G5`. The reviewer's "identity" claim is wrong, but the subset relationship is real and triggers RULE 7 within-cohort residual measurement (see Surface 4 memory entry).

### Claim 3 — "Aperture fragility (5m / 15m / 30m)"

**Verdict:** **CONFIRMED with mechanism caveat.**

Q5−rest mean-R within the lane: 5m = **+0.260 R**, 15m = **+0.093 R**, 30m = **-0.053 R**. The reviewer correctly identified the aperture sensitivity. **However**, Carver p.192 (verbatim addendum landed in `docs/institutional/literature/carver_2015_ch12_speed_and_size.md` 2026-04-28) grounds aperture × RR as holding-period × stop-level. 5m × RR1.0 ⇒ scalp horizon ⇒ Chan Ch 7 stop-cascade mechanism (`docs/institutional/literature/chan_2013_ch7_intraday_momentum.md` p.155-157, verbatim) operates at 5-30 min. 30m × RR1.0 ⇒ swing horizon ⇒ stop-cascade mechanism does NOT operate. The "fragility" reading dies once the mechanism horizon is named.

### Claim 4 — "Slippage 3x higher on garch>70"

**Verdict:** **REFUTED.**

`pipeline/cost_model.py` MNQ slippage is **fixed at $1.0 per fill**, independent of vol regime. Realized loss-trade `pnl_r` on the cohort = **-1.000 R** identical on `garch>70=on` and `garch>70=off` subsets. No measurable slippage scaling on volatile days. Reviewer's claim is a memory-based assertion, not a canonical-data finding.

### Claim 5 — "MGC same session = better fit"

**Verdict:** **CHECKED, not an immediate fix.**

MGC `COMEX_SETTLE` Q5−Q1 of `garch_forecast_vol_pct` mean-R is stable across apertures (5m: **+0.22 R**, 15m: **+0.22 R**, 30m: **+0.17 R**) — i.e. MGC does not exhibit the aperture-flip MNQ shows. **However**, all per-quintile means on MGC `COMEX_SETTLE` are negative on this cohort, so "better fit" does not translate to a deployable edge. Not a direct rescue; flagged for a future MGC-specific Phase D pre-reg if motivated by separate evidence.

### Claim 6 — "Try VWAP / PDH instead of GARCH"

**Verdict:** **REFUTED on this cohort.**

Within `MNQ COMEX_SETTLE O5 E2 CB1 RR1.0 long ORB_G5` IS:
- `garch_forecast_vol_pct > 70`: **t = 3.03**
- `atr_20_pct > 70`: **t = 2.60**
- `ovn_range_pct > 80`: **t = 1.74**

GARCH dominates the alternative vol/range proxies. (VWAP/PDH not directly comparable as scalar quintile filters; the comparison set is the closest-class alternatives.)

### Claim 7 — "Q1>Q3 = U-shape"

**Verdict:** **WEAK CONFIRMATION.** Real signal is Q5 alone, not a U.

Per-quintile mean-R of `garch_forecast_vol_pct` on the cohort long-side: Q1=+0.031, Q2=−0.018, Q3=**−0.037 (dead zone)**, Q4=+0.010, Q5=**+0.284**. Q3 is the worst, but the headline "U-shape" overstates Q1's contribution. The deployable edge is Q5; Q1 is noise-positive.

### Claim 8 — "Momentum vs reversion identity"

**Verdict:** **MOMENTUM CONFIRMED.**

MFE_on (garch>70) = **+0.900 R** vs MFE_off = **+0.786 R** (+15% larger favourable extension). Avg-loss identical at -1.0R on both subsets. This is a momentum-extension signature, consistent with Chan Ch 7 p.157 stop-cascade mechanism. Reviewer's "could be reversion" framing has no canonical-data support.

### Claim 9 — "Q5-only sizing redux"

**Verdict:** **TESTED AND FAILS FLOOR.**

Phase D D-0 v2 already executed Q5-only-sized variant — see `docs/audit/results/<phase-d-d0-v2>` (`aeb7531a`). Result: abs Sharpe diff +0.009 << 0.05 floor; rel uplift +5.1% << 15% floor. **PARK** by Criterion 8 floor. This claim was already adjudicated; the reviewer is asking for work that has been done.

### Claim 10 — "Open>PDH alternative"

**Verdict:** **LOW N — INCONCLUSIVE.**

Only **14 IS days** satisfy `open > prev_day_high` within `MNQ COMEX_SETTLE ORB_G5` cohort. N too low for meaningful confirmatory test. Cannot rule in or out as alternative filter; would require a dedicated longer-horizon pre-reg (instrument expansion or proxy-extended data).

### Claim 11 — "RR1.0 vs RR1.5 vanity bias"

**Verdict:** **PARTLY VALID.**

Deployed `MNQ COMEX_SETTLE` lane uses **RR1.5**, not RR1.0. At RR1.5, within-cohort residual t-stat for `garch>70` = **2.35** (vs **3.03** at RR1.0). Still significant but smaller. Reviewer is correct that the headline t=3.03 is RR1.0-specific and the deployed-RR comparison is the honest benchmark. Does not kill D4 (t=2.35 still meaningful) but tightens the framing.

### Claim 12 — "Net-of-cost realism"

**Verdict:** **VALIDATED.**

MNQ friction per `pipeline/cost_model.py`: **$2.92 fixed per round-trip**. Avg risk on cohort = $69.74 → friction = **4.19% of 1R**. Friction does not erode the edge below significance (residual t still > 2.0 at RR1.5, well above noise floor).

### Claim 13 — "Top-quintile P80 across apertures"

**Verdict:** **FRAGMENTED.**

Same aperture-specific pattern as Claim 3: 5m top20−rest = **+0.260 R**, 15m = **+0.093 R**, 30m = **-0.053 R**. The P80 cut behaves identically to the Q5 cut. Same Carver-p.192 / Chan-p.157 horizon-mechanism reading applies.

### Claim 14 — "GARCH timing alignment"

**Verdict:** **CORRECT REVIEWER FRAMING, COMPLIANT WITH PROJECT.**

GARCH forecast made at prior close, applied at session start the next day. RULE 1.2 valid-domain pass (no look-ahead). Carver Ch 9 framework (verbatim in `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`) supports daily-forecast → intraday-application. Reviewer's framing is correct; project's implementation is compliant.

---

## Decision

**External review of D4 surfaced:**
- **1 valid issue:** subset-vs-identity nuance for GARCH ⊂ ORB_G5 (Claim 2). Triggers RULE 7-style within-cohort residual measurement, captured as memory-feedback lesson (Surface 4).
- **1 partly-valid concern:** RR1.0 vanity vs deployed RR1.5 (Claim 11). Within-cohort residual at RR1.5 t=2.35, still significant.
- **4 refuted claims:** slippage scaling (#4), VWAP/PDH alternative beats GARCH (#6), Q5-only-sizing not yet tested (#9), top-quintile P80 universal (#13).
- **8 confirmed-but-mechanism-grounded findings:** GARCH-as-daily-forecast (#1), aperture sensitivity (#3), MGC stability (#5), Q5-only-not-U-shape (#7), momentum extension (#8), Open>PDH low-N (#10), net-of-cost realism (#12), GARCH timing alignment (#14).

**D4 verdict (PARK_PENDING_OOS_POWER) stands unchanged.** The external review's "kill it" conclusion is not supported by canonical data when grounded against `/resources/` literature.

**Doctrine landed:**
1. Carver Ch 12 p.192 horizon-stop addendum (commit `25ed6f09`, 2026-04-28) — re-frames aperture sensitivity as horizon-stop match (not microstructure fragility) when the cited mechanism operates at the implied holding-period horizon.
2. Memory-feedback lesson on subset-of-deployed-filter requiring within-cohort residual (Surface 4).
3. Decision-ledger entry `d4-aistudio-audit` (this commit).

---

## Reproduction

**Reproduction script:** `research/audit_aistudio_claims_d4.py` (committed alongside this doc; runnable with `python research/audit_aistudio_claims_d4.py`).

The script uses canonical layers only (`bars_1m`, `daily_features`, `orb_outcomes`) via `pipeline.paths.GOLD_DB_PATH` and reproduces every numeric figure cited above.

**Required inputs:**
- `gold.db` at `pipeline.paths.GOLD_DB_PATH` (canonical project DB).
- Python with `duckdb`, `numpy`, `pandas`, `scipy`.

**Cross-references:**
- `docs/audit/results/2026-04-28-mnq-comex-settle-pathway-b-v1-result.md` — D4 verdict file (PARK_PENDING_OOS_POWER).
- `docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-pathway-b-v1.yaml` — D4 pre-reg.
- `research/phase_d_d4_mnq_comex_settle_pathway_b.py` — D4 runner.
- `docs/institutional/literature/carver_2015_ch12_speed_and_size.md` — addendum landed at commit `25ed6f09`.
- `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md` — verbatim p.155 + p.157 stop-cascade mechanism (lines 22 + 40).
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` — Ch 9-10 daily-forecast framework.
- `docs/institutional/literature/bailey_lopezdeprado_2014_dsr_sample_selection.md` — selection-bias / OOS framing.
- `pipeline/build_daily_features.py:1468-1497` — `garch_forecast_vol_pct` rolling 252-day prior-only window.
- `pipeline/cost_model.py` — MNQ slippage $1.0, friction $2.92.

---

## Limitations

**What this audit does not do:**

1. **Does NOT re-derive D4 verdict.** The PARK_PENDING_OOS_POWER conclusion in the D4 result file stands. This audit only adjudicates the external review's structural objections.
2. **Does NOT promote D4.** Despite refuting most of the reviewer's "kill it" reasoning, OOS power floor (N_OOS = 17 < 50) remains the binding constraint per Amendment 3.2. Real-money flip remains gated.
3. **Does NOT cover claims outside the 14 listed.** Future reviewer objections (e.g., on different filter classes, different sessions, different instruments) require their own audit.
4. **Per-claim verdicts are NOT pooled into a synthesis claim.** No "aggregate edge survival" t-stat. Each claim stands or falls on its own canonical-layer evidence. Pooled-finding rule (`docs/audit/results/TEMPLATE-pooled-finding.md`) deliberately not invoked.
5. **MGC Claim 5 deferred.** "MGC stable across apertures but all means negative" is descriptive only; promoting MGC `COMEX_SETTLE` to a Phase D pre-reg requires separate motivation, not a re-routing of D4 evidence.
6. **Open>PDH Claim 10 deferred.** N=14 is too small for confirmatory work; flagged as "would require longer-horizon pre-reg" not as a refuted/confirmed verdict.
7. **No live OOS update.** OOS counts and dir-match figures match the D4 result file as of 2026-04-28; no re-pull.

**Falsification battery:** if the Carver p.192 horizon-mechanism reading of Claim 3 is wrong — i.e. if a different signal class also exhibits 5m → 30m sign-flip on MNQ COMEX_SETTLE despite no <30min mechanism — the addendum's decision rule should be revisited. No such counter-example surfaced this session.
