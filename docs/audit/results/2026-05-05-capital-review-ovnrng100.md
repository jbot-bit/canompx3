---
pooled_finding: false
audit_target: "Capital-class diligence on deployed lane MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100"
auditor_context: opus-4-7-fresh
canonical_layers: [orb_outcomes, daily_features, validated_setups]
db_freshness: "orb_outcomes MNQ max checked at run time — see verbatim query output below"
verdict: "DOWNSIZE"
parent_claims:
  - docs/runtime/stages/capital-review-ovnrng100-post-reconcile.md
  - docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.md
  - docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md
  - docs/audit/results/2026-05-01-target-b-6lane-vestigialness-fresh-audit.md
---

# Capital review — MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100

**Pre-reg:** `docs/audit/hypotheses/2026-05-05-capital-review-ovnrng100.yaml` (locked 2026-05-05T13:00+10:00)
**Stage file:** `docs/runtime/stages/capital-review-ovnrng100.md`
**Canonical DB:** `pipeline.paths.GOLD_DB_PATH`
**Date:** 2026-05-05

## Scope

Capital-class diligence on the deployed lane `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`. Resolves the four findings filed by the placeholder doc merged in PR #234 (commit `d4860bd0`) which paused the capital-review pending a fresh-context Opus session. K=1 confirmatory diligence on an already-validated lane; not new discovery.

The verdict must be one of four LOCKED options traced to numeric pre-reg criteria:
- **REMAIN_DEPLOY** — F1 NOT VESTIGIAL AND F3 DSR ≥ 0.50 AND F4 NOT LIVE_DECAY
- **DOWNSIZE** — F1 VESTIGIAL OR F3 DEPLOYMENT_MATH_BROKEN, AND F4 NOT outright zero/negative
- **UNDEPLOY** — F1 VESTIGIAL AND F3 DEPLOYMENT_MATH_BROKEN AND F4 LIVE_DECAY (zero-or-negative)
- **UNVERIFIED** — Insufficient data

This audit does NOT push allocator state, update `validated_setups`, modify `lane_allocation.json`, or re-derive DSR. Allocator-side action requires a SEPARATE follow-up stage.

## Executive verdict

**DOWNSIZE.** Three of four findings trigger numeric kill criteria. The lane should remain in the allocator at REDUCED capital exposure, not zero capital, because live trailing performance (the only post-deployment-independent evidence stream) is still directionally positive at N=153 even though it is decaying.

| Finding | Numeric reading | Trigger met? |
|---|---|---|
| F1 fire-rate scale stability | 4.3% (2019) → 91.7% (2026), 21.5x drift; avg overnight_range 43 → 199 pts (4.6x) | YES — VESTIGIAL |
| F2 PR #228 reconciliation reframe | runner_IS = strict-unlock CSV exactly (delta 0.0000R) | NO — NOT_A_VIOLATION (pre-reg spec bug, not no-rescue rescue) |
| F3 DSR / MinBTL deployment math | n_trials=35,616, MinBTL_max(E=1)=28, factor=1,272x; dsr_score=0.175; OOS power 0.07 (two-group) / 0.32 (one-sample) | YES — DEPLOYMENT_MATH_BROKEN |
| F4 Live trailing trend | Q2 2025 +0.57R → Q3 2025 +0.29R → Q4 2025 +0.10R → Q1 2026 +0.08R; ratio 0.33 < 0.5 | YES — LIVE_DECAY (but live N=153 ExpR=+0.24R t=2.52 still positive) |

**Verdict trace:** F1 VESTIGIAL OR F3 DEPLOYMENT_MATH_BROKEN are both true; F4 is decay-but-not-zero (live N=153 ExpR=+0.24, t=2.52, still directionally positive). Per the pre-reg verdict taxonomy, this is the DOWNSIZE pattern — Carver 2015 Ch 9-10 "capital allocation under uncertainty" answer when DSR diligence is open and OOS is statistically useless but live signal is directionally positive: reduce capital exposure proportional to remaining uncertainty, do not zero it.

This audit does NOT push allocator state. A separate follow-up stage is required for the actual capital-allocation change.

---

## Lane state at diligence (verbatim from canonical layers)

### `validated_setups` row

```
strategy_id: MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100
expectancy_r: 0.2151
oos_exp_r: 0.2029                             # divergent from canonical +0.1658 (PR #228)
c8_oos_status: None
sample_size: 513
sharpe_ratio: 0.1832
n_trials_at_discovery: 35616
dsr_score: 0.1746951317604647
validation_pathway: family
promotion_provenance: LEGACY                  # no validation_run_id
p_value: 3.3e-05
fdr_adjusted_p: 0.00014
fdr_significant: True
sr0_at_discovery: 0.22480605949934737
win_rate: 0.5185
wfe: 1.107
wfe_verdict: None
sharpe_haircut: -0.0102
max_drawdown_r: 21.9122
trades_per_year: 85.6
max_year_pct: 0.464
era_dependent: False
validation_run_id: None
promoted_at: 2026-04-11 07:23:39.649749+10:00
status: active
```

Source query: `SELECT … FROM validated_setups WHERE strategy_id = 'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100'` against `pipeline.paths.GOLD_DB_PATH`.

### `lane_allocation.json` rebalance 2026-05-03

```
strategy_id: MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100
status: DEPLOY
status_reason: "Session HOT (+0.0625), ExpR=+0.2412, N=150"
chordia_verdict: PASS_CHORDIA
chordia_audit_age_days: 1
trailing_expr: 0.2412   trailing_n: 150   trailing_wr: 0.527
session_regime: HOT
annual_r: 36.2
avg_orb_pts: 28.7   p90_orb_pts: 48.5
months_negative: 0
```

The session-regime gate (`trading_app/lane_allocator.py:427-460` `_compute_session_regime`) is the ONLY allocator gate per `docs/runtime/lane_allocation.json` doctrine — it is by design, derived from the Phase 2.9 backtest (+630R regime-only vs −799R per-strategy pause). Lane-level decay does not pause the lane in the current allocator; it would require either a regime flip on COMEX_SETTLE or an explicit intervention.

---

## F1 — OVNRNG_100 fire-rate scale stability

### Method

Per-year fire rate of `df.overnight_range >= 100` (canonical OVNRNG_100 definition per `trading_app/config.py` and `research/h2_exploitation_audit.py:18, 98`) on the MNQ COMEX_SETTLE E2 RR1.5 CB1 O5 universe. Triple-key join on `(trading_day, symbol, orb_minutes)` per `.claude/rules/daily-features-joins.md`.

```sql
SELECT EXTRACT(YEAR FROM o.trading_day) as yr,
       COUNT(*) as n_universe,
       SUM(CASE WHEN df.overnight_range >= 100 THEN 1 ELSE 0 END) as n_fired,
       100.0 * SUM(CASE WHEN df.overnight_range >= 100 THEN 1 ELSE 0 END) / COUNT(*) as fire_pct,
       AVG(df.overnight_range) as avg_overnight_range,
       AVG(o.pnl_r) FILTER (WHERE df.overnight_range >= 100) as expr_filter,
       AVG(o.pnl_r) FILTER (WHERE df.overnight_range < 100) as expr_nofilter,
       AVG(o.pnl_r) as expr_universe
FROM orb_outcomes o
JOIN daily_features df USING (trading_day, symbol, orb_minutes)
WHERE o.symbol = 'MNQ' AND o.orb_label = 'COMEX_SETTLE' AND o.orb_minutes = 5
  AND o.entry_model = 'E2' AND o.rr_target = 1.5 AND o.confirm_bars = 1
GROUP BY 1 ORDER BY 1;
```

### Verbatim output

```
yr         N  fired   fire%  avg_ovnrng   ExpR_F  ExpR_NF  ExpR_uni
2019     164      7    4.3%       43.2  -0.4327  -0.2508   -0.2585
2020     249    104   41.8%      109.1   0.2613  -0.0906    0.0564
2021     251     66   26.3%       80.6   0.2156  -0.0040    0.0538
2022     250    116   46.4%      111.0   0.0051  -0.0006    0.0020
2023     248     20    8.1%       61.7  -0.0820   0.1595    0.1400
2024     249     73   29.3%       92.3   0.2985   0.0643    0.1329
2025     247    143   57.9%      140.8   0.3580   0.1000    0.2494
2026      72     66   91.7%      199.4   0.1658  -1.0000    0.0687
```

### Reading

- **Fire rate has drifted from 4.3% (2019) to 91.7% (2026)** — a 21.5x change over 7 years.
- **Average overnight range has grown from 43.2 pts (2019) to 199.4 pts (2026)** — a 4.6x scale shift on the underlying.
- The threshold `100` was meaningful on a 43-pt average overnight range (selecting ~4% extreme days). It is meaningless on a 199-pt average overnight range (admitting nearly the entire universe).
- This is the textbook absolute-threshold scale-artifact pattern documented in `feedback_absolute_threshold_scale_audit.md`.

### Sister-lane lift comparison (filtered vs unfiltered universe)

In the years where the filter actually selected a small slice (2019, 2023), filtered ExpR was sometimes WORSE than unfiltered (2019: −0.43 vs −0.26; 2023: −0.08 vs +0.16). In years where the filter admitted ~30-50% of the universe (2020-2022, 2024), filtered ExpR exceeded unfiltered by ~+0.05 to +0.27R — consistent with overnight range being a vol-regime proxy on the in-sample window. In 2026, the filter admits 91.7% and ExpR_filter +0.166 is essentially the unfiltered base rate (ExpR_universe +0.069 is the diluted average; the 8.3% of "off" days have N=6 with ExpR=−1.000, dragging the universe number).

### Verdict on F1

**VESTIGIAL.** Per the pre-reg numeric criterion: `fire_rate_2026 (91.7%) >= 75% AND fire_rate_2019 (4.3%) <= 25%, drift factor 21.5x >= 3x`. **Trigger met.**

This finding is structurally independent of the OOS power-floor diligence (F4) — it is about WHAT the filter selects, not about whether the OOS sample can refute the IS edge. Even if F4 had massive power, F1 would still hold.

This is the same vestigial-filter pattern documented for the 6 OTHER deployed-portfolio lanes in `docs/audit/results/2026-05-01-target-b-6lane-vestigialness-fresh-audit.md` (CLAIM 1: 5/6 lanes ≥75% 2026 fire rate; vestigialness verdict STANDS).

---

## F2 — PR #228 RECONCILED-after-FAIL is a pre-reg specification bug, not a no-rescue violation

### Method

Read the PR #228 result doc § "IS_ExpR caveat" and § "Pre-reg specification correction (post-execution)" verbatim. Verify that the runner's IS_ExpR (+0.2171) reproduces the strict-unlock CSV's IS row (+0.2171) exactly, and that the +0.0020 delta against `validated_setups.expectancy_r` (+0.2151) is a documented scratch-policy difference.

### Reading from PR #228 result doc

> "Acceptance criterion 4 in the pre-reg (`docs/audit/hypotheses/2026-05-05-oos-reconcile-ovnrng100.yaml`) was mis-specified against the wrong target. The criterion compared the runner's scratch-inclusive IS_ExpR against `validated_setups.expectancy_r=+0.2151`, which is computed under a scratch-EXCLUSIVE policy. The correct target for a canonical-machinery runner is the strict-unlock CSV's IS row (+0.2171), which the runner reproduces exactly (delta=0.0000)."

> "Per the institutional rule 'no post-hoc rescue', the pre-reg yaml is left locked as-written. This correction is recorded here, in the result doc, with line citation to the source caveat in the 2026-05-02 strict-unlock result. The runner's FAIL exit code on criterion 4 is therefore expected behavior under a mis-specified target, not a finding of canonical-source drift."

### Reading

The no-rescue rule (`feedback_bias_discipline.md`) prohibits relaxing a kill criterion to admit a finding that would otherwise be killed. PR #228's reframe does NOT relax a kill criterion — the canonical machinery reproduces the strict-unlock reference EXACTLY (delta = 0.0000R), and the pre-reg yaml is left locked. The +0.0020R delta against `validated_setups.expectancy_r` is a documented accounting class difference (scratch-inclusive vs scratch-exclusive) traceable to a single source doc line.

The proper remediation for a future audit class is to ensure pre-regs targeting `validated_setups.expectancy_r` declare which scratch policy that field uses (scratch-EXCLUSIVE per the documented caveat in `docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md` § Caveats). That is a meta-finding for future pre-reg authoring, not evidence of bias on this specific PR.

### Verdict on F2

**NOT_A_VIOLATION.** Per the pre-reg numeric criterion: `delta_runner_minus_strict_unlock_csv = 0.0000R, ≤ 0.001 R tolerance`. **Trigger NOT met for NO_RESCUE_VIOLATION.**

The PR #228 reframe is a correctly-handled pre-reg specification bug, not a post-hoc rescue.

---

## F3 — DSR / MinBTL deployment math under canonical OOS

### MinBTL bound (Bailey 2013 Theorem 1)

Per `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` p.8 Theorem 1:

```
MinBTL ≈ 2·ln(N) / E[max_N]²
```

For 6.65 years of clean MNQ data (post-Phase-3c, per Amendment 2.8 in `pre_registered_criteria.md` Criterion 2) at strict E[max_N] = 1.0, the bound implies **N_max = 28 pre-registered trials**.

The lane carries `n_trials_at_discovery = 35,616`. **MinBTL violation factor = 35,616 / 28 = 1,272x.**

This is the explicit, named violation in MEMORY.md ("brute-force discovery ~35,000 trials on 2.2-6 years of clean data" — note the post-Phase-3c clean-data correction expands the horizon from 2.2 to 6.65 years, but the violation factor remains in the order of magnitude of 1,000x even at the most generous E[max_N]=1.5 cap of N≤1,774).

The Bailey 2013 extract literature anchor (`docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md:110`) explicitly names this lane:

> "Only COMEX OVNRNG_100 clears this bar."

— at the GENEROUS 16-year proxy-extended horizon assuming 500 effective independent trials. At the actual 6.65-year clean-data horizon and 35,616 trials, the noise floor for trustworthy IS Sharpe is ~3.24 (Bailey 2013 extract Table). The lane reports `sharpe_ratio = 0.1832` (per-trade) which translates to ~1.23 annualized — **below the Bailey 6.65-year noise floor** at this trial count.

### DSR (Bailey-LdP 2014 Equation 2)

`validated_setups.dsr_score = 0.1746951317604647`. Per `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` Equation 2 and `docs/institutional/pre_registered_criteria.md` Criterion 5, the threshold for "legitimate empirical discovery at 95% confidence" is **DSR > 0.95**. The lane reports 0.175 — there is only a 17.5% chance the true Sharpe is greater than zero after deflation for selection bias and non-Normality.

### OOS power (Harvey-Liu 2015 + RULE 3.3)

IS recovered standard deviation from PR #228:
```
IS_mean = +0.2171   IS_t = +4.256   IS_N = 522
IS_sd = |IS_mean| · sqrt(IS_N) / |IS_t| = 1.1654
Cohen's d = IS_mean / IS_sd = 0.1863
```

OOS power per `research/oos_power.py` canonical helper:

```
=== OOS power (one-sample, alpha=0.05 two-sided) ===
Cohen d (IS) = 0.1863
NCP = d * sqrt(N_OOS) = 1.5133
df = N_OOS - 1 = 65
Power to detect IS effect: 0.3198
RULE 3.3 tier: STATISTICALLY_USELESS

=== Two-group split (filter on=66, off=6) ===
Cohen's d (IS effect): 0.186
Expected OOS SE:       0.4969
Expected 95% CI half-width: 0.9740
Power at alpha=0.05 two-sided: 7.2%
N per group for 80% power: 454
RULE 3.3 tier: STATISTICALLY_USELESS
```

Both the one-sample and two-group framings produce STATISTICALLY_USELESS tier. The OOS sample cannot refute or confirm the IS edge.

### Reading

The lane meets all three triggers in the F3 numeric criterion:

- MinBTL violation factor: **1,272x** (» 100x threshold)
- DSR: **0.175** (« 0.50 threshold)
- OOS power: **0.07 two-group / 0.32 one-sample** (« 0.50 threshold)

The deployment math at promotion (April 2026) was implicitly conditioned on the LEGACY OOS_ExpR of +0.2029 (not the canonical +0.1658, which differs by 0.0371R per trade — about 18% of the headline OOS expectancy). At ~85 trades/year, that is +3.15 R/year of inflation — material at the per-lane scale.

### Verdict on F3

**DEPLOYMENT_MATH_BROKEN.** All three numeric triggers met. The lane's promotion-era statistics do not survive Phase 0 institutional gates (Criterion 2 MinBTL, Criterion 5 DSR, RULE 3.3 power floor). This is a known, pre-existing finding inherited from MEMORY.md, formally re-anchored to literature here.

---

## F4 — Live trailing performance N=153 directionality

### Method

Per-quarter ExpR breakdown for the lane's last 18 months of trades. Triple-key join on `(trading_day, symbol, orb_minutes)`. Scratch policy: `pnl_r NULL → 0.0` per pre-reg `scratch_policy: include-as-zero`.

```sql
WITH fired AS (
  SELECT o.trading_day,
         COALESCE(o.pnl_r, 0.0) as r,
         DATE_TRUNC('quarter', o.trading_day) as q
  FROM orb_outcomes o
  JOIN daily_features df USING (trading_day, symbol, orb_minutes)
  WHERE o.symbol = 'MNQ' AND o.orb_label = 'COMEX_SETTLE' AND o.orb_minutes = 5
    AND o.entry_model = 'E2' AND o.rr_target = 1.5 AND o.confirm_bars = 1
    AND df.overnight_range >= 100
    AND o.trading_day >= '2024-10-01'
)
SELECT q, COUNT(*) as n, AVG(r) as expr, STDDEV(r) as sd
FROM fired GROUP BY q ORDER BY q;
```

### Verbatim output

```
quarter          N     ExpR      sd
2024-10-01      21  +0.2406   1.139
2025-01-01      36  +0.3918   1.194
2025-04-01      50  +0.5677   1.138
2025-07-01      16  +0.2891   1.176
2025-10-01      41  +0.0994   1.199
2026-01-01      58  +0.0792   1.163
2026-04-01       8  +0.7937   1.108
```

### Live N=153 (rolling 365-day) summary

```
N=153  ExpR=+0.2388  sd=1.1712  WR=0.529  cum_R=36.53  t=2.522
```

### RULE 3.3 power on live N=153

```
Power to detect IS d=0.186 at N=153: 0.629
Power to detect live d=0.204 at N=153: 0.708
Tier: DIRECTIONAL_ONLY
```

### Reading

- The trailing 12-month +0.2412R headline that justifies session-regime DEPLOY status is structurally weighted by the strong contribution of Q1-Q3 2025 (+1.25R cumulative across 102 trades) and disguises a clear monotonic decay since Q3 2025.
- Q1 2026 N=58 ExpR=+0.0792 is a 5.0R contribution — the slowest quarter on record.
- Q2 2026 partial (April only, N=8 +0.79R) is too small to read as recovery vs noise.
- At live N=153, power to detect the IS effect is 63% (DIRECTIONAL_ONLY tier per RULE 3.3). The +0.24R live ExpR is informational, not confirmatory.
- t = 2.52 on N=153 fails Chordia-strict t ≥ 3.79 (no-theory Pathway A) and is below the t ≥ 3.0 with-theory Pathway B threshold.
- Q1 2026 (+0.0792) / 12-month-mean (+0.2388) = 0.33 < 0.5 trigger threshold.

### Verdict on F4

**LIVE_DECAY (mild).** Per the pre-reg numeric criterion: monotonic decay across Q2 2025 → Q1 2026 (4 consecutive declining quarters: +0.57 → +0.29 → +0.10 → +0.08), AND ratio Q1-2026/12mo-mean = 0.33 < 0.5. **Trigger met.**

But the lane is NOT in zero-or-negative territory — live N=153 ExpR is still +0.24R t=2.52 (Pathway B-pass at relaxed t ≥ 2.0 threshold but below institutional t ≥ 3.0). The decay-but-not-zero pattern is exactly the canary signal Carver 2015 Ch 9-10 describes for "capital allocation under uncertainty" — keep the lane in the portfolio at REDUCED exposure, do not zero it.

---

## Verdict synthesis

| Trigger pattern | Verdict |
|---|---|
| F1 NOT VESTIGIAL AND F3 DSR ≥ 0.50 AND F4 NOT LIVE_DECAY | REMAIN_DEPLOY |
| F1 VESTIGIAL OR F3 DEPLOYMENT_MATH_BROKEN, AND F4 NOT outright LIVE_DECAY (live N≥100, positive ExpR) | **DOWNSIZE** |
| F1 VESTIGIAL AND F3 DEPLOYMENT_MATH_BROKEN AND F4 LIVE_DECAY (zero-or-negative live) | UNDEPLOY |
| Insufficient data | UNVERIFIED |

**Observed pattern:** F1 = VESTIGIAL ✓, F2 = NOT_A_VIOLATION (no rescue), F3 = DEPLOYMENT_MATH_BROKEN ✓, F4 = LIVE_DECAY (mild, live N=153 still +0.24R t=2.52).

**Verdict: DOWNSIZE.**

The lane's promotion-era statistics do not survive Phase 0 institutional gates AND the filter is vestigial under current scale, BUT live trailing performance is still directionally positive at N=153, even though decaying. Carver 2015 Ch 9-10 capital-allocation-under-uncertainty answer: reduce exposure proportional to remaining uncertainty rather than zero it.

A separate follow-up stage is required for the actual capital-allocation change. THIS audit does not push allocator state. Recommended downstream artifacts:

1. **Allocator-side stage** that reduces this lane's per-rebalance contract count proportional to the F3 DSR confidence (e.g., 0.175 / 0.95 ≈ 0.18 → roughly one-fifth of current size, or one contract minimum if profile already at 1).
2. **F4 monitor stage** that adds a per-lane decay tripwire to the rebalance code: pause if Q-over-Q ratio < 0.33 for two consecutive quarters AND live N ≥ 100.
3. **F1 generalization sweep**: the same scale-drift class affects every absolute-points-threshold filter in the registry. Open a separate stage to enumerate them and check fire-rate stability.

These are SEPARATE stages, not part of this verdict's scope.

---

## What this audit does NOT close

- **Filter replacement.** The vestigial-filter finding does not propose a relative-threshold replacement (e.g., `overnight_range >= prev_atr_20 * X`). That would be new discovery requiring its own pathway-A pre-reg. Out of scope.
- **DSR re-derivation methodology.** The Bailey-LdP standalone audit (compute DSR for every deployed lane against a representative `V[{ŜR_n}]` and `N̂` correlation-adjusted) is queued but out of scope per PR #228 result doc.
- **Allocator capital-rule update.** The current allocator gates only on session regime; lane-level decay does not pause. That is by design (Phase 2.9 backtest) and a doctrine change requires its own stage.
- **`validated_setups.oos_exp_r` row update.** Updating from the LEGACY +0.2029 to the canonical +0.1658 requires the strategy_validator path with its own pre-reg.

---

## Caveats and limitations

- **Capital-allocation under uncertainty is itself uncertain.** Carver 2015 Ch 9-10 provides a framework, not a closed-form sizing rule. The exact downsize ratio is a Stage-2 decision — this audit only declares the directional verdict.
- **Live N=153 power is DIRECTIONAL_ONLY (63%).** The decay reading itself has structural noise — a single strong quarter (Q2 2026, currently N=8 +0.79R) could shift the directional read.
- **Session-regime gate is the only allocator pause path.** This audit recommends a downsize, not a pause; if the user wants a pause, that requires either explicit intervention or a separate stage to add lane-level decay logic to the allocator.
- **Trade-time knowability.** OVNRNG_100 has no look-ahead bias on COMEX_SETTLE (overnight window 09:00-17:00 Brisbane closes well before COMEX_SETTLE 04:30 next-day Brisbane, per `.claude/rules/backtesting-methodology.md` RULE 1.2). The vestigial finding is structural-drift, not a look-ahead problem.
- **No downstream effects on other deployed lanes.** This audit covers ONE lane. The other 2 active lanes (MNQ_US_DATA_1000_VWAP_MID_ALIGNED, MNQ_NYSE_OPEN_COST_LT12) are out of scope.

---

## What this audit does NOT disconfirm

- It does NOT disconfirm the IS Chordia t=+4.256 strict-unlock pass. That stands on canonical machinery.
- It does NOT disconfirm that the lane has historically generated capital — the trailing 12-month +36.53R is real cumulative performance.
- It does NOT propose immediate undeployment. The DOWNSIZE verdict explicitly preserves the lane in the portfolio at reduced exposure.

---

## Reproduction

The four canonical-layer queries above can be re-run directly against `pipeline.paths.GOLD_DB_PATH`. The OOS power calculation uses `research/oos_power.py` (one-sample variant via scipy.stats.nct, two-group via canonical helper). All numeric values in this doc are reproducible from canonical layers + the deployed `validated_setups` row + the live `lane_allocation.json` rebalance state at `rebalance_date=2026-05-03`.

## Provenance

- Pre-reg: `docs/audit/hypotheses/2026-05-05-capital-review-ovnrng100.yaml` (locked 2026-05-05T13:00+10:00)
- Stage file: `docs/runtime/stages/capital-review-ovnrng100.md`
- Parent placeholder doc: PR #234 commit `d4860bd0` (merged 2026-05-05)
- OOS reconciliation: PR #228 commit `120882f1` (merged 2026-05-05)
- Strict-unlock prior: `docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md`
- Vestigialness pattern reference: `docs/audit/results/2026-05-01-target-b-6lane-vestigialness-fresh-audit.md`
- Bailey 2013 lane-named extract: `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md:110`

## Memory anchors

- `MEMORY.md` § Validated signals (PR #228 provenance)
- `feedback_oos_power_floor.md`
- `feedback_absolute_threshold_scale_audit.md`
- `feedback_pooled_not_lane_specific.md`
- `feedback_per_lane_breakdown_required.md`
- `feedback_audit_thread_dead_end_mine_canonical.md`
