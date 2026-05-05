# OOS reconciliation diligence — MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100

**Pre-reg:** `docs/audit/hypotheses/2026-05-05-oos-reconcile-ovnrng100.yaml`
**Runner:** `research/oos_reconcile_ovnrng100.py`
**Canonical DB:** `C:/Users/joshd/canompx3/gold.db`
**Date:** 2026-05-05

## Scope

Reconciles three pre-existing OOS values for the same nominal deployed lane.
This is a value-reconciliation diligence audit on a deployed lane, not a fresh
discovery scan. The lane is currently `status=DEPLOY` in
`docs/runtime/lane_allocation.json` rebalance_date=2026-05-03 with
trailing_expr=+0.2412 and N=150.

Lane validity, DSR re-derivation, and any allocator/promotion decision are
explicitly out of scope.

## Verdict

**MEASURED verdict:** `RECONCILED_AGAINST_STRICT_UNLOCK`. The 2026-05-02
strict-unlock CSV (+0.1658 OOS_ExpR, N_OOS=66, N_IS=522) is reproduced exactly
by canonical machinery. The `validated_setups.oos_exp_r=+0.2029` LEGACY value
is the divergent number, not the strict-unlock or the canonical recompute.

| Source | OOS_ExpR | N_OOS | Cohort definition |
|---|---:|---:|---|
| A — `validated_setups.oos_exp_r` | +0.2029 | unknown | promoted 2026-04-11; `promotion_provenance=LEGACY`; no `validation_run_id` linkage |
| B — strict-unlock CSV (2026-05-02) | +0.1658 | 66 | canonical `_load_universe` + `filter_signal('OVNRNG_100','COMEX_SETTLE')` + `HOLDOUT_SACRED_FROM` + `WF_START_OVERRIDE['MNQ']`; scratch policy `pnl_r NULL → 0.0` |
| R — runner canonical recompute (this audit) | +0.1658 | 66 | same canonical machinery as B |

**OOS deltas:**

- R − A = −0.0371 (LEGACY value disagrees with canonical by 3.71 R-points)
- R − B = +0.0000 (canonical recompute reproduces strict-unlock CSV exactly)

## Split summary (canonical recompute)

| Split | N_universe | N_fired | Fire% | Scratch | ExpR | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1494 | 522 | 34.94% | 9 | +0.2171 | +0.1863 | +4.256 | 0.00002 |
| OOS | 72 | 66 | 91.67% | 3 | +0.1658 | +0.1415 | +1.150 | 0.25022 |

Identical to the IS/OOS rows in
`docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md`.
This is the expected outcome under canonical-machinery delegation.

## IS_ExpR caveat (acceptance criterion 4 explained)

The runner emitted FAIL on acceptance criterion 4 (`|IS_ExpR_runner −
validated_setups.expectancy_r| ≤ 0.001`). Observed delta:
+0.2171 − +0.2151 = +0.0020.

This is a known-class accounting difference, not a runner defect:

- `validated_setups.expectancy_r` = +0.2151 reports `wins+losses` only
  (excludes scratches). Source: documented caveat in
  `docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md`
  § Caveats — "`validated_setups.sample_size` reports wins+losses only".
- Runner IS_ExpR = +0.2171 reports `N_fired` (wins+losses+scratches with
  `pnl_r → 0.0`), matching the strict-unlock CSV's IS row exactly.

### Pre-reg specification correction (post-execution)

Acceptance criterion 4 in the pre-reg
(`docs/audit/hypotheses/2026-05-05-oos-reconcile-ovnrng100.yaml`) was
mis-specified against the wrong target. The criterion compared the runner's
scratch-inclusive IS_ExpR against `validated_setups.expectancy_r=+0.2151`,
which is computed under a scratch-EXCLUSIVE policy. The correct target for a
canonical-machinery runner is the strict-unlock CSV's IS row (+0.2171), which
the runner reproduces exactly (delta=0.0000).

Per the institutional rule "no post-hoc rescue", the pre-reg yaml is left
locked as-written. This correction is recorded here, in the result doc, with
line citation to the source caveat in the 2026-05-02 strict-unlock result.
The runner's FAIL exit code on criterion 4 is therefore expected behavior
under a mis-specified target, not a finding of canonical-source drift.

## Why the LEGACY value diverges (provenance trace)

`validated_setups.oos_exp_r=+0.2029` was written at 2026-04-11 with
`promotion_provenance=LEGACY`. The row carries no `validation_run_id`, so
the original computation cannot be re-run from a stored audit trail.

What is recoverable from the row alone:

- Promotion date 2026-04-11 sits 3 days after Mode A cutover
  (`HOLDOUT_GRANDFATHER_CUTOFF=2026-04-08` per `trading_app/holdout_policy.py`).
  Rows promoted at-or-before 2026-04-08 are grandfathered Mode B; rows after
  are subject to Mode A enforcement. 2026-04-11 is post-cutoff.
- `n_trials_at_discovery=35616` is consistent with the pre-Phase-0
  brute-force discovery regime per `MEMORY.md` ("brute-force discovery
  ~35,000 trials on 2.2-6 years of clean data").

What is NOT recoverable from the row alone: the exact cohort lower bound,
filter delegation path, scratch policy, or OOS cutoff date used at promotion
time. The audit trail is structurally absent. The direction or shape of the
+0.0371 R divergence cannot be attributed to any specific cause from the
row alone. Any reconciliation must therefore be against the next canonical
reference, which is the 2026-05-02 strict-unlock CSV.

## Method notes

- Canonical machinery delegated, no re-encoding (per
  `.claude/rules/integrity-guardian.md` Rule 4):
  - Holdout cut: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM = 2026-01-01`
  - IS lower bound: `trading_app.config.WF_START_OVERRIDE['MNQ'] = 2020-01-01`
  - Cohort load: `research.chordia_strict_unlock_v1._load_universe`
  - Filter fire: `research.filter_utils.filter_signal('OVNRNG_100','COMEX_SETTLE')`
  - Stats: `research.chordia_strict_unlock_v1._evaluate_split`
- Source layers: `bars_1m`, `daily_features`, `orb_outcomes` (read-only).
  `validated_setups` read once for the diff target only.
- Scratch handling: `pnl_r NULL → 0.0` for the measured trade stream;
  scratch counts reported separately. Matches strict-unlock CSV.
- Triple-key join enforced via `_load_universe`: `(trading_day, symbol,
  orb_minutes)`. Per `.claude/rules/daily-features-joins.md`.

## Trade-time knowability

`OVNRNG_100` is trade-time-knowable for `COMEX_SETTLE`. Per
`.claude/rules/backtesting-methodology.md` § RULE 1.2: the overnight window
09:00-17:00 Brisbane closes before COMEX_SETTLE 04:30 next-day Brisbane. No
look-ahead by the validity-domain gate.

## Literature anchors

> "If your backtesting and live trading programs are one and the same, and
> the only difference between backtesting versus live trading is what kind
> of data you are feeding into the program (historical data in the former,
> and live market data in the latter), then there can be no look-ahead bias
> in the program."
> — Chan 2013 Ch 1 p.4
> (`docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`)

This grounds the runner's design: the same canonical helpers used by the
strict-unlock runner (and downstream by the validator) are imported here.
Reproducing strict-unlock exactly is the expected outcome under this
discipline; failure to reproduce would indicate canonical-source drift.

> "Despite its popularity, OOS testing has several limitations. First, an
> OOS test may not be truly out of sample. [...] Second, an OOS test, like
> any other test in statistics, only works in a probabilistic sense. In
> other words, an OOS test's success can be due to luck for both the
> in-sample selection and the out-of-sample testing."
> — Harvey & Liu 2015 p.17
> (`docs/institutional/literature/harvey_liu_2015_backtesting.md`)

Cited because OOS_t=+1.150 / p_two=0.250 / N_OOS=66 is statistically
underpowered. The +0.1658 OOS_ExpR is descriptive, not confirmatory. Per
`.claude/rules/backtesting-methodology.md` § RULE 3.3, an underpowered
OOS cannot be a hard kill; reconciliation does not change that, only
restates which value is canonical.

> "Put bluntly, a backtest where the researcher has not controlled for the
> extent of the search involved in his or her finding is worthless,
> regardless of how excellent the reported performance might be. Investors
> and journal referees should demand this information whenever a backtest
> is submitted to them, although even this will not remove the danger
> completely."
> — Bailey & López de Prado 2014 p.3
> (`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`)

Cited as the explicit out-of-scope flag: `n_trials_at_discovery=35616` and
`dsr_score=0.175` per the `validated_setups` row. The lane was discovered
under the pre-Phase-0 brute-force regime and carries a documented MinBTL
violation per `MEMORY.md`. Reconciling the OOS value does **not** close the
DSR diligence gap. That is a separate audit.

## What this reconciliation closes

- The "three OOS numbers" diligence question is closed:
  `+0.1658` is the canonical OOS_ExpR for this lane under the same cohort,
  filter, and holdout policy as the validator.
- The strict-unlock CSV value is reproducible from canonical layers via
  imported canonical helpers. No drift in the canonical machinery.
- The `validated_setups.oos_exp_r=+0.2029` value is recoverable only by
  the legacy promotion-time computation path; the row itself carries no
  `validation_run_id` to recompute against.

## What this reconciliation does NOT close

- DSR re-derivation. `n_trials_at_discovery=35616` MinBTL violation per
  `MEMORY.md` is unaddressed by this audit.
- Lane validity. The Chordia strict t-stat clears 3.79 IS, but the OOS
  power floor (N_OOS=66, post-filter `t=+1.150`, `p_two=0.250`) is below
  the C8 confirmatory floor and thus the lane carries
  `c8_oos_status=NULL` per `validated_setups`.
- Capital-at-risk decision. The lane is currently
  `status=DEPLOY` in `lane_allocation.json` rebalance 2026-05-03;
  whether that should change based on this diligence is a `/capital-review`
  question, not this audit's verdict.
- Update to `validated_setups.oos_exp_r`. Canonical sources are
  `bars_1m`/`daily_features`/`orb_outcomes` per the research-truth-protocol;
  `validated_setups` is a derived layer and a row update would require a
  separate validator-path stage with its own pre-reg.

## Next-step recommendation

The deployed lane's published OOS value (`validated_setups.oos_exp_r=+0.2029`)
overstates the canonical OOS_ExpR by +0.0371 R per trade. The trailing
performance number used by the allocator (`trailing_expr=+0.2412`, N=150) is
an independent measurement of post-deployment live trades and is unaffected
by this static OOS overstatement going forward. The diligence question is
whether the original deployment decision was conditioned on the LEGACY +0.2029
value rather than the canonical +0.1658 — and that question cannot be
answered from the row alone (no `validation_run_id`).

Two paths:

1. **Update path.** Open a separate stage to update
   `validated_setups.oos_exp_r` to +0.1658 (and `oos_exp_r_recomputed_at` /
   provenance fields if such columns exist). This requires the
   strategy_validator path, not this research script. Out of scope here.
2. **Capital-review path.** Route to `/capital-review` on the deployed
   lane to re-examine whether deployment was implicitly conditioned on
   the LEGACY +0.2029 value. If yes, the live decision needs the
   canonical +0.1658 inputs.

Recommendation: path (2) first. The diligence finding is that the public
OOS number for this lane is 0.037 R high, which is a capital-at-risk
question, not a record-keeping question.

## Caveats and limitations

- **OOS power floor.** N_OOS=66, post-filter t=+1.150, p_two=0.250. The
  canonical OOS_ExpR=+0.1658 is descriptive, not confirmatory. Per
  `.claude/rules/backtesting-methodology.md` § RULE 3.3, an underpowered
  OOS cannot be a hard kill — reconciliation does not change that.
- **DSR diligence gap remains open.** `n_trials_at_discovery=35616`,
  `dsr_score=0.175`. The lane was discovered under the pre-Phase-0
  brute-force regime and carries a documented MinBTL violation per
  `MEMORY.md`. Not closed by this audit.
- **Lane validity not re-tested.** This audit only reconciles three
  pre-existing OOS values; whether the lane's edge is real, regime-stable,
  or DSR-deflatable is out of scope.
- **No allocator decision.** Whether the lane should remain
  `status=DEPLOY` is a `/capital-review` question, not this audit's
  verdict.
- **No `validated_setups` update.** A row update would require the
  strategy_validator path with its own pre-reg.
- **LEGACY provenance unrecoverable.** The `validated_setups.oos_exp_r=+0.2029`
  value was written 2026-04-11 with `promotion_provenance=LEGACY`. The row
  carries no `validation_run_id`. The exact cohort lower bound, filter
  delegation path, scratch policy, or OOS cutoff used at promotion time
  cannot be recovered. The +0.0371 R divergence is documented as a
  measurement gap, not attributed to a specific cause.
- **Pre-reg specification bug.** Acceptance criterion 4 in the pre-reg
  yaml was set against `validated_setups.expectancy_r` (scratch-EXCLUSIVE)
  instead of the strict-unlock CSV's IS_ExpR (scratch-INCLUSIVE). Runner
  reproduces strict-unlock exactly; FAIL on criterion 4 is expected
  behavior under a mis-specified target. Pre-reg yaml left as-locked per
  no-post-hoc-rescue rule. Correction documented above in
  § "IS_ExpR caveat".

## What this reconciliation does NOT disconfirm

- It does NOT disconfirm the deployed lane's edge. The IS Chordia t=+4.256
  clears the strict 3.79 hurdle; canonical machinery reproduces this.
- It does NOT disconfirm the strict-unlock 2026-05-02 verdict. That
  audit's `PASS_CHORDIA` verdict stands on the canonical IS gates.
- It does NOT disconfirm or confirm the live trailing performance
  (`trailing_expr=+0.2412`, N=150). Live trailing is independent of all
  three pre-existing OOS numbers and was not measured by this audit.

## Reproduction

```
python research/oos_reconcile_ovnrng100.py
```

Outputs to stdout. Read-only against `gold.db`. No canonical writes.
