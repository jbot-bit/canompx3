# CPCV Infrastructure Calibration Postmortem (v1)

**Date:** 2026-04-21
**Pre-reg:** `docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1.yaml`
**Authority:** `docs/institutional/pre_registered_criteria.md` Amendment 3.2
**Seeds per hypothesis:** 10
**Trades per seed:** 2000 | n_splits=6 | n_test_splits=2 | alpha=0.05

## Overall verdict: **FAIL**

## H1 — Known-null reject rate matches alpha

- Mean reject fraction: **0.0867**
- Pass band: [0.025, 0.1]
- Verdict: **PASS**
- Per-seed reject fractions: [0.0667, 0.0667, 0.2, 0.2, 0.0, 0.1333, 0.0667, 0.0667, 0.0, 0.0667]

## H2 — Known-edge recovery matches theoretical power

- Effect size: +0.15 R per trade, sd=1.0
- Mean per-fold N: 668.0
- Observed mean reject fraction: **0.9800**
- Theoretical power at equivalent N: **0.9720**
- |gap|: 0.0080 (pass ≤ 0.10, kill > 0.20)
- Verdict: **PASS**

## H3 — Embargo sensitivity on AR(1)

- AR(1) ρ: 0.15
- embargo=0:  mean reject = **0.1000**
- embargo=5:  mean reject = **0.1000**
- embargo=10: mean reject = **0.1000**
- embargo=20: mean reject = **0.1000**
- gap (embargo=0 − embargo=5): 0.0000 (pass ≥ 0.03)
- check (a) embargo=0 > embargo=5: **False**
- check (b) embargo∈{5,10,20} within [0.025, 0.10]: **True**
- production embargo chosen: **None**
- Verdict: **KILL**

## Integration decision

One or more calibration hypotheses failed. The CPCV implementation is **PARKED** — do NOT wire into `strategy_validator._check_criterion_8_oos`.

- H3: KILL — embargo sensitivity on AR(1) serial correlation

## Interpretation — why H3 killed (and why that matters)

The H3 result is the pre-registered "CPCV is unnecessary" kill branch firing, not a bug. It reflects a real structural mismatch between LdP 2020 Ch 7 CPCV and the use case Amendment 3.2 wants to address.

**What CPCV-with-embargo is designed for:** cross-validating a *trained model* whose parameters are fit on the train fold and then evaluated on the test fold. Embargo excludes train indices near test boundaries so that serial correlation in labels does not leak train-set information into the test-set evaluation via the fitted parameters.

**What Amendment 3.2 wanted it for:** generating multiple OOS estimates of per-fold ExpR / Sharpe from a *fixed, already-realised* return stream (the deployed lane's historical pnl_r) — to back-stop the 15–30-trade 2026 live OOS with additional probabilistic evidence from IS folds.

In that second use case, the train fold is never fitted to — `cpcv_evaluate` computes the test-fold t-statistic directly on the realised returns, the train indices are never consulted. Embargo therefore has nothing to do. The observed uniform 0.100 reject rate at embargo ∈ {0, 5, 10, 20} under AR(1) ρ=0.15 is driven entirely by the *within-test-fold* autocorrelation inflating t-stats (computed under iid assumption) — a distortion the CPCV-style embargo cannot mitigate.

**Arithmetic sanity check:** under AR(1) with ρ=0.15, the true SE of the sample mean is approximately `sd · √((1+ρ)/(1−ρ))/√N` ≈ 1.16× larger than the iid estimate, so computed t-stats inflate by ≈1.16× and two-tailed reject rate at α=0.05 rises above nominal. The observed 0.10 reject rate is consistent with this mild inflation; embargo cannot fix it because embargo operates on train indices that do not enter the evaluation.

## What this means for Amendment 3.2

Amendment 3.2's tiered Criterion 8 and opt-in power-floor (committed in `0e33df2d`) **remain binding** — they did not depend on CPCV. The amendment simply lists CPCV as a *candidate* second-opinion mechanism; that candidate is now falsified for this use case.

The honest options for generating additional OOS evidence without burning the 2026 sacred holdout are:

1. **Wait for time.** At current fire rates, Tier-1 confirmatory OOS (N ≥ 100 filtered trades per lane) is reachable ~2027-09. Unchanged from Amendment 3.2.
2. **Block bootstrap on realised returns** (Politis-Romano stationary bootstrap, or circular block bootstrap). Addresses the actual contamination — within-sample serial correlation — which H3 showed embargo does not. This is a different technique, not a tweak of CPCV; would need its own pre-reg.
3. **Bayesian shrinkage / hierarchical pooling across deployed lanes** to borrow strength without false-pooling. Also a different technique, own pre-reg.

**Do NOT** adjust H3 post-hoc to get it to pass; `pre_registered_criteria.md:5` forbids post-hoc relaxation and that rule is the reason this project has any rigor left.

## Status of the artifacts

- `trading_app/cpcv.py` — **PARKED.** Tested, documented, but not imported by any production path. Retained as an audit artifact: someone revisiting this direction in the future should see that it was built and measured, not re-speculate.
- `tests/test_trading_app/test_cpcv.py` — **ACTIVE.** 22 tests pass; keeps the parked module from silently breaking under dependency upgrades.
- `research/cpcv_calibration_v1.py` — **ACTIVE one-off.** Reproduces this postmortem deterministically from fixed seeds.
- Integration into `strategy_validator._check_criterion_8_oos` — **DO NOT PROCEED.**

## Follow-on questions (not this commit)

- Is the AR(1) autocorrelation in the 6 deployed lanes' actual 2020–2025 pnl_r streams ≥ 0.15, < 0.15, or negligible? Measure before committing to a block-bootstrap pre-reg. If autocorrelation is effectively zero, plain k-fold on realised returns is sufficient and the whole extra machinery is unneeded.
- If block bootstrap is pursued, the block size should be calibrated from the *measured* autocorrelation decay on real lane returns, not from a synthetic ρ=0.15 assumption.
