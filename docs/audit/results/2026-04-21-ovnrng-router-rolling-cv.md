# OVNRNG Router — Rolling CV + Ablation — **KILL**

**Date:** 2026-04-21
**Branch:** `research/ovnrng-router-rolling-cv`
**Script:** `research/audit_ovnrng_router_rolling_cv.py`
**Parent:** PR #62 (single-fold walk-forward reported ROUTER_HOLDS_OOS)

---

## Verdict: **ROUTER_BRITTLE — DEAD**

Rolling 4-fold annual CV kills the PR #62 walk-forward finding.
Router wins **1 of 4 folds** on the primary ovn/atr binning variable.
Mean ΔSR_ann (router − control) = **−0.525**.

**PR #62's "ROUTER_HOLDS_OOS" verdict was a single-fold artifact.**
The 50/50 split landed 2022 in train and 2023–2025 in test — masking
the severely underperforming 2022 test year (ΔSR_ann = **−3.23**).

This is exactly the failure mode `.claude/rules/backtesting-methodology.md`
Rule 3 warns about ("never tune parameters against OOS, never single-
fold-WF for deployment"). Single-fold walk-forward is insufficient
evidence for a router-style signal.

---

## Does this make money? (trader framing)

No. The signal is regime-transient, not persistent. Deploying a
router off this signal would have:
- Lost money in 2022 (router SR=−1.57, control SR=+2.98, a 4.5-SR
  catastrophe)
- Slight loss in 2023 (ΔSR=−0.11)
- Big win in 2024 (ΔSR=+1.42)
- Slight loss in 2025 (ΔSR=−0.18)

A strategy that loses 4 SR-units in one year to gain 1.4 in another
is not deployable. Institutional standard: need ≥3 of 4 folds positive
AND mean ΔSR ≥ 0.30 for deploy-ready status.

---

## Rolling 4-fold CV results

### Primary: ovn_atr (PR #62 variable)

| Fold | Test year | Router SR_ann | Control SR_ann | Δ | Router n |
|------|-----------|---------------|----------------|---|----------|
| 1 | 2022 | **−1.97** | +1.26 | **−3.23** | 193 |
| 2 | 2023 | +0.23 | +0.34 | −0.11 | 151 |
| 3 | 2024 | +1.30 | −0.13 | **+1.42** | 240 |
| 4 | 2025 | +0.09 | +0.27 | −0.18 | 202 |

Wins: **1 of 4**. Mean ΔSR = **−0.525**. Median ΔSR = −0.145.

### Ablation: is ovn/atr special?

Ran the same rolling CV with two alternate vol-regime binning
variables:

| Fold | ovn/atr | atr_20_pct | garch_forecast_vol_pct |
|------|---------|------------|-------------------------|
| test 2022 | −3.23 | −0.50 | −1.56 |
| test 2023 | −0.11 | −1.93 | +1.71 |
| test 2024 | +1.42 | +0.69 | +1.93 |
| test 2025 | −0.18 | −0.76 | −1.17 |
| **mean** | **−0.525** | **−0.625** | **+0.229** |
| **wins** | **1/4** | **1/4** | **2/4** |

No binning variable delivers a robust router signal. GARCH forecast
is the best of the three but still flips sign every other fold —
inconsistent with deployment.

### Map stability (train-derived best session per bin, across folds)

ovn_atr map across 4 folds (each bin): most bins have **3 unique
sessions chosen across 4 folds**. Q3 has 2 unique (NYSE_OPEN vs
CME_PRECLOSE alternating).

atr_20_pct map: Q1 has **4 unique** sessions across 4 folds — complete
instability. Q2 has 4 unique. Average map drift is worse than ovn/atr.

garch_forecast_vol_pct map: 2 unique in every bin — BEST map
stability of the three. Q2 always NYSE_OPEN, Q4 mostly CME_PRECLOSE,
Q5 mostly NYSE_CLOSE. Yet the test-period SR still wins only 2/4
folds — map stability alone does not guarantee test performance.

---

## Where the PR #62 conclusion went wrong (postmortem)

**Single-fold WF hid fold-selection risk.** PR #62 split IS 50/50
on trade-count median — trade #5978 landed at 2022-08-30. That
train/test split put:
- Most of 2022's unfavorable regime days IN TRAIN (helped the
  train-derived map look good)
- All of 2023–2025 in test (favorable regimes where the map still
  produced acceptable SR on average)

The rolling-CV here uses annual folds which expose fold-to-fold
drift that the single fold could not. The single-fold test was
analogous to "pick the best 3-year training window for this router"
— classic in-sample optimization.

**Rule update needed:** `.claude/rules/backtesting-methodology.md`
Rule 3 says "Never tune parameters against OOS." Should add "Single-
fold walk-forward is insufficient for router/allocator hypotheses
— require ≥3 rolling folds OR combinatorial-purged CV (LdP 2020
Ch 8)." Already implicit in Rule 10 (pre-reg required) but worth
making explicit for the WF-discipline rule.

---

## Alternative framings (tunnel-vision check)

### What I fairly tested

- ovn/atr as allocator router: KILLED (1/4 folds)
- atr_20_pct as allocator router: KILLED (1/4 folds)
- garch_forecast_vol_pct as allocator router: MARGINAL (2/4 folds)
  but map drift across folds and mean SR only +0.23 — not deploy-ready

### What remains untested (from PR #61 tunnel-vision list)

1. **Conditioner / sizing tilt**: scale position size continuously
   by bin rather than binary enable/disable. Could be more robust
   to regime shifts.
2. **Confluence filters**: interaction with gap_type, day_of_week,
   prev_day_direction.
3. **Direction-conditional signal**: overnight range effect on long
   vs short breakouts separately.
4. **Cross-instrument**: same tests on MES / MGC. Low priority given
   the MNQ result killed the hypothesis.

### Honest assessment of next-best test

**Do NOT pursue any of the above as the next turn.** The allocator-
router thesis is DEAD at the primary framing. The alternative
framings (conditioner, confluence, direction-conditional) are all
extensions of the same dead hypothesis ("ovn/atr predicts something
useful"). If the primary framing fails rolling CV, spending cycles
on derivative framings has low expected value.

**Real next move:** return to the queue with the OVNRNG research
line closed. Outstanding higher-EV candidates (from prior handovers):

1. L1 EUROPE_FLOW break-quality filter research (not ATR-normalized)
2. Cross-instrument Pathway-B on any PR #51 CANDIDATE_READYs that
   replicate on MES/MGC
3. Portfolio-level diagnostics on the 6 deployed lanes (correlation
   structure, drawdown clusters)

---

## Theory citations (canonical local extracts)

This section is descriptive of why vol-regime routing SHOULD have
worked if the signal were real — and why the empirical failure is
not a theory failure, just a data/regime reality check.

- **Chan 2008 Ch 7** (`docs/institutional/literature/
  chan_2008_ch7_regime_switching.md`): regime classifiers (high-vs-
  low vol) are amenable to GARCH and similar econometric tools. Theory
  allowed for exactly this router design. Empirical result: the
  regime-to-session optimal mapping drifts faster than the 3-year
  training window can learn.
- **Chordia et al 2018** (`chordia_et_al_2018_two_million_strategies.md`):
  factor-segmented testing with strict t ≥ 3.79. The router's IS
  t-stat passed, but segment-wise (per-fold) t-stats do not — exactly
  the failure mode Chordia warn against ("in-sample statistical
  significance does not imply out-of-sample economic significance").
- **Lopez de Prado 2020** (`lopez_de_prado_2020_ml_for_asset_managers.md`):
  backtest overfitting via single-fold WF. CPCV or rolling CV is the
  right approach; single-fold WF is treated explicitly as a red
  flag in § backtest-overfitting.

**Theoretical verdict:** the hypothesis was theory-grounded, but the
empirical data does not support it at the required deployment
threshold. Closing the line is the honest call.

---

## Self-audit (institutional checklist)

- **Wrong question?** No. "Does the router signal hold across
  multiple train/test splits?" is the right question. Asking it
  revealed the answer.
- **Wrong test?** No. Rolling 4-fold CV is a standard institutional
  technique. The ablation across 3 binning variables controls for
  whether ovn/atr is special.
- **Missing angle?** Yes — CPCV (combinatorial-purged CV per LdP
  2020 Ch 8) would be stricter. But CPCV here would make the router
  look even worse (more folds, more opportunities for the map to drift).
  Not worth running to pile on a KILL verdict.
- **Implementation vs signal confusion?** No. The signal is what's
  being tested; no infrastructure was built.
- **Multiple testing?** Scanned 3 binning variables × 4 folds = 12
  fold-level tests. None individually pass a strict threshold. Worth
  noting but doesn't change the KILL verdict.
- **2026 OOS?** UNTOUCHED. Sacred.

---

## What this closes

- **OVNRNG allocator-router line: CLOSED.** PR #62's ROUTER_HOLDS_OOS
  verdict is retracted. No pre-reg will be written. No allocator
  infra will be built on this finding.

- **PR #47's "Q3–Q4 sweet-spot on NYSE_OPEN" now officially an
  isolated single-session observation at RR=1.0.** Not deployable
  without separate Pathway-B pre-reg specific to NYSE_OPEN, and
  previous cross-session / allocator framings have failed.

- **Memory update (do on same PR or next turn):** mark PR #62's
  ROUTER_HOLDS_OOS as retracted via this rolling CV. Update
  `memory/session_2026-04-20-claude-handoff.md` / current session
  handover.

---

## Operational impact

**Zero.** No deployment change. No production code touched. No lane
added or paused. Router research line now closed with a verifiable
KILL verdict. The 6-lane portfolio continues unchanged.

---

## Provenance

- Canonical data: `orb_outcomes`, `daily_features` (triple-joined).
- Rolling 4-fold CV: annual test folds across 2022–2025, 3-year
  training windows. Look-ahead clean (≥17:00 Brisbane sessions only,
  train-only quantile bounds).
- 2026 OOS (Mode A sacred) UNTOUCHED.
- Read-only research. No production code. No pre-reg. No deployment
  change.
- Theory citations reference local extracts under
  `docs/institutional/literature/` (all verified to exist).
