# 2026-04-21 Phase B Calibration Probe

Purpose:
- Test whether the Phase B gate stack is intrinsically calibration-biased against any candidate, or whether it can emit `KEEP` for a clean, strong signal.
- This is a synthetic control probe, not a strategy discovery result and not a canonical edge claim.

Method:
- Replayed the same gate logic structure used in `origin/research/pr48-sizer-rule-oos-backtest:research/phase_b_live_lane_verdicts.py` rather than routing a synthetic candidate through that script directly.
- Gate logic mirrored: `holdout_clean`, strict Chordia `t >= 3.79`, `WFE >= 0.50`, conservative `DSR(rho=0.7) > 0.95`, and `sr_state != ALARM`.
- DSR used the repo-native implementation from `trading_app.dsr` with `estimate_var_sr_from_db()` against the canonical database.

Inputs:
- Canonical DB for variance calibration: `/mnt/c/Users/joshd/canompx3/gold.db`
- Phase B lineage head checked read-only: `aa8b838c4024a3f6c76291a933bdb2d2c70cda6c`
- Current hunt branch head: `33e462dde117b076706d71c628176bd353c91309`
- `var_sr` from canonical DB: `0.006712097170233`
- `n_trials` replayed at live-book scale: `35616`

Synthetic cases:
- `clean_strong_signal`: strong train/OOS Sharpe, clean holdout, `CONTINUE` SR state
- `contaminated_strong_signal`: same signal strength, but holdout marked dirty
- `clean_weak_signal`: clean holdout but weaker Sharpe profile that should fail strength gates
- `alarm_strong_signal`: strong clean signal but SR state forced to `ALARM`

## Results

| case_id | train_n | oos_n | train_sr_ann | oos_sr_ann | WFE | t | DSR@0.7 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `clean_strong_signal` | 2000 | 1500 | 1.9500 | 1.8000 | 0.9231 | 4.3916 | 1.000000 | `KEEP` |
| `contaminated_strong_signal` | 2000 | 1500 | 1.9500 | 1.8000 | 0.9231 | 4.3916 | 1.000000 | `DEGRADE` |
| `clean_weak_signal` | 2000 | 1500 | 0.4500 | 0.2000 | 0.4444 | 0.4880 | 0.000003 | `DEGRADE` |
| `alarm_strong_signal` | 2000 | 1500 | 1.9500 | 1.8000 | 0.9231 | 4.3916 | 1.000000 | `PAUSE-PENDING-REVIEW` |

## Case notes

### `clean_strong_signal`
- Description: Clean-holdout synthetic with strong OOS Sharpe and no alarm state.
- Verdict: `KEEP`
- Reasons:
  - All replayed Phase B gates clear.
- DSR grid:
  - rho=0.3, n_eff=24931.5, sr0=0.334141, dsr=1.000000
  - rho=0.5, n_eff=17808.5, sr0=0.327673, dsr=1.000000
  - rho=0.7, n_eff=10685.5, sr0=0.317625, dsr=1.000000

### `contaminated_strong_signal`
- Description: Same signal quality as clean_strong_signal, but holdout provenance marked dirty.
- Verdict: `DEGRADE`
- Reasons:
  - Holdout integrity fails (synthetic discovery marked post-holdout).
- DSR grid:
  - rho=0.3, n_eff=24931.5, sr0=0.334141, dsr=1.000000
  - rho=0.5, n_eff=17808.5, sr0=0.327673, dsr=1.000000
  - rho=0.7, n_eff=10685.5, sr0=0.317625, dsr=1.000000

### `clean_weak_signal`
- Description: Clean-holdout synthetic with weak OOS Sharpe that should fail both Chordia and conservative DSR.
- Verdict: `DEGRADE`
- Reasons:
  - Chordia strict band fails (t=0.488 < 3.79).
  - WFE fails (0.444 < 0.50).
  - Conservative DSR fails (rho=0.7 DSR=0.000003 <= 0.95).
- DSR grid:
  - rho=0.3, n_eff=24931.5, sr0=0.334141, dsr=0.000000
  - rho=0.5, n_eff=17808.5, sr0=0.327673, dsr=0.000001
  - rho=0.7, n_eff=10685.5, sr0=0.317625, dsr=0.000003

### `alarm_strong_signal`
- Description: Strong clean-holdout synthetic forced into SR ALARM to test pause behavior.
- Verdict: `PAUSE-PENDING-REVIEW`
- Reasons:
  - Criterion 12 SR state is ALARM.
- DSR grid:
  - rho=0.3, n_eff=24931.5, sr0=0.334141, dsr=1.000000
  - rho=0.5, n_eff=17808.5, sr0=0.327673, dsr=1.000000
  - rho=0.7, n_eff=10685.5, sr0=0.317625, dsr=1.000000

## Verdict

- `clean_strong_signal` returns `KEEP`, so the replayed Phase B framework is not intrinsically incapable of producing a keep verdict.
- `contaminated_strong_signal` flips to `DEGRADE` on holdout contamination alone, which shows the live-book posture failure is structurally separable from signal strength.
- `clean_weak_signal` degrades on strength gates, which shows the replay still discriminates weak candidates rather than handing out unconditional keeps.
- `alarm_strong_signal` pauses immediately, matching Criterion 12 behavior in the live-book lane audit.

Conclusion:
- No framework-calibration bias was found in the gate stack itself.
- The stronger interpretation remains the same as the Phase B institutional re-evaluation: the live six are posture-blocked, not automatically edge-dead.
