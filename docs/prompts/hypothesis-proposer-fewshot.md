# Fewshot Exemplars — Hypothesis Proposer

These are real pre-registered hypothesis YAMLs from `docs/audit/hypotheses/`. They are shown to the LLM as positive examples of shape, depth, and theory-grounding. The LLM must produce YAML matching this style.

---

## Example 1 — Standalone-edge, MES, 4 hypotheses, Bailey-strict

```yaml
metadata:
  name: "mes-final"
  date_locked: "2026-04-09T16:00:00+10:00"
  holdout_date: "2026-01-01"
  total_expected_trials: 4
  testing_mode: "family"
  data_horizon_years_clean: 6.66

hypotheses:
  - id: 1
    name: "MES CME_PRECLOSE G8 RR1.0"
    theory_citation: "chordia_et_al_2018_two_million_strategies"
    economic_basis: "S&P 500 forced rebalancing before close. Unfiltered delta is -0.264 but G8 filter creates +0.178 edge by removing cost-killed trades."
    filter: {type: ORB_G8, column: orb_size_points}
    scope: {instruments: [MES], sessions: [CME_PRECLOSE], rr_targets: [1.0], entry_models: [E2], confirm_bars: [1], stop_multipliers: [1.0]}
    expected_trial_count: 1
    kill_criteria:
      - "BH FDR q=0.05 fails on K=4 family"
      - "WFE < 0.50"
      - "2026 OOS ExpR < 0"
      - "Any era N>=50 ExpR < -0.05"

total_hypothesis_count: 4
total_expected_trials: 4
```

---

## Example 2 — Single-hypothesis, MNQ, theory-grounded, conservative budget

```yaml
metadata:
  name: "mnq-overnight-flow"
  date_locked: "2026-04-11T10:00:00+10:00"
  holdout_date: "2026-01-01"
  total_expected_trials: 5
  testing_mode: "family"
  data_horizon_years_clean: 6.65

hypotheses:
  - id: 1
    name: "Overnight range predicts EUROPE_FLOW follow-through"
    theory_citation: "chan_2008_ch7_regime_switching"
    economic_basis: |
      Asian-session overnight range reflects institutional positioning. When
      large, London-open participants more likely continue the directional move.
    filter:
      type: OVNRNG
      column: overnight_range
      thresholds: [50, 75, 100, 125, 150]
    scope:
      instruments: [MNQ]
      sessions: [EUROPE_FLOW]
      rr_targets: [1.0, 1.5, 2.0]
      entry_models: [E2]
      confirm_bars: [1]
      stop_multipliers: [1.0]
      orb_minutes: [5]
    expected_trial_count: 5
    kill_criteria:
      - "BH FDR q=0.05 fails on K=5 family"
      - "WFE < 0.50"
      - "2026 OOS ExpR < 0"
      - "Any era N>=50 ExpR < -0.05"

total_hypothesis_count: 1
total_expected_trials: 5
```

---

## What the LLM must avoid

- Citing papers from training memory (e.g., "Markowitz 1952"). Only corpus slugs.
- Setting `holdout_date` to anything other than `2026-01-01`.
- Using `filter.type` like `VOL_RV_HIGH` or `BRK_FAST` with `entry_models: [E2]`.
- Setting `total_expected_trials` above 28 unless explicitly asked.
- Adding instruments outside {MNQ, MES, MGC} — MCL/SIL/M6E/MBT/M2K are dead for ORB.
