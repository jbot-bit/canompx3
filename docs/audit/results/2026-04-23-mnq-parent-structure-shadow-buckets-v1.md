# MNQ parent structure shadow buckets v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-23-mnq-parent-structure-shadow-buckets-v1.yaml`
**Pre-reg commit SHA field:** `50fabff3777a54e39e0f4006890907c644c01c12`
**Pre-reg git lock commit:** `094adda9b00e79f64064da89a1c72a6445478f81`
**Canonical DB path:** `/mnt/c/users/joshd/canompx3/gold.db`
**Latest canonical trading day:** `2026-04-16`
**Canonical layers:** `daily_features`, `orb_outcomes`
**Family K:** `6` exact IS tests (`H1 1x2` + `H2 2x2`) with BH at `q=0.05`

**Family verdict:** CONTINUE=0 | PARK=0 | KILL=2

## Scope / Question

Evaluate the exact preregistered `conditional_role` family on exact MNQ deployed-parent lane-side populations and ask one question only: do frozen ORB-end / pre-entry-safe prior-day structure buckets improve parent policy value as shadow-only conditioners?

## Verdict / Decision

**Decision:** `DEAD`

No exact-parent shadow bucket in this `K=6` family improved `policy_ev_per_opportunity_r` enough to survive the prereg decision rule. Do not reopen this family under renamed score / confluence language.

## Family summary

| hypothesis | test | pass | N_parent | N_on_IS | policy_EV_delta_IS | delta_mean_IS | t_IS | p_IS | BH | N_on_OOS | policy_EV_delta_OOS | delta_mean_OOS |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| H1 | PD_CLEAR_LONG_fire | filtered | 861 | 320 | -0.0145 | -0.0145 | -0.466 | 0.641262 | N | 13 | +0.1440 | +0.1440 |
| H1 | PD_CLEAR_LONG_fire | unfiltered | 901 | 337 | -0.0079 | -0.0079 | -0.264 | 0.791563 | N | 13 | +0.1440 | +0.1440 |
| H2 | score_eq_2 | filtered | 831 | 84 | -0.0489 | -0.0489 | -1.223 | 0.221550 | N | 5 | -0.3496 | -0.3496 |
| H2 | score_eq_2 | unfiltered | 832 | 84 | -0.0500 | -0.0500 | -1.251 | 0.211138 | N | 5 | -0.3496 | -0.3496 |
| H2 | score_ge_1 | filtered | 831 | 330 | -0.0133 | -0.0133 | -0.414 | 0.679331 | N | 12 | -0.2583 | -0.2583 |
| H2 | score_ge_1 | unfiltered | 832 | 330 | -0.0145 | -0.0145 | -0.449 | 0.653832 | N | 12 | -0.2583 | -0.2583 |

## MNQ COMEX_SETTLE RR1.5 long PD_CLEAR_LONG shadow bucket

**Verdict:** `KILL`

| hypothesis | test | pass | N_parent | N_on_IS | policy_EV_delta_IS | delta_mean_IS | t_IS | p_IS | BH | N_on_OOS | policy_EV_delta_OOS | delta_mean_OOS |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| H1 | PD_CLEAR_LONG_fire | filtered | 861 | 320 | -0.0145 | -0.0145 | -0.466 | 0.641262 | N | 13 | +0.1440 | +0.1440 |
| H1 | PD_CLEAR_LONG_fire | unfiltered | 901 | 337 | -0.0079 | -0.0079 | -0.264 | 0.791563 | N | 13 | +0.1440 | +0.1440 |

### PD_CLEAR_LONG_fire / filtered

- IS selected-trade mean: `+0.2015R` vs parent `+0.0922R`
- IS policy EV per opportunity delta: `-0.0145R`
- IS daily delta test: `n_days=829`, `mean=-0.0145R`, `t=-0.466`, `p=0.641262`, `BH=N`
- OOS selected N: `13`; selected mean `+0.0771R`; policy EV delta `+0.1440R`
- OOS daily delta: `+0.1440R`, `t=+0.925`, `p=0.362283`

### PD_CLEAR_LONG_fire / unfiltered

- IS selected-trade mean: `+0.1781R` vs parent `+0.0770R`
- IS policy EV per opportunity delta: `-0.0079R`
- IS daily delta test: `n_days=869`, `mean=-0.0079R`, `t=-0.264`, `p=0.791563`, `BH=N`
- OOS selected N: `13`; selected mean `+0.0771R`; policy EV delta `+0.1440R`
- OOS daily delta: `+0.1440R`, `t=+0.925`, `p=0.362283`

## MNQ US_DATA_1000 O15 RR1.5 long prior-day structure score buckets

**Verdict:** `KILL`

| hypothesis | test | pass | N_parent | N_on_IS | policy_EV_delta_IS | delta_mean_IS | t_IS | p_IS | BH | N_on_OOS | policy_EV_delta_OOS | delta_mean_OOS |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| H2 | score_eq_2 | filtered | 831 | 84 | -0.0489 | -0.0489 | -1.223 | 0.221550 | N | 5 | -0.3496 | -0.3496 |
| H2 | score_eq_2 | unfiltered | 832 | 84 | -0.0500 | -0.0500 | -1.251 | 0.211138 | N | 5 | -0.3496 | -0.3496 |
| H2 | score_ge_1 | filtered | 831 | 330 | -0.0133 | -0.0133 | -0.414 | 0.679331 | N | 12 | -0.2583 | -0.2583 |
| H2 | score_ge_1 | unfiltered | 832 | 330 | -0.0145 | -0.0145 | -0.449 | 0.653832 | N | 12 | -0.2583 | -0.2583 |

### score_eq_2 / filtered

- IS selected-trade mean: `+0.0727R` vs parent `+0.0565R`
- IS policy EV per opportunity delta: `-0.0489R`
- IS daily delta test: `n_days=800`, `mean=-0.0489R`, `t=-1.223`, `p=0.221550`, `BH=N`
- OOS selected N: `5`; selected mean `-1.0000R`; policy EV delta `-0.3496R`
- OOS daily delta: `-0.3496R`, `t=-1.707`, `p=0.098139`

### score_eq_2 / unfiltered

- IS selected-trade mean: `+0.0727R` vs parent `+0.0576R`
- IS policy EV per opportunity delta: `-0.0500R`
- IS daily delta test: `n_days=801`, `mean=-0.0500R`, `t=-1.251`, `p=0.211138`, `BH=N`
- OOS selected N: `5`; selected mean `-1.0000R`; policy EV delta `-0.3496R`
- OOS daily delta: `-0.3496R`, `t=-1.707`, `p=0.098139`

### score_ge_1 / filtered

- IS selected-trade mean: `+0.1047R` vs parent `+0.0565R`
- IS policy EV per opportunity delta: `-0.0133R`
- IS daily delta test: `n_days=800`, `mean=-0.0133R`, `t=-0.414`, `p=0.679331`, `BH=N`
- OOS selected N: `12`; selected mean `-0.1809R`; policy EV delta `-0.2583R`
- OOS daily delta: `-0.2583R`, `t=-1.457`, `p=0.155468`

### score_ge_1 / unfiltered

- IS selected-trade mean: `+0.1047R` vs parent `+0.0576R`
- IS policy EV per opportunity delta: `-0.0145R`
- IS daily delta test: `n_days=801`, `mean=-0.0145R`, `t=-0.449`, `p=0.653832`, `BH=N`
- OOS selected N: `12`; selected mean `-0.1809R`; policy EV delta `-0.2583R`
- OOS daily delta: `-0.2583R`, `t=-1.457`, `p=0.155468`

## Closeout

SURVIVED SCRUTINY:
PARKED:
DID NOT SURVIVE:
- MNQ COMEX_SETTLE RR1.5 long PD_CLEAR_LONG shadow bucket
- MNQ US_DATA_1000 O15 RR1.5 long prior-day structure score buckets
## Caveats / Limitations

CAVEATS:
- OOS from 2026-01-01 onward is descriptive only and remains thin for every bucket.
- This script evaluates exact lane-side parent populations only; it does not reopen same-session routing or live sizing questions.
- ORB-end / pre-entry-safe structure predicates are used; no post-entry or post-session features are admitted.
NEXT STEPS:
- If any hypothesis is `CONTINUE`, the honest next move is a bounded shadow-monitor / translation follow-through on that exact object.
- If a hypothesis is `PARK`, leave it as descriptive-only and do not broaden the family without a new prereg.
- If a hypothesis is `KILL`, do not reopen it under renamed score language.

## Reproduction

- Runner: `research/mnq_parent_structure_shadow_buckets_v1.py`
- Command: `./.venv-wsl/bin/python research/mnq_parent_structure_shadow_buckets_v1.py --output docs/audit/results/2026-04-23-mnq-parent-structure-shadow-buckets-v1.md`
