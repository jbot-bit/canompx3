# Garch A4b Binding-Budget Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-17-garch-a4b-binding-budget.yaml`
**Universe:** deployable validated shelf (`38` lanes; no profile/session/instrument filter inside A4b).
**Binding budget:** `max_slots=5`, `max_dd=$2,500` derived from `bulenox_50k` as the real profile scalar matching the locked 5-slot budget, without reusing that profile's lane universe.
**Stop policy:** fixed `SM=0.75` applied via canonical `apply_tight_stop()` path.
**Verdict:** `NULL_BY_CONSTRUCTION`

## Binding Preflight

- Rebalance months in IS: `72`
- Binding pass count: `50` / `72`
- Binding pass ratio: `0.694` (need `>= 0.80`)
- Pass rule: candidate-eligible lanes `> 10` on at least `80%` of IS rebalance dates.

| Month | Rebalance | Deployable | Candidate Eligible | Binding Pass |
|---|---:|---:|---:|---:|
| 2022-07-01 | 2022-07-01 | 38 | 38 | True |
| 2022-08-01 | 2022-08-01 | 38 | 38 | True |
| 2022-09-01 | 2022-09-01 | 38 | 38 | True |
| 2023-05-01 | 2023-05-01 | 38 | 38 | True |
| 2025-05-01 | 2025-05-01 | 38 | 38 | True |
| 2022-10-01 | 2022-10-02 | 36 | 36 | True |
| 2022-11-01 | 2022-11-01 | 36 | 36 | True |
| 2025-01-01 | 2025-01-02 | 36 | 36 | True |
| 2025-02-01 | 2025-02-03 | 36 | 36 | True |
| 2025-03-01 | 2025-03-03 | 36 | 36 | True |
| 2023-06-01 | 2023-06-01 | 38 | 35 | True |
| 2022-05-01 | 2022-05-01 | 34 | 34 | True |

## IS Results

| Route | Annualized R | Sharpe | MaxDD R | Slot hit/day | Trading days |
|---|---:|---:|---:|---:|---:|
| BASELINE_LIT_GROUNDED | +47.34 | +1.466 | 42.46 | 2.924 | 1755 |
| CANDIDATE_GARCH_RANK | +40.04 | +1.420 | 48.09 | 2.287 | 1755 |
| DESTRUCTION_SHUFFLE | +42.86 | +1.321 | 48.09 | 2.910 | 1755 |
| POSITIVE_CONTROL_TRAILING_EXPR | +43.58 | +1.651 | 42.32 | 2.064 | 1755 |

## Decision Rules

- Candidate annualized R delta vs baseline: `-7.30` (need `>= +20.0`)
- Candidate Sharpe delta vs baseline: `-0.046` (need `>= +0.05`)
- Candidate DD ratio vs baseline: `1.133` (need `<= 1.20`)
- Mean selection churn: `0.377` (need `<= 0.50`)
- Primary IS pass: `False`
- Destruction shuffle passes primary: `False` (must be `False`)
- Positive control passes primary: `False` (must be `True`)

## OOS Descriptive

| Route | Annualized R | Sharpe | MaxDD R | Trading days |
|---|---:|---:|---:|---:|
| BASELINE_LIT_GROUNDED | +51.35 | +1.465 | 14.60 | 67 |
| CANDIDATE_GARCH_RANK | +51.35 | +1.465 | 14.60 | 67 |

- OOS direction match vs IS effect: `False`
- OOS effect ratio vs IS effect: `-0.000`

SURVIVED SCRUTINY:
- Destruction shuffle failed to pass, so the garch term did not survive randomization.

DID NOT SURVIVE:
- Binding preflight failed; the stage is null by construction.
- Positive control failed, so the harness cannot claim a clean utility test.
- Candidate failed the locked IS pass rule.
- OOS direction flipped versus the IS effect.
- OOS effect ratio -0.000 is below the 0.40 kill line.

CAVEATS:
- A4b uses the canonical allocator surface but a shelf-level fixed stop policy (`SM=0.75`) and a profile-derived DD scalar only; it is not a deployment claim.
- Early IS rebalances face thinner garch history, so candidate eligibility can be sparse before the feature fully matures.
- The comparator remains a validated-shelf utility test, not profile translation, not forward shadow proof, and not standalone garch edge evidence.

NEXT STEPS:
- Redesign the scarce-resource surface before interpreting utility; do not rescue this stage with narrative.
