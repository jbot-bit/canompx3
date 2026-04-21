# G2 — MinBTL Certificate

**Candidate:** ________________________
**Hypothesis family:** ________________________
**Pre-reg:** `docs/audit/hypotheses/________.yaml`

---

## Purpose

Per Bailey et al 2013 Theorem 1 Eq. 6 (`docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`), data horizon must exceed:

```
MinBTL ≈ 2 · ln(N_trials) / E[max_N]²
```

where `N_trials` is the number of hypotheses searched over to find the candidate, and `E[max_N]` is the expected maximum Sharpe ratio under the null across those trials.

## Live computation

| Input | Value | Source |
|---|---|---|
| N_trials (candidate's hypothesis family K) | | pre-reg yaml `K_family` field |
| E[max_N] (expected max-of-K SR under null) | | `numpy.random` simulation OR Bailey 2013 Table 1 |
| Observed IS horizon (years) | | query: `SELECT MIN(trading_day), MAX(trading_day) FROM orb_outcomes WHERE ...` |
| MinBTL threshold (years) | | `2 * ln(K) / E[max_N]^2` |

Evidence (live-query + computation):

```
$ python -c "
import numpy as np, math
K = <paste K_family>
# E[max_N] via simulation (10000 reps, T trades per trial)
T = <paste T_obs>
sims = np.max(np.abs(np.random.randn(10000, K)) * (1/math.sqrt(T)), axis=1)
E_max = sims.mean()
minbtl = 2 * math.log(K) / (E_max ** 2)
print(f'K={K} T={T} E[max_N]={E_max:.4f} MinBTL={minbtl:.2f} years')
"
<paste output>
```

## Current horizon constraint (pre_registered_criteria.md Amendment 3.1)

- Strict: `N_trials ≤ 48` (5 years of daily data, per current horizon)
- Relaxed: `N_trials ≤ 120` (pre-registered theory-grounded only)

## Verdict

- [ ] PASS — observed IS horizon ≥ MinBTL AND N_trials ≤ 48 strict (or ≤ 120 with theory cite)
- [ ] THEORY-JUSTIFIED RELAXED — N_trials > 48 but with pre-registered literature citation justifying relaxed threshold
- [ ] FAIL — N_trials exceeds both bands OR horizon < MinBTL

## Failure disposition

FAIL → candidate cannot advance. Either (a) narrow N_trials via pre-registration tightening, or (b) accept as "insufficient horizon — cannot distinguish edge from luck at this K."

## Literature citation

- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` Theorem 1 (Eq. 6).
- Amendment 3.1 horizon bands: `docs/institutional/pre_registered_criteria.md`.

## Authored by / committed

- Author: ____________________________
- Commit SHA of candidate's script: ________________
- Commit SHA of this certificate: ________________
- Pinned `pre_registered_criteria.md` commit SHA at eval time: ________________
