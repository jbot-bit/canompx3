# Noise Check Snapshot (2026-02-22)

Method: permutation sanity check on current KEEP lead-lag candidates.
- Null model: shuffle leader directions within year, recompute uplift ON-OFF.
- 300 permutations per candidate.
- Metric: observed uplift vs permutation distribution (z-score + p_ge_obs).

## Results

- A2 `MES_US_DATA_OPEN -> M2K_US_DATA_OPEN`
  - observed uplift: `+0.3018`
  - perm mean/std: `-0.0046 / 0.0832`
  - z-score: `3.68`
  - p(perm >= obs): `0.0000`

- A1 `M6E_US_EQUITY_OPEN -> M2K_US_POST_EQUITY`
  - observed uplift: `+0.2877`
  - perm mean/std: `-0.0013 / 0.0752`
  - z-score: `3.85`
  - p(perm >= obs): `0.0000`

- A0 `M6E_US_EQUITY_OPEN -> MES_US_EQUITY_OPEN`
  - observed uplift: `+0.2751`
  - perm mean/std: `-0.0058 / 0.1404`
  - z-score: `2.00`
  - p(perm >= obs): `0.0267`

- A3 `MES_1000 -> M2K_US_POST_EQUITY`
  - observed uplift: `+0.1997`
  - perm mean/std: `-0.0017 / 0.0733`
  - z-score: `2.75`
  - p(perm >= obs): `0.0033`

- B1 `M2K_1000 -> MES_1000`
  - observed uplift: `+0.1490`
  - perm mean/std: `-0.0016 / 0.0824`
  - z-score: `1.83`
  - p(perm >= obs): `0.0333`

## Read
- These are unlikely to be pure random alignment for this candidate set.
- Still not immunity from data-snooping bias across the broader search process.
- Promotion should continue to require strict OOS + forward shadow checks.
