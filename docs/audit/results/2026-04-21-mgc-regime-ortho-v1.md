# MGC regime-orthogonal ORB filters — Phase 3 result

**Hypothesis:** `docs/audit/hypotheses/2026-04-21-mgc-regime-ortho-v1.yaml`
**Parent commit:** `f4da26b4`
**Run UTC:** `2026-04-21T03:35:29.277225+00:00`
**Holdout fence:** `2026-01-01` from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`

## Scope

- Instrument: `MGC`
- Sessions: `BRISBANE_1025`, `LONDON_METALS`, `US_DATA_830`
- Entry model: `E2` / `confirm_bars=1` / `RR=1.0` / `orb_minutes=5`
- Family K: `12` cells (`3 sessions × 2 predicates × 2 directions`)

## Canonical thresholds (IS only)

- `ATR_VEL_RATIO_Q67_HIGH` threshold: `1.021300`
- `PREV_WEEK_RANGE_Q67_HIGH` threshold: `93.676000`

## Coverage

- Total scoped rows: `1917`
- IS rows: `1777`
- OOS rows: `140`
- Session counts: `{'LONDON_METALS': 988, 'US_DATA_830': 929}`
- `BRISBANE_1025` has zero scoped `orb_outcomes` rows for this exact family and therefore fails as zero-coverage cells rather than being imputed.

## Cell table

| Cell | Session | Feature | Dir | N_on_IS | N_on_OOS | ExpR_on_IS | ExpR_on_OOS | Δ_IS | t_IS | p | q_family | WFE_holdout | 1.5x | 2.0x |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BRI_ATRVEL_LONG` | `BRISBANE_1025` | `ATR_VEL_RATIO_Q67_HIGH` | `long` | 0 | 0 | nan | nan | nan | nan | nan | 1.0000 | nan | nan | nan |
| `BRI_ATRVEL_SHORT` | `BRISBANE_1025` | `ATR_VEL_RATIO_Q67_HIGH` | `short` | 0 | 0 | nan | nan | nan | nan | nan | 1.0000 | nan | nan | nan |
| `BRI_PREVWEEK_LONG` | `BRISBANE_1025` | `PREV_WEEK_RANGE_Q67_HIGH` | `long` | 0 | 0 | nan | nan | nan | nan | nan | 1.0000 | nan | nan | nan |
| `BRI_PREVWEEK_SHORT` | `BRISBANE_1025` | `PREV_WEEK_RANGE_Q67_HIGH` | `short` | 0 | 0 | nan | nan | nan | nan | nan | 1.0000 | nan | nan | nan |
| `LDM_ATRVEL_LONG` | `LONDON_METALS` | `ATR_VEL_RATIO_Q67_HIGH` | `long` | 161 | 22 | -0.1495 | +0.2696 | +0.0034 | +0.044 | 0.4825 | 0.8271 | 213.378 | -0.2299 | -0.3102 |
| `LDM_ATRVEL_SHORT` | `LONDON_METALS` | `ATR_VEL_RATIO_Q67_HIGH` | `short` | 141 | 15 | -0.0403 | +0.2419 | +0.1136 | +1.428 | 0.0772 | 0.4633 | 3.243 | -0.1175 | -0.1947 |
| `LDM_PREVWEEK_LONG` | `LONDON_METALS` | `PREV_WEEK_RANGE_Q67_HIGH` | `long` | 154 | 39 | -0.0121 | -0.0461 | +0.2079 | +2.602 | 0.0049 | 0.0587 | nan | -0.0753 | -0.1385 |
| `LDM_PREVWEEK_SHORT` | `LONDON_METALS` | `PREV_WEEK_RANGE_Q67_HIGH` | `short` | 141 | 32 | -0.0611 | +0.0461 | +0.0832 | +1.017 | 0.1550 | 0.5040 | nan | -0.1262 | -0.1913 |
| `USD830_ATRVEL_LONG` | `US_DATA_830` | `ATR_VEL_RATIO_Q67_HIGH` | `long` | 133 | 21 | -0.0765 | -0.2813 | -0.0187 | -0.204 | 0.5808 | 0.8711 | nan | -0.1217 | -0.1668 |
| `USD830_ATRVEL_SHORT` | `US_DATA_830` | `ATR_VEL_RATIO_Q67_HIGH` | `short` | 150 | 14 | -0.0524 | -0.3182 | +0.0849 | +0.964 | 0.1680 | 0.5040 | -7.125 | -0.0972 | -0.1420 |
| `USD830_PREVWEEK_LONG` | `US_DATA_830` | `PREV_WEEK_RANGE_Q67_HIGH` | `long` | 148 | 36 | -0.0482 | -0.2174 | +0.0238 | +0.266 | 0.3953 | 0.8271 | nan | -0.0862 | -0.1241 |
| `USD830_PREVWEEK_SHORT` | `US_DATA_830` | `PREV_WEEK_RANGE_Q67_HIGH` | `short` | 140 | 33 | -0.1014 | +0.0302 | +0.0100 | +0.110 | 0.4563 | 0.8271 | nan | -0.1432 | -0.1850 |

## Family-level gates

- BH-FDR family survivors (`q<0.05`): `0`
- Positive control (`ORB_G5` sanity control, verified from current canonical data in this run): `{'status': 'PASS', 'session': 'COMEX_SETTLE', 'n_on': 1555, 'delta_is': 0.5021675550791951, 't_is': 5.114648501857901, 'raw_p': 8.022372368605497e-07}`
- Negative control (calendar parity): `PASS`

## Nulls

- Observed max t across 12 cells: `2.6017`
- `destruction_shuffle` empirical p on family max-t: `0.3466` from `250` replicates
- `rng_null` empirical p on family max-t: `0.3586` from `250` replicates

## Rolling blocked CV

- Fold count: `4`
- Mean train delta: `0.05865662514556565`
- Mean test delta: `0.07621286198579276`
- CV WFE: `1.2993052668246519`

## DSR bracket (cross-check only)

- `BRI_ATRVEL_LONG`: rho=0.3 `None`, rho=0.5 `None`, rho=0.7 `None`
- `BRI_ATRVEL_SHORT`: rho=0.3 `None`, rho=0.5 `None`, rho=0.7 `None`
- `BRI_PREVWEEK_LONG`: rho=0.3 `None`, rho=0.5 `None`, rho=0.7 `None`
- `BRI_PREVWEEK_SHORT`: rho=0.3 `None`, rho=0.5 `None`, rho=0.7 `None`
- `LDM_ATRVEL_LONG`: rho=0.3 `0.0003993710541210782`, rho=0.5 `0.00011638801592411818`, rho=0.7 `5.4343104337273296e-05`
- `LDM_ATRVEL_SHORT`: rho=0.3 `0.06041199016545806`, rho=0.5 `0.03144819146763306`, rho=0.7 `0.020728714609944965`
- `LDM_PREVWEEK_LONG`: rho=0.3 `0.12047067509474185`, rho=0.5 `0.06740021906328975`, rho=0.7 `0.04621884142407312`
- `LDM_PREVWEEK_SHORT`: rho=0.3 `0.03438501197998245`, rho=0.5 `0.016655772221594534`, rho=0.7 `0.01052228264000643`
- `USD830_ATRVEL_LONG`: rho=0.3 `0.027740871754122542`, rho=0.5 `0.013401691460913578`, rho=0.7 `0.008470877736418636`
- `USD830_ATRVEL_SHORT`: rho=0.3 `0.044225012815398534`, rho=0.5 `0.021592279844584894`, rho=0.7 `0.013672962931952248`
- `USD830_PREVWEEK_LONG`: rho=0.3 `0.05187066977453664`, rho=0.5 `0.0259975510987564`, rho=0.7 `0.016727392232936844`
- `USD830_PREVWEEK_SHORT`: rho=0.3 `0.011852683983457846`, rho=0.5 `0.005105464481067856`, rho=0.7 `0.0030115332752952417`

## Verdict

**Current family verdict: DEAD at Phase 3.** No cell survived family-level BH-FDR on canonical IS. This is not a post-hoc rescue surface.
