# MGC ORB-formation microstructure conditioners — Phase 3 result

**Hypothesis:** `docs/audit/hypotheses/2026-04-21-mgc-microstructure-v1.yaml`
**Parent commit:** `f4da26b4`
**Run UTC:** `2026-04-21T03:49:50.894465+00:00`
**Holdout fence:** `2026-01-01` from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`

## Timing validity

- `ORB_RANGE_CONCENTRATION_Q67_HIGH` is computed from the five 1-minute bars in `[orb_start_utc, orb_end_utc)` only.
- `ORB_VOLUME_CONCENTRATION_Q67_HIGH` is computed from the same ORB formation bars only.
- Both features are fully known at ORB close and therefore strictly precede any `E2` entry scan after ORB close.
- No break-bar, break-delay, or post-ORB field is used to derive the family features.

## Scope

- Instrument: `MGC`
- Sessions: `BRISBANE_1025`, `US_DATA_830`
- Entry model: `E2` / `confirm_bars=1` / `RR=1.0` / `orb_minutes=5`
- Family K: `8` cells (`2 sessions × 2 predicates × 2 directions`)

## Canonical thresholds (IS only)

- `ORB_RANGE_CONCENTRATION_Q67_HIGH`: `0.777778`
- `ORB_VOLUME_CONCENTRATION_Q67_HIGH`: `0.368309`

## Coverage

- Total scoped rows: `929`
- IS rows: `860`
- OOS rows: `69`
- Session counts: `{'US_DATA_830': 929}`
- Feature coverage IS: `{'ORB_RANGE_CONCENTRATION_Q67_HIGH': 1.0, 'ORB_VOLUME_CONCENTRATION_Q67_HIGH': 1.0}`
- `BRISBANE_1025` has zero scoped `orb_outcomes` rows for this exact family and therefore remains a zero-coverage slice rather than being imputed.

## Cell table

| Cell | Session | Feature | Dir | N_on_IS | N_on_OOS | ExpR_on_IS | ExpR_on_OOS | Δ_IS | t_IS | p | q_family | WFE_holdout | 1.5x | 2.0x |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BRI_RANGECONC_LONG` | `BRISBANE_1025` | `ORB_RANGE_CONCENTRATION_Q67_HIGH` | `long` | 0 | 0 | nan | nan | nan | nan | nan | 1.0000 | nan | nan | nan |
| `BRI_RANGECONC_SHORT` | `BRISBANE_1025` | `ORB_RANGE_CONCENTRATION_Q67_HIGH` | `short` | 0 | 0 | nan | nan | nan | nan | nan | 1.0000 | nan | nan | nan |
| `BRI_VOLCONC_LONG` | `BRISBANE_1025` | `ORB_VOLUME_CONCENTRATION_Q67_HIGH` | `long` | 0 | 0 | nan | nan | nan | nan | nan | 1.0000 | nan | nan | nan |
| `BRI_VOLCONC_SHORT` | `BRISBANE_1025` | `ORB_VOLUME_CONCENTRATION_Q67_HIGH` | `short` | 0 | 0 | nan | nan | nan | nan | nan | 1.0000 | nan | nan | nan |
| `USD830_RANGECONC_LONG` | `US_DATA_830` | `ORB_RANGE_CONCENTRATION_Q67_HIGH` | `long` | 140 | 14 | -0.0084 | -0.3451 | +0.0828 | +0.924 | 0.1782 | 1.0000 | -2.525 | -0.0498 | -0.0912 |
| `USD830_RANGECONC_SHORT` | `US_DATA_830` | `ORB_RANGE_CONCENTRATION_Q67_HIGH` | `short` | 144 | 10 | -0.1514 | -0.0572 | -0.0645 | -0.725 | 0.7653 | 1.0000 | nan | -0.1969 | -0.2423 |
| `USD830_VOLCONC_LONG` | `US_DATA_830` | `ORB_VOLUME_CONCENTRATION_Q67_HIGH` | `long` | 136 | 8 | -0.0703 | -0.2755 | -0.0098 | -0.107 | 0.5426 | 1.0000 | nan | -0.1124 | -0.1545 |
| `USD830_VOLCONC_SHORT` | `US_DATA_830` | `ORB_VOLUME_CONCENTRATION_Q67_HIGH` | `short` | 148 | 2 | -0.0745 | +0.9206 | +0.0509 | +0.579 | 0.2816 | 1.0000 | 18.608 | -0.1216 | -0.1687 |

## Family-level gates

- BH-FDR family survivors (`q<0.05`): `0`
- Positive control (`ORB_G5` sanity control, verified from current canonical data in this run): `{'status': 'PASS', 'session': 'COMEX_SETTLE', 'n_on': 1555, 'delta_is': 0.5021675550791951, 't_is': 5.114648501857901, 'raw_p': 8.022372368605497e-07}`
- Negative control (calendar parity): `PASS`

## Nulls

- Observed max t across 8 cells: `0.9238`
- `destruction_shuffle` empirical p on family max-t: `0.4263` from `250` replicates
- `rng_null` empirical p on family max-t: `0.3944` from `250` replicates

## Rolling blocked CV

- Fold count: `4`
- Mean train delta: `0.031238780780480178`
- Mean test delta: `0.07724322911701265`
- CV WFE: `2.472671057805135`

## DSR bracket (cross-check only)

- `BRI_RANGECONC_LONG`: rho=0.3 `None`, rho=0.5 `None`, rho=0.7 `None`
- `BRI_RANGECONC_SHORT`: rho=0.3 `None`, rho=0.5 `None`, rho=0.7 `None`
- `BRI_VOLCONC_LONG`: rho=0.3 `None`, rho=0.5 `None`, rho=0.7 `None`
- `BRI_VOLCONC_SHORT`: rho=0.3 `None`, rho=0.5 `None`, rho=0.7 `None`
- `USD830_RANGECONC_LONG`: rho=0.3 `0.2210073469314121`, rho=0.5 `0.12940482945179754`, rho=0.7 `0.09171645663136574`
- `USD830_RANGECONC_SHORT`: rho=0.3 `0.003487282071685749`, rho=0.5 `0.0011031288924677196`, rho=0.7 `0.0005504102831229152`
- `USD830_VOLCONC_LONG`: rho=0.3 `0.05871032160859835`, rho=0.5 `0.027407940227635708`, rho=0.7 `0.017080658755517786`
- `USD830_VOLCONC_SHORT`: rho=0.3 `0.04426622305369582`, rho=0.5 `0.019078249049813645`, rho=0.7 `0.011317025369161793`

## Verdict

**Current family verdict: DEAD at Phase 3.** No cell survived family-level BH-FDR on canonical IS. This family does not earn a post-hoc rescue.
