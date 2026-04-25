# MNQ US_DATA_1000 O15 VWAP lane — O5 auction-close confluence v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-mnq-us-data-1000-o5-auction-close-on-vwap-o15-v1.yaml` (LOCKED, commit_sha=5536258b)
**Script:** `research/mnq_us_data_1000_o5_auction_close_on_vwap_o15_v1.py`
**DB:** `/mnt/c/Users/joshd/canompx3/gold.db`

**Verdict:** `KILL`

## Scope

- instrument: `MNQ`
- session: `US_DATA_1000`
- target lane: `O15 E2 RR1.5 CB1 VWAP_MID_ALIGNED`
- confluence feature: `O5` close-location-in-range aligned to O15 break direction
- OOS window: `2026-01-01` through `2026-04-16`

## Truth / Calculation Check

- canonical target lane membership delegated through `research.filter_utils.filter_signal` -> `trading_app.config.ALL_FILTERS['VWAP_MID_ALIGNED']`
- O5 confluence computed from raw `bars_1m` over canonical `pipeline.dst.orb_utc_window(trading_day, 'US_DATA_1000', 5)`
- no O5/O15 break-delay, break-bar, outcome, MAE, or MFE columns used in the feature
- result is local to this lane and does NOT claim anything about O15 as a class

## Results

- lane rows total: `810` | IS: `771` | OOS: `39`
- feature fire rows IS resolved: `285` | off rows IS resolved: `486`
- lane ExpR IS resolved: `0.2131` | scratch-inclusive: `0.2131`
- on-signal ExpR IS resolved: `0.2651`
- off-signal ExpR IS resolved: `0.1826`
- delta IS resolved: `0.0825` | scratch-inclusive: `0.0825`
- Welch t/p: `t=0.914` `p=0.3611`
- OOS on-signal N resolved: `11` | delta OOS resolved: `-0.5039`
- T0 tautology max |corr|: `0.118` vs `o15_size_q3`
- T6 null-floor p: `0.2028` | observed delta: `0.0825`
- T7 positive IS years: `5/7`
- T8 scratch rate on/off: `0.000` / `0.000`

## Year Delta

| Year | Delta |
|---|---:|
| 2019 | 0.1688 |
| 2020 | -0.0819 |
| 2021 | 0.0285 |
| 2022 | -0.0252 |
| 2023 | 0.2422 |
| 2024 | 0.1400 |
| 2025 | 0.2285 |

## T0 Proxy Correlations

| Proxy | corr |
|---|---:|
| o5_size_q3 | 0.082 |
| o15_size_q3 | 0.118 |
| atr70_fire | -0.015 |
| ovn80_fire | 0.002 |

## Interpretation

- This is a nested same-session ORB-state test, not an aperture-family reopen.
- A failure here does not kill other sessions, assets, O15 lanes, or other O5-derived features.
- A pass here would still only support this exact lane/feature/role combination.
