# 2026-04-21 Data Landscape

Scope: Phase 0 of the orthogonal canonical golden-egg hunt.

Canonical truth inputs:
- `bars_1m`
- `daily_features`
- `orb_outcomes`

Posture-only inputs:
- `active_validated_setups` for occupied-surface fencing
- active profile live-six lane IDs for negative-space exclusion

Data mode: read-only DuckDB against `/mnt/c/Users/joshd/canompx3/gold.db`
Holdout fence: `2026-01-01`

## 0.1 Shape and Coverage

### Table summaries

#### bars_1m
| symbol | first_bar | last_bar | bar_rows |
| --- | --- | --- | --- |
| MES | 2019-05-06 10:00:00+10:00 | 2026-04-17 09:59:00+10:00 | 2442118 |
| MGC | 2022-06-13 10:00:00+10:00 | 2026-04-17 09:59:00+10:00 | 1347181 |
| MNQ | 2019-05-06 10:00:00+10:00 | 2026-04-20 09:59:00+10:00 | 2444410 |

#### daily_features
| symbol | first_day | last_day | feature_rows |
| --- | --- | --- | --- |
| MES | 2019-05-06 00:00:00 | 2026-04-17 00:00:00 | 6099 |
| MGC | 2022-06-13 00:00:00 | 2026-04-17 00:00:00 | 3360 |
| MNQ | 2019-05-06 00:00:00 | 2026-04-19 00:00:00 | 6102 |

#### orb_outcomes
| symbol | first_day | last_day | outcome_rows |
| --- | --- | --- | --- |
| MES | 2019-05-06 00:00:00 | 2026-04-16 00:00:00 | 1908252 |
| MGC | 2022-06-13 00:00:00 | 2026-04-16 00:00:00 | 918684 |
| MNQ | 2019-05-06 00:00:00 | 2026-04-19 00:00:00 | 2110032 |

### Monthly density samples

#### bars_1m monthly summary
| symbol | min_rows | median_rows | max_rows |
| --- | --- | --- | --- |
| MES | 12495 | 29427.5000 | 31800 |
| MGC | 11580 | 29310.0000 | 31740 |
| MNQ | 12615 | 29436.5000 | 31800 |

#### daily_features monthly summary
| symbol | min_rows | median_rows | max_rows |
| --- | --- | --- | --- |
| MES | 45 | 78.0000 | 81 |
| MGC | 42 | 75.0000 | 81 |
| MNQ | 48 | 78.0000 | 81 |

#### orb_outcomes monthly summary
| symbol | min_rows | median_rows | max_rows |
| --- | --- | --- | --- |
| MES | 5652 | 22950.0000 | 25092 |
| MGC | 4320 | 19980.0000 | 21816 |
| MNQ | 6336 | 25362.0000 | 27684 |

## 0.2 Feature Inventory

Daily feature column count: `289`
Used-in-live-six feature columns: `7`

Used-in-live-six columns derived from current live filter implementations:
- `atr_20_pct` referenced by `1` live-six lane filter(s)
- `orb_COMEX_SETTLE_size` referenced by `1` live-six lane filter(s)
- `orb_EUROPE_FLOW_size` referenced by `1` live-six lane filter(s)
- `orb_NYSE_OPEN_size` referenced by `1` live-six lane filter(s)
- `orb_TOKYO_OPEN_size` referenced by `1` live-six lane filter(s)
- `orb_US_DATA_1000_size` referenced by `1` live-six lane filter(s)
- `symbol` referenced by `5` live-six lane filter(s)

Feature inventory sample:
| column_name | column_type | status |
| --- | --- | --- |
| trading_day | DATE | canonical_but_unused |
| symbol | VARCHAR | used_in_live_six |
| orb_minutes | INTEGER | canonical_but_unused |
| bar_count_1m | INTEGER | canonical_but_unused |
| session_asia_high | DOUBLE | canonical_but_unused |
| session_asia_low | DOUBLE | canonical_but_unused |
| session_london_high | DOUBLE | canonical_but_unused |
| session_london_low | DOUBLE | canonical_but_unused |
| session_ny_high | DOUBLE | canonical_but_unused |
| session_ny_low | DOUBLE | canonical_but_unused |
| rsi_14_at_CME_REOPEN | DOUBLE | canonical_but_unused |
| orb_TOKYO_OPEN_high | DOUBLE | canonical_but_unused |
| orb_TOKYO_OPEN_low | DOUBLE | canonical_but_unused |
| orb_TOKYO_OPEN_size | DOUBLE | used_in_live_six |
| orb_TOKYO_OPEN_break_dir | VARCHAR | canonical_but_unused |
| orb_TOKYO_OPEN_break_ts | TIMESTAMP WITH TIME ZONE | canonical_but_unused |
| orb_TOKYO_OPEN_outcome | VARCHAR | canonical_but_unused |
| orb_TOKYO_OPEN_mae_r | DOUBLE | canonical_but_unused |
| orb_TOKYO_OPEN_mfe_r | DOUBLE | canonical_but_unused |
| orb_SINGAPORE_OPEN_high | DOUBLE | canonical_but_unused |
| orb_SINGAPORE_OPEN_low | DOUBLE | canonical_but_unused |
| orb_SINGAPORE_OPEN_size | DOUBLE | canonical_but_unused |
| orb_SINGAPORE_OPEN_break_dir | VARCHAR | canonical_but_unused |
| orb_SINGAPORE_OPEN_break_ts | TIMESTAMP WITH TIME ZONE | canonical_but_unused |
| orb_SINGAPORE_OPEN_outcome | VARCHAR | canonical_but_unused |
| orb_SINGAPORE_OPEN_mae_r | DOUBLE | canonical_but_unused |
| orb_SINGAPORE_OPEN_mfe_r | DOUBLE | canonical_but_unused |
| daily_open | DOUBLE | canonical_but_unused |
| daily_high | DOUBLE | canonical_but_unused |
| daily_low | DOUBLE | canonical_but_unused |
| daily_close | DOUBLE | canonical_but_unused |
| gap_open_points | DOUBLE | canonical_but_unused |
| orb_TOKYO_OPEN_double_break | BOOLEAN | canonical_but_unused |
| orb_SINGAPORE_OPEN_double_break | BOOLEAN | canonical_but_unused |
| atr_20 | DOUBLE | canonical_but_unused |
| us_dst | BOOLEAN | canonical_but_unused |
| uk_dst | BOOLEAN | canonical_but_unused |
| orb_NYSE_OPEN_high | DOUBLE | canonical_but_unused |
| orb_NYSE_OPEN_low | DOUBLE | canonical_but_unused |
| orb_NYSE_OPEN_size | DOUBLE | used_in_live_six |

## 0.3 Session Coverage Map

Coverage uses non-null `daily_features.orb_{SESSION}_size` before the holdout fence as the session-availability proxy.
| instrument | session | trade_days_pre_holdout | coverage_pct |
| --- | --- | --- | --- |
| MES | BRISBANE_1025 | 5166 | 88.2600 |
| MES | TOKYO_OPEN | 5166 | 88.2600 |
| MES | SINGAPORE_OPEN | 5165 | 88.2500 |
| MES | EUROPE_FLOW | 5163 | 88.2100 |
| MES | LONDON_METALS | 5162 | 88.1900 |
| MES | US_DATA_830 | 5162 | 88.1900 |
| MES | CME_REOPEN | 5160 | 88.1600 |
| MES | NYSE_OPEN | 5157 | 88.1100 |
| MES | US_DATA_1000 | 5157 | 88.1100 |
| MES | CME_PRECLOSE | 4974 | 84.9800 |
| MES | COMEX_SETTLE | 4974 | 84.9800 |
| MES | NYSE_CLOSE | 4974 | 84.9800 |
| MGC | BRISBANE_1025 | 2754 | 88.3500 |
| MGC | NYSE_OPEN | 2754 | 88.3500 |
| MGC | SINGAPORE_OPEN | 2754 | 88.3500 |
| MGC | TOKYO_OPEN | 2754 | 88.3500 |
| MGC | US_DATA_1000 | 2754 | 88.3500 |
| MGC | US_DATA_830 | 2754 | 88.3500 |
| MGC | COMEX_SETTLE | 2751 | 88.2600 |
| MGC | EUROPE_FLOW | 2751 | 88.2600 |
| MGC | LONDON_METALS | 2751 | 88.2600 |
| MGC | CME_REOPEN | 2748 | 88.1600 |
| MGC | CME_PRECLOSE | 2661 | 85.3700 |
| MGC | NYSE_CLOSE | 2661 | 85.3700 |
| MNQ | BRISBANE_1025 | 5166 | 88.2600 |
| MNQ | SINGAPORE_OPEN | 5166 | 88.2600 |
| MNQ | TOKYO_OPEN | 5166 | 88.2600 |
| MNQ | US_DATA_830 | 5163 | 88.2100 |
| MNQ | LONDON_METALS | 5162 | 88.1900 |
| MNQ | CME_REOPEN | 5160 | 88.1600 |
| MNQ | EUROPE_FLOW | 5159 | 88.1400 |
| MNQ | NYSE_OPEN | 5157 | 88.1100 |
| MNQ | US_DATA_1000 | 5157 | 88.1100 |
| MNQ | CME_PRECLOSE | 4974 | 84.9800 |
| MNQ | COMEX_SETTLE | 4974 | 84.9800 |
| MNQ | NYSE_CLOSE | 4974 | 84.9800 |

## 0.4 Regime Descriptors Available

- `atr_20`
- `atr_20_pct`
- `atr_vel_ratio`
- `atr_vel_regime`
- `day_of_week`
- `gap_open_points`
- `overnight_range_pct`
- `pit_range_atr`
- `garch_forecast_vol_pct`
- `prev_week_range`
- `prev_month_range`
- `is_friday`
- `is_monday`
- `is_tuesday`
- `is_nfp_day`
- `is_opex_day`

## 0.5 Holdout-Window Descriptor

Distinct trade days before the sacred holdout, from `orb_outcomes` trades only:
| instrument | session | trade_days_pre_holdout |
| --- | --- | --- |
| MES | SINGAPORE_OPEN | 1721 |
| MES | TOKYO_OPEN | 1721 |
| MES | EUROPE_FLOW | 1719 |
| MES | LONDON_METALS | 1718 |
| MES | NYSE_OPEN | 1702 |
| MES | US_DATA_1000 | 1701 |
| MES | US_DATA_830 | 1678 |
| MES | COMEX_SETTLE | 1654 |
| MES | CME_PRECLOSE | 1426 |
| MES | CME_REOPEN | 869 |
| MES | NYSE_CLOSE | 754 |
| MGC | SINGAPORE_OPEN | 918 |
| MGC | TOKYO_OPEN | 918 |
| MGC | EUROPE_FLOW | 917 |
| MGC | LONDON_METALS | 917 |
| MGC | NYSE_OPEN | 913 |
| MGC | COMEX_SETTLE | 884 |
| MGC | US_DATA_1000 | 869 |
| MGC | US_DATA_830 | 860 |
| MGC | CME_REOPEN | 470 |
| MNQ | BRISBANE_1025 | 1722 |
| MNQ | SINGAPORE_OPEN | 1722 |
| MNQ | TOKYO_OPEN | 1722 |
| MNQ | EUROPE_FLOW | 1719 |
| MNQ | LONDON_METALS | 1718 |
| MNQ | US_DATA_1000 | 1701 |
| MNQ | NYSE_OPEN | 1693 |
| MNQ | US_DATA_830 | 1684 |
| MNQ | COMEX_SETTLE | 1650 |
| MNQ | CME_PRECLOSE | 1454 |
| MNQ | CME_REOPEN | 888 |
| MNQ | NYSE_CLOSE | 807 |

## 0.6 Negative-Space Heatmap

Heatmap file: `outputs/negative_space_heatmap.csv`

Top uncovered cells by pre-holdout day count:
| instrument | session | day_of_week | vol_regime | day_count |
| --- | --- | --- | --- | --- |
| MNQ | BRISBANE_1025 | 2 | Stable | 627 |
| MNQ | CME_REOPEN | 2 | Stable | 627 |
| MNQ | LONDON_METALS | 2 | Stable | 627 |
| MNQ | US_DATA_830 | 2 | Stable | 627 |
| MNQ | CME_PRECLOSE | 2 | Stable | 624 |
| MNQ | NYSE_CLOSE | 2 | Stable | 624 |
| MNQ | BRISBANE_1025 | 3 | Stable | 615 |
| MNQ | LONDON_METALS | 3 | Stable | 615 |
| MNQ | US_DATA_830 | 3 | Stable | 615 |
| MNQ | CME_REOPEN | 3 | Stable | 609 |
| MNQ | CME_PRECLOSE | 3 | Stable | 600 |
| MNQ | NYSE_CLOSE | 3 | Stable | 600 |
| MES | BRISBANE_1025 | 2 | Stable | 573 |
| MES | CME_REOPEN | 2 | Stable | 573 |
| MES | EUROPE_FLOW | 2 | Stable | 573 |
| MES | LONDON_METALS | 2 | Stable | 573 |
| MES | NYSE_OPEN | 2 | Stable | 573 |
| MES | SINGAPORE_OPEN | 2 | Stable | 573 |
| MES | TOKYO_OPEN | 2 | Stable | 573 |
| MES | US_DATA_1000 | 2 | Stable | 573 |

## Live-Six Filter Spec

Current live-six filter columns inferred from code and active lane IDs:
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` → `ORB_G5` / `OrbSizeFilter` / columns `orb_COMEX_SETTLE_size, symbol`
- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` → `ORB_G5` / `OrbSizeFilter` / columns `orb_EUROPE_FLOW_size, symbol`
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` → `COST_LT12` / `CostRatioFilter` / columns `orb_NYSE_OPEN_size, symbol`
- `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` → `ATR_P50` / `OwnATRPercentileFilter` / columns `atr_20_pct`
- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` → `COST_LT12` / `CostRatioFilter` / columns `orb_TOKYO_OPEN_size, symbol`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` → `ORB_G5` / `OrbSizeFilter` / columns `orb_US_DATA_1000_size, symbol`

