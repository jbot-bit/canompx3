# 2026-05-30 ORB Edge Inventory Rebuild - Locked Run Manifest

Purpose: rebuild the ORB edge inventory for MNQ, MES, and MGC from canonical tables only.

Truth tables allowed:
- `orb_outcomes`
- `daily_features`
- `bars_1m`

Banned truth inputs:
- `validated_setups`
- `edge_families`
- `live_config`
- current strategy rankings
- post-2026-01-01 data for selection or tuning
- future-looking feature columns

Holdout:
- Selection/scoring/trial ranking uses `trading_day < 2026-01-01`.
- `trading_day >= 2026-01-01` is descriptive only for power/OOS-warning labels.

Hard exclusions from NO-GO registry:
- E0 is purged.
- E3 is retired and not part of this inventory rebuild.
- Dead instruments MCL, SIL, M6E, MBT, M2K excluded.
- Break delay/speed, late-entry timing, NR4/NR7, IBS, gap/fill, EMA20, broad ML/meta-labeling, RR4.0, and killed VWAP/OVNRNG variants are not reopened.

Core base grid:
- Instruments: MNQ, MES, MGC.
- Sessions: canonical enabled sessions per `pipeline.asset_configs.ASSET_CONFIGS` where `orb_outcomes` has pre-holdout rows.
- Entry models: E1 and E2 only.
- Confirm bars: CB1 only; confirm-bar tuning is outside this inventory pass.
- ORB apertures: O5, O15, O30 where rows exist.
- RR targets: 1.0, 1.5, 2.0.
- Baseline role: standalone ORB edge inventory.

Mechanism families and roles:
1. Baseline ORB continuation.
   - Role: standalone candidate family.
   - Feature rule: no feature gate.
2. ORB size relative to friction.
   - Role: filter/conditioner, not a separate standalone edge.
   - Feature rule: pre-entry cost-to-risk thresholds <= 0.08, <= 0.10, <= 0.12, <= 0.15.
3. Volatility state.
   - Role: conditioner.
   - Feature rule: pre-entry volatility feature, chosen from canonical available columns in this order: `atr_vel_ratio`, `atr_pct`, `atr_percentile`, `daily_atr_pct`. Test high/low terciles only.
4. Session participation/liquidity.
   - Role: conditioner.
   - Feature rule: `orb_{session}_volume` divided by that instrument/session/aperture's prior-20-trading-day median ORB volume. Test high/low terciles only.
   - Explicit exclusion: `rel_vol_{session}` is break-bar volume and is not used because it is not pre-entry-knowable for E2.
5. Cross-market flow.
   - Role: confluence/allocator, not standalone.
   - Feature rule: same trading_day/session/aperture peer ORB direction concordance and peer cost-to-risk <= 0.12, using only peer rows whose ORB window is complete before or at the tested ORB aperture end.

Trial accounting:
- Base cell count K_base = available instrument x session x entry_model x aperture x RR cells.
- Size/friction K_size = K_base x 4 thresholds.
- Volatility K_vol = K_base x 2 tercile tails when a valid column exists.
- Participation K_participation = K_base x 2 tercile tails when a valid column exists.
- Cross-market K_cross = eligible cross-market cells x 2 rules: direction concordance, direction concordance plus peer cost-to-risk <= 0.12.
- Family-level BH-FDR uses the family K above.
- Global headline BH-FDR uses K_total = sum of all tested family cells.

Classification rules:
- Deployable: pre-holdout family-level positive after costs, BH-FDR q < 0.05, t >= 3.00 with mechanism, WFE >= 0.50, no dead era with N >= 50, 2026 descriptive OOS not negative with adequate power, and cost-to-risk not fragile.
- Research-provisional: pre-holdout positive after costs and mechanism-plausible but fails one or more deploy gates, lacks OOS power, or is role-limited as filter/conditioner/confluence.
- Unsupported: negative/zero family mean, no BH survivor, t below threshold, WFE below 0.50, era instability, cost fragility, or insufficient sample for inference.

Scratch policy:
- Include scratches as 0R if `pnl_r` is NULL and `outcome='scratch'`; otherwise use `pnl_r`.

Reporting unit:
- Report family/instrument/session/mechanism summaries. Individual cells are diagnostic only and cannot rescue a family.
