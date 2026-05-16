# Deployment Coverage Audit — Orphan Validated Strategies

**Rebalance date:** 2026-05-17
**Total active validated strategies:** 844
**Profiles in roster:** 10 (1 active, 9 inactive)

## Delta vs prior snapshot (2026-05-12)

Refresh trigger: 296 stale `last_trade_day` rows on `validated_setups` (Drift Check 52). Refresh via canonical `scripts/migrations/backfill_validated_trade_windows.py` (idempotent DELETE+INSERT, inspected=844, drifted=296, updated=296). All 130 drift checks PASS post-refresh.

Routing-class counts vs 2026-05-12 snapshot:

| Class | 2026-05-12 | 2026-05-17 | Δ |
|---|---:|---:|---:|
| ROUTABLE_ACTIVE | 766 (14093.6 R) | 766 (13990.3 R) | 0 lanes, −103.3 R annual_r |
| ROUTABLE_DORMANT | 78 (809.4 R) | 78 (809.4 R) | unchanged |
| ORPHAN_FIRM_GAP | 0 | 0 | unchanged |
| ORPHAN_NO_FIRM | 0 | 0 | unchanged |

Key observation: **no lanes moved between classes**. The 78 ROUTABLE_DORMANT cohort and 809.4 R blocked-capital figure are stable across the 5-day refresh window. `ROUTABLE_ACTIVE` annual_r drifted modestly (−0.7%) as recent trades reweight the trailing-window formula; no qualitative change to the deployment-coverage picture.

Minor rank churn inside the DORMANT cohort: `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P50` dropped 27.6 R → 23.2 R (still DORMANT), `MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075` rose 11.4 R → 12.1 R. No rank change affects the activation/PARK decision; both rows remain blocked by the same single active profile.

## Deferred-decision marker

Activation-vs-PARK decision on the 78 ROUTABLE_DORMANT lanes is **deferred to a follow-up session** (HANDOFF Next-Step #6). Refresh-first ordering (user-chosen 2026-05-17) eliminates the stale-snapshot bias risk for that decision. This file is INFORMATIONAL only — no `prop_profiles.py` or `lane_allocation.json` mutation this session.

Prior snapshot retained at `docs/audit/results/2026-05-12-deployment-coverage-orphans.md` for evidence trail.

Read-only audit. Does not mutate prop_profiles, validated_setups, or lane_allocation.json.
annual_r computed via canonical `trading_app.lane_allocator.compute_lane_scores`.

## Scope and limitations (read first)

- **`annual_r` is live, not stale.** Numbers recomputed every run from the canonical trailing-window formula in `lane_allocator.py:373`. Any prior cited value (e.g. memory or plan documents) may differ — trust this report.
- **`minimal_fix` = cheapest single-whitelist edit, NOT an activation recommendation.** It identifies the profile already closest to admitting the strategy. It does NOT verify broker-fit, cost-model viability, or whether the profile is inactive for capital reasons.
- **`ROUTABLE_DORMANT` does not mean "deploy this".** A profile may be inactive because of capital, broker rules, or a deliberate prior decision. This audit only proves the (instrument, session) pair is whitelisted by an inactive profile — activation is a separate decision.
- **No new statistical claims.** This audit makes zero new p-value or significance assertions. Every annual_r already passed validation upstream — we're only joining sources.

## Notable findings

- **`topstep_50k`** (inactive, firm=topstep) admits ZERO active validated strategies as currently configured. Its (instrument, session) whitelist does not intersect validated_setups.
- Active profile **`topstep_50k_mnq_auto`** does not admit 78/844 validated strategies (sum annual_r blocked = 809.4R).
- Active-profile reach: instruments=['MNQ'] sessions=['CME_PRECLOSE', 'COMEX_SETTLE', 'EUROPE_FLOW', 'NYSE_OPEN', 'SINGAPORE_OPEN', 'TOKYO_OPEN', 'US_DATA_1000']. Strategies on instruments outside that set cannot be live-traded today.

## Summary by routing class

| Class | Count | Sum annual_r |
|---|---:|---:|
| ROUTABLE_ACTIVE | 766 | 13990.3 |
| ROUTABLE_DORMANT | 78 | 809.4 |
| ORPHAN_FIRM_GAP | 0 | 0.0 |
| ORPHAN_NO_FIRM | 0 | 0.0 |

## Top 30 non-active-routable strategies by annual_r

Strategies the live broker (topstep_50k_mnq_auto) cannot trade. Rank order = how much $/yr edge sits behind a profile-config decision.

| strategy_id | instrument | session | annual_r | N | ExpR | class | minimal_fix |
|---|---|---|---:|---:|---:|---|---|
| MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 39.5 | 72 | 0.412 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MGC_LONDON_METALS_E2_RR1.5_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 36.5 | 76 | 0.401 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MES_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15 | MES | US_DATA_1000 | 32.5 | 181 | 0.179 | ROUTABLE_DORMANT | add session 'US_DATA_1000' to topstep_50k_mes_auto [INACTIVE] |
| MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15 | MES | US_DATA_1000 | 32.1 | 195 | 0.165 | ROUTABLE_DORMANT | add session 'US_DATA_1000' to topstep_50k_mes_auto [INACTIVE] |
| MGC_LONDON_METALS_E2_RR1.5_CB1_ORB_VOL_8K_O30_S075 | MGC | LONDON_METALS | 28.0 | 76 | 0.307 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15 | MES | US_DATA_1000 | 27.8 | 165 | 0.169 | ROUTABLE_DORMANT | add session 'US_DATA_1000' to topstep_50k_mes_auto [INACTIVE] |
| MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_4K_O15 | MGC | LONDON_METALS | 27.8 | 92 | 0.277 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30_S075 | MGC | LONDON_METALS | 27.1 | 72 | 0.283 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MGC_LONDON_METALS_E1_RR1.5_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 25.9 | 73 | 0.295 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70 | MNQ | NYSE_CLOSE | 25.3 | 46 | 0.321 | ROUTABLE_DORMANT | add session 'NYSE_CLOSE' to topstep_50k_mnq_auto [ACTIVE] |
| MGC_LONDON_METALS_E2_RR1.0_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 24.8 | 76 | 0.272 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P50 | MNQ | NYSE_CLOSE | 23.2 | 66 | 0.264 | ROUTABLE_DORMANT | add session 'NYSE_CLOSE' to topstep_50k_mnq_auto [ACTIVE] |
| MGC_LONDON_METALS_E2_RR1.0_CB1_ORB_VOL_4K_O15 | MGC | LONDON_METALS | 22.6 | 94 | 0.220 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4 | MGC | CME_REOPEN | 21.6 | 101 | 0.196 | ROUTABLE_DORMANT | add session 'CME_REOPEN' to topstep_50k [INACTIVE] |
| MGC_LONDON_METALS_E1_RR1.0_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 19.4 | 75 | 0.216 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MGC_LONDON_METALS_E1_RR1.0_CB3_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 18.6 | 74 | 0.209 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MGC_LONDON_METALS_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075 | MGC | LONDON_METALS | 18.6 | 76 | 0.204 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G4_NOMON_O15 | MNQ | LONDON_METALS | 17.1 | 197 | 0.087 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k_mnq_auto [ACTIVE] |
| MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G5_NOMON_O15 | MNQ | LONDON_METALS | 17.1 | 197 | 0.087 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k_mnq_auto [ACTIVE] |
| MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G6_NOMON_O15 | MNQ | LONDON_METALS | 17.1 | 197 | 0.087 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k_mnq_auto [ACTIVE] |
| MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G8_NOMON_O15 | MNQ | LONDON_METALS | 17.1 | 197 | 0.087 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k_mnq_auto [ACTIVE] |
| MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075 | MNQ | NYSE_CLOSE | 17.1 | 20 | 0.498 | ROUTABLE_DORMANT | add session 'NYSE_CLOSE' to topstep_50k_mnq_auto [ACTIVE] |
| MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075 | MGC | LONDON_METALS | 15.9 | 92 | 0.159 | ROUTABLE_DORMANT | add session 'LONDON_METALS' to topstep_50k [INACTIVE] |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O30_S075 | MES | CME_PRECLOSE | 12.8 | 11 | 0.581 | ROUTABLE_DORMANT | add session 'CME_PRECLOSE' to tradeify_100k_type_b [INACTIVE] |
| MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075 | MES | COMEX_SETTLE | 12.1 | 70 | 0.188 | ROUTABLE_DORMANT | add session 'COMEX_SETTLE' to topstep_50k_mes_auto [INACTIVE] |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O30 | MES | CME_PRECLOSE | 11.8 | 11 | 0.535 | ROUTABLE_DORMANT | add session 'CME_PRECLOSE' to tradeify_100k_type_b [INACTIVE] |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_O30_S075 | MES | CME_PRECLOSE | 11.4 | 13 | 0.511 | ROUTABLE_DORMANT | add session 'CME_PRECLOSE' to tradeify_100k_type_b [INACTIVE] |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_O30_S075 | MES | CME_PRECLOSE | 11.4 | 13 | 0.511 | ROUTABLE_DORMANT | add session 'CME_PRECLOSE' to tradeify_100k_type_b [INACTIVE] |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_O30 | MES | CME_PRECLOSE | 10.1 | 13 | 0.454 | ROUTABLE_DORMANT | add session 'CME_PRECLOSE' to tradeify_100k_type_b [INACTIVE] |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_O30 | MES | CME_PRECLOSE | 10.1 | 13 | 0.454 | ROUTABLE_DORMANT | add session 'CME_PRECLOSE' to tradeify_100k_type_b [INACTIVE] |

## Per-firm coverage matrix

Best annual_r reachable by each profile across (instrument, session) pairs in validated_setups.
`-` = profile does not admit that pair.

| profile | active | MES/CME_PRECLOSE | MES/COMEX_SETTLE | MES/NYSE_CLOSE | MES/NYSE_OPEN | MES/US_DATA_1000 | MES/US_DATA_830 | MGC/CME_REOPEN | MGC/LONDON_METALS | MNQ/CME_PRECLOSE | MNQ/COMEX_SETTLE | MNQ/EUROPE_FLOW | MNQ/LONDON_METALS | MNQ/NYSE_CLOSE | MNQ/NYSE_OPEN | MNQ/SINGAPORE_OPEN | MNQ/TOKYO_OPEN | MNQ/US_DATA_1000 | MNQ/US_DATA_830 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| topstep_50k_mnq_auto | YES | - | - | - | - | - | - | - | - | 38.0 | 34.8 | 47.1 | - | - | 34.0 | 66.0 | 62.1 | 38.1 | - |
| bulenox_50k | no | - | - | - | - | - | - | 21.6 | - | - | 34.8 | 47.1 | - | - | - | 66.0 | 62.1 | - | - |
| self_funded_tradovate | no | 12.8 | 12.1 | - | 3.8 | 32.5 | - | 21.6 | - | 38.0 | 34.8 | 47.1 | - | - | 34.0 | 66.0 | 62.1 | 38.1 | - |
| topstep_100k_type_a | no | 12.8 | 12.1 | - | 3.8 | 32.5 | -7.1 | 21.6 | 39.5 | 38.0 | 34.8 | - | 17.1 | - | 34.0 | - | 62.1 | 38.1 | 2.4 |
| topstep_50k | no | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| topstep_50k_mes_auto | no | 12.8 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| topstep_50k_type_a | no | 12.8 | 12.1 | - | 3.8 | 32.5 | -7.1 | 21.6 | 39.5 | 38.0 | 34.8 | - | 17.1 | - | 34.0 | - | 62.1 | 38.1 | 2.4 |
| tradeify_100k_type_b | no | - | 12.1 | 3.5 | 3.8 | 32.5 | -7.1 | 21.6 | - | - | 34.8 | 47.1 | - | 25.3 | 34.0 | 66.0 | - | 38.1 | 2.4 |
| tradeify_50k | no | - | - | - | - | - | - | - | - | 38.0 | 34.8 | - | - | 25.3 | - | - | 62.1 | 38.1 | - |
| tradeify_50k_type_b | no | - | 12.1 | 3.5 | 3.8 | 32.5 | -7.1 | 21.6 | - | - | 34.8 | 47.1 | - | 25.3 | 34.0 | 66.0 | - | 38.1 | 2.4 |

## Dead whitelist entries

Sessions allowed by a profile but with zero active validated strategies for that profile's instruments. = empty capacity in the whitelist; tightening cost-free.

- **topstep_50k** (inactive): TOKYO_OPEN

## Cross-broker arbitrage

Strategies blocked on every active profile but routable on ≥1 inactive profile. Activation candidates ranked by annual_r.

| strategy_id | instrument | session | annual_r | N | inactive profiles that admit |
|---|---|---|---:|---:|---|
| MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 39.5 | 72 | topstep_50k_type_a, topstep_100k_type_a |
| MGC_LONDON_METALS_E2_RR1.5_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 36.5 | 76 | topstep_50k_type_a, topstep_100k_type_a |
| MES_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15 | MES | US_DATA_1000 | 32.5 | 181 | topstep_50k_type_a, topstep_100k_type_a, tradeify_50k_type_b, tradeify_100k_type_b, self_funded_tradovate |
| MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15 | MES | US_DATA_1000 | 32.1 | 195 | topstep_50k_type_a, topstep_100k_type_a, tradeify_50k_type_b, tradeify_100k_type_b, self_funded_tradovate |
| MGC_LONDON_METALS_E2_RR1.5_CB1_ORB_VOL_8K_O30_S075 | MGC | LONDON_METALS | 28.0 | 76 | topstep_50k_type_a, topstep_100k_type_a |
| MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15 | MES | US_DATA_1000 | 27.8 | 165 | topstep_50k_type_a, topstep_100k_type_a, tradeify_50k_type_b, tradeify_100k_type_b, self_funded_tradovate |
| MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_4K_O15 | MGC | LONDON_METALS | 27.8 | 92 | topstep_50k_type_a, topstep_100k_type_a |
| MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30_S075 | MGC | LONDON_METALS | 27.1 | 72 | topstep_50k_type_a, topstep_100k_type_a |
| MGC_LONDON_METALS_E1_RR1.5_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 25.9 | 73 | topstep_50k_type_a, topstep_100k_type_a |
| MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70 | MNQ | NYSE_CLOSE | 25.3 | 46 | tradeify_50k, tradeify_50k_type_b, tradeify_100k_type_b |
| MGC_LONDON_METALS_E2_RR1.0_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 24.8 | 76 | topstep_50k_type_a, topstep_100k_type_a |
| MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P50 | MNQ | NYSE_CLOSE | 23.2 | 66 | tradeify_50k, tradeify_50k_type_b, tradeify_100k_type_b |
| MGC_LONDON_METALS_E2_RR1.0_CB1_ORB_VOL_4K_O15 | MGC | LONDON_METALS | 22.6 | 94 | topstep_50k_type_a, topstep_100k_type_a |
| MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4 | MGC | CME_REOPEN | 21.6 | 101 | topstep_50k_type_a, topstep_100k_type_a, tradeify_50k_type_b, tradeify_100k_type_b, bulenox_50k, self_funded_tradovate |
| MGC_LONDON_METALS_E1_RR1.0_CB1_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 19.4 | 75 | topstep_50k_type_a, topstep_100k_type_a |
| MGC_LONDON_METALS_E1_RR1.0_CB3_ORB_VOL_8K_O30 | MGC | LONDON_METALS | 18.6 | 74 | topstep_50k_type_a, topstep_100k_type_a |
| MGC_LONDON_METALS_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075 | MGC | LONDON_METALS | 18.6 | 76 | topstep_50k_type_a, topstep_100k_type_a |
| MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G4_NOMON_O15 | MNQ | LONDON_METALS | 17.1 | 197 | topstep_50k_type_a, topstep_100k_type_a |
| MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G5_NOMON_O15 | MNQ | LONDON_METALS | 17.1 | 197 | topstep_50k_type_a, topstep_100k_type_a |
| MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G6_NOMON_O15 | MNQ | LONDON_METALS | 17.1 | 197 | topstep_50k_type_a, topstep_100k_type_a |
| MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G8_NOMON_O15 | MNQ | LONDON_METALS | 17.1 | 197 | topstep_50k_type_a, topstep_100k_type_a |
| MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075 | MNQ | NYSE_CLOSE | 17.1 | 20 | tradeify_50k, tradeify_50k_type_b, tradeify_100k_type_b |
| MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075 | MGC | LONDON_METALS | 15.9 | 92 | topstep_50k_type_a, topstep_100k_type_a |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O30_S075 | MES | CME_PRECLOSE | 12.8 | 11 | topstep_50k_mes_auto, topstep_50k_type_a, topstep_100k_type_a, self_funded_tradovate |
| MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075 | MES | COMEX_SETTLE | 12.1 | 70 | topstep_50k_type_a, topstep_100k_type_a, tradeify_50k_type_b, tradeify_100k_type_b, self_funded_tradovate |

## Minimal-delta fix queue

Ranked by sum-of-annual_r unlocked per single profile edit. **Each edit is gated on broker-fit + cost-model verification** — this list is awareness, not approval.

| fix | strategies unlocked | sum annual_r | top strategy |
|---|---:|---:|---|
| add session 'LONDON_METALS' to topstep_50k [INACTIVE] | 12 | 304.7 | MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30 (39.5) |
| add session 'CME_PRECLOSE' to tradeify_100k_type_b [INACTIVE] | 35 | 195.3 | MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O30_S075 (12.8) |
| add session 'NYSE_CLOSE' to topstep_50k_mnq_auto [ACTIVE] | 10 | 104.7 | MNQ_NYSE_CLOSE_E2_RR1.0_CB1_X_MES_ATR70 (25.3) |
| add session 'US_DATA_1000' to topstep_50k_mes_auto [INACTIVE] | 3 | 92.4 | MES_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15 (32.5) |
| add session 'LONDON_METALS' to topstep_50k_mnq_auto [ACTIVE] | 5 | 71.2 | MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G4_NOMON_O15 (17.1) |
| add session 'CME_REOPEN' to topstep_50k [INACTIVE] | 1 | 21.6 | MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4 (21.6) |
| add session 'COMEX_SETTLE' to topstep_50k_mes_auto [INACTIVE] | 2 | 21.0 | MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075 (12.1) |
| add session 'NYSE_CLOSE' to topstep_100k_type_a [INACTIVE] | 2 | 5.6 | MES_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G5_O15 (3.5) |
| add session 'US_DATA_830' to topstep_50k_mnq_auto [ACTIVE] | 2 | 0.3 | MNQ_US_DATA_830_E2_RR1.0_CB1_X_MGC_ATR70_S075 (2.4) |
| add session 'NYSE_OPEN' to topstep_50k_mes_auto [INACTIVE] | 5 | -0.3 | MES_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15_S075 (3.8) |
| add session 'US_DATA_830' to topstep_50k_mes_auto [INACTIVE] | 1 | -7.1 | MES_US_DATA_830_E2_RR1.0_CB1_OVNRNG_50_S075 (-7.1) |

---

Generated by `scripts/tools/deployment_coverage_audit.py`. Re-run to refresh; output is deterministic.
