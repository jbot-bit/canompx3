# Handoff: Current Keepers / Contenders

Last updated: 2026-02-22
Purpose: quick takeover file for another AI/session.

## Canonical sources
- `research/output/shinies_shortlist.md` (human narrative + statuses)
- `research/output/shinies_registry.csv` (structured registry)
- `research/output/shinies_overlay_rank.csv` (cross-strategy add-on ranking)
- `research/output/shinies_overlay_detail.csv` (per-strategy overlay metrics)

## KEEP set (at-a-glance)
- A0: `M6E_US_EQUITY_OPEN -> MES_US_EQUITY_OPEN` | `E0/CB1/RR3.0`
- A1: `M6E_US_EQUITY_OPEN -> M2K_US_POST_EQUITY` | `E1/CB5/RR1.5`
- A2: `MES_US_DATA_OPEN -> M2K_US_DATA_OPEN` | `E0/CB1/RR1.5`
- A3: `MES_1000 -> M2K_US_POST_EQUITY` | `E1/CB5/RR1.5`
- B1: `M2K_1000 -> MES_1000` | `E0/CB1/RR2.5` (higher-frequency)
- B2: `MES_1000 fast<=15` | `E0/CB1/RR2.5` (higher-frequency)

## WATCH
- W1: `M6E_US_EQUITY_OPEN -> MES_US_EQUITY_OPEN` asymmetry challenger `same_short_fast15`

## KILL
- K1: relay chain (`M6E + MES -> M2K_US_POST_EQUITY`) underperformed single-filter core avgR.

## Add-on overlays to test for deployment presets
- `OV_f_vol60` (follower breakout volume impulse top 40%)
- `OV_f_fast15` (follower break delay <= 15m)

## Guardrails
- No-lookahead enforced in lead-lag: `leader_break_ts <= follower entry_ts`
- Baseline-positive alone is NOT sufficient; require filtered uplift + practical tradability.
