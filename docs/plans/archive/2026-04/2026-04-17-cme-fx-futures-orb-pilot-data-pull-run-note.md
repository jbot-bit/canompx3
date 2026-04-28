---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# CME FX ORB Pilot Data Pull / Run Note

Databento request set: `GLBX.MDP3`, `ohlcv-1m`, research-only, `90-180` trading days. Pull parent products `6J` for `TOKYO_OPEN`; `6B` and `6A` for `LONDON_METALS`, `US_DATA_830`, `US_DATA_1000`.

Raw storage paths:
- `data/raw/databento/ohlcv-1m/6J/`
- `data/raw/databento/ohlcv-1m/6B/`
- `data/raw/databento/ohlcv-1m/6A/`

Minimal run steps:
1. Download raw 1m bars only for the locked products and pilot window.
2. Store raw files under the locked paths with request metadata beside them.
3. Build a research-only pilot slice from raw bars to session-level ORB metrics.
4. Fill the comparison table only; no validator wiring, no live config, no asset onboarding.

Pilot comparison table schema:
`asset | session | pilot_days | benchmark_dead_fx | benchmark_live_proxy | double_break_rate | fakeout_share | continuation_E1 | continuation_E2 | cost_adjusted_ExpR | median_ORB_risk_usd | friction_pct_of_median_risk | monthly_stability_note | verdict`
