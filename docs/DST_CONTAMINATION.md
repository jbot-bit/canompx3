# DST Contamination in Fixed Sessions

> Extracted from CLAUDE.md — Feb 2026 finding. This is the authoritative reference for DST session behavior.

## Problem

Four fixed sessions (0900, 1800, 0030, 2300) have their relationship to market events change with DST. Three (0900/1800/0030) align with their event in winter but miss by 1 hour in summer. 2300 is a special case — it NEVER aligns with the US data release but sits on opposite sides of it depending on DST. Every metric computed on these sessions (avgR, Sharpe, WR, totR) is a blended average of two different market contexts.

## Which Sessions Are Affected

| Session | Winter (std time) | Summer (DST) | Shift source |
|---------|------------------|--------------|-------------|
| 0900 | = CME open (5PM CST = 23:00 UTC = 09:00 Bris) | CME opened at 0800 Bris (5PM CDT = 22:00 UTC) | US DST |
| 1800 | = London open (8AM GMT = 08:00 UTC = 18:00 Bris) | London opened at 1700 Bris (8AM BST = 07:00 UTC) | UK DST |
| 0030 | = US equity open (9:30AM EST = 14:30 UTC = 00:30 Bris) | US equity opened at 2330 Bris (9:30AM EDT = 13:30 UTC) | US DST |
| 2300 | 30min BEFORE US data (8:30 EST = 13:30 UTC; 2300 = 13:00 UTC) | 30min AFTER US data (8:30 EDT = 12:30 UTC; 2300 = 13:00 UTC) | US DST |

### 2300 Special Case

2300 Brisbane (13:00 UTC) never catches the US data release (8:30 ET). DST flips which side: pre-data in winter, post-data in summer. Summer has 76-90% MORE volume. Winter/summer split is meaningful; `"US"` classification in `dst.py` is correct.

## Clean vs. Contaminated

**Clean sessions (no DST issue):** 1000/1100/1130 (Asia, no DST). All dynamic sessions: CME_OPEN, US_EQUITY_OPEN, US_DATA_OPEN, LONDON_OPEN, US_POST_EQUITY, CME_CLOSE (resolvers adjust per-day).

**Contaminated:** All daily_features/orb_outcomes/experimental_strategies/validated_setups/edge_families for 0900/1800/0030/2300. All pre-Feb 2026 blended research findings for those sessions.

**Not contaminated:** 1000/1100/1130 results. Dynamic session results. Raw bars_1m/bars_5m (correct UTC data). ORB computation is correct — the clock time maps to different market events depending on DST, that's the issue.

## Remediation Status (Feb 2026 — DONE)

- ✅ Validator split: DST columns on both strategy tables, auto-migrated by `init_trading_app_schema()`.
- ✅ Revalidation: 1272 strategies — 275 STABLE, 155 WINTER-DOM, 130 SUMMER-DOM. No validated broken. CSV: `research/output/dst_strategy_revalidation.csv`.
- ✅ Volume analysis: event-driven edges confirmed. `research/output/volume_dst_findings.md`.
- ✅ Time scan: `research/research_orb_time_scan.py`. New candidates all rejected.
- ✅ DST columns live in production gold.db (942 validated_setups, 464 with DST splits; 12,996 experimental, 2,304 with DST splits).

**937 validated strategies exist** across all instruments. 2300: 4 (MGC G8+). 0030: 44 (MES 31, MNQ 13). Do NOT deprecate.

## Canonical Time Model

All bars are stored in UTC. All session labels (0900, 1000, 1800, etc.) are Brisbane-local clock slots. DST verdict is computed per trading day and attached to outcomes and strategies. The mapping from clock slot to market event shifts with DST — that's the contamination. The data itself is correct; the *interpretation* changes.

## Implementation Rules

All DST logic lives in `pipeline/dst.py`. **Never hardcode UTC offsets for market events.**

Key functions:
- `is_us_dst(trading_day)` / `is_uk_dst(trading_day)` — boolean DST check per date
- `is_winter_for_session(trading_day, orb_label)` — returns True/False/None (None = clean session, no DST issue)
- `classify_dst_verdict(winter_avg_r, summer_avg_r, ...)` — labels strategies STABLE/WINTER-DOM/SUMMER-DOM
- Dynamic session resolvers (return Brisbane `(hour, minute)` per trading day):
  - `cme_open_brisbane()`, `us_equity_open_brisbane()`, `us_data_open_brisbane()`
  - `london_open_brisbane()`, `us_post_equity_brisbane()`, `cme_close_brisbane()`

## DST Regime Lookup

| Session | DST source | Regime function |
|---------|-----------|----------------|
| 0900 | US | `pipeline/dst.py` → `classify_dst("US", date)` |
| 1800 | UK | `pipeline/dst.py` → `classify_dst("UK", date)` |
| 0030 | US | `pipeline/dst.py` → `classify_dst("US", date)` |
| 2300 | US | `pipeline/dst.py` → `classify_dst("US", date)` |
