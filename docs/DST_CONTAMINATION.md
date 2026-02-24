# DST Contamination — FULLY RESOLVED (Feb 2026)

> Historical reference document. DST contamination is no longer an active issue. All sessions are now event-based with per-day time resolution.

## Resolution

All sessions have been migrated from fixed-clock names to event-based names. Session times are resolved per-day from `pipeline/dst.py` SESSION_CATALOG, eliminating DST contamination entirely.

| Old Name | New Name | Event |
|----------|----------|-------|
| 0900 | CME_REOPEN | CME daily reopen (5:00 PM CT) |
| 1800 | LONDON_METALS | London metals open (8:00 AM UK) |
| 0030 | NYSE_OPEN | NYSE equity open (9:30 AM ET) |
| 2300 | US_DATA_830 | US economic data release (8:30 AM ET) |
| 1000 | TOKYO_OPEN | Tokyo open (always clean — no DST in Asia) |
| 1100/1130 | SINGAPORE_OPEN | Singapore/HK open (always clean) |

## Historical Problem (for reference)

Four old fixed sessions (0900, 1800, 0030, 2300) had their relationship to market events change with DST. Three (0900/1800/0030) aligned with their event in winter but missed by 1 hour in summer. 2300 was a special case — it NEVER aligned with the US data release but sat on opposite sides of it depending on DST.

### US_DATA_830 (formerly 2300) Special Case

The old 2300 Brisbane (13:00 UTC) never caught the US data release (8:30 ET). DST flipped which side: pre-data in winter, post-data in summer. Summer had 76-90% MORE volume. The new US_DATA_830 session resolves to the actual 8:30 AM ET time each day.

## Remediation History (Feb 2026)

- Validator split: DST columns on both strategy tables, auto-migrated by `init_trading_app_schema()`.
- Revalidation: 1272 strategies — 275 STABLE, 155 WINTER-DOM, 130 SUMMER-DOM. No validated broken. CSV: `research/output/dst_strategy_revalidation.csv`.
- Volume analysis: event-driven edges confirmed. `research/output/volume_dst_findings.md`.
- Time scan: `research/research_orb_time_scan.py`. New candidates all rejected.
- DST columns remain in production gold.db for historical reference.
- All sessions migrated to event-based names (Feb 2026). DST contamination is fully resolved.

## Canonical Time Model

All bars are stored in UTC. Session labels are now event-based names (CME_REOPEN, TOKYO_OPEN, etc.) with per-day time resolution from `pipeline/dst.py`. DST verdict columns remain in the database for historical analysis. The old fixed-clock labels (0900, 1000, 1800, etc.) have been replaced.

## Implementation

All session resolution logic lives in `pipeline/dst.py`. **Never hardcode UTC offsets for market events.**

Key functions:
- `is_us_dst(trading_day)` / `is_uk_dst(trading_day)` — boolean DST check per date
- `is_winter_for_session(trading_day, orb_label)` — returns True/False/None (None = no DST relevance)
- `classify_dst_verdict(winter_avg_r, summer_avg_r, ...)` — labels strategies STABLE/WINTER-DOM/SUMMER-DOM
- Session resolvers in SESSION_CATALOG resolve per-day times for all event-based sessions

## DST Regime Lookup (Historical Reference)

| Session | DST source | Regime function |
|---------|-----------|----------------|
| CME_REOPEN | US | `pipeline/dst.py` → `classify_dst("US", date)` |
| LONDON_METALS | UK | `pipeline/dst.py` → `classify_dst("UK", date)` |
| NYSE_OPEN | US | `pipeline/dst.py` → `classify_dst("US", date)` |
| US_DATA_830 | US | `pipeline/dst.py` → `classify_dst("US", date)` |
