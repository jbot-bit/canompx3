# Pipeline Data Audit — 2026-02-25

**Scope:** Full pipeline audit against live `gold.db` data. All 13 tables profiled. Cross-table integrity verified.
**Database:** gold.db (DuckDB), 13 tables
**Trigger:** Post session-rename migration + recent pipeline changes

---

## Executive Summary

The pipeline is structurally sound. Bars data has zero integrity violations (no high<low, no null OHLC). Entry model rules are correctly enforced (E0 CB1-only, E3 CB1-only, E0 CB2+ blocked). Cross-table joins have zero orphans. One critical finding requires immediate attention: the compression tier columns are entirely NULL at orb_minutes=5 for CME_REOPEN and LONDON_METALS, meaning the ATR Velocity AVOID signal cannot fire for the primary trading aperture. A secondary finding: 297 SINGAPORE_OPEN validated strategies remain active despite the session being permanently excluded.

---

## Findings by Severity

### CRITICAL — Compression Tier NULL at orb_minutes=5

| Session | orb_minutes=5 (N=11,277) | orb_minutes=15 (N=7,016) |
|---------|--------------------------|--------------------------|
| CME_REOPEN | compression_z: **0/11,277** | compression_z: 1,287/7,016 |
| TOKYO_OPEN | compression_z: 8,237/11,277 | compression_z: 1,286/7,016 |
| LONDON_METALS | compression_z: **0/11,277** | compression_z: 1,285/7,016 |

**Impact:** `TRADING_RULES.md` documents the ATR Velocity + ORB Compression AVOID signal as VALIDATED and wired into execution for CME_REOPEN and LONDON_METALS. The columns `orb_CME_REOPEN_compression_z`, `orb_CME_REOPEN_compression_tier`, `orb_LONDON_METALS_compression_z`, and `orb_LONDON_METALS_compression_tier` are ALL NULL at orb_minutes=5 — the primary ORB aperture. Only TOKYO_OPEN is populated at 5m (8,237 of 11,277 rows). All three sessions are populated at orb_minutes=15 and 30.

**Root cause hypothesis:** `build_daily_features.py` likely computes compression using a lookback window that requires prior ORB data from the same session. At orb_minutes=5, CME_REOPEN and LONDON_METALS may not have enough qualifying bars to compute the z-score on the first pass. Alternatively, the compression calculation may be gated on a session that only exists at 15m+.

**Action required:** Investigate `build_daily_features.py` compression calculation. If this is by design (compression needs a wider aperture), document it and remove the AVOID signal from 5m execution specs. If it's a bug, fix and rebuild.

---

### WARNING — 297 SINGAPORE_OPEN Validated Strategies Still Active

`validated_setups` contains 297 rows with `orb_label = 'SINGAPORE_OPEN'`, all with `status = 'active'`. TRADING_RULES.md documents SINGAPORE_OPEN as **PERMANENTLY OFF** (74% double-break rate). Breakdown:

| Instrument | Count |
|-----------|-------|
| MNQ | 234 |
| MES | 57 |
| M2K | 6 |

These strategies would never fire in live trading (session is excluded), but they inflate validated_setups counts and could confuse edge family analysis. They should be retired with `retirement_reason = 'SINGAPORE_OPEN_EXCLUDED'`.

---

### INFO — MNQ 15m ORB Coverage Gap

MNQ bars_1m data starts Feb 2021 (native MNQ contracts), but the 15m ORB aperture in `orb_outcomes` only has 584 rows — roughly from Feb 2024 onward. The 5m aperture has 1,462 rows (full history) and the 30m has 1,462 rows. This suggests `build_daily_features.py` or `outcome_builder.py` started generating 15m ORB data for MNQ later than 5m/30m. Not blocking for current trading (5m is the primary aperture), but worth investigating if 15m strategies are ever pursued for MNQ.

---

## Table-by-Table Audit Results

### bars_1m — PASS

| Check | Result |
|-------|--------|
| Total rows | 13,553,175 |
| Instruments | 7 (MGC, MES, MNQ, M2K, M6E, MCL, SIL) |
| high < low violations | **0** |
| NULL OHLC | **0** |
| Source symbol mapping | Correct (GC→MGC, ES→MES, RTY→M2K, 6E→M6E, SI→SIL; MNQ/MCL native) |

Coverage by instrument:
- **MGC**: Feb 2016 → Feb 2026 (~10 years, via GC full-size)
- **MES**: Feb 2019 → Feb 2026 (~7 years, ES→MES hybrid at Feb 2024)
- **MNQ**: Feb 2021 → Feb 2026 (~5 years, NQ→MNQ hybrid at Feb 2024)
- **M2K**: Feb 2021 → Feb 2026 (~5 years, via RTY)
- **M6E**: Feb 2021 → Feb 2026 (~5 years, via 6E) — NO-GO instrument
- **MCL**: Jul 2021 → Feb 2026 (~4.5 years, native) — NO-GO instrument
- **SIL**: Feb 2024 → Feb 2026 (~2 years, via SI) — NO-GO instrument

### bars_5m — PASS

| Check | Result |
|-------|--------|
| Total rows | 2,717,736 |
| 5m:1m ratio | ~5.0x for core instruments (correct) |
| Slightly below 5.0x | M2K, M6E, MCL, SIL (expected: partial first/last bars at contract boundaries) |
| high < low violations | **0** |

### daily_features — PASS (except compression — see CRITICAL)

| Check | Result |
|-------|--------|
| Total rows | 26,187 |
| Grain | (trading_day, symbol, orb_minutes) — **verified unique** |
| orb_minutes values | 5, 15, 30 (3 rows per day per symbol) |
| Session coverage | All 10 event-based sessions present |
| ATR columns | Populated, no nulls on active instruments |
| ORB size columns | Correct — M6E shows sub-pip values (e.g., 0.0005), not literal zeros |

### orb_outcomes — PASS

| Check | Result |
|-------|--------|
| Total rows | 5,802,294 |
| Entry models | E0, E1, E3 all present |
| E0 confirm_bars | CB1 only — **correct** (E0 CB2+ removed for look-ahead bias) |
| E0 CB2+ rows | **0** — confirmed purged |
| E3 confirm_bars | CB1 only — **correct** (CB1-5 identical for limit retrace) |
| E1 confirm_bars | CB1, CB2, CB3, CB4, CB5 — full range |
| RR targets | 6 levels (1.0, 1.5, 2.0, 2.5, 3.0, 4.0) — balanced |
| Outcome types | win, loss, scratch, time_stop — correct |
| scratch pnl_r | 100% NULL — **correct** (scratches have no P&L) |
| early_exit outcomes | Only M6E and MCL (NO-GO instruments) |

### validated_setups — PASS (except SINGAPORE_OPEN — see WARNING)

| Check | Result |
|-------|--------|
| Total rows | 1,322 |
| Status | All `active` (no retired, no NULL) |
| Sessions represented | 8 sessions (CME_PRECLOSE, CME_REOPEN, LONDON_METALS, NYSE_CLOSE, NYSE_OPEN, SINGAPORE_OPEN, TOKYO_OPEN, US_DATA_830, US_DATA_1000) |
| Active instruments | MGC (339), MES (273), MNQ (648), M2K (96) — no dead instruments |
| COMEX_SETTLE / NYSE_CLOSE | Present (new sessions from recent migration) |

### edge_families — PASS

| Check | Result |
|-------|--------|
| Total rows | 472 |
| Orphan family hashes (vs validated_setups) | **0** |
| All families link back to validated strategies | **Yes** |

### Cross-Table Integrity — PASS

| Check | Result |
|-------|--------|
| orb_outcomes days with no daily_features match | **0 orphans** |
| validated_setups with no edge_family match | **0 orphans** |
| edge_families with no validated_setup match | **0 orphans** |

---

## Action Items

| Priority | Item | Owner |
|----------|------|-------|
| **P0** | Investigate compression_z/compression_tier NULL at orb_minutes=5 for CME_REOPEN and LONDON_METALS in `build_daily_features.py`. Either fix the calculation or update TRADING_RULES.md to remove the AVOID signal from 5m execution specs. | Pipeline |
| **P1** | Retire 297 SINGAPORE_OPEN validated strategies (`UPDATE validated_setups SET status='retired', retired_at=NOW(), retirement_reason='session_permanently_excluded' WHERE orb_label='SINGAPORE_OPEN'`) | Trading App |
| **P2** | Investigate MNQ 15m ORB coverage gap (584 rows vs 1,462 for 5m/30m) — low priority unless 15m strategies are pursued | Pipeline |

---

## Audit Methodology

All queries run directly against `gold.db` via Python/DuckDB (read-only). No data was modified. Profiling covered: row counts, PK uniqueness, null rates, value distributions, cross-table foreign key integrity, and business rule verification (entry model constraints, filter logic, outcome types). MCP tools used for schema discovery and table counts; direct SQL for detailed profiling.

**Date:** 2026-02-25
**Auditor:** Claude (data-exploration skill)
