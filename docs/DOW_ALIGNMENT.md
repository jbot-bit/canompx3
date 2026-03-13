# DOW Alignment

> Extracted from CLAUDE.md — Feb 2026 verification. This is the authoritative reference for day-of-week filter alignment.

## Summary

`day_of_week` uses Brisbane trading day. For most sessions Brisbane DOW = exchange DOW. Exception: **NYSE_OPEN** (crosses midnight Brisbane → Brisbane DOW = US DOW + 1). Brisbane-Friday at NYSE_OPEN = US Thursday.

## Active DOW Filters

One active DOW filter in the discovery grid (Mar 2026):

| Filter | Session | Meaning | Status |
|--------|---------|---------|--------|
| NOMON | LONDON_METALS | Skip Brisbane Monday (= exchange Monday) | Active (PLAUSIBLE BUT UNPROVEN) |

NOFRI (CME_REOPEN) and NOTUE (TOKYO_OPEN) were removed from the discovery grid in Mar 2026 after the DOW Filter Stress Test found them to be LIKELY NOISE. Their definitions and ALL_FILTERS entries are retained for DB row compatibility. See `research/output/DOW_FILTER_STRESS_TEST.md`.

## Runtime Guard

`validate_dow_filter_alignment()` in `pipeline/dst.py` prevents DOW filters on misaligned sessions at runtime. This ensures no new DOW filter can be applied to a session where Brisbane DOW ≠ exchange DOW without explicit handling.

## Investigation

Full investigation: `research/research_dow_alignment.py`
Full DOW mapping: `TRADING_RULES.md` § DOW Alignment
DOW stress test: `research/output/DOW_FILTER_STRESS_TEST.md`
