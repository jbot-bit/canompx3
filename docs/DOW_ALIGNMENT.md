# DOW Alignment

> Extracted from CLAUDE.md — Feb 2026 verification. This is the authoritative reference for day-of-week filter alignment.

## Summary

`day_of_week` uses Brisbane trading day. For most sessions Brisbane DOW = exchange DOW. Exception: **0030** (crosses midnight Brisbane → Brisbane DOW = US DOW + 1). Brisbane-Friday at 0030 = US Thursday.

## Active DOW Filters

All three active DOW filters are correctly aligned:

| Filter | Session | Meaning |
|--------|---------|---------|
| NOFRI | 0900 | Skip Brisbane Friday (= exchange Friday) |
| NOMON | 1800 | Skip Brisbane Monday (= exchange Monday) |
| NOTUE | 1000 | Skip Brisbane Tuesday (= exchange Tuesday) |

## Runtime Guard

`validate_dow_filter_alignment()` in `pipeline/dst.py` prevents DOW filters on misaligned sessions at runtime. This ensures no new DOW filter can be applied to a session where Brisbane DOW ≠ exchange DOW without explicit handling.

## Investigation

Full investigation: `research/research_dow_alignment.py`
Full DOW mapping: `TRADING_RULES.md` § DOW Alignment
