# System Audit Report — 2026-02-25

## Executive Summary

**Audit Scope:** Database schema, outcome/strategy pipeline, session time configurations.
**Triggered by:** Accumulated changes from session architecture overhaul, strategy audit fixes, E0 CB2+ purge, FDR correction, T80 time-stop additions.

### System Health Baseline

| Check | Result |
|-------|--------|
| Drift detection (31 checks) | ALL PASSED |
| Test suite (1801 items) | 1800 passed, 1 skipped |
| Migration status (old session names) | CLEAN — 0 old names in DB |
| Schema integrity (old columns) | CLEAN — 0 old columns in daily_features |

### Findings Summary

| Severity | Count | Details |
|----------|-------|---------|
| HIGH | 2 | F-05 prospective_signals schema, F-15 prospective_tracker old names |
| MEDIUM | 5 | F-01 uncommitted changes, F-03 outcome counts stale in MEMORY, F-07 SINGAPORE_OPEN exclusion, F-12 ORB_LABELS sync guard, F-13 edge families |
| LOW | 6 | F-06 test fixtures, F-08 VolumeFilter, F-09 .gitignore, F-10 dead instruments, F-11 comment, F-14 DST verdicts |
| RESOLVED | 1 | F-04 T80 already wired |

### Top 3 Items Requiring Attention

1. **F-05 + F-15 (HIGH):** `prospective_signals.session` is INTEGER (should be VARCHAR) and `prospective_tracker.py` references old `orb_0900_*` columns that no longer exist after migration.
2. **F-12 (MEDIUM):** No drift check guards ORB_LABELS vs SESSION_CATALOG synchronization — a new session added to one but not the other would silently fail.
3. **F-07 (MEDIUM):** 67 active SINGAPORE_OPEN strategies are excluded from fitness monitoring by a hardcoded filter with no explanatory comment.

---

## Issue Registry

| ID | Severity | Title | Status | Remediated |
|----|----------|-------|--------|------------|
| F-01 | MEDIUM | 6 uncommitted files (session cleanup) | CONFIRMED | YES — committed |
| F-02 | HIGH | Migration execution verification | VERIFIED CLEAN | N/A |
| F-03 | MEDIUM | Outcome counts differ from MEMORY.md | CONFIRMED | YES — MEMORY updated |
| F-04 | ~~LOW~~ | T80 not wired in paper_trader | RESOLVED | Already in execution_engine.py:901 |
| F-05 | HIGH | `prospective_signals.session` is INTEGER, should be VARCHAR | CONFIRMED | YES — schema fixed |
| F-06 | LOW | Old session names in ~25 test fixtures | CONFIRMED | DEFERRED |
| F-07 | MEDIUM | SINGAPORE_OPEN hardcoded exclusion in strategy_fitness.py:417 | CONFIRMED | YES — documented + config constant |
| F-08 | LOW | VolumeFilter returns empty in fitness context | CONFIRMED | DEFERRED |
| F-09 | LOW | `gold.db.bak*` not in .gitignore | CONFIRMED | YES — pattern added |
| F-10 | LOW | Dead instruments (MCL/SIL/M6E) in asset_configs.py | CONFIRMED | NO ACTION NEEDED |
| F-11 | LOW | SESSION_WINDOWS comment could be clearer | CONFIRMED | NO ACTION NEEDED |
| F-12 | MEDIUM | No guard: ORB_LABELS vs SESSION_CATALOG sync | CONFIRMED | YES — drift check #32 added |
| F-13 | MEDIUM | Edge family rebuild after migration | VERIFIED | Edge families are current |
| F-14 | LOW | DST verdict columns always NULL for post-migration strategies | EXPECTED | NO ACTION NEEDED |
| F-15 | HIGH | `prospective_tracker.py` uses old session names entirely | CONFIRMED | YES — rewritten |

---

## Detailed Findings by Area

### A. Database Schema

**F-02: Migration Execution — VERIFIED CLEAN**

DB queries confirm complete migration:
- 0 old session names (`0900`, `1000`, `1800`, `2300`, `0030`, etc.) in `orb_outcomes`
- 10 event-based names present: CME_REOPEN, TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS, US_DATA_830, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE
- 0 old column names (`orb_0900_*`, etc.) in `daily_features`

**F-05 + F-15: Prospective Signals Schema + Tracker (HIGH)**

The `prospective_signals` table defines `session INTEGER NOT NULL` but post-migration session identifiers are text strings like "CME_REOPEN". The `prospective_tracker.py` SIGNALS dict used:
- `"session": 900` (integer)
- `"orb_label": "0900"` (old name)
- Column references `orb_0900_outcome`, `orb_0900_size` (old column names)

This subsystem was completely broken post-migration. **Remediated:** Schema changed to VARCHAR, tracker rewritten with event-based names.

**F-09: .gitignore Gap**

`gold.db.bak.phase1` (2.5GB) and `gold.db.bak.pre-migration` (2.5GB) were untracked. Pattern `gold.db.bak*` added to `.gitignore`.

**F-12: ORB_LABELS vs SESSION_CATALOG Sync Guard**

No drift check verified that `ORB_LABELS` (in `init_db.py`, drives schema generation) matches `SESSION_CATALOG` dynamic entries (in `dst.py`, drives time resolution). If someone adds a session to one but not the other:
- In ORB_LABELS only: schema has columns, but `build_daily_features.py` raises ValueError (fail-closed)
- In SESSION_CATALOG only: time is resolved, but no columns exist to store results (silent data loss)

**Remediated:** Drift check #32 added to `check_drift.py`.

### B. Outcomes & Strategies

**F-01: Uncommitted Changes**

7 files with pending session-rename cleanup changes:
- `pipeline/asset_configs.py` — comment updates (1000/1100 to TOKYO_OPEN/SINGAPORE_OPEN)
- `pipeline/build_daily_features.py` — removes dead `ORB_TIMES_LOCAL = {}`
- `pipeline/init_db.py` — removes dead `ORB_LABELS_FIXED = []`
- `tests/test_pipeline/test_init_db.py` — removes dead tests for `ORB_LABELS_FIXED`
- `trading_app/mcp_server.py` — updates example strategy_id in docstring
- `trading_app/strategy_fitness.py` — updates example strategy_id in docstring
- `research/output/forward_gate_status_latest.md` — content update

**Remediated:** Committed together.

**F-03: Outcome Counts Differ from MEMORY.md**

| Instrument | MEMORY.md | Actual | Delta |
|-----------|-----------|--------|-------|
| MGC | 522K | 743,778 | +42% |
| MES | 742K | 655,914 | -12% |
| MNQ | 190K | 416,430 | +119% |
| M2K | 531K | 471,786 | -11% |

MEMORY.md values were from a snapshot taken during an earlier rebuild. The current counts reflect outcomes rebuilt after the session migration (which added new sessions like COMEX_SETTLE, NYSE_CLOSE and expanded NQ backfill for MNQ). Additionally, dead instruments remain: MCL (133K), M6E (510K), SIL (74K).

Strategy count is 618 (not 999 per MEMORY.md) — this reflects post-migration re-validation with stricter criteria.

**Remediated:** MEMORY.md updated with current counts.

**F-07: SINGAPORE_OPEN Hardcoded Exclusion**

`strategy_fitness.py:417` excludes SINGAPORE_OPEN from portfolio fitness with:
```python
AND orb_label != 'SINGAPORE_OPEN'
```

DB shows 67 active SINGAPORE_OPEN strategies. The exclusion rationale: SINGAPORE_OPEN is documented as "inconsistent edge" in config.py line 17, and was flagged as no FDR-confirmed edges in MEMORY.md.

**Remediated:** Extracted to `EXCLUDED_FROM_FITNESS` constant in `config.py` with explanatory comment. `strategy_fitness.py` now references the constant.

**F-13: Edge Family Status**

202 edge families across 4 instruments (MGC: 74, MES: 55, MNQ: 56, M2K: 17). These correspond to the 618 validated strategies. The counts are internally consistent — edge families were rebuilt after migration.

### C. Session Time Configurations

**F-04: T80 Time-Stop — ALREADY RESOLVED**

The ACTION QUEUE in MEMORY.md listed "T80 time-stop in paper_trader" as pending. Investigation reveals `execution_engine.py:901` already checks `EARLY_EXIT_MINUTES.get(trade.orb_label)` and `paper_trader.py` handles the `"early_exit_timed"` outcome. This item is stale.

**Remediated:** MEMORY.md ACTION QUEUE updated to mark as DONE.

**F-06: Old Session Names in Test Fixtures (LOW — DEFERRED)**

~25 test files contain old session names ("0900", "1800", etc.) in synthetic test fixtures. These tests pass because they create self-contained in-memory databases. Not broken, but cosmetically stale.

**F-10/F-11: Dead Instruments and Comment Clarity (LOW — NO ACTION)**

MCL, SIL, M6E remain in `asset_configs.py` for data pipeline compatibility. `SESSION_WINDOWS` comment is adequate. No changes needed.

**F-14: DST Verdict Columns (LOW — EXPECTED)**

All DST verdicts in `validated_setups` are NULL post-migration. This is expected: `DST_AFFECTED_SESSIONS` is now empty (all sessions are dynamic/clean). The `classify_dst_verdict()` function would return "CLEAN" for any new validation run. Historical verdicts from pre-migration are gone (strategies were re-validated).

---

## Dependency Map

```
F-01 (commit cleanup) ─────────────────── standalone
F-02 (verify migration) ──┬── gates ──── F-03 (outcome counts)
                          └── gates ──── F-13 (edge families)
F-05 + F-15 (prospective) ─────────────── coupled pair
F-07 (SINGAPORE exclusion) ─────────────── standalone
F-09 (.gitignore) ─────────────────────── standalone
F-12 (drift check #32) ───────────────── standalone
```

---

## Confirmed Healthy Systems

- **Session architecture:** 10 dynamic event-based sessions, all resolved per-day from `pipeline/dst.py SESSION_CATALOG`. Zero production code references old names.
- **DST handling:** Fully resolved via `zoneinfo`. `DST_AFFECTED_SESSIONS = {}`.
- **DOW alignment:** Runtime guard `validate_dow_filter_alignment()` catches misaligned DOW filters at strategy creation time.
- **Daily features JOIN rules:** 3-column join (trading_day, symbol, orb_minutes) documented in `.claude/rules/daily-features-joins.md` and enforced in MCP tools.
- **Strategy pipeline:** 7-phase validation, FDR correction, regime waivers. 618 active validated strategies across 4 instruments.
- **MCP server:** 4 read-only tools, template-only queries, row-capped at 5000, parameter-allowlisted.
- **Drift detection:** 31 checks all passing (now 32 with F-12 addition).
- **Test suite:** 1800 tests passing, 1 skipped.
- **Execution engine:** T80 early exit wired, calendar filters, ATR velocity overlay functional.
- **E0 CB1-only guard:** Drift check #30 prevents look-ahead CB2+ regression.

---

## Remediation Roadmap

### Batch A — Immediate (completed)
- [x] F-01: Committed 7 pending cleanup files
- [x] F-02: Verified migration clean via DB queries
- [x] F-03: Updated MEMORY.md with current counts
- [x] F-09: Added `gold.db.bak*` to `.gitignore`

### Batch B — Short-term (completed)
- [x] F-12: Added drift check #32 (ORB_LABELS vs SESSION_CATALOG sync)
- [x] F-05 + F-15: Fixed prospective_signals schema (INTEGER → VARCHAR) + rewrote tracker
- [x] F-07: Extracted `EXCLUDED_FROM_FITNESS` constant, documented rationale

### Batch C — Low-priority (deferred)
- [ ] F-06: Batch-update ~25 test fixtures to new session names
- [ ] F-08: Add warning log for VolumeFilter strategies in fitness context

---

## Best Practices Recommendations

1. **Session-name sync guard (F-12):** Drift check #32 now prevents future session additions from silently failing. Any new session must be added to BOTH `ORB_LABELS` in `init_db.py` AND `SESSION_CATALOG` in `dst.py`.

2. **Prospective tracker config:** The tracker should import session parameters from `config.py` rather than hardcoding them. The current fix uses event-based names but still hardcodes strategy parameters in the SIGNALS dict.

3. **MEMORY.md hygiene:** The ACTION QUEUE and counts drifted significantly from reality. Consider a periodic "MEMORY refresh" as part of major milestone completion.

4. **Pre-commit hook coverage:** The existing `.githooks/pre-commit` hook runs drift detection. Consider adding a lightweight session-name grep check to catch old names in new code before they reach CI.
