# Discovery Dedup & Cleanup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land the uncommitted discovery dedup feature (trade_day_hash + canonical/alias marking inline during discovery) and clean up scratch files from root.

**Architecture:** strategy_discovery.py now computes trade_day_hash and marks canonical vs alias strategies during grid iteration, eliminating the need for a separate backfill step. The validator already skips aliases (sets validation_status='SKIPPED'). The portfolio already supports `--family-heads-only` via edge_families. This commit completes the "discovery auto-populate" item.

**Tech Stack:** Python, DuckDB, pytest

---

## Context for the implementing engineer

### What changed in strategy_discovery.py (uncommitted)

- Added `_compute_trade_day_hash()` — MD5 of sorted trade days, same algorithm as `build_edge_families.py`
- Added `_mark_canonical()` — groups strategies by (instrument, orb_label, entry_model, rr_target, confirm_bars, trade_day_hash), picks highest filter specificity as canonical (G8 > G6 > G5 > G4 > VOL > NO_FILTER)
- Added `_FILTER_SPECIFICITY` ranking dict and `_INSERT_SQL` constant
- Grid iteration now collects all strategies in memory first, then calls `_mark_canonical()`, then batch-writes with `trade_day_hash`, `is_canonical`, `canonical_strategy_id` columns populated
- Old code flushed to DB every 500 rows during iteration; new code collects all, dedup, then writes

### What changed in test_strategy_discovery.py (uncommitted)

- `TestComputeMetricsScratchCounts` — 3 tests for entry_signals/scratch_count/early_exit_count
- `TestDedup` — 2 tests for `_mark_canonical` and `_compute_trade_day_hash`
- `TestValidatorSkipsAliases` — 1 integration test that alias rows get SKIPPED status

### Schema (already committed)

`experimental_strategies` already has columns: `trade_day_hash TEXT`, `is_canonical BOOLEAN DEFAULT TRUE`, `canonical_strategy_id TEXT` (added in db_manager.py).

### Downstream already wired (already committed)

- `strategy_validator.py:269` — checks `is_canonical is False` → skips with SKIPPED status
- `portfolio.py:253` — `family_heads_only` param uses `edge_families` table
- `db_manager.py:375` — `get_family_head_ids()` helper

### Scratch files to delete

5 files in project root that are external AI notes, not code:
- `fixes.txt`, `fixes1.txt`, `notes.txt`, `notes1.txt`, `walk.txt`

---

## Tasks

### Task 1: Run tests on uncommitted changes

**Files:**
- Read: `trading_app/strategy_discovery.py` (modified)
- Read: `tests/test_trading_app/test_strategy_discovery.py` (modified)

**Step 1: Run discovery tests only**

Run: `python -m pytest tests/test_trading_app/test_strategy_discovery.py -v`
Expected: All tests PASS including the 6 new ones (TestComputeMetricsScratchCounts x3, TestDedup x2, TestValidatorSkipsAliases x1)

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All ~1,045+ tests PASS, 0 failures

**Step 3: Run drift check**

Run: `python pipeline/check_drift.py`
Expected: 21/21 checks PASS

---

### Task 2: Delete scratch files

**Files:**
- Delete: `fixes.txt`
- Delete: `fixes1.txt`
- Delete: `notes.txt`
- Delete: `notes1.txt`
- Delete: `walk.txt`

**Step 1: Delete all 5 scratch files**

```bash
rm fixes.txt fixes1.txt notes.txt notes1.txt walk.txt
```

**Step 2: Verify root is clean**

```bash
ls *.txt
```
Expected: No .txt files in project root (only .md, .py, .gitignore, etc.)

---

### Task 3: Commit

**Files:**
- Stage: `trading_app/strategy_discovery.py`
- Stage: `tests/test_trading_app/test_strategy_discovery.py`

**Step 1: Stage the 2 modified files**

```bash
git add trading_app/strategy_discovery.py tests/test_trading_app/test_strategy_discovery.py
```

**Step 2: Commit with descriptive message**

```bash
git commit -m "feat: inline trade-day hash + canonical dedup in strategy discovery

Discovery now computes trade_day_hash and marks canonical vs alias
strategies during grid iteration. Eliminates separate backfill step.
Validator skips aliases (SKIPPED status). Highest filter specificity
wins canonical election (G8 > G6 > G5 > G4 > VOL > NO_FILTER).

Adds 6 tests: scratch/early_exit counts, dedup logic, alias skipping."
```

**Step 3: Verify clean state**

```bash
git status
```
Expected: Only untracked files remaining (if any). No modified tracked files.

---

### Task 4: Update memory

**Files:**
- Modify: `C:\Users\joshd\.claude\projects\C--Users-joshd-OneDrive-Desktop-Canompx3\memory\MEMORY.md`

**Step 1: Update pending section**

Remove "discovery auto-populate" from pending list. Mark edge families integration as complete. Update project state with new test count.

---

## Post-plan: what's next after this lands

With discovery dedup + portfolio family_heads_only + edge_families all wired:

1. **Re-run discovery + validation** for all instruments with new dedup (optional — quantifies alias reduction)
2. **Decisions**: LONDON_OPEN in live_config? MNQ US_EQUITY_OPEN?
3. **Parked**: Full MGC rebuild 2016-2026 for CORE-class sample sizes
4. **Nested research**: Rebuild nested tables for 1000 15m ORB
