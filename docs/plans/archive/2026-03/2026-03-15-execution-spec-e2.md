---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# ExecutionSpec E2 Fix

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix `execution_spec.py` to allow E2 (live stop-market) and reject E3 (soft-retired). Currently it's backwards.

**Architecture:** Single file change + 5 test updates. 15-minute job.

**Tech Stack:** Python, pytest

---

## Task 1: Fix ExecutionSpec and Update Tests

**Files:**
- Modify: `trading_app/execution_spec.py` line 46
- Modify: `tests/test_trading_app/test_execution_spec.py` (5 locations)

**Context:** `execution_spec.py:46` only allows `["E1", "E3"]`. E2 is live, E3 is soft-retired. All 5 existing tests that reference E3 as valid must be updated to use E2.

**Step 1: Run current tests to baseline**
```bash
python -m pytest tests/test_trading_app/test_execution_spec.py -v
```
All should PASS (confirms current state before breaking anything).

**Step 2: Fix `execution_spec.py` line 46**

Change:
```python
        if self.entry_model not in ["E1", "E3"]:
            raise ValueError(f"entry_model must be E1/E3, got {self.entry_model}")
```
To:
```python
        if self.entry_model not in ["E1", "E2"]:
            raise ValueError(f"entry_model must be E1/E2, got {self.entry_model}")
```

**Step 3: Update `test_execution_spec.py` — 5 locations**

1. `test_valid_entry_models` — change `["E1", "E3"]` to `["E1", "E2"]`
2. `test_invalid_entry_model` — change match string to `"entry_model must be E1/E2"`
3. `test_invalid_entry_model_legacy` — same match string update
4. `test_from_json` — change `entry_model="E3"` to `entry_model="E2"`
5. `test_roundtrip` — change `entry_model="E3"` to `entry_model="E2"`
6. `test_str_includes_entry_model` — change `entry_model="E3"` / assert `"E2" in str(spec)`

**Step 4: Run to verify all PASS**
```bash
python -m pytest tests/test_trading_app/test_execution_spec.py -v
python pipeline/check_drift.py
```

**Step 5: Commit**
```bash
git add trading_app/execution_spec.py tests/test_trading_app/test_execution_spec.py
git commit -m "fix: allow E2 and reject E3 in ExecutionSpec validation (dead code trap)"
```
