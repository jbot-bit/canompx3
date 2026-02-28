# WF Start-Date Override Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Allow per-instrument walk-forward start-date override so MGC's WF windows skip the pre-2022 low-gold regime where G4+ filters produce <15 trades per window.

**Architecture:** Config-level `WF_START_OVERRIDE` dict in `config.py`. Only WF window generation is affected — full-sample validation (Phase A) still uses all data. The override is threaded through `strategy_validator.py` → `_walkforward_worker()` → `run_walkforward()`.

**Tech Stack:** Python, DuckDB, pytest

**Design Doc:** `docs/plans/2026-02-28-wf-start-override-design.md`

---

### Task 0: Add WF_START_OVERRIDE to config.py

**Files:**
- Modify: `trading_app/config.py:76` (after existing imports)

**Step 1: Add the constant**

`config.py` already imports `dataclass` and `json`. Add `date` import and the override dict near the top of the file, after the existing imports (line 76-77):

```python
from datetime import date

# Walk-forward start-date override per instrument.
# Full-sample validation (Phase A) uses ALL data. Only WF window generation
# starts from max(earliest_outcome, override_date).
# Rationale: Gold tripled $1,300→$3,500+ (2016→2026). G4+ filters produce
# <15 trades/window before 2022 = INVALID windows under anchored WF.
WF_START_OVERRIDE: dict[str, date] = {
    "MGC": date(2022, 1, 1),  # Gold <$1800 pre-2022 = tiny ORBs, G4+ windows invalid
}
```

**Step 2: Verify import works**

Run: `python -c "from trading_app.config import WF_START_OVERRIDE; print(WF_START_OVERRIDE)"`
Expected: `{'MGC': datetime.date(2022, 1, 1)}`

**Step 3: Run tests**

Run: `pytest tests/ -x -q --timeout=30`
Expected: All pass (no behavior change yet)

**Step 4: Commit**

```bash
git add trading_app/config.py
git commit -m "feat: add WF_START_OVERRIDE config for per-instrument WF anchoring"
```

---

### Task 1: Add wf_start_date param to walkforward.py

**Files:**
- Modify: `trading_app/walkforward.py:52-121`
- Test: `tests/test_trading_app/test_walkforward.py`

**Step 1: Write the failing test**

Add to `tests/test_trading_app/test_walkforward.py` inside `class TestWalkForward`:

```python
def test_wf_start_date_override(self, con):
    """wf_start_date shifts window anchor forward, skipping early data."""
    # 2016-2025: 10 years of data
    outcomes = _monthly_outcomes(2016, 2025)
    _insert_outcomes(con, outcomes)

    # With override=2022-01-01:
    # anchor = max(2016-01-10, 2022-01-01) = 2022-01-01
    # First test window starts: 2022-01-01 + 12mo = 2023-01-01
    result = run_walkforward(
        con=con, strategy_id="TEST_OVERRIDE", instrument="MGC",
        wf_start_date=date(2022, 1, 1),
        **_WF_BASE,
    )

    # Should have ~6 valid windows (2023-01 to 2025-12)
    assert result.n_valid_windows >= 3
    assert result.passed is True
    # First window should start at or after 2023-01-01
    first_window = result.windows[0]
    assert first_window["window_start"] >= "2023-01-01"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading_app/test_walkforward.py::TestWalkForward::test_wf_start_date_override -v`
Expected: FAIL — `run_walkforward() got an unexpected keyword argument 'wf_start_date'`

**Step 3: Implement the change**

In `trading_app/walkforward.py`, modify `run_walkforward()`:

1. Add parameter after `dst_regime` (line 67):
```python
    wf_start_date: date | None = None,
```

2. Replace line 121:
```python
    # Old: window_start = _add_months(earliest, min_train_months)
    # New: Apply per-instrument WF start override
    anchor = max(earliest, wf_start_date) if wf_start_date else earliest
    window_start = _add_months(anchor, min_train_months)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trading_app/test_walkforward.py::TestWalkForward::test_wf_start_date_override -v`
Expected: PASS

**Step 5: Run full WF test suite**

Run: `pytest tests/test_trading_app/test_walkforward.py -v`
Expected: All existing tests still pass (None default = no behavior change)

**Step 6: Commit**

```bash
git add trading_app/walkforward.py tests/test_trading_app/test_walkforward.py
git commit -m "feat: add wf_start_date parameter to run_walkforward()"
```

---

### Task 2: Thread override through strategy_validator.py

**Files:**
- Modify: `trading_app/strategy_validator.py` (3 touch points)

**Step 1: Add import**

At the top of `strategy_validator.py`, add to the existing config imports:

```python
from trading_app.config import WF_START_OVERRIDE
```

**Step 2: Add wf_start_date to _walkforward_worker()**

In `_walkforward_worker()` (line 456), add parameter:

```python
def _walkforward_worker(
    strategy_id: str,
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
    filter_params: str | None,
    orb_minutes: int,
    db_path_str: str,
    wf_params: dict,
    dst_regime: str | None,
    dst_verdict_from_discovery: str | None,
    dst_cols_from_discovery: dict | None,
    wf_start_date: date | None = None,  # <-- NEW
) -> dict:
```

Add `from datetime import date` import inside the worker (or at module level if not already there).

Then pass it to `run_walkforward()` call (line 496):

```python
            wf_result = run_walkforward(
                con=con,
                strategy_id=strategy_id,
                instrument=instrument,
                orb_label=orb_label,
                entry_model=entry_model,
                rr_target=rr_target,
                confirm_bars=confirm_bars,
                filter_type=filter_type,
                orb_minutes=orb_minutes,
                test_window_months=wf_params["test_window_months"],
                min_train_months=wf_params["min_train_months"],
                min_trades_per_window=wf_params["min_trades"],
                min_valid_windows=wf_params["min_windows"],
                min_pct_positive=wf_params["min_pct_positive"],
                dst_regime=dst_regime,
                wf_start_date=wf_start_date,  # <-- NEW
            )
```

**Step 3: Thread through _build_worker_kwargs() and run_validation()**

In `run_validation()`, after looking up the cost_spec (line 581), look up the override:

```python
    wf_start_date = WF_START_OVERRIDE.get(instrument)
```

In `_build_worker_kwargs()` (line 698), add to the returned dict:

```python
                wf_start_date=wf_start_date,
```

Note: `wf_start_date` is captured from the enclosing `run_validation()` scope via closure.

**Step 4: Run tests**

Run: `pytest tests/ -x -q --timeout=30`
Expected: All pass

**Step 5: Commit**

```bash
git add trading_app/strategy_validator.py
git commit -m "feat: thread WF_START_OVERRIDE through validator to walkforward"
```

---

### Task 3: Write comprehensive tests for WF start-date override

**Files:**
- Modify: `tests/test_trading_app/test_walkforward.py`

**Step 1: Add test_wf_start_date_none_unchanged**

```python
def test_wf_start_date_none_unchanged(self, con):
    """No override -> windows start from earliest data (backwards compat)."""
    outcomes = _monthly_outcomes(2016, 2025)
    _insert_outcomes(con, outcomes)

    result = run_walkforward(
        con=con, strategy_id="TEST_NO_OVERRIDE", instrument="MGC",
        **_WF_BASE,
    )

    # Without override, first window starts 2017-01 (earliest + 12mo)
    first_window = result.windows[0]
    assert first_window["window_start"] >= "2017-01-01"
    assert first_window["window_start"] < "2017-07-01"
```

**Step 2: Add test_wf_start_date_after_latest (fail-closed edge case)**

```python
def test_wf_start_date_after_latest(self, con):
    """Override date after all data -> no windows, fail-closed."""
    outcomes = _monthly_outcomes(2020, 2023)
    _insert_outcomes(con, outcomes)

    result = run_walkforward(
        con=con, strategy_id="TEST_FUTURE_OVERRIDE", instrument="MGC",
        wf_start_date=date(2030, 1, 1),
        **_WF_BASE,
    )

    assert result.passed is False
    assert result.n_total_windows == 0
```

**Step 3: Add test_wf_start_date_before_earliest (no-op edge case)**

```python
def test_wf_start_date_before_earliest(self, con):
    """Override before earliest data -> max() picks earliest, no change."""
    outcomes = _monthly_outcomes(2020, 2023)
    _insert_outcomes(con, outcomes)

    result_with = run_walkforward(
        con=con, strategy_id="TEST_EARLY_OVERRIDE", instrument="MGC",
        wf_start_date=date(2015, 1, 1),
        **_WF_BASE,
    )
    result_without = run_walkforward(
        con=con, strategy_id="TEST_NO_OVERRIDE2", instrument="MGC",
        **_WF_BASE,
    )

    assert result_with.n_total_windows == result_without.n_total_windows
    assert result_with.n_valid_windows == result_without.n_valid_windows
```

**Step 4: Run all tests**

Run: `pytest tests/test_trading_app/test_walkforward.py -v`
Expected: All pass (4 new + all existing)

**Step 5: Run full suite**

Run: `pytest tests/ -x -q --timeout=30`
Expected: All pass

**Step 6: Commit**

```bash
git add tests/test_trading_app/test_walkforward.py
git commit -m "test: comprehensive tests for WF start-date override"
```

---

### Task 4: Final verification + MGC rebuild

**Files:**
- None (execution only)

**Step 1: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 2: Run drift check**

Run: `python pipeline/check_drift.py`
Expected: All checks pass

**Step 3: Rebuild MGC validation (WITH walk-forward — override now active)**

```bash
python trading_app/strategy_validator.py --instrument MGC \
  --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75
```

Expected: Significantly more WF passes than the 10 from tonight's build (expect 30-60+).

**Step 4: Rebuild MGC edge families**

```bash
python scripts/tools/build_edge_families.py --instrument MGC
```

**Step 5: Report results**

Compare before/after:
- Before (tonight): 10 WF passed, 4 FDR, 45 families
- After: [report actual numbers]

---

## Summary

| Task | Files | What Changes |
|------|-------|-------------|
| 0 | config.py | Add `WF_START_OVERRIDE` dict |
| 1 | walkforward.py + test | Add `wf_start_date` parameter, change anchor logic |
| 2 | strategy_validator.py | Import override, thread through worker pipeline |
| 3 | test_walkforward.py | 4 new tests: override, none, after-latest, before-earliest |
| 4 | (execution) | Full test suite + MGC rebuild |

**Total: ~30 lines of production code, ~60 lines of test code, 3 files modified.**
