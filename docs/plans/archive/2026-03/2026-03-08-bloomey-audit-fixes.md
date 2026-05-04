---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Bloomey Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 5 findings from the Bloomey seven-sins review of the strategy pipeline.

**Architecture:** Pure hygiene — no new files, no new data models, no new interfaces. Tightens existing code in 3 files: `strategy_validator.py`, `strategy_discovery.py`, `build_daily_features.py`.

**Tech Stack:** Python, DuckDB

---

### Task 0: Add LOOK-AHEAD comment to detect_double_break

**Files:**
- Modify: `pipeline/build_daily_features.py:374-384`

**Step 1: Add docstring line**

Add `NOTE: This is LOOK-AHEAD` to `detect_double_break()` docstring, matching the pattern used by `day_type` at line 537.

```python
def detect_double_break(
    bars_df: pd.DataFrame,
    trading_day: date,
    orb_label: str,
    orb_minutes: int,
    orb_high: float | None,
    orb_low: float | None,
) -> bool | None:
    """
    Detect if BOTH the ORB high and low were breached during the session.

    NOTE: This is LOOK-AHEAD relative to intraday entry — it checks the FULL
    session after trade entry. Do NOT use as a live trading filter.
    """
```

**Step 2: Verify**

Run: `python -m pytest tests/test_pipeline/ -x -q`
Expected: All pass (docstring-only change)

---

### Task 1: Delete dead _INSERT_SQL

**Files:**
- Modify: `trading_app/strategy_discovery.py:71-96`

**Step 1: Remove dead code**

Delete the `_INSERT_SQL` constant (lines 71-96). It is defined but never called — the actual insert uses `_BATCH_COLUMNS` and a different mechanism.

**Step 2: Verify no references**

Run: `grep -rn "_INSERT_SQL" trading_app/`
Expected: Zero matches

**Step 3: Run tests**

Run: `python -m pytest tests/test_trading_app/ -x -q`
Expected: All pass

---

### Task 2: Add @research-source annotations to ATR thresholds

**Files:**
- Modify: `trading_app/strategy_validator.py:302-308,394`

**Step 1: Annotate classify_regime thresholds**

```python
def classify_regime(atr_20: float) -> str:
    """Classify market regime from mean ATR(20)."""
    # @research-source: MGC regime analysis (Mar 2026), see memory/mgc_regime_analysis.md
    # ATR < 20 = dormant (pre-2022 MGC regime), ATR 20-30 = marginal transition zone
    if atr_20 < 20.0:
        return "DORMANT"
    elif atr_20 < 30.0:
        return "MARGINAL"
    return "ACTIVE"
```

**Step 2: Annotate 5-trade waiver cap**

```python
                # @research-source: MGC regime analysis (Mar 2026) — DORMANT years
                # typically have 0-5 trades; waiving years with >5 trades risks
                # masking real negative signal
                if mean_atr is not None and classify_regime(mean_atr) == "DORMANT" and trades <= 5:
```

**Step 3: Run drift check**

Run: `python pipeline/check_drift.py`
Expected: All pass (annotations satisfy drift check #45)

---

### Task 3: Fix all_years_positive DB flag inconsistency

**Files:**
- Modify: `trading_app/strategy_validator.py:955-964`

**Step 1: Apply min_trades_per_year filter to DB flag computation**

Currently line 964 computes `all_positive` from raw `yearly_data` without filtering low-trade years. Phase 3 at line 375 filters `d.get("trades", 0) >= min_trades_per_year`. The DB flag should match.

```python
                if status == "PASSED":
                    yearly = rd.get("yearly_results", "{}")
                    try:
                        yearly_data = json.loads(yearly) if isinstance(yearly, str) else yearly
                    except (json.JSONDecodeError, TypeError):
                        yearly_data = {}

                    included = {
                        y: d for y, d in yearly_data.items()
                        if int(y) not in (exclude_years or set())
                        and d.get("trades", 0) >= min_trades_per_year
                    }
                    years_tested = len(included)
                    all_positive = all(d.get("avg_r", 0) > 0 for d in included.values())
```

Note: `min_trades_per_year` is already available in scope — it's a parameter of `run_validation()` which contains this code block.

**Step 2: Verify min_trades_per_year is in scope**

Check that the variable is accessible at line 962. It's a parameter of the enclosing function.

**Step 3: Run tests**

Run: `python -m pytest tests/test_trading_app/test_strategy_validator.py -x -v`
Expected: All pass

---

### Task 4: Log corrupt filter_params instead of silent pass

**Files:**
- Modify: `trading_app/strategy_validator.py:128-129,173-174`

**Step 1: Add logging import**

Verify `import logging` and `log = logging.getLogger(__name__)` exist at top of file.

**Step 2: Add warning log to _extract_size_bounds**

```python
        except (json.JSONDecodeError, TypeError, ValueError):
            log.warning("Corrupt filter_params in _extract_size_bounds: %s", filter_params)
```

**Step 3: Add warning log to _parse_skip_days**

```python
    except (json.JSONDecodeError, TypeError, ValueError):
        log.warning("Corrupt filter_params in _parse_skip_days: %s", filter_params)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_trading_app/test_strategy_validator.py -x -v`
Expected: All pass

---

### Task 5: Run verification gates

**Step 1: Drift check**

Run: `python pipeline/check_drift.py`
Expected: All pass

**Step 2: Behavioral audit**

Run: `python scripts/tools/audit_behavioral.py`
Expected: All pass

**Step 3: Full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

---

### Task 6: Commit all fixes

```bash
git add trading_app/strategy_validator.py trading_app/strategy_discovery.py pipeline/build_daily_features.py
git commit -m "fix(bloomey): audit fixes — annotations, fail-closed, dead code, consistency"
```
