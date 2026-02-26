# E2 Stop-Market Entry Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace dishonest E0 (limit-on-confirm) with honest E2 (stop-market at ORB level + configurable slippage) across the entire codebase and rebuild all outcome data.

**Architecture:** E2 uses a new `detect_break_touch()` function that finds the first bar whose range crosses the ORB level (no close requirement), then fills at ORB level + N ticks slippage. E0 is fully purged (code + data). Grid slot reused (CB1 only).

**Tech Stack:** Python, DuckDB, pandas, numpy, pytest

---

### Task 1: Add E2 Config Constants

**Files:**
- Modify: `trading_app/config.py:601-607`

**Step 1: Write the failing test**

In `tests/test_app_sync.py`, update the ENTRY_MODELS assertion:

```python
# Find the existing test (line ~306):
#   assert ENTRY_MODELS == ["E0", "E1", "E3"]
# Replace with:
assert ENTRY_MODELS == ["E1", "E2", "E3"]
```

Also find the combo count test (line ~291) that mentions E0 and update the comment to reference E2.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_sync.py::TestConfigSync::test_entry_models -xvs`
Expected: FAIL — ENTRY_MODELS still has E0

**Step 3: Implement config changes**

In `trading_app/config.py`:

1. Update the ENTRY MODELS docstring (lines 35-46) — replace E0 description with:
```python
#  E2 (Stop-Market) - Stop order at ORB level + N ticks slippage. Triggers on
#     first bar whose range crosses the ORB level. No confirmation needed.
#     Fakeouts included as trades. Industry-standard honest breakout entry.
```

2. Add slippage constants (after line 607):
```python
# E2 stop-market slippage: number of ticks beyond ORB level for fill-through
# Default 1 = industry standard. Use 2 for stress testing.
E2_SLIPPAGE_TICKS = 1
E2_STRESS_TICKS = 2
```

3. Replace ENTRY_MODELS (line 607):
```python
# E0 was PURGED (Feb 2026): 3 compounding optimistic biases (fill-on-touch,
# fakeout exclusion, fill-bar wins). E0 won 33/33 combos = structural artifact.
# E2 replaces E0 as the honest stop-market entry at the ORB level.
ENTRY_MODELS = ["E1", "E2", "E3"]
```

4. Update the comment block (lines 601-606) to remove E0 references and document E2.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app_sync.py::TestConfigSync::test_entry_models -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add trading_app/config.py tests/test_app_sync.py
git commit -m "config: replace E0 with E2 in ENTRY_MODELS, add slippage constants"
```

---

### Task 2: Build detect_break_touch() and _resolve_e2()

**Files:**
- Modify: `trading_app/entry_rules.py`
- Test: `tests/test_trading_app/test_entry_rules.py`

**Step 1: Write the failing tests**

Add to `tests/test_trading_app/test_entry_rules.py`:

```python
from trading_app.entry_rules import detect_break_touch, BreakTouchResult

class TestDetectBreakTouch:
    """detect_break_touch finds the first bar whose range crosses the ORB level."""

    def _make_bars(self, data):
        """Helper: create bars DataFrame from list of (ts, open, high, low, close)."""
        return pd.DataFrame(data, columns=["ts_utc", "open", "high", "low", "close"])

    def test_long_touch_on_first_bar(self):
        """Bar high crosses orb_high — detected as break touch."""
        bars = self._make_bars([
            (datetime(2024, 1, 1, 0, 5), 2348, 2351, 2347, 2349),  # high > orb_high, but closes inside
        ])
        result = detect_break_touch(
            bars, orb_high=2350, orb_low=2340, break_dir="long",
            detection_window_start=datetime(2024, 1, 1, 0, 5),
            detection_window_end=datetime(2024, 1, 1, 8, 0),
        )
        assert result.touched is True
        assert result.touch_bar_ts == datetime(2024, 1, 1, 0, 5)

    def test_long_no_touch(self):
        """Bar high never reaches orb_high — no touch."""
        bars = self._make_bars([
            (datetime(2024, 1, 1, 0, 5), 2345, 2349, 2344, 2348),
        ])
        result = detect_break_touch(
            bars, orb_high=2350, orb_low=2340, break_dir="long",
            detection_window_start=datetime(2024, 1, 1, 0, 5),
            detection_window_end=datetime(2024, 1, 1, 8, 0),
        )
        assert result.touched is False

    def test_short_touch_fakeout(self):
        """Bar low crosses orb_low but closes back above — still a valid touch."""
        bars = self._make_bars([
            (datetime(2024, 1, 1, 0, 5), 2342, 2343, 2339, 2341),  # low < orb_low, closes inside
        ])
        result = detect_break_touch(
            bars, orb_high=2350, orb_low=2340, break_dir="short",
            detection_window_start=datetime(2024, 1, 1, 0, 5),
            detection_window_end=datetime(2024, 1, 1, 8, 0),
        )
        assert result.touched is True

    def test_touch_returns_first_bar(self):
        """Multiple bars touch — returns the FIRST one."""
        bars = self._make_bars([
            (datetime(2024, 1, 1, 0, 5), 2348, 2349, 2347, 2348),  # no touch
            (datetime(2024, 1, 1, 0, 6), 2349, 2351, 2348, 2349),  # touch (fakeout)
            (datetime(2024, 1, 1, 0, 7), 2349, 2352, 2349, 2352),  # touch (real)
        ])
        result = detect_break_touch(
            bars, orb_high=2350, orb_low=2340, break_dir="long",
            detection_window_start=datetime(2024, 1, 1, 0, 5),
            detection_window_end=datetime(2024, 1, 1, 8, 0),
        )
        assert result.touched is True
        assert result.touch_bar_ts == datetime(2024, 1, 1, 0, 6)  # first touch, even fakeout

    def test_respects_window(self):
        """Bars outside detection window are ignored."""
        bars = self._make_bars([
            (datetime(2024, 1, 1, 0, 4), 2349, 2351, 2348, 2351),  # before window
            (datetime(2024, 1, 1, 0, 5), 2348, 2349, 2347, 2348),  # in window, no touch
        ])
        result = detect_break_touch(
            bars, orb_high=2350, orb_low=2340, break_dir="long",
            detection_window_start=datetime(2024, 1, 1, 0, 5),
            detection_window_end=datetime(2024, 1, 1, 8, 0),
        )
        assert result.touched is False


class TestResolveE2:
    """_resolve_e2 fills at ORB level + N ticks slippage."""

    def _make_bars(self, data):
        return pd.DataFrame(data, columns=["ts_utc", "open", "high", "low", "close"])

    def test_long_entry_price_includes_slippage(self):
        """Long E2: entry = orb_high + slippage_ticks * tick_size."""
        bars = self._make_bars([
            (datetime(2024, 1, 1, 0, 5), 2348, 2351, 2347, 2349),
        ])
        result = detect_break_touch(
            bars, orb_high=2350, orb_low=2340, break_dir="long",
            detection_window_start=datetime(2024, 1, 1, 0, 5),
            detection_window_end=datetime(2024, 1, 1, 8, 0),
        )
        signal = _resolve_e2(result, slippage_ticks=1, tick_size=0.10)
        assert signal.triggered is True
        assert signal.entry_price == 2350.10  # orb_high + 1 tick
        assert signal.stop_price == 2340      # opposite ORB level
        assert signal.entry_model == "E2"

    def test_short_entry_price_includes_slippage(self):
        """Short E2: entry = orb_low - slippage_ticks * tick_size."""
        bars = self._make_bars([
            (datetime(2024, 1, 1, 0, 5), 2342, 2343, 2339, 2341),
        ])
        result = detect_break_touch(
            bars, orb_high=2350, orb_low=2340, break_dir="short",
            detection_window_start=datetime(2024, 1, 1, 0, 5),
            detection_window_end=datetime(2024, 1, 1, 8, 0),
        )
        signal = _resolve_e2(result, slippage_ticks=1, tick_size=0.10)
        assert signal.triggered is True
        assert signal.entry_price == 2339.90  # orb_low - 1 tick
        assert signal.stop_price == 2350

    def test_stress_slippage_2_ticks(self):
        """2-tick stress test: wider slippage."""
        bars = self._make_bars([
            (datetime(2024, 1, 1, 0, 5), 2348, 2351, 2347, 2352),
        ])
        result = detect_break_touch(
            bars, orb_high=2350, orb_low=2340, break_dir="long",
            detection_window_start=datetime(2024, 1, 1, 0, 5),
            detection_window_end=datetime(2024, 1, 1, 8, 0),
        )
        signal = _resolve_e2(result, slippage_ticks=2, tick_size=0.10)
        assert signal.entry_price == 2350.20  # orb_high + 2 ticks

    def test_no_touch_returns_no_fill(self):
        """No break touch → no E2 fill."""
        result = BreakTouchResult(
            touched=False, touch_bar_ts=None, touch_bar_idx=None,
            orb_high=2350, orb_low=2340, break_dir="long",
        )
        signal = _resolve_e2(result, slippage_ticks=1, tick_size=0.10)
        assert signal.triggered is False
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trading_app/test_entry_rules.py::TestDetectBreakTouch -xvs`
Expected: FAIL — ImportError (detect_break_touch not defined)

**Step 3: Implement detect_break_touch() and _resolve_e2()**

In `trading_app/entry_rules.py`:

1. Add `BreakTouchResult` dataclass:
```python
@dataclass(frozen=True)
class BreakTouchResult:
    """Result of break-touch detection (range crosses ORB, no close requirement)."""
    touched: bool
    touch_bar_ts: datetime | None
    touch_bar_idx: int | None
    orb_high: float
    orb_low: float
    break_dir: str
```

2. Add `detect_break_touch()`:
```python
def detect_break_touch(
    bars_df: pd.DataFrame,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    detection_window_start: datetime,
    detection_window_end: datetime,
) -> BreakTouchResult:
    """
    Detect first bar whose range crosses the ORB level.

    Unlike detect_confirm(), this does NOT require the bar to close outside
    the ORB. A bar whose high > orb_high (long) or low < orb_low (short)
    counts as a touch, even if it closes back inside (fakeout).

    This is the detection path for E2 (stop-market), where a resting stop
    order triggers on any intra-bar touch of the level.
    """
    no_touch = BreakTouchResult(
        touched=False, touch_bar_ts=None, touch_bar_idx=None,
        orb_high=orb_high, orb_low=orb_low, break_dir=break_dir,
    )

    if break_dir not in ("long", "short"):
        raise ValueError(f"break_dir must be 'long' or 'short', got {break_dir}")

    candidate_bars = bars_df[
        (bars_df["ts_utc"] >= pd.Timestamp(detection_window_start))
        & (bars_df["ts_utc"] < pd.Timestamp(detection_window_end))
    ].sort_values("ts_utc")

    if candidate_bars.empty:
        return no_touch

    if break_dir == "long":
        touch_mask = candidate_bars["high"].values > orb_high
    else:
        touch_mask = candidate_bars["low"].values < orb_low

    if not touch_mask.any():
        return no_touch

    idx = int(np.argmax(touch_mask))
    ts = candidate_bars.iloc[idx]["ts_utc"].to_pydatetime()

    return BreakTouchResult(
        touched=True, touch_bar_ts=ts, touch_bar_idx=idx,
        orb_high=orb_high, orb_low=orb_low, break_dir=break_dir,
    )
```

3. Add `_resolve_e2()`:
```python
def _resolve_e2(
    touch: BreakTouchResult,
    slippage_ticks: int,
    tick_size: float,
) -> EntrySignal:
    """E2: Stop-Market. Entry at ORB level + N ticks slippage.

    A stop order sits at the ORB boundary before the break. It fills
    the moment the bar's range crosses the level. Fill price includes
    slippage (fill-through, not fill-on-touch).

    Fakeout bars (close back inside ORB) ARE valid fills — the stop
    triggered intra-bar regardless of where the bar closes.
    """
    if not touch.touched:
        return EntrySignal(
            triggered=False, entry_ts=None, entry_price=None,
            stop_price=None, entry_model="E2", confirm_bar_ts=None,
        )

    slippage = slippage_ticks * tick_size

    if touch.break_dir == "long":
        entry_price = touch.orb_high + slippage
        stop_price = touch.orb_low
    else:
        entry_price = touch.orb_low - slippage
        stop_price = touch.orb_high

    return EntrySignal(
        triggered=True,
        entry_ts=touch.touch_bar_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        entry_model="E2",
        confirm_bar_ts=touch.touch_bar_ts,
    )
```

4. Delete `_resolve_e0()` function entirely (lines 196-239).

5. Update `resolve_entry()` dispatch — remove E0 branch, add E2 raise:
```python
if entry_model == "E0":
    raise ValueError("E0 was purged (Feb 2026). Use E2 for stop-market entry.")
elif entry_model == "E2":
    raise ValueError("E2 uses detect_break_touch(), not detect_confirm(). "
                     "Call detect_break_touch() + _resolve_e2() directly.")
```

6. Update `detect_entry_with_confirm_bars()` — remove E0 CB>1 guard, add E2 guard:
```python
if entry_model == "E2":
    raise ValueError("E2 uses detect_break_touch(), not detect_entry_with_confirm_bars().")
```

7. Update module docstring at top of file.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trading_app/test_entry_rules.py -xvs`
Expected: PASS (new tests pass, old E0 tests removed)

**Step 5: Commit**

```bash
git add trading_app/entry_rules.py tests/test_trading_app/test_entry_rules.py
git commit -m "feat: add detect_break_touch() + _resolve_e2(), purge _resolve_e0()"
```

---

### Task 3: Remove E0-Specific Tests

**Files:**
- Modify: `tests/test_trading_app/test_entry_rules.py`
- Modify: `tests/test_trading_app/test_outcome_builder.py`
- Modify: `tests/test_trading_app/test_paper_trader.py`
- Modify: `tests/test_app_sync.py`
- Modify: `tests/test_trading_app/test_live_config.py`
- Modify: `tests/test_tools/test_prospective_tracker.py`
- Modify: `tests/test_research/test_lib.py`
- Modify: `tests/test_trading_app/test_ai/test_sql_adapter.py`

**Step 1: Update each test file**

For each file with E0 references:
- Delete `TestE0CB2PlusGuard` class in test_entry_rules.py (lines 286-312)
- Replace `"E0"` with `"E2"` in test fixture data where used as a parameter
- Replace `assert ENTRY_MODELS == ["E0", "E1", "E3"]` with `["E1", "E2", "E3"]`
- Update combo count comments referencing E0 to reference E2
- Update strategy ID strings: `"MGC_CME_REOPEN_E0_..."` → `"MGC_CME_REOPEN_E2_..."`
- In test_outcome_builder.py: `assert models == {"E0", "E1", "E3"}` → `{"E1", "E2", "E3"}`
- In test_research/test_lib.py: change `outcomes_query("MGC", "1000", "E0")` → `"E1"` (E0 is no longer valid)

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All existing tests pass with E0→E2 references updated

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: replace all E0 references with E2 across test suite"
```

---

### Task 4: Wire E2 into outcome_builder.py

**Files:**
- Modify: `trading_app/outcome_builder.py:740-758`

**Step 1: Write the failing test**

Add to `tests/test_trading_app/test_outcome_builder.py`:

```python
def test_e2_outcome_uses_break_touch(self):
    """E2 outcomes use detect_break_touch (range-based), not detect_confirm (close-based)."""
    # Create a fakeout bar: high crosses ORB high, but closes back inside
    bars = self._make_bars([
        # ORB formation bars (before break)
        (datetime(2024, 1, 1, 0, 0), 2345, 2348, 2342, 2346),
        # Break bar: high > 2350 but closes at 2349 (fakeout)
        (datetime(2024, 1, 1, 0, 5), 2348, 2351, 2347, 2349),
        # Post-entry bars
        (datetime(2024, 1, 1, 0, 6), 2349, 2355, 2348, 2354),
        (datetime(2024, 1, 1, 0, 7), 2354, 2360, 2353, 2359),
        (datetime(2024, 1, 1, 0, 8), 2359, 2365, 2358, 2364),
    ])
    result = compute_single_outcome(
        bars_df=bars,
        break_ts=datetime(2024, 1, 1, 0, 5),
        orb_high=2350.0, orb_low=2340.0,
        break_dir="long", rr_target=2.0, confirm_bars=1,
        trading_day_end=datetime(2024, 1, 1, 23, 0),
        cost_spec=get_cost_spec("MGC"),
        entry_model="E2",
        orb_label="CME_REOPEN",
    )
    # E2 should fill on the fakeout bar (E0/E1 would NOT fill here)
    assert result["entry_price"] is not None
    assert result["entry_price"] == pytest.approx(2350.10, abs=0.01)  # orb_high + 1 tick
    assert result["entry_ts"] == datetime(2024, 1, 1, 0, 5)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_outcome_builder.py::TestOutcomeBuilder::test_e2_outcome_uses_break_touch -xvs`
Expected: FAIL — E2 not handled in outcome_builder

**Step 3: Implement E2 path in outcome_builder**

In `trading_app/outcome_builder.py`:

1. Add imports at top:
```python
from trading_app.entry_rules import detect_break_touch, _resolve_e2, BreakTouchResult
from trading_app.config import E2_SLIPPAGE_TICKS
```

2. In the main build loop (lines ~744-758), add E2 branch BEFORE the E1/E3 path:
```python
for em in ENTRY_MODELS:
    if em == "E2":
        # E2: stop-market at ORB level, no confirm needed
        touch = detect_break_touch(
            bars_df, orb_high=orb_high, orb_low=orb_low,
            break_dir=break_dir,
            detection_window_start=break_ts,
            detection_window_end=td_end,
        )
        signal = _resolve_e2(touch, slippage_ticks=E2_SLIPPAGE_TICKS,
                             tick_size=cost_spec.tick_size)
        # E2 is always CB1 in the grid
        cb_options = [1]
        for cb in cb_options:
            outcomes = _compute_outcomes_all_rr(
                bars_df=bars_df, signal=signal,
                orb_high=orb_high, orb_low=orb_low,
                break_dir=break_dir, rr_targets=RR_TARGETS,
                trading_day_end=td_end, cost_spec=cost_spec,
                entry_model=em, orb_label=orb_label, break_ts=break_ts,
            )
            for rr_target, outcome in zip(RR_TARGETS, outcomes):
                day_batch.append([...])  # same columns as existing
                total_written += 1
        continue  # skip the E1/E3 path below

    # E1/E3: confirm-based path (unchanged)
    cb_options = [1] if em == "E3" else CONFIRM_BARS_OPTIONS
    for cb in cb_options:
        signal = detect_entry_with_confirm_bars(...)
        ...
```

3. Also update `compute_single_outcome()` (line ~441) with the same E2 branch:
```python
if entry_model == "E2":
    touch = detect_break_touch(
        bars_df, orb_high=orb_high, orb_low=orb_low,
        break_dir=break_dir,
        detection_window_start=break_ts,
        detection_window_end=trading_day_end,
    )
    signal = _resolve_e2(touch, slippage_ticks=E2_SLIPPAGE_TICKS,
                         tick_size=cost_spec.tick_size)
else:
    signal = detect_entry_with_confirm_bars(...)
```

4. Remove E0 references from comments.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trading_app/test_outcome_builder.py -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add trading_app/outcome_builder.py tests/test_trading_app/test_outcome_builder.py
git commit -m "feat: wire E2 stop-market path into outcome_builder"
```

---

### Task 5: Wire E2 into strategy_discovery.py

**Files:**
- Modify: `trading_app/strategy_discovery.py:840-854`

**Step 1: Update grid loop**

Replace E0 references in the grid iteration:

1. Line 841: `total_combos += nf * len(RR_TARGETS) * 2  # E0+E3 (CB1 only)` → `# E2+E3 (CB1 only)`
2. Line 747 (outcome_builder loop): `em in ("E0", "E3")` → `em in ("E2", "E3")`
3. Line 852: `if em in ("E0", "E3") and cb > 1:` → `if em in ("E2", "E3") and cb > 1:`

**Step 2: Run discovery in dry-run/smoke mode**

Run: `python -m pytest tests/test_trading_app/test_strategy_discovery.py -xvs`
Expected: PASS

**Step 3: Commit**

```bash
git add trading_app/strategy_discovery.py
git commit -m "refactor: replace E0 with E2 in strategy_discovery grid loop"
```

---

### Task 6: Wire E2 into nested/builder.py

**Files:**
- Modify: `trading_app/nested/builder.py:284-306`

**Step 1: Update nested builder**

1. Line 287: `em in ("E0", "E3") and cb > 1` → `em in ("E2", "E3") and cb > 1`
2. Add E2 branch in the entry_cache loop: E2 uses `detect_break_touch()` on 5m bars instead of `detect_confirm()` + `resolve_entry()`.
3. Import `detect_break_touch`, `_resolve_e2`, `E2_SLIPPAGE_TICKS` at top of file.

The E2 path in nested builder:
```python
if em == "E2":
    touch = detect_break_touch(
        bars_5m_df, orb_high=orb_high, orb_low=orb_low,
        break_dir=break_dir,
        detection_window_start=break_ts,
        detection_window_end=td_end,
    )
    signal = _resolve_e2(touch, slippage_ticks=E2_SLIPPAGE_TICKS,
                         tick_size=cost_spec.tick_size)
    entry_cache[(cb, em)] = signal
    continue
```

**Step 2: Run nested tests**

Run: `python -m pytest tests/test_trading_app/test_nested/ -xvs`
Expected: PASS

**Step 3: Commit**

```bash
git add trading_app/nested/builder.py
git commit -m "refactor: replace E0 with E2 in nested/builder.py"
```

---

### Task 7: Wire E2 into execution_engine.py

**Files:**
- Modify: `trading_app/execution_engine.py:529-638`

**Step 1: Replace E0 branch with E2**

The E0 branch (lines 529-621) handles confirm-bar entry at ORB level. E2 in live execution is similar but with slippage:

Replace `if trade.entry_model == "E0":` block with:
```python
if trade.entry_model == "E2":
    # E2: Stop-Market — fill at ORB edge + slippage ON the break bar.
    # The stop order was resting at the ORB level before the break.
    from trading_app.config import E2_SLIPPAGE_TICKS
    tick_size = trade.strategy.tick_size if hasattr(trade.strategy, 'tick_size') else 0.10

    if trade.direction == "long":
        entry_price = orb.high + E2_SLIPPAGE_TICKS * tick_size
        if confirm_bar["high"] <= orb.high:
            return events  # Bar didn't reach level — no trigger
    else:
        entry_price = orb.low - E2_SLIPPAGE_TICKS * tick_size
        if confirm_bar["low"] >= orb.low:
            return events  # Bar didn't reach level — no trigger

    # Rest of entry logic: risk calc, risk manager, target, state change
    # (same structure as old E0, just different entry_price and reason)
    ...
    reason="stop_market_E2"
```

Note: In live trading, E2 should trigger on the BREAK bar (first touch), not the confirm bar. The execution engine's `_try_entry` is called when confirm bars are met. For E2 with CB1, the confirm bar IS the break bar, so the logic is equivalent. The key difference is entry_price includes slippage.

**Step 2: Run execution engine tests**

Run: `python -m pytest tests/test_trading_app/test_execution_engine.py -xvs`
Expected: PASS (after updating E0→E2 in test fixtures)

**Step 3: Commit**

```bash
git add trading_app/execution_engine.py
git commit -m "feat: replace E0 execution branch with E2 stop-market in execution_engine"
```

---

### Task 8: Update paper_trader.py

**Files:**
- Modify: `trading_app/paper_trader.py`

**Step 1: Update entry model references**

1. In `_entry_model_from_strategy()`: ensure it can parse "E2" from strategy IDs
2. Replace any E0 comments/references with E2

This should be minimal — the function parses the entry model token from the strategy ID string, and "E2" is already a valid 2-char token matching the same pattern as "E0".

**Step 2: Run paper trader tests**

Run: `python -m pytest tests/test_trading_app/test_paper_trader.py -xvs`
Expected: PASS

**Step 3: Commit**

```bash
git add trading_app/paper_trader.py
git commit -m "refactor: update paper_trader E0 references to E2"
```

---

### Task 9: Update Drift Checks

**Files:**
- Modify: `pipeline/check_drift.py:532-551, 1225-1275`

**Step 1: Update check_entry_models_sync()**

Line 543: `expected = ["E0", "E1", "E3"]` → `expected = ["E1", "E2", "E3"]`

**Step 2: Update check_e0_cb1_only()**

Rename to `check_e2_e3_cb1_only()`. Update all E0 references to E2:
- `em in ("E0", "E3")` → `em in ("E2", "E3")`
- Comments updated

**Step 3: Add new check: check_no_e0_in_db()**

```python
def check_no_e0_in_db() -> list[str]:
    """Check #35: No E0 rows should exist in any trading table.

    E0 (limit-on-confirm) was purged Feb 2026. If E0 rows reappear,
    it means a dirty rebuild or manual insertion occurred.
    """
    violations = []
    db_path = _get_db_path()
    if db_path is None or not Path(db_path).exists():
        return violations

    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for table in ["orb_outcomes", "experimental_strategies", "validated_setups"]:
            count = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE entry_model = 'E0'"
            ).fetchone()[0]
            if count > 0:
                violations.append(
                    f"  {table}: {count} rows with entry_model='E0' (purged Feb 2026)"
                )
    finally:
        con.close()

    return violations
```

Add to the `main()` runner list.

**Step 4: Run drift checks**

Run: `python pipeline/check_drift.py`
Expected: All 35 checks pass (E0 data not yet purged — this check will fail until Task 11)

**Step 5: Commit**

```bash
git add pipeline/check_drift.py
git commit -m "drift: update entry model checks for E2, add no-E0-in-db guard"
```

---

### Task 10: Update Documentation

**Files:**
- Modify: `TRADING_RULES.md` — entry model table
- Modify: `CLAUDE.md` — if entry models are mentioned
- Modify: `docs/plans/2026-02-26-e2-entry-model-design.md` — mark as implemented

**Step 1: Update TRADING_RULES.md entry model section**

Replace E0 entry in the entry model table with E2. Add slippage documentation.

**Step 2: Commit**

```bash
git add TRADING_RULES.md CLAUDE.md
git commit -m "docs: update entry model documentation for E2 replacement"
```

---

### Task 11: Purge E0 Data from Database

**WARNING: This is a destructive database operation. Execute carefully.**

**Step 1: Count E0 rows before purge**

```bash
python -c "
import duckdb
con = duckdb.connect('gold.db', read_only=True)
for t in ['orb_outcomes', 'experimental_strategies', 'validated_setups', 'edge_families']:
    try:
        n = con.execute(f\"SELECT COUNT(*) FROM {t} WHERE entry_model = 'E0'\").fetchone()[0]
    except:
        n = con.execute(f\"SELECT COUNT(*) FROM {t} WHERE family_id LIKE '%_E0_%'\").fetchone()[0]
    print(f'{t}: {n} E0 rows')
con.close()
"
```

**Step 2: Execute purge**

```bash
python -c "
import duckdb
con = duckdb.connect('gold.db')
con.execute(\"DELETE FROM orb_outcomes WHERE entry_model = 'E0'\")
con.execute(\"DELETE FROM experimental_strategies WHERE entry_model = 'E0'\")
con.execute(\"DELETE FROM validated_setups WHERE entry_model = 'E0'\")
con.execute(\"DELETE FROM edge_families WHERE family_id LIKE '%_E0_%'\")
con.commit()
print('E0 purge complete')
con.close()
"
```

**Step 3: Verify purge**

Run: `python pipeline/check_drift.py`
Expected: All checks pass including the new no-E0-in-db check (#35)

**Step 4: Do NOT commit gold.db** (it's in .gitignore)

---

### Task 12: Rebuild E2 Outcomes for All Instruments

**WARNING: Long-running. Use scratch DB path for safety.**

**Step 1: Copy DB to scratch location**

```bash
cp gold.db C:/db/gold.db
export DUCKDB_PATH=C:/db/gold.db
```

**Step 2: Rebuild outcomes per instrument**

```bash
# MGC (10 years)
python trading_app/outcome_builder.py --instrument MGC --force --start 2016-02-01 --end 2026-02-04

# MES (7 years)
python trading_app/outcome_builder.py --instrument MES --force --start 2019-02-12 --end 2026-02-11

# MNQ (2 years)
python trading_app/outcome_builder.py --instrument MNQ --force --start 2024-02-05 --end 2026-02-04

# M2K (5 years)
python trading_app/outcome_builder.py --instrument M2K --force --start 2021-02-05 --end 2026-02-04
```

**Step 3: Verify E2 outcomes exist**

```bash
python -c "
import duckdb
con = duckdb.connect('C:/db/gold.db', read_only=True)
for em in ['E1', 'E2', 'E3']:
    n = con.execute(f\"SELECT COUNT(*) FROM orb_outcomes WHERE entry_model = '{em}'\").fetchone()[0]
    print(f'{em}: {n} outcomes')
con.close()
"
```

**Step 4: Copy back**

```bash
cp C:/db/gold.db gold.db
```

---

### Task 13: Rediscovery + Revalidation + Edge Families

**Step 1: Run discovery per instrument**

```bash
python trading_app/strategy_discovery.py --instrument MGC
python trading_app/strategy_discovery.py --instrument MES
python trading_app/strategy_discovery.py --instrument MNQ
python trading_app/strategy_discovery.py --instrument M2K
```

**Step 2: Run validation per instrument**

```bash
python trading_app/strategy_validator.py --instrument MGC --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument MES --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument MNQ --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument M2K --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
```

**Step 3: Build edge families**

```bash
python scripts/tools/build_edge_families.py --instrument MGC
python scripts/tools/build_edge_families.py --instrument MES
python scripts/tools/build_edge_families.py --instrument MNQ
python scripts/tools/build_edge_families.py --instrument M2K
```

**Step 4: Verify new strategy counts**

```bash
python -c "
import duckdb
con = duckdb.connect('gold.db', read_only=True)
for em in ['E1', 'E2', 'E3']:
    n = con.execute(f\"SELECT COUNT(*) FROM validated_setups WHERE entry_model = '{em}' AND is_active = true\").fetchone()[0]
    print(f'{em}: {n} validated active')
n = con.execute('SELECT COUNT(DISTINCT family_id) FROM edge_families').fetchone()[0]
print(f'Edge families: {n}')
con.close()
"
```

---

### Task 14: Run Full Health Check

**Step 1: Drift checks**

Run: `python pipeline/check_drift.py`
Expected: All 35 checks pass

**Step 2: Full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 3: Health check**

Run: `python pipeline/health_check.py`
Expected: All checks pass

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: E2 stop-market entry model — E0 purged, E2 built, full rebuild complete"
```

---

### Task 15: Update Memory Files

**Files:**
- Modify: `~/.claude/projects/C--Users-joshd-canompx3/memory/MEMORY.md`
- Modify: `~/.claude/projects/C--Users-joshd-canompx3/memory/entry_model_audit.md`

Update:
- ENTRY_MODELS: E1, E2, E3 (E0 purged)
- Strategy counts from Task 13 verification
- E2 mechanics documentation
- Action queue: mark E0 purge as DONE
