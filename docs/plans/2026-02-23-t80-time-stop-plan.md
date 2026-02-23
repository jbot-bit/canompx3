# T80 Time-Stop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add conditional time-stop columns to `orb_outcomes` so backtested metrics reflect T80 early exit.

**Architecture:** Three new nullable columns (`ts_outcome`, `ts_pnl_r`, `ts_exit_ts`) computed alongside existing outcomes. Uses `EARLY_EXIT_MINUTES` from config.py. At the time-stop bar, if MTM < 0, exit at bar close. Otherwise trade continues to normal resolution.

**Tech Stack:** DuckDB, pandas, numpy, pytest

---

### Task 1: Schema migration — add time-stop columns

**Files:**
- Modify: `trading_app/db_manager.py:371-376` (add migration after ambiguous_bar)

**Step 1: Write the failing test**

Add to `tests/test_trading_app/test_outcome_builder.py`:

```python
class TestTimeStopColumns:
    """Tests that time-stop columns exist in orb_outcomes schema."""

    def test_time_stop_columns_in_schema(self, tmp_path):
        """Schema migration adds ts_outcome, ts_pnl_r, ts_exit_ts."""
        db_path = tmp_path / "test.db"
        import duckdb
        with duckdb.connect(str(db_path)) as con:
            from trading_app.db_manager import init_trading_app_schema
            init_trading_app_schema(db_path=db_path)
            cols = {r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'orb_outcomes'"
            ).fetchall()}
        assert "ts_outcome" in cols
        assert "ts_pnl_r" in cols
        assert "ts_exit_ts" in cols
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_outcome_builder.py::TestTimeStopColumns -x -v`
Expected: FAIL — columns don't exist yet

**Step 3: Add migration to db_manager.py**

In `trading_app/db_manager.py`, after the `ambiguous_bar` migration (line ~376), add:

```python
        # Migration: add time-stop columns (T80 conditional exit)
        # Stores outcome/pnl_r/exit_ts if EARLY_EXIT_MINUTES time-stop fires.
        # NULL = no time-stop configured for this session.
        for col, typedef in [
            ("ts_outcome", "VARCHAR"),
            ("ts_pnl_r", "DOUBLE"),
            ("ts_exit_ts", "TIMESTAMPTZ"),
        ]:
            try:
                con.execute(f"ALTER TABLE orb_outcomes ADD COLUMN {col} {typedef}")
            except duckdb.CatalogException:
                pass  # column already exists
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trading_app/test_outcome_builder.py::TestTimeStopColumns -x -v`
Expected: PASS

**Step 5: Commit**

```bash
git add trading_app/db_manager.py tests/test_trading_app/test_outcome_builder.py
git commit -m "feat: add time-stop columns to orb_outcomes schema"
```

---

### Task 2: Time-stop helper function

**Files:**
- Modify: `trading_app/outcome_builder.py` (add helper before `_compute_outcomes_all_rr`)

**Step 1: Write the failing test**

Add to `tests/test_trading_app/test_outcome_builder.py`:

```python
from trading_app.outcome_builder import _annotate_time_stop


class TestAnnotateTimeStop:
    """Tests for _annotate_time_stop helper."""

    def test_no_threshold_returns_nulls(self):
        """Session with no time-stop → all ts_* are None."""
        result = {"outcome": "win", "pnl_r": 1.5, "exit_ts": datetime(2024, 1, 5, 0, 30, tzinfo=timezone.utc)}
        _annotate_time_stop(result, threshold_minutes=None, post_entry=pd.DataFrame(),
                            entry_ts=datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc),
                            entry_price=100.0, stop_price=90.0, break_dir="long",
                            cost_spec=_cost())
        assert result["ts_outcome"] is None
        assert result["ts_pnl_r"] is None
        assert result["ts_exit_ts"] is None

    def test_exit_before_threshold(self):
        """Trade exits (win) before T80 bar → ts_* = baseline."""
        entry_ts = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 6, tzinfo=timezone.utc),
            [(101, 115, 100, 112, 100)] * 60,  # 60 bars after entry
        )
        result = {"outcome": "win", "pnl_r": 1.5,
                  "exit_ts": datetime(2024, 1, 5, 0, 15, tzinfo=timezone.utc)}
        _annotate_time_stop(result, threshold_minutes=30, post_entry=bars,
                            entry_ts=entry_ts, entry_price=100.0, stop_price=90.0,
                            break_dir="long", cost_spec=_cost())
        assert result["ts_outcome"] == "win"
        assert result["ts_pnl_r"] == 1.5

    def test_underwater_at_threshold_fires(self):
        """Trade open and underwater at T80 → time_stop fires."""
        entry_ts = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)
        # 60 bars: all with close below entry (underwater)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 6, tzinfo=timezone.utc),
            [(99, 100, 95, 97, 100)] * 60,
        )
        result = {"outcome": "loss", "pnl_r": -1.0,
                  "exit_ts": datetime(2024, 1, 5, 1, 30, tzinfo=timezone.utc)}
        _annotate_time_stop(result, threshold_minutes=30, post_entry=bars,
                            entry_ts=entry_ts, entry_price=100.0, stop_price=90.0,
                            break_dir="long", cost_spec=_cost())
        assert result["ts_outcome"] == "time_stop"
        assert result["ts_pnl_r"] < 0
        assert result["ts_pnl_r"] > -1.0  # better than full loss
        assert result["ts_exit_ts"] is not None

    def test_positive_at_threshold_no_fire(self):
        """Trade open but positive at T80 → keeps running, ts_* = baseline."""
        entry_ts = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)
        # 60 bars: all with close above entry (positive)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 6, tzinfo=timezone.utc),
            [(101, 105, 100, 103, 100)] * 60,
        )
        result = {"outcome": "scratch", "pnl_r": None,
                  "exit_ts": None}
        _annotate_time_stop(result, threshold_minutes=30, post_entry=bars,
                            entry_ts=entry_ts, entry_price=100.0, stop_price=90.0,
                            break_dir="long", cost_spec=_cost())
        assert result["ts_outcome"] == "scratch"
        assert result["ts_pnl_r"] is None

    def test_short_trade_underwater_fires(self):
        """Short trade underwater at T80 (price above entry) → fires."""
        entry_ts = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 6, tzinfo=timezone.utc),
            [(101, 105, 100, 103, 100)] * 60,
        )
        result = {"outcome": "loss", "pnl_r": -1.0,
                  "exit_ts": datetime(2024, 1, 5, 1, 30, tzinfo=timezone.utc)}
        _annotate_time_stop(result, threshold_minutes=30, post_entry=bars,
                            entry_ts=entry_ts, entry_price=100.0, stop_price=110.0,
                            break_dir="short", cost_spec=_cost())
        assert result["ts_outcome"] == "time_stop"
        assert result["ts_pnl_r"] < 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_outcome_builder.py::TestAnnotateTimeStop -x -v`
Expected: FAIL — `_annotate_time_stop` doesn't exist

**Step 3: Implement the helper**

In `trading_app/outcome_builder.py`, add before `_compute_outcomes_all_rr`:

```python
def _annotate_time_stop(
    result: dict,
    threshold_minutes: int | None,
    post_entry: pd.DataFrame,
    entry_ts: datetime,
    entry_price: float,
    stop_price: float,
    break_dir: str,
    cost_spec,
) -> None:
    """Annotate an outcome dict with conditional time-stop fields.

    Mutates result in-place, adding ts_outcome, ts_pnl_r, ts_exit_ts.
    Mirrors execution_engine EARLY_EXIT logic: at the first bar past threshold,
    if MTM < 0, exit at bar close.
    """
    result["ts_outcome"] = None
    result["ts_pnl_r"] = None
    result["ts_exit_ts"] = None

    if threshold_minutes is None:
        return

    if post_entry.empty:
        # No post-entry bars → time-stop can't fire; copy baseline
        result["ts_outcome"] = result["outcome"]
        result["ts_pnl_r"] = result["pnl_r"]
        result["ts_exit_ts"] = result["exit_ts"]
        return

    # Find the first bar at or after the time-stop threshold
    ts_cutoff = pd.Timestamp(entry_ts + timedelta(minutes=threshold_minutes))
    ts_mask = post_entry["ts_utc"] >= ts_cutoff
    if not ts_mask.any():
        # Session ended before threshold → time-stop never reached; copy baseline
        result["ts_outcome"] = result["outcome"]
        result["ts_pnl_r"] = result["pnl_r"]
        result["ts_exit_ts"] = result["exit_ts"]
        return

    ts_bar = post_entry.loc[ts_mask.idxmax()]
    ts_bar_ts = ts_bar["ts_utc"].to_pydatetime()
    ts_bar_close = float(ts_bar["close"])

    # Did the trade already resolve before the time-stop bar?
    normal_exit_ts = result.get("exit_ts")
    if normal_exit_ts is not None and normal_exit_ts < ts_bar_ts:
        # Resolved before time-stop → no change
        result["ts_outcome"] = result["outcome"]
        result["ts_pnl_r"] = result["pnl_r"]
        result["ts_exit_ts"] = result["exit_ts"]
        return

    # Trade is still open at the time-stop bar. Check MTM.
    if break_dir == "long":
        mtm_points = ts_bar_close - entry_price
    else:
        mtm_points = entry_price - ts_bar_close

    if mtm_points < 0:
        # Underwater → time-stop fires
        result["ts_outcome"] = "time_stop"
        result["ts_pnl_r"] = round(
            to_r_multiple(cost_spec, entry_price, stop_price, mtm_points), 4
        )
        result["ts_exit_ts"] = ts_bar_ts
    else:
        # Positive → trade keeps running to normal resolution
        result["ts_outcome"] = result["outcome"]
        result["ts_pnl_r"] = result["pnl_r"]
        result["ts_exit_ts"] = result["exit_ts"]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trading_app/test_outcome_builder.py::TestAnnotateTimeStop -x -v`
Expected: PASS

**Step 5: Commit**

```bash
git add trading_app/outcome_builder.py tests/test_trading_app/test_outcome_builder.py
git commit -m "feat: add _annotate_time_stop helper for T80 conditional exit"
```

---

### Task 3: Wire time-stop into outcome computation

**Files:**
- Modify: `trading_app/outcome_builder.py:132-309` (`_compute_outcomes_all_rr`)
- Modify: `trading_app/outcome_builder.py:311-474` (`compute_single_outcome`)

**Step 1: Write the failing test**

Add to `tests/test_trading_app/test_outcome_builder.py`:

```python
class TestTimeStopIntegration:
    """End-to-end tests that compute_single_outcome includes ts_* columns."""

    def test_outcome_has_time_stop_keys(self):
        """compute_single_outcome returns ts_outcome, ts_pnl_r, ts_exit_ts."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [(2698, 2701, 2695, 2701, 100),
             (2703, 2710, 2700, 2710, 100),
             (2718, 2735, 2717, 2730, 100)],
        )
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
            orb_label="1000",
        )
        assert "ts_outcome" in result
        assert "ts_pnl_r" in result
        assert "ts_exit_ts" in result

    def test_1000_session_loss_after_30m_gets_time_stopped(self):
        """MGC 1000 trade that loses after 30+ minutes gets ts_outcome=time_stop."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 8, 0, tzinfo=timezone.utc)

        # Bar 0: confirm (close above ORB high)
        # Bar 1: E1 entry at open. Entry=2703, Stop=2690, Risk=13
        # Bars 2-60: price drifts down slowly, close < entry but above stop
        # Eventually hits stop after ~90 minutes
        bar_data = [
            (2698, 2701, 2695, 2701, 100),  # confirm
            (2703, 2705, 2698, 2700, 100),  # entry bar, close < entry
        ]
        # 58 more bars: slowly drifting down to 2695 (above stop=2690)
        for i in range(58):
            price = 2700 - i * 0.1
            bar_data.append((price, price + 1, price - 1, price, 100))
        # Bar 61: hits stop
        bar_data.append((2691, 2692, 2688, 2689, 100))

        bars = _make_bars(datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc), bar_data)
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
            orb_label="1000",
        )
        # Baseline: full loss
        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0
        # Time-stop: should fire at ~30m mark (bar 31) with partial loss
        assert result["ts_outcome"] == "time_stop"
        assert result["ts_pnl_r"] < 0
        assert result["ts_pnl_r"] > -1.0  # better than full loss

    def test_no_threshold_session_leaves_ts_null(self):
        """Session like '1100' with no time-stop → ts_* = None."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [(2698, 2701, 2695, 2701, 100),
             (2703, 2710, 2700, 2710, 100),
             (2718, 2735, 2717, 2730, 100)],
        )
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
            orb_label="1100",
        )
        assert result["ts_outcome"] is None
        assert result["ts_pnl_r"] is None
        assert result["ts_exit_ts"] is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_outcome_builder.py::TestTimeStopIntegration -x -v`
Expected: FAIL — `ts_outcome` not in result keys

**Step 3: Wire into both compute functions**

**3a. `_compute_outcomes_all_rr`** (~line 132):

After computing `post_entry` (line 196), add the threshold lookup:

```python
    # Time-stop threshold for this session
    ts_threshold = EARLY_EXIT_MINUTES.get(orb_label)
```

Then at the end of the RR loop (after `results.append(result)` on line 307), add:

```python
        _annotate_time_stop(
            result, ts_threshold, post_entry,
            entry_ts, entry_price, stop_price, break_dir, cost_spec,
        )
```

Also add `ts_outcome`, `ts_pnl_r`, `ts_exit_ts` with value `None` to the `null_result` dict (line 150-156) and to each `result` dict (line 214-220).

**3b. `compute_single_outcome`** (~line 311):

After computing the normal outcome and before `return result` (line 474), add:

```python
    ts_threshold = EARLY_EXIT_MINUTES.get(orb_label)
    _annotate_time_stop(
        result, ts_threshold, post_entry,
        entry_ts, entry_price, stop_price, break_dir, cost_spec,
    )
```

Also add `ts_outcome`, `ts_pnl_r`, `ts_exit_ts` to the initial `result` dict (line 329-343).

Note: for the early-return paths in `compute_single_outcome` (no signal at line 358, zero risk at line 366, fill-bar exit at line 388, empty post_entry at line 400), the ts_* keys must also be present. The `_annotate_time_stop` call only runs on the normal path (line ~474). For early returns, the initial dict has ts_* = None, which is correct (no entry or no data → no time-stop applies). **But** for fill-bar exits (line 388), if the session has a threshold, the time-stop is moot (trade resolved immediately). We should annotate these too.

Add after `result.update(fill_exit)` at line 389, before the return:

```python
        # Fill-bar exit → resolved before any time-stop
        if ts_threshold is not None:
            result["ts_outcome"] = result["outcome"]
            result["ts_pnl_r"] = result["pnl_r"]
            result["ts_exit_ts"] = result["exit_ts"]
```

And add the `ts_threshold` lookup earlier (after `result` dict initialization, ~line 344):

```python
    ts_threshold = EARLY_EXIT_MINUTES.get(orb_label)
```

Also ensure `EARLY_EXIT_MINUTES` is imported at the top of outcome_builder.py:

```python
from trading_app.config import EARLY_EXIT_MINUTES
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trading_app/test_outcome_builder.py::TestTimeStopIntegration -x -v`
Expected: PASS

**Step 5: Commit**

```bash
git add trading_app/outcome_builder.py tests/test_trading_app/test_outcome_builder.py
git commit -m "feat: wire T80 time-stop into outcome computation"
```

---

### Task 4: Update batch insert to include time-stop columns

**Files:**
- Modify: `trading_app/outcome_builder.py:641-677` (build_outcomes batch insert)

**Step 1: Update the batch append**

In `build_outcomes()`, update the `day_batch.append` (line 642-652) to include the 3 new columns:

```python
                            day_batch.append([
                                trading_day, symbol, orb_label, orb_minutes,
                                rr_target, cb, em,
                                outcome["entry_ts"], outcome["entry_price"],
                                outcome["stop_price"], outcome["target_price"],
                                outcome["outcome"], outcome["exit_ts"],
                                outcome["exit_price"], outcome["pnl_r"],
                                outcome["risk_dollars"], outcome["pnl_dollars"],
                                outcome["mae_r"], outcome["mfe_r"],
                                outcome.get("ambiguous_bar", False),
                                outcome.get("ts_outcome"),
                                outcome.get("ts_pnl_r"),
                                outcome.get("ts_exit_ts"),
                            ])
```

Update the `columns=` list (line 659-667):

```python
                    columns=[
                        'trading_day', 'symbol', 'orb_label', 'orb_minutes',
                        'rr_target', 'confirm_bars', 'entry_model',
                        'entry_ts', 'entry_price', 'stop_price', 'target_price',
                        'outcome', 'exit_ts', 'exit_price', 'pnl_r',
                        'risk_dollars', 'pnl_dollars',
                        'mae_r', 'mfe_r',
                        'ambiguous_bar',
                        'ts_outcome', 'ts_pnl_r', 'ts_exit_ts',
                    ],
```

Update the INSERT SQL (line 669-677):

```python
                con.execute("""
                    INSERT OR REPLACE INTO orb_outcomes
                    SELECT trading_day, symbol, orb_label, orb_minutes,
                           rr_target, confirm_bars, entry_model,
                           entry_ts, entry_price, stop_price, target_price,
                           outcome, exit_ts, exit_price, pnl_r,
                           risk_dollars, pnl_dollars,
                           mae_r, mfe_r, ambiguous_bar,
                           ts_outcome, ts_pnl_r, ts_exit_ts
                    FROM batch_df
                """)
```

**Step 2: Run full test suite**

Run: `python -m pytest tests/test_trading_app/test_outcome_builder.py -x -v`
Expected: ALL PASS

**Step 3: Run drift check**

Run: `python pipeline/check_drift.py`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add trading_app/outcome_builder.py
git commit -m "feat: include time-stop columns in batch insert"
```

---

### Task 5: Full validation

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass (except known flaky/broken tests noted in MEMORY.md)

**Step 2: Run drift check**

Run: `python pipeline/check_drift.py`
Expected: 29/29 pass

**Step 3: Smoke test — dry run outcome build**

Run: `python trading_app/outcome_builder.py --instrument MGC --start 2025-01-01 --end 2025-01-05 --dry-run`
Expected: Runs without error, reports row counts

**Step 4: Final commit if any fixups needed**

```bash
git add -A && git commit -m "fix: address validation feedback for T80 time-stop"
```
