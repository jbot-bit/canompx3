# Parallel Walkforward Validation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Parallelize walkforward validation across multiple CPU cores, reducing runtime from hours to ~15-30 minutes.

**Architecture:** Serial phases 1-5 cull cheap failures, survivors go to ProcessPoolExecutor for walkforward + DST split. Workers open read-only DuckDB connections. Results batched, written in single transaction at end.

**Tech Stack:** Python `concurrent.futures.ProcessPoolExecutor`, DuckDB read-only connections, `time` module for speedup logging.

---

### Task 1: Add `--workers` CLI flag

**Files:**
- Modify: `trading_app/strategy_validator.py:786-788` (argparse section)
- Modify: `trading_app/strategy_validator.py:793-811` (main() call)

**Step 1: Add the argument to argparse**

After the `--no-regime-waivers` argument (line 787), add:

```python
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for walkforward (default: min(8, cpu_count-1), 1=serial)")
```

**Step 2: Pass workers to run_validation in main()**

Add `workers=args.workers` to the `run_validation()` call.

**Step 3: Add `workers` parameter to `run_validation()` signature**

Add `workers: int | None = None` to the function signature (line 453).

**Step 4: Run tests to verify no breakage**

Run: `python -m pytest tests/test_trading_app/ -x -q`
Expected: All existing tests pass (no behavioral change yet)

**Step 5: Commit**

```bash
git add trading_app/strategy_validator.py
git commit -m "feat: add --workers CLI flag to strategy_validator (no-op)"
```

---

### Task 2: Create the worker function

**Files:**
- Modify: `trading_app/strategy_validator.py` (add function before `run_validation`)

**Step 1: Add imports at top of file**

Add to imports section (after line 24):
```python
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
```

**Step 2: Write `_walkforward_worker()` function**

Add before `run_validation()` (around line 452):

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
) -> dict:
    """Worker function for parallel walkforward. Runs in a subprocess.

    Opens its own read-only DuckDB connection. Returns a plain dict
    (must be serialization-safe — no connection objects, no DataFrames).
    """
    t0 = time.monotonic()

    import duckdb
    from pipeline.db_config import configure_connection
    from trading_app.walkforward import run_walkforward
    from trading_app.strategy_discovery import parse_dst_regime

    result = {
        "strategy_id": strategy_id,
        "wf_result": None,
        "dst_split": None,
        "wf_duration_s": 0.0,
        "error": None,
    }

    try:
        with duckdb.connect(db_path_str, read_only=True) as con:
            configure_connection(con, writing=False)

            # Phase 4b: Walk-forward
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
            )
            result["wf_result"] = {
                "passed": wf_result.passed,
                "rejection_reason": wf_result.rejection_reason,
                "as_dict": {
                    k: v for k, v in wf_result.__dict__.items()
                },
            }

            # DST split (recompute only for blended strategies missing data)
            if dst_verdict_from_discovery is not None:
                result["dst_split"] = dst_cols_from_discovery
            elif dst_regime is None:
                dst_split = compute_dst_split(
                    con, strategy_id, instrument,
                    orb_label=orb_label,
                    entry_model=entry_model,
                    rr_target=rr_target,
                    confirm_bars=confirm_bars,
                    filter_type=filter_type,
                    filter_params=filter_params,
                    orb_minutes=orb_minutes,
                )
                result["dst_split"] = dst_split
            else:
                result["dst_split"] = {
                    "winter_n": None, "winter_avg_r": None,
                    "summer_n": None, "summer_avg_r": None,
                    "verdict": None,
                }

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    result["wf_duration_s"] = time.monotonic() - t0
    return result
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_trading_app/ -x -q`
Expected: Pass (function exists but not called yet)

**Step 4: Commit**

```bash
git add trading_app/strategy_validator.py
git commit -m "feat: add _walkforward_worker function for parallel validation"
```

---

### Task 3: Refactor `run_validation()` — split serial and parallel phases

This is the core change. The current `for row in rows:` loop (lines 516-697) gets split into:
1. **Phase A (serial):** Phases 1-5 cull, collect survivors
2. **Phase B (parallel):** Walkforward + DST for survivors via pool
3. **Phase C (serial):** Batch write all results

**Files:**
- Modify: `trading_app/strategy_validator.py:453-751` (run_validation body)

**Step 1: Resolve default workers at top of `run_validation()`**

After `cost_spec = get_cost_spec(instrument)` (line 481), add:

```python
    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    use_parallel = workers > 1 and enable_walkforward
```

**Step 2: Refactor the main loop into Phase A (serial cull)**

Replace the body of `run_validation()` from `with duckdb.connect(...)` (line 483) through end. The new structure:

```python
    # ── Phase A: Load strategies + serial cull (phases 1-5) ──────────
    with duckdb.connect(str(db_path)) as con:
        from pipeline.db_config import configure_connection
        configure_connection(con, writing=True)

        if not dry_run:
            init_trading_app_schema(db_path=db_path)

        rows = con.execute(
            """SELECT * FROM experimental_strategies
               WHERE instrument = ?
               AND (validation_status IS NULL OR validation_status = '')
               ORDER BY strategy_id""",
            [instrument],
        ).fetchall()
        col_names = [desc[0] for desc in con.description]

        atr_by_year = {}
        if enable_regime_waivers:
            atr_rows = con.execute("""
                SELECT EXTRACT(YEAR FROM trading_day) as yr, AVG(atr_20) as mean_atr
                FROM daily_features
                WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL
                GROUP BY yr
            """, [instrument]).fetchall()
            atr_by_year = {int(r[0]): r[1] for r in atr_rows}

    # Connection closed here — DuckDB requires no write connection open
    # while worker processes open read-only connections.

    passed = 0
    rejected = 0
    skipped_aliases = 0
    passed_strategy_ids = []

    # Phase A results: list of dicts with serial validation outcome
    serial_results = []  # Each: {row_dict, status, notes, regime_waivers, strat_dst_regime}
    wf_candidates = []   # Survivors needing walkforward

    for row in rows:
        row_dict = dict(zip(col_names, row))
        strategy_id = row_dict["strategy_id"]

        if row_dict.get("is_canonical") is False:
            skipped_aliases += 1
            serial_results.append({
                "row_dict": row_dict, "status": "SKIPPED",
                "notes": "Alias (non-canonical)", "regime_waivers": [],
                "dst_split": {"winter_n": None, "winter_avg_r": None,
                              "summer_n": None, "summer_avg_r": None, "verdict": None},
            })
            continue

        status, notes, regime_waivers = validate_strategy(
            row_dict, cost_spec,
            stress_multiplier=stress_multiplier,
            min_sample=min_sample,
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown,
            exclude_years=exclude_years,
            min_years_positive_pct=min_years_positive_pct,
            min_trades_per_year=min_trades_per_year,
            atr_by_year=atr_by_year if enable_regime_waivers else None,
            enable_regime_waivers=enable_regime_waivers,
        )

        strat_dst_regime = parse_dst_regime(strategy_id)

        if status == "PASSED" and enable_walkforward:
            wf_candidates.append({
                "row_dict": row_dict, "status": status, "notes": notes,
                "regime_waivers": regime_waivers,
                "strat_dst_regime": strat_dst_regime,
            })
        else:
            # Rejected in phases 1-5 or walkforward disabled
            # Still need DST split for the record
            dst_split = {"winter_n": None, "winter_avg_r": None,
                         "summer_n": None, "summer_avg_r": None, "verdict": None}
            serial_results.append({
                "row_dict": row_dict, "status": status, "notes": notes,
                "regime_waivers": regime_waivers, "dst_split": dst_split,
            })
            if status == "PASSED":
                passed += 1
                passed_strategy_ids.append(strategy_id)
            else:
                rejected += 1

    logger.info(
        f"Phase A complete: {len(wf_candidates)} survivors for walkforward, "
        f"{rejected} rejected, {skipped_aliases} aliases skipped "
        f"(of {len(rows)} strategies)"
    )
```

**Step 3: Add Phase B — parallel walkforward**

```python
    # ── Phase B: Parallel walkforward for survivors ──────────────────
    wf_results_map = {}  # strategy_id -> worker result dict

    if wf_candidates:
        wf_params = {
            "test_window_months": wf_test_months,
            "min_train_months": wf_min_train_months,
            "min_trades": wf_min_trades,
            "min_windows": wf_min_windows,
            "min_pct_positive": wf_min_pct_positive,
        }

        wall_start = time.monotonic()
        total_wf_duration = 0.0

        if use_parallel:
            logger.info(f"Starting parallel walkforward with {workers} workers for {len(wf_candidates)} strategies...")

            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_sid = {}
                for cand in wf_candidates:
                    rd = cand["row_dict"]
                    sid = rd["strategy_id"]
                    future = executor.submit(
                        _walkforward_worker,
                        strategy_id=sid,
                        instrument=instrument,
                        orb_label=rd["orb_label"],
                        entry_model=rd.get("entry_model", "E1"),
                        rr_target=rd["rr_target"],
                        confirm_bars=rd["confirm_bars"],
                        filter_type=rd.get("filter_type", "NO_FILTER"),
                        filter_params=rd.get("filter_params"),
                        orb_minutes=rd.get("orb_minutes", 5),
                        db_path_str=str(db_path),
                        wf_params=wf_params,
                        dst_regime=cand["strat_dst_regime"],
                        dst_verdict_from_discovery=rd.get("dst_verdict"),
                        dst_cols_from_discovery={
                            "winter_n": rd.get("dst_winter_n"),
                            "winter_avg_r": rd.get("dst_winter_avg_r"),
                            "summer_n": rd.get("dst_summer_n"),
                            "summer_avg_r": rd.get("dst_summer_avg_r"),
                            "verdict": rd.get("dst_verdict"),
                        } if rd.get("dst_verdict") is not None else None,
                    )
                    future_to_sid[future] = sid

                for future in as_completed(future_to_sid):
                    sid = future_to_sid[future]
                    try:
                        result = future.result()
                        wf_results_map[sid] = result
                        total_wf_duration += result.get("wf_duration_s", 0)
                    except Exception as e:
                        logger.error(f"Worker exception for {sid}: {e}")
                        wf_results_map[sid] = {
                            "strategy_id": sid, "wf_result": None,
                            "dst_split": None, "error": str(e),
                            "wf_duration_s": 0,
                        }
        else:
            # Serial fallback (--workers 1 or --no-walkforward already handled)
            logger.info(f"Running walkforward serially for {len(wf_candidates)} strategies...")
            with duckdb.connect(str(db_path), read_only=True) as con:
                from pipeline.db_config import configure_connection
                configure_connection(con, writing=False)

                for cand in wf_candidates:
                    rd = cand["row_dict"]
                    sid = rd["strategy_id"]
                    result = _walkforward_worker(
                        strategy_id=sid,
                        instrument=instrument,
                        orb_label=rd["orb_label"],
                        entry_model=rd.get("entry_model", "E1"),
                        rr_target=rd["rr_target"],
                        confirm_bars=rd["confirm_bars"],
                        filter_type=rd.get("filter_type", "NO_FILTER"),
                        filter_params=rd.get("filter_params"),
                        orb_minutes=rd.get("orb_minutes", 5),
                        db_path_str=str(db_path),
                        wf_params=wf_params,
                        dst_regime=cand["strat_dst_regime"],
                        dst_verdict_from_discovery=rd.get("dst_verdict"),
                        dst_cols_from_discovery={
                            "winter_n": rd.get("dst_winter_n"),
                            "winter_avg_r": rd.get("dst_winter_avg_r"),
                            "summer_n": rd.get("dst_summer_n"),
                            "summer_avg_r": rd.get("dst_summer_avg_r"),
                            "verdict": rd.get("dst_verdict"),
                        } if rd.get("dst_verdict") is not None else None,
                    )
                    wf_results_map[sid] = result
                    total_wf_duration += result.get("wf_duration_s", 0)

        wall_elapsed = time.monotonic() - wall_start
        speedup = total_wf_duration / wall_elapsed if wall_elapsed > 0 else 1.0
        logger.info(
            f"Walkforward complete: wall={wall_elapsed:.1f}s, "
            f"sum(worker)={total_wf_duration:.1f}s, "
            f"speedup={speedup:.1f}x ({workers} workers)"
        )

        # Merge walkforward results into serial_results
        for cand in wf_candidates:
            rd = cand["row_dict"]
            sid = rd["strategy_id"]
            wr = wf_results_map.get(sid, {})

            status = cand["status"]
            notes = cand["notes"]

            if wr.get("error"):
                status = "REJECTED"
                notes = f"Phase 4b: Worker error: {wr['error']}"
            elif wr.get("wf_result") and not wr["wf_result"]["passed"]:
                status = "REJECTED"
                notes = f"Phase 4b: {wr['wf_result']['rejection_reason']}"

            dst_split = wr.get("dst_split") or {
                "winter_n": None, "winter_avg_r": None,
                "summer_n": None, "summer_avg_r": None, "verdict": None,
            }

            serial_results.append({
                "row_dict": rd, "status": status, "notes": notes,
                "regime_waivers": cand["regime_waivers"],
                "dst_split": dst_split,
                "wf_result_dict": wr.get("wf_result"),
            })

            if status == "PASSED":
                passed += 1
                passed_strategy_ids.append(sid)
            else:
                rejected += 1
```

**Step 4: Add Phase C — batch writes**

```python
    # ── Phase C: Batch write all results ─────────────────────────────
    if not dry_run:
        with duckdb.connect(str(db_path)) as con:
            from pipeline.db_config import configure_connection
            configure_connection(con, writing=True)

            for sr in serial_results:
                rd = sr["row_dict"]
                sid = rd["strategy_id"]
                status = sr["status"]
                notes = sr["notes"]
                dst_split = sr["dst_split"]

                if status == "SKIPPED":
                    con.execute(
                        """UPDATE experimental_strategies
                           SET validation_status = 'SKIPPED',
                               validation_notes = 'Alias (non-canonical)'
                           WHERE strategy_id = ?""",
                        [sid],
                    )
                    continue

                # Update experimental_strategies
                con.execute(
                    """UPDATE experimental_strategies
                       SET validation_status = ?, validation_notes = ?,
                           dst_winter_n = ?, dst_winter_avg_r = ?,
                           dst_summer_n = ?, dst_summer_avg_r = ?,
                           dst_verdict = ?
                       WHERE strategy_id = ?""",
                    [status, notes,
                     dst_split.get("winter_n"), dst_split.get("winter_avg_r"),
                     dst_split.get("summer_n"), dst_split.get("summer_avg_r"),
                     dst_split.get("verdict"),
                     sid],
                )

                if status == "PASSED":
                    yearly = rd.get("yearly_results", "{}")
                    try:
                        yearly_data = json.loads(yearly) if isinstance(yearly, str) else yearly
                    except (json.JSONDecodeError, TypeError):
                        yearly_data = {}

                    included = {y: d for y, d in yearly_data.items()
                                if int(y) not in (exclude_years or set())}
                    years_tested = len(included)
                    all_positive = all(
                        d.get("avg_r", 0) > 0 for d in included.values()
                    )
                    regime_waivers = sr["regime_waivers"]

                    con.execute(
                        """INSERT OR REPLACE INTO validated_setups
                           (strategy_id, promoted_from, instrument, orb_label,
                            orb_minutes, rr_target, confirm_bars, entry_model,
                            filter_type, filter_params,
                            sample_size, win_rate, expectancy_r,
                            years_tested, all_years_positive, stress_test_passed,
                            sharpe_ratio, max_drawdown_r,
                            trades_per_year, sharpe_ann,
                            yearly_results, status,
                            median_risk_dollars, avg_risk_dollars,
                            avg_win_dollars, avg_loss_dollars,
                            regime_waivers, regime_waiver_count,
                            dst_winter_n, dst_winter_avg_r,
                            dst_summer_n, dst_summer_avg_r,
                            dst_verdict)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            sid, sid,
                            rd["instrument"], rd["orb_label"],
                            rd["orb_minutes"], rd["rr_target"],
                            rd["confirm_bars"], rd.get("entry_model", "E1"),
                            rd.get("filter_type", ""),
                            rd.get("filter_params", ""),
                            rd.get("sample_size", 0),
                            rd.get("win_rate", 0),
                            rd.get("expectancy_r", 0),
                            years_tested, all_positive, True,
                            rd.get("sharpe_ratio"),
                            rd.get("max_drawdown_r"),
                            rd.get("trades_per_year"),
                            rd.get("sharpe_ann"),
                            yearly, "active",
                            rd.get("median_risk_dollars"),
                            rd.get("avg_risk_dollars"),
                            rd.get("avg_win_dollars"),
                            rd.get("avg_loss_dollars"),
                            json.dumps(regime_waivers) if regime_waivers else None,
                            len(regime_waivers),
                            dst_split.get("winter_n"), dst_split.get("winter_avg_r"),
                            dst_split.get("summer_n"), dst_split.get("summer_avg_r"),
                            dst_split.get("verdict"),
                        ],
                    )

            # Write walkforward JSONL (batch)
            from trading_app.walkforward import WalkForwardResult, append_walkforward_result
            for sr in serial_results:
                wfr = sr.get("wf_result_dict")
                if wfr and wfr.get("as_dict"):
                    wd = wfr["as_dict"]
                    wf_obj = WalkForwardResult(**{
                        k: v for k, v in wd.items()
                        if k in WalkForwardResult.__dataclass_fields__
                    })
                    append_walkforward_result(wf_obj, wf_output_path)

            # FDR correction (unchanged)
            if passed_strategy_ids:
                logger.info("Computing FDR correction (Benjamini-Hochberg)...")
                all_p_values = con.execute(
                    """SELECT strategy_id, p_value FROM experimental_strategies
                       WHERE instrument = ?
                       AND is_canonical = TRUE
                       AND p_value IS NOT NULL""",
                    [instrument],
                ).fetchall()
                p_value_list = [(r[0], r[1]) for r in all_p_values]
                fdr_results = benjamini_hochberg(p_value_list, alpha=0.05)

                n_fdr_sig = 0
                n_fdr_insig = 0
                for sid in passed_strategy_ids:
                    fdr = fdr_results.get(sid)
                    if fdr is not None:
                        con.execute(
                            """UPDATE validated_setups
                               SET fdr_significant = ?,
                                   fdr_adjusted_p = ?
                               WHERE strategy_id = ?""",
                            [fdr["fdr_significant"], fdr["adjusted_p"], sid],
                        )
                        if fdr["fdr_significant"]:
                            n_fdr_sig += 1
                        else:
                            n_fdr_insig += 1

                logger.info(
                    f"  FDR results: {n_fdr_sig} significant, {n_fdr_insig} not significant "
                    f"(of {len(passed_strategy_ids)} passed, from {len(p_value_list)} tested)"
                )
                if n_fdr_insig > 0:
                    logger.info(
                        f"  WARNING: {n_fdr_insig} validated strategies do NOT survive "
                        f"FDR correction — potential false positives"
                    )

            con.commit()

    logger.info(f"Validation complete: {passed} PASSED, {rejected} REJECTED, "
                f"{skipped_aliases} aliases skipped "
                f"(of {len(rows)} strategies)")
    if dry_run:
        logger.info("  (DRY RUN — no data written)")

    return passed, rejected
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_trading_app/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add trading_app/strategy_validator.py
git commit -m "feat: parallel walkforward validation with --workers flag"
```

---

### Task 4: Smoke test with real database

**Step 1: Dry-run serial (baseline)**

```bash
python trading_app/strategy_validator.py --instrument MGC --min-sample 50 \
  --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward --dry-run
```

Verify: Runs without error, shows cull counts.

**Step 2: Dry-run parallel (verify workers start)**

```bash
python trading_app/strategy_validator.py --instrument MGC --min-sample 50 \
  --no-regime-waivers --min-years-positive-pct 0.75 --workers 4 --dry-run
```

Verify: Shows "Starting parallel walkforward with 4 workers" and speedup log.

**Step 3: Verify serial fallback**

```bash
python trading_app/strategy_validator.py --instrument MGC --min-sample 50 \
  --no-regime-waivers --min-years-positive-pct 0.75 --workers 1 --dry-run
```

Verify: Shows "Running walkforward serially" message.

**Step 4: Commit (if smoke tests pass)**

```bash
git add trading_app/strategy_validator.py
git commit -m "test: smoke test parallel walkforward — verified with dry-run"
```

---

### Task 5: Run drift checks and full test suite

**Step 1: Drift check**

```bash
python pipeline/check_drift.py
```

Expected: All 28+ checks pass.

**Step 2: Full test suite**

```bash
python -m pytest tests/ -x -q
```

Expected: All pass.

**Step 3: Final commit if any cleanup needed**

---

## Notes

- The worker function imports modules inside the function body because subprocess forking requires fresh imports
- `WalkForwardResult` reconstruction from dict is needed because the dataclass crosses the process boundary as a plain dict
- The serial fallback (`--workers 1`) should produce byte-identical DB output to the current code — this is the regression test
- DuckDB `read_only=True` is the key safety guarantee — workers physically cannot write
