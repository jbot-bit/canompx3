#!/usr/bin/env python3
"""
Synthetic Null Pipeline Test — the most important test in this project.

Generates random-walk bars (zero drift, zero signal), runs the full ORB
breakout pipeline, and asserts ZERO strategies validate. A pipeline that
extracts "edge" from noise is fundamentally broken.

This is a *falsification* test: it doesn't verify the code does what we
want — it verifies the code doesn't do the thing we're afraid of
(manufacturing false edges from structureless data).

Null model: Gaussian random walk, i.i.d. increments ~ N(0, σ), zero mean.
No autocorrelation, no vol clustering, no fat tails. If the pipeline can't
even reject this weakest possible null, nothing it produces can be trusted.

Usage:
    python scripts/tests/test_synthetic_null.py                     # 1 seed, all apertures
    python scripts/tests/test_synthetic_null.py --seeds 5           # 5 seeds
    python scripts/tests/test_synthetic_null.py --apertures 5       # fast: 5m only
    python scripts/tests/test_synthetic_null.py --keep-db           # keep temp DB for inspection
    python scripts/tests/test_synthetic_null.py --sigma 2.0         # wider bars (bigger ORBs)

Expected runtime: 20-60 min per seed depending on apertures and hardware.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Bar generation
# ---------------------------------------------------------------------------


def generate_synthetic_bars(
    db_path: Path,
    *,
    seed: int,
    symbol: str = "MGC",
    start_date: date = date(2020, 1, 1),
    end_date: date = date(2025, 12, 31),
    sigma: float = 1.2,
    start_price: float = 2000.0,
    tick_size: float = 0.10,
) -> dict:
    """Generate random-walk 1m bars with zero drift and insert into bars_1m.

    The random walk is the weakest null hypothesis in quantitative finance
    (Fama 1970). It produces:
    - ORB ranges (from random intrabar variation)
    - ORB breaks (price randomly crosses range boundaries)
    - Wins and losses (purely by chance)
    - Occasional streaks that *look* like edge

    The question: can the validation chain reject all of it?

    Calibration (sigma=1.2):
    - 5m ORB range: ~4-8 points (some pass G4/G6 filters — intentional)
    - Daily ATR: ~40-60 points (mid-range for gold)
    - Cost drag is realistic, not dominant — statistical gates must do the work

    Returns dict with row_count, price stats, and null-property verification.
    """
    rng = np.random.default_rng(seed)

    # Business days only (Mon-Fri). Includes market holidays — harmless for null test.
    bdays = pd.bdate_range(start_date, end_date)
    n_days = len(bdays)
    n_bars_per_day = 1440  # 00:00 - 23:59 UTC, every minute
    total_bars = n_days * n_bars_per_day

    print(f"  Generating {total_bars:,} bars ({n_days} days x {n_bars_per_day} min/day)...")

    # --- Vectorized random walk (zero drift) ---
    increments = rng.normal(0, sigma, size=total_bars)
    closes_raw = start_price + np.cumsum(increments)

    # Opens: each bar opens at previous bar's close
    opens_raw = np.empty(total_bars)
    opens_raw[0] = start_price
    opens_raw[1:] = closes_raw[:-1]

    # Round to tick size
    closes = np.round(closes_raw / tick_size) * tick_size
    opens = np.round(opens_raw / tick_size) * tick_size

    # Intrabar high/low: extend beyond open-close range
    # Uses 0.3*sigma — enough to create realistic wicks without inflating ORBs excessively
    high_ext = np.abs(rng.normal(0, sigma * 0.3, size=total_bars))
    low_ext = np.abs(rng.normal(0, sigma * 0.3, size=total_bars))

    highs = np.round((np.maximum(opens, closes) + high_ext) / tick_size) * tick_size
    lows = np.round((np.minimum(opens, closes) - low_ext) / tick_size) * tick_size

    # Enforce OHLC validity after rounding
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    # Floor at tick_size (random walk could theoretically go negative with σ=1.2
    # after 2.25M steps starting at 2000 — extremely unlikely but defensive)
    floor = tick_size
    opens = np.maximum(opens, floor)
    closes = np.maximum(closes, floor)
    highs = np.maximum(highs, floor)
    lows = np.maximum(lows, floor)

    # Volumes: random integers (no signal in volume either)
    volumes = rng.integers(50, 500, size=total_bars).astype(np.int64)

    # --- Timestamps: every minute of every business day ---
    print("  Building timestamps...")
    ts_arrays = []
    for d in bdays:
        base = pd.Timestamp(d.date(), tz="UTC")
        ts_arrays.append(pd.date_range(base, periods=n_bars_per_day, freq="min"))
    timestamps = np.concatenate(ts_arrays)

    # --- Insert into DB in chunks ---
    print("  Inserting into bars_1m...")
    con = duckdb.connect(str(db_path))
    chunk_size = 200_000

    try:
        for i in range(0, total_bars, chunk_size):
            j = min(i + chunk_size, total_bars)
            chunk = pd.DataFrame(
                {
                    "ts_utc": timestamps[i:j],
                    "symbol": symbol,
                    "source_symbol": symbol,
                    "open": opens[i:j],
                    "high": highs[i:j],
                    "low": lows[i:j],
                    "close": closes[i:j],
                    "volume": volumes[i:j],
                }
            )
            con.register("chunk_view", chunk)
            con.execute("INSERT INTO bars_1m SELECT * FROM chunk_view")
            con.unregister("chunk_view")
    finally:
        con.close()

    # --- Null-property verification ---
    returns = np.diff(closes_raw) / np.where(closes_raw[:-1] != 0, closes_raw[:-1], 1)
    autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1]) if len(returns) > 1 else 0.0

    stats = {
        "row_count": total_bars,
        "n_days": n_days,
        "price_min": float(np.min(lows)),
        "price_max": float(np.max(highs)),
        "price_final": float(closes[-1]),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "autocorr_lag1": autocorr,
    }

    return stats


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def run_step(name: str, cmd: list[str], env: dict, timeout: int = 3600) -> bool:
    """Run a pipeline step as subprocess. Returns True on success."""
    print(f"\n--- {name} ---")
    t0 = time.time()

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        if result.stderr:
            # Show last 800 chars of stderr for diagnostics
            print(f"  stderr: ...{result.stderr[-800:]}")
        if result.stdout:
            print(f"  stdout: ...{result.stdout[-500:]}")
        return False

    print(f"  OK ({elapsed:.1f}s)")
    # Show last few lines of stdout for progress visibility
    if result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-3:]:
            print(f"  {line}")

    return True


# ---------------------------------------------------------------------------
# Results query
# ---------------------------------------------------------------------------


def query_results(db_path: Path) -> dict:
    """Query the temp DB for diagnostics."""
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        results = {}

        # Table row counts
        for table in [
            "bars_1m",
            "bars_5m",
            "daily_features",
            "orb_outcomes",
            "experimental_strategies",
            "validated_setups",
        ]:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                n = -1
            results[f"{table}_rows"] = n

        # --- Discovery funnel ---

        # Strategies with any trades (sample_size > 0)
        try:
            results["exp_with_trades"] = con.execute(
                "SELECT COUNT(*) FROM experimental_strategies WHERE sample_size > 0"
            ).fetchone()[0]
        except Exception:
            results["exp_with_trades"] = -1

        # Strategies with positive expectancy
        try:
            results["exp_positive_expr"] = con.execute(
                "SELECT COUNT(*) FROM experimental_strategies WHERE expectancy_r > 0 AND sample_size > 0"
            ).fetchone()[0]
        except Exception:
            results["exp_positive_expr"] = -1

        # Strategies passing Phase 1 threshold (sample >= 30 AND positive ExpR)
        try:
            results["exp_pass_phase1"] = con.execute(
                "SELECT COUNT(*) FROM experimental_strategies WHERE sample_size >= 30 AND expectancy_r > 0"
            ).fetchone()[0]
        except Exception:
            results["exp_pass_phase1"] = -1

        # The key metric: validated strategies
        try:
            results["validated_count"] = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
        except Exception:
            results["validated_count"] = -1

        try:
            results["validated_promoted"] = con.execute(
                "SELECT COUNT(*) FROM validated_setups WHERE status = 'PROMOTED'"
            ).fetchone()[0]
        except Exception:
            results["validated_promoted"] = -1

        # --- Survivors (if any) ---
        try:
            survivors = con.execute(
                """
                SELECT strategy_id, instrument, orb_label, orb_minutes,
                       entry_model, confirm_bars, filter_type, rr_target,
                       sample_size, win_rate, expectancy_r, sharpe_ratio,
                       sharpe_ann, max_drawdown_r, trades_per_year, status
                FROM validated_setups
                ORDER BY expectancy_r DESC
                """
            ).fetchdf()
            results["survivors"] = survivors
        except Exception:
            results["survivors"] = pd.DataFrame()

        # --- ORB break statistics (sanity check: are ORBs actually forming?) ---
        try:
            break_stats = con.execute(
                """
                SELECT orb_minutes,
                       COUNT(*) as total_days,
                       COUNT(orb_CME_REOPEN_break_dir) as cme_reopen_breaks,
                       COUNT(orb_TOKYO_OPEN_break_dir) as tokyo_breaks,
                       COUNT(orb_NYSE_OPEN_break_dir) as nyse_breaks,
                       AVG(orb_CME_REOPEN_size) as avg_cme_reopen_orb,
                       AVG(orb_TOKYO_OPEN_size) as avg_tokyo_orb,
                       AVG(atr_20) as avg_atr
                FROM daily_features
                GROUP BY orb_minutes
                ORDER BY orb_minutes
                """
            ).fetchdf()
            results["break_stats"] = break_stats
        except Exception:
            results["break_stats"] = pd.DataFrame()

        # --- Outcome distribution (sanity: are outcomes computing?) ---
        try:
            outcome_stats = con.execute(
                """
                SELECT entry_model,
                       COUNT(*) as total,
                       COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
                       COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses,
                       COUNT(CASE WHEN outcome IS NULL THEN 1 END) as no_entry,
                       ROUND(AVG(CASE WHEN outcome IN ('win','loss') THEN pnl_r END), 4) as avg_pnl_r
                FROM orb_outcomes
                GROUP BY entry_model
                ORDER BY entry_model
                """
            ).fetchdf()
            results["outcome_stats"] = outcome_stats
        except Exception:
            results["outcome_stats"] = pd.DataFrame()

        # --- ExpR distribution of experimental strategies (how much false edge exists?) ---
        try:
            expr_dist = con.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN expectancy_r > 0 THEN 1 END) as positive,
                    COUNT(CASE WHEN expectancy_r > 0.1 THEN 1 END) as gt_0_1,
                    COUNT(CASE WHEN expectancy_r > 0.2 THEN 1 END) as gt_0_2,
                    ROUND(AVG(expectancy_r), 4) as mean_expr,
                    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY expectancy_r), 4) as median_expr,
                    ROUND(MAX(expectancy_r), 4) as max_expr
                FROM experimental_strategies
                WHERE sample_size > 0
                """
            ).fetchdf()
            results["expr_distribution"] = expr_dist
        except Exception:
            results["expr_distribution"] = pd.DataFrame()

        # --- Rejection phase breakdown (where did the validator catch false positives?) ---
        try:
            # validated_setups only has survivors. Check experimental_strategies for rejection info.
            # The validator sets validation_status on experimental_strategies.
            rejection_phases = con.execute(
                """
                SELECT
                    CASE
                        WHEN validation_status IS NULL THEN 'not_validated'
                        WHEN validation_status = 'PROMOTED' THEN 'PROMOTED'
                        ELSE validation_status
                    END as status,
                    COUNT(*) as cnt
                FROM experimental_strategies
                GROUP BY 1
                ORDER BY cnt DESC
                """
            ).fetchdf()
            results["rejection_phases"] = rejection_phases
        except Exception:
            results["rejection_phases"] = pd.DataFrame()

        return results

    finally:
        con.close()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(seed: int, bar_stats: dict, results: dict) -> None:
    """Print formatted report for one seed."""
    print(f"\n{'='*60}")
    print(f"RESULTS — Seed {seed}")
    print(f"{'='*60}")

    # Bar generation stats
    print(f"\n  Bars generated:     {bar_stats['row_count']:,} ({bar_stats['n_days']} days)")
    print(f"  Price range:        {bar_stats['price_min']:.1f} — {bar_stats['price_max']:.1f}")
    print(f"  Final price:        {bar_stats['price_final']:.1f}")
    print(f"  Mean 1m return:     {bar_stats['mean_return']:.8f} (should be ~0)")
    print(f"  Autocorr (lag 1):   {bar_stats['autocorr_lag1']:.6f} (should be ~0)")

    # Null verification
    if abs(bar_stats["mean_return"]) > 0.001:
        print("  WARNING: Mean return is suspiciously far from zero!")
    if abs(bar_stats["autocorr_lag1"]) > 0.05:
        print("  WARNING: Significant autocorrelation detected in synthetic data!")

    # Table row counts
    print("\n  Table rows:")
    for table in ["bars_1m", "bars_5m", "daily_features", "orb_outcomes", "experimental_strategies", "validated_setups"]:
        key = f"{table}_rows"
        print(f"    {table:30s} {results.get(key, '?'):>10,}")

    # Break stats (sanity: ORBs forming?)
    if not results.get("break_stats", pd.DataFrame()).empty:
        print("\n  ORB formation (sanity check):")
        bs = results["break_stats"]
        for _, row in bs.iterrows():
            print(
                f"    O{int(row['orb_minutes']):2d}: "
                f"{int(row['total_days'])} days, "
                f"CME_REOPEN breaks={int(row['cme_reopen_breaks'])}, "
                f"TOKYO breaks={int(row['tokyo_breaks'])}, "
                f"NYSE breaks={int(row['nyse_breaks'])}, "
                f"avg ORB(CME)={row['avg_cme_reopen_orb']:.1f}pts, "
                f"avg ATR={row['avg_atr']:.1f}pts"
            )

    # Outcome stats
    if not results.get("outcome_stats", pd.DataFrame()).empty:
        print("\n  Outcomes by entry model:")
        for _, row in results["outcome_stats"].iterrows():
            total = int(row["total"])
            wins = int(row["wins"])
            losses = int(row["losses"])
            no_entry = int(row["no_entry"])
            avg_pnl = row["avg_pnl_r"]
            avg_pnl_str = f"{avg_pnl:.4f}" if pd.notna(avg_pnl) else "N/A"
            wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            print(
                f"    {row['entry_model']}: "
                f"total={total:,}, W={wins:,}, L={losses:,}, no_entry={no_entry:,}, "
                f"WR={wr:.1f}%, avg_pnl_r={avg_pnl_str}"
            )

    # ExpR distribution
    if not results.get("expr_distribution", pd.DataFrame()).empty:
        ed = results["expr_distribution"].iloc[0]
        print("\n  ExpR distribution (experimental strategies with trades):")
        print(f"    Total:     {int(ed['total']):,}")
        print(f"    Positive:  {int(ed['positive']):,} ({int(ed['positive'])/max(int(ed['total']),1)*100:.1f}%)")
        print(f"    ExpR>0.1:  {int(ed['gt_0_1']):,}")
        print(f"    ExpR>0.2:  {int(ed['gt_0_2']):,}")
        print(f"    Mean ExpR: {ed['mean_expr']}")
        print(f"    Max ExpR:  {ed['max_expr']}")

    # Discovery funnel
    print("\n  VALIDATION FUNNEL:")
    print(f"    Discovered (sample>0):          {results.get('exp_with_trades', '?'):>6,}")
    print(f"    Positive ExpR (noise luck):     {results.get('exp_positive_expr', '?'):>6,}")
    print(f"    Pass Phase 1 (N>=30, ExpR>0):   {results.get('exp_pass_phase1', '?'):>6,}")
    print(f"    VALIDATED (all gates):           {results.get('validated_count', '?'):>6}")
    print(f"    PROMOTED:                        {results.get('validated_promoted', '?'):>6}")

    # Rejection breakdown
    if not results.get("rejection_phases", pd.DataFrame()).empty:
        print("\n  Rejection breakdown:")
        for _, row in results["rejection_phases"].iterrows():
            print(f"    {row['status']:30s} {int(row['cnt']):>6,}")

    # THE KEY CHECK — any row in validated_setups from noise is a failure.
    # Do not distinguish 'active' vs 'PROMOTED' — the table is called
    # validated_setups, not maybe_validated. If noise gets in, it's broken.
    validated = results.get("validated_count", -1)
    promoted = results.get("validated_promoted", 0)

    print(f"\n  {'='*50}")
    if validated == 0:
        print("  PASS: Zero strategies in validated_setups from noise")
    elif validated > 0:
        print(f"  FAIL: {validated} strategies in validated_setups from NOISE")
        print(f"        ({promoted} PROMOTED, {validated - promoted} active)")
        print("        The pipeline manufactures false edges.")
        if not results.get("survivors", pd.DataFrame()).empty:
            print("\n  FALSE POSITIVE SURVIVORS (top 20):")
            print(results["survivors"].head(20).to_string(index=False))
    else:
        print("  ERROR: Could not determine validation count")
    print(f"  {'='*50}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic null pipeline test: generates random walk bars, "
        "runs the full pipeline, verifies zero strategies validate."
    )
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds to test (default: 1)")
    parser.add_argument("--start-seed", type=int, default=42, help="First seed value (default: 42)")
    parser.add_argument(
        "--apertures",
        type=int,
        nargs="+",
        default=[5, 15, 30],
        choices=[5, 15, 30],
        help="ORB apertures to test (default: 5 15 30)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.2,
        help="Per-minute bar volatility in points (default: 1.2, produces ATR ~40-60)",
    )
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--keep-db", action="store_true", help="Keep temp database after test for inspection")
    parser.add_argument("--output-dir", type=str, default=None, help="Write DB to this directory instead of temp (implies --keep-db)")
    parser.add_argument("--instrument", type=str, default="MGC", help="Instrument symbol (default: MGC)")

    args = parser.parse_args()

    start_dt = date.fromisoformat(args.start_date)
    end_dt = date.fromisoformat(args.end_date)

    print("=" * 60)
    print("SYNTHETIC NULL PIPELINE TEST")
    print("=" * 60)
    print(f"  Seeds:       {args.seeds} (starting at {args.start_seed})")
    print(f"  Instrument:  {args.instrument}")
    print(f"  Date range:  {start_dt} to {end_dt}")
    print(f"  Apertures:   {args.apertures}")
    print(f"  Sigma:       {args.sigma} pts/min")
    print("  Null model:  Gaussian random walk, i.i.d. N(0, sigma), zero drift")
    print(f"  Keep DB:     {args.keep_db}")
    print()

    all_results = []
    any_failure = False

    for i in range(args.seeds):
        seed = args.start_seed + i
        t_seed_start = time.time()

        print(f"\n{'#'*60}")
        print(f"# SEED {seed} ({i+1}/{args.seeds})")
        print(f"{'#'*60}")

        # Create output directory: --output-dir (permanent) or temp
        if args.output_dir:
            tmpdir = str(Path(args.output_dir) / f"seed_{seed:04d}")
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
            args.keep_db = True  # implied
        else:
            tmpdir = tempfile.mkdtemp(prefix=f"null_test_seed{seed}_")
        db_path = Path(tmpdir) / "null_test.db"

        try:
            # Ensure DBN source directories exist (pipeline scripts check for them
            # even though they only read from the database, not DBN files).
            # These dirs are in .gitignore — creating empty ones is harmless.
            dbn_dir = PROJECT_ROOT / "DB" / "GOLD_DB_FULLSIZE"
            dbn_dir.mkdir(parents=True, exist_ok=True)

            # Create the DuckDB file so paths.py can find it via DUCKDB_PATH
            con = duckdb.connect(str(db_path))
            con.close()
            assert db_path.exists(), f"Failed to create temp DB at {db_path}"
            print(f"  Temp DB: {db_path}")

            # Safety: verify DUCKDB_PATH resolves correctly
            # (If this points to production gold.db, abort immediately)
            env_check = os.environ.copy()
            env_check["DUCKDB_PATH"] = str(db_path)
            # Ensure pipeline/trading_app are importable in subprocesses
            env_check["PYTHONPATH"] = str(PROJECT_ROOT)
            check_result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pipeline.paths import GOLD_DB_PATH; print(GOLD_DB_PATH)",
                ],
                env=env_check,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            resolved_path = check_result.stdout.strip()
            if resolved_path and Path(resolved_path) != db_path:
                print(f"  ABORT: GOLD_DB_PATH resolved to {resolved_path}, not {db_path}")
                print("         This would write to the wrong database!")
                return 1
            print(f"  GOLD_DB_PATH resolves to: {resolved_path}")

            # Step 1: Initialize schema
            ok = run_step("init_db", [sys.executable, "pipeline/init_db.py"], env_check)
            if not ok:
                all_results.append({"seed": seed, "status": "PIPELINE_ERROR", "step": "init_db"})
                any_failure = True
                continue

            ok = run_step(
                "init_trading_schema",
                [
                    sys.executable,
                    "-c",
                    "from trading_app.db_manager import init_trading_app_schema; init_trading_app_schema()",
                ],
                env_check,
            )
            if not ok:
                all_results.append({"seed": seed, "status": "PIPELINE_ERROR", "step": "init_trading_schema"})
                any_failure = True
                continue

            # Step 2: Generate and insert synthetic bars
            print(f"\n--- Generate synthetic bars (seed={seed}, sigma={args.sigma}) ---")
            t0 = time.time()
            # Per-instrument calibration for realistic price/tick scale
            _inst_defaults = {
                "MGC": {"start_price": 2000.0, "tick_size": 0.10},
                "MNQ": {"start_price": 20000.0, "tick_size": 0.25},
                "MES": {"start_price": 5000.0, "tick_size": 0.25},
            }
            _id = _inst_defaults.get(args.instrument, _inst_defaults["MGC"])
            bar_stats = generate_synthetic_bars(
                db_path,
                seed=seed,
                symbol=args.instrument,
                start_date=start_dt,
                end_date=end_dt,
                sigma=args.sigma,
                start_price=_id["start_price"],
                tick_size=_id["tick_size"],
            )
            gen_time = time.time() - t0
            print(f"  Generated {bar_stats['row_count']:,} bars in {gen_time:.1f}s")
            print(f"  Price: {bar_stats['price_min']:.1f} - {bar_stats['price_max']:.1f}")
            print(f"  Mean return: {bar_stats['mean_return']:.8f}")
            print(f"  Autocorr(1): {bar_stats['autocorr_lag1']:.6f}")

            # Step 3: Run the rest of the pipeline
            start_str = start_dt.isoformat()
            end_str = end_dt.isoformat()
            python = sys.executable

            # Build 5m bars
            ok = run_step(
                "build_bars_5m",
                [python, "pipeline/build_bars_5m.py", "--instrument", args.instrument, "--start", start_str, "--end", end_str],
                env_check,
            )
            if not ok:
                all_results.append({"seed": seed, "status": "PIPELINE_ERROR", "step": "build_bars_5m"})
                any_failure = True
                continue

            # Per-aperture steps
            pipeline_ok = True
            for orb_min in args.apertures:
                tag = f"O{orb_min}"

                ok = run_step(
                    f"build_daily_features ({tag})",
                    [
                        python,
                        "pipeline/build_daily_features.py",
                        "--instrument",
                        args.instrument,
                        "--start",
                        start_str,
                        "--end",
                        end_str,
                        "--orb-minutes",
                        str(orb_min),
                    ],
                    env_check,
                )
                if not ok:
                    pipeline_ok = False
                    break

                ok = run_step(
                    f"outcome_builder ({tag})",
                    [
                        python,
                        "trading_app/outcome_builder.py",
                        "--instrument",
                        args.instrument,
                        "--start",
                        start_str,
                        "--end",
                        end_str,
                        "--orb-minutes",
                        str(orb_min),
                        "--force",
                    ],
                    env_check,
                )
                if not ok:
                    pipeline_ok = False
                    break

                ok = run_step(
                    f"strategy_discovery ({tag})",
                    [
                        python,
                        "trading_app/strategy_discovery.py",
                        "--instrument",
                        args.instrument,
                        "--orb-minutes",
                        str(orb_min),
                    ],
                    env_check,
                )
                if not ok:
                    pipeline_ok = False
                    break

            if not pipeline_ok:
                all_results.append({"seed": seed, "status": "PIPELINE_ERROR", "step": "per_aperture"})
                any_failure = True
                continue

            # Strategy validation
            ok = run_step(
                "strategy_validator",
                [python, "trading_app/strategy_validator.py", "--instrument", args.instrument],
                env_check,
            )
            if not ok:
                all_results.append({"seed": seed, "status": "PIPELINE_ERROR", "step": "validator"})
                any_failure = True
                continue

            # Step 4: Query and report results
            results = query_results(db_path)
            results["seed"] = seed
            results["bar_stats"] = bar_stats

            print_report(seed, bar_stats, results)

            # Determine pass/fail — ANY row in validated_setups from noise = FAIL
            validated = results.get("validated_count", -1)
            if validated == 0:
                results["status"] = "PASS"
            elif validated > 0:
                results["status"] = "FAIL"
                any_failure = True
            else:
                results["status"] = "ERROR"
                any_failure = True

            all_results.append(results)

            seed_time = time.time() - t_seed_start
            print(f"\n  Seed {seed} total time: {seed_time:.0f}s ({seed_time/60:.1f}min)")

        except Exception as exc:
            print(f"\n  EXCEPTION during seed {seed}: {exc}")
            import traceback

            traceback.print_exc()
            all_results.append({"seed": seed, "status": "EXCEPTION", "error": str(exc)})
            any_failure = True

        finally:
            if args.keep_db:
                print(f"\n  DB retained at: {db_path}")
            else:
                shutil.rmtree(tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for r in all_results:
        status = r.get("status", "UNKNOWN")
        seed = r.get("seed", "?")
        validated = r.get("validated_count", "?")
        positive = r.get("exp_positive_expr", "?")
        pass_p1 = r.get("exp_pass_phase1", "?")
        step = r.get("step", "")
        error = r.get("error", "")

        detail = ""
        if status == "PIPELINE_ERROR":
            detail = f" (failed at: {step})"
        elif status == "EXCEPTION":
            detail = f" ({error})"
        elif status in ("PASS", "FAIL"):
            detail = f" (validated={validated}, noise_positive_expr={positive}, pass_phase1={pass_p1})"

        print(f"  Seed {seed}: {status}{detail}")

    print()
    if any_failure:
        failures = sum(1 for r in all_results if r.get("status") != "PASS")
        print(f"OVERALL: FAIL ({failures}/{len(all_results)} seeds failed)")
        print("The pipeline may be manufacturing false edges from noise.")
        return 1
    else:
        print(f"OVERALL: PASS ({len(all_results)}/{len(all_results)} seeds, 0 false validations)")
        print("The pipeline correctly rejects zero-signal data at all gates.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
