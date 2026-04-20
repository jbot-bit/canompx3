#!/usr/bin/env python3
"""MNQ E2 slippage validation pilot.

The MNQ baseline edge is ~+0.044R = ~$2.28/trade. At $0.50/tick (MNQ tick),
1 extra tick of slippage costs $1 and kills ~44% of the edge. We need the
ACTUAL number.

MGC pilot showed median=0 ticks slippage (mean 6.75 dominated by 2018-01-18
gap-open event). MNQ is a different market (tech equities vs gold,
different book depth, different participant mix) — needs its own measurement.

Two operating modes:
1. Databento pull + reprice (COSTS MONEY) — requires DATABENTO_API_KEY,
   pulls new TBBO windows, caches under `research/data/tbbo_mnq_pilot/`.
2. --reprice-cache (FREE) — reuses the existing 119-file cache. Reverse-
   engineers a manifest from cached filenames by joining to daily_features
   for orb_high/orb_low/break_dir. This is the default for honest analysis
   of measurements we already paid for.

Canonical delegation (institutional-rigor Rule 4):
- `reprice_e2_entry` from `research.databento_microstructure` — do NOT
  re-encode the first-cross / BBO logic.
- `_orb_utc_window` from `pipeline.build_daily_features` — do NOT re-encode
  ORB window timing.
- Cost specs from `pipeline.cost_model.get_cost_spec("MNQ")`.

Usage:
    python -m research.research_mnq_e2_slippage_pilot --estimate-cost
    python -m research.research_mnq_e2_slippage_pilot --pull
    python -m research.research_mnq_e2_slippage_pilot --reprice
    python -m research.research_mnq_e2_slippage_pilot --reprice-cache
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.build_daily_features import _orb_utc_window
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH, PROJECT_ROOT
from research.databento_microstructure import load_tbbo_df, reprice_e2_entry

DATASET = "GLBX.MDP3"
INSTRUMENT = "MNQ"
DATABENTO_SYMBOL = "MNQ.FUT"
NQ_SYMBOL = "NQ.FUT"  # Full-size for comparison (deeper book)
ORB_MINUTES = 5
SAMPLE_PER_BUCKET = 5  # 5 per (session × atr_regime) bucket
DEFAULT_SEED = 42
WINDOW_MINUTES_BEFORE = 2
WINDOW_MINUTES_AFTER = 30
CACHE_DIR = PROJECT_ROOT / "research" / "data" / "tbbo_mnq_pilot"

# Sessions to sample from (chronological, covers Asian + US)
PILOT_SESSIONS = [
    "TOKYO_OPEN",  # Asian — thinner book
    "SINGAPORE_OPEN",  # Asian — thinner book
    "LONDON_METALS",  # European — medium book
    "NYSE_OPEN",  # US — deepest book
    "US_DATA_830",  # US — deep book
    "CME_PRECLOSE",  # US — medium book
]

CACHE_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_([A-Z_0-9]+)_MNQ\.dbn\.zst$")


@dataclass(frozen=True)
class PilotDay:
    trading_day: str
    orb_label: str
    break_dir: str
    orb_level: float  # The ORB boundary that was crossed
    atr_20: float
    atr_regime: str  # "high" or "low"
    window_start_utc: str
    window_end_utc: str


def parse_cache_filename(filename: str) -> tuple[str, str] | None:
    """Parse `YYYY-MM-DD_SESSION_MNQ.dbn.zst` → (day, session) or None."""
    m = CACHE_FILENAME_RE.match(filename)
    if not m:
        return None
    return m.group(1), m.group(2)


def build_manifest_from_cache(cache_dir: Path) -> list[dict]:
    """Reverse-engineer a pilot manifest from cached filenames.

    For each `YYYY-MM-DD_SESSION_MNQ.dbn.zst`, joins to `daily_features` for
    canonical `orb_{session}_high/low/break_dir` + `atr_20`. Days with no
    daily_features row are marked `error='daily_features missing'`, not
    silently dropped.

    Zero Databento spend — uses only already-cached data.
    """
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        rows: list[dict] = []
        for cache_path in sorted(cache_dir.glob("*_MNQ.dbn.zst")):
            parsed = parse_cache_filename(cache_path.name)
            if parsed is None:
                rows.append({
                    "trading_day": None,
                    "orb_label": None,
                    "cache_path": str(cache_path),
                    "orb_high": None,
                    "orb_low": None,
                    "break_dir": None,
                    "atr_20": None,
                    "error": f"filename_regex_failed: {cache_path.name}",
                })
                continue

            day, session = parsed
            high_col = f"orb_{session}_high"
            low_col = f"orb_{session}_low"
            dir_col = f"orb_{session}_break_dir"

            try:
                df_row = con.execute(
                    f"""
                    SELECT {high_col}, {low_col}, {dir_col}, atr_20
                    FROM daily_features
                    WHERE symbol = 'MNQ'
                      AND orb_minutes = 5
                      AND trading_day = CAST(? AS DATE)
                    """,
                    [day],
                ).fetchone()
            except duckdb.Error as exc:
                rows.append({
                    "trading_day": day,
                    "orb_label": session,
                    "cache_path": str(cache_path),
                    "orb_high": None,
                    "orb_low": None,
                    "break_dir": None,
                    "atr_20": None,
                    "error": f"duckdb_query_failed: {exc}",
                })
                continue

            if df_row is None or df_row[0] is None or df_row[1] is None or df_row[2] is None:
                rows.append({
                    "trading_day": day,
                    "orb_label": session,
                    "cache_path": str(cache_path),
                    "orb_high": None,
                    "orb_low": None,
                    "break_dir": None,
                    "atr_20": None,
                    "error": "daily_features missing or incomplete",
                })
                continue

            orb_high, orb_low, break_dir, atr_20 = df_row
            rows.append({
                "trading_day": day,
                "orb_label": session,
                "cache_path": str(cache_path),
                "orb_high": float(orb_high),
                "orb_low": float(orb_low),
                "break_dir": str(break_dir),
                "atr_20": float(atr_20) if atr_20 is not None else None,
                "error": None,
            })
        return rows
    finally:
        con.close()


def reprice_cache_manifest(manifest: list[dict]) -> list[dict]:
    """For each manifest row, load cached tbbo and call canonical reprice_e2_entry.

    Delegates to `research.databento_microstructure.reprice_e2_entry`; does
    NOT re-encode first-cross / BBO logic.

    orb_end_utc is computed via canonical `_orb_utc_window` (no re-encoding).
    model_entry_ts_utc is passed as orb_end_utc (it's metadata-only inside
    reprice_e2_entry; only orb_end_utc gates the trade scan).
    """
    spec = get_cost_spec(INSTRUMENT)
    results: list[dict] = []

    for row in manifest:
        if row.get("error") is not None:
            results.append({
                "trading_day": row.get("trading_day"),
                "orb_label": row.get("orb_label"),
                "error": row["error"],
            })
            continue

        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            results.append({
                "trading_day": row["trading_day"],
                "orb_label": row["orb_label"],
                "error": "cache_file_missing",
            })
            continue

        try:
            tbbo_df = load_tbbo_df(cache_path)
        except Exception as exc:
            results.append({
                "trading_day": row["trading_day"],
                "orb_label": row["orb_label"],
                "error": f"load_tbbo_failed: {exc}",
            })
            continue

        if tbbo_df.empty:
            results.append({
                "trading_day": row["trading_day"],
                "orb_label": row["orb_label"],
                "error": "empty_tbbo_after_front_month_filter",
            })
            continue

        try:
            _, orb_end_dt = _orb_utc_window(
                date.fromisoformat(row["trading_day"]), row["orb_label"], ORB_MINUTES
            )
        except Exception as exc:
            results.append({
                "trading_day": row["trading_day"],
                "orb_label": row["orb_label"],
                "error": f"orb_utc_window_failed: {exc}",
            })
            continue

        orb_end_utc = orb_end_dt.isoformat()
        orb_high = float(row["orb_high"])
        orb_low = float(row["orb_low"])
        break_dir = str(row["break_dir"])
        orb_level = orb_high if break_dir == "long" else orb_low
        modeled_slippage_ticks = int(spec.slippage / spec.point_value / spec.tick_size)

        entry = reprice_e2_entry(
            tbbo_df=tbbo_df,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir=break_dir,
            model_entry_price=orb_level + (spec.tick_size if break_dir == "long" else -spec.tick_size),
            model_entry_ts_utc=orb_end_utc,  # metadata-only in reprice_e2_entry
            trading_day=row["trading_day"],
            symbol_pulled=DATABENTO_SYMBOL,
            tick_size=spec.tick_size,
            modeled_slippage_ticks=modeled_slippage_ticks,
            orb_end_utc=orb_end_utc,
        )

        results.append({
            "trading_day": entry.trading_day,
            "orb_label": row["orb_label"],
            "break_dir": entry.break_dir,
            "atr_20": row.get("atr_20"),
            "orb_high": orb_high,
            "orb_low": orb_low,
            "orb_level": entry.orb_level,
            "trigger_price": entry.trigger_trade_price,
            "bid_at_trigger": entry.bbo_at_trigger_bid,
            "ask_at_trigger": entry.bbo_at_trigger_ask,
            "spread_ticks": entry.bbo_at_trigger_spread,
            "estimated_fill": entry.estimated_fill_price,
            "slippage_pts": entry.actual_slippage_points,
            "slippage_ticks": entry.actual_slippage_ticks,
            "n_tbbo_records": entry.tbbo_records_in_window,
            "error": entry.error,
        })

    return results


def build_pilot_manifest(seed: int = DEFAULT_SEED) -> list[PilotDay]:
    """Select stratified sample of MNQ E2 touch days (used for fresh Databento pulls)."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    all_days = []
    for session in PILOT_SESSIONS:
        break_dir_col = f"orb_{session}_break_dir"
        high_col = f"orb_{session}_high"
        low_col = f"orb_{session}_low"

        session_rows = con.execute(f"""
            SELECT
                o.trading_day,
                '{session}' as orb_label,
                df.{break_dir_col} as break_dir,
                CASE WHEN df.{break_dir_col} = 'long'
                     THEN df.{high_col} ELSE df.{low_col} END as orb_level,
                df.atr_20
            FROM orb_outcomes o
            JOIN daily_features df
                ON o.trading_day = df.trading_day
                AND df.symbol = 'MNQ' AND df.orb_minutes = 5
            WHERE o.symbol = 'MNQ'
                AND o.orb_label = '{session}'
                AND o.entry_model = 'E2'
                AND o.rr_target = 2.0
                AND o.confirm_bars = 1
                AND o.orb_minutes = 5
                AND o.outcome IS NOT NULL
                AND df.atr_20 IS NOT NULL
                AND df.{break_dir_col} IS NOT NULL
        """).fetchall()

        for r in session_rows:
            all_days.append(
                {
                    "trading_day": str(r[0]),
                    "orb_label": r[1],
                    "break_dir": r[2],
                    "orb_level": float(r[3]) if r[3] else None,
                    "atr_20": float(r[4]),
                }
            )

    con.close()

    if not all_days:
        print("No MNQ E2 touch days found!")
        return []

    df = pd.DataFrame(all_days)
    df = df.dropna(subset=["orb_level"])

    # Stratify: ATR median split × session
    atr_median = df["atr_20"].median()
    df["atr_regime"] = np.where(df["atr_20"] >= atr_median, "high", "low")

    # Sample from each bucket
    rng = np.random.default_rng(seed)
    sampled = []

    for session in PILOT_SESSIONS:
        for regime in ["high", "low"]:
            bucket = df[(df["orb_label"] == session) & (df["atr_regime"] == regime)]
            if len(bucket) == 0:
                continue
            n = min(SAMPLE_PER_BUCKET, len(bucket))
            sample = bucket.sample(n=n, random_state=rng.integers(0, 2**31))

            for _, row in sample.iterrows():
                td = date.fromisoformat(row["trading_day"])
                orb_start, orb_end = _orb_utc_window(td, session, ORB_MINUTES)
                window_start = orb_end - timedelta(minutes=WINDOW_MINUTES_BEFORE)
                window_end = orb_end + timedelta(minutes=WINDOW_MINUTES_AFTER)

                sampled.append(
                    PilotDay(
                        trading_day=row["trading_day"],
                        orb_label=session,
                        break_dir=row["break_dir"],
                        orb_level=row["orb_level"],
                        atr_20=row["atr_20"],
                        atr_regime=regime,
                        window_start_utc=window_start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        window_end_utc=window_end.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    )
                )

    print(f"Pilot manifest: {len(sampled)} days")
    print(f"  Sessions: {set(d.orb_label for d in sampled)}")
    print(
        f"  ATR regimes: high={sum(1 for d in sampled if d.atr_regime == 'high')}, "
        f"low={sum(1 for d in sampled if d.atr_regime == 'low')}"
    )
    print(
        f"  Break dirs: long={sum(1 for d in sampled if d.break_dir == 'long')}, "
        f"short={sum(1 for d in sampled if d.break_dir == 'short')}"
    )

    return sampled


def estimate_cost(manifest: list[PilotDay]) -> float:
    """Estimate Databento cost for the pilot."""
    import databento as db

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("DATABENTO_API_KEY")

    client = db.Historical(api_key)
    total = 0.0

    for day in manifest[:3]:
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET,
                symbols=[DATABENTO_SYMBOL],
                schema="tbbo",
                stype_in="parent",
                start=day.window_start_utc,
                end=day.window_end_utc,
            )
            total += float(cost)
        except Exception as e:
            print(f"  Cost estimate failed for {day.trading_day}: {e}")

    per_day = total / 3 if total > 0 else 0
    estimated_total = per_day * len(manifest)
    print("\nCost estimate:")
    print(f"  Sample (3 days): ${total:.2f}")
    print(f"  Per day: ${per_day:.3f}")
    print(f"  Total ({len(manifest)} days): ${estimated_total:.2f}")
    return estimated_total


def pull_tbbo(manifest: list[PilotDay], force: bool = False) -> list[Path]:
    """Pull tbbo data for all pilot days."""
    import databento as db

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("DATABENTO_API_KEY")

    client = db.Historical(api_key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    errors = []

    for i, day in enumerate(manifest):
        cache_path = CACHE_DIR / f"{day.trading_day}_{day.orb_label}_MNQ.dbn.zst"

        if cache_path.exists() and not force:
            print(f"  [{i + 1}/{len(manifest)}] {day.trading_day} {day.orb_label} ... cached")
            paths.append(cache_path)
            continue

        print(f"  [{i + 1}/{len(manifest)}] {day.trading_day} {day.orb_label} ... pulling")
        try:
            client.timeseries.get_range(
                dataset=DATASET,
                symbols=[DATABENTO_SYMBOL],
                schema="tbbo",
                stype_in="parent",
                start=day.window_start_utc,
                end=day.window_end_utc,
                path=str(cache_path),
            )
            paths.append(cache_path)
        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append((day.trading_day, day.orb_label, str(e)))

    if errors:
        print(f"\n{len(errors)} pull failures")
        for td, ol, err in errors:
            print(f"  {td} {ol}: {err}")

    return paths


def reprice_entries(manifest: list[PilotDay]) -> pd.DataFrame:
    """Reprice a fresh-pull manifest against the canonical reprice_e2_entry.

    For the --reprice mode. For the no-Databento-spend path, use
    reprice_cache_manifest + build_manifest_from_cache instead.
    """
    spec = get_cost_spec(INSTRUMENT)
    results = []

    for day in manifest:
        cache_path = CACHE_DIR / f"{day.trading_day}_{day.orb_label}_MNQ.dbn.zst"
        if not cache_path.exists():
            results.append({"trading_day": day.trading_day, "orb_label": day.orb_label, "error": "no cache file"})
            continue

        tbbo_df = load_tbbo_df(cache_path)
        if tbbo_df.empty:
            results.append(
                {"trading_day": day.trading_day, "orb_label": day.orb_label, "error": "empty after filtering"}
            )
            continue

        td = date.fromisoformat(day.trading_day)
        _, orb_end = _orb_utc_window(td, day.orb_label, ORB_MINUTES)

        # Fetch real orb_high and orb_low from daily_features (no dummies)
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        try:
            hc = f"orb_{day.orb_label}_high"
            lc = f"orb_{day.orb_label}_low"
            df_row = con.execute(
                f"""
                SELECT {hc}, {lc} FROM daily_features
                WHERE symbol='MNQ' AND orb_minutes=5 AND trading_day = CAST(? AS DATE)
                """,
                [day.trading_day],
            ).fetchone()
        finally:
            con.close()

        if df_row is None or df_row[0] is None or df_row[1] is None:
            results.append({"trading_day": day.trading_day, "orb_label": day.orb_label, "error": "daily_features missing"})
            continue

        orb_high, orb_low = float(df_row[0]), float(df_row[1])
        modeled_slippage_ticks = int(spec.slippage / spec.point_value / spec.tick_size)

        entry = reprice_e2_entry(
            tbbo_df=tbbo_df,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir=day.break_dir,
            model_entry_price=day.orb_level + (spec.tick_size if day.break_dir == "long" else -spec.tick_size),
            model_entry_ts_utc=orb_end.isoformat(),
            trading_day=day.trading_day,
            symbol_pulled=DATABENTO_SYMBOL,
            tick_size=spec.tick_size,
            modeled_slippage_ticks=modeled_slippage_ticks,
            orb_end_utc=orb_end.isoformat(),
        )

        row = {
            "trading_day": entry.trading_day,
            "orb_label": day.orb_label,
            "break_dir": entry.break_dir,
            "atr_regime": day.atr_regime,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "orb_level": entry.orb_level,
            "trigger_price": entry.trigger_trade_price,
            "bid_at_trigger": entry.bbo_at_trigger_bid,
            "ask_at_trigger": entry.bbo_at_trigger_ask,
            "spread_ticks": entry.bbo_at_trigger_spread,
            "estimated_fill": entry.estimated_fill_price,
            "slippage_pts": entry.actual_slippage_points,
            "slippage_ticks": entry.actual_slippage_ticks,
            "n_tbbo_records": entry.tbbo_records_in_window,
            "error": entry.error,
        }
        results.append(row)

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> None:
    """Print analysis of slippage results. Reports median + mean + MAD (the
    mean alone is misleading for skewed distributions — MGC pilot mean=6.75
    was 1-outlier-driven, median=0)."""
    valid = results_df[results_df["error"].isna()].copy()
    errors = results_df[results_df["error"].notna()]

    print(f"\n{'=' * 60}")
    print("MNQ E2 SLIPPAGE PILOT RESULTS")
    print(f"{'=' * 60}")
    print(f"  Valid samples: {len(valid)}")
    print(f"  Errors: {len(errors)}")

    if valid.empty:
        print("  No valid results!")
        return

    slip = valid["slippage_ticks"].astype(float).to_numpy()
    median = float(np.median(slip))
    mean = float(np.mean(slip))
    mad = float(np.median(np.abs(slip - median)))
    print("\n  Slippage (ticks):")
    print(f"    Median: {median:.2f}     (ROBUST central tendency)")
    print(f"    MAD:    {mad:.2f}     (robust dispersion)")
    print(f"    Mean:   {mean:.2f}     (outlier-sensitive — inspect if > median)")
    print(f"    Std:    {np.std(slip):.2f}")
    print(f"    Min:    {np.min(slip):.2f}")
    print(f"    p25:    {np.quantile(slip, 0.25):.2f}")
    print(f"    p75:    {np.quantile(slip, 0.75):.2f}")
    print(f"    p95:    {np.quantile(slip, 0.95):.2f}")
    print(f"    Max:    {np.max(slip):.2f}")
    print(f"    % <= 1 tick: {(slip <= 1).mean() * 100:.1f}%")
    print(f"    % <= 2 ticks: {(slip <= 2).mean() * 100:.1f}%")

    spec = get_cost_spec(INSTRUMENT)
    modeled = spec.slippage / spec.point_value / spec.tick_size
    print(f"\n  Modeled slippage: {modeled:.0f} ticks")
    print(f"  Actual median:   {median:.1f} ticks")
    print(f"  Verdict: {'CONSERVATIVE (safe)' if median <= modeled else 'OPTIMISTIC (danger)'}")

    # Per-session breakdown
    print("\n  Per session:")
    for session in sorted(valid["orb_label"].unique()):
        s = valid[valid["orb_label"] == session]["slippage_ticks"].astype(float)
        print(f"    {session:20s}: median={np.median(s):.1f}, mean={np.mean(s):.1f}, N={len(s)}")

    # Per direction (MGC showed long=11, short=0.25 asymmetry)
    print("\n  Per break_dir:")
    for direction in ["long", "short"]:
        d = valid[valid["break_dir"] == direction]["slippage_ticks"].astype(float)
        if len(d) > 0:
            print(f"    {direction:5s}: median={np.median(d):.1f}, mean={np.mean(d):.1f}, N={len(d)}")

    # Outlier flag (per-day inspection needed if > p95)
    p95 = np.quantile(slip, 0.95)
    outliers = valid[valid["slippage_ticks"].astype(float) > max(p95, 10.0)]
    if len(outliers) > 0:
        print(f"\n  OUTLIERS (>p95 or >10 ticks, require per-day investigation):")
        for _, row in outliers.iterrows():
            print(
                f"    {row['trading_day']} {row['orb_label']} {row['break_dir']}: "
                f"slippage={row['slippage_ticks']:.0f} ticks  "
                f"orb_level={row['orb_level']} trigger={row['trigger_price']}"
            )

    # Spread
    if "spread_ticks" in valid.columns:
        spreads = valid["spread_ticks"].astype(float).dropna()
        if len(spreads) > 0:
            print("\n  Spread at trigger (ticks):")
            print(f"    Mean:   {spreads.mean():.2f}")
            print(f"    Median: {spreads.median():.2f}")
            print(f"    Max:    {spreads.max():.2f}")


def main():
    parser = argparse.ArgumentParser(description="MNQ E2 slippage validation pilot")
    parser.add_argument("--estimate-cost", action="store_true")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--reprice", action="store_true")
    parser.add_argument(
        "--reprice-cache",
        action="store_true",
        help="Reprice all cached days via reverse-engineered manifest (no Databento spend).",
    )
    parser.add_argument("--all", action="store_true", help="Full fresh-pull pipeline: manifest + cost + pull + reprice")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.reprice_cache:
        manifest_cache = build_manifest_from_cache(CACHE_DIR)
        print(f"Cache manifest: {len(manifest_cache)} entries, {sum(1 for r in manifest_cache if r['error'] is None)} valid")
        results = reprice_cache_manifest(manifest_cache)
        results_df = pd.DataFrame(results)
        out_path = CACHE_DIR / "slippage_results_cache_v2.csv"
        results_df.to_csv(out_path, index=False)
        print(f"Results saved: {out_path}")
        analyze_results(results_df)
        return

    # Fresh-pull path (costs Databento $$)
    manifest = build_pilot_manifest(seed=args.seed)
    if not manifest:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps([asdict(d) for d in manifest], indent=2))
    print(f"Manifest saved: {manifest_path}")

    if args.estimate_cost or args.all:
        estimate_cost(manifest)

    if args.pull or args.all:
        pull_tbbo(manifest, force=args.force)

    if args.reprice or args.all:
        results = reprice_entries(manifest)
        results_path = CACHE_DIR / "slippage_results.csv"
        results.to_csv(results_path, index=False)
        print(f"Results saved: {results_path}")
        analyze_results(results)


if __name__ == "__main__":
    main()
