#!/usr/bin/env python3
"""
MNQ E2 slippage validation pilot — the Achilles heel test.

The MNQ baseline edge is +0.044R = ~$2.28/trade. At $0.50/tick (MNQ tick),
1 extra tick of slippage costs $1 and kills 44% of the edge. The edge dies
at 5 ticks/side. We need the ACTUAL number.

MGC pilot showed median=0 ticks slippage. MNQ is a different market
(tech equities vs gold, different book depth, different participant mix).

Approach:
1. Sample 40 E2 touch days from MNQ across multiple sessions
2. Stratify by ATR regime (high/low) and session time (Asian/US)
3. Pull Databento tbbo around each break moment
4. Measure: what's the BBO when price first crosses the ORB boundary?
5. Compare to modeled 1-tick slippage assumption

Usage:
    python -m research.research_mnq_e2_slippage_pilot --estimate-cost
    python -m research.research_mnq_e2_slippage_pilot --pull
    python -m research.research_mnq_e2_slippage_pilot --reprice
"""

from __future__ import annotations

import argparse
import json
import os
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
    "TOKYO_OPEN",       # Asian — thinner book
    "SINGAPORE_OPEN",   # Asian — thinner book
    "LONDON_METALS",    # European — medium book
    "NYSE_OPEN",        # US — deepest book
    "US_DATA_830",      # US — deep book
    "CME_PRECLOSE",     # US — medium book
]


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


def build_pilot_manifest(seed: int = DEFAULT_SEED) -> list[PilotDay]:
    """Select stratified sample of MNQ E2 touch days."""
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
            all_days.append({
                "trading_day": str(r[0]),
                "orb_label": r[1],
                "break_dir": r[2],
                "orb_level": float(r[3]) if r[3] else None,
                "atr_20": float(r[4]),
            })

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
                # Compute UTC window around the break
                td = date.fromisoformat(row["trading_day"])
                orb_start, orb_end = _orb_utc_window(td, session, ORB_MINUTES)
                window_start = orb_end - timedelta(minutes=WINDOW_MINUTES_BEFORE)
                window_end = orb_end + timedelta(minutes=WINDOW_MINUTES_AFTER)

                sampled.append(PilotDay(
                    trading_day=row["trading_day"],
                    orb_label=session,
                    break_dir=row["break_dir"],
                    orb_level=row["orb_level"],
                    atr_20=row["atr_20"],
                    atr_regime=regime,
                    window_start_utc=window_start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    window_end_utc=window_end.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                ))

    print(f"Pilot manifest: {len(sampled)} days")
    print(f"  Sessions: {set(d.orb_label for d in sampled)}")
    print(f"  ATR regimes: high={sum(1 for d in sampled if d.atr_regime=='high')}, "
          f"low={sum(1 for d in sampled if d.atr_regime=='low')}")
    print(f"  Break dirs: long={sum(1 for d in sampled if d.break_dir=='long')}, "
          f"short={sum(1 for d in sampled if d.break_dir=='short')}")

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

    for day in manifest[:3]:  # Sample 3 days for cost estimate, extrapolate
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
    print(f"\nCost estimate:")
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
            print(f"  [{i+1}/{len(manifest)}] {day.trading_day} {day.orb_label} ... cached")
            paths.append(cache_path)
            continue

        print(f"  [{i+1}/{len(manifest)}] {day.trading_day} {day.orb_label} ... pulling")
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
    """Measure actual slippage using the PROVEN MGC pilot functions.

    Delegates to databento_microstructure.load_tbbo_df (front-month filtering)
    and databento_microstructure.reprice_entry (crossing detection + BBO).
    """
    from research.databento_microstructure import load_tbbo_df, reprice_entry

    spec = get_cost_spec(INSTRUMENT)
    results = []

    for i, day in enumerate(manifest):
        cache_path = CACHE_DIR / f"{day.trading_day}_{day.orb_label}_MNQ.dbn.zst"
        if not cache_path.exists():
            results.append({"trading_day": day.trading_day, "orb_label": day.orb_label,
                            "error": "no cache file"})
            continue

        tbbo_df = load_tbbo_df(cache_path)
        if tbbo_df.empty:
            results.append({"trading_day": day.trading_day, "orb_label": day.orb_label,
                            "error": "empty after filtering"})
            continue

        # Compute ORB end time for post-ORB filtering
        td = date.fromisoformat(day.trading_day)
        _, orb_end = _orb_utc_window(td, day.orb_label, ORB_MINUTES)

        # Use the PROVEN reprice function from the MGC pilot
        # It needs orb_high/orb_low — derive from break_dir + orb_level
        if day.break_dir == "long":
            orb_high = day.orb_level
            orb_low = day.orb_level - 1.0  # dummy — only orb_high matters for long
        else:
            orb_low = day.orb_level
            orb_high = day.orb_level + 1.0  # dummy — only orb_low matters for short

        entry = reprice_entry(
            tbbo_df=tbbo_df,
            trading_day=day.trading_day,
            break_dir=day.break_dir,
            orb_high=orb_high,
            orb_low=orb_low,
            model_entry_price=day.orb_level + (spec.tick_size if day.break_dir == "long" else -spec.tick_size),
            modeled_slippage_ticks=int(spec.slippage / spec.point_value / spec.tick_size),
            tick_size=spec.tick_size,
            symbol_pulled="MNQ.FUT",
            orb_end_utc=orb_end.isoformat(),
        )

        row = {
            "trading_day": entry.trading_day,
            "orb_label": day.orb_label,
            "break_dir": entry.break_dir,
            "atr_regime": day.atr_regime,
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
    """Print analysis of slippage results."""
    valid = results_df[results_df["error"].isna()].copy()
    errors = results_df[results_df["error"].notna()]

    print(f"\n{'='*60}")
    print(f"MNQ E2 SLIPPAGE PILOT RESULTS")
    print(f"{'='*60}")
    print(f"  Valid samples: {len(valid)}")
    print(f"  Errors: {len(errors)}")

    if valid.empty:
        print("  No valid results!")
        return

    slip = valid["slippage_ticks"].values
    print(f"\n  Slippage (ticks):")
    print(f"    Mean:   {np.mean(slip):.2f}")
    print(f"    Median: {np.median(slip):.2f}")
    print(f"    Std:    {np.std(slip):.2f}")
    print(f"    Min:    {np.min(slip):.2f}")
    print(f"    Max:    {np.max(slip):.2f}")
    print(f"    % <= 1 tick: {(slip <= 1).mean()*100:.1f}%")
    print(f"    % <= 2 ticks: {(slip <= 2).mean()*100:.1f}%")

    spec = get_cost_spec(INSTRUMENT)
    modeled = spec.slippage / spec.point_value / spec.tick_size
    print(f"\n  Modeled slippage: {modeled:.0f} ticks")
    print(f"  Actual median:   {np.median(slip):.1f} ticks")
    print(f"  Verdict: {'CONSERVATIVE (safe)' if np.median(slip) <= modeled else 'OPTIMISTIC (danger)'}")

    # Per-session breakdown
    print(f"\n  Per session:")
    for session in sorted(valid["orb_label"].unique()):
        s = valid[valid["orb_label"] == session]["slippage_ticks"]
        print(f"    {session:20s}: median={np.median(s):.1f}, mean={np.mean(s):.1f}, N={len(s)}")

    # Per ATR regime
    print(f"\n  Per ATR regime:")
    for regime in ["low", "high"]:
        r = valid[valid["atr_regime"] == regime]["slippage_ticks"]
        if len(r) > 0:
            print(f"    {regime:5s}: median={np.median(r):.1f}, mean={np.mean(r):.1f}, N={len(r)}")

    # Spread at trigger
    if "spread_ticks" in valid.columns:
        spreads = valid["spread_ticks"].dropna()
        if len(spreads) > 0:
            print(f"\n  Spread at trigger (ticks):")
            print(f"    Mean:   {np.mean(spreads):.2f}")
            print(f"    Median: {np.median(spreads):.2f}")
            print(f"    Max:    {np.max(spreads):.2f}")

    # Impact on edge
    print(f"\n  EDGE IMPACT:")
    baseline = 0.044  # MNQ baseline ExpR
    avg_risk = 51.87  # From fresh agent audit
    for ticks in [np.median(slip), np.mean(slip), 2.0, 3.0]:
        extra_cost = ticks * spec.tick_size * spec.point_value * 2  # entry + exit
        extra_cost_r = extra_cost / avg_risk
        net_expr = baseline - extra_cost_r + modeled * spec.tick_size * spec.point_value * 2 / avg_risk
        print(f"    At {ticks:.1f} ticks: extra cost = ${extra_cost:.2f} = {extra_cost_r:.3f}R, "
              f"net baseline = {baseline:.3f}R (already includes modeled)")


def main():
    parser = argparse.ArgumentParser(description="MNQ E2 slippage validation pilot")
    parser.add_argument("--estimate-cost", action="store_true")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--reprice", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: manifest + cost + pull + reprice")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = build_pilot_manifest(seed=args.seed)
    if not manifest:
        return

    # Save manifest
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
