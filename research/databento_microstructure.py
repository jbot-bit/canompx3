#!/usr/bin/env python3
"""
Databento tbbo data pull and E2 slippage repricing for microstructure pilot.

Pulls tick-level BBO data around ORB break moments and measures actual
stop-market fill quality vs the modeled 1-tick slippage assumption.

Key design decisions:
- Pulls BOTH MGC.FUT and GC.FUT for comparison (MGC is what we trade,
  GC has deeper book — comparing them reveals the bias)
- Caches .dbn.zst files locally under research/data/ to avoid re-downloading
- BBO at trade crossing = OPTIMISTIC lower bound for stop-market fill, not truth
- Each pilot day gets its own file to keep costs traceable

Usage:
    # Cost estimate only (no data download)
    python -m research.databento_microstructure --estimate-cost

    # Pull tbbo for all pilot days
    python -m research.databento_microstructure --pull

    # Reprice E2 entries from cached data
    python -m research.databento_microstructure --reprice
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH, PROJECT_ROOT
from trading_app.config import E2_SLIPPAGE_TICKS

DATASET = "GLBX.MDP3"
SYMBOLS = ["MGC.FUT", "GC.FUT"]
SCHEMA = "tbbo"
STYPE_IN = "parent"
CACHE_DIR = PROJECT_ROOT / "research" / "data" / "tbbo_pilot"


@dataclass(frozen=True)
class RepricedEntry:
    """Result of repricing one E2 entry against tick data."""

    trading_day: str
    break_dir: str
    symbol_pulled: str  # MGC.FUT or GC.FUT
    orb_level: float  # The ORB boundary that was touched
    modeled_entry_price: float  # From outcome_builder (ORB + 1 tick)
    modeled_slippage_ticks: int
    # Tick-level measurements
    trigger_trade_price: float | None  # First trade crossing ORB level
    trigger_trade_ts: str | None
    bbo_at_trigger_bid: float | None
    bbo_at_trigger_ask: float | None
    bbo_at_trigger_spread: float | None  # In ticks
    # What a stop-market would actually fill at (optimistic)
    estimated_fill_price: float | None  # ask (long) or bid (short)
    actual_slippage_points: float | None
    actual_slippage_ticks: float | None
    # Context
    tick_size: float
    tbbo_records_in_window: int
    error: str | None = None


def _cache_path(trading_day: str, symbol: str) -> Path:
    """Deterministic cache path for a (day, symbol) pull."""
    sym_clean = symbol.replace(".", "_")
    return CACHE_DIR / f"{trading_day}_{sym_clean}.dbn.zst"


def _get_client():
    """Create Databento Historical client."""
    try:
        import databento as db
    except ImportError as e:
        raise ImportError("databento package required: pip install databento") from e

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not found in environment or .env")
    return db.Historical(api_key)


def estimate_cost_per_day(
    manifest_df: pd.DataFrame,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Estimate Databento cost per pilot day per symbol.

    Uses the Python client (not raw requests) with per-day granularity
    instead of year-wide buckets to avoid timeouts.
    """
    client = _get_client()
    symbols = symbols or SYMBOLS
    rows: list[dict] = []

    for _, day_row in manifest_df.iterrows():
        for sym in symbols:
            try:
                cost = client.metadata.get_cost(
                    dataset=DATASET,
                    symbols=[sym],
                    schema=SCHEMA,
                    stype_in=STYPE_IN,
                    start=day_row["window_start_utc"],
                    end=day_row["window_end_utc"],
                )
                rows.append({
                    "trading_day": day_row["trading_day"],
                    "symbol": sym,
                    "window_start_utc": day_row["window_start_utc"],
                    "window_end_utc": day_row["window_end_utc"],
                    "estimated_cost_usd": float(cost),
                    "error": None,
                })
            except Exception as e:
                rows.append({
                    "trading_day": day_row["trading_day"],
                    "symbol": sym,
                    "window_start_utc": day_row["window_start_utc"],
                    "window_end_utc": day_row["window_end_utc"],
                    "estimated_cost_usd": None,
                    "error": str(e),
                })

    return pd.DataFrame(rows)


def pull_tbbo_for_day(
    trading_day: str,
    window_start_utc: str,
    window_end_utc: str,
    symbol: str = "MGC.FUT",
    *,
    force: bool = False,
) -> Path:
    """Download tbbo data for one pilot day window. Returns path to cached file.

    Skips download if cache file already exists (unless force=True).
    """
    cache_path = _cache_path(trading_day, symbol)
    if cache_path.exists() and not force:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    client = _get_client()

    client.timeseries.get_range(
        dataset=DATASET,
        symbols=[symbol],
        schema=SCHEMA,
        stype_in=STYPE_IN,
        start=window_start_utc,
        end=window_end_utc,
        path=str(cache_path),
    )
    return cache_path


def pull_all_pilot_days(
    manifest_df: pd.DataFrame,
    symbols: list[str] | None = None,
    *,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Pull tbbo for all pilot days and symbols. Returns {symbol: [paths]}.

    Raises RuntimeError if any pulls fail (fail-closed). Error details
    are printed before the exception.
    """
    symbols = symbols or SYMBOLS
    result: dict[str, list[Path]] = {sym: [] for sym in symbols}
    errors: list[tuple[str, str, str]] = []

    total = len(manifest_df) * len(symbols)
    done = 0

    for _, row in manifest_df.iterrows():
        for sym in symbols:
            done += 1
            day = str(row["trading_day"])
            cache = _cache_path(day, sym)
            status = "cached" if (cache.exists() and not force) else "pulling"
            print(f"  [{done}/{total}] {day} {sym} ... {status}")

            try:
                path = pull_tbbo_for_day(
                    day,
                    row["window_start_utc"],
                    row["window_end_utc"],
                    sym,
                    force=force,
                )
                result[sym].append(path)
            except Exception as e:
                print(f"    ERROR: {e}")
                errors.append((day, sym, str(e)))

    if errors:
        print(f"\n{len(errors)} pull failures:")
        for day, sym, err in errors:
            print(f"  {day} {sym}: {err}")
        raise RuntimeError(f"{len(errors)} tbbo pulls failed — see above for details")

    return result


def load_tbbo_df(cache_path: Path) -> pd.DataFrame:
    """Load a cached .dbn.zst file, filtering to front-month outright only.

    Parent symbol requests (e.g. MGC.FUT) return ALL instruments: front month,
    back months, calendar spreads (e.g. MGCJ5-MGCM5). Spread trades have
    negative prices and must be excluded. Back months trade at different
    prices. We keep only the most-traded outright symbol (= front contract).
    """
    import databento as db

    store = db.DBNStore.from_file(cache_path)
    df = store.to_df()

    if df.empty:
        return df

    # Drop calendar spreads (symbol contains '-')
    df = df[~df["symbol"].str.contains("-", na=False)]

    if df.empty:
        return df

    # Keep only the most-traded outright symbol (front month by volume)
    symbol_volume = df.groupby("symbol")["size"].sum()
    front_symbol = symbol_volume.idxmax()
    df = df[df["symbol"] == front_symbol]

    return df


def reprice_e2_entry(
    tbbo_df: pd.DataFrame,
    orb_high: float,
    orb_low: float,
    break_dir: str,
    model_entry_price: float,
    model_entry_ts_utc: str,
    trading_day: str,
    symbol_pulled: str,
    tick_size: float,
    modeled_slippage_ticks: int = E2_SLIPPAGE_TICKS,
    orb_end_utc: str | None = None,
) -> RepricedEntry:
    """Reprice one E2 entry against actual tick data.

    For a stop-market order:
    - Long: stop sits at orb_high. Triggers when trade >= orb_high.
      Fill = ask at that moment (optimistic lower bound).
    - Short: stop sits at orb_low. Triggers when trade <= orb_low.
      Fill = bid at that moment (optimistic lower bound).

    CRITICAL: Only trades AFTER orb_end_utc are considered. The ORB range
    is not established until the aperture window closes. Trades before that
    are pre-ORB and must not be matched as trigger events.

    The BBO at the trigger trade is the BEST CASE — a real stop-market
    competes with other triggered orders and HFT. Actual fills are likely worse.
    """
    orb_level = orb_high if break_dir == "long" else orb_low

    base = {
        "trading_day": trading_day,
        "break_dir": break_dir,
        "symbol_pulled": symbol_pulled,
        "orb_level": orb_level,
        "modeled_entry_price": model_entry_price,
        "modeled_slippage_ticks": modeled_slippage_ticks,
        "tick_size": tick_size,
        "tbbo_records_in_window": len(tbbo_df),
    }

    if tbbo_df.empty:
        return RepricedEntry(
            **base,
            trigger_trade_price=None,
            trigger_trade_ts=None,
            bbo_at_trigger_bid=None,
            bbo_at_trigger_ask=None,
            bbo_at_trigger_spread=None,
            estimated_fill_price=None,
            actual_slippage_points=None,
            actual_slippage_ticks=None,
            error="no_tbbo_records",
        )

    # Sort by event timestamp
    df = tbbo_df.sort_index() if tbbo_df.index.name == "ts_event" else tbbo_df.sort_values("ts_event")

    # Filter to only post-ORB trades (ORB range not set until aperture closes)
    if orb_end_utc is not None:
        orb_end_ts = pd.Timestamp(orb_end_utc)
        if df.index.name == "ts_event":
            df = df[df.index >= orb_end_ts]
        else:
            df = df[df["ts_event"] >= orb_end_ts]

    if df.empty:
        return RepricedEntry(
            **base,
            trigger_trade_price=None,
            trigger_trade_ts=None,
            bbo_at_trigger_bid=None,
            bbo_at_trigger_ask=None,
            bbo_at_trigger_spread=None,
            estimated_fill_price=None,
            actual_slippage_points=None,
            actual_slippage_ticks=None,
            error="no_post_orb_records",
        )

    # Find the first trade that crosses the ORB level
    if break_dir == "long":
        trigger_mask = df["price"] >= orb_level
    else:
        trigger_mask = df["price"] <= orb_level

    if not trigger_mask.any():
        return RepricedEntry(
            **base,
            trigger_trade_price=None,
            trigger_trade_ts=None,
            bbo_at_trigger_bid=None,
            bbo_at_trigger_ask=None,
            bbo_at_trigger_spread=None,
            estimated_fill_price=None,
            actual_slippage_points=None,
            actual_slippage_ticks=None,
            error="no_trigger_trade_found",
        )

    # Use positional index — ts_event index may have duplicate timestamps
    trigger_pos = int(trigger_mask.values.argmax())
    trigger_row = df.iloc[trigger_pos]

    trigger_price = float(trigger_row["price"])
    trigger_ts = str(trigger_row.name if df.index.name == "ts_event" else trigger_row["ts_event"])
    bid = float(trigger_row["bid_px_00"])
    ask = float(trigger_row["ask_px_00"])
    spread_ticks = round((ask - bid) / tick_size, 1) if tick_size > 0 else None

    # Stop-market fill estimate: ask for long, bid for short
    if break_dir == "long":
        fill_price = ask
        slippage_points = fill_price - orb_level
    else:
        fill_price = bid
        slippage_points = orb_level - fill_price

    slippage_ticks = round(slippage_points / tick_size, 1) if tick_size > 0 else None

    return RepricedEntry(
        **base,
        trigger_trade_price=trigger_price,
        trigger_trade_ts=trigger_ts,
        bbo_at_trigger_bid=bid,
        bbo_at_trigger_ask=ask,
        bbo_at_trigger_spread=spread_ticks,
        estimated_fill_price=fill_price,
        actual_slippage_points=round(slippage_points, 10),
        actual_slippage_ticks=slippage_ticks,
        error=None,
    )


def reprice_all_pilot_days(
    manifest_df: pd.DataFrame,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Reprice all pilot days from cached tbbo data."""
    symbols = symbols or SYMBOLS
    cost_spec = get_cost_spec("MGC")
    results: list[dict] = []

    for _, row in manifest_df.iterrows():
        day = str(row["trading_day"])
        for sym in symbols:
            cache = _cache_path(day, sym)
            if not cache.exists():
                results.append(asdict(RepricedEntry(
                    trading_day=day,
                    break_dir=row["break_dir"],
                    symbol_pulled=sym,
                    orb_level=row["orb_CME_REOPEN_high"] if row["break_dir"] == "long" else row["orb_CME_REOPEN_low"],
                    modeled_entry_price=row["model_entry_price"],
                    modeled_slippage_ticks=int(row["modeled_entry_slippage_ticks"]),
                    tick_size=cost_spec.tick_size,
                    tbbo_records_in_window=0,
                    trigger_trade_price=None,
                    trigger_trade_ts=None,
                    bbo_at_trigger_bid=None,
                    bbo_at_trigger_ask=None,
                    bbo_at_trigger_spread=None,
                    estimated_fill_price=None,
                    actual_slippage_points=None,
                    actual_slippage_ticks=None,
                    error="cache_file_missing",
                )))
                continue

            tbbo_df = load_tbbo_df(cache)
            # ORB range is not set until aperture closes
            orb_end = (
                pd.Timestamp(row["orb_start_utc"])
                + pd.Timedelta(minutes=int(row["orb_minutes"]))
            ).isoformat()
            repriced = reprice_e2_entry(
                tbbo_df=tbbo_df,
                orb_high=row["orb_CME_REOPEN_high"],
                orb_low=row["orb_CME_REOPEN_low"],
                break_dir=row["break_dir"],
                model_entry_price=row["model_entry_price"],
                model_entry_ts_utc=row["model_entry_ts_utc"],
                trading_day=day,
                symbol_pulled=sym,
                tick_size=cost_spec.tick_size,
                modeled_slippage_ticks=int(row["modeled_entry_slippage_ticks"]),
                orb_end_utc=orb_end,
            )
            results.append(asdict(repriced))

    return pd.DataFrame(results)


def analyze_slippage(repriced_df: pd.DataFrame) -> dict:
    """Analyze slippage distribution from repriced entries."""
    analysis: dict = {}

    for sym in repriced_df["symbol_pulled"].unique():
        sym_df = repriced_df[
            (repriced_df["symbol_pulled"] == sym) & repriced_df["actual_slippage_ticks"].notna()
        ].copy()

        if sym_df.empty:
            analysis[sym] = {"n": 0, "error": "no_valid_repriced_entries"}
            continue

        ticks = sym_df["actual_slippage_ticks"]
        spreads = sym_df["bbo_at_trigger_spread"]

        analysis[sym] = {
            "n": len(sym_df),
            "slippage_ticks": {
                "mean": round(float(ticks.mean()), 2),
                "median": round(float(ticks.median()), 2),
                "std": round(float(ticks.std()), 2),
                "min": round(float(ticks.min()), 2),
                "max": round(float(ticks.max()), 2),
                "p25": round(float(ticks.quantile(0.25)), 2),
                "p75": round(float(ticks.quantile(0.75)), 2),
                "p95": round(float(ticks.quantile(0.95)), 2),
                "pct_above_1_tick": round(float((ticks > 1.0).mean()) * 100, 1),
                "pct_above_2_ticks": round(float((ticks > 2.0).mean()) * 100, 1),
            },
            "spread_ticks": {
                "mean": round(float(spreads.mean()), 2),
                "median": round(float(spreads.median()), 2),
                "max": round(float(spreads.max()), 2),
            },
            "by_direction": {},
        }

        for direction in ["long", "short"]:
            dir_df = sym_df[sym_df["break_dir"] == direction]
            if not dir_df.empty:
                dir_ticks = dir_df["actual_slippage_ticks"]
                analysis[sym]["by_direction"][direction] = {
                    "n": len(dir_df),
                    "mean_slippage_ticks": round(float(dir_ticks.mean()), 2),
                    "median_slippage_ticks": round(float(dir_ticks.median()), 2),
                }

    # Modeled vs actual comparison
    valid = repriced_df[repriced_df["actual_slippage_ticks"].notna()].copy()
    if not valid.empty:
        analysis["_comparison"] = {
            "modeled_assumption_ticks": E2_SLIPPAGE_TICKS,
            "caveat": "BBO at trigger trade is optimistic lower bound — real fills likely worse",
        }

    return analysis


def write_results(
    repriced_df: pd.DataFrame,
    analysis: dict,
    cost_df: pd.DataFrame | None = None,
) -> None:
    """Write repricing results to research/output/."""
    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    repriced_df.to_csv(out_dir / "mgc_e2_repriced_entries.csv", index=False)

    payload = {"slippage_analysis": analysis}
    if cost_df is not None:
        payload["cost_summary"] = {
            "total_estimated_usd": cost_df["estimated_cost_usd"].sum(),
            "per_symbol": cost_df.groupby("symbol")["estimated_cost_usd"].sum().to_dict(),
            "errors": int(cost_df["error"].notna().sum()),
        }

    (out_dir / "mgc_e2_slippage_analysis.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )


def render_analysis(analysis: dict) -> None:
    """Print analysis to console."""
    print("=" * 80)
    print("E2 SLIPPAGE REPRICING — MICROSTRUCTURE ANALYSIS")
    print("=" * 80)

    for sym, data in analysis.items():
        if sym.startswith("_"):
            continue
        print(f"\n{sym}:")
        if data.get("error"):
            print(f"  {data['error']}")
            continue

        s = data["slippage_ticks"]
        print(f"  N = {data['n']} repriced entries")
        print(f"  Slippage: mean={s['mean']} ticks, median={s['median']}, "
              f"std={s['std']}, range=[{s['min']}, {s['max']}]")
        print(f"  P25={s['p25']}, P75={s['p75']}, P95={s['p95']}")
        print(f"  Above 1 tick: {s['pct_above_1_tick']}%, Above 2 ticks: {s['pct_above_2_ticks']}%")

        sp = data["spread_ticks"]
        print(f"  Spread at trigger: mean={sp['mean']} ticks, median={sp['median']}, max={sp['max']}")

        for direction, dd in data.get("by_direction", {}).items():
            print(f"  {direction}: N={dd['n']}, mean={dd['mean_slippage_ticks']}, "
                  f"median={dd['median_slippage_ticks']}")

    if "_comparison" in analysis:
        c = analysis["_comparison"]
        print(f"\nModeled assumption: {c['modeled_assumption_ticks']} tick")
        print(f"Caveat: {c['caveat']}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Databento microstructure pilot")
    parser.add_argument("--estimate-cost", action="store_true",
                        help="Estimate Databento cost per pilot day")
    parser.add_argument("--pull", action="store_true",
                        help="Pull tbbo data for all pilot days")
    parser.add_argument("--reprice", action="store_true",
                        help="Reprice E2 entries from cached tbbo data")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cached")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        help="Symbols to pull (default: MGC.FUT GC.FUT)")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    return parser.parse_args()


def _load_manifest(db_path: Path) -> pd.DataFrame:
    """Load the pilot manifest (rebuild if needed)."""
    from research.research_mgc_e2_microstructure_pilot import (
        add_manifest_windows,
        load_e2_touch_candidates,
        stratified_sample_days,
    )

    candidates = load_e2_touch_candidates(db_path)
    sample = stratified_sample_days(candidates)
    return add_manifest_windows(sample)


def main() -> None:
    args = parse_args()

    if not any([args.estimate_cost, args.pull, args.reprice]):
        print("Specify --estimate-cost, --pull, and/or --reprice")
        return

    manifest = _load_manifest(args.db_path)
    print(f"Pilot manifest: {len(manifest)} days")

    if args.estimate_cost:
        print("\nEstimating costs...")
        cost_df = estimate_cost_per_day(manifest, args.symbols)
        total = cost_df["estimated_cost_usd"].sum()
        errors = cost_df["error"].notna().sum()
        print(f"Total estimated cost: ${total:.4f} ({errors} errors)")
        for sym in args.symbols:
            sym_cost = cost_df[cost_df["symbol"] == sym]["estimated_cost_usd"].sum()
            print(f"  {sym}: ${sym_cost:.4f}")
        cost_df.to_csv(
            PROJECT_ROOT / "research" / "output" / "mgc_e2_tbbo_cost_estimate.csv",
            index=False,
        )

    if args.pull:
        print("\nPulling tbbo data...")
        pull_all_pilot_days(manifest, args.symbols, force=args.force)
        print("Done.")

    if args.reprice:
        print("\nRepricing E2 entries...")
        repriced = reprice_all_pilot_days(manifest, args.symbols)
        analysis = analyze_slippage(repriced)
        cost_df = None
        cost_path = PROJECT_ROOT / "research" / "output" / "mgc_e2_tbbo_cost_estimate.csv"
        if cost_path.exists():
            cost_df = pd.read_csv(cost_path)
        write_results(repriced, analysis, cost_df)
        render_analysis(analysis)


if __name__ == "__main__":
    main()
