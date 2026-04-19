#!/usr/bin/env python3
"""
MGC CME_REOPEN E2 microstructure pilot pack.

Builds a falsification-first pilot around one question:
"Is the current MGC CME_REOPEN E2 1-tick entry-slippage assumption too optimistic?"

Default behavior is deliberately local and cheap:
- selects a deterministic stratified sample of real E2 touch days
- writes a pilot-day manifest with UTC windows for microstructure pulls
- summarizes active MGC CME_REOPEN E2 strategies against the sampled days

Optional network step:
- `--check-databento` verifies schema access and estimates pilot cost

This script does NOT auto-download quote/trade data by default. Cost-incurring
data pulls should remain explicit until the pilot pack is reviewed.

Usage:
    python -m research.research_mgc_e2_microstructure_pilot
    python -m research.research_mgc_e2_microstructure_pilot --check-databento
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests

from pipeline.build_daily_features import _orb_utc_window
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH, PROJECT_ROOT
from trading_app.config import ALL_FILTERS, E2_SLIPPAGE_TICKS

DATASET = "GLBX.MDP3"
INSTRUMENT = "MGC"
ORB_LABEL = "CME_REOPEN"
ORB_MINUTES = 5
ATR_THRESHOLD = 50.0
HIGH_VOL_SAMPLE_N = 20
LOW_VOL_SAMPLE_N = 20
DEFAULT_SEED = 7
WINDOW_MINUTES_BEFORE = 2
WINDOW_MINUTES_AFTER = 50  # Must cover max break_delay (39min observed) + buffer
PREFERRED_SCHEMAS = ("tbbo", "mbp-1")
DATABENTO_PARENT_SYMBOL = "GC.FUT"


@dataclass(frozen=True)
class PilotSummary:
    instrument: str
    orb_label: str
    orb_minutes: int
    total_sample_days: int
    high_vol_days: int
    low_vol_days: int
    atr_threshold: float
    seed: int
    modeled_entry_slippage_ticks: int
    modeled_tick_size: float
    sample_year_counts: dict[str, int]
    sample_break_dir_counts: dict[str, int]
    caveats: list[str]


def choose_microstructure_schema(available_schemas: Iterable[str]) -> str | None:
    """Prefer the cheapest useful trigger-truth layer first."""
    available = {schema.lower() for schema in available_schemas}
    for schema in PREFERRED_SCHEMAS:
        if schema in available:
            return schema
    return None


def _year_counts(sample_df: pd.DataFrame) -> dict[str, int]:
    counts = sample_df["trading_day"].map(lambda d: str(d.year)).value_counts().sort_index()
    return {str(year): int(count) for year, count in counts.items()}


def _break_dir_counts(sample_df: pd.DataFrame) -> dict[str, int]:
    counts = sample_df["break_dir"].value_counts(dropna=False).sort_index()
    return {str(direction): int(count) for direction, count in counts.items()}


def load_e2_touch_candidates(
    db_path: Path,
    instrument: str = INSTRUMENT,
    orb_label: str = ORB_LABEL,
    orb_minutes: int = ORB_MINUTES,
) -> pd.DataFrame:
    """Load real E2 touch days joined to daily_features context.

    Important bias guard:
    We sample from `orb_outcomes` E2 touch rows, not `daily_features` close-break rows.
    E2 triggers on range touch, so close-based break timestamps would undercount fakeouts.
    """
    query = f"""
        SELECT
            df.*,
            o.entry_ts AS model_entry_ts,
            o.entry_price AS model_entry_price,
            o.stop_price AS model_stop_price,
            o.outcome AS model_outcome_rr1,
            o.pnl_r AS model_pnl_r_rr1
        FROM orb_outcomes o
        JOIN daily_features df
          ON o.trading_day = df.trading_day
         AND o.symbol = df.symbol
         AND o.orb_minutes = df.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = 'E2'
          AND o.confirm_bars = 1
          AND o.rr_target = 1.0
          AND o.entry_ts IS NOT NULL
          AND df.atr_20_pct IS NOT NULL
        ORDER BY df.trading_day
    """
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(query, [instrument, orb_label, orb_minutes]).fetchdf()

    if df.empty:
        raise ValueError(f"No E2 touch candidates found for {instrument} {orb_label} O{orb_minutes}")

    cost_spec = get_cost_spec(instrument)
    orb_high_col = f"orb_{orb_label}_high"
    orb_low_col = f"orb_{orb_label}_low"

    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["model_entry_ts"] = pd.to_datetime(df["model_entry_ts"], utc=True)
    df["break_dir"] = df.apply(
        lambda row: "long" if row["model_entry_price"] > row[orb_high_col] else "short",
        axis=1,
    )
    trigger_level = df.apply(
        lambda row: row[orb_high_col] if row["break_dir"] == "long" else row[orb_low_col],
        axis=1,
    )
    df["modeled_entry_slippage_points"] = (df["model_entry_price"] - trigger_level).abs().round(10)
    df["modeled_entry_slippage_ticks"] = (df["modeled_entry_slippage_points"] / cost_spec.tick_size).round().astype(int)
    df["atr_bucket"] = df["atr_20_pct"].apply(lambda v: "high_vol" if v >= ATR_THRESHOLD else "low_vol")

    return df


def stratified_sample_days(
    candidates_df: pd.DataFrame,
    *,
    high_count: int = HIGH_VOL_SAMPLE_N,
    low_count: int = LOW_VOL_SAMPLE_N,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Select a deterministic ATR-stratified sample."""
    high = candidates_df[candidates_df["atr_bucket"] == "high_vol"]
    low = candidates_df[candidates_df["atr_bucket"] == "low_vol"]

    if len(high) < high_count:
        raise ValueError(f"Need {high_count} high-vol days, found {len(high)}")
    if len(low) < low_count:
        raise ValueError(f"Need {low_count} low-vol days, found {len(low)}")

    sample_high = high.sample(n=high_count, random_state=seed)
    sample_low = low.sample(n=low_count, random_state=seed + 1)

    sample = pd.concat([sample_high, sample_low], ignore_index=True)
    sort_cols = ["trading_day"]
    if "break_dir" in sample.columns:
        sort_cols.append("break_dir")
    sample = sample.sort_values(sort_cols).reset_index(drop=True)
    return sample


def add_manifest_windows(
    sample_df: pd.DataFrame,
    *,
    orb_label: str = ORB_LABEL,
    orb_minutes: int = ORB_MINUTES,
    minutes_before: int = WINDOW_MINUTES_BEFORE,
    minutes_after: int = WINDOW_MINUTES_AFTER,
) -> pd.DataFrame:
    """Attach UTC windows for targeted Databento pulls.

    Window end = max(orb_start + minutes_after, model_entry_ts + 5min).
    This ensures tick data always covers the actual break moment even for
    late-breaking days (break_delay up to 39min observed in pilot).
    """
    entry_buffer = timedelta(minutes=5)
    rows: list[dict] = []
    for row in sample_df.to_dict("records"):
        orb_start_utc, _ = _orb_utc_window(row["trading_day"], orb_label, orb_minutes)
        window_start_utc = orb_start_utc - timedelta(minutes=minutes_before)
        fixed_end = orb_start_utc + timedelta(minutes=minutes_after)
        entry_ts = pd.Timestamp(row["model_entry_ts"]).tz_convert("UTC")
        entry_end = entry_ts + entry_buffer
        window_end_utc = max(fixed_end, entry_end)

        out = dict(row)
        out["orb_start_utc"] = orb_start_utc.isoformat()
        out["window_start_utc"] = window_start_utc.isoformat()
        out["window_end_utc"] = window_end_utc.isoformat()
        out["model_entry_ts_utc"] = entry_ts.isoformat()
        rows.append(out)

    return pd.DataFrame(rows)


def build_strategy_panel(
    db_path: Path,
    candidates_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    *,
    instrument: str = INSTRUMENT,
    orb_label: str = ORB_LABEL,
    orb_minutes: int = ORB_MINUTES,
) -> pd.DataFrame:
    """Summarize active strategies against the sampled touch days.

    This is still a pre-microstructure baseline. It tells us what live/active
    MGC CME_REOPEN E2 families look like on the sampled days before any quote-level
    repricing is applied.
    """
    with duckdb.connect(str(db_path), read_only=True) as con:
        strategies = con.execute(
            """
            SELECT strategy_id, filter_type, rr_target, confirm_bars,
                   sample_size, win_rate, expectancy_r, avg_risk_dollars
            FROM validated_setups
            WHERE instrument = ?
              AND orb_label = ?
              AND orb_minutes = ?
              AND entry_model = 'E2'
              AND status = 'active'
            ORDER BY expectancy_r DESC NULLS LAST, strategy_id
            """,
            [instrument, orb_label, orb_minutes],
        ).fetchdf()

        outcomes = con.execute(
            """
            SELECT trading_day, rr_target, confirm_bars, pnl_r, risk_dollars, outcome
            FROM orb_outcomes
            WHERE symbol = ?
              AND orb_label = ?
              AND orb_minutes = ?
              AND entry_model = 'E2'
              AND confirm_bars = 1
              AND entry_ts IS NOT NULL
            """,
            [instrument, orb_label, orb_minutes],
        ).fetchdf()

    if strategies.empty:
        return strategies

    sample_days = set(sample_df["trading_day"])
    sample_features = candidates_df[candidates_df["trading_day"].isin(sample_days)].copy()
    outcomes["trading_day"] = pd.to_datetime(outcomes["trading_day"]).dt.date
    outcome_lookup = outcomes.groupby(["rr_target", "confirm_bars"])

    rows: list[dict] = []
    total_sample_days = len(sample_features)

    for strategy in strategies.to_dict("records"):
        filter_obj = ALL_FILTERS.get(strategy["filter_type"])
        if filter_obj is None:
            raise KeyError(f"validated_setups filter_type '{strategy['filter_type']}' missing from ALL_FILTERS")

        qualifying_mask = filter_obj.matches_df(sample_features, orb_label)
        qualifying_days = sample_features.loc[qualifying_mask, "trading_day"].tolist()

        try:
            strategy_outcomes = outcome_lookup.get_group((strategy["rr_target"], strategy["confirm_bars"]))
        except KeyError:
            strategy_outcomes = outcomes.iloc[0:0].copy()

        strategy_outcomes = strategy_outcomes[strategy_outcomes["trading_day"].isin(qualifying_days)]
        pilot_n = int(len(strategy_outcomes))
        pilot_avg_r = float(strategy_outcomes["pnl_r"].mean()) if pilot_n else None
        pilot_wr = float((strategy_outcomes["pnl_r"] > 0).mean()) if pilot_n else None
        pilot_avg_risk_dollars = float(strategy_outcomes["risk_dollars"].mean()) if pilot_n else None

        rows.append(
            {
                "strategy_id": strategy["strategy_id"],
                "filter_type": strategy["filter_type"],
                "rr_target": float(strategy["rr_target"]),
                "confirm_bars": int(strategy["confirm_bars"]),
                "full_sample_size": int(strategy["sample_size"]),
                "full_win_rate": float(strategy["win_rate"]) if strategy["win_rate"] is not None else None,
                "full_expectancy_r": float(strategy["expectancy_r"]) if strategy["expectancy_r"] is not None else None,
                "full_avg_risk_dollars": (
                    float(strategy["avg_risk_dollars"]) if strategy["avg_risk_dollars"] is not None else None
                ),
                "pilot_qualifying_days": int(len(qualifying_days)),
                "pilot_qualification_rate": len(qualifying_days) / total_sample_days if total_sample_days else None,
                "pilot_n": pilot_n,
                "pilot_avg_r": pilot_avg_r,
                "pilot_wr": pilot_wr,
                "pilot_avg_risk_dollars": pilot_avg_risk_dollars,
            }
        )

    return pd.DataFrame(rows).sort_values(["pilot_avg_r", "full_expectancy_r"], ascending=[False, False])


def estimate_databento_cost(sample_manifest_df: pd.DataFrame) -> dict[str, object]:
    """Optional network step: schema access + cost estimate.

    Uses Databento's documented metadata HTTP endpoints directly rather than the
    Python client. This keeps the pilot script resilient if the client package
    misbehaves on interpreter shutdown.
    """
    api_key = os.getenv("DATABENTO_API_KEY")
    results: dict[str, object] = {
        "dataset": DATASET,
        "available_schemas": [],
        "preferred_schema": None,
        "symbol": DATABENTO_PARENT_SYMBOL,
        "estimated_total_cost_usd": None,
        "cost_method": "year_bucket_upper_bound",
        "estimated_bucket_costs_usd": [],
        "error": None,
    }
    if not api_key:
        results["error"] = "DATABENTO_API_KEY not found in environment or .env"
        return results

    base_url = "https://hist.databento.com/v0"
    auth = (api_key, "")
    headers = {"accept": "application/json"}

    try:
        with requests.Session() as session:
            session.headers.update(headers)
            schema_resp = session.get(
                f"{base_url}/metadata.list_schemas",
                params={"dataset": DATASET},
                auth=auth,
                timeout=(5, 5),
            )
            schema_resp.raise_for_status()
            available_schemas = schema_resp.json()
            preferred_schema = choose_microstructure_schema(available_schemas)
            results["available_schemas"] = available_schemas
            results["preferred_schema"] = preferred_schema

            if preferred_schema is None:
                return results

            bucket_costs: list[dict[str, object]] = []
            total_cost = 0.0

            manifest = sample_manifest_df.copy()
            manifest["sample_year"] = pd.to_datetime(manifest["trading_day"]).dt.year

            for year, year_rows in manifest.groupby("sample_year"):
                start_utc = year_rows["window_start_utc"].min()
                end_utc = year_rows["window_end_utc"].max()
                cost_resp = session.post(
                    f"{base_url}/metadata.get_cost",
                    data={
                        "dataset": DATASET,
                        "start": start_utc,
                        "end": end_utc,
                        "symbols": DATABENTO_PARENT_SYMBOL,
                        "schema": preferred_schema,
                        "stype_in": "parent",
                        "stype_out": "instrument_id",
                    },
                    auth=auth,
                    timeout=(5, 5),
                )
                cost_resp.raise_for_status()
                cost_value = float(cost_resp.json())
                total_cost += cost_value
                bucket_costs.append(
                    {
                        "sample_year": int(year),
                        "schema": preferred_schema,
                        "bucket_start_utc": start_utc,
                        "bucket_end_utc": end_utc,
                        "sample_days_in_bucket": int(len(year_rows)),
                        "estimated_upper_bound_cost_usd": round(cost_value, 6),
                    }
                )

            results["estimated_total_cost_usd"] = round(total_cost, 6)
            results["estimated_bucket_costs_usd"] = bucket_costs
            return results
    except requests.RequestException as exc:
        results["error"] = str(exc)
        return results


def write_outputs(
    output_dir: Path,
    sample_manifest_df: pd.DataFrame,
    strategy_panel_df: pd.DataFrame,
    summary: PilotSummary,
    databento_info: dict[str, object] | None,
) -> None:
    """Write the pilot pack artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "mgc_e2_microstructure_pilot_days.csv"
    strategy_path = output_dir / "mgc_e2_microstructure_pilot_strategies.csv"
    summary_path = output_dir / "mgc_e2_microstructure_pilot_summary.json"

    manifest_cols = [
        "trading_day",
        "atr_20_pct",
        "atr_bucket",
        "break_dir",
        "model_entry_ts_utc",
        "model_entry_price",
        "modeled_entry_slippage_points",
        "modeled_entry_slippage_ticks",
        "orb_CME_REOPEN_high",
        "orb_CME_REOPEN_low",
        "orb_CME_REOPEN_size",
        "orb_CME_REOPEN_break_delay_min",
        "window_start_utc",
        "window_end_utc",
    ]
    sample_manifest_df[manifest_cols].to_csv(sample_path, index=False)
    strategy_panel_df.to_csv(strategy_path, index=False)

    payload = {
        "summary": asdict(summary),
        "databento": databento_info,
        "artifacts": {
            "sample_days_csv": str(sample_path.relative_to(PROJECT_ROOT)),
            "strategy_panel_csv": str(strategy_path.relative_to(PROJECT_ROOT)),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def render_console_summary(
    summary: PilotSummary,
    strategy_panel_df: pd.DataFrame,
    databento_info: dict[str, object] | None,
) -> None:
    """Print a concise, honest summary for terminal use."""
    print("=" * 80)
    print("MGC CME_REOPEN E2 MICROSTRUCTURE PILOT")
    print("=" * 80)
    print(
        f"Sample days: {summary.total_sample_days} ({summary.high_vol_days} high-vol, {summary.low_vol_days} low-vol)"
    )
    print(f"ATR threshold: {summary.atr_threshold:.1f} percentile")
    print(f"Modeled entry slippage: {summary.modeled_entry_slippage_ticks} tick @ {summary.modeled_tick_size:.2f}")
    print(f"Sample years: {summary.sample_year_counts}")
    print(f"Break directions: {summary.sample_break_dir_counts}")
    print()

    if not strategy_panel_df.empty:
        print("Active MGC CME_REOPEN E2 strategies on sampled touch days:")
        print("  strategy_id | pilot_n | pilot_avg_r | full_expectancy_r | filter_type")
        for row in strategy_panel_df.head(8).itertuples(index=False):
            pilot_avg = "N/A" if pd.isna(row.pilot_avg_r) else f"{row.pilot_avg_r:.4f}"
            full_avg = "N/A" if pd.isna(row.full_expectancy_r) else f"{row.full_expectancy_r:.4f}"
            print(f"  {row.strategy_id} | {row.pilot_n:>3} | {pilot_avg:>10} | {full_avg:>16} | {row.filter_type}")
        print()

    if databento_info is not None:
        print("Databento:")
        print(f"  available_schemas={databento_info.get('available_schemas')}")
        print(f"  preferred_schema={databento_info.get('preferred_schema')}")
        print(f"  estimated_total_cost_usd={databento_info.get('estimated_total_cost_usd')}")
        if databento_info.get("error"):
            print(f"  error={databento_info.get('error')}")
        print()

    print("Caveats:")
    for item in summary.caveats:
        print(f"  - {item}")
    print()
    print("Pilot outputs written under research/output/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MGC CME_REOPEN E2 microstructure pilot pack")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--high-count", type=int, default=HIGH_VOL_SAMPLE_N)
    parser.add_argument("--low-count", type=int, default=LOW_VOL_SAMPLE_N)
    parser.add_argument("--check-databento", action="store_true", help="Check schema access and estimate pilot cost")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    candidates_df = load_e2_touch_candidates(args.db_path)
    sample_df = stratified_sample_days(
        candidates_df,
        high_count=args.high_count,
        low_count=args.low_count,
        seed=args.seed,
    )
    sample_manifest_df = add_manifest_windows(sample_df)
    strategy_panel_df = build_strategy_panel(args.db_path, candidates_df, sample_df)

    databento_info = estimate_databento_cost(sample_manifest_df) if args.check_databento else None

    summary = PilotSummary(
        instrument=INSTRUMENT,
        orb_label=ORB_LABEL,
        orb_minutes=ORB_MINUTES,
        total_sample_days=len(sample_manifest_df),
        high_vol_days=int((sample_manifest_df["atr_bucket"] == "high_vol").sum()),
        low_vol_days=int((sample_manifest_df["atr_bucket"] == "low_vol").sum()),
        atr_threshold=ATR_THRESHOLD,
        seed=args.seed,
        modeled_entry_slippage_ticks=E2_SLIPPAGE_TICKS,
        modeled_tick_size=get_cost_spec(INSTRUMENT).tick_size,
        sample_year_counts=_year_counts(sample_manifest_df),
        sample_break_dir_counts=_break_dir_counts(sample_manifest_df),
        caveats=[
            "This pilot pack is built from actual E2 touch days, not close-based break rows.",
            "It still does not reprice entries from quote-level data until the Databento pull is run.",
            "ATR stratification controls one regime axis only; it does not isolate event-day or contract-roll effects.",
            "Strategy-panel metrics are baseline sampled outcomes before any microstructure repricing.",
        ],
    )

    output_dir = PROJECT_ROOT / "research" / "output"
    write_outputs(output_dir, sample_manifest_df, strategy_panel_df, summary, databento_info)
    render_console_summary(summary, strategy_panel_df, databento_info)


if __name__ == "__main__":
    main()
