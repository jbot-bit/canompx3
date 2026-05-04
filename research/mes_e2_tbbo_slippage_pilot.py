#!/usr/bin/env python3
"""MES E2 slippage validation pilot.

Purpose:
- close the open MES portion of `cost-realism-slippage-pilot`
- measure E2 stop-market fill quality against TBBO data
- avoid any new alpha discovery or threshold selection

Two operating modes:
1. `--estimate-cost` / `--pull` / `--reprice` for a fresh Databento TBBO
   sample. Always estimate cost before pulling paid data.
2. `--reprice-cache` for already-cached MES TBBO windows, with no Databento
   spend.

Canonical delegation:
- `research.databento_microstructure.reprice_e2_entry`
- `pipeline.build_daily_features._orb_utc_window`
- `pipeline.cost_model.get_cost_spec("MES")`
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
INSTRUMENT = "MES"
DATABENTO_SYMBOL = "MES.FUT"
ORB_MINUTES = 5
SAMPLE_PER_BUCKET = 5
DEFAULT_SEED = 42
WINDOW_MINUTES_BEFORE = 2
WINDOW_MINUTES_AFTER = 30
CACHE_DIR = PROJECT_ROOT / "research" / "data" / "tbbo_mes_pilot"
RESULT_DOC = PROJECT_ROOT / "docs" / "audit" / "results" / "2026-04-24-mes-e2-slippage-pilot-v1.md"

# Current deployable MES sessions from deployable_validated_relation() audit.
PILOT_SESSIONS = [
    "CME_PRECLOSE",
    "COMEX_SETTLE",
    "SINGAPORE_OPEN",
    "US_DATA_830",
]
ALLOWED_PILOT_SESSIONS = frozenset(PILOT_SESSIONS)

CACHE_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_([A-Z_0-9]+)_MES\.dbn\.zst$")


@dataclass(frozen=True)
class PilotDay:
    trading_day: str
    orb_label: str
    break_dir: str
    orb_level: float
    atr_20: float
    atr_regime: str
    window_start_utc: str
    window_end_utc: str


def parse_cache_filename(filename: str) -> tuple[str, str] | None:
    """Parse `YYYY-MM-DD_SESSION_MES.dbn.zst` into `(day, session)`."""
    match = CACHE_FILENAME_RE.match(filename)
    if not match:
        return None
    return match.group(1), match.group(2)


def _modeled_slippage_ticks() -> int:
    spec = get_cost_spec(INSTRUMENT)
    return int(round(spec.slippage / spec.point_value / spec.tick_size))


def _cache_path(day: PilotDay) -> Path:
    return CACHE_DIR / f"{day.trading_day}_{day.orb_label}_MES.dbn.zst"


def _validate_pilot_sessions(sessions: list[str]) -> list[str]:
    unknown = sorted(set(sessions) - ALLOWED_PILOT_SESSIONS)
    if unknown:
        allowed = ", ".join(sorted(ALLOWED_PILOT_SESSIONS))
        raise ValueError(f"Unsupported MES pilot session(s): {', '.join(unknown)}. Allowed: {allowed}")
    return sessions


def build_manifest_from_cache(cache_dir: Path) -> list[dict]:
    """Build a no-spend manifest from cached MES TBBO filenames."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        rows: list[dict] = []
        for cache_path in sorted(cache_dir.glob("*_MES.dbn.zst")):
            parsed = parse_cache_filename(cache_path.name)
            if parsed is None:
                rows.append(
                    {
                        "trading_day": None,
                        "orb_label": None,
                        "cache_path": str(cache_path),
                        "orb_high": None,
                        "orb_low": None,
                        "break_dir": None,
                        "atr_20": None,
                        "error": f"filename_regex_failed: {cache_path.name}",
                    }
                )
                continue

            day, session = parsed
            if session not in ALLOWED_PILOT_SESSIONS:
                rows.append(
                    {
                        "trading_day": day,
                        "orb_label": session,
                        "cache_path": str(cache_path),
                        "orb_high": None,
                        "orb_low": None,
                        "break_dir": None,
                        "atr_20": None,
                        "error": f"unsupported_session: {session}",
                    }
                )
                continue

            high_col = f"orb_{session}_high"
            low_col = f"orb_{session}_low"
            dir_col = f"orb_{session}_break_dir"

            try:
                df_row = con.execute(
                    f"""
                    SELECT {high_col}, {low_col}, {dir_col}, atr_20
                    FROM daily_features
                    WHERE symbol = 'MES'
                      AND orb_minutes = 5
                      AND trading_day = CAST(? AS DATE)
                    """,
                    [day],
                ).fetchone()
            except duckdb.Error as exc:
                rows.append(
                    {
                        "trading_day": day,
                        "orb_label": session,
                        "cache_path": str(cache_path),
                        "orb_high": None,
                        "orb_low": None,
                        "break_dir": None,
                        "atr_20": None,
                        "error": f"duckdb_query_failed: {exc}",
                    }
                )
                continue

            if df_row is None or df_row[0] is None or df_row[1] is None or df_row[2] is None:
                rows.append(
                    {
                        "trading_day": day,
                        "orb_label": session,
                        "cache_path": str(cache_path),
                        "orb_high": None,
                        "orb_low": None,
                        "break_dir": None,
                        "atr_20": None,
                        "error": "daily_features missing or incomplete",
                    }
                )
                continue

            orb_high, orb_low, break_dir, atr_20 = df_row
            rows.append(
                {
                    "trading_day": day,
                    "orb_label": session,
                    "cache_path": str(cache_path),
                    "orb_high": float(orb_high),
                    "orb_low": float(orb_low),
                    "break_dir": str(break_dir),
                    "atr_20": float(atr_20) if atr_20 is not None else None,
                    "error": None,
                }
            )
        return rows
    finally:
        con.close()


def reprice_cache_manifest(manifest: list[dict]) -> list[dict]:
    """Reprice cached MES TBBO windows using the canonical E2 repricer."""
    spec = get_cost_spec(INSTRUMENT)
    modeled_slippage_ticks = _modeled_slippage_ticks()
    results: list[dict] = []

    for row in manifest:
        if row.get("error") is not None:
            results.append(
                {
                    "trading_day": row.get("trading_day"),
                    "orb_label": row.get("orb_label"),
                    "error": row["error"],
                }
            )
            continue

        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            results.append(
                {
                    "trading_day": row["trading_day"],
                    "orb_label": row["orb_label"],
                    "error": "cache_file_missing",
                }
            )
            continue

        try:
            tbbo_df = load_tbbo_df(cache_path)
        except Exception as exc:
            results.append(
                {
                    "trading_day": row["trading_day"],
                    "orb_label": row["orb_label"],
                    "error": f"load_tbbo_failed: {exc}",
                }
            )
            continue

        if tbbo_df.empty:
            results.append(
                {
                    "trading_day": row["trading_day"],
                    "orb_label": row["orb_label"],
                    "error": "empty_tbbo_after_front_month_filter",
                }
            )
            continue

        try:
            _, orb_end_dt = _orb_utc_window(date.fromisoformat(row["trading_day"]), row["orb_label"], ORB_MINUTES)
        except Exception as exc:
            results.append(
                {
                    "trading_day": row["trading_day"],
                    "orb_label": row["orb_label"],
                    "error": f"orb_utc_window_failed: {exc}",
                }
            )
            continue

        orb_high = float(row["orb_high"])
        orb_low = float(row["orb_low"])
        break_dir = str(row["break_dir"])
        orb_level = orb_high if break_dir == "long" else orb_low
        model_entry_price = orb_level + (spec.tick_size if break_dir == "long" else -spec.tick_size)

        entry = reprice_e2_entry(
            tbbo_df=tbbo_df,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir=break_dir,
            model_entry_price=model_entry_price,
            model_entry_ts_utc=orb_end_dt.isoformat(),
            trading_day=row["trading_day"],
            symbol_pulled=DATABENTO_SYMBOL,
            tick_size=spec.tick_size,
            modeled_slippage_ticks=modeled_slippage_ticks,
            orb_end_utc=orb_end_dt.isoformat(),
        )
        results.append(_entry_result_row(entry, row["orb_label"], row.get("atr_20"), orb_high, orb_low))

    return results


def _entry_result_row(entry, orb_label: str, atr_20: float | None, orb_high: float, orb_low: float) -> dict:
    return {
        "trading_day": entry.trading_day,
        "orb_label": orb_label,
        "break_dir": entry.break_dir,
        "atr_20": atr_20,
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


def build_pilot_manifest(seed: int = DEFAULT_SEED) -> list[PilotDay]:
    """Select MES E2 touch days across deployable MES sessions."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        all_days = []
        for session in PILOT_SESSIONS:
            break_dir_col = f"orb_{session}_break_dir"
            high_col = f"orb_{session}_high"
            low_col = f"orb_{session}_low"

            session_rows = con.execute(
                f"""
                SELECT
                    o.trading_day,
                    '{session}' AS orb_label,
                    df.{break_dir_col} AS break_dir,
                    CASE WHEN df.{break_dir_col} = 'long'
                         THEN df.{high_col} ELSE df.{low_col} END AS orb_level,
                    df.atr_20
                FROM orb_outcomes o
                JOIN daily_features df
                    ON o.trading_day = df.trading_day
                    AND df.symbol = 'MES'
                    AND df.orb_minutes = 5
                WHERE o.symbol = 'MES'
                    AND o.orb_label = '{session}'
                    AND o.entry_model = 'E2'
                    AND o.confirm_bars = 1
                    AND o.orb_minutes = 5
                    AND o.outcome IS NOT NULL
                    AND df.atr_20 IS NOT NULL
                    AND df.{break_dir_col} IS NOT NULL
                """
            ).fetchall()

            for row in session_rows:
                all_days.append(
                    {
                        "trading_day": str(row[0]),
                        "orb_label": row[1],
                        "break_dir": row[2],
                        "orb_level": float(row[3]) if row[3] is not None else None,
                        "atr_20": float(row[4]),
                    }
                )
    finally:
        con.close()

    if not all_days:
        print("No MES E2 touch days found!")
        return []

    df = pd.DataFrame(all_days).dropna(subset=["orb_level"])
    df = df.sort_values(["orb_label", "trading_day", "break_dir", "orb_level"]).reset_index(drop=True)
    atr_median = df["atr_20"].median()
    df["atr_regime"] = np.where(df["atr_20"] >= atr_median, "high", "low")

    rng = np.random.default_rng(seed)
    sampled: list[PilotDay] = []
    for session in PILOT_SESSIONS:
        for regime in ["high", "low"]:
            bucket = df[(df["orb_label"] == session) & (df["atr_regime"] == regime)]
            if bucket.empty:
                continue
            sample = bucket.sample(n=min(SAMPLE_PER_BUCKET, len(bucket)), random_state=rng.integers(0, 2**31))
            for _, row in sample.iterrows():
                trading_day = date.fromisoformat(row["trading_day"])
                _, orb_end = _orb_utc_window(trading_day, session, ORB_MINUTES)
                sampled.append(
                    PilotDay(
                        trading_day=row["trading_day"],
                        orb_label=session,
                        break_dir=row["break_dir"],
                        orb_level=float(row["orb_level"]),
                        atr_20=float(row["atr_20"]),
                        atr_regime=regime,
                        window_start_utc=(orb_end - timedelta(minutes=WINDOW_MINUTES_BEFORE)).strftime(
                            "%Y-%m-%dT%H:%M:%S+00:00"
                        ),
                        window_end_utc=(orb_end + timedelta(minutes=WINDOW_MINUTES_AFTER)).strftime(
                            "%Y-%m-%dT%H:%M:%S+00:00"
                        ),
                    )
                )

    print(f"Pilot manifest: {len(sampled)} days")
    print(f"  Sessions: {sorted({d.orb_label for d in sampled})}")
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
    """Estimate Databento TBBO pull cost from up to the first three windows."""
    import databento as db

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("DATABENTO_API_KEY")

    client = db.Historical(api_key)
    total = 0.0
    sample = manifest[: min(3, len(manifest))]
    for day in sample:
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
        except Exception as exc:
            print(f"  Cost estimate failed for {day.trading_day} {day.orb_label}: {exc}")

    per_day = total / len(sample) if total > 0 and sample else 0.0
    estimated_total = per_day * len(manifest)
    print("\nCost estimate:")
    print(f"  Sample ({len(sample)} days): ${total:.2f}")
    print(f"  Per day: ${per_day:.3f}")
    print(f"  Total ({len(manifest)} days): ${estimated_total:.2f}")
    return estimated_total


def pull_tbbo(manifest: list[PilotDay], force: bool = False) -> list[Path]:
    """Pull paid MES TBBO data for the pilot manifest."""
    import databento as db

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("DATABENTO_API_KEY")

    client = db.Historical(api_key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    errors: list[tuple[str, str, str]] = []

    for i, day in enumerate(manifest):
        cache_path = _cache_path(day)
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
        except Exception as exc:
            print(f"    ERROR: {exc}")
            errors.append((day.trading_day, day.orb_label, str(exc)))

    if errors:
        print(f"\n{len(errors)} pull failures")
        for trading_day, session, err in errors:
            print(f"  {trading_day} {session}: {err}")
    return paths


def reprice_entries(manifest: list[PilotDay]) -> pd.DataFrame:
    """Reprice freshly pulled MES TBBO cache files."""
    spec = get_cost_spec(INSTRUMENT)
    modeled_slippage_ticks = _modeled_slippage_ticks()
    results = []

    for day in manifest:
        cache_path = _cache_path(day)
        if not cache_path.exists():
            results.append({"trading_day": day.trading_day, "orb_label": day.orb_label, "error": "no cache file"})
            continue

        tbbo_df = load_tbbo_df(cache_path)
        if tbbo_df.empty:
            results.append(
                {"trading_day": day.trading_day, "orb_label": day.orb_label, "error": "empty after filtering"}
            )
            continue

        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        try:
            high_col = f"orb_{day.orb_label}_high"
            low_col = f"orb_{day.orb_label}_low"
            df_row = con.execute(
                f"""
                SELECT {high_col}, {low_col}
                FROM daily_features
                WHERE symbol = 'MES'
                  AND orb_minutes = 5
                  AND trading_day = CAST(? AS DATE)
                """,
                [day.trading_day],
            ).fetchone()
        finally:
            con.close()

        if df_row is None or df_row[0] is None or df_row[1] is None:
            results.append(
                {"trading_day": day.trading_day, "orb_label": day.orb_label, "error": "daily_features missing"}
            )
            continue

        orb_high = float(df_row[0])
        orb_low = float(df_row[1])
        _, orb_end = _orb_utc_window(date.fromisoformat(day.trading_day), day.orb_label, ORB_MINUTES)
        model_entry_price = day.orb_level + (spec.tick_size if day.break_dir == "long" else -spec.tick_size)
        entry = reprice_e2_entry(
            tbbo_df=tbbo_df,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir=day.break_dir,
            model_entry_price=model_entry_price,
            model_entry_ts_utc=orb_end.isoformat(),
            trading_day=day.trading_day,
            symbol_pulled=DATABENTO_SYMBOL,
            tick_size=spec.tick_size,
            modeled_slippage_ticks=modeled_slippage_ticks,
            orb_end_utc=orb_end.isoformat(),
        )
        row = _entry_result_row(entry, day.orb_label, day.atr_20, orb_high, orb_low)
        row["atr_regime"] = day.atr_regime
        results.append(row)

    return pd.DataFrame(results)


def summarize_results(results_df: pd.DataFrame) -> dict:
    valid = results_df[results_df["error"].isna()].copy()
    errors = results_df[results_df["error"].notna()].copy()
    modeled = _modeled_slippage_ticks()
    if valid.empty:
        return {
            "valid_n": 0,
            "error_n": int(len(errors)),
            "modeled_ticks": modeled,
            "verdict": "INCONCLUSIVE",
            "reason": "no valid repriced entries",
        }

    slip = valid["slippage_ticks"].astype(float).to_numpy()
    median = float(np.median(slip))
    mean = float(np.mean(slip))
    p95 = float(np.quantile(slip, 0.95))
    max_slip = float(np.max(slip))
    if median <= modeled and p95 <= modeled * 2:
        verdict = "PASS"
        reason = "median is modeled-conservative and p95 stays within 2x modeled slippage"
    elif median <= modeled:
        verdict = "WARN"
        reason = "median is modeled-conservative but tail slippage needs session/outlier review"
    else:
        verdict = "FAIL"
        reason = "median slippage exceeds modeled slippage"

    return {
        "valid_n": int(len(valid)),
        "error_n": int(len(errors)),
        "modeled_ticks": modeled,
        "median_ticks": median,
        "mean_ticks": mean,
        "p95_ticks": p95,
        "max_ticks": max_slip,
        "pct_le_1_tick": float((slip <= 1).mean() * 100),
        "pct_le_2_ticks": float((slip <= 2).mean() * 100),
        "verdict": verdict,
        "reason": reason,
    }


def analyze_results(results_df: pd.DataFrame) -> None:
    summary = summarize_results(results_df)
    print(f"\n{'=' * 60}")
    print("MES E2 SLIPPAGE PILOT RESULTS")
    print(f"{'=' * 60}")
    print(f"  Valid samples: {summary['valid_n']}")
    print(f"  Errors: {summary['error_n']}")
    print(f"  Modeled slippage: {summary['modeled_ticks']} ticks")
    print(f"  Verdict: {summary['verdict']} — {summary['reason']}")

    if summary["valid_n"] == 0:
        return

    print("\n  Slippage (ticks):")
    print(f"    Median: {summary['median_ticks']:.2f}")
    print(f"    Mean:   {summary['mean_ticks']:.2f}")
    print(f"    p95:    {summary['p95_ticks']:.2f}")
    print(f"    Max:    {summary['max_ticks']:.2f}")
    print(f"    % <= 1 tick: {summary['pct_le_1_tick']:.1f}%")
    print(f"    % <= 2 ticks: {summary['pct_le_2_ticks']:.1f}%")

    valid = results_df[results_df["error"].isna()].copy()
    print("\n  Per session:")
    for session in sorted(valid["orb_label"].unique()):
        ticks = valid[valid["orb_label"] == session]["slippage_ticks"].astype(float)
        print(f"    {session:20s}: median={np.median(ticks):.1f}, mean={np.mean(ticks):.1f}, N={len(ticks)}")

    print("\n  Per break_dir:")
    for direction in ["long", "short"]:
        ticks = valid[valid["break_dir"] == direction]["slippage_ticks"].astype(float)
        if len(ticks) > 0:
            print(f"    {direction:5s}: median={np.median(ticks):.1f}, mean={np.mean(ticks):.1f}, N={len(ticks)}")


def build_result_doc(results_df: pd.DataFrame, *, source_csv: Path) -> str:
    summary = summarize_results(results_df)
    try:
        source_label = source_csv.relative_to(PROJECT_ROOT)
    except ValueError:
        source_label = source_csv
    lines = [
        "# MES E2 TBBO slippage pilot v1",
        "",
        "**Script:** `research/mes_e2_tbbo_slippage_pilot.py`",
        f"**Result CSV:** `{source_label}`",
        "**Scope:** MES | E2 | O5 | deployable MES sessions | TBBO stop-market repricing",
        f"**Sessions:** {', '.join(PILOT_SESSIONS)}",
        "",
        f"## Verdict: **{summary['verdict']}**",
        "",
        summary["reason"],
        "",
        "## Summary",
        "",
        f"- Valid repriced samples: `{summary['valid_n']}`",
        f"- Error rows: `{summary['error_n']}`",
        f"- Modeled slippage: `{summary['modeled_ticks']}` ticks",
    ]
    if summary["valid_n"]:
        lines.extend(
            [
                f"- Median slippage: `{summary['median_ticks']:.2f}` ticks",
                f"- Mean slippage: `{summary['mean_ticks']:.2f}` ticks",
                f"- p95 slippage: `{summary['p95_ticks']:.2f}` ticks",
                f"- Max slippage: `{summary['max_ticks']:.2f}` ticks",
                f"- Percent <= 1 tick: `{summary['pct_le_1_tick']:.1f}%`",
                f"- Percent <= 2 ticks: `{summary['pct_le_2_ticks']:.1f}%`",
            ]
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- This measures fill-quality / cost realism only; it is not alpha discovery.",
            "- Repricing delegates to `research.databento_microstructure.reprice_e2_entry`.",
            "- ORB timing delegates to `pipeline.build_daily_features._orb_utc_window`.",
            '- Cost assumptions come from `pipeline.cost_model.get_cost_spec("MES")`.',
            "",
            "## Follow-up rule",
            "",
            "- `PASS`: close the MES portion of `cost-realism-slippage-pilot`.",
            "- `WARN`: keep the debt open for session/tail review before cost-model changes.",
            "- `FAIL`: keep the debt open and review MES cost assumptions before trusting MES ExpR.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="MES E2 TBBO slippage validation pilot")
    parser.add_argument("--estimate-cost", action="store_true")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--reprice", action="store_true")
    parser.add_argument(
        "--reprice-cache", action="store_true", help="Reprice cached MES TBBO files; no Databento spend."
    )
    parser.add_argument("--all", action="store_true", help="Fresh-pull flow: manifest + cost + pull + reprice.")
    parser.add_argument("--write-result-doc", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sessions", nargs="+", default=None, metavar="SESSION")
    args = parser.parse_args()

    if args.sessions:
        global PILOT_SESSIONS
        PILOT_SESSIONS = _validate_pilot_sessions(list(args.sessions))
        print(f"PILOT_SESSIONS override: {PILOT_SESSIONS}")

    if args.reprice_cache:
        manifest_cache = build_manifest_from_cache(CACHE_DIR)
        valid_manifest_n = sum(1 for row in manifest_cache if row["error"] is None)
        print(f"Cache manifest: {len(manifest_cache)} entries, {valid_manifest_n} valid")
        results_df = pd.DataFrame(reprice_cache_manifest(manifest_cache))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CACHE_DIR / "slippage_results_cache_v1.csv"
        results_df.to_csv(out_path, index=False)
        print(f"Results saved: {out_path}")
        analyze_results(results_df)
        if args.write_result_doc:
            RESULT_DOC.write_text(build_result_doc(results_df, source_csv=out_path), encoding="utf-8")
            print(f"Result doc saved: {RESULT_DOC}")
        return

    manifest = build_pilot_manifest(seed=args.seed)
    if not manifest:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps([asdict(day) for day in manifest], indent=2), encoding="utf-8")
    print(f"Manifest saved: {manifest_path}")

    if args.estimate_cost or args.all:
        estimate_cost(manifest)
    if args.pull or args.all:
        pull_tbbo(manifest, force=args.force)
    if args.reprice or args.all:
        results_df = reprice_entries(manifest)
        results_path = CACHE_DIR / "slippage_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Results saved: {results_path}")
        analyze_results(results_df)
        if args.write_result_doc:
            RESULT_DOC.write_text(build_result_doc(results_df, source_csv=results_path), encoding="utf-8")
            print(f"Result doc saved: {RESULT_DOC}")


if __name__ == "__main__":
    main()
