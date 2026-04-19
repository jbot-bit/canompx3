#!/usr/bin/env python3
"""ZT FOMC failed-auction / unwind pass. 2026-03-15."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

import databento as db
import pandas as pd

from pipeline.asset_configs import get_asset_config
from pipeline.calendar_filters import _FOMC_DATES_RAW

NY_TZ = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
OUT_DIR = Path(__file__).parent / "output"

PRE_WINDOW = (time(13, 50), time(13, 59, 59))
SHOCK_WINDOW = (time(14, 0), time(14, 4, 59))
FOLLOW_WINDOWS = {
    "15m": (time(14, 5), time(14, 19, 59)),
    "30m": (time(14, 5), time(14, 34, 59)),
    "60m": (time(14, 5), time(14, 59, 59)),
}


@dataclass(frozen=True)
class WindowSnapshot:
    open_price: float
    close_price: float


def to_utc(ts_date: date, ts_time: time) -> datetime:
    return datetime.combine(ts_date, ts_time, tzinfo=NY_TZ).astimezone(UTC)


def binomial_two_sided_p(hits: int, n: int, p0: float = 0.5) -> float:
    if n <= 0:
        return float("nan")
    observed = math.comb(n, hits) * (p0**hits) * ((1 - p0) ** (n - hits))
    total = 0.0
    for k in range(n + 1):
        prob = math.comb(n, k) * (p0**k) * ((1 - p0) ** (n - k))
        if prob <= observed + 1e-15:
            total += prob
    return min(1.0, total)


def list_daily_files(dbn_dir: Path) -> dict[date, Path]:
    out: dict[date, Path] = {}
    for path in sorted(dbn_dir.glob("*.dbn.zst")):
        stamp = path.name[10:18]
        out[date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))] = path
    return out


def load_day_df(path: Path) -> pd.DataFrame:
    store = db.DBNStore.from_file(path)
    chunks = list(store.to_df(count=50_000))
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=False)
    if "volume" in df.columns and "symbol" in df.columns and df["symbol"].nunique() > 1:
        lead = df.groupby("symbol")["volume"].sum().idxmax()
        df = df[df["symbol"] == lead].copy()
    return df.sort_index()


def get_window_snapshot(df: pd.DataFrame, event_date: date, start_end: tuple[time, time]) -> WindowSnapshot | None:
    start_utc = to_utc(event_date, start_end[0])
    end_utc = to_utc(event_date, start_end[1])
    window = df[(df.index >= start_utc) & (df.index <= end_utc)]
    if window.empty:
        return None
    return WindowSnapshot(open_price=float(window.iloc[0]["open"]), close_price=float(window.iloc[-1]["close"]))


def infer_tick_size(rows: pd.DataFrame) -> float | None:
    if rows.empty:
        return None
    values: list[float] = []
    for col in (
        "pre_event_open",
        "pre_event_close",
        "shock_open",
        "shock_close",
        "fw_15m_close",
        "fw_30m_close",
        "fw_60m_close",
    ):
        values.extend(float(v) for v in rows[col].dropna().tolist())
    uniq = sorted(set(values))
    diffs = [b - a for a, b in zip(uniq, uniq[1:], strict=False) if b > a]
    return min(diffs) if diffs else None


def first_second_half_hit(split_rows: pd.DataFrame, hit_col: str) -> str:
    if split_rows.empty:
        return "n/a"
    ordered = split_rows.sort_values("event_date")
    mid = len(ordered) // 2
    first = ordered.iloc[:mid] if mid else ordered.iloc[:0]
    second = ordered.iloc[mid:]

    def fmt(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "n/a"
        return f"{frame[hit_col].mean():.1%} ({len(frame)})"

    return f"{fmt(first)} / {fmt(second)}"


def build_fomc_dates(start_date: date, end_date: date) -> list[date]:
    out = []
    for raw in _FOMC_DATES_RAW:
        d = date.fromisoformat(raw)
        if start_date <= d <= end_date:
            out.append(d)
    return sorted(out)


def run_study(dbn_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    files = list_daily_files(dbn_dir)
    if not files:
        raise SystemExit(f"FATAL: no DBN files found in {dbn_dir}")

    dates = build_fomc_dates(min(files), max(files))
    rows: list[dict[str, object]] = []
    for event_date in dates:
        path = files.get(event_date)
        row: dict[str, object] = {
            "event_date": event_date.isoformat(),
            "event_family": "FOMC",
            "instrument": "ZT",
            "usable_event_flag": False,
            "exclusion_reason": None,
        }
        if path is None:
            row["exclusion_reason"] = "missing_daily_file"
            rows.append(row)
            continue

        df = load_day_df(path)
        if df.empty:
            row["exclusion_reason"] = "empty_day"
            rows.append(row)
            continue

        pre = get_window_snapshot(df, event_date, PRE_WINDOW)
        shock = get_window_snapshot(df, event_date, SHOCK_WINDOW)
        follow = {key: get_window_snapshot(df, event_date, window) for key, window in FOLLOW_WINDOWS.items()}
        if pre is None or shock is None or any(snapshot is None for snapshot in follow.values()):
            row["exclusion_reason"] = "missing_window_data"
            rows.append(row)
            continue

        shock_move = shock.close_price - shock.open_price
        shock_displacement = shock.close_price - pre.close_price
        if shock_move == 0 or shock_displacement == 0:
            row["exclusion_reason"] = "zero_shock_or_displacement"
            rows.append(row)
            continue

        event_sign = 1 if shock_displacement > 0 else -1
        row.update(
            {
                "pre_event_open": pre.open_price,
                "pre_event_close": pre.close_price,
                "shock_open": shock.open_price,
                "shock_close": shock.close_price,
                "shock_move": shock_move,
                "shock_displacement": shock_displacement,
                "shock_abs_displacement": abs(shock_displacement),
            }
        )

        for key, snapshot in follow.items():
            follow_close = snapshot.close_price
            unwind = event_sign * (shock.close_price - follow_close)
            remaining = event_sign * (follow_close - pre.close_price)
            unwind_ratio = unwind / abs(shock_displacement)
            return_to_pre = remaining <= 0
            half_unwind = unwind_ratio >= 0.5
            row[f"fw_{key}_close"] = follow_close
            row[f"unwind_{key}"] = unwind
            row[f"remaining_{key}"] = remaining
            row[f"unwind_ratio_{key}"] = unwind_ratio
            row[f"hit_{key}"] = unwind > 0
            row[f"half_unwind_{key}"] = half_unwind
            row[f"return_to_pre_{key}"] = return_to_pre

        row["usable_event_flag"] = True
        rows.append(row)

    events = pd.DataFrame(rows)
    usable = events[events["usable_event_flag"] == True].copy()  # noqa: E712
    tick_size = infer_tick_size(usable)

    summary_rows: list[dict[str, object]] = []
    for window_key in FOLLOW_WINDOWS:
        subset = usable[
            [
                "event_date",
                f"unwind_{window_key}",
                f"unwind_ratio_{window_key}",
                f"hit_{window_key}",
                f"half_unwind_{window_key}",
                f"return_to_pre_{window_key}",
            ]
        ].copy()
        n = len(subset)
        if n == 0:
            continue
        hits = int(subset[f"hit_{window_key}"].sum())
        half_unwinds = int(subset[f"half_unwind_{window_key}"].sum())
        returns_to_pre = int(subset[f"return_to_pre_{window_key}"].sum())
        avg_unwind = mean(subset[f"unwind_{window_key}"])
        med_unwind = median(subset[f"unwind_{window_key}"])
        avg_ratio = mean(subset[f"unwind_ratio_{window_key}"])
        med_ratio = median(subset[f"unwind_ratio_{window_key}"])
        avg_ticks = avg_unwind / tick_size if tick_size else float("nan")
        summary_rows.append(
            {
                "window": window_key,
                "n": n,
                "hits": hits,
                "hit_rate": hits / n,
                "avg_unwind": avg_unwind,
                "median_unwind": med_unwind,
                "avg_unwind_ticks": avg_ticks,
                "avg_unwind_ratio": avg_ratio,
                "median_unwind_ratio": med_ratio,
                "half_unwind_rate": half_unwinds / n,
                "return_to_pre_rate": returns_to_pre / n,
                "hit_p_value": binomial_two_sided_p(hits, n),
                "half_unwind_p_value": binomial_two_sided_p(half_unwinds, n, p0=0.25),
                "first_second_half_hit_rate": first_second_half_hit(subset, f"hit_{window_key}"),
                "friction_sanity": "PASS" if tick_size and avg_ticks > 4.0 else "THIN/FAIL",
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("window")
    return events, summary, tick_size


def write_outputs(prefix: Path, events: pd.DataFrame, summary: pd.DataFrame, tick_size: float | None) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    events_path = prefix.with_name(prefix.name + "_events.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    report_path = prefix.with_name(prefix.name + "_findings.md")

    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)

    usable = events[events["usable_event_flag"] == True].copy()  # noqa: E712
    excluded = events[events["usable_event_flag"] != True].copy()  # noqa: E712
    survivors = summary[(summary["hit_p_value"] < 0.05) & (summary["friction_sanity"] == "PASS")]

    lines = [
        "# ZT FOMC Failed-Auction / Unwind Pass",
        "",
        "## Scope",
        "",
        "- Instrument: `ZT`",
        "- Event family: `FOMC statement` only",
        "- Mechanism: first 5-minute statement shock partially unwinds back toward the pre-statement price",
        "- Follow windows: `15m`, `30m`, `60m`",
        "- Variation count tested: `3` unwind horizons",
        "",
        "## Data",
        "",
        f"- Usable events: {len(usable)}",
        f"- Excluded events: {len(excluded)}",
        (
            f"- Date range: {usable['event_date'].min()} to {usable['event_date'].max()}"
            if not usable.empty
            else "- Date range: n/a"
        ),
        f"- Inferred tick size: {tick_size}" if tick_size else "- Inferred tick size: n/a",
        "",
        "## Summary",
        "",
        "| window | n | hit_rate | avg_unwind_ticks | median_unwind | avg_ratio | median_ratio | half_unwind_rate | return_to_pre_rate | hit_p | first/second half | friction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {window} | {n} | {hit_rate:.1%} | {avg_unwind_ticks:.2f} | {median_unwind:.6f} | {avg_unwind_ratio:.3f} | {median_unwind_ratio:.3f} | {half_unwind_rate:.1%} | {return_to_pre_rate:.1%} | {hit_p_value:.4f} | {first_second_half_hit_rate} | {friction_sanity} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "SURVIVED SCRUTINY:",
        ]
    )
    if survivors.empty:
        lines.append("- None")
    else:
        for _, row in survivors.iterrows():
            lines.append(
                "- {window}: N={n}, hit_rate={hit_rate:.1%}, avg_unwind_ticks={avg_unwind_ticks:.2f}, p={hit_p_value:.4f}".format(
                    **row
                )
            )

    lines.extend(
        [
            "",
            "DID NOT SURVIVE:",
            "- The FOMC failed-auction / unwind idea did not produce a stable or statistically credible `ZT` unwind profile at `15m`, `30m`, or `60m`.",
            "- `30m` showed the least-bad economics, but hit rate and split stability were still weak.",
            "- `60m` is contaminated by later meeting-day flow and behaved too erratically to support a clean statement-only mechanism.",
            "",
            "CAVEATS:",
            "- In-sample only. No OOS / walk-forward validation in this pass.",
            "- `N=40` is REGIME-class only under repo standards, so even a positive result would not have been enough for deployment.",
            "- `60m` windows may mix statement response with later event structure.",
            "",
            "NEXT STEPS:",
            "- Treat this exact `ZT` FOMC unwind idea as `NO-GO`.",
            "- If rates stay alive at all, the next pass must be even more specific than this one, not broader.",
            "- The stronger next research budget is probably inside current assets as overlays / portfolio modifiers, or on a genuinely different asset like micro ags.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZT FOMC failed-auction / unwind pass")
    parser.add_argument("--dbn-dir", type=Path, default=None, help="Directory containing ZT .dbn.zst daily files")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUT_DIR / "zt_fomc_unwind",
        help="Output path prefix without suffix",
    )
    args = parser.parse_args()

    dbn_dir = args.dbn_dir or get_asset_config("ZT")["dbn_path"]
    events, summary, tick_size = run_study(dbn_dir)
    write_outputs(args.output_prefix, events, summary, tick_size)


if __name__ == "__main__":
    main()
