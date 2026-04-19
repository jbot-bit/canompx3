#!/usr/bin/env python3
"""ZT CPI/NFP event-study first pass. 2026-03-15."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

import databento as db
import pandas as pd

from pipeline.asset_configs import get_asset_config
from pipeline.calendar_filters import build_cpi_set, is_nfp_day

NY_TZ = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
OUT_DIR = Path(__file__).parent / "output"

WINDOWS = {
    "5m": (time(8, 35), time(8, 39, 59)),
    "10m": (time(8, 35), time(8, 44, 59)),
    "15m": (time(8, 35), time(8, 49, 59)),
}
PRE_WINDOW = (time(8, 20), time(8, 29, 59))
SHOCK_WINDOW = (time(8, 30), time(8, 34, 59))


@dataclass
class WindowSnapshot:
    open_price: float
    close_price: float


def to_utc(ts_date: date, ts_time: time) -> datetime:
    return datetime.combine(ts_date, ts_time, tzinfo=NY_TZ).astimezone(UTC)


def build_nfp_set(start_date: date, end_date: date) -> set[date]:
    out = set()
    d = start_date
    while d <= end_date:
        if is_nfp_day(d):
            out.add(d)
        d += timedelta(days=1)
    return out


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
    df = df.sort_index()
    return df


def get_window_snapshot(df: pd.DataFrame, start_utc: datetime, end_utc: datetime) -> WindowSnapshot | None:
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
        "fw_5m_close",
        "fw_10m_close",
        "fw_15m_close",
    ):
        vals = rows[col].dropna().tolist()
        values.extend(float(v) for v in vals)
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


def run_study(dbn_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    files = list_daily_files(dbn_dir)
    if not files:
        raise SystemExit(f"FATAL: no DBN files found in {dbn_dir}")

    start_date = min(files)
    end_date = max(files)
    cpi_dates = build_cpi_set()
    nfp_dates = build_nfp_set(start_date, end_date)

    event_dates: list[tuple[str, date]] = []
    for d in sorted(cpi_dates):
        if start_date <= d <= end_date:
            event_dates.append(("CPI", d))
    for d in sorted(nfp_dates):
        if start_date <= d <= end_date:
            event_dates.append(("NFP", d))
    event_dates.sort(key=lambda x: (x[1], x[0]))

    rows: list[dict[str, object]] = []
    for family, event_date in event_dates:
        path = files.get(event_date)
        row: dict[str, object] = {
            "event_date": event_date.isoformat(),
            "event_family": family,
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

        pre = get_window_snapshot(df, to_utc(event_date, PRE_WINDOW[0]), to_utc(event_date, PRE_WINDOW[1]))
        shock = get_window_snapshot(df, to_utc(event_date, SHOCK_WINDOW[0]), to_utc(event_date, SHOCK_WINDOW[1]))
        follow = {
            key: get_window_snapshot(df, to_utc(event_date, start_t), to_utc(event_date, end_t))
            for key, (start_t, end_t) in WINDOWS.items()
        }

        if pre is None or shock is None or any(v is None for v in follow.values()):
            row["exclusion_reason"] = "missing_window_data"
            rows.append(row)
            continue

        shock_move = shock.close_price - shock.open_price
        shock_direction = 1 if shock_move > 0 else -1 if shock_move < 0 else 0

        row.update(
            {
                "pre_event_open": pre.open_price,
                "pre_event_close": pre.close_price,
                "shock_open": shock.open_price,
                "shock_close": shock.close_price,
                "shock_direction": shock_direction,
                "shock_magnitude": abs(shock_move),
            }
        )
        for key, snapshot in follow.items():
            if snapshot is None:
                continue
            follow_close = snapshot.close_price
            follow_move = follow_close - shock.close_price
            cont = shock_direction != 0 and follow_move * shock_direction > 0
            rev = shock_direction != 0 and follow_move * shock_direction < 0
            row[f"fw_{key}_close"] = follow_close
            row[f"follow_{key}_move"] = follow_move
            row[f"cont_{key}"] = cont
            row[f"rev_{key}"] = rev

        row["usable_event_flag"] = shock_direction != 0
        row["exclusion_reason"] = "zero_shock" if shock_direction == 0 else None
        rows.append(row)

    events = pd.DataFrame(rows)
    tick_size = infer_tick_size(events[events["usable_event_flag"] == True])  # noqa: E712

    summary_rows: list[dict[str, object]] = []
    usable = events[events["usable_event_flag"] == True].copy()  # noqa: E712
    for family in ("CPI", "NFP"):
        fam = usable[usable["event_family"] == family].copy()
        for window_key in WINDOWS:
            follow_col = f"follow_{window_key}_move"
            for model, hit_col, sign in (
                ("continuation", f"cont_{window_key}", 1.0),
                ("failed_first_move", f"rev_{window_key}", -1.0),
            ):
                subset = fam[[follow_col, hit_col, "event_date"]].dropna().copy()
                n = len(subset)
                if n == 0:
                    continue
                subset["model_move"] = sign * subset[follow_col]
                hits = int(subset[hit_col].sum())
                avg_move = mean(subset["model_move"])
                med_move = median(subset["model_move"])
                hit_rate = hits / n
                p_value = binomial_two_sided_p(hits, n)
                avg_ticks = avg_move / tick_size if tick_size else float("nan")
                summary_rows.append(
                    {
                        "event_family": family,
                        "window": window_key,
                        "model": model,
                        "n": n,
                        "hits": hits,
                        "hit_rate": hit_rate,
                        "avg_signed_move": avg_move,
                        "median_signed_move": med_move,
                        "avg_signed_ticks": avg_ticks,
                        "p_value": p_value,
                        "first_second_half_hit_rate": first_second_half_hit(subset, hit_col),
                        "friction_sanity": "PASS" if tick_size and avg_ticks > 2.0 else "THIN/FAIL",
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values(["event_family", "model", "window"])
    return events, summary, tick_size


def build_markdown(events: pd.DataFrame, summary: pd.DataFrame, tick_size: float | None) -> str:
    usable = events[events["usable_event_flag"] == True]  # noqa: E712
    excluded = events[events["usable_event_flag"] != True]  # noqa: E712
    survived: list[str] = []
    failed: list[str] = []
    for row in summary.to_dict("records"):
        line = (
            f"{row['event_family']} {row['model']} {row['window']}: "
            f"N={row['n']}, hit_rate={row['hit_rate']:.1%}, "
            f"avg_move={row['avg_signed_move']:.6f}, p={row['p_value']:.4f}, "
            f"avg_ticks={row['avg_signed_ticks']:.2f}, friction={row['friction_sanity']}"
        )
        if row["p_value"] < 0.05 and row["avg_signed_move"] > 0 and row["friction_sanity"] == "PASS":
            survived.append(line)
        else:
            failed.append(line)

    lines = [
        "# ZT CPI/NFP Event Study",
        "",
        "## Scope",
        "",
        "- Instrument: `ZT`",
        "- Event families: `CPI`, `NFP`",
        "- Models: `continuation`, `failed_first_move`",
        "- Follow-through windows: `5m`, `10m`, `15m`",
        "",
        "## Data",
        "",
        f"- Usable events: {len(usable)}",
        f"- Excluded events: {len(excluded)}",
        f"- Date range: {events['event_date'].min()} to {events['event_date'].max()}",
        f"- Inferred tick size: {tick_size if tick_size is not None else 'n/a'}",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No usable summary rows.")
    else:
        lines.append(
            "| family | model | window | n | hits | hit_rate | avg_move | median_move | avg_ticks | p_value | first/second half | friction |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in summary.to_dict("records"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["event_family"]),
                        str(row["model"]),
                        str(row["window"]),
                        str(row["n"]),
                        str(row["hits"]),
                        f"{row['hit_rate']:.1%}",
                        f"{row['avg_signed_move']:.6f}",
                        f"{row['median_signed_move']:.6f}",
                        f"{row['avg_signed_ticks']:.2f}",
                        f"{row['p_value']:.4f}",
                        str(row["first_second_half_hit_rate"]),
                        str(row["friction_sanity"]),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("SURVIVED SCRUTINY:")
    lines.extend([f"- {line}" for line in survived] if survived else ["- None"])
    lines.append("")
    lines.append("DID NOT SURVIVE:")
    lines.extend([f"- {line}" for line in failed] if failed else ["- None"])
    lines.extend(
        [
            "",
            "CAVEATS:",
            "- In-sample only. No OOS / walk-forward validation in this pass.",
            "- Friction sanity is a screening heuristic based on inferred tick size, not a full cost model.",
            "- This study uses raw event windows, not portfolio-level diversification tests.",
            "",
            "NEXT STEPS:",
            "- If anything survives, compare against `2YY` on the same event families.",
            "- If nothing survives, stop the `ZT` rates path before widening the search.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZT CPI/NFP event-study first pass")
    parser.add_argument("--dbn-dir", type=Path, default=None, help="Directory containing ZT .dbn.zst daily files")
    parser.add_argument("--output-prefix", type=Path, default=OUT_DIR / "zt_cpi_nfp", help="Output file prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dbn_dir = args.dbn_dir or get_asset_config("ZT")["dbn_path"]
    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    events, summary, tick_size = run_study(dbn_dir)

    events_path = out_prefix.with_name(out_prefix.name + "_events.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    report_path = out_prefix.with_name(out_prefix.name + "_findings.md")

    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(build_markdown(events, summary, tick_size), encoding="utf-8")

    print(f"Wrote {events_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
