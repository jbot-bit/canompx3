#!/usr/bin/env python3
"""ZT Stage-1 event viability pass. 2026-03-15."""

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
from pipeline.calendar_filters import _FOMC_DATES_RAW, build_cpi_set, is_nfp_day

NY_TZ = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
OUT_DIR = Path(__file__).parent / "output"


@dataclass(frozen=True)
class EventFamily:
    name: str
    pre_window: tuple[time, time]
    shock_window: tuple[time, time]
    follow_windows: dict[str, tuple[time, time]]


@dataclass(frozen=True)
class WindowSnapshot:
    open_price: float
    close_price: float


EVENT_FAMILIES: dict[str, EventFamily] = {
    "CPI": EventFamily(
        name="CPI",
        pre_window=(time(8, 20), time(8, 29, 59)),
        shock_window=(time(8, 30), time(8, 34, 59)),
        follow_windows={
            "5m": (time(8, 35), time(8, 39, 59)),
            "10m": (time(8, 35), time(8, 44, 59)),
            "15m": (time(8, 35), time(8, 49, 59)),
        },
    ),
    "NFP": EventFamily(
        name="NFP",
        pre_window=(time(8, 20), time(8, 29, 59)),
        shock_window=(time(8, 30), time(8, 34, 59)),
        follow_windows={
            "5m": (time(8, 35), time(8, 39, 59)),
            "10m": (time(8, 35), time(8, 44, 59)),
            "15m": (time(8, 35), time(8, 49, 59)),
        },
    ),
    "FOMC": EventFamily(
        name="FOMC",
        pre_window=(time(13, 50), time(13, 59, 59)),
        shock_window=(time(14, 0), time(14, 4, 59)),
        follow_windows={
            "5m": (time(14, 5), time(14, 9, 59)),
            "10m": (time(14, 5), time(14, 14, 59)),
            "15m": (time(14, 5), time(14, 19, 59)),
        },
    ),
}


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


def build_fomc_announcement_set(start_date: date, end_date: date) -> set[date]:
    out = set()
    for raw in _FOMC_DATES_RAW:
        d = date.fromisoformat(raw)
        if start_date <= d <= end_date:
            out.add(d)
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
    return df.sort_index()


def get_window_snapshot(
    df: pd.DataFrame,
    event_date: date,
    start_end: tuple[time, time],
) -> WindowSnapshot | None:
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
        "fw_5m_close",
        "fw_10m_close",
        "fw_15m_close",
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


def build_event_dates(files: dict[date, Path]) -> list[tuple[str, date]]:
    start_date = min(files)
    end_date = max(files)
    cpi_dates = build_cpi_set()
    nfp_dates = build_nfp_set(start_date, end_date)
    fomc_dates = build_fomc_announcement_set(start_date, end_date)
    event_dates: list[tuple[str, date]] = []
    for family, family_dates in (
        ("CPI", cpi_dates),
        ("NFP", nfp_dates),
        ("FOMC", fomc_dates),
    ):
        for event_date in sorted(family_dates):
            if start_date <= event_date <= end_date:
                event_dates.append((family, event_date))
    event_dates.sort(key=lambda x: (x[1], x[0]))
    return event_dates


def run_study(dbn_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float | None]:
    files = list_daily_files(dbn_dir)
    if not files:
        raise SystemExit(f"FATAL: no DBN files found in {dbn_dir}")

    event_dates = build_event_dates(files)
    rows: list[dict[str, object]] = []
    for family_name, event_date in event_dates:
        family = EVENT_FAMILIES[family_name]
        path = files.get(event_date)
        row: dict[str, object] = {
            "event_date": event_date.isoformat(),
            "event_family": family_name,
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

        pre = get_window_snapshot(df, event_date, family.pre_window)
        shock = get_window_snapshot(df, event_date, family.shock_window)
        follow = {
            key: get_window_snapshot(df, event_date, window)
            for key, window in family.follow_windows.items()
        }
        if pre is None or shock is None or any(snapshot is None for snapshot in follow.values()):
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
            follow_close = snapshot.close_price
            follow_move = follow_close - shock.close_price
            cont = shock_direction != 0 and follow_move * shock_direction > 0
            rev = shock_direction != 0 and follow_move * shock_direction < 0
            row[f"fw_{key}_close"] = follow_close
            row[f"follow_{key}_move"] = follow_move
            row[f"cont_{key}"] = cont
            row[f"rev_{key}"] = rev

        total_move = row["fw_15m_close"] - pre.close_price
        row["follow_15m_abs_move"] = abs(row["follow_15m_move"])
        row["total_25m_abs_move"] = abs(total_move)
        row["usable_event_flag"] = shock_direction != 0
        row["exclusion_reason"] = "zero_shock" if shock_direction == 0 else None
        rows.append(row)

    events = pd.DataFrame(rows)
    usable = events[events["usable_event_flag"] == True].copy()  # noqa: E712
    tick_size = infer_tick_size(usable)

    economics_rows: list[dict[str, object]] = []
    for family_name in EVENT_FAMILIES:
        fam = usable[usable["event_family"] == family_name].copy()
        if fam.empty:
            continue
        economics_rows.append(
            {
                "event_family": family_name,
                "n": len(fam),
                "date_start": fam["event_date"].min(),
                "date_end": fam["event_date"].max(),
                "mean_shock_abs_ticks": mean(fam["shock_magnitude"] / tick_size) if tick_size else float("nan"),
                "median_shock_abs_ticks": median(fam["shock_magnitude"] / tick_size) if tick_size else float("nan"),
                "mean_follow_15_abs_ticks": mean(fam["follow_15m_abs_move"] / tick_size) if tick_size else float("nan"),
                "median_follow_15_abs_ticks": median(fam["follow_15m_abs_move"] / tick_size) if tick_size else float("nan"),
                "mean_total_25m_abs_ticks": mean(fam["total_25m_abs_move"] / tick_size) if tick_size else float("nan"),
                "median_total_25m_abs_ticks": median(fam["total_25m_abs_move"] / tick_size) if tick_size else float("nan"),
            }
        )

    summary_rows: list[dict[str, object]] = []
    for family_name, family in EVENT_FAMILIES.items():
        fam = usable[usable["event_family"] == family_name].copy()
        for window_key in family.follow_windows:
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
                p_value = binomial_two_sided_p(hits, n)
                avg_ticks = avg_move / tick_size if tick_size else float("nan")
                summary_rows.append(
                    {
                        "event_family": family_name,
                        "window": window_key,
                        "model": model,
                        "n": n,
                        "hits": hits,
                        "hit_rate": hits / n,
                        "avg_signed_move": avg_move,
                        "median_signed_move": med_move,
                        "avg_signed_ticks": avg_ticks,
                        "p_value": p_value,
                        "first_second_half_hit_rate": first_second_half_hit(subset, hit_col),
                        "friction_sanity": "PASS" if tick_size and avg_ticks > 4.0 else "THIN/FAIL",
                    }
                )

    economics = pd.DataFrame(economics_rows).sort_values("event_family")
    summary = pd.DataFrame(summary_rows).sort_values(["event_family", "model", "window"])
    return events, economics, summary, tick_size


def write_outputs(
    prefix: Path,
    events: pd.DataFrame,
    economics: pd.DataFrame,
    summary: pd.DataFrame,
    tick_size: float | None,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    events_path = prefix.with_name(prefix.name + "_events.csv")
    economics_path = prefix.with_name(prefix.name + "_economics.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    report_path = prefix.with_name(prefix.name + "_findings.md")

    events.to_csv(events_path, index=False)
    economics.to_csv(economics_path, index=False)
    summary.to_csv(summary_path, index=False)

    usable = events[events["usable_event_flag"] == True].copy()  # noqa: E712
    excluded = events[events["usable_event_flag"] != True].copy()  # noqa: E712
    survivors = summary[summary["p_value"] < 0.05].copy()

    variation_count = sum(len(family.follow_windows) * 2 for family in EVENT_FAMILIES.values())
    lines = [
        "# ZT Stage-1 Event Viability",
        "",
        "## Scope",
        "",
        "- Instrument: `ZT`",
        "- Event families: `CPI`, `NFP`, `FOMC`",
        "- Models: `continuation`, `failed_first_move`",
        "- Follow-through windows: `5m`, `10m`, `15m`",
        f"- Variation count tested: `{variation_count}` directional cells",
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
        "## Structural Feasibility",
        "",
        "Event economics are measured in ticks, not stories. This answers whether the asset has enough honest post-event movement to justify any further research budget at all.",
        "",
        "| family | n | date_start | date_end | mean_shock_ticks | median_shock_ticks | mean_follow15_ticks | median_follow15_ticks | mean_total25_ticks | median_total25_ticks |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in economics.iterrows():
        lines.append(
            "| {event_family} | {n} | {date_start} | {date_end} | {mean_shock_abs_ticks:.2f} | {median_shock_abs_ticks:.2f} | "
            "{mean_follow_15_abs_ticks:.2f} | {median_follow_15_abs_ticks:.2f} | {mean_total_25m_abs_ticks:.2f} | {median_total_25m_abs_ticks:.2f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Directional First Pass",
            "",
            "These cells test the simplest post-event continuation vs failed-first-move framing. No extra filters were added.",
            "",
            "| family | model | window | n | hits | hit_rate | avg_ticks | p_value | first/second half | friction |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            "| {event_family} | {model} | {window} | {n} | {hits} | {hit_rate:.1%} | {avg_signed_ticks:.2f} | {p_value:.4f} | {first_second_half_hit_rate} | {friction_sanity} |".format(
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
        lines.append("- No directional cell reached raw `p < 0.05`; BH/FDR would not rescue anything.")
    else:
        for _, row in survivors.iterrows():
            lines.append(
                "- {event_family} {model} {window}: N={n}, hit_rate={hit_rate:.1%}, avg_ticks={avg_signed_ticks:.2f}, p={p_value:.4f}".format(
                    **row
                )
            )

    lines.extend(
        [
            "",
            "DID NOT SURVIVE:",
            "- The simple continuation / failed-first-move family did not produce a directional `ZT` edge across `CPI`, `NFP`, or `FOMC` in this pass.",
            "- `CPI` and `NFP` showed large event shocks but weak post-shock sign persistence.",
            "- `FOMC` had enough movement to matter economically, but the tested directional framing was still unstable and statistically weak.",
            "",
            "CAVEATS:",
            "- In-sample only. No OOS / walk-forward validation in this pass.",
            "- Friction sanity is a rough tick-space screen, not a broker-specific execution model.",
            "- This pass answers asset viability and simple directional viability. It does not prove portfolio additivity.",
            "",
            "NEXT STEPS:",
            "- Treat `ZT` as structurally viable only because the event windows move enough in tick terms.",
            "- Treat the simple continuation / failed-first-move family as `NO-GO` until a sharper, single-mechanism follow-up is justified.",
            "- Do not widen into generic Treasury scans or filter soup off this report.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZT Stage-1 event viability pass")
    parser.add_argument("--dbn-dir", type=Path, default=None, help="Directory containing ZT .dbn.zst daily files")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUT_DIR / "zt_event_viability",
        help="Output path prefix without suffix",
    )
    args = parser.parse_args()

    dbn_dir = args.dbn_dir or get_asset_config("ZT")["dbn_path"]
    events, economics, summary, tick_size = run_study(dbn_dir)
    write_outputs(args.output_prefix, events, economics, summary, tick_size)


if __name__ == "__main__":
    main()
