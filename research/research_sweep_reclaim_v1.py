#!/usr/bin/env python3
"""Sweep/reclaim v1 — narrow prior-day high/low reversal event study.

Research-only. Not a strategy backtest.

Locked by:
  docs/audit/hypotheses/2026-04-19-sweep-reclaim-v1.yaml
  docs/specs/level_interaction_v1.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.dst import orb_utc_window
from research.lib import bh_fdr, connect_db, resolve_level_reference, ttest_1s
from research.lib.level_interactions import classify_level_interaction

LOCKED_INSTRUMENTS = [symbol for symbol in ACTIVE_ORB_INSTRUMENTS if symbol in {"MES", "MGC", "MNQ"}]
LOCKED_SESSIONS = [
    "CME_PRECLOSE",
    "COMEX_SETTLE",
    "EUROPE_FLOW",
    "NYSE_OPEN",
    "TOKYO_OPEN",
    "US_DATA_1000",
]
LOCKED_LEVELS = {
    "prev_day_high": "below",
    "prev_day_low": "above",
}
COMMON_START = date(2019, 5, 6)
HOLDOUT_START = date(2026, 1, 1)
EVENT_WINDOW_MINUTES = 30
RECLAIM_LOOKAHEAD_BARS = 2
RESPONSE_HORIZON_BARS = 2
OUTPUT_DIR = Path("research/output")
OUTPUT_CSV = OUTPUT_DIR / "sweep_reclaim_v1_cells.csv"
OUTPUT_MD = Path("docs/audit/results/2026-04-19-sweep-reclaim-v1.md")


@dataclass(frozen=True)
class EventObservation:
    instrument: str
    session: str
    trading_day: date
    sample: str
    level_name: str
    signed_return: float
    hit: int
    interaction_bar_index: int
    reclaim_bar_index: int


def reclaim_direction(reference_side: str) -> int:
    if reference_side == "below":
        return -1
    if reference_side == "above":
        return 1
    raise ValueError(f"Unsupported reference_side: {reference_side}")


def signed_response(reclaim_close: float, future_close: float, atr_20: float, direction: int) -> float | None:
    if atr_20 is None or atr_20 <= 0:
        return None
    return direction * ((future_close - reclaim_close) / atr_20)


def load_feature_rows(con) -> pd.DataFrame:
    instruments_sql = ", ".join(f"'{symbol}'" for symbol in LOCKED_INSTRUMENTS)
    sql = f"""
    SELECT trading_day, symbol, atr_20,
           prev_day_high, prev_day_low, prev_day_close
    FROM daily_features
    WHERE orb_minutes = 5
      AND symbol IN ({instruments_sql})
      AND trading_day >= ?
    ORDER BY symbol, trading_day
    """
    df = con.execute(sql, [COMMON_START]).fetchdf()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    return df


def load_bars_for_day(con, symbol: str, trading_day: date) -> pd.DataFrame:
    from pipeline.dst import compute_trading_day_utc_range

    day_start, day_end = compute_trading_day_utc_range(trading_day)
    sql = """
    SELECT ts_utc, open, high, low, close
    FROM bars_1m
    WHERE symbol = ?
      AND ts_utc >= ?
      AND ts_utc < ?
    ORDER BY ts_utc
    """
    df = con.execute(sql, [symbol, day_start, day_end]).fetchdf()
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def session_window(bars_day: pd.DataFrame, trading_day: date, session: str) -> pd.DataFrame:
    start_utc, _ = orb_utc_window(trading_day, session, 1)
    end_utc = start_utc + timedelta(minutes=EVENT_WINDOW_MINUTES)
    return bars_day[(bars_day["ts_utc"] >= pd.Timestamp(start_utc)) & (bars_day["ts_utc"] < pd.Timestamp(end_utc))].reset_index(drop=True)


def collect_events() -> list[EventObservation]:
    events: list[EventObservation] = []
    bar_cache: dict[tuple[str, date], pd.DataFrame] = {}

    with connect_db() as con:
        feature_rows = load_feature_rows(con)
        total_rows = len(feature_rows)

        for idx, row in enumerate(feature_rows.itertuples(index=False), start=1):
            if idx % 500 == 0:
                print(f"Progress: {idx}/{total_rows} feature rows", flush=True)

            symbol = row.symbol
            trading_day = row.trading_day
            atr_20 = float(row.atr_20) if row.atr_20 is not None else None
            sample = "IS" if trading_day < HOLDOUT_START else "OOS"

            cache_key = (symbol, trading_day)
            if cache_key not in bar_cache:
                bar_cache[cache_key] = load_bars_for_day(con, symbol, trading_day)
            bars_day = bar_cache[cache_key]
            if bars_day.empty:
                continue

            feature_map = {
                "prev_day_high": row.prev_day_high,
                "prev_day_low": row.prev_day_low,
                "prev_day_close": row.prev_day_close,
            }

            for session in LOCKED_SESSIONS:
                bars_window = session_window(bars_day, trading_day, session)
                if len(bars_window) <= RESPONSE_HORIZON_BARS:
                    continue

                for level_name, reference_side in LOCKED_LEVELS.items():
                    ref = resolve_level_reference(feature_map, level_name, target_session=session)
                    if ref.unavailable_reason is not None or ref.price is None:
                        continue

                    event = classify_level_interaction(
                        bars_window,
                        level_name=level_name,
                        level_price=ref.price,
                        reference_side=reference_side,
                        sweep_epsilon=0.0,
                        reclaim_lookahead_bars=RECLAIM_LOOKAHEAD_BARS,
                    )
                    if event.unavailable_reason is not None:
                        continue
                    if event.interaction_kind != "close_through" or not event.swept or not event.reclaimed:
                        continue
                    if event.reclaim_bar_index is None:
                        continue
                    future_idx = event.reclaim_bar_index + RESPONSE_HORIZON_BARS
                    if future_idx >= len(bars_window):
                        continue

                    reclaim_close = float(bars_window.iloc[event.reclaim_bar_index]["close"])
                    future_close = float(bars_window.iloc[future_idx]["close"])
                    direction = reclaim_direction(reference_side)
                    signed_ret = signed_response(reclaim_close, future_close, atr_20, direction)
                    if signed_ret is None:
                        continue

                    events.append(
                        EventObservation(
                            instrument=symbol,
                            session=session,
                            trading_day=trading_day,
                            sample=sample,
                            level_name=level_name,
                            signed_return=float(signed_ret),
                            hit=int(signed_ret > 0),
                            interaction_bar_index=int(event.bar_index),
                            reclaim_bar_index=int(event.reclaim_bar_index),
                        )
                    )

    return events


def summarize_events(events: list[EventObservation]) -> pd.DataFrame:
    rows = [event.__dict__ for event in events]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    cell_index = [
        (instrument, session, level_name)
        for instrument in LOCKED_INSTRUMENTS
        for session in LOCKED_SESSIONS
        for level_name in LOCKED_LEVELS
    ]

    summary_rows = []
    pvals = []
    for instrument, session, level_name in cell_index:
        sub_is = df[
            (df["instrument"] == instrument)
            & (df["session"] == session)
            & (df["level_name"] == level_name)
            & (df["sample"] == "IS")
        ]
        sub_oos = df[
            (df["instrument"] == instrument)
            & (df["session"] == session)
            & (df["level_name"] == level_name)
            & (df["sample"] == "OOS")
        ]

        is_vals = sub_is["signed_return"].to_numpy(dtype=float)
        oos_vals = sub_oos["signed_return"].to_numpy(dtype=float)
        n_is, mean_is, wr_is, t_is, p_is = ttest_1s(is_vals)
        n_oos, mean_oos, wr_oos, t_oos, p_oos = ttest_1s(oos_vals)
        pvals.append(1.0 if n_is < 30 or pd.isna(p_is) else float(p_is))

        summary_rows.append(
            {
                "instrument": instrument,
                "session": session,
                "level_name": level_name,
                "n_is": int(n_is),
                "avg_is": None if pd.isna(mean_is) else float(mean_is),
                "wr_is": None if pd.isna(wr_is) else float(wr_is),
                "t_is": None if pd.isna(t_is) else float(t_is),
                "p_is": None if pd.isna(p_is) else float(p_is),
                "n_oos": int(n_oos),
                "avg_oos": None if pd.isna(mean_oos) else float(mean_oos),
                "wr_oos": None if pd.isna(wr_oos) else float(wr_oos),
                "t_oos": None if pd.isna(t_oos) else float(t_oos),
                "p_oos": None if pd.isna(p_oos) else float(p_oos),
                "dir_match_oos": bool((mean_is > 0 and mean_oos > 0) or (mean_is < 0 and mean_oos < 0))
                if (not pd.isna(mean_is) and not pd.isna(mean_oos) and n_oos >= 20)
                else None,
            }
        )

    rejected = bh_fdr(pvals, q=0.05)
    for idx, row in enumerate(summary_rows):
        row["p_for_bh"] = pvals[idx]
        row["bh_survivor"] = idx in rejected
        row["passes_primary"] = bool(
            row["bh_survivor"]
            and row["n_is"] >= 50
            and row["avg_is"] is not None
            and row["avg_is"] > 0
        )

    return pd.DataFrame(summary_rows)


def write_outputs(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    survivors = summary[summary["passes_primary"]].copy()
    survivors = survivors.sort_values(["avg_is", "n_is"], ascending=[False, False])

    lines = [
        "# Sweep Reclaim V1",
        "",
        "Research-only event study locked by `docs/audit/hypotheses/2026-04-19-sweep-reclaim-v1.yaml`.",
        "",
        "## Scope",
        "",
        f"- Instruments: {', '.join(LOCKED_INSTRUMENTS)}",
        f"- Sessions: {', '.join(LOCKED_SESSIONS)}",
        "- Levels: prev_day_high / prev_day_low only",
        "- Event definition: swept close-through that reclaims back inside within 2 bars",
        f"- Event window: first {EVENT_WINDOW_MINUTES} minutes of the session",
        f"- Response metric: signed next-{RESPONSE_HORIZON_BARS}-bar close-to-close return from reclaim close normalized by ATR20",
        "- Selection uses pre-2026 only; 2026 is diagnostic OOS only",
        "",
        "## Family Verdict",
        "",
        f"- Locked family K: {len(summary)}",
        f"- Primary survivors (BH + N>=50 + avg_is>0): {len(survivors)}",
        "",
    ]

    if survivors.empty:
        warm = summary[
            (summary["avg_is"].notna())
            & (summary["avg_is"] > 0)
            & (summary["n_is"] >= 20)
        ].copy()
        warm = warm.sort_values(["p_is", "avg_is"], ascending=[True, False]).head(5)
        lines.extend(
            [
                "No primary survivors.",
                "",
                "The family remains useful as a cleaner trapped-side re-test, but this narrow sweep-reclaim pass did not surface a clear positive cell under the locked standards.",
            ]
        )
        if not warm.empty:
            lines.extend(["", "## Warm Cells (Informational Only)", ""])
            for row in warm.itertuples(index=False):
                lines.append(
                    f"- {row.instrument} {row.session} {row.level_name}: "
                    f"IS n={row.n_is}, avg={row.avg_is:+.4f}, WR={row.wr_is:.1%}, "
                    f"t={row.t_is:.2f}, p={row.p_is:.4f}, BH_survivor={row.bh_survivor}"
                )
    else:
        lines.extend(["## Primary Survivors", ""])
        for row in survivors.itertuples(index=False):
            oos_note = (
                f"OOS n={row.n_oos}, avg={row.avg_oos:+.4f}, dir_match={row.dir_match_oos}"
                if row.n_oos > 0 and row.avg_oos is not None
                else "OOS sparse or unavailable"
            )
            lines.append(
                f"- {row.instrument} {row.session} {row.level_name}: "
                f"IS n={row.n_is}, avg={row.avg_is:+.4f}, WR={row.wr_is:.1%}, "
                f"t={row.t_is:.2f}, p={row.p_is:.4f}; {oos_note}"
            )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is not a trade strategy or deployability result.",
            "- No costs or trade geometry are applied; this is a short-horizon reversal event study only.",
            "- Protocol B skepticism applies because exact sweep-reclaim theory is not separately grounded in local literature resources.",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    events = collect_events()
    summary = summarize_events(events)
    if summary.empty:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(OUTPUT_CSV, index=False)
        OUTPUT_MD.write_text(
            "# Sweep Reclaim V1\n\nNo events were collected under the locked scope.",
            encoding="utf-8",
        )
        print("No events collected under locked scope.")
        return 0

    write_outputs(summary)
    primary = summary[summary["passes_primary"]]
    print(summary.sort_values(["passes_primary", "avg_is"], ascending=[False, False]).to_string(index=False))
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_MD}")
    print(f"Primary survivors: {len(primary)} / {len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
