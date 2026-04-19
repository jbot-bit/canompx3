#!/usr/bin/env python3
"""Path-accurate native MGC sub-R audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec, pnl_points_to_r, to_r_multiple
from research.lib import bh_fdr, connect_db, ttest_1s, write_csv
from research.research_mgc_payoff_compression_audit import passes_filter

OVERLAP_START = "2022-06-13"
HOLDOUT_START = pd.Timestamp("2026-01-01")
BH_Q = 0.10
MIN_N = 50
RESULT_PATH = Path("docs/audit/results/2026-04-19-mgc-path-accurate-subr-v1.md")
OUTPUT_PREFIX = "mgc_path_accurate_subr_v1"
MGC_SPEC = get_cost_spec("MGC")


@dataclass(frozen=True)
class Candidate:
    family_id: str
    orb_label: str
    filter_type: str | None
    target_r: float
    family_kind: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("NYSE_OPEN_OVNRNG_50_RR1", "NYSE_OPEN", "OVNRNG_50", 0.75, "warm"),
    Candidate("US_DATA_1000_ATR_P70_RR1", "US_DATA_1000", "ATR_P70", 0.5, "warm"),
    Candidate("US_DATA_1000_OVNRNG_10_RR1", "US_DATA_1000", "OVNRNG_10", 0.5, "warm"),
    Candidate("US_DATA_1000_BROAD_RR1", "US_DATA_1000", None, 0.5, "broad"),
    Candidate("NYSE_OPEN_BROAD_RR1", "NYSE_OPEN", None, 0.5, "broad"),
)


def load_rows() -> pd.DataFrame:
    sql = f"""
    SELECT
        o.trading_day,
        o.orb_label,
        o.entry_ts,
        o.entry_price,
        o.stop_price,
        o.pnl_r AS rr1_pnl_r,
        o.ambiguous_bar AS rr1_ambiguous_bar,
        d.atr_20_pct,
        d.overnight_range,
        d.orb_NYSE_OPEN_size,
        d.orb_US_DATA_1000_size
    FROM orb_outcomes o
    JOIN daily_features d
      ON d.trading_day = o.trading_day
     AND d.symbol = o.symbol
     AND d.orb_minutes = o.orb_minutes
    WHERE o.symbol = 'MGC'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.0
      AND o.orb_label IN ('NYSE_OPEN', 'US_DATA_1000')
      AND o.trading_day >= DATE '{OVERLAP_START}'
      AND o.outcome IS NOT NULL
      AND o.entry_ts IS NOT NULL
      AND o.entry_price IS NOT NULL
      AND o.stop_price IS NOT NULL
    ORDER BY o.trading_day, o.orb_label
    """
    with connect_db() as con:
        return con.execute(sql).fetchdf()


def filter_rows(rows: pd.DataFrame) -> pd.DataFrame:
    tagged: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        row_family = pd.Series(
            {
                "family_id": candidate.family_id,
                "orb_label": candidate.orb_label,
                "kind": candidate.family_kind,
                "filter_type": candidate.filter_type,
            }
        )
        mask = rows.apply(
            lambda r, family=row_family: passes_filter(r, family),  # type: ignore[arg-type]
            axis=1,
        )
        subset = rows.loc[mask].copy()
        if subset.empty:
            continue
        subset["family_id"] = candidate.family_id
        subset["family_kind"] = candidate.family_kind
        subset["target_r"] = candidate.target_r
        tagged.append(subset)
    return pd.concat(tagged, ignore_index=True) if tagged else pd.DataFrame()


def load_day_bars(con, trading_day) -> pd.DataFrame:
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)
    sql = """
        SELECT ts_utc, high, low
        FROM bars_1m
        WHERE symbol = 'MGC'
          AND ts_utc >= ?
          AND ts_utc < ?
        ORDER BY ts_utc
    """
    return con.execute(sql, [start_utc, end_utc]).fetchdf()


def rebuild_outcome(day_bars: pd.DataFrame, row: pd.Series) -> dict:
    entry_ts = pd.Timestamp(row["entry_ts"])
    entry_price = float(row["entry_price"])
    stop_price = float(row["stop_price"])
    target_r = float(row["target_r"])
    risk_points = abs(entry_price - stop_price)
    if risk_points <= 0:
        return {"path_outcome": None, "path_pnl_r": None, "path_ambiguous_bar": False}
    long_side = entry_price > stop_price
    if long_side:
        target_price = entry_price + risk_points * target_r
    else:
        target_price = entry_price - risk_points * target_r

    fill_bar = day_bars[day_bars["ts_utc"] == entry_ts]
    if not fill_bar.empty:
        bar = fill_bar.iloc[0]
        if long_side:
            hit_tgt = bar["high"] >= target_price
            hit_stp = bar["low"] <= stop_price
        else:
            hit_tgt = bar["low"] <= target_price
            hit_stp = bar["high"] >= stop_price
        if hit_tgt and hit_stp:
            return {"path_outcome": "loss", "path_pnl_r": -1.0, "path_ambiguous_bar": True}
        if hit_tgt:
            return {
                "path_outcome": "win",
                "path_pnl_r": round(to_r_multiple(MGC_SPEC, entry_price, stop_price, risk_points * target_r), 4),
                "path_ambiguous_bar": False,
            }
        if hit_stp:
            return {"path_outcome": "loss", "path_pnl_r": -1.0, "path_ambiguous_bar": False}

    _, td_end = compute_trading_day_utc_range(row["trading_day"])
    td_end_ts = pd.Timestamp(td_end)
    post_entry = day_bars[(day_bars["ts_utc"] > entry_ts) & (day_bars["ts_utc"] < td_end_ts)]
    if post_entry.empty:
        return {"path_outcome": "scratch", "path_pnl_r": None, "path_ambiguous_bar": False}

    highs = post_entry["high"].to_numpy(dtype=float)
    lows = post_entry["low"].to_numpy(dtype=float)
    if long_side:
        hit_target = highs >= target_price
        hit_stop = lows <= stop_price
    else:
        hit_target = lows <= target_price
        hit_stop = highs >= stop_price
    any_hit = hit_target | hit_stop
    if not any_hit.any():
        return {"path_outcome": "scratch", "path_pnl_r": None, "path_ambiguous_bar": False}
    first_hit_idx = int(np.argmax(any_hit))
    if hit_target[first_hit_idx] and hit_stop[first_hit_idx]:
        return {"path_outcome": "loss", "path_pnl_r": -1.0, "path_ambiguous_bar": True}
    if hit_target[first_hit_idx]:
        return {
            "path_outcome": "win",
            "path_pnl_r": round(to_r_multiple(MGC_SPEC, entry_price, stop_price, risk_points * target_r), 4),
            "path_ambiguous_bar": False,
        }
    return {"path_outcome": "loss", "path_pnl_r": -1.0, "path_ambiguous_bar": False}


def rebuild_trade_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    rebuilt_frames: list[pd.DataFrame] = []
    with connect_db() as con:
        for trading_day, group in rows.groupby("trading_day"):
            day_bars = load_day_bars(con, trading_day)
            rebuilt = group.copy()
            outcomes = rebuilt.apply(
                lambda r, day_bars=day_bars: rebuild_outcome(day_bars, r),
                axis=1,
                result_type="expand",
            )
            rebuilt = pd.concat([rebuilt.reset_index(drop=True), outcomes.reset_index(drop=True)], axis=1)
            rebuilt_frames.append(rebuilt)
    return pd.concat(rebuilt_frames, ignore_index=True)


def summarize(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate in CANDIDATES:
        grp = matrix[matrix["family_id"] == candidate.family_id].copy()
        if grp.empty:
            continue
        grp["trading_day"] = pd.to_datetime(grp["trading_day"])
        is_vals = grp.loc[grp["trading_day"] < HOLDOUT_START, "path_pnl_r"].dropna().values
        oos_vals = grp.loc[grp["trading_day"] >= HOLDOUT_START, "path_pnl_r"].dropna().values
        n_is, avg_is, wr_is, t_is, p_is = ttest_1s(is_vals)
        n_oos, avg_oos, wr_oos, t_oos, p_oos = ttest_1s(oos_vals)
        rows.append(
            {
                "family_id": candidate.family_id,
                "family_kind": candidate.family_kind,
                "orb_label": candidate.orb_label,
                "target_r": candidate.target_r,
                "n_is": n_is,
                "avg_is": avg_is,
                "wr_is": wr_is,
                "t_is": t_is,
                "p_is": p_is,
                "n_oos": n_oos,
                "avg_oos": avg_oos,
                "wr_oos": wr_oos,
                "t_oos": t_oos,
                "p_oos": p_oos,
                "scratch_rate": (grp["path_outcome"] == "scratch").mean(),
                "ambiguous_rate": grp["path_ambiguous_bar"].mean(),
                "avg_rr1_rewrite": grp["rr1_pnl_r"].mean(),
                "avg_path_gap": grp["path_pnl_r"].fillna(0).mean() - grp["rr1_pnl_r"].fillna(0).mean(),
            }
        )
    result = pd.DataFrame(rows)
    rejected = bh_fdr(result["p_is"].fillna(1.0).tolist(), q=BH_Q)
    result["bh_survive"] = [idx in rejected for idx in range(len(result))]
    result["primary_survivor"] = result["bh_survive"] & (result["n_is"] >= MIN_N) & (result["avg_is"] > 0)
    return result.sort_values(["primary_survivor", "avg_is"], ascending=[False, False]).reset_index(drop=True)


def build_markdown(summary: pd.DataFrame) -> str:
    survivors = summary[summary["primary_survivor"]].copy()
    lines = [
        "# MGC Path-Accurate Sub-R v1",
        "",
        "Date: 2026-04-19",
        "",
        "## Scope",
        "",
        "Hard-proof rebuild of the 5 native MGC low-R survivors from actual 1-minute",
        "price path. This replaces the earlier MFE-based rewrite approximation with",
        "canonical fill-bar + post-entry target/stop sequencing.",
        "",
        "Locked matrix:",
        "",
        "- 5 families carried forward from native low-R v1",
        "- K = 5 with global BH at q=0.10",
        "- same-bar target/stop conflicts fail closed as losses",
        "- scratches remain scratches if neither target nor stop is reached by trading-day end",
        "",
        "## Executive Verdict",
        "",
    ]
    if survivors.empty:
        lines += [
            "No families survive after path-accurate sub-R reconstruction.",
            "",
            "That kills the current native low-R MGC path as a validated edge claim.",
            "",
        ]
    else:
        lines += [
            f"{len(survivors)} families survive the path-accurate rebuild.",
            "",
            "This is the first version of the low-R MGC story that is strong enough to be",
            "called more than a diagnostic rewrite. It is still not a deployment memo, but",
            "the path-accurate rebuild removes the biggest remaining methodology gap.",
            "",
        ]
    lines += [
        "## Matrix",
        "",
        "| Family | Target | N IS | Avg IS | p IS | BH | Primary | N OOS | Avg OOS | Scratch | Ambig |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        p_is = f"{row.p_is:.4f}" if pd.notna(row.p_is) else "NA"
        lines.append(
            f"| {row.family_id} | {row.target_r:.2f} | {int(row.n_is)} | {row.avg_is:+.4f} | "
            f"{p_is} | {'Y' if row.bh_survive else 'N'} | {'Y' if row.primary_survivor else 'N'} | "
            f"{int(row.n_oos)} | {row.avg_oos:+.4f} | {row.scratch_rate:.1%} | {row.ambiguous_rate:.1%} |"
        )
    lines += [
        "",
        "## Guardrails",
        "",
        "- This still does not reopen broad GC proxy discovery.",
        "- This still does not promote live deployment by itself.",
        "- But this is the correct proof layer for sub-1R target claims.",
        "",
        "## Outputs",
        "",
        "- `research/output/mgc_path_accurate_subr_v1_summary.csv`",
        "- `research/output/mgc_path_accurate_subr_v1_trade_matrix.csv`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    base = filter_rows(load_rows())
    matrix = rebuild_trade_matrix(base)
    summary = summarize(matrix)
    write_csv(summary, f"{OUTPUT_PREFIX}_summary.csv")
    write_csv(matrix, f"{OUTPUT_PREFIX}_trade_matrix.csv")
    RESULT_PATH.write_text(build_markdown(summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
