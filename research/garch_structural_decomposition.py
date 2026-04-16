"""Structural decomposition of garch regime families.

Question:
  For the strongest families from the 2026-04-16 regime audit, is garch mostly
  a proxy for existing regime variables (ATR percentile, overnight-range
  percentile, gap/calendar flags), or does it behave like a distinct input?

Method:
  - Use the top session-side families from the pre-registered regime audit.
  - Load exact canonical trade populations for each underlying cell.
  - Use trade-time-knowable features only.
  - Report pooled family-level correlations and cell-level stratified sign
    persistence. No causal narrative injection.

Output:
  docs/audit/results/2026-04-16-garch-structural-decomposition.md
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research import garch_broad_exact_role_exhaustion as broad

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-structural-decomposition.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

IS_END = "2026-01-01"
FAMILIES = [
    ("COMEX_SETTLE", "high"),
    ("EUROPE_FLOW", "high"),
    ("TOKYO_OPEN", "high"),
    ("SINGAPORE_OPEN", "high"),
    ("LONDON_METALS", "high"),
    ("NYSE_OPEN", "low"),
]


def load_row_universe(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = broad.load_rows(con)
    return rows[rows["filter_type"].map(broad.in_scope)].copy()


def load_trades_with_context(con, row: pd.Series, direction: str) -> pd.DataFrame:
    filter_sql, join_sql = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
    if filter_sql is None:
        return pd.DataFrame()
    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      d.garch_forecast_vol_pct AS gp,
      d.atr_20_pct,
      d.overnight_range_pct,
      d.atr_vel_ratio,
      d.gap_type,
      d.is_nfp_day,
      d.is_opex_day,
      d.is_friday
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    {join_sql}
    WHERE o.symbol = '{row["instrument"]}'
      AND o.orb_minutes = {row["orb_minutes"]}
      AND o.orb_label = '{row["orb_label"]}'
      AND o.entry_model = '{row["entry_model"]}'
      AND o.rr_target = {row["rr_target"]}
      AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
      AND {filter_sql}
      AND o.trading_day < DATE '{IS_END}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["pnl_r"] = df["pnl_r"].astype(float)
    df["gp"] = df["gp"].astype(float)
    return df


def lift(df: pd.DataFrame, side: str, *, column: str = "gp", threshold: float | None = None, mask: pd.Series | None = None) -> float | None:
    if mask is not None:
        df = df.loc[mask].copy()
    if len(df) < 20:
        return None
    if threshold is None:
        threshold = 70 if side == "high" else 30
    gate = df[column] >= threshold if side == "high" else df[column] <= threshold
    on = df.loc[gate, "pnl_r"]
    off = df.loc[~gate, "pnl_r"]
    if len(on) < 10 or len(off) < 10:
        return None
    s_on = on.std(ddof=1)
    s_off = off.std(ddof=1)
    sr_on = on.mean() / s_on if s_on > 0 else 0.0
    sr_off = off.mean() / s_off if s_off > 0 else 0.0
    return float(sr_on - sr_off)


def family_report(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame, session: str, side: str) -> dict[str, object]:
    fam_rows = rows[rows["orb_label"] == session]
    pooled = []
    cell_rows = []
    for _, row in fam_rows.iterrows():
        for direction in ["long", "short"]:
            df = load_trades_with_context(con, row, direction)
            if len(df) < broad.MIN_TOTAL:
                continue
            srl = lift(df, side)
            if srl is None:
                continue
            df = df.copy()
            df["instrument"] = row["instrument"]
            df["direction"] = direction
            df["filter_type"] = row["filter_type"]
            pooled.append(df)

            atr_hi = lift(df, side, mask=(df["atr_20_pct"] >= 80).fillna(False))
            atr_lo = lift(df, side, mask=(df["atr_20_pct"] <= 20).fillna(False))
            ovn_hi = lift(df, side, mask=(df["overnight_range_pct"] >= 80).fillna(False))
            ovn_lo = lift(df, side, mask=(df["overnight_range_pct"] <= 20).fillna(False))
            cell_rows.append(
                {
                    "instrument": row["instrument"],
                    "direction": direction,
                    "filter_type": row["filter_type"],
                    "overall": srl,
                    "atr_hi": atr_hi,
                    "atr_lo": atr_lo,
                    "ovn_hi": ovn_hi,
                    "ovn_lo": ovn_lo,
                }
            )

    pooled_df = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
    cells_df = pd.DataFrame(cell_rows)
    if len(pooled_df) == 0:
        return {"session": session, "side": side, "empty": True}

    if side == "high":
        gflag = (pooled_df["gp"] >= 70).astype(int)
    else:
        gflag = (pooled_df["gp"] <= 30).astype(int)

    def safe_corr(a, b) -> float:
        a = pd.Series(a).astype(float)
        b = pd.Series(b).astype(float)
        if a.nunique(dropna=True) <= 1 or b.nunique(dropna=True) <= 1:
            return 0.0
        val = a.corr(b)
        return 0.0 if pd.isna(val) else float(val)

    corrs = {
        "corr_atr_20_pct": safe_corr(pooled_df["gp"], pooled_df["atr_20_pct"]),
        "corr_overnight_range_pct": safe_corr(pooled_df["gp"], pooled_df["overnight_range_pct"]),
        "corr_atr_vel_ratio": safe_corr(pooled_df["gp"], pooled_df["atr_vel_ratio"]),
        "corr_gap_up_flag": safe_corr(gflag, (pooled_df["gap_type"] == "gap_up").astype(int)),
        "corr_gap_down_flag": safe_corr(gflag, (pooled_df["gap_type"] == "gap_down").astype(int)),
        "corr_nfp_flag": safe_corr(gflag, pooled_df["is_nfp_day"].fillna(False).astype(int)),
        "corr_opex_flag": safe_corr(gflag, pooled_df["is_opex_day"].fillna(False).astype(int)),
        "corr_friday_flag": safe_corr(gflag, pooled_df["is_friday"].fillna(False).astype(int)),
    }

    def support_frac(series: pd.Series, expect_positive: bool) -> str:
        vals = series.dropna()
        if len(vals) == 0:
            return "n/a"
        good = (vals > 0).sum() if expect_positive else (vals < 0).sum()
        return f"{good}/{len(vals)}"

    expect_positive = side == "high"
    return {
        "session": session,
        "side": side,
        "empty": False,
        "n_cells": len(cells_df),
        "n_trades": len(pooled_df),
        "mean_gp": float(pooled_df["gp"].mean()),
        "mean_atr_pct": float(pooled_df["atr_20_pct"].dropna().mean()),
        "mean_ovn_pct": float(pooled_df["overnight_range_pct"].dropna().mean()),
        "mean_atr_vel": float(pooled_df["atr_vel_ratio"].dropna().mean()),
        **corrs,
        "overall_support": support_frac(cells_df["overall"], expect_positive),
        "atr_hi_support": support_frac(cells_df["atr_hi"], expect_positive),
        "atr_lo_support": support_frac(cells_df["atr_lo"], expect_positive),
        "ovn_hi_support": support_frac(cells_df["ovn_hi"], expect_positive),
        "ovn_lo_support": support_frac(cells_df["ovn_lo"], expect_positive),
    }


def emit(results: list[dict[str, object]]) -> None:
    lines = [
        "# Garch Structural Decomposition",
        "",
        "**Date:** 2026-04-16",
        "**Source families:** strongest directional / monotonicity families from `garch_regime_family_audit.py`.",
        "**Theory grounding:** Chan 2008 Ch 7 on high/low volatility regimes and GARCH-based regime tracking; Carver 2015 for continuous forecast/sizing interpretation.",
        "",
        "Trade-time-knowable proxies only:",
        "- `atr_20_pct`",
        "- `overnight_range_pct`",
        "- `atr_vel_ratio`",
        "- gap flags",
        "- NFP / OPEX / Friday flags",
        "",
        "No `day_type` or post-entry columns used.",
        "",
    ]
    for r in results:
        if r["empty"]:
            continue
        lines += [
            f"## {r['session']} {r['side']}",
            "",
            f"- Cells: **{r['n_cells']}**",
            f"- Pooled trades: **{r['n_trades']}**",
            f"- Mean `garch_pct`: **{r['mean_gp']:.2f}**",
            f"- Mean `atr_20_pct`: **{r['mean_atr_pct']:.2f}**",
            f"- Mean `overnight_range_pct`: **{r['mean_ovn_pct']:.2f}**",
            f"- Mean `atr_vel_ratio`: **{r['mean_atr_vel']:.3f}**",
            "",
            "### Correlations / overlap",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| corr(garch_pct, atr_20_pct) | {r['corr_atr_20_pct']:+.3f} |",
            f"| corr(garch_pct, overnight_range_pct) | {r['corr_overnight_range_pct']:+.3f} |",
            f"| corr(garch_pct, atr_vel_ratio) | {r['corr_atr_vel_ratio']:+.3f} |",
            f"| corr(garch_flag, gap_up) | {r['corr_gap_up_flag']:+.3f} |",
            f"| corr(garch_flag, gap_down) | {r['corr_gap_down_flag']:+.3f} |",
            f"| corr(garch_flag, is_nfp_day) | {r['corr_nfp_flag']:+.3f} |",
            f"| corr(garch_flag, is_opex_day) | {r['corr_opex_flag']:+.3f} |",
            f"| corr(garch_flag, is_friday) | {r['corr_friday_flag']:+.3f} |",
            "",
            "### Sign persistence within other regime strata",
            "",
            "| Check | Supporting cells |",
            "|---|---|",
            f"| Overall expected sign | {r['overall_support']} |",
            f"| Within ATR high stratum | {r['atr_hi_support']} |",
            f"| Within ATR low stratum | {r['atr_lo_support']} |",
            f"| Within overnight-range high stratum | {r['ovn_hi_support']} |",
            f"| Within overnight-range low stratum | {r['ovn_lo_support']} |",
            "",
        ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUTPUT_MD}")


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    rows = load_row_universe(con)
    results = [family_report(con, rows, session, side) for session, side in FAMILIES]
    con.close()
    emit(results)


if __name__ == "__main__":
    main()
