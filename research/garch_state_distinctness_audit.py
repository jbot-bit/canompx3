"""State-distinctness audit for garch vs adjacent vol-state proxies.

Purpose:
  On a locked garch-anchored family set, determine whether garch_forecast_vol_pct
  looks distinct, complementary, or mostly subsumed relative to:
    - atr_20_pct
    - overnight_range_pct
    - atr_vel_ratio / atr_vel_regime

Methodology constraints:
  - D1 canonical distinctness uses exact canonical populations on the locked
    family set.
  - D2 utility distinctness uses validated_setups only and remains research-
    provisional, not production truth.
  - 2026 forward/OOS readouts are descriptive only and do not decide verdicts.
  - atr_vel is handled canonically: regime strata first, IS terciles only for
    the four-cell decomposition.
  - No new broad search, no deployment geometry, no confluence fishing.

Output:
  docs/audit/results/2026-04-16-garch-state-distinctness-audit.md
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
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

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-state-distinctness-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

GARCH_HIGH = 70.0
GARCH_LOW = 30.0
PCT_HIGH = 80.0
PCT_LOW = 20.0
MIN_TOTAL = 50
MIN_ON_OFF = 10
MIN_FOUR_CELL = 20

FAMILIES = [
    ("COMEX_SETTLE", "high"),
    ("EUROPE_FLOW", "high"),
    ("TOKYO_OPEN", "high"),
    ("SINGAPORE_OPEN", "high"),
    ("LONDON_METALS", "high"),
    ("NYSE_OPEN", "low"),
]


@dataclass(frozen=True)
class FamilyContext:
    session: str
    side: str

    @property
    def label(self) -> str:
        return f"{self.session}_{self.side}"

    @property
    def expect_positive(self) -> bool:
        return self.side == "high"


FAMILY_CONTEXTS = [FamilyContext(*item) for item in FAMILIES]


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    if a.nunique(dropna=True) <= 1 or b.nunique(dropna=True) <= 1:
        return 0.0
    val = a.corr(b)
    return 0.0 if pd.isna(val) else float(val)


def sharpe_like(arr: pd.Series) -> float:
    arr = pd.Series(arr).astype(float)
    if len(arr) < 2:
        return 0.0
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd) if sd and not pd.isna(sd) else 0.0


def mean_diff(on: pd.Series, off: pd.Series) -> float:
    return float(pd.Series(on).mean() - pd.Series(off).mean())


def sr_lift(on: pd.Series, off: pd.Series) -> float:
    return float(sharpe_like(on) - sharpe_like(off))


def garch_gate(df: pd.DataFrame, side: str) -> pd.Series:
    if side == "high":
        return df["gp"] >= GARCH_HIGH
    return df["gp"] <= GARCH_LOW


def pct_gate(df: pd.DataFrame, column: str, side: str) -> pd.Series:
    if side == "high":
        return df[column] >= PCT_HIGH
    return df[column] <= PCT_LOW


def score_side_metrics(df: pd.DataFrame, side: str) -> dict[str, float | int | bool | None]:
    if len(df) < MIN_TOTAL:
        return {"skip": True}
    mask = garch_gate(df, side)
    on = df.loc[mask, "pnl_r"]
    off = df.loc[~mask, "pnl_r"]
    if len(on) < MIN_ON_OFF or len(off) < MIN_ON_OFF:
        return {"skip": True}
    return {
        "skip": False,
        "n_on": int(len(on)),
        "n_off": int(len(off)),
        "lift": mean_diff(on, off),
        "sr_lift": sr_lift(on, off),
    }


def proxy_side_metrics(df: pd.DataFrame, column: str, side: str) -> dict[str, float | int | bool]:
    if len(df) < MIN_TOTAL:
        return {"skip": True}
    mask = pct_gate(df, column, side)
    on = df.loc[mask, "pnl_r"]
    off = df.loc[~mask, "pnl_r"]
    if len(on) < MIN_ON_OFF or len(off) < MIN_ON_OFF:
        return {"skip": True}
    return {
        "skip": False,
        "n_on": int(len(on)),
        "n_off": int(len(off)),
        "lift": mean_diff(on, off),
        "sr_lift": sr_lift(on, off),
    }


def ternary_bucket(series: pd.Series) -> pd.Series:
    vals = pd.Series(series).astype(float)
    out = pd.Series(index=vals.index, dtype="object")
    good = vals.dropna()
    if len(good) < 30 or good.nunique() < 3:
        return out
    q33 = float(np.nanpercentile(good, 33))
    q67 = float(np.nanpercentile(good, 67))
    out.loc[vals <= q33] = "low"
    out.loc[vals >= q67] = "high"
    return out


def load_rows(con: duckdb.DuckDBPyConnection, *, validated_only: bool) -> pd.DataFrame:
    rows = broad.load_rows(con)
    rows = rows[rows["filter_type"].map(broad.in_scope)].copy()
    rows = rows[rows["orb_label"].isin({ctx.session for ctx in FAMILY_CONTEXTS})].copy()
    if validated_only:
        rows = rows[rows["src"] == "validated"].copy()
    else:
        rows["_src_rank"] = rows["src"].map({"validated": 0, "experimental": 1}).fillna(9)
        rows = rows.sort_values(["_src_rank", "strategy_id"]).drop_duplicates(
            subset=[
                "strategy_id",
                "instrument",
                "orb_label",
                "orb_minutes",
                "rr_target",
                "entry_model",
                "filter_type",
            ],
            keep="first",
        )
        rows = rows.drop(columns="_src_rank")
    return rows.reset_index(drop=True)


def load_context_trades(
    con: duckdb.DuckDBPyConnection,
    row: pd.Series,
    direction: str,
    *,
    is_oos: bool,
) -> pd.DataFrame:
    filter_sql, join_sql = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
    if filter_sql is None:
        return pd.DataFrame()
    date_clause = ">=" if is_oos else "<"
    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      d.garch_forecast_vol_pct AS gp,
      d.atr_20_pct,
      d.overnight_range_pct,
      d.atr_vel_ratio,
      d.atr_vel_regime
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
      AND d.atr_20_pct IS NOT NULL
      AND d.overnight_range_pct IS NOT NULL
      AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
      AND {filter_sql}
      AND o.trading_day {date_clause} DATE '{broad.IS_END}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    for col in ["pnl_r", "gp", "atr_20_pct", "overnight_range_pct", "atr_vel_ratio"]:
        df[col] = df[col].astype(float)
    df["atr_vel_regime"] = df["atr_vel_regime"].astype("string")
    return df


def collect_family_data(con: duckdb.DuckDBPyConnection, *, validated_only: bool) -> dict[str, dict[str, object]]:
    rows = load_rows(con, validated_only=validated_only)
    out: dict[str, dict[str, object]] = {}
    for ctx in FAMILY_CONTEXTS:
        pooled_frames: list[pd.DataFrame] = []
        cell_meta: list[dict[str, object]] = []
        fam_rows = rows[rows["orb_label"] == ctx.session].copy()
        for _, row in fam_rows.iterrows():
            for direction in ["long", "short"]:
                df_is = load_context_trades(con, row, direction, is_oos=False)
                if len(df_is) < MIN_TOTAL:
                    continue
                df_oos = load_context_trades(con, row, direction, is_oos=True)
                df_is = df_is.copy()
                df_is["instrument"] = row["instrument"]
                df_is["direction"] = direction
                df_is["filter_type"] = row["filter_type"]
                df_is["src"] = row["src"]
                pooled_frames.append(df_is)
                base = score_side_metrics(df_is, ctx.side)
                if base["skip"]:
                    continue
                cell_meta.append(
                    {
                        "instrument": row["instrument"],
                        "direction": direction,
                        "filter_type": row["filter_type"],
                        "src": row["src"],
                        "overall_sr_lift": float(base["sr_lift"]),
                        "overall_lift": float(base["lift"]),
                        "oos_sr_lift": None
                        if len(df_oos) < MIN_TOTAL
                        else score_side_metrics(df_oos, ctx.side).get("sr_lift"),
                    }
                )
        pooled = pd.concat(pooled_frames, ignore_index=True) if pooled_frames else pd.DataFrame()
        out[ctx.label] = {"ctx": ctx, "pooled": pooled, "cells": pd.DataFrame(cell_meta)}
    return out


def overlap_table(pooled: pd.DataFrame, side: str) -> dict[str, float | int]:
    if len(pooled) == 0:
        return {"n_trades": 0}
    gflag = garch_gate(pooled, side).astype(int)
    return {
        "n_trades": int(len(pooled)),
        "mean_gp": float(pooled["gp"].mean()),
        "mean_atr_pct": float(pooled["atr_20_pct"].mean()),
        "mean_ovn_pct": float(pooled["overnight_range_pct"].mean()),
        "mean_atr_vel": float(pooled["atr_vel_ratio"].dropna().mean())
        if pooled["atr_vel_ratio"].notna().any()
        else float("nan"),
        "corr_garch_atr": safe_corr(pooled["gp"], pooled["atr_20_pct"]),
        "corr_garch_ovn": safe_corr(pooled["gp"], pooled["overnight_range_pct"]),
        "corr_garch_atr_vel": safe_corr(pooled["gp"], pooled["atr_vel_ratio"]),
        "corr_garch_contracting": safe_corr(gflag, (pooled["atr_vel_regime"] == "Contracting").astype(int)),
        "corr_garch_expanding": safe_corr(gflag, (pooled["atr_vel_regime"] == "Expanding").astype(int)),
    }


def stratum_lift(df: pd.DataFrame, side: str) -> dict[str, object]:
    base = score_side_metrics(df, side)
    if base["skip"]:
        return {"status": "thin", "n": int(len(df))}
    return {
        "status": "ok",
        "n": int(len(df)),
        "lift": float(base["lift"]),
        "sr_lift": float(base["sr_lift"]),
        "support": bool(base["sr_lift"] > 0) if side == "high" else bool(base["sr_lift"] < 0),
    }


def conditional_persistence(pooled: pd.DataFrame, ctx: FamilyContext) -> dict[str, dict[str, object]]:
    if len(pooled) == 0:
        return {}
    out: dict[str, dict[str, object]] = {}
    out["overall"] = stratum_lift(pooled, ctx.side)
    out["atr_high"] = stratum_lift(pooled.loc[pct_gate(pooled, "atr_20_pct", "high")], ctx.side)
    out["atr_low"] = stratum_lift(pooled.loc[pct_gate(pooled, "atr_20_pct", "low")], ctx.side)
    out["ovn_high"] = stratum_lift(pooled.loc[pct_gate(pooled, "overnight_range_pct", "high")], ctx.side)
    out["ovn_low"] = stratum_lift(pooled.loc[pct_gate(pooled, "overnight_range_pct", "low")], ctx.side)
    for regime in ["Expanding", "Stable", "Contracting"]:
        out[f"atr_vel_{regime.lower()}"] = stratum_lift(
            pooled.loc[pooled["atr_vel_regime"] == regime],
            ctx.side,
        )
    return out


def four_cell_tail_pair(
    pooled: pd.DataFrame,
    *,
    proxy_col: str | None = None,
    proxy_label: str,
    proxy_mode: str,
) -> pd.DataFrame:
    work = pooled.copy()
    work["g_state"] = pd.Series(index=work.index, dtype="object")
    work.loc[work["gp"] <= GARCH_LOW, "g_state"] = "low"
    work.loc[work["gp"] >= GARCH_HIGH, "g_state"] = "high"

    work["p_state"] = pd.Series(index=work.index, dtype="object")
    if proxy_mode == "pct":
        assert proxy_col is not None
        work.loc[work[proxy_col] <= PCT_LOW, "p_state"] = "low"
        work.loc[work[proxy_col] >= PCT_HIGH, "p_state"] = "high"
    elif proxy_mode == "atr_vel_tercile":
        work["p_state"] = ternary_bucket(work["atr_vel_ratio"])
    else:
        raise ValueError(proxy_mode)

    work = work.dropna(subset=["g_state", "p_state"]).copy()
    if len(work) == 0:
        return pd.DataFrame()

    rows = []
    for p_state in ["low", "high"]:
        for g_state in ["low", "high"]:
            sub = work[(work["p_state"] == p_state) & (work["g_state"] == g_state)]
            if len(sub) == 0:
                rows.append(
                    {
                        "pair": proxy_label,
                        "proxy_state": p_state,
                        "garch_state": g_state,
                        "n": 0,
                        "exp_r": np.nan,
                        "sr": np.nan,
                        "total_r": np.nan,
                        "status": "empty",
                    }
                )
                continue
            rows.append(
                {
                    "pair": proxy_label,
                    "proxy_state": p_state,
                    "garch_state": g_state,
                    "n": int(len(sub)),
                    "exp_r": float(sub["pnl_r"].mean()),
                    "sr": sharpe_like(sub["pnl_r"]),
                    "total_r": float(sub["pnl_r"].sum()),
                    "status": "thin" if len(sub) < MIN_FOUR_CELL else "ok",
                }
            )
    return pd.DataFrame(rows)


def utility_rows(
    con: duckdb.DuckDBPyConnection,
    *,
    family: FamilyContext,
    score_name: str,
    score_col: str,
    threshold_side: str,
    threshold_value: float,
) -> pd.DataFrame:
    rows = load_rows(con, validated_only=True)
    rows = rows[rows["orb_label"] == family.session].copy()
    out = []
    for _, row in rows.iterrows():
        for direction in ["long", "short"]:
            df = load_context_trades(con, row, direction, is_oos=False)
            if len(df) < MIN_TOTAL or score_col not in df:
                continue
            df = df.rename(columns={score_col: "score"}).copy()
            if threshold_side == "high":
                mask = df["score"] >= threshold_value
            else:
                mask = df["score"] <= threshold_value
            on = df.loc[mask, "pnl_r"]
            off = df.loc[~mask, "pnl_r"]
            if len(on) < MIN_ON_OFF or len(off) < MIN_ON_OFF:
                continue
            srl = sr_lift(on, off)
            out.append(
                {
                    "family": family.label,
                    "score": score_name,
                    "instrument": row["instrument"],
                    "direction": direction,
                    "filter_type": row["filter_type"],
                    "n_total": int(len(df)),
                    "n_on": int(len(on)),
                    "n_off": int(len(off)),
                    "lift": mean_diff(on, off),
                    "sr_lift": srl,
                    "support": bool(srl > 0) if family.expect_positive else bool(srl < 0),
                }
            )
    return pd.DataFrame(out)


def utility_summary(con: duckdb.DuckDBPyConnection, family: FamilyContext) -> pd.DataFrame:
    frames = [
        utility_rows(
            con,
            family=family,
            score_name="garch",
            score_col="gp",
            threshold_side=family.side,
            threshold_value=GARCH_HIGH if family.side == "high" else GARCH_LOW,
        ),
        utility_rows(
            con,
            family=family,
            score_name="atr",
            score_col="atr_20_pct",
            threshold_side=family.side,
            threshold_value=PCT_HIGH if family.side == "high" else PCT_LOW,
        ),
        utility_rows(
            con,
            family=family,
            score_name="overnight",
            score_col="overnight_range_pct",
            threshold_side=family.side,
            threshold_value=PCT_HIGH if family.side == "high" else PCT_LOW,
        ),
    ]
    df = (
        pd.concat([f for f in frames if len(f) > 0], ignore_index=True)
        if any(len(f) > 0 for f in frames)
        else pd.DataFrame()
    )
    if len(df) == 0:
        return df
    return (
        df.groupby("score", as_index=False)
        .agg(
            cells=("score", "size"),
            support_cells=("support", "sum"),
            mean_lift=("lift", "mean"),
            mean_sr_lift=("sr_lift", "mean"),
            mean_n_on=("n_on", "mean"),
        )
        .sort_values(["mean_sr_lift", "mean_lift"], ascending=False)
        .reset_index(drop=True)
    )


def four_cell_signal(df: pd.DataFrame, family: FamilyContext) -> dict[str, object]:
    if len(df) == 0:
        return {"status": "thin", "signal": "no_data"}
    pivot = {(r["proxy_state"], r["garch_state"]): r for _, r in df.iterrows()}
    hh = pivot.get(("high", "high"))
    hl = pivot.get(("high", "low"))
    lh = pivot.get(("low", "high"))
    ll = pivot.get(("low", "low"))
    if any(item is None for item in [hh, hl, lh, ll]):
        return {"status": "thin", "signal": "missing_cells"}
    if any(r["status"] != "ok" for r in [hh, hl, lh, ll]):
        return {"status": "thin", "signal": "thin_cells"}
    if family.side == "high":
        if hh["exp_r"] > hl["exp_r"] and lh["exp_r"] > ll["exp_r"]:
            return {"status": "ok", "signal": "garch_marginal"}
        if hh["exp_r"] > max(hl["exp_r"], lh["exp_r"], ll["exp_r"]):
            return {"status": "ok", "signal": "interaction_like"}
        return {"status": "ok", "signal": "shared_or_flat"}
    if hh["exp_r"] < hl["exp_r"] and lh["exp_r"] < ll["exp_r"]:
        return {"status": "ok", "signal": "garch_marginal"}
    if ll["exp_r"] < min(hl["exp_r"], lh["exp_r"], hh["exp_r"]):
        return {"status": "ok", "signal": "interaction_like"}
    return {"status": "ok", "signal": "shared_or_flat"}


def verdict_for_family(
    ctx: FamilyContext,
    persistence: dict[str, dict[str, object]],
    utility: pd.DataFrame,
    pair_signals: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    out = []
    for proxy in ["atr", "overnight", "atr_vel"]:
        cond_keys = {
            "atr": ["atr_high", "atr_low"],
            "overnight": ["ovn_high", "ovn_low"],
            "atr_vel": ["atr_vel_expanding", "atr_vel_stable", "atr_vel_contracting"],
        }[proxy]
        non_thin = [persistence[k] for k in cond_keys if persistence.get(k, {}).get("status") == "ok"]
        supportive = sum(1 for item in non_thin if item["support"])
        overall_ok = persistence.get("overall", {}).get("status") == "ok"
        overall_support = bool(persistence.get("overall", {}).get("support", False))

        utility_rank = None
        if len(utility) > 0 and proxy != "atr_vel":
            util_row = utility[utility["score"] == ("atr" if proxy == "atr" else "overnight")]
            g_row = utility[utility["score"] == "garch"]
            if len(util_row) > 0 and len(g_row) > 0:
                utility_rank = (
                    "garch_ge_proxy"
                    if float(g_row.iloc[0]["mean_sr_lift"]) >= float(util_row.iloc[0]["mean_sr_lift"])
                    else "proxy_gt_garch"
                )

        pair_signal = pair_signals[proxy]["signal"]
        pair_ok = pair_signals[proxy]["status"] == "ok"

        if (
            overall_ok
            and overall_support
            and supportive >= 1
            and pair_ok
            and pair_signal in {"garch_marginal", "interaction_like"}
            and utility_rank != "proxy_gt_garch"
        ):
            verdict = "distinct"
        elif overall_ok and overall_support and pair_ok and pair_signal == "interaction_like":
            verdict = "complementary"
        elif (
            (not overall_support or supportive == 0)
            and pair_ok
            and pair_signal == "shared_or_flat"
            and utility_rank == "proxy_gt_garch"
        ):
            verdict = "subsumed"
        else:
            verdict = "unclear"

        role = {
            "distinct": "R3/R7",
            "complementary": "R7/R8",
            "subsumed": "no_standalone_role",
            "unclear": "unclear",
        }[verdict]
        out.append({"proxy": proxy, "verdict": verdict, "role": role})
    return out


def emit(
    d1: dict[str, dict[str, object]],
    d2: dict[str, dict[str, object]],
    utility_by_family: dict[str, pd.DataFrame],
    pair_tables: dict[str, dict[str, pd.DataFrame]],
    verdicts: dict[str, list[dict[str, str]]],
) -> None:
    lines = [
        "# Garch State Distinctness Audit",
        "",
        "**Date:** 2026-04-16",
        "**Purpose:** local distinctness audit on the locked garch-anchored family set.",
        "**Boundary:** not a global proxy ranking, not a deployment proof, not a new search.",
        "",
        "## Scope rules",
        "",
        "- D1 canonical distinctness uses canonical populations on the locked family set.",
        "- D2 utility distinctness uses `validated_setups` only and remains research-provisional.",
        "- 2026 forward/OOS is descriptive only and does not decide the verdicts here.",
        "- `atr_vel` is treated canonically via regime strata first; no fake 70/30 thresholds.",
        "",
    ]

    for label, payload in d1.items():
        ctx: FamilyContext = payload["ctx"]
        pooled: pd.DataFrame = payload["pooled"]
        overlap = overlap_table(pooled, ctx.side)
        persistence = conditional_persistence(pooled, ctx)
        utility = utility_by_family.get(label, pd.DataFrame())
        lines.extend(
            [
                f"## {ctx.session} {ctx.side}",
                "",
                f"- D1 pooled trades: **{overlap['n_trades']}**",
                f"- Mean `garch_pct`: **{overlap.get('mean_gp', float('nan')):.2f}**",
                f"- Mean `atr_20_pct`: **{overlap.get('mean_atr_pct', float('nan')):.2f}**",
                f"- Mean `overnight_range_pct`: **{overlap.get('mean_ovn_pct', float('nan')):.2f}**",
                f"- Mean `atr_vel_ratio`: **{overlap.get('mean_atr_vel', float('nan')):.3f}**",
                "",
                "### Overlap",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| corr(garch, atr_20_pct) | {overlap.get('corr_garch_atr', float('nan')):+.3f} |",
                f"| corr(garch, overnight_range_pct) | {overlap.get('corr_garch_ovn', float('nan')):+.3f} |",
                f"| corr(garch, atr_vel_ratio) | {overlap.get('corr_garch_atr_vel', float('nan')):+.3f} |",
                f"| corr(garch_flag, atr_vel_contracting) | {overlap.get('corr_garch_contracting', float('nan')):+.3f} |",
                f"| corr(garch_flag, atr_vel_expanding) | {overlap.get('corr_garch_expanding', float('nan')):+.3f} |",
                "",
                "### Conditional sign persistence",
                "",
                "| Stratum | Status | N | Lift | SR lift | Support |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for key in [
            "overall",
            "atr_high",
            "atr_low",
            "ovn_high",
            "ovn_low",
            "atr_vel_expanding",
            "atr_vel_stable",
            "atr_vel_contracting",
        ]:
            row = persistence.get(key, {"status": "thin", "n": 0})
            lines.append(
                f"| {key} | {row.get('status', 'thin')} | {int(row.get('n', 0))} | "
                f"{'' if row.get('status') != 'ok' else f'{row.get("lift", float("nan")):+.3f}'} | "
                f"{'' if row.get('status') != 'ok' else f'{row.get("sr_lift", float("nan")):+.3f}'} | "
                f"{'' if row.get('status') != 'ok' else ('Y' if row.get('support') else 'N')} |"
            )

        lines.extend(["", "### Four-cell decompositions", ""])
        for proxy in ["atr", "overnight", "atr_vel"]:
            table = pair_tables[label][proxy]
            lines.append(f"**{proxy}**")
            lines.append("")
            lines.append("| Proxy state | Garch state | N | ExpR | SR | Total R | Status |")
            lines.append("|---|---|---:|---:|---:|---:|---|")
            if len(table) == 0:
                lines.append("| n/a | n/a | 0 |  |  |  | empty |")
            else:
                for _, row in table.iterrows():
                    lines.append(
                        f"| {row['proxy_state']} | {row['garch_state']} | {int(row['n'])} | "
                        f"{'' if pd.isna(row['exp_r']) else f'{row["exp_r"]:+.3f}'} | "
                        f"{'' if pd.isna(row['sr']) else f'{row["sr"]:+.3f}'} | "
                        f"{'' if pd.isna(row['total_r']) else f'{row["total_r"]:+.1f}'} | "
                        f"{row['status']} |"
                    )
            lines.append("")

        lines.extend(["### D2 local utility comparison", ""])
        if len(utility) == 0:
            lines.append("No validated-family utility rows met minimum support.")
        else:
            lines.append("| Score | Cells | Support cells | Mean lift | Mean SR lift | Mean N_on |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for _, row in utility.iterrows():
                lines.append(
                    f"| {row['score']} | {int(row['cells'])} | {int(row['support_cells'])} | "
                    f"{row['mean_lift']:+.3f} | {row['mean_sr_lift']:+.3f} | {row['mean_n_on']:.1f} |"
                )
        lines.extend(["", "### Verdicts", ""])
        lines.append("| Proxy | Verdict | Allowed role |")
        lines.append("|---|---|---|")
        for item in verdicts[label]:
            lines.append(f"| {item['proxy']} | {item['verdict']} | {item['role']} |")
        lines.append("")

    lines.extend(
        [
            "## Guardrails",
            "",
            "- `distinct` / `complementary` / `subsumed` are local to the locked family set.",
            "- `validated_setups` evidence here is research-provisional, not production truth.",
            "- No 2026 forward/OOS figure was allowed to decide the verdicts.",
            "- Pairwise correlations are descriptive only; verdicts required persistence plus four-cell or utility support.",
            "",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    d1 = collect_family_data(con, validated_only=False)
    d2 = collect_family_data(con, validated_only=True)

    utility_by_family: dict[str, pd.DataFrame] = {}
    pair_tables: dict[str, dict[str, pd.DataFrame]] = {}
    verdicts: dict[str, list[dict[str, str]]] = {}

    for ctx in FAMILY_CONTEXTS:
        label = ctx.label
        pooled = d1[label]["pooled"]
        utility = utility_summary(con, ctx)
        utility_by_family[label] = utility
        pair_tables[label] = {
            "atr": four_cell_tail_pair(pooled, proxy_col="atr_20_pct", proxy_label="atr", proxy_mode="pct"),
            "overnight": four_cell_tail_pair(
                pooled, proxy_col="overnight_range_pct", proxy_label="overnight", proxy_mode="pct"
            ),
            "atr_vel": four_cell_tail_pair(pooled, proxy_label="atr_vel", proxy_mode="atr_vel_tercile"),
        }
        persistence = conditional_persistence(pooled, ctx)
        pair_signals = {proxy: four_cell_signal(tbl, ctx) for proxy, tbl in pair_tables[label].items()}
        verdicts[label] = verdict_for_family(ctx, persistence, utility, pair_signals)

    con.close()
    emit(d1, d2, utility_by_family, pair_tables, verdicts)
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
