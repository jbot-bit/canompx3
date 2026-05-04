"""Shared helpers for conditional-role research studies.

Canonical truth only:
- daily_features
- orb_outcomes

Purpose:
- avoid re-copying the same lane-relative-volume role-study logic across
  bounded follow-through scripts
- keep prereg loading, canonical row assembly, frozen-IS quintiles, daily
  policy series, and simple execution diagnostics consistent
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml

from pipeline.cost_model import get_cost_spec, risk_in_dollars
from research.lib.stats import bh_fdr, compute_metrics, ttest_1s
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.hypothesis_loader import check_mode_a_consistency, load_hypothesis_metadata
from trading_app.topstep_scaling_plan import lots_for_position


@dataclass(frozen=True)
class DailyDeltaResult:
    n_days: int
    mean_delta_r: float
    t_stat: float
    p_two_tailed: float
    bh_survives: bool
    direction_positive: bool


def resolve_research_db_path(default_path: Path) -> Path:
    """Resolve the canonical gold.db path for managed worktree research.

    Managed worktrees may not carry their own hardlinked `gold.db`. In that
    case, fall back to the canonical repo db in `/mnt/c/Users/joshd/canompx3`.
    """
    if default_path.exists():
        return default_path
    fallback = Path("/mnt/c/Users/joshd/canompx3/gold.db")
    if fallback.exists():
        return fallback
    return default_path


def load_prereg_meta(prereg_path: Path) -> tuple[dict[str, Any], str]:
    meta = load_hypothesis_metadata(prereg_path)
    check_mode_a_consistency(meta)
    body = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    commit_sha = str(body.get("metadata", {}).get("commit_sha", "UNSTAMPED"))
    return meta, commit_sha


def list_sessions(con: duckdb.DuckDBPyConnection, symbol: str, orb_minutes: int) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT orb_label
        FROM orb_outcomes
        WHERE symbol = ? AND orb_minutes = ? AND pnl_r IS NOT NULL
        ORDER BY orb_label
        """,
        [symbol, orb_minutes],
    ).fetchall()
    return [r[0] for r in rows]


def load_symbol_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    symbol: str,
    orb_minutes: int,
    entry_model: str,
    confirm_bars: int,
    rr_target: float,
    include_timestamps: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for session in list_sessions(con, symbol, orb_minutes):
        rel_col = f"rel_vol_{session}"
        ts_cols = ", o.entry_ts, o.exit_ts" if include_timestamps else ""
        sql = f"""
        WITH df AS (
          SELECT d.trading_day, d.symbol, d.{rel_col} AS rel_vol
          FROM daily_features d
          WHERE d.symbol = '{symbol}' AND d.orb_minutes = {orb_minutes}
        )
        SELECT o.trading_day{ts_cols}, o.pnl_r, o.entry_price, o.stop_price, o.orb_label, df.rel_vol
        FROM orb_outcomes o
        JOIN df ON o.trading_day = df.trading_day AND o.symbol = df.symbol
        WHERE o.symbol = '{symbol}'
          AND o.orb_label = '{session}'
          AND o.orb_minutes = {orb_minutes}
          AND o.entry_model = '{entry_model}'
          AND o.confirm_bars = {confirm_bars}
          AND o.rr_target = {rr_target}
          AND o.pnl_r IS NOT NULL
        """
        sub = con.sql(sql).to_df()
        if sub.empty:
            continue
        sub["direction"] = np.where(sub["entry_price"] > sub["stop_price"], "long", "short")
        sub["lane"] = sub["orb_label"] + "_" + sub["direction"]
        sub["trading_day"] = pd.to_datetime(sub["trading_day"])
        spec = get_cost_spec(symbol)
        sub["risk_dollars"] = sub.apply(
            lambda row, spec=spec: risk_in_dollars(spec, float(row["entry_price"]), float(row["stop_price"])),
            axis=1,
        )
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["rel_vol", "pnl_r"]).reset_index(drop=True)


def assign_is_quintiles(
    df: pd.DataFrame,
    *,
    holdout: pd.Timestamp | None = None,
    desired_weight_map: dict[int, float] | None = None,
    executable_contract_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    holdout_ts = holdout or pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_df = out[out["trading_day"] < holdout_ts]
    cuts: dict[str, np.ndarray] = {}
    for lane, group in is_df.groupby("lane"):
        cuts[lane] = np.quantile(group["rel_vol"].astype(float), [0.2, 0.4, 0.6, 0.8])

    def assign(row: pd.Series) -> int:
        q = cuts[row["lane"]]
        value = float(row["rel_vol"])
        if value <= q[0]:
            return 1
        if value <= q[1]:
            return 2
        if value <= q[2]:
            return 3
        if value <= q[3]:
            return 4
        return 5

    out["q_is"] = out.apply(assign, axis=1)
    out["w_parent"] = 1.0
    out["w_q45"] = np.where(out["q_is"] >= 4, 1.0, 0.0)
    out["w_q5"] = np.where(out["q_is"] == 5, 1.0, 0.0)
    if desired_weight_map is not None:
        out["w_cont_desired"] = out["q_is"].map(desired_weight_map).astype(float)
    if executable_contract_map is not None:
        out["contracts_cont_exec"] = out["q_is"].map(executable_contract_map).astype(int)
    out["contracts_parent"] = 1
    out["contracts_q45"] = np.where(out["q_is"] >= 4, 1, 0)
    out["contracts_q5"] = np.where(out["q_is"] == 5, 1, 0)
    return out


def rank_slope(df: pd.DataFrame) -> tuple[int, float, float]:
    import statsmodels.api as sm

    ranked = df.copy()
    ranked["rank_rel_vol"] = (
        ranked.groupby("lane")["rel_vol"].rank(method="average").div(ranked.groupby("lane")["lane"].transform("count"))
    )
    X = ranked[["rank_rel_vol"]].astype(float).copy()
    if ranked["lane"].nunique() > 1:
        X = pd.concat([X, pd.get_dummies(ranked["lane"], drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(ranked["pnl_r"].astype(float), X).fit(cov_type="HC3")
    return len(ranked), float(model.params["rank_rel_vol"]), float(model.tvalues["rank_rel_vol"])


def quintile_means(df: pd.DataFrame) -> dict[int, tuple[int, float]]:
    grouped = df.groupby("q_is")["pnl_r"].agg(["count", "mean"]).reset_index()
    return {int(row["q_is"]): (int(row["count"]), float(row["mean"])) for _, row in grouped.iterrows()}


def daily_policy_series(df: pd.DataFrame, weight_col: str) -> pd.Series:
    weighted = df["pnl_r"].astype(float) * df[weight_col].astype(float)
    return weighted.groupby(df["trading_day"]).sum().sort_index()


def daily_dollar_series(df: pd.DataFrame, weight_col: str) -> pd.Series:
    weighted = df["pnl_r"].astype(float) * df["risk_dollars"].astype(float) * df[weight_col].astype(float)
    return weighted.groupby(df["trading_day"]).sum().sort_index()


def max_drawdown(series: pd.Series) -> float:
    vals = series.to_numpy(dtype=float)
    if len(vals) == 0:
        return 0.0
    cumulative = np.cumsum(vals)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return float(drawdown.max())


def role_metrics(df: pd.DataFrame, weight_col: str) -> dict[str, float]:
    selected = df[df[weight_col] > 0]
    trade_share = float((df[weight_col] > 0).mean())
    selected_avg = float(selected["pnl_r"].mean()) if not selected.empty else float("nan")
    policy_ev = float((df["pnl_r"].astype(float) * df[weight_col].astype(float)).mean())
    avg_weight = float(df[weight_col].mean())
    capital_norm = float(policy_ev / avg_weight) if avg_weight > 0 else float("nan")
    daily_r = daily_policy_series(df, weight_col)
    daily_dollars = daily_dollar_series(df, weight_col)
    r_metrics = compute_metrics(daily_r.tolist()) or {}
    return {
        "selected_n": int((df[weight_col] > 0).sum()),
        "trade_share": trade_share,
        "selected_avg_r": selected_avg,
        "policy_ev_per_opp_r": policy_ev,
        "avg_weight": avg_weight,
        "capital_normalized_ev_r": capital_norm,
        "daily_total_r": float(daily_r.sum()),
        "daily_max_dd_r": float(r_metrics.get("max_dd", 0.0)),
        "daily_total_dollars": float(daily_dollars.sum()),
        "daily_max_dd_dollars": max_drawdown(daily_dollars),
    }


def delta_test(parent_daily: pd.Series, candidate_daily: pd.Series) -> DailyDeltaResult:
    aligned = pd.concat(
        [parent_daily.rename("parent"), candidate_daily.rename("candidate")],
        axis=1,
    ).fillna(0.0)
    delta = (aligned["candidate"] - aligned["parent"]).to_numpy(dtype=float)
    n_days, mean_delta, _wr, t_stat, p_two = ttest_1s(delta, 0.0)
    return DailyDeltaResult(
        n_days=n_days,
        mean_delta_r=float(mean_delta),
        t_stat=float(t_stat),
        p_two_tailed=float(p_two),
        bh_survives=False,
        direction_positive=bool(mean_delta > 0) if not np.isnan(mean_delta) else False,
    )


def apply_bh(
    results: dict[tuple[str, str], DailyDeltaResult], q: float = 0.05
) -> dict[tuple[str, str], DailyDeltaResult]:
    ordered_keys = list(results.keys())
    p_values = [results[key].p_two_tailed for key in ordered_keys]
    reject_ix = bh_fdr(p_values, q=q)
    updated: dict[tuple[str, str], DailyDeltaResult] = {}
    for i, key in enumerate(ordered_keys):
        res = results[key]
        updated[key] = DailyDeltaResult(
            n_days=res.n_days,
            mean_delta_r=res.mean_delta_r,
            t_stat=res.t_stat,
            p_two_tailed=res.p_two_tailed,
            bh_survives=i in reject_ix,
            direction_positive=res.direction_positive,
        )
    return updated


def max_open_contracts_and_lots(df: pd.DataFrame, contract_col: str, instrument: str) -> tuple[int, int]:
    max_contracts = 0
    max_lots = 0
    for _day, day_df in df.groupby("trading_day"):
        trades = day_df[day_df[contract_col] > 0].copy()
        if trades.empty:
            continue
        events: list[tuple[pd.Timestamp, int, int]] = []
        for _, row in trades.iterrows():
            entry_ts = row.get("entry_ts")
            exit_ts = row.get("exit_ts")
            contracts = int(row[contract_col])
            if pd.isna(entry_ts):
                continue
            if pd.isna(exit_ts):
                exit_ts = entry_ts
            events.append((pd.Timestamp(entry_ts), 0, contracts))
            events.append((pd.Timestamp(exit_ts), 1, contracts))
        events.sort(key=lambda item: (item[0], item[1]))
        open_contracts = 0
        for _ts, event_type, contracts in events:
            if event_type == 0:
                open_contracts += contracts
            else:
                open_contracts = max(0, open_contracts - contracts)
            max_contracts = max(max_contracts, open_contracts)
            max_lots = max(max_lots, lots_for_position(instrument, open_contracts))
    return max_contracts, max_lots
