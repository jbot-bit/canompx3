#!/usr/bin/env python3
"""Read-only quarantine audit for pre-trade context hypotheses.

This script is intentionally separate from discovery/validation mutation paths.
It reads only canonical layers (`daily_features`, `orb_outcomes`) and tests a
small pre-registered family of hypotheses using:

- strict pre-trade feature safety checks
- no look-ahead columns
- fixed E2 / O5 / RR1.0 scope
- pre-2026 in-sample (IS) and 2026+ holdout (OOS) split
- permutation test on filtered-vs-excluded lift
- BH FDR across the full family

The goal is not promotion. The goal is to decide whether these context families
survive a first honest quarantine pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from pipeline.paths import GOLD_DB_PATH
from pipeline.session_guard import is_feature_safe
from trading_app.config import ALL_FILTERS

HOLDOUT_DATE = pd.Timestamp("2026-01-01")
ROOT = Path(__file__).resolve().parent.parent.parent
HYP_PATH = ROOT / "docs" / "audit" / "hypotheses" / "2026-04-11-external-context-quarantine.yaml"
N_PERM = 5000
MIN_GROUP_N = 20
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.0
ORB_MINUTES = 5

# Explicit feature lineage for safety + valid-sample masking. Do not infer.
FILTER_INPUTS: dict[str, tuple[str, ...]] = {
    "GAP_R015": ("gap_open_points", "atr_20"),
    "PDR_R105": ("prev_day_range", "atr_20"),
    "ATR_VEL_GE105": ("atr_vel_ratio",),
}


@dataclass(frozen=True)
class Trial:
    hypothesis_id: int
    hypothesis_name: str
    filter_type: str
    instrument: str
    session: str
    rr_target: float
    entry_model: str
    confirm_bars: int
    stop_multiplier: float


def load_trials(path: Path) -> list[Trial]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    trials: list[Trial] = []
    for h in data["hypotheses"]:
        scope = h["scope"]
        filter_type = str(h["filter"]["type"])
        for inst in scope["instruments"]:
            for sess in scope["sessions"]:
                for rr in scope["rr_targets"]:
                    for em in scope["entry_models"]:
                        for cb in scope["confirm_bars"]:
                            for sm in scope["stop_multipliers"]:
                                trials.append(
                                    Trial(
                                        hypothesis_id=int(h["id"]),
                                        hypothesis_name=str(h["name"]),
                                        filter_type=filter_type,
                                        instrument=str(inst),
                                        session=str(sess),
                                        rr_target=float(rr),
                                        entry_model=str(em),
                                        confirm_bars=int(cb),
                                        stop_multiplier=float(sm),
                                    )
                                )
    return trials


def assert_feature_safety(trial: Trial) -> None:
    cols = FILTER_INPUTS[trial.filter_type]
    unsafe = [c for c in cols if not is_feature_safe(c, trial.session)]
    if unsafe:
        raise ValueError(
            f"{trial.filter_type} for {trial.instrument} {trial.session} uses look-ahead columns: {unsafe}"
        )


def load_trial_frame(con: duckdb.DuckDBPyConnection, trial: Trial) -> pd.DataFrame:
    query = """
        SELECT
            o.trading_day AS outcome_day,
            o.pnl_r,
            df.*
        FROM orb_outcomes o
        JOIN daily_features df
          ON o.trading_day = df.trading_day
         AND o.symbol = df.symbol
         AND o.orb_minutes = df.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND o.orb_minutes = ?
          AND o.pnl_r IS NOT NULL
    """
    rows = con.execute(
        query,
        [
            trial.instrument,
            trial.session,
            trial.entry_model,
            trial.confirm_bars,
            trial.rr_target,
            ORB_MINUTES,
        ],
    ).fetchall()
    cols = [d[0] for d in con.description]
    df = pd.DataFrame.from_records(rows, columns=cols)
    if df.empty:
        return df
    df["outcome_day"] = pd.to_datetime(df["outcome_day"])
    df["year"] = df["outcome_day"].dt.year
    return df


def valid_mask(df: pd.DataFrame, filter_type: str) -> pd.Series:
    if filter_type == "GAP_R015":
        return df["gap_open_points"].notna() & df["atr_20"].notna() & (df["atr_20"] > 0)
    if filter_type == "PDR_R105":
        return df["prev_day_range"].notna() & df["atr_20"].notna() & (df["atr_20"] > 0)
    if filter_type == "ATR_VEL_GE105":
        return df["atr_vel_ratio"].notna()
    raise KeyError(filter_type)


def apply_filter(df: pd.DataFrame, trial: Trial) -> pd.Series:
    filt = ALL_FILTERS[trial.filter_type]
    flags = []
    for row in df.to_dict("records"):
        flags.append(bool(filt.matches_row(row, trial.session)))
    return pd.Series(flags, index=df.index)


def mean_or_nan(values: pd.Series) -> float:
    if len(values) == 0:
        return math.nan
    return float(values.mean())


def win_rate(values: pd.Series) -> float:
    if len(values) == 0:
        return math.nan
    return float((values > 0).mean())


def permutation_test(filtered: np.ndarray, excluded: np.ndarray, seed: int = 42) -> tuple[float, float]:
    if len(filtered) < MIN_GROUP_N or len(excluded) < MIN_GROUP_N:
        return math.nan, math.nan
    observed = float(filtered.mean() - excluded.mean())
    combined = np.concatenate([filtered, excluded])
    n_f = len(filtered)
    rng = np.random.default_rng(seed)
    null_diffs = np.empty(N_PERM, dtype=float)
    for i in range(N_PERM):
        perm = rng.permutation(combined)
        null_diffs[i] = perm[:n_f].mean() - perm[n_f:].mean()
    p_two_tailed = float((np.sum(np.abs(null_diffs) >= abs(observed)) + 1) / (len(null_diffs) + 1))
    return observed, p_two_tailed


def split_metrics(df: pd.DataFrame, mask: pd.Series) -> dict[str, float | int]:
    filtered = df.loc[mask, "pnl_r"]
    excluded = df.loc[~mask, "pnl_r"]
    lift, p_val = permutation_test(filtered.to_numpy(dtype=float), excluded.to_numpy(dtype=float))
    return {
        "n_total": int(len(df)),
        "n_filtered": int(len(filtered)),
        "n_excluded": int(len(excluded)),
        "wr_filtered": win_rate(filtered),
        "wr_excluded": win_rate(excluded),
        "wr_lift_pp": (win_rate(filtered) - win_rate(excluded)) * 100 if len(filtered) and len(excluded) else math.nan,
        "expr_filtered": mean_or_nan(filtered),
        "expr_excluded": mean_or_nan(excluded),
        "expr_lift": lift,
        "p_value": p_val,
    }


def year_stability(df: pd.DataFrame, mask: pd.Series) -> tuple[int, int]:
    positive = 0
    evaluable = 0
    for _, grp in df.groupby("year", sort=True):
        sub_mask = mask.loc[grp.index]
        filtered = grp.loc[sub_mask, "pnl_r"]
        excluded = grp.loc[~sub_mask, "pnl_r"]
        if len(filtered) < 10 or len(excluded) < 10:
            continue
        evaluable += 1
        if float(filtered.mean() - excluded.mean()) > 0:
            positive += 1
    return positive, evaluable


def bh_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    ranked = np.asarray(p_values, dtype=float)[order]
    adj = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adj[i] = running
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out.tolist()


def classify(row: dict) -> str:
    if row["is_n_filtered"] < 30:
        return "FAIL_N"
    if math.isnan(row["is_p_value"]) or math.isnan(row["is_expr_lift"]):
        return "FAIL_STATS"
    if row["is_expr_lift"] <= 0:
        return "FAIL_IS_DIR"
    if row["is_p_bh"] >= 0.05:
        return "FAIL_BH"
    if row["oos_n_filtered"] < 20:
        return "UNRESOLVED_OOS_N"
    if math.isnan(row["oos_expr_lift"]):
        return "FAIL_OOS_STATS"
    if row["oos_expr_lift"] <= 0:
        return "FAIL_OOS_DIR"
    if row["year_evaluable"] > 0 and row["year_positive"] / row["year_evaluable"] < 0.60:
        return "FAIL_YEAR_STAB"
    return "SURVIVES"


def main() -> int:
    trials = load_trials(HYP_PATH)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    results: list[dict] = []

    for trial in trials:
        assert_feature_safety(trial)
        frame = load_trial_frame(con, trial)
        if frame.empty:
            results.append(
                {
                    "hypothesis_id": trial.hypothesis_id,
                    "hypothesis_name": trial.hypothesis_name,
                    "filter_type": trial.filter_type,
                    "instrument": trial.instrument,
                    "session": trial.session,
                    "status": "NO_DATA",
                }
            )
            continue

        valid = frame.loc[valid_mask(frame, trial.filter_type)].copy()
        if valid.empty:
            results.append(
                {
                    "hypothesis_id": trial.hypothesis_id,
                    "hypothesis_name": trial.hypothesis_name,
                    "filter_type": trial.filter_type,
                    "instrument": trial.instrument,
                    "session": trial.session,
                    "status": "NO_VALID_ROWS",
                }
            )
            continue

        mask = apply_filter(valid, trial)
        is_df = valid.loc[valid["outcome_day"] < HOLDOUT_DATE].copy()
        oos_df = valid.loc[valid["outcome_day"] >= HOLDOUT_DATE].copy()
        is_mask = mask.loc[is_df.index]
        oos_mask = mask.loc[oos_df.index]

        is_metrics = split_metrics(is_df, is_mask) if not is_df.empty else {}
        oos_metrics = split_metrics(oos_df, oos_mask) if not oos_df.empty else {}
        year_pos, year_eval = year_stability(is_df, is_mask) if not is_df.empty else (0, 0)

        results.append(
            {
                "hypothesis_id": trial.hypothesis_id,
                "hypothesis_name": trial.hypothesis_name,
                "filter_type": trial.filter_type,
                "instrument": trial.instrument,
                "session": trial.session,
                "status": "OK",
                "is_n_total": is_metrics.get("n_total", 0),
                "is_n_filtered": is_metrics.get("n_filtered", 0),
                "is_n_excluded": is_metrics.get("n_excluded", 0),
                "is_wr_lift_pp": is_metrics.get("wr_lift_pp", math.nan),
                "is_expr_filtered": is_metrics.get("expr_filtered", math.nan),
                "is_expr_excluded": is_metrics.get("expr_excluded", math.nan),
                "is_expr_lift": is_metrics.get("expr_lift", math.nan),
                "is_p_value": is_metrics.get("p_value", math.nan),
                "oos_n_total": oos_metrics.get("n_total", 0),
                "oos_n_filtered": oos_metrics.get("n_filtered", 0),
                "oos_n_excluded": oos_metrics.get("n_excluded", 0),
                "oos_wr_lift_pp": oos_metrics.get("wr_lift_pp", math.nan),
                "oos_expr_filtered": oos_metrics.get("expr_filtered", math.nan),
                "oos_expr_excluded": oos_metrics.get("expr_excluded", math.nan),
                "oos_expr_lift": oos_metrics.get("expr_lift", math.nan),
                "oos_p_value": oos_metrics.get("p_value", math.nan),
                "year_positive": year_pos,
                "year_evaluable": year_eval,
            }
        )

    con.close()

    ok_rows = [r for r in results if r["status"] == "OK"]
    bh_vals = bh_adjust([float(r["is_p_value"]) for r in ok_rows])
    for row, p_bh in zip(ok_rows, bh_vals, strict=True):
        row["is_p_bh"] = p_bh
        row["decision"] = classify(row)

    out = pd.DataFrame(results)
    if "decision" not in out.columns:
        out["decision"] = None
    if "is_p_bh" not in out.columns:
        out["is_p_bh"] = math.nan
    out = out.sort_values(["decision", "is_p_bh", "hypothesis_id", "instrument", "session"], na_position="last")

    display_cols = [
        "hypothesis_id",
        "filter_type",
        "instrument",
        "session",
        "is_n_filtered",
        "is_expr_lift",
        "is_p_value",
        "is_p_bh",
        "oos_n_filtered",
        "oos_expr_lift",
        "year_positive",
        "year_evaluable",
        "decision",
    ]
    print(out[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDecision counts:")
    print(out["decision"].fillna(out["status"]).value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
