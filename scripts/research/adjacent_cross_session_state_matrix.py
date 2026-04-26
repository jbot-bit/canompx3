#!/usr/bin/env python3
"""Read-only adjacent cross-session state matrix audit.

This is the broad, literature-grounded non-ML pass:

- adjacent prior->later handoff only
- four-state continuation/reversal family
- internal canonical ORB state only
- pre-trade safety enforced via session_guard
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

HOLDOUT_DATE = pd.Timestamp("2026-01-01")
ROOT = Path(__file__).resolve().parent.parent.parent
HYP_PATH = ROOT / "docs" / "audit" / "hypotheses" / "2026-04-11-adjacent-cross-session-state-matrix.yaml"
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.0
N_PERM = 5000
MIN_GROUP_N = 20


@dataclass(frozen=True)
class Trial:
    group: str
    prior_session: str
    later_session: str
    mode: str
    state: str


def load_trials(path: Path) -> tuple[list[Trial], dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    trials: list[Trial] = []
    for group in data["groups"]:
        for pair in data["pairs"]:
            for state_spec in data["state_family"]:
                trials.append(
                    Trial(
                        group=str(group),
                        prior_session=str(pair["prior_session"]),
                        later_session=str(pair["later_session"]),
                        mode=str(state_spec["mode"]),
                        state=str(state_spec["state"]),
                    )
                )
    return trials, data


def symbols_for_group(group: str) -> list[str]:
    if group == "MES+MNQ":
        return ["MES", "MNQ"]
    return [group]


def assert_feature_safety(trial: Trial) -> None:
    prior_dir_col = f"orb_{trial.prior_session}_break_dir"
    prior_outcome_col = f"orb_{trial.prior_session}_outcome"
    unsafe = [c for c in (prior_dir_col, prior_outcome_col) if not is_feature_safe(c, trial.later_session)]
    if unsafe:
        raise ValueError(
            f"{trial.prior_session}->{trial.later_session} for {trial.group} uses look-ahead columns: {unsafe}"
        )


def load_trial_frame(con: duckdb.DuckDBPyConnection, trial: Trial) -> pd.DataFrame:
    symbols = symbols_for_group(trial.group)
    sym_list = ", ".join(f"'{s}'" for s in symbols)
    prior_dir_col = f"orb_{trial.prior_session}_break_dir"
    prior_outcome_col = f"orb_{trial.prior_session}_outcome"
    query = f"""
        SELECT
            o.symbol,
            o.trading_day AS outcome_day,
            o.entry_price,
            o.target_price,
            o.pnl_r,
            df.{prior_dir_col} AS prior_dir,
            df.{prior_outcome_col} AS prior_outcome
        FROM orb_outcomes o
        JOIN daily_features df
          ON o.trading_day = df.trading_day
         AND o.symbol = df.symbol
         AND o.orb_minutes = df.orb_minutes
        WHERE o.symbol IN ({sym_list})
          AND o.orb_label = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND o.orb_minutes = ?
          AND o.outcome IN ('win', 'loss')
          AND o.pnl_r IS NOT NULL
          AND o.entry_price IS NOT NULL
          AND o.target_price IS NOT NULL
    """
    rows = con.execute(
        query,
        [trial.later_session, ENTRY_MODEL, CONFIRM_BARS, RR_TARGET, ORB_MINUTES],
    ).fetchall()
    cols = [d[0] for d in con.description]
    df = pd.DataFrame.from_records(rows, columns=cols)
    if df.empty:
        return df
    df["outcome_day"] = pd.to_datetime(df["outcome_day"])
    df["year"] = df["outcome_day"].dt.year
    df["trade_direction"] = np.where(df["target_price"] > df["entry_price"], "long", "short")
    return df


def valid_mask(df: pd.DataFrame) -> pd.Series:
    return df["prior_dir"].isin(["long", "short"]) & df["prior_outcome"].isin(["win", "loss"])


def state_mask(df: pd.DataFrame, state: str) -> pd.Series:
    same_dir = df["prior_dir"] == df["trade_direction"]
    opposed_dir = df["prior_dir"] != df["trade_direction"]
    if state == "PRIOR_WIN_ALIGN":
        return (df["prior_outcome"] == "win") & same_dir
    if state == "PRIOR_WIN_OPPOSED":
        return (df["prior_outcome"] == "win") & opposed_dir
    if state == "PRIOR_LOSS_ALIGN":
        return (df["prior_outcome"] == "loss") & same_dir
    if state == "PRIOR_LOSS_OPPOSED":
        return (df["prior_outcome"] == "loss") & opposed_dir
    raise KeyError(state)


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


def split_metrics(df: pd.DataFrame, mask: pd.Series, mode: str) -> dict[str, float | int]:
    filtered = df.loc[mask, "pnl_r"]
    excluded = df.loc[~mask, "pnl_r"]
    raw_lift, p_val = permutation_test(filtered.to_numpy(dtype=float), excluded.to_numpy(dtype=float))
    if mode == "TAKE":
        benefit = raw_lift
    elif mode == "VETO":
        benefit = -raw_lift if not math.isnan(raw_lift) else math.nan
    else:
        raise KeyError(mode)
    return {
        "n_total": int(len(df)),
        "n_filtered": int(len(filtered)),
        "n_excluded": int(len(excluded)),
        "benefit": benefit,
        "p_value": p_val,
    }


def oos_benefit(df: pd.DataFrame, mask: pd.Series, mode: str) -> float:
    filtered = df.loc[mask, "pnl_r"]
    excluded = df.loc[~mask, "pnl_r"]
    if len(filtered) == 0 or len(excluded) == 0:
        return math.nan
    raw = float(filtered.mean() - excluded.mean())
    return raw if mode == "TAKE" else -raw


def year_stability(df: pd.DataFrame, mask: pd.Series, mode: str) -> tuple[int, int]:
    positive = 0
    evaluable = 0
    for _, grp in df.groupby("year", sort=True):
        sub_mask = mask.loc[grp.index]
        filtered = grp.loc[sub_mask, "pnl_r"]
        excluded = grp.loc[~sub_mask, "pnl_r"]
        if len(filtered) < 10 or len(excluded) < 10:
            continue
        evaluable += 1
        raw = float(filtered.mean() - excluded.mean())
        benefit = raw if mode == "TAKE" else -raw
        if benefit > 0:
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
    if math.isnan(row["is_benefit"]) or math.isnan(row["is_p_value"]):
        return "FAIL_STATS"
    if row["is_benefit"] <= 0:
        return "FAIL_IS_DIR"
    if row["is_p_bh"] >= 0.05:
        return "FAIL_BH"
    if row["oos_n_filtered"] < 10:
        return "UNRESOLVED_OOS_N"
    if math.isnan(row["oos_benefit"]):
        return "FAIL_OOS_STATS"
    if row["oos_benefit"] <= 0:
        return "FAIL_OOS_DIR"
    if row["year_evaluable"] > 0 and row["year_positive"] / row["year_evaluable"] < 0.60:
        return "FAIL_YEAR_STAB"
    return "SURVIVES"


def main() -> int:
    trials, meta = load_trials(HYP_PATH)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    results: list[dict] = []

    for trial in trials:
        assert_feature_safety(trial)
        frame = load_trial_frame(con, trial)
        if frame.empty:
            results.append(
                {
                    "group": trial.group,
                    "prior_session": trial.prior_session,
                    "later_session": trial.later_session,
                    "mode": trial.mode,
                    "state": trial.state,
                    "status": "NO_DATA",
                }
            )
            continue

        valid = frame.loc[valid_mask(frame)].copy()
        if valid.empty:
            results.append(
                {
                    "group": trial.group,
                    "prior_session": trial.prior_session,
                    "later_session": trial.later_session,
                    "mode": trial.mode,
                    "state": trial.state,
                    "status": "NO_VALID_ROWS",
                }
            )
            continue

        mask = state_mask(valid, trial.state)
        is_df = valid.loc[valid["outcome_day"] < HOLDOUT_DATE].copy()
        oos_df = valid.loc[valid["outcome_day"] >= HOLDOUT_DATE].copy()
        is_mask = mask.loc[is_df.index]
        oos_mask = mask.loc[oos_df.index]

        is_metrics = split_metrics(is_df, is_mask, trial.mode) if not is_df.empty else {}
        year_pos, year_eval = year_stability(is_df, is_mask, trial.mode) if not is_df.empty else (0, 0)

        results.append(
            {
                "group": trial.group,
                "prior_session": trial.prior_session,
                "later_session": trial.later_session,
                "mode": trial.mode,
                "state": trial.state,
                "status": "OK",
                "is_n_filtered": is_metrics.get("n_filtered", 0),
                "is_benefit": is_metrics.get("benefit", math.nan),
                "is_p_value": is_metrics.get("p_value", math.nan),
                "oos_n_filtered": int(oos_mask.sum()) if not oos_df.empty else 0,
                "oos_benefit": oos_benefit(oos_df, oos_mask, trial.mode) if not oos_df.empty else math.nan,
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
    out = out.sort_values(
        ["decision", "is_p_bh", "group", "later_session", "prior_session", "mode", "state"],
        na_position="last",
    )

    display_cols = [
        "group",
        "prior_session",
        "later_session",
        "mode",
        "state",
        "is_n_filtered",
        "is_benefit",
        "is_p_value",
        "is_p_bh",
        "oos_n_filtered",
        "oos_benefit",
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
