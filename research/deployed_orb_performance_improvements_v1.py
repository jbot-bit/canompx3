"""Targeted replay for 2026-04-21 exact deployed ORB performance improvements.

Reads the locked hypothesis family from:
  docs/audit/hypotheses/2026-04-21-deployed-orb-performance-improvements-v1.yaml

Method:
- canonical truth only: daily_features + orb_outcomes
- exact live-lane filtered universes only (not a broad session scan)
- thresholds fit on pre-2025 rows only
- 2025 used as OOS-CV
- 2026+ descriptive only
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from pipeline.session_guard import is_feature_safe
from trading_app.strategy_fitness import _load_strategy_outcomes
from trading_app.strategy_validator import benjamini_hochberg

HYP_PATH = Path("docs/audit/hypotheses/2026-04-21-deployed-orb-performance-improvements-v1.yaml")
RESULT_MD = Path("docs/audit/results/2026-04-21-deployed-orb-performance-improvements-v1.md")

IS_END = pd.Timestamp("2024-12-31").date()
OOS_START = pd.Timestamp("2025-01-01").date()
OOS_END = pd.Timestamp("2025-12-31").date()
FWD_START = pd.Timestamp("2026-01-01").date()
BH_Q = 0.10


@dataclass(frozen=True)
class WindowStats:
    n_on: int
    n_off: int
    avg_on: float | None
    avg_off: float | None
    delta: float | None
    sharpe_on: float | None
    p_one_tailed: float | None


def _mean(series: pd.Series) -> float | None:
    if len(series) == 0:
        return None
    return float(series.mean())


def _sharpe(series: pd.Series) -> float | None:
    if len(series) < 2:
        return None
    std = float(series.std(ddof=1))
    if std <= 0 or math.isnan(std):
        return None
    return float(series.mean() / std)


def _one_tailed_p(on: pd.Series, off: pd.Series) -> float | None:
    if len(on) < 2 or len(off) < 2:
        return None
    t_stat, p_two = stats.ttest_ind(on, off, equal_var=False)
    if math.isnan(t_stat) or math.isnan(p_two):
        return None
    if t_stat > 0:
        return float(p_two / 2.0)
    return float(1.0 - (p_two / 2.0))


def _window_stats(df: pd.DataFrame, mask: pd.Series) -> WindowStats:
    sub = df.loc[mask].copy()
    on = sub.loc[sub["signal"] == 1, "pnl_r"].astype(float)
    off = sub.loc[sub["signal"] == 0, "pnl_r"].astype(float)
    avg_on = _mean(on)
    avg_off = _mean(off)
    delta = None if avg_on is None or avg_off is None else float(avg_on - avg_off)
    return WindowStats(
        n_on=int(len(on)),
        n_off=int(len(off)),
        avg_on=avg_on,
        avg_off=avg_off,
        delta=delta,
        sharpe_on=_sharpe(on),
        p_one_tailed=_one_tailed_p(on, off),
    )


def _threshold_rule_to_quantile(rule: str) -> float:
    mapping = {
        "IS_P75": 0.75,
        "IS_P90": 0.90,
    }
    try:
        return mapping[rule]
    except KeyError as exc:
        raise ValueError(f"Unknown threshold rule {rule!r}") from exc


def _load_lane_frame(con: duckdb.DuckDBPyConnection, lane: dict[str, Any]) -> pd.DataFrame:
    outcomes = _load_strategy_outcomes(
        con,
        instrument=lane["instrument"],
        orb_label=lane["session"],
        orb_minutes=lane["orb_minutes"],
        entry_model="E2",
        rr_target=lane["rr_target"],
        confirm_bars=lane["confirm_bars"],
        filter_type=lane["base_filter"],
        start_date="2019-01-01",
        end_date="2026-12-31",
    )
    out_df = pd.DataFrame(outcomes)
    if out_df.empty:
        return out_df
    out_df["trading_day"] = pd.to_datetime(out_df["trading_day"]).dt.date
    feat_df = con.execute(
        "SELECT * FROM daily_features WHERE symbol = ? AND orb_minutes = ? AND trading_day <= DATE '2026-12-31'",
        [lane["instrument"], lane["orb_minutes"]],
    ).df()
    feat_df["trading_day"] = pd.to_datetime(feat_df["trading_day"]).dt.date
    df = out_df.merge(feat_df, on="trading_day", how="inner", suffixes=("", "_feat"))
    df = df.sort_values("trading_day").reset_index(drop=True)
    return df


def _build_signal(df: pd.DataFrame, feature_name: str, rule: str) -> tuple[pd.Series, float]:
    q = _threshold_rule_to_quantile(rule)
    feature = df[feature_name]
    fit_mask = (df["trading_day"] <= IS_END) & feature.notna()
    if int(fit_mask.sum()) == 0:
        raise ValueError(f"No pre-2025 rows available to fit {feature_name} {rule}")
    threshold = float(feature.loc[fit_mask].quantile(q))
    signal = (feature.astype(float) >= threshold).astype(int)
    return signal, threshold


def main() -> int:
    if not HYP_PATH.exists():
        raise FileNotFoundError(f"Hypothesis file missing: {HYP_PATH}")

    hyp = yaml.safe_load(HYP_PATH.read_text(encoding="utf-8"))
    lanes = {lane["lane_id"]: lane for lane in hyp["scope"]["exact_live_lanes"]}
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    lane_frames: dict[str, pd.DataFrame] = {}
    lane_baselines: dict[str, dict[str, float | int | None]] = {}
    for lane_id, lane in lanes.items():
        frame = _load_lane_frame(con, lane)
        if frame.empty:
            raise ValueError(f"Lane frame empty for {lane_id}")
        lane_frames[lane_id] = frame
        is_mask = frame["trading_day"] <= IS_END
        oos_mask = (frame["trading_day"] >= OOS_START) & (frame["trading_day"] <= OOS_END)
        lane_baselines[lane_id] = {
            "n_is": int(is_mask.sum()),
            "avg_r_is": _mean(frame.loc[is_mask, "pnl_r"].astype(float)),
            "sharpe_is": _sharpe(frame.loc[is_mask, "pnl_r"].astype(float)),
            "n_oos": int(oos_mask.sum()),
            "avg_r_oos": _mean(frame.loc[oos_mask, "pnl_r"].astype(float)),
            "sharpe_oos": _sharpe(frame.loc[oos_mask, "pnl_r"].astype(float)),
        }

    raw_results: list[dict[str, Any]] = []
    p_values: list[tuple[str, float]] = []

    for hypothesis in hyp["hypotheses"]:
        lane = lanes[hypothesis["lane_id"]]
        feature_name = hypothesis["feature_name"]
        if not is_feature_safe(feature_name, lane["session"]):
            raise ValueError(f"Unsafe feature {feature_name} for session {lane['session']}")
        df = lane_frames[hypothesis["lane_id"]].copy()
        if feature_name not in df.columns:
            raise ValueError(f"Feature {feature_name} missing for {hypothesis['lane_id']}")
        for rule in hypothesis["threshold_rules"]:
            signal, threshold = _build_signal(df, feature_name, rule)
            df["signal"] = signal
            is_stats = _window_stats(df, df["trading_day"] <= IS_END)
            oos_stats = _window_stats(
                df,
                (df["trading_day"] >= OOS_START) & (df["trading_day"] <= OOS_END),
            )
            fwd_stats = _window_stats(df, df["trading_day"] >= FWD_START)
            strategy_key = f"{hypothesis['id']}::{rule}"
            p_raw = is_stats.p_one_tailed
            if p_raw is not None:
                p_values.append((strategy_key, p_raw))
            wfe = None
            if (
                is_stats.sharpe_on is not None
                and oos_stats.sharpe_on is not None
                and is_stats.sharpe_on > 0
            ):
                wfe = float(oos_stats.sharpe_on / is_stats.sharpe_on)
            raw_results.append(
                {
                    "strategy_key": strategy_key,
                    "hypothesis_id": hypothesis["id"],
                    "lane_id": hypothesis["lane_id"],
                    "session": lane["session"],
                    "base_filter": lane["base_filter"],
                    "feature_name": feature_name,
                    "threshold_rule": rule,
                    "threshold_value": threshold,
                    "baseline_is_n": lane_baselines[hypothesis["lane_id"]]["n_is"],
                    "baseline_is_avg_r": lane_baselines[hypothesis["lane_id"]]["avg_r_is"],
                    "baseline_oos_n": lane_baselines[hypothesis["lane_id"]]["n_oos"],
                    "baseline_oos_avg_r": lane_baselines[hypothesis["lane_id"]]["avg_r_oos"],
                    "n_is_on": is_stats.n_on,
                    "n_is_off": is_stats.n_off,
                    "avg_r_is_on": is_stats.avg_on,
                    "avg_r_is_off": is_stats.avg_off,
                    "delta_is": is_stats.delta,
                    "sharpe_is_on": is_stats.sharpe_on,
                    "p_raw_one_tailed": p_raw,
                    "n_oos_on": oos_stats.n_on,
                    "n_oos_off": oos_stats.n_off,
                    "avg_r_oos_on": oos_stats.avg_on,
                    "avg_r_oos_off": oos_stats.avg_off,
                    "delta_oos": oos_stats.delta,
                    "sharpe_oos_on": oos_stats.sharpe_on,
                    "wfe": wfe,
                    "n_fwd_on": fwd_stats.n_on,
                    "avg_r_fwd_on": fwd_stats.avg_on,
                    "delta_fwd": fwd_stats.delta,
                }
            )

    bh = benjamini_hochberg(p_values, alpha=BH_Q, total_tests=hyp["family_k"]["global_k"])
    for row in raw_results:
        bh_row = bh.get(row["strategy_key"])
        row["p_bh"] = None if bh_row is None else bh_row["adjusted_p"]
        row["bh_significant"] = False if bh_row is None else bool(bh_row["fdr_significant"])
        row["survives"] = bool(
            row["bh_significant"]
            and row["delta_is"] is not None
            and row["delta_is"] > 0
            and row["n_oos_on"] >= 30
            and row["avg_r_oos_on"] is not None
            and row["avg_r_oos_on"] > 0
            and row["wfe"] is not None
            and row["wfe"] >= 0.50
        )

    results_df = pd.DataFrame(raw_results).sort_values(
        ["survives", "p_bh", "delta_oos", "delta_is"],
        ascending=[False, True, False, False],
        na_position="last",
    )

    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Exact Deployed ORB Performance Improvements — V1")
    lines.append("")
    lines.append(f"- Hypothesis file: `{HYP_PATH}`")
    lines.append(f"- Global K: {hyp['family_k']['global_k']}")
    lines.append(f"- BH q: {hyp['family_k']['bh_q']}")
    lines.append("- IS window: pre-2025")
    lines.append("- OOS-CV window: 2025 calendar year")
    lines.append("- Forward window: 2026+ descriptive only")
    lines.append("")
    survivors = results_df[results_df["survives"]]
    lines.append(f"- Survivors: {len(survivors)}/{len(results_df)}")
    lines.append("")
    if len(results_df):
        lines.append("| Hypothesis | Lane | Feature | Rule | N_IS_on | ΔIS | p_raw | p_bh | N_OOS_on | OOS_on | ΔOOS | WFE | Survivor |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in results_df.to_dict("records"):
            lines.append(
                "| {hypothesis_id} | {lane_id} | {feature_name} | {threshold_rule} | {n_is_on} | {delta_is} | {p_raw_one_tailed} | {p_bh} | {n_oos_on} | {avg_r_oos_on} | {delta_oos} | {wfe} | {survives} |".format(
                    hypothesis_id=row["hypothesis_id"],
                    lane_id=row["lane_id"],
                    feature_name=row["feature_name"],
                    threshold_rule=row["threshold_rule"],
                    n_is_on=row["n_is_on"],
                    delta_is="-" if row["delta_is"] is None else f"{row['delta_is']:.6f}",
                    p_raw_one_tailed="-" if row["p_raw_one_tailed"] is None else f"{row['p_raw_one_tailed']:.6f}",
                    p_bh="-" if row["p_bh"] is None else f"{row['p_bh']:.6f}",
                    n_oos_on=row["n_oos_on"],
                    avg_r_oos_on="-" if row["avg_r_oos_on"] is None else f"{row['avg_r_oos_on']:.6f}",
                    delta_oos="-" if row["delta_oos"] is None else f"{row['delta_oos']:.6f}",
                    wfe="-" if row["wfe"] is None else f"{row['wfe']:.3f}",
                    survives="YES" if row["survives"] else "NO",
                )
            )

    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(raw_results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
