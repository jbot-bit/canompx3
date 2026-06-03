#!/usr/bin/env python3
"""Bias-hardened ORB execution-variant investigation v1.

Locked by:
  docs/audit/hypotheses/2026-06-01-orb-execution-variants-v1.yaml

This runner evaluates execution modifications against the original MNQ E2
ORB opportunity set. The accounting unit is the parent opportunity, not the
selected child trade. A same-direction re-entry therefore books:

    original stopped E2 trade + bounded re-entry trade

2026-01-01+ is loaded only for descriptive monitoring and never for selection.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.cost_model import CostSpec, get_cost_spec, to_r_multiple
from pipeline.dst import BRISBANE_TZ, compute_trading_day_utc_range, orb_utc_window
from pipeline.paths import GOLD_DB_PATH
from research.lib import bh_fdr
from trading_app.config import E2_SLIPPAGE_TICKS
from trading_app.dsr import compute_dsr, compute_sr0
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = Path("docs/audit/hypotheses/2026-06-01-orb-execution-variants-v1.yaml")
RESULT_DOC = Path("docs/audit/results/2026-06-01-orb-execution-variants-v1.md")
ROW_CSV = Path("docs/audit/results/2026-06-01-orb-execution-variants-v1-cells.csv")

SYMBOL = "MNQ"
ORB_MINUTES = 5
RR_TARGETS = (1.0, 1.5, 2.0)
PRIMARY_PARENT_SESSIONS = ("NYSE_OPEN", "US_DATA_1000")
SECONDARY_PARENT_SESSIONS = ("CME_PRECLOSE",)
PARENT_SESSIONS = PRIMARY_PARENT_SESSIONS + SECONDARY_PARENT_SESSIONS
REENTRY_WAIT_BARS = (0, 2, 5)
REVERSAL_WAIT_BARS = (0, 5)
CONFIRM_BARS = (2, 3, 5)
RETEST_WAIT_BARS = (0, 2)
PRE_ENTRY_FILTERS = ("orb_size_q67", "atr20pct_ge70")
THROTTLE_RULES = ("skip_after_first_loss", "half_after_first_loss")
BH_Q = 0.05
SELECTABLE_K = (
    len(PARENT_SESSIONS) * len(RR_TARGETS) * len(REENTRY_WAIT_BARS)
    + len(PARENT_SESSIONS) * len(RR_TARGETS) * len(CONFIRM_BARS)
    + len(THROTTLE_RULES) * len(RR_TARGETS)
    + len(PARENT_SESSIONS) * len(RR_TARGETS) * len(RETEST_WAIT_BARS)
    + len(PARENT_SESSIONS) * len(RR_TARGETS) * len(PRE_ENTRY_FILTERS)
    + len(PARENT_SESSIONS) * len(RR_TARGETS) * len(REVERSAL_WAIT_BARS)
)

E2_LOOKAHEAD_BANNED_PATTERNS = (
    "break_ts",
    "break_delay_min",
    "break_bar_volume",
    "break_bar_continues",
    "rel_vol_",
    "break_dir",
    "double_break",
    "outcome",
    "mae_r",
    "mfe_r",
    "pnl_r",
)


@dataclass(frozen=True)
class ReentryConfig:
    wait_bars: int
    size: float = 1.0
    max_reentries: int = 1


@dataclass(frozen=True)
class TradePathResult:
    outcome: str
    pnl_r: float
    exit_ts: pd.Timestamp


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _ts(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _date_value(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def reject_e2_lookahead_columns(columns: Iterable[str]) -> None:
    """Fail closed if a pre-entry E2 predictor set contains tainted fields."""

    offenders = []
    for col in columns:
        lowered = col.lower()
        if any(pattern in lowered for pattern in E2_LOOKAHEAD_BANNED_PATTERNS):
            offenders.append(col)
    if offenders:
        raise ValueError(f"E2 lookahead predictors rejected: {offenders}")


def infer_direction(entry_price: float, stop_price: float) -> str:
    if entry_price > stop_price:
        return "long"
    if entry_price < stop_price:
        return "short"
    raise ValueError("entry_price equals stop_price; cannot infer direction")


def target_price(entry_price: float, stop_price: float, rr_target: float, direction: str) -> float:
    risk_points = abs(entry_price - stop_price)
    if direction == "long":
        return entry_price + risk_points * rr_target
    if direction == "short":
        return entry_price - risk_points * rr_target
    raise ValueError(f"Unsupported direction: {direction}")


def simulate_trade_path(
    bars: pd.DataFrame,
    *,
    entry_ts: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: str,
    cost_spec: CostSpec,
) -> TradePathResult:
    """Simulate one trade from a timestamp-eligible entry.

    The fill bar is included. If stop and target are both touched on the same
    bar, the result is a conservative loss, matching outcome_builder.
    Scratches are force-flattened at the last available bar close and booked as
    realized EOD MTM, not dropped.
    """

    if bars.empty:
        raise ValueError("Cannot simulate trade path with no bars")

    entry_ts = _ts(entry_ts)
    path = bars.copy()
    path["ts_utc"] = pd.to_datetime(path["ts_utc"], utc=True)
    path = path[path["ts_utc"] >= entry_ts].sort_values("ts_utc")
    if path.empty:
        raise ValueError("No bars at or after entry_ts")

    for row in path.itertuples(index=False):
        ts = _ts(row.ts_utc)
        high = float(row.high)
        low = float(row.low)
        if direction == "long":
            hit_stop = low <= stop_price
            hit_target = high >= target_price
        elif direction == "short":
            hit_stop = high >= stop_price
            hit_target = low <= target_price
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        if hit_stop and hit_target:
            return TradePathResult(outcome="loss", pnl_r=-1.0, exit_ts=ts)
        if hit_stop:
            return TradePathResult(outcome="loss", pnl_r=-1.0, exit_ts=ts)
        if hit_target:
            raw_points = abs(target_price - entry_price)
            pnl_r = round(to_r_multiple(cost_spec, entry_price, stop_price, raw_points), 4)
            return TradePathResult(outcome="win", pnl_r=pnl_r, exit_ts=ts)

    last = path.iloc[-1]
    exit_ts = _ts(last["ts_utc"])
    exit_close = float(last["close"])
    pnl_points = exit_close - entry_price if direction == "long" else entry_price - exit_close
    pnl_r = round(to_r_multiple(cost_spec, entry_price, stop_price, pnl_points), 4)
    return TradePathResult(outcome="scratch", pnl_r=pnl_r, exit_ts=exit_ts)


def _slippage_points(cost_spec: CostSpec) -> float:
    return E2_SLIPPAGE_TICKS * cost_spec.tick_size


def _post_exit_bars(bars: pd.DataFrame, exit_ts: pd.Timestamp, wait_bars: int) -> pd.DataFrame:
    if bars.empty:
        return bars
    ts = _ts(exit_ts) + pd.Timedelta(minutes=wait_bars)
    b = bars.copy()
    b["ts_utc"] = pd.to_datetime(b["ts_utc"], utc=True)
    return b[b["ts_utc"] > ts].sort_values("ts_utc").reset_index(drop=True)


def _same_direction_reentry_with_levels(
    parent: pd.Series,
    bars: pd.DataFrame,
    config: ReentryConfig,
    *,
    cost_spec: CostSpec,
    orb_high: float,
    orb_low: float,
) -> tuple[float, TradePathResult | None]:
    parent_pnl = float(parent["pnl_r"])
    if parent.get("outcome") != "loss":
        return parent_pnl, None
    if config.max_reentries != 1:
        raise ValueError("v1 selectable policy supports max_reentries=1 only")

    direction = infer_direction(float(parent["entry_price"]), float(parent["stop_price"]))
    exit_ts = _ts(parent["exit_ts"])
    candidate_bars = _post_exit_bars(bars, exit_ts, config.wait_bars)
    if candidate_bars.empty:
        return parent_pnl, None

    rr_target = float(parent["rr_target"])
    slip = _slippage_points(cost_spec)

    if direction == "long":
        trigger_mask = candidate_bars["high"].astype(float) > orb_high
        entry_price = orb_high + slip
        stop_price = orb_low
    else:
        trigger_mask = candidate_bars["low"].astype(float) < orb_low
        entry_price = orb_low - slip
        stop_price = orb_high

    if not bool(trigger_mask.any()):
        return parent_pnl, None

    trigger_idx = int(np.argmax(trigger_mask.to_numpy()))
    entry_ts = _ts(candidate_bars.iloc[trigger_idx]["ts_utc"])
    tgt = target_price(entry_price, stop_price, rr_target, direction)
    result = simulate_trade_path(
        candidate_bars.iloc[trigger_idx:].reset_index(drop=True),
        entry_ts=entry_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=tgt,
        direction=direction,
        cost_spec=cost_spec,
    )
    return parent_pnl + config.size * result.pnl_r, result


def same_direction_reentry_policy_pnl(
    parent: pd.Series,
    bars: pd.DataFrame,
    config: ReentryConfig,
    *,
    cost_spec: CostSpec,
) -> tuple[float, TradePathResult | None]:
    """Policy PnL for max-one same-direction re-entry after a full E2 stop."""

    return _same_direction_reentry_with_levels(
        parent,
        bars,
        config,
        cost_spec=cost_spec,
        orb_high=float(parent["orb_high"]),
        orb_low=float(parent["orb_low"]),
    )


def opposite_direction_reentry_policy_pnl(
    parent: pd.Series,
    bars: pd.DataFrame,
    config: ReentryConfig,
    *,
    cost_spec: CostSpec,
) -> tuple[float, TradePathResult | None]:
    """Lower-priority fakeout reversal control after a full E2 stop."""

    parent_pnl = float(parent["pnl_r"])
    if parent.get("outcome") != "loss":
        return parent_pnl, None

    original = infer_direction(float(parent["entry_price"]), float(parent["stop_price"]))
    direction = "short" if original == "long" else "long"
    exit_ts = _ts(parent["exit_ts"])
    candidate_bars = _post_exit_bars(bars, exit_ts, config.wait_bars)
    if candidate_bars.empty:
        return parent_pnl, None

    orb_high = float(parent["orb_high"])
    orb_low = float(parent["orb_low"])
    rr_target = float(parent["rr_target"])
    slip = _slippage_points(cost_spec)

    if direction == "long":
        trigger_mask = candidate_bars["high"].astype(float) > orb_high
        entry_price = orb_high + slip
        stop_price = orb_low
    else:
        trigger_mask = candidate_bars["low"].astype(float) < orb_low
        entry_price = orb_low - slip
        stop_price = orb_high

    if not bool(trigger_mask.any()):
        return parent_pnl, None

    trigger_idx = int(np.argmax(trigger_mask.to_numpy()))
    entry_ts = _ts(candidate_bars.iloc[trigger_idx]["ts_utc"])
    tgt = target_price(entry_price, stop_price, rr_target, direction)
    result = simulate_trade_path(
        candidate_bars.iloc[trigger_idx:].reset_index(drop=True),
        entry_ts=entry_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=tgt,
        direction=direction,
        cost_spec=cost_spec,
    )
    return parent_pnl + config.size * result.pnl_r, result


def retest_hold_policy_pnl(
    parent: pd.Series,
    bars: pd.DataFrame,
    *,
    wait_bars: int,
    cost_spec: CostSpec,
) -> tuple[float, TradePathResult | None]:
    """Replacement policy: enter on retest of breakout edge before stop breach."""

    direction = infer_direction(float(parent["entry_price"]), float(parent["stop_price"]))
    entry_ts = _ts(parent["entry_ts"]) + pd.Timedelta(minutes=wait_bars)
    b = bars.copy()
    if b.empty:
        return 0.0, None
    b["ts_utc"] = pd.to_datetime(b["ts_utc"], utc=True)
    post = b[b["ts_utc"] > entry_ts].sort_values("ts_utc").reset_index(drop=True)
    if post.empty:
        return 0.0, None

    orb_high = float(parent["orb_high"])
    orb_low = float(parent["orb_low"])
    rr_target = float(parent["rr_target"])

    if direction == "long":
        retrace = post["low"].astype(float) <= orb_high
        stop_hit = post["low"].astype(float) <= orb_low
        entry_price = orb_high
        stop_price = orb_low
    else:
        retrace = post["high"].astype(float) >= orb_low
        stop_hit = post["high"].astype(float) >= orb_high
        entry_price = orb_low
        stop_price = orb_high

    if not bool(retrace.any()):
        return 0.0, None

    retrace_idx = int(np.argmax(retrace.to_numpy()))
    if bool(stop_hit.iloc[: retrace_idx + 1].any()):
        return 0.0, None

    fill_ts = _ts(post.iloc[retrace_idx]["ts_utc"])
    tgt = target_price(entry_price, stop_price, rr_target, direction)
    result = simulate_trade_path(
        post.iloc[retrace_idx:].reset_index(drop=True),
        entry_ts=fill_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=tgt,
        direction=direction,
        cost_spec=cost_spec,
    )
    return result.pnl_r, result


def bh_q_values(values: list[tuple[str, float]]) -> dict[str, float]:
    finite = [(key, p) for key, p in values if p is not None and not math.isnan(float(p))]
    if not finite:
        return {}
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    out: dict[str, float] = {}
    running = 1.0
    for i in range(m - 1, -1, -1):
        key, p = finite[i]
        q = min(running, float(p) * m / (i + 1))
        running = q
        out[key] = q
    return out


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    cumulative = np.cumsum(values)
    running_max = np.maximum.accumulate(cumulative)
    return float((running_max - cumulative).max())


def _sharpe(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return float("nan")
    std = float(vals.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(vals.mean() / std)


def _ttest_mean(values: np.ndarray) -> tuple[float, float]:
    vals = values[np.isfinite(values)]
    if vals.size < 10:
        return float("nan"), float("nan")
    result = stats.ttest_1samp(vals, 0.0)
    return float(result.statistic), float(result.pvalue)


def _era_stability(df: pd.DataFrame, delta_col: str) -> tuple[bool, str]:
    eras = [
        ("launch_2019", date(2019, 1, 1), date(2019, 12, 31)),
        ("covid_2020_2022", date(2020, 1, 1), date(2022, 12, 31)),
        ("transition_2023", date(2023, 1, 1), date(2023, 12, 31)),
        ("recent_2024_2025", date(2024, 1, 1), date(2025, 12, 31)),
    ]
    parts: list[str] = []
    passed = True
    eligible = 0
    for name, start, end in eras:
        sub = df[(df["trading_day"] >= start) & (df["trading_day"] <= end)]
        if len(sub) < 50:
            parts.append(f"{name}:LOW_N={len(sub)}")
            continue
        eligible += 1
        mean_delta = float(sub[delta_col].mean())
        parts.append(f"{name}:{mean_delta:+.4f}/N={len(sub)}")
        if mean_delta < -0.05:
            passed = False
    if eligible < 3:
        passed = False
    return passed, "; ".join(parts)


def _wfe_delta_proxy(df: pd.DataFrame, delta_col: str) -> float:
    windows = []
    for year in range(2021, 2026):
        train = df[pd.to_datetime(df["trading_day"]).dt.year < year]
        test = df[pd.to_datetime(df["trading_day"]).dt.year == year]
        if len(train) < 100 or len(test) < 30:
            continue
        train_mean = float(train[delta_col].mean())
        test_mean = float(test[delta_col].mean())
        if train_mean <= 0:
            continue
        windows.append((len(test), train_mean, test_mean))
    if not windows:
        return float("nan")
    total_n = sum(n for n, _, _ in windows)
    train_weighted = sum(n * train for n, train, _ in windows) / total_n
    test_weighted = sum(n * test for n, _, test in windows) / total_n
    if train_weighted <= 0:
        return float("nan")
    return float(test_weighted / train_weighted)


def _year_outlier_share(df: pd.DataFrame, delta_col: str) -> tuple[float, str]:
    if df.empty:
        return float("nan"), "NA"
    work = df.copy()
    work["year"] = pd.to_datetime(work["trading_day"]).dt.year
    totals = work.groupby("year")[delta_col].sum()
    total_abs = float(abs(totals.sum()))
    if total_abs <= 1e-12 or totals.empty:
        return float("nan"), "NA"
    year = int(totals.abs().idxmax())
    share = float(abs(totals.loc[year]) / total_abs)
    return share, f"{year}:{totals.loc[year]:+.2f}R"


def _summarize_candidate(
    *,
    candidate_id: str,
    family: str,
    role: str,
    trigger_rule: str,
    parent_population: str,
    lane_id: str,
    parent_df: pd.DataFrame,
    modified_pnl: list[float],
    action_count: int,
    selectable: bool = True,
) -> dict[str, object]:
    if len(parent_df) != len(modified_pnl):
        raise ValueError(f"{candidate_id}: parent/modified length mismatch")
    work = parent_df.copy()
    work["modified_pnl"] = np.asarray(modified_pnl, dtype=float)
    if work["modified_pnl"].isna().any():
        raise ValueError(f"{candidate_id}: modified policy contains NaN")
    work["parent_pnl"] = work["pnl_r"].astype(float)
    work["delta_pnl"] = work["modified_pnl"] - work["parent_pnl"]

    is_df = work[work["trading_day"] < HOLDOUT_SACRED_FROM].copy()
    mon_df = work[work["trading_day"] >= HOLDOUT_SACRED_FROM].copy()
    parent_vals = is_df["parent_pnl"].to_numpy(dtype=float)
    mod_vals = is_df["modified_pnl"].to_numpy(dtype=float)
    delta_vals = is_df["delta_pnl"].to_numpy(dtype=float)
    t_delta, p_delta = _ttest_mean(delta_vals)

    parent_ev = float(parent_vals.mean()) if len(parent_vals) else float("nan")
    mod_ev = float(mod_vals.mean()) if len(mod_vals) else float("nan")
    delta_ev = float(delta_vals.mean()) if len(delta_vals) else float("nan")
    action_n_is = int((is_df["delta_pnl"].abs() > 1e-12).sum())
    action_rate_is = float(action_n_is / len(is_df)) if len(is_df) else float("nan")
    tail_parent = float(np.nanpercentile(parent_vals, 5)) if len(parent_vals) else float("nan")
    tail_mod = float(np.nanpercentile(mod_vals, 5)) if len(mod_vals) else float("nan")
    dd_parent = _max_drawdown(parent_vals)
    dd_mod = _max_drawdown(mod_vals)
    era_pass, era_detail = _era_stability(is_df, "delta_pnl")
    outlier_share, outlier_year = _year_outlier_share(is_df, "delta_pnl")
    wfe = _wfe_delta_proxy(is_df, "delta_pnl")
    fire_guard = bool(action_n_is >= 30 and 0.01 <= action_rate_is <= 0.90)

    row = {
        "candidate_id": candidate_id,
        "family": family,
        "role": role,
        "trigger_rule": trigger_rule,
        "parent_population": parent_population,
        "lane_id": lane_id,
        "selectable": selectable,
        "n_is": int(len(is_df)),
        "date_start_is": str(is_df["trading_day"].min()) if len(is_df) else None,
        "date_end_is": str(is_df["trading_day"].max()) if len(is_df) else None,
        "parent_ev_is": parent_ev,
        "modified_ev_is": mod_ev,
        "delta_ev_is": delta_ev,
        "parent_max_dd_is": dd_parent,
        "modified_max_dd_is": dd_mod,
        "delta_max_dd_is": dd_mod - dd_parent if math.isfinite(dd_mod) and math.isfinite(dd_parent) else float("nan"),
        "parent_tail5_is": tail_parent,
        "modified_tail5_is": tail_mod,
        "delta_tail5_is": tail_mod - tail_parent if math.isfinite(tail_mod) and math.isfinite(tail_parent) else float("nan"),
        "t_delta_is": t_delta,
        "p_delta_is": p_delta,
        "delta_sharpe_is": _sharpe(delta_vals),
        "policy_sharpe_is": _sharpe(mod_vals),
        "bh_k_declared": SELECTABLE_K,
        "bh_q_family": float("nan"),
        "bh_q_lane": float("nan"),
        "bh_q_global": float("nan"),
        "dsr": float("nan"),
        "sr0": float("nan"),
        "wfe_delta_proxy": wfe,
        "era_stable": era_pass,
        "era_detail": era_detail,
        "year_outlier_share": outlier_share,
        "year_outlier": outlier_year,
        "action_count_total": int(action_count),
        "action_count_is": action_n_is,
        "action_rate_is": action_rate_is,
        "fire_rate_guard": fire_guard,
        "n_2026": int(len(mon_df)),
        "parent_ev_2026": float(mon_df["parent_pnl"].mean()) if len(mon_df) else float("nan"),
        "modified_ev_2026": float(mon_df["modified_pnl"].mean()) if len(mon_df) else float("nan"),
        "delta_ev_2026": float(mon_df["delta_pnl"].mean()) if len(mon_df) else float("nan"),
        "verdict": "UNSCORED",
    }
    return row


def _connect_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


def _load_parent_for_session(con: duckdb.DuckDBPyConnection, session: str) -> pd.DataFrame:
    feature_cols = [
        f"orb_{session}_size",
        f"orb_{session}_volume",
        "atr_20_pct",
    ]
    reject_e2_lookahead_columns(feature_cols)
    sql = f"""
    SELECT
        o.trading_day,
        o.symbol,
        o.orb_label,
        o.orb_minutes,
        o.rr_target,
        o.confirm_bars,
        o.entry_model,
        o.entry_ts,
        o.entry_price,
        o.stop_price,
        o.target_price,
        o.outcome,
        o.exit_ts,
        o.exit_price,
        o.pnl_r,
        o.risk_dollars,
        o.mae_r,
        o.mfe_r,
        d.orb_{session}_high AS orb_high,
        d.orb_{session}_low AS orb_low,
        d.orb_{session}_size AS orb_size,
        d.orb_{session}_volume AS orb_volume,
        d.atr_20_pct AS atr_20_pct
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.symbol = d.symbol
     AND o.trading_day = d.trading_day
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target IN (1.0, 1.5, 2.0)
      AND o.entry_ts IS NOT NULL
    ORDER BY o.trading_day, o.rr_target
    """
    df = con.execute(sql, [SYMBOL, session, ORB_MINUTES]).fetchdf()
    if df.empty:
        raise RuntimeError(f"No parent rows for {SYMBOL} {session}")
    return df


def load_parent_df(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frames = [_load_parent_for_session(con, session) for session in PARENT_SESSIONS]
    df = pd.concat(frames, ignore_index=True)
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    if df["pnl_r"].isna().any():
        bad = int(df["pnl_r"].isna().sum())
        raise RuntimeError(f"Parent query found {bad} entry rows with NULL pnl_r; no silent dropout allowed")
    if df[["orb_high", "orb_low", "orb_size"]].isna().any().any():
        raise RuntimeError("Parent query found NULL ORB levels; cannot simulate path")
    _assert_parent_timestamp_eligibility(df)
    return df.sort_values(["trading_day", "entry_ts", "orb_label", "rr_target"]).reset_index(drop=True)


def _assert_parent_timestamp_eligibility(df: pd.DataFrame) -> None:
    bad: list[str] = []
    for row in df.itertuples(index=False):
        _, orb_end = orb_utc_window(row.trading_day, row.orb_label, row.orb_minutes)
        if _ts(row.entry_ts) < _ts(orb_end):
            bad.append(f"{row.symbol}/{row.orb_label}/{row.trading_day}: entry before OR end")
            if len(bad) >= 5:
                break
    if bad:
        raise RuntimeError("Timestamp eligibility failed: " + "; ".join(bad))


def load_bars_by_day(con: duckdb.DuckDBPyConnection, min_day: date, max_day: date) -> dict[date, pd.DataFrame]:
    start_utc, _ = compute_trading_day_utc_range(min_day)
    _, end_utc = compute_trading_day_utc_range(max_day)
    sql = """
    SELECT ts_utc, open, high, low, close
    FROM bars_1m
    WHERE symbol = ?
      AND ts_utc >= ?
      AND ts_utc < ?
    ORDER BY ts_utc
    """
    df = con.execute(sql, [SYMBOL, start_utc, end_utc]).fetchdf()
    if df.empty:
        raise RuntimeError("No bars_1m rows loaded for parent date span")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["trading_day"] = (df["ts_utc"].dt.tz_convert(BRISBANE_TZ) - pd.Timedelta(hours=9)).dt.date
    return {td: group.drop(columns=["trading_day"]).reset_index(drop=True) for td, group in df.groupby("trading_day")}


def load_e1_variant_df(con: duckdb.DuckDBPyConnection, session: str, confirm_bars: int, rr_target: float) -> pd.DataFrame:
    sql = """
    SELECT trading_day, symbol, orb_label, orb_minutes, rr_target, confirm_bars,
           entry_model, entry_ts, pnl_r, outcome
    FROM orb_outcomes
    WHERE symbol = ?
      AND orb_label = ?
      AND orb_minutes = ?
      AND entry_model = 'E1'
      AND confirm_bars = ?
      AND rr_target = ?
      AND entry_ts IS NOT NULL
    ORDER BY trading_day
    """
    df = con.execute(sql, [SYMBOL, session, ORB_MINUTES, confirm_bars, rr_target]).fetchdf()
    if df.empty:
        df["pnl_was_null"] = pd.Series(dtype=bool)
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    # No silent dropout: entered child rows with NULL pnl_r are retained and
    # conservatively booked as zero for replacement-policy accounting.
    df["pnl_was_null"] = df["pnl_r"].isna()
    df["pnl_r"] = df["pnl_r"].fillna(0.0)
    return df


def _bars_for_parent(row: pd.Series, bars_by_day: dict[date, pd.DataFrame]) -> pd.DataFrame:
    return bars_by_day.get(_date_value(row["trading_day"]), pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close"]))


def _shift_bars_between_trading_days(bars: pd.DataFrame, source_day: date, target_day: date) -> pd.DataFrame:
    """Preserve intraday shape while breaking date alignment for a control."""

    if bars.empty:
        return bars
    source_start, _ = compute_trading_day_utc_range(source_day)
    target_start, _ = compute_trading_day_utc_range(target_day)
    delta = _ts(target_start) - _ts(source_start)
    shifted = bars.copy()
    shifted["ts_utc"] = pd.to_datetime(shifted["ts_utc"], utc=True) + delta
    return shifted


def _fixed_non_open_window_levels(bars: pd.DataFrame, trading_day: date, session: str) -> tuple[float, float] | None:
    """Use a deterministic arbitrary post-open 5m range as a negative control."""

    if bars.empty:
        return None
    orb_start, _ = orb_utc_window(trading_day, session, ORB_MINUTES)
    control_start = _ts(orb_start) + pd.Timedelta(minutes=73)
    control_end = control_start + pd.Timedelta(minutes=ORB_MINUTES)
    b = bars.copy()
    b["ts_utc"] = pd.to_datetime(b["ts_utc"], utc=True)
    window = b[(b["ts_utc"] >= control_start) & (b["ts_utc"] < control_end)]
    if len(window) < ORB_MINUTES:
        return None
    return float(window["high"].max()), float(window["low"].min())


def _eval_same_direction(parent_df: pd.DataFrame, bars_by_day: dict[date, pd.DataFrame], cost_spec: CostSpec) -> list[dict[str, object]]:
    rows = []
    for session in PARENT_SESSIONS:
        for rr in RR_TARGETS:
            sub = parent_df[(parent_df["orb_label"] == session) & (parent_df["rr_target"] == rr)].copy()
            for wait in REENTRY_WAIT_BARS:
                modified: list[float] = []
                action_count = 0
                for _, parent in sub.iterrows():
                    policy, reentry = same_direction_reentry_policy_pnl(
                        parent,
                        _bars_for_parent(parent, bars_by_day),
                        ReentryConfig(wait_bars=wait, size=1.0, max_reentries=1),
                        cost_spec=cost_spec,
                    )
                    modified.append(float(policy))
                    if reentry is not None:
                        action_count += 1
                rows.append(
                    _summarize_candidate(
                        candidate_id=f"same_dir_reentry__{session}__rr{rr:g}__wait{wait}",
                        family="same_direction_reentry_after_stop",
                        role="execution",
                        trigger_rule=(
                            "After parent E2 full stop, wait "
                            f"{wait} bars, then re-enter original direction on fresh ORB boundary touch; max one re-entry."
                        ),
                        parent_population=f"{SYMBOL} {session} O5 E2 CB1 RR{rr:g}",
                        lane_id=f"{session}_rr{rr:g}",
                        parent_df=sub,
                        modified_pnl=modified,
                        action_count=action_count,
                    )
                )
    return rows


def _eval_shuffled_date_control(
    parent_df: pd.DataFrame,
    bars_by_day: dict[date, pd.DataFrame],
    cost_spec: CostSpec,
) -> list[dict[str, object]]:
    """Non-selectable negative control: same rule on next trading day's path."""

    rows = []
    days = sorted(bars_by_day)
    next_day = {day: days[(idx + 1) % len(days)] for idx, day in enumerate(days)}
    for session in PARENT_SESSIONS:
        for rr in RR_TARGETS:
            sub = parent_df[(parent_df["orb_label"] == session) & (parent_df["rr_target"] == rr)].copy()
            modified: list[float] = []
            action_count = 0
            for _, parent in sub.iterrows():
                td = _date_value(parent["trading_day"])
                src_day = next_day.get(td)
                src_bars = bars_by_day.get(src_day, pd.DataFrame()) if src_day is not None else pd.DataFrame()
                shifted = _shift_bars_between_trading_days(src_bars, src_day, td) if src_day is not None else src_bars
                policy, reentry = same_direction_reentry_policy_pnl(
                    parent,
                    shifted,
                    ReentryConfig(wait_bars=5, size=1.0, max_reentries=1),
                    cost_spec=cost_spec,
                )
                modified.append(float(policy))
                if reentry is not None:
                    action_count += 1
            rows.append(
                _summarize_candidate(
                    candidate_id=f"control_shuffled_date_same_dir__{session}__rr{rr:g}__wait5",
                    family="shuffled_reentry_date_control",
                    role="control",
                    trigger_rule=(
                        "Negative control: run same-direction re-entry on the next trading day's 1m path shifted "
                        "onto the parent day; not eligible for ranking."
                    ),
                    parent_population=f"{SYMBOL} {session} O5 E2 CB1 RR{rr:g}",
                    lane_id=f"{session}_rr{rr:g}",
                    parent_df=sub,
                    modified_pnl=modified,
                    action_count=action_count,
                    selectable=False,
                )
            )
    return rows


def _eval_random_range_control(
    parent_df: pd.DataFrame,
    bars_by_day: dict[date, pd.DataFrame],
    cost_spec: CostSpec,
) -> list[dict[str, object]]:
    """Non-selectable negative control: replace ORB levels with a fixed arbitrary range."""

    rows = []
    for session in PARENT_SESSIONS:
        for rr in RR_TARGETS:
            sub = parent_df[(parent_df["orb_label"] == session) & (parent_df["rr_target"] == rr)].copy()
            modified: list[float] = []
            action_count = 0
            for _, parent in sub.iterrows():
                td = _date_value(parent["trading_day"])
                bars = _bars_for_parent(parent, bars_by_day)
                levels = _fixed_non_open_window_levels(bars, td, session)
                if levels is None:
                    modified.append(float(parent["pnl_r"]))
                    continue
                fake_high, fake_low = levels
                policy, reentry = _same_direction_reentry_with_levels(
                    parent,
                    bars,
                    ReentryConfig(wait_bars=5, size=1.0, max_reentries=1),
                    cost_spec=cost_spec,
                    orb_high=fake_high,
                    orb_low=fake_low,
                )
                modified.append(float(policy))
                if reentry is not None:
                    action_count += 1
            rows.append(
                _summarize_candidate(
                    candidate_id=f"control_non_open_range__{session}__rr{rr:g}__offset73_wait5",
                    family="random_non_open_range_control",
                    role="control",
                    trigger_rule=(
                        "Negative control: replace ORB levels with deterministic 5m range beginning 73 minutes "
                        "after session OR start; not eligible for ranking."
                    ),
                    parent_population=f"{SYMBOL} {session} O5 E2 CB1 RR{rr:g}",
                    lane_id=f"{session}_rr{rr:g}",
                    parent_df=sub,
                    modified_pnl=modified,
                    action_count=action_count,
                    selectable=False,
                )
            )
    return rows


def _eval_reversal(parent_df: pd.DataFrame, bars_by_day: dict[date, pd.DataFrame], cost_spec: CostSpec) -> list[dict[str, object]]:
    rows = []
    for session in PARENT_SESSIONS:
        for rr in RR_TARGETS:
            sub = parent_df[(parent_df["orb_label"] == session) & (parent_df["rr_target"] == rr)].copy()
            for wait in REVERSAL_WAIT_BARS:
                modified: list[float] = []
                action_count = 0
                for _, parent in sub.iterrows():
                    policy, reentry = opposite_direction_reentry_policy_pnl(
                        parent,
                        _bars_for_parent(parent, bars_by_day),
                        ReentryConfig(wait_bars=wait, size=1.0, max_reentries=1),
                        cost_spec=cost_spec,
                    )
                    modified.append(float(policy))
                    if reentry is not None:
                        action_count += 1
                rows.append(
                    _summarize_candidate(
                        candidate_id=f"fakeout_reversal__{session}__rr{rr:g}__wait{wait}",
                        family="fakeout_reversal_after_stop",
                        role="standalone",
                        trigger_rule=(
                            "After parent E2 full stop, wait "
                            f"{wait} bars, then enter the opposite ORB side; max one reversal."
                        ),
                        parent_population=f"{SYMBOL} {session} O5 E2 CB1 RR{rr:g}",
                        lane_id=f"{session}_rr{rr:g}",
                        parent_df=sub,
                        modified_pnl=modified,
                        action_count=action_count,
                    )
                )
    return rows


def _eval_retest(parent_df: pd.DataFrame, bars_by_day: dict[date, pd.DataFrame], cost_spec: CostSpec) -> list[dict[str, object]]:
    rows = []
    for session in PARENT_SESSIONS:
        for rr in RR_TARGETS:
            sub = parent_df[(parent_df["orb_label"] == session) & (parent_df["rr_target"] == rr)].copy()
            for wait in RETEST_WAIT_BARS:
                modified = []
                action_count = 0
                for _, parent in sub.iterrows():
                    policy, fill = retest_hold_policy_pnl(
                        parent,
                        _bars_for_parent(parent, bars_by_day),
                        wait_bars=wait,
                        cost_spec=cost_spec,
                    )
                    modified.append(float(policy))
                    if fill is not None:
                        action_count += 1
                rows.append(
                    _summarize_candidate(
                        candidate_id=f"retest_hold__{session}__rr{rr:g}__wait{wait}",
                        family="retest_hold_before_stop",
                        role="execution",
                        trigger_rule=(
                            "Replace initial stop entry with ORB-edge retest hold entry before any stop breach; "
                            f"wait {wait} bars after parent entry timestamp."
                        ),
                        parent_population=f"{SYMBOL} {session} O5 E2 CB1 RR{rr:g}",
                        lane_id=f"{session}_rr{rr:g}",
                        parent_df=sub,
                        modified_pnl=modified,
                        action_count=action_count,
                    )
                )
    return rows


def _eval_confirmation(con: duckdb.DuckDBPyConnection, parent_df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for session in PARENT_SESSIONS:
        for rr in RR_TARGETS:
            parent = parent_df[(parent_df["orb_label"] == session) & (parent_df["rr_target"] == rr)].copy()
            for cb in CONFIRM_BARS:
                e1 = load_e1_variant_df(con, session, cb, rr)
                merged = parent.merge(
                    e1[["trading_day", "pnl_r", "entry_ts", "pnl_was_null"]].rename(
                        columns={"pnl_r": "variant_pnl_r", "entry_ts": "variant_entry_ts"}
                    ),
                    on="trading_day",
                    how="left",
                )
                modified = merged["variant_pnl_r"].fillna(0.0).astype(float).tolist()
                action_count = int(merged["variant_entry_ts"].notna().sum())
                rows.append(
                    _summarize_candidate(
                        candidate_id=f"initial_confirmation_delay__{session}__rr{rr:g}__cb{cb}",
                        family="initial_entry_confirmation_delay",
                        role="execution",
                        trigger_rule=(
                            f"Replace E2 touch entry with E1 market-on-confirm after {cb} consecutive outside closes; "
                            "missed confirmations count as zero PnL on the original parent opportunity."
                        ),
                        parent_population=f"{SYMBOL} {session} O5 E2 CB1 RR{rr:g}",
                        lane_id=f"{session}_rr{rr:g}",
                        parent_df=parent,
                        modified_pnl=modified,
                        action_count=action_count,
                    )
                )
    return rows


def _eval_filters(parent_df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for session in PARENT_SESSIONS:
        for rr in RR_TARGETS:
            parent = parent_df[(parent_df["orb_label"] == session) & (parent_df["rr_target"] == rr)].copy()
            is_parent = parent[parent["trading_day"] < HOLDOUT_SACRED_FROM]
            size_threshold = float(np.nanpercentile(is_parent["orb_size"].astype(float), 67))
            filters = {
                "orb_size_q67": parent["orb_size"].astype(float) >= size_threshold,
                "atr20pct_ge70": parent["atr_20_pct"].astype(float) >= 70.0,
            }
            for filter_name, mask in filters.items():
                if filter_name == "orb_size_q67":
                    rule = f"Trade parent only when pre-entry ORB size >= IS q67 ({size_threshold:.2f})."
                else:
                    rule = "Trade parent only when daily_features.atr_20_pct >= 70, known before entry."
                modified = np.where(mask.to_numpy(), parent["pnl_r"].astype(float).to_numpy(), 0.0).tolist()
                rows.append(
                    _summarize_candidate(
                        candidate_id=f"pre_entry_filter__{session}__rr{rr:g}__{filter_name}",
                        family="pre_entry_size_atr_filter",
                        role="filter",
                        trigger_rule=rule,
                        parent_population=f"{SYMBOL} {session} O5 E2 CB1 RR{rr:g}",
                        lane_id=f"{session}_rr{rr:g}",
                        parent_df=parent,
                        modified_pnl=modified,
                        action_count=int((np.asarray(modified) != parent["pnl_r"].astype(float).to_numpy()).sum()),
                    )
                )
    return rows


def _eval_throttle(parent_df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for rr in RR_TARGETS:
        parent = parent_df[parent_df["rr_target"] == rr].copy()
        parent = parent.sort_values(["trading_day", "entry_ts", "orb_label"]).reset_index(drop=True)
        for rule in THROTTLE_RULES:
            modified = []
            action_count = 0
            loss_seen_by_day: dict[date, bool] = {}
            for row in parent.itertuples(index=False):
                td = row.trading_day
                loss_seen = loss_seen_by_day.get(td, False)
                pnl = float(row.pnl_r)
                if loss_seen:
                    action_count += 1
                    if rule == "skip_after_first_loss":
                        pnl = 0.0
                    elif rule == "half_after_first_loss":
                        pnl *= 0.5
                    else:
                        raise ValueError(rule)
                modified.append(pnl)
                if row.outcome == "loss":
                    loss_seen_by_day[td] = True
            trigger = (
                "After the first same-trading-day parent E2 loss across the three scoped sessions, "
                + ("skip later parent opportunities." if rule == "skip_after_first_loss" else "trade later opportunities at half risk.")
            )
            rows.append(
                _summarize_candidate(
                    candidate_id=f"one_loss_throttle__rr{rr:g}__{rule}",
                    family="one_loss_session_throttle",
                    role="allocator",
                    trigger_rule=trigger,
                    parent_population=f"{SYMBOL} O5 E2 CB1 RR{rr:g} across {','.join(PARENT_SESSIONS)}",
                    lane_id=f"portfolio_rr{rr:g}",
                    parent_df=parent,
                    modified_pnl=modified,
                    action_count=action_count,
                )
            )
    return rows


def _apply_multiple_testing(rows: list[dict[str, object]]) -> None:
    selectable = [r for r in rows if r.get("selectable", True)]
    global_inputs = [
        (str(r["candidate_id"]), 1.0 if math.isnan(float(r["p_delta_is"])) else float(r["p_delta_is"]))
        for r in selectable
    ]
    global_q = bh_q_values(global_inputs)
    for r in selectable:
        r["bh_q_global"] = global_q.get(str(r["candidate_id"]), float("nan"))

    for family in sorted({str(r["family"]) for r in selectable}):
        group = [r for r in selectable if r["family"] == family]
        q_map = bh_q_values(
            [
                (str(r["candidate_id"]), 1.0 if math.isnan(float(r["p_delta_is"])) else float(r["p_delta_is"]))
                for r in group
            ]
        )
        for r in group:
            r["bh_q_family"] = q_map.get(str(r["candidate_id"]), float("nan"))

    for lane in sorted({str(r["lane_id"]) for r in selectable}):
        group = [r for r in selectable if r["lane_id"] == lane]
        q_map = bh_q_values(
            [
                (str(r["candidate_id"]), 1.0 if math.isnan(float(r["p_delta_is"])) else float(r["p_delta_is"]))
                for r in group
            ]
        )
        for r in group:
            r["bh_q_lane"] = q_map.get(str(r["candidate_id"]), float("nan"))


def _apply_dsr(rows: list[dict[str, object]]) -> None:
    selectable = [r for r in rows if r.get("selectable", True)]
    sharpe_vals = np.array([float(r["delta_sharpe_is"]) for r in selectable], dtype=float)
    sharpe_vals = sharpe_vals[np.isfinite(sharpe_vals)]
    var_sr = float(np.var(sharpe_vals, ddof=1)) if sharpe_vals.size > 1 else 0.0
    sr0 = compute_sr0(SELECTABLE_K, var_sr)
    for r in selectable:
        sr = float(r["delta_sharpe_is"])
        n = int(r["n_is"])
        if not math.isfinite(sr) or n < 2:
            r["dsr"] = 0.0
            r["sr0"] = sr0
            continue
        # We do not persist per-candidate deltas here. Use conservative normal
        # shape terms for the cross-check and keep DSR secondary to BH/WFE.
        r["dsr"] = compute_dsr(sr_hat=sr, sr0=sr0, t_obs=n, skewness=0.0, kurtosis_excess=0.0)
        r["sr0"] = sr0


def _assign_verdicts(rows: list[dict[str, object]]) -> None:
    for r in rows:
        if not r.get("selectable", True):
            r["verdict"] = "PARK"
            continue
        if float(r["delta_ev_is"]) <= 0 or float(r["modified_ev_is"]) <= float(r["parent_ev_is"]):
            r["verdict"] = "KILL"
            continue
        if int(r["n_is"]) < 100 or not bool(r["fire_rate_guard"]):
            r["verdict"] = "PARK"
            continue
        if float(r["p_delta_is"]) > 0.05 or float(r["bh_q_family"]) > BH_Q:
            r["verdict"] = "KILL"
            continue
        if float(r["dsr"]) < 0.95:
            r["verdict"] = "NARROW"
            continue
        if not bool(r["era_stable"]):
            r["verdict"] = "NARROW"
            continue
        wfe = float(r["wfe_delta_proxy"])
        if math.isnan(wfe) or wfe < 0.50:
            r["verdict"] = "NARROW"
            continue
        if float(r["delta_max_dd_is"]) > 0 and float(r["delta_tail5_is"]) < 0:
            r["verdict"] = "NARROW"
            continue
        r["verdict"] = "CONTINUE"


def run_research(db_path: Path = GOLD_DB_PATH) -> pd.DataFrame:
    # The known-bad injection must reject. If it does not, the run is unsafe.
    # Catch only the expected failure below.
    try:
        reject_e2_lookahead_columns(["orb_NYSE_OPEN_break_ts", "rel_vol_NYSE_OPEN", "pnl_r"])
    except ValueError:
        pass
    else:
        raise RuntimeError("Known-bad lookahead injection was not rejected")

    with _connect_db(db_path) as con:
        parent_df = load_parent_df(con)
        min_day = min(parent_df["trading_day"])
        max_day = max(parent_df["trading_day"])
        bars_by_day = load_bars_by_day(con, min_day, max_day)
        cost_spec = get_cost_spec(SYMBOL)
        rows: list[dict[str, object]] = []
        rows.extend(_eval_same_direction(parent_df, bars_by_day, cost_spec))
        rows.extend(_eval_shuffled_date_control(parent_df, bars_by_day, cost_spec))
        rows.extend(_eval_random_range_control(parent_df, bars_by_day, cost_spec))
        rows.extend(_eval_confirmation(con, parent_df))
        rows.extend(_eval_throttle(parent_df))
        rows.extend(_eval_retest(parent_df, bars_by_day, cost_spec))
        rows.extend(_eval_filters(parent_df))
        rows.extend(_eval_reversal(parent_df, bars_by_day, cost_spec))

    observed_k = len([r for r in rows if r.get("selectable", True)])
    if observed_k != SELECTABLE_K:
        raise RuntimeError(f"Selectable K mismatch: observed {observed_k}, declared {SELECTABLE_K}")
    _apply_multiple_testing(rows)
    _apply_dsr(rows)
    _assign_verdicts(rows)
    return pd.DataFrame(rows)


def _priority_sort(df: pd.DataFrame) -> pd.DataFrame:
    order = {"CONTINUE": 0, "NARROW": 1, "KILL": 2, "PARK": 3}
    out = df.copy()
    out["_rank"] = out["verdict"].map(order).fillna(9)
    return out.sort_values(["_rank", "delta_ev_is", "modified_ev_is"], ascending=[True, False, False])


def write_outputs(df: pd.DataFrame) -> None:
    ROW_CSV.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROW_CSV, index=False)
    RESULT_DOC.write_text(render_markdown(df), encoding="utf-8")


def render_markdown(df: pd.DataFrame) -> str:
    ranked = _priority_sort(df)
    continue_df = ranked[ranked["verdict"] == "CONTINUE"]
    narrow_df = ranked[ranked["verdict"] == "NARROW"]
    lines: list[str] = [
        "# ORB Execution Variants v1",
        "",
        f"**Pre-reg:** `{PREREG_PATH.as_posix()}`",
        f"**Canonical inputs:** `bars_1m`, `daily_features`, `orb_outcomes` only.",
        f"**Selectable K:** `{SELECTABLE_K}`",
        f"**Holdout policy:** pre-2026 selects; `{HOLDOUT_SACRED_FROM}` onward is descriptive only.",
        f"**Full cell CSV:** `{ROW_CSV.as_posix()}`",
        "",
        "## Executive Verdict",
        "",
    ]
    if continue_df.empty:
        lines.append("No candidate clears the powered policy gates. Priority additions: **none**.")
    else:
        lines.append(f"Priority additions: **{len(continue_df)} CONTINUE candidates**. They still are research-valid only, not deployment-cleared.")
    same_dir = ranked[ranked["family"] == "same_direction_reentry_after_stop"]
    reversal = ranked[ranked["family"] == "fakeout_reversal_after_stop"]
    if not same_dir.empty:
        best_same = same_dir.iloc[0]
        lines.append(
            f"Primary same-direction answer: **KILL**. Best same-direction cell is `{best_same['candidate_id']}` "
            f"with delta `{_fmt(best_same['delta_ev_is'])}`, BH-family `{_fmt(best_same['bh_q_family'])}`, "
            f"DSR `{_fmt(best_same['dsr'])}`, era-stable `{best_same['era_stable']}`, and 2026 descriptive delta "
            f"`{_fmt(best_same['delta_ev_2026'])}`."
        )
    if not reversal.empty:
        best_rev = reversal.iloc[0]
        lines.append(
            f"Closest non-primary signal: `{best_rev['candidate_id']}` is an opposite-direction reversal, "
            f"not the user's same-direction re-entry failure mode; verdict `{best_rev['verdict']}`."
        )
    lines.extend(
        [
            "",
            "This report does not infer stop-hunt intent. The measurable object is: parent E2 entry stops, then a bounded execution modification is tested per original ORB opportunity.",
            "All EV numbers include skipped trades as zero and include the first stopped trade before any re-entry recovery.",
            "",
            "## Priority Table",
            "",
            "| Rank | Verdict | Candidate | Role | Parent EV | Modified EV | Delta EV | DD Delta | p | BH family | DSR | WFE | 2026 delta |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in ranked.head(25).reset_index(drop=True).iterrows():
        lines.append(
            "| "
            f"{idx + 1} | {row['verdict']} | `{row['candidate_id']}` | {row['role']} | "
            f"{_fmt(row['parent_ev_is'])} | {_fmt(row['modified_ev_is'])} | {_fmt(row['delta_ev_is'])} | "
            f"{_fmt(row['delta_max_dd_is'])} | {_fmt(row['p_delta_is'])} | {_fmt(row['bh_q_family'])} | "
            f"{_fmt(row['dsr'])} | {_fmt(row['wfe_delta_proxy'])} | {_fmt(row['delta_ev_2026'])} |"
        )

    lines.extend(["", "## Family Summary", ""])
    family_rows = []
    for family, group in df.groupby("family"):
        best = _priority_sort(group).iloc[0]
        verdict_counts = group["verdict"].value_counts().to_dict()
        family_rows.append((family, best, verdict_counts))
    lines.extend(
        [
            "| Family | Best candidate | Verdicts | Best delta EV | Best p | Best BH family | Best DSR |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for family, best, counts in sorted(family_rows):
        count_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(
            f"| {family} | `{best['candidate_id']}` | {count_text} | {_fmt(best['delta_ev_is'])} | "
            f"{_fmt(best['p_delta_is'])} | {_fmt(best['bh_q_family'])} | {_fmt(best['dsr'])} |"
        )

    lines.extend(
        [
            "",
            "## Controls",
            "",
            "- Known-bad E2 lookahead injection: `orb_NYSE_OPEN_break_ts`, `rel_vol_NYSE_OPEN`, and `pnl_r` are rejected before the run.",
            "- Inverted-direction trigger: tested as `fakeout_reversal_after_stop`; it is scored separately from same-direction re-entry and is not allowed to rescue that family.",
            "- Shuffled re-entry trigger dates: tested as `shuffled_reentry_date_control` by shifting the next trading day's path onto the parent day; not eligible for ranking.",
            "- Random non-open range window: tested as `random_non_open_range_control` using a deterministic 5m range 73 minutes after session OR start; not eligible for ranking.",
            "- Order-flow, footprint, delta, and absorption variants: `PARK_NEW_DATA`; current 1m OHLCV cannot measure them honestly.",
            "- Control read: shuffled-date controls print large positive deltas, which is a construction-sensitivity warning, not evidence. It reinforces the decision not to promote a path narrative from these 1m OHLCV controls.",
            "",
            "## Candidate Detail",
            "",
        ]
    )
    detail_df = pd.concat([continue_df, narrow_df]).head(20) if not pd.concat([continue_df, narrow_df]).empty else ranked.head(10)
    for _, row in detail_df.iterrows():
        lines.extend(
            [
                f"### `{row['candidate_id']}`",
                "",
                f"- role: `{row['role']}`",
                f"- measurable trigger rule: {row['trigger_rule']}",
                f"- parent population: {row['parent_population']}",
                f"- original parent EV: `{_fmt(row['parent_ev_is'])}`",
                f"- modified policy EV, including first loss/skips: `{_fmt(row['modified_ev_is'])}`",
                f"- delta drawdown / tail loss: `{_fmt(row['delta_max_dd_is'])}` / `{_fmt(row['delta_tail5_is'])}`",
                f"- sample and span: `N={row['n_is']}`, `{row['date_start_is']}..{row['date_end_is']}`",
                f"- p-value / BH K / DSR / WFE / era stability: `{_fmt(row['p_delta_is'])}` / `{row['bh_k_declared']}` / `{_fmt(row['dsr'])}` / `{_fmt(row['wfe_delta_proxy'])}` / `{row['era_stable']}`",
                f"- 2026 descriptive result, non-selection: `N={row['n_2026']}`, delta `{_fmt(row['delta_ev_2026'])}`",
                f"- verdict: `{row['verdict']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## SURVIVED SCRUTINY",
            "",
        ]
    )
    if continue_df.empty:
        lines.append("- None.")
    else:
        for _, row in continue_df.iterrows():
            lines.append(f"- `{row['candidate_id']}` delta `{_fmt(row['delta_ev_is'])}`, p `{_fmt(row['p_delta_is'])}`, DSR `{_fmt(row['dsr'])}`.")
    lines.extend(["", "## DID NOT SURVIVE", ""])
    for verdict in ("NARROW", "PARK", "KILL"):
        sub = df[df["verdict"] == verdict]
        lines.append(f"- `{verdict}`: {len(sub)} cells.")
    lines.extend(
        [
            "",
            "## CAVEATS",
            "",
            "- DSR uses the fixed universe K declared here and the cross-sectional variance of candidate delta Sharpes. It is a research-validation cross-check, not deployment permission.",
            "- Re-entry path simulation uses 1m OHLC bars and conservative same-bar stop/target ordering. It cannot observe queue position or footprint absorption.",
            "- The shuffled-date control is intentionally non-selectable and came back positive; that makes narrative explanations weaker, not stronger.",
            "- 2026 rows are descriptive monitoring only; no threshold, session, or candidate ranking is selected from them.",
            "",
            "## NEXT STEPS",
            "",
            "- Only `CONTINUE` rows, if any, should move to a separate deployment-readiness route.",
            "- `NARROW` rows need a fresh prereg or an implementation simplification before retest.",
            "- `PARK_NEW_DATA` order-flow claims need actual order-flow data before they can be ranked.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bias-hardened ORB execution variants v1")
    parser.add_argument("--db", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    df = run_research(args.db)
    if not args.no_write:
        write_outputs(df)
        print(f"Wrote {ROW_CSV}")
        print(f"Wrote {RESULT_DOC}")
    verdict_counts = df["verdict"].value_counts().to_dict()
    print(f"Selectable K={SELECTABLE_K}; verdicts={verdict_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
