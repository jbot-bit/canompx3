"""MNQ live-context overlay family replay on exact deployed lanes.

Scope is locked by:
  docs/audit/hypotheses/2026-04-20-mnq-live-context-overlays-v1.yaml

This runner evaluates exactly five pre-registered hypotheses on two exact
deployed MNQ lanes using canonical orb_outcomes + daily_features joins and
canonical filter delegation via research.filter_utils.filter_signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from research.result_doc_header import build_header
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-20-mnq-live-context-overlays-v1.yaml"
OUTPUT_PATH = Path("docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md")
BH_Q = 0.05
IS_START_YEAR = 2019
IS_END_YEAR = 2025

LANES = {
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12": {
        "session": "NYSE_OPEN",
        "orb_minutes": 5,
        "entry_model": "E2",
        "confirm_bars": 1,
        "rr_target": 1.0,
        "filter_key": "COST_LT12",
    },
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5": {
        "session": "COMEX_SETTLE",
        "orb_minutes": 5,
        "entry_model": "E2",
        "confirm_bars": 1,
        "rr_target": 1.5,
        "filter_key": "ORB_G5",
    },
}


@dataclass(frozen=True)
class HypothesisPlan:
    id: str
    lane_id: str
    short_only: bool
    signal_kind: str
    delta_mode: str


HYPOTHESIS_PLANS = [
    HypothesisPlan("H01_NYO_SHORT_PREV_BEAR", "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", True, "prev_day_bear", "on_minus_off"),
    HypothesisPlan("H02_NYO_LANE_OPEX_TRUE", "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", False, "is_opex", "on_minus_off"),
    HypothesisPlan("H03_CMX_SHORT_RELVOL_Q3", "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", True, "rel_vol_high_q3", "on_minus_off"),
    HypothesisPlan("H04_CMX_SHORT_RELVOL_Q3_AND_F6", "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", True, "rel_vol_high_q3_and_f6", "on_minus_off"),
    HypothesisPlan("H05_CMX_LANE_OPEX_TRUE", "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", False, "is_opex", "off_minus_on"),
]


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _bh_fdr(values: list[tuple[str, float]]) -> dict[str, float]:
    finite = [(key, p) for key, p in values if p is not None and not math.isnan(p)]
    if not finite:
        return {}
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    out: dict[str, float] = {}
    running = 1.0
    for i in range(m - 1, -1, -1):
        key, p = finite[i]
        q = min(running, p * m / (i + 1))
        running = q
        out[key] = q
    return out


def _welch_p(on: pd.Series, off: pd.Series) -> float:
    if len(on) < 2 or len(off) < 2:
        return float("nan")
    res = stats.ttest_ind(on, off, equal_var=False)
    return float(np.asarray(res.pvalue))


def _compute_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    corr = float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])
    if math.isnan(corr):
        return 0.0
    return corr


def _load_lane(con: duckdb.DuckDBPyConnection, lane_id: str) -> pd.DataFrame:
    spec = LANES[lane_id]
    session = spec["session"]
    query = f"""
    SELECT
        o.trading_day,
        o.symbol,
        o.orb_minutes,
        o.orb_label,
        o.entry_model,
        o.confirm_bars,
        o.rr_target,
        o.entry_price,
        o.stop_price,
        o.pnl_r,
        d.prev_day_direction,
        d.is_opex_day,
        d.prev_day_high,
        d.prev_day_low,
        d.rel_vol_{session} AS lane_rel_vol,
        d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = {spec["orb_minutes"]}
      AND o.entry_model = '{spec["entry_model"]}'
      AND o.confirm_bars = {spec["confirm_bars"]}
      AND o.rr_target = {spec["rr_target"]}
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(query).df()
    if len(df) == 0:
        return df
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["is_is"] = df["trading_day"].dt.date < HOLDOUT_SACRED_FROM
    df["is_oos"] = ~df["is_is"]
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    return df


def _assert_active_lanes(con: duckdb.DuckDBPyConnection) -> None:
    placeholders = ", ".join(f"'{lane}'" for lane in LANES)
    query = f"""
    SELECT strategy_id
    FROM validated_setups
    WHERE strategy_id IN ({placeholders})
      AND COALESCE(status, 'active') != 'retired'
    ORDER BY strategy_id
    """
    rows = con.execute(query).fetchall()
    found = {row[0] for row in rows}
    missing = sorted(set(LANES) - found)
    if missing:
        raise RuntimeError(f"Missing active validated_setups rows for: {missing}")


def _f6_inside_pdr(df: pd.DataFrame, session: str) -> np.ndarray:
    mid = (df[f"orb_{session}_high"].astype(float) + df[f"orb_{session}_low"].astype(float)) / 2.0
    pdh = df["prev_day_high"].astype(float)
    pdl = df["prev_day_low"].astype(float)
    return ((mid > pdl) & (mid < pdh)).fillna(False).to_numpy(dtype=bool)


def _calibrate_rel_vol_threshold(df: pd.DataFrame) -> float:
    vals = df.loc[df["is_is"], "lane_rel_vol"].astype(float).dropna()
    if len(vals) < 20:
        raise RuntimeError("Insufficient IS rel_vol rows for threshold calibration")
    return float(np.nanpercentile(vals, 67))


def _build_signal(df: pd.DataFrame, plan: HypothesisPlan, rel_vol_thresholds: dict[str, float]) -> np.ndarray:
    lane = LANES[plan.lane_id]
    session = lane["session"]
    if plan.signal_kind == "prev_day_bear":
        return df["prev_day_direction"].eq("bear").to_numpy(dtype=bool)
    if plan.signal_kind == "is_opex":
        return df["is_opex_day"].fillna(False).astype(bool).to_numpy()
    if plan.signal_kind == "rel_vol_high_q3":
        thr = rel_vol_thresholds[plan.lane_id]
        return (df["lane_rel_vol"].astype(float) > thr).fillna(False).to_numpy(dtype=bool)
    if plan.signal_kind == "rel_vol_high_q3_and_f6":
        thr = rel_vol_thresholds[plan.lane_id]
        rel = (df["lane_rel_vol"].astype(float) > thr).fillna(False).to_numpy(dtype=bool)
        return rel & _f6_inside_pdr(df, session)
    raise ValueError(f"unsupported signal_kind: {plan.signal_kind}")


def _signed_delta(expr_on: float, expr_off: float, mode: str) -> float:
    if math.isnan(expr_on) or math.isnan(expr_off):
        return float("nan")
    if mode == "on_minus_off":
        return expr_on - expr_off
    if mode == "off_minus_on":
        return expr_off - expr_on
    raise ValueError(f"unsupported delta_mode: {mode}")


def _signed_wr_delta(wr_on: float, wr_off: float, mode: str) -> float:
    if math.isnan(wr_on) or math.isnan(wr_off):
        return float("nan")
    if mode == "on_minus_off":
        return wr_on - wr_off
    if mode == "off_minus_on":
        return wr_off - wr_on
    raise ValueError(f"unsupported delta_mode: {mode}")


def _group_stats(series: pd.Series) -> tuple[float, float]:
    if len(series) == 0:
        return float("nan"), float("nan")
    return float(series.mean()), float((series > 0).mean())


def _year_rows(df: pd.DataFrame, signal: np.ndarray, mode: str) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    positive_years = 0
    for year in range(IS_START_YEAR, IS_END_YEAR + 1):
        yr = df[(df["is_is"]) & (df["year"] == year)].copy()
        if len(yr) == 0:
            rows.append({"year": year, "n_on": 0, "n_off": 0, "delta": float("nan"), "eligible": False})
            continue
        sig = signal[yr.index.to_numpy()]
        on = yr.loc[sig, "pnl_r"]
        off = yr.loc[~sig, "pnl_r"]
        delta = _signed_delta(float(on.mean()) if len(on) else float("nan"), float(off.mean()) if len(off) else float("nan"), mode)
        eligible = len(on) >= 10 and len(off) >= 10 and not math.isnan(delta)
        if eligible and delta > 0:
            positive_years += 1
        rows.append({"year": year, "n_on": int(len(on)), "n_off": int(len(off)), "delta": delta, "eligible": eligible})
    return rows, positive_years


def _evaluate_pass(
    plan: HypothesisPlan,
    hyp_cfg: dict,
    pass_name: str,
    lane_df: pd.DataFrame,
    signal_all: np.ndarray,
    deployed_fire_all: np.ndarray,
    corr_vs_filter: float,
) -> dict[str, object]:
    if pass_name == "filtered":
        mask = deployed_fire_all.astype(bool)
    else:
        mask = np.ones(len(lane_df), dtype=bool)

    sub = lane_df.loc[mask].copy().reset_index(drop=True)
    signal = signal_all[mask]
    if len(sub) == 0:
        raise RuntimeError(f"{plan.id} {pass_name}: zero rows after pass mask")

    sig_is = signal[sub["is_is"].to_numpy()]
    sig_oos = signal[sub["is_oos"].to_numpy()]
    is_df = sub.loc[sub["is_is"]].copy()
    oos_df = sub.loc[sub["is_oos"]].copy()

    on_is = is_df.loc[sig_is, "pnl_r"]
    off_is = is_df.loc[~sig_is, "pnl_r"]
    on_oos = oos_df.loc[sig_oos, "pnl_r"]
    off_oos = oos_df.loc[~sig_oos, "pnl_r"]

    expr_on_is, wr_on_is = _group_stats(on_is)
    expr_off_is, wr_off_is = _group_stats(off_is)
    expr_on_oos, wr_on_oos = _group_stats(on_oos)
    expr_off_oos, wr_off_oos = _group_stats(off_oos)

    delta_is = _signed_delta(expr_on_is, expr_off_is, plan.delta_mode)
    delta_oos = _signed_delta(expr_on_oos, expr_off_oos, plan.delta_mode)
    wr_spread = _signed_wr_delta(wr_on_is, wr_off_is, plan.delta_mode)
    raw_p = _welch_p(on_is, off_is)
    fire_rate = float(signal.mean()) if len(signal) else float("nan")
    extreme_fire = bool(fire_rate < 0.05 or fire_rate > 0.95) if not math.isnan(fire_rate) else False
    arithmetic_only = bool(abs(wr_spread) < 0.03 and abs(delta_is) > 0.10) if not math.isnan(wr_spread) and not math.isnan(delta_is) else False
    oos_dir_match = bool(np.sign(delta_is) == np.sign(delta_oos)) if len(on_oos) >= 10 and len(off_oos) >= 10 and not math.isnan(delta_oos) else None
    year_rows, years_positive = _year_rows(sub, signal, plan.delta_mode)

    thresholds_gte = hyp_cfg["pass_metric"]["threshold_gte"]
    thresholds_lt = hyp_cfg["pass_metric"]["threshold_lt"]
    delta_key = next(key for key in thresholds_gte if key.startswith(pass_name))
    raw_p_key = next(key for key in thresholds_lt if key.startswith(pass_name))
    delta_ok = not math.isnan(delta_is) and delta_is >= float(thresholds_gte[delta_key])
    raw_p_ok = not math.isnan(raw_p) and raw_p < float(thresholds_lt[raw_p_key])
    years_ok = True
    if pass_name == "filtered" and "years_positive_filtered" in thresholds_gte:
        years_ok = years_positive >= int(thresholds_gte["years_positive_filtered"])
    extra_gate = hyp_cfg["pass_metric"].get("extra_gate")
    extra_ok = True
    if extra_gate == "filtered_ExpR_on_IS > 0":
        extra_ok = expr_on_is > 0
    elif extra_gate == "filtered_ExpR_off_IS > 0":
        extra_ok = expr_off_is > 0

    return {
        "hypothesis_id": plan.id,
        "lane_id": plan.lane_id,
        "pass_name": pass_name,
        "n_total": int(len(sub)),
        "n_is": int(len(is_df)),
        "n_oos": int(len(oos_df)),
        "n_on_is": int(len(on_is)),
        "n_off_is": int(len(off_is)),
        "n_on_oos": int(len(on_oos)),
        "n_off_oos": int(len(off_oos)),
        "expr_on_is": expr_on_is,
        "expr_off_is": expr_off_is,
        "expr_on_oos": expr_on_oos,
        "expr_off_oos": expr_off_oos,
        "wr_on_is": wr_on_is,
        "wr_off_is": wr_off_is,
        "delta_is": delta_is,
        "delta_oos": delta_oos,
        "wr_spread_is": wr_spread,
        "raw_p_is": raw_p,
        "fire_rate": fire_rate,
        "extreme_fire": extreme_fire,
        "arithmetic_only": arithmetic_only,
        "corr_vs_filter": corr_vs_filter,
        "oos_dir_match": oos_dir_match,
        "years_positive": years_positive,
        "year_rows": year_rows,
        "delta_ok": delta_ok,
        "raw_p_ok": raw_p_ok,
        "years_ok": years_ok,
        "extra_ok": extra_ok,
    }


def _soft_fail_count(filtered_result: dict[str, object]) -> int:
    fails = 0
    if not bool(filtered_result["raw_p_ok"]):
        fails += 1
    if not bool(filtered_result["years_ok"]):
        fails += 1
    oos_dir_match = filtered_result["oos_dir_match"]
    n_on_oos = int(filtered_result["n_on_oos"])
    if n_on_oos >= 10 and oos_dir_match is False:
        fails += 1
    return fails


def _summarize_hypothesis(unfiltered: dict[str, object], filtered: dict[str, object]) -> str:
    unfiltered_core = (
        all(bool(unfiltered[key]) for key in ["delta_ok", "raw_p_ok", "extra_ok"])
        and not bool(unfiltered["extreme_fire"])
        and not bool(unfiltered["arithmetic_only"])
        and float(unfiltered.get("q_family", 1.0)) < BH_Q
    )
    filtered_core = (
        all(bool(filtered[key]) for key in ["delta_ok", "raw_p_ok", "years_ok", "extra_ok"])
        and not bool(filtered["extreme_fire"])
        and not bool(filtered["arithmetic_only"])
        and float(filtered.get("q_family", 1.0)) < BH_Q
    )
    filtered_oos_ok = int(filtered["n_on_oos"]) < 10 or filtered["oos_dir_match"] is not False
    if unfiltered_core and filtered_core and filtered_oos_ok:
        return "CONTINUE"
    if unfiltered_core and _soft_fail_count(filtered) == 1:
        return "PARK"
    return "KILL"


def _render_pass_table(result: dict[str, object]) -> list[str]:
    return [
        "| Pass | N_total | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | q_lane | years_pos | dir_match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {result['pass_name']} | {result['n_total']} | {result['n_on_is']} | {result['n_off_is']} | "
            f"{result['n_on_oos']} | {_fmt(result['expr_on_is'])} | {_fmt(result['expr_off_is'])} | "
            f"{_fmt(result['delta_is'])} | {_fmt(result['raw_p_is'])} | {_fmt(result.get('q_family'))} | "
            f"{_fmt(result.get('q_lane'))} | {result['years_positive']} | {result['oos_dir_match']} |"
        ),
    ]


def _render_year_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Year | N_on | N_off | Delta | Eligible_for_years_positive |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['year']} | {row['n_on']} | {row['n_off']} | {_fmt(row['delta'])} | {row['eligible']} |"
        )
    return lines


def main() -> int:
    prereg = _load_yaml(PREREG_PATH)
    if prereg.get("status") != "LOCKED":
        raise RuntimeError(f"pre-reg must be LOCKED before execution: {PREREG_PATH}")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    _assert_active_lanes(con)

    lane_frames: dict[str, pd.DataFrame] = {}
    deployed_fires: dict[str, np.ndarray] = {}
    rel_vol_thresholds: dict[str, float] = {}
    for lane_id in LANES:
        df = _load_lane(con, lane_id)
        if len(df) == 0:
            raise RuntimeError(f"Lane {lane_id} has zero canonical rows")
        lane_frames[lane_id] = df
        deployed_fires[lane_id] = filter_signal(df, LANES[lane_id]["filter_key"], LANES[lane_id]["session"]).astype(bool)
    con.close()

    cmx_short_base = lane_frames["MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5"].copy()
    cmx_short_base = cmx_short_base.loc[cmx_short_base["direction"] == "short"].copy()
    rel_vol_thresholds["MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5"] = _calibrate_rel_vol_threshold(cmx_short_base)

    results: list[dict[str, object]] = []
    for plan in HYPOTHESIS_PLANS:
        lane_df = lane_frames[plan.lane_id].copy()
        deployed_fire = deployed_fires[plan.lane_id]
        if plan.short_only:
            short_mask = lane_df["direction"].eq("short").to_numpy(dtype=bool)
            lane_df = lane_df.loc[short_mask].copy()
            deployed_fire = deployed_fire[short_mask]
        signal = _build_signal(lane_df, plan, rel_vol_thresholds)
        corr_vs_filter = _compute_corr(signal.astype(int), deployed_fire.astype(int))
        hyp_cfg = next(h for h in prereg["hypotheses"] if h["id"] == plan.id)
        for pass_name in ("unfiltered", "filtered"):
            results.append(_evaluate_pass(plan, hyp_cfg, pass_name, lane_df, signal, deployed_fire, corr_vs_filter))

    family_q = _bh_fdr([(f"{r['hypothesis_id']}::{r['pass_name']}", float(r["raw_p_is"])) for r in results])
    lane_q: dict[str, float] = {}
    for lane_id in LANES:
        lane_results = [r for r in results if r["lane_id"] == lane_id]
        lane_q.update(_bh_fdr([(f"{r['hypothesis_id']}::{r['pass_name']}", float(r["raw_p_is"])) for r in lane_results]))
    for result in results:
        key = f"{result['hypothesis_id']}::{result['pass_name']}"
        result["q_family"] = family_q.get(key, float("nan"))
        result["q_lane"] = lane_q.get(key, float("nan"))

    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result["hypothesis_id"]), {})[str(result["pass_name"])] = result

    verdicts: list[tuple[str, str]] = []
    continue_ids: list[str] = []
    park_ids: list[str] = []
    kill_ids: list[str] = []
    for hyp_id, passes in grouped.items():
        verdict = _summarize_hypothesis(passes["unfiltered"], passes["filtered"])
        verdicts.append((hyp_id, verdict))
        if verdict == "CONTINUE":
            continue_ids.append(hyp_id)
        elif verdict == "PARK":
            park_ids.append(hyp_id)
        else:
            kill_ids.append(hyp_id)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = build_header(
        prereg_path=PREREG_PATH,
        script_path=__file__,
        extra_lines=[
            f"**IS window:** trading_day < {HOLDOUT_SACRED_FROM}",
            f"**Observed tests:** {len(results)}",
        ],
        observed_cell_count=len(results),
    )
    lines.extend(
        [
            f"**Family verdict:** CONTINUE={len(continue_ids)} | PARK={len(park_ids)} | KILL={len(kill_ids)}",
            "",
            "## Summary",
            "",
            f"- `H03/H04` frozen IS-only rel_vol threshold on COMEX short lane: `{rel_vol_thresholds['MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5']:.6f}`",
            f"- `q_family` computed across all `{len(results)}` two-pass tests",
            "- `q_lane` computed within each physical lane bucket only",
            "",
            "| Hypothesis | Lane | Verdict | Unfiltered q_family | Filtered q_family | Filtered q_lane |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for hyp_id, verdict in verdicts:
        filtered = grouped[hyp_id]["filtered"]
        unfiltered = grouped[hyp_id]["unfiltered"]
        lines.append(
            f"| {hyp_id} | {filtered['lane_id']} | {verdict} | {_fmt(unfiltered['q_family'])} | {_fmt(filtered['q_family'])} | {_fmt(filtered['q_lane'])} |"
        )

    for hyp_id, verdict in verdicts:
        lines.extend(["", f"## {hyp_id}", ""])
        lines.append(f"**Verdict:** {verdict}")
        lines.append("")
        for pass_name in ("unfiltered", "filtered"):
            result = grouped[hyp_id][pass_name]
            lines.append(f"### {pass_name.title()} Pass")
            lines.append("")
            lines.extend(_render_pass_table(result))
            lines.append("")
            lines.append(
                f"- `corr_vs_filter={_fmt(result['corr_vs_filter'])}` | `extreme_fire={result['extreme_fire']}` | `arithmetic_only={result['arithmetic_only']}`"
            )
            lines.append("")
            lines.extend(_render_year_table(result["year_rows"]))  # type: ignore[arg-type]
            lines.append("")

    lines.extend(["## Closeout", "", "SURVIVED SCRUTINY:"])
    if continue_ids:
        lines.extend(f"- {hyp_id}" for hyp_id in continue_ids)
    else:
        lines.append("- none")
    lines.extend(["PARKED FOR MORE OOS:"])
    if park_ids:
        lines.extend(f"- {hyp_id}" for hyp_id in park_ids)
    else:
        lines.append("- none")
    lines.extend(["DID NOT SURVIVE:"])
    if kill_ids:
        lines.extend(f"- {hyp_id}" for hyp_id in kill_ids)
    else:
        lines.append("- none")
    lines.extend(
        [
            "CAVEATS:",
            "- OOS remains descriptive only under Mode A and several filtered branches are thin.",
            "- Family q-values are the load-bearing multiple-testing gate; lane q-values are secondary framing only.",
            "NEXT STEPS:",
            "- If no hypothesis survives both passes, kill this exact overlay bundle and do not broaden scope without a new pre-reg.",
            "- If a hypothesis parks, treat it as shadow-only until a fresh pre-reg narrows the unresolved criterion.",
        ]
    )
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    for hyp_id, verdict in verdicts:
        filtered = grouped[hyp_id]["filtered"]
        print(
            f"{hyp_id}: {verdict} | filtered delta={_fmt(filtered['delta_is'])} "
            f"raw_p={_fmt(filtered['raw_p_is'])} q_family={_fmt(filtered['q_family'])} "
            f"q_lane={_fmt(filtered['q_lane'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
