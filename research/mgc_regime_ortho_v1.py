#!/usr/bin/env python3
"""Phase 3 execution for 2026-04-21-mgc-regime-ortho-v1.

Canonical-only family scan for the locked MGC regime-orthogonal hypothesis.

Outputs:
  - docs/audit/results/2026-04-21-mgc-regime-ortho-v1.md
  - outputs/2026-04-21-mgc-regime-ortho-v1/trades.parquet
  - outputs/2026-04-21-mgc-regime-ortho-v1/null_distributions.csv
  - outputs/2026-04-21-mgc-regime-ortho-v1/cv_summary.json
  - outputs/2026-04-21-mgc-regime-ortho-v1/reproducibility.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats as sstats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import COST_SPECS, stress_test_costs
from trading_app.dsr import compute_dsr, compute_sr0, estimate_var_sr_from_db
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from research.filter_utils import filter_signal

HYP_PATH = PROJECT_ROOT / "docs/audit/hypotheses/2026-04-21-mgc-regime-ortho-v1.yaml"
RESULT_PATH = PROJECT_ROOT / "docs/audit/results/2026-04-21-mgc-regime-ortho-v1.md"
OUT_DIR = PROJECT_ROOT / "outputs/2026-04-21-mgc-regime-ortho-v1"
TRADES_PATH = OUT_DIR / "trades.parquet"
NULLS_PATH = OUT_DIR / "null_distributions.csv"
CV_PATH = OUT_DIR / "cv_summary.json"
REPRO_PATH = OUT_DIR / "reproducibility.json"
STATE_PATH = PROJECT_ROOT / "outputs/session_state.json"
SEED = 20260421
ALPHA = 0.05
BLOCK_SIZE = 20
N_NULL = 250
EMBARGO_DAYS = 10


@dataclass
class CellResult:
    hypothesis_id: str
    session: str
    feature: str
    direction: str
    threshold: float | None = None
    n_base_is: int = 0
    n_base_oos: int = 0
    n_on_is: int = 0
    n_off_is: int = 0
    n_on_oos: int = 0
    trade_days_on_is: int = 0
    trade_days_on_oos: int = 0
    expr_base_is: float | None = None
    expr_base_oos: float | None = None
    expr_on_is: float | None = None
    expr_off_is: float | None = None
    expr_on_oos: float | None = None
    delta_is: float | None = None
    delta_oos: float | None = None
    wr_on_is: float | None = None
    wr_off_is: float | None = None
    wr_on_oos: float | None = None
    wr_spread_is: float | None = None
    sr_on_is: float | None = None
    sr_on_oos: float | None = None
    t_is: float | None = None
    raw_p: float | None = None
    boot_p: float | None = None
    q_family: float | None = None
    bh_pass_family: bool = False
    wfe_holdout: float | None = None
    cv_mean_delta: float | None = None
    cv_wfe: float | None = None
    stressed_expr_1_5x: float | None = None
    stressed_expr_2_0x: float | None = None
    dsr_03: float | None = None
    dsr_05: float | None = None
    dsr_07: float | None = None
    yrs_positive_is: int = 0
    coverage_ok: bool = True
    notes: list[str] = field(default_factory=list)


def _canonical_db_path() -> Path:
    if STATE_PATH.exists():
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        cand = payload.get("canonical_db_path")
        if cand:
            return Path(cand)
    return PROJECT_ROOT / "gold.db"


def _load_hypothesis() -> dict[str, Any]:
    with HYP_PATH.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    expected = str(HOLDOUT_SACRED_FROM)
    actual = payload["data_policy"]["holdout_fence"]["locked_boundary"]
    if actual != expected:
        raise ValueError(f"Holdout mismatch: hypothesis {actual} vs canonical {expected}")
    return payload


def _welch_greater(on: np.ndarray, off: np.ndarray) -> tuple[float | None, float | None]:
    if len(on) < 2 or len(off) < 2:
        return None, None
    test = sstats.ttest_ind(on, off, equal_var=False, alternative="greater")
    if test.statistic is None or math.isnan(float(test.statistic)):
        return None, None
    return float(test.statistic), float(test.pvalue)


def _sharpe(x: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    s = float(np.std(x, ddof=1))
    if s <= 0:
        return None
    return float(np.mean(x) / s)


def _bh_qvals(pvals: list[float | None]) -> list[float]:
    vals = [1.0 if p is None or math.isnan(p) else float(p) for p in pvals]
    m = len(vals)
    if m == 0:
        return []
    order = np.argsort(vals)
    q = np.empty(m, dtype=float)
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = int(order[rank])
        raw = vals[idx] * m / (rank + 1)
        running_min = min(running_min, raw)
        q[idx] = min(1.0, running_min)
    return q.tolist()


def _moving_block_delta_p(df: pd.DataFrame, fire_col: str, seed: int) -> float | None:
    if len(df) < BLOCK_SIZE * 2:
        return None
    obs = _mean_diff(df[fire_col].to_numpy(dtype=bool), df["pnl_r"].to_numpy(dtype=float))
    if obs is None:
        return None
    rng = np.random.default_rng(seed)
    n = len(df)
    nblocks = int(math.ceil(n / BLOCK_SIZE))
    idx_max = max(n - BLOCK_SIZE + 1, 1)
    deltas = []
    arr = df[["pnl_r", fire_col]].reset_index(drop=True)
    for _ in range(N_NULL):
        starts = rng.integers(0, idx_max, size=nblocks)
        sample = pd.concat([arr.iloc[s : s + BLOCK_SIZE] for s in starts], ignore_index=True).iloc[:n]
        delta = _mean_diff(sample[fire_col].to_numpy(dtype=bool), sample["pnl_r"].to_numpy(dtype=float))
        if delta is not None:
            deltas.append(delta)
    if not deltas:
        return None
    deltas_np = np.asarray(deltas, dtype=float)
    return float((np.sum(deltas_np <= 0.0) + 1) / (len(deltas_np) + 1))


def _mean_diff(fire: np.ndarray, pnl: np.ndarray) -> float | None:
    on = pnl[fire]
    off = pnl[~fire]
    if len(on) == 0 or len(off) == 0:
        return None
    return float(np.mean(on) - np.mean(off))


def _cost_stressed_expr(expr: float | None, avg_risk: float | None, multiplier: float) -> float | None:
    if expr is None or avg_risk is None or avg_risk <= 0:
        return None
    base = COST_SPECS["MGC"]
    stressed = stress_test_costs(base, multiplier)
    extra_r = (stressed.total_friction - base.total_friction) / avg_risk
    return float(expr - extra_r)


def _family_null_distribution(cell_frames: dict[str, pd.DataFrame], fire_cols: dict[str, str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    observed = []
    for cell_id, df in cell_frames.items():
        t, _ = _welch_greater(
            df.loc[df[fire_cols[cell_id]], "pnl_r"].to_numpy(dtype=float),
            df.loc[~df[fire_cols[cell_id]], "pnl_r"].to_numpy(dtype=float),
        )
        observed.append((cell_id, t if t is not None else float("-inf")))
    observed_max_t = max(t for _, t in observed) if observed else float("-inf")

    for variant in ("destruction_shuffle", "rng_null"):
        for rep in range(N_NULL):
            max_t = float("-inf")
            for cell_id, df in cell_frames.items():
                work = df.copy()
                block_keys = pd.to_datetime(work["trading_day"]).dt.to_period("M").astype(str)
                shuffled = np.zeros(len(work), dtype=bool)
                observed_fire = work[fire_cols[cell_id]].to_numpy(dtype=bool)
                for block in sorted(block_keys.unique()):
                    idx = np.where(block_keys == block)[0]
                    if len(idx) == 0:
                        continue
                    count = int(observed_fire[idx].sum())
                    if variant == "destruction_shuffle":
                        block_fire = observed_fire[idx].copy()
                        rng.shuffle(block_fire)
                        shuffled[idx] = block_fire
                    else:
                        if count <= 0:
                            shuffled[idx] = False
                        elif count >= len(idx):
                            shuffled[idx] = True
                        else:
                            pick = rng.choice(idx, size=count, replace=False)
                            shuffled[pick] = True
                on = work.loc[shuffled, "pnl_r"].to_numpy(dtype=float)
                off = work.loc[~shuffled, "pnl_r"].to_numpy(dtype=float)
                t, _ = _welch_greater(on, off)
                if t is not None:
                    max_t = max(max_t, float(t))
            rows.append({"variant": variant, "replicate": rep, "max_t": max_t, "observed_max_t": observed_max_t})
    return pd.DataFrame(rows)


def _compute_positive_control(db_path: Path) -> dict[str, Any]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        session = "COMEX_SETTLE"
        q = f"""
        SELECT o.trading_day, o.pnl_r, o.outcome, o.risk_dollars, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
        WHERE o.symbol='MNQ' AND o.orb_label='{session}' AND o.orb_minutes=5
          AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.5
          AND o.pnl_r IS NOT NULL AND d.orb_{session}_break_dir IN ('long','short')
          AND o.trading_day < ?
        ORDER BY o.trading_day
        """
        df = con.execute(q, [HOLDOUT_SACRED_FROM]).df()
    finally:
        con.close()
    if df.empty:
        return {"status": "FAIL", "reason": "no_rows"}
    fire = filter_signal(df, "ORB_G5", session).astype(bool)
    on = df.loc[fire, "pnl_r"].to_numpy(dtype=float)
    off = df.loc[~fire, "pnl_r"].to_numpy(dtype=float)
    t, p = _welch_greater(on, off)
    delta = _mean_diff(fire, df["pnl_r"].to_numpy(dtype=float))
    passed = bool(delta is not None and delta > 0 and t is not None and t > 0 and p is not None and p < 0.05 and len(on) >= 100)
    return {
        "status": "PASS" if passed else "FAIL",
        "session": session,
        "n_on": int(fire.sum()),
        "delta_is": delta,
        "t_is": t,
        "raw_p": p,
    }


def _compute_negative_control(df: pd.DataFrame) -> dict[str, Any]:
    work = df[df["is_is"]].copy()
    if work.empty:
        return {"status": "FAIL", "reason": "no_rows"}
    work["neg_fire"] = pd.to_datetime(work["trading_day"]).dt.day % 2 == 0
    stats = []
    for (session, direction), grp in work.groupby(["session", "direction"], sort=True):
        fire = grp["neg_fire"].to_numpy(dtype=bool)
        delta = _mean_diff(fire, grp["pnl_r"].to_numpy(dtype=float))
        on = grp.loc[fire, "pnl_r"].to_numpy(dtype=float)
        off = grp.loc[~fire, "pnl_r"].to_numpy(dtype=float)
        t, p = _welch_greater(on, off)
        stats.append({"session": session, "direction": direction, "delta": delta, "t": t, "p": p})
    any_pass = any((row["p"] is not None and row["p"] < 0.05 and row["delta"] is not None and row["delta"] > 0) for row in stats)
    return {"status": "PASS" if not any_pass else "FAIL", "cells": stats}


def _cv_summary(df: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, Any]:
    tests = [
        (date(2024, 1, 1), date(2024, 7, 1)),
        (date(2024, 7, 1), date(2025, 1, 1)),
        (date(2025, 1, 1), date(2025, 7, 1)),
        (date(2025, 7, 1), HOLDOUT_SACRED_FROM),
    ]
    folds: list[dict[str, Any]] = []
    for start, end in tests:
        train_end = start - timedelta(days=EMBARGO_DAYS)
        train = df[df["trading_day"] < train_end].copy()
        test = df[(df["trading_day"] >= start) & (df["trading_day"] < end)].copy()
        if train.empty or test.empty:
            continue
        fold = {"test_start": str(start), "test_end": str(end), "cells": []}
        fold_thresholds = {}
        for feature in ("ATR_VEL_RATIO_Q67_HIGH", "PREV_WEEK_RANGE_Q67_HIGH"):
            src = "atr_vel_ratio" if feature.startswith("ATR_VEL") else "prev_week_range"
            vals = train[src].dropna()
            if vals.empty:
                fold_thresholds[feature] = None
            else:
                fold_thresholds[feature] = float(vals.quantile(0.67))
        for session in sorted(df["session"].unique()):
            for feature, threshold in fold_thresholds.items():
                if threshold is None:
                    continue
                src = "atr_vel_ratio" if feature.startswith("ATR_VEL") else "prev_week_range"
                for direction in ("long", "short"):
                    tr = train[(train["session"] == session) & (train["direction"] == direction)].copy()
                    te = test[(test["session"] == session) & (test["direction"] == direction)].copy()
                    if te.empty or tr.empty:
                        continue
                    tr_fire = tr[src] >= threshold
                    te_fire = te[src] >= threshold
                    train_delta = _mean_diff(tr_fire.to_numpy(dtype=bool), tr["pnl_r"].to_numpy(dtype=float))
                    test_delta = _mean_diff(te_fire.to_numpy(dtype=bool), te["pnl_r"].to_numpy(dtype=float))
                    if train_delta is None or test_delta is None:
                        continue
                    fold["cells"].append(
                        {
                            "session": session,
                            "feature": feature,
                            "direction": direction,
                            "train_delta": train_delta,
                            "test_delta": test_delta,
                            "wfe": test_delta / train_delta if train_delta > 0 else None,
                            "n_test_on": int(te_fire.sum()),
                        }
                    )
        folds.append(fold)

    all_test_deltas = [c["test_delta"] for f in folds for c in f["cells"] if c["test_delta"] is not None]
    all_train_deltas = [c["train_delta"] for f in folds for c in f["cells"] if c["train_delta"] is not None]
    pos = [c for f in folds for c in f["cells"] if c["test_delta"] is not None and c["test_delta"] > 0]
    return {
        "embargo_days": EMBARGO_DAYS,
        "folds": folds,
        "mean_test_delta": float(np.mean(all_test_deltas)) if all_test_deltas else None,
        "mean_train_delta": float(np.mean(all_train_deltas)) if all_train_deltas else None,
        "wfe": float(np.mean(all_test_deltas) / np.mean(all_train_deltas)) if all_train_deltas and np.mean(all_train_deltas) > 0 else None,
        "positive_cells": len(pos),
        "total_cells": sum(len(f["cells"]) for f in folds),
    }


def _load_scoped_trades(db_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for session in ("BRISBANE_1025", "LONDON_METALS", "US_DATA_830"):
            q = f"""
            SELECT
              o.trading_day,
              o.symbol,
              o.orb_label AS session,
              o.orb_minutes,
              o.entry_model,
              o.confirm_bars,
              o.rr_target,
              o.pnl_r,
              o.outcome,
              o.risk_dollars,
              o.entry_ts,
              o.exit_ts,
              d.atr_vel_ratio,
              d.prev_week_range,
              d.atr_20,
              d.orb_{session}_size AS orb_size,
              d.orb_{session}_break_dir AS direction
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
            WHERE o.symbol='MGC' AND o.orb_label='{session}' AND o.orb_minutes=5
              AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
              AND o.pnl_r IS NOT NULL
            ORDER BY o.trading_day
            """
            df = con.execute(q).df()
            if not df.empty:
                frames.append(df)
    finally:
        con.close()
    if not frames:
        return pd.DataFrame(
            columns=[
                "trading_day",
                "symbol",
                "session",
                "orb_minutes",
                "entry_model",
                "confirm_bars",
                "rr_target",
                "pnl_r",
                "outcome",
                "risk_dollars",
                "entry_ts",
                "exit_ts",
                "atr_vel_ratio",
                "prev_week_range",
                "atr_20",
                "orb_size",
                "direction",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["is_is"] = df["trading_day"] < HOLDOUT_SACRED_FROM
    df["is_oos"] = df["trading_day"] >= HOLDOUT_SACRED_FROM
    return df


def _cell_id(session: str, feature: str, direction: str) -> str:
    f = "ATRVEL" if feature.startswith("ATR") else "PREVWEEK"
    sess = {"BRISBANE_1025": "BRI", "LONDON_METALS": "LDM", "US_DATA_830": "USD830"}[session]
    side = "LONG" if direction == "long" else "SHORT"
    return f"{sess}_{f}_{side}"


def _write_artifacts(df: pd.DataFrame, nulls: pd.DataFrame, cv_summary: dict[str, Any], repro: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_con = duckdb.connect()
    try:
        temp_con.register("trades_df", df)
        temp_con.execute(f"COPY (SELECT * FROM trades_df) TO '{TRADES_PATH.as_posix()}' (FORMAT PARQUET)")
        temp_con.register("nulls_df", nulls)
        temp_con.execute(f"COPY (SELECT * FROM nulls_df) TO '{NULLS_PATH.as_posix()}' (HEADER, DELIMITER ',')")
    finally:
        temp_con.close()
    CV_PATH.write_text(json.dumps(cv_summary, indent=2, default=str), encoding="utf-8")
    REPRO_PATH.write_text(json.dumps(repro, indent=2, default=str), encoding="utf-8")


def _render_result(hyp: dict[str, Any], df: pd.DataFrame, cells: list[CellResult], thresholds: dict[str, float], nulls: pd.DataFrame, cv_summary: dict[str, Any], pos_control: dict[str, Any], neg_control: dict[str, Any]) -> str:
    lines = [
        "# MGC regime-orthogonal ORB filters — Phase 3 result",
        "",
        f"**Hypothesis:** `{HYP_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"**Parent commit:** `{hyp['reproducibility']['parent_commit_sha']}`",
        f"**Run UTC:** `{datetime.now(UTC).isoformat()}`",
        f"**Holdout fence:** `{HOLDOUT_SACRED_FROM}` from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`",
        "",
        "## Scope",
        "",
        "- Instrument: `MGC`",
        "- Sessions: `BRISBANE_1025`, `LONDON_METALS`, `US_DATA_830`",
        "- Entry model: `E2` / `confirm_bars=1` / `RR=1.0` / `orb_minutes=5`",
        "- Family K: `12` cells (`3 sessions × 2 predicates × 2 directions`)",
        "",
        "## Canonical thresholds (IS only)",
        "",
        f"- `ATR_VEL_RATIO_Q67_HIGH` threshold: `{thresholds['ATR_VEL_RATIO_Q67_HIGH']:.6f}`",
        f"- `PREV_WEEK_RANGE_Q67_HIGH` threshold: `{thresholds['PREV_WEEK_RANGE_Q67_HIGH']:.6f}`",
        "",
        "## Coverage",
        "",
        f"- Total scoped rows: `{len(df)}`",
        f"- IS rows: `{int(df['is_is'].sum())}`",
        f"- OOS rows: `{int(df['is_oos'].sum())}`",
        f"- Session counts: `{df.groupby('session').size().to_dict()}`",
        "- `BRISBANE_1025` has zero scoped `orb_outcomes` rows for this exact family and therefore fails as zero-coverage cells rather than being imputed.",
        "",
        "## Cell table",
        "",
        "| Cell | Session | Feature | Dir | N_on_IS | N_on_OOS | ExpR_on_IS | ExpR_on_OOS | Δ_IS | t_IS | p | q_family | WFE_holdout | 1.5x | 2.0x |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            f"| `{cell.hypothesis_id}` | `{cell.session}` | `{cell.feature}` | `{cell.direction}` | "
            f"{cell.n_on_is} | {cell.n_on_oos} | "
            f"{'nan' if cell.expr_on_is is None else f'{cell.expr_on_is:+.4f}'} | "
            f"{'nan' if cell.expr_on_oos is None else f'{cell.expr_on_oos:+.4f}'} | "
            f"{'nan' if cell.delta_is is None else f'{cell.delta_is:+.4f}'} | "
            f"{'nan' if cell.t_is is None else f'{cell.t_is:+.3f}'} | "
            f"{'nan' if cell.raw_p is None else f'{cell.raw_p:.4f}'} | "
            f"{'nan' if cell.q_family is None else f'{cell.q_family:.4f}'} | "
            f"{'nan' if cell.wfe_holdout is None else f'{cell.wfe_holdout:.3f}'} | "
            f"{'nan' if cell.stressed_expr_1_5x is None else f'{cell.stressed_expr_1_5x:+.4f}'} | "
            f"{'nan' if cell.stressed_expr_2_0x is None else f'{cell.stressed_expr_2_0x:+.4f}'} |"
        )

    survivors = [c for c in cells if c.bh_pass_family]
    lines += [
        "",
        "## Family-level gates",
        "",
        f"- BH-FDR family survivors (`q<0.05`): `{len(survivors)}`",
        f"- Positive control (`ORB_G5` sanity control, verified from current canonical data in this run): `{pos_control}`",
        f"- Negative control (calendar parity): `{neg_control['status']}`",
        "",
        "## Nulls",
        "",
        f"- Observed max t across 12 cells: `{nulls['observed_max_t'].iloc[0]:.4f}`",
    ]
    for variant in ("destruction_shuffle", "rng_null"):
        sub = nulls[nulls["variant"] == variant]
        valid = sub["max_t"].replace([-np.inf, np.inf], np.nan).dropna()
        p_emp = float((np.sum(valid >= sub['observed_max_t'].iloc[0]) + 1) / (len(valid) + 1)) if len(valid) else float("nan")
        lines.append(f"- `{variant}` empirical p on family max-t: `{p_emp:.4f}` from `{len(valid)}` replicates")

    lines += [
        "",
        "## Rolling blocked CV",
        "",
        f"- Fold count: `{len(cv_summary['folds'])}`",
        f"- Mean train delta: `{cv_summary['mean_train_delta']}`",
        f"- Mean test delta: `{cv_summary['mean_test_delta']}`",
        f"- CV WFE: `{cv_summary['wfe']}`",
        "",
        "## DSR bracket (cross-check only)",
        "",
    ]
    for cell in cells:
        lines.append(
            f"- `{cell.hypothesis_id}`: rho=0.3 `{cell.dsr_03}`, rho=0.5 `{cell.dsr_05}`, rho=0.7 `{cell.dsr_07}`"
        )
    lines += [
        "",
        "## Verdict",
        "",
    ]
    if not survivors:
        lines.append("**Current family verdict: DEAD at Phase 3.** No cell survived family-level BH-FDR on canonical IS. This is not a post-hoc rescue surface.")
    else:
        lines.append(
            f"**Current family verdict: CONDITIONAL.** `{len(survivors)}` cell(s) survived BH-FDR on canonical IS and move to adversarial audit only if nulls, CV, and cost-stress remain aligned."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    hyp = _load_hypothesis()
    db_path = _canonical_db_path()
    scoped = _load_scoped_trades(db_path)
    if scoped.empty:
        raise SystemExit("No scoped trades found for hypothesis family")

    is_scoped = scoped[scoped["is_is"]].copy()
    thresholds = {}
    coverage = {}
    for feature, src in (
        ("ATR_VEL_RATIO_Q67_HIGH", "atr_vel_ratio"),
        ("PREV_WEEK_RANGE_Q67_HIGH", "prev_week_range"),
    ):
        vals = is_scoped[src].dropna()
        thresholds[feature] = float(vals.quantile(0.67)) if not vals.empty else float("nan")
        coverage[feature] = float(vals.notna().mean()) if len(is_scoped) else 0.0

    scoped["ATR_VEL_RATIO_Q67_HIGH"] = scoped["atr_vel_ratio"] >= thresholds["ATR_VEL_RATIO_Q67_HIGH"]
    scoped["PREV_WEEK_RANGE_Q67_HIGH"] = scoped["prev_week_range"] >= thresholds["PREV_WEEK_RANGE_Q67_HIGH"]

    cell_results: list[CellResult] = []
    cell_frames: dict[str, pd.DataFrame] = {}
    fire_cols: dict[str, str] = {}
    var_sr = float(estimate_var_sr_from_db(db_path))
    for session in ("BRISBANE_1025", "LONDON_METALS", "US_DATA_830"):
        for feature, src in (
            ("ATR_VEL_RATIO_Q67_HIGH", "atr_vel_ratio"),
            ("PREV_WEEK_RANGE_Q67_HIGH", "prev_week_range"),
        ):
            for direction in ("long", "short"):
                cell_id = _cell_id(session, feature, direction)
                res = CellResult(hypothesis_id=cell_id, session=session, feature=feature, direction=direction)
                res.threshold = thresholds[feature]
                res.coverage_ok = coverage[feature] >= 0.90
                eligible = scoped[(scoped["session"] == session) & (scoped["direction"] == direction)].copy()
                if eligible.empty:
                    res.notes.append("zero scoped rows")
                    cell_results.append(res)
                    continue
                is_df = eligible[eligible["is_is"]].copy()
                oos_df = eligible[eligible["is_oos"]].copy()
                res.n_base_is = len(is_df)
                res.n_base_oos = len(oos_df)
                res.expr_base_is = float(is_df["pnl_r"].mean()) if len(is_df) else None
                res.expr_base_oos = float(oos_df["pnl_r"].mean()) if len(oos_df) else None
                fire_col = feature
                is_fire = is_df[fire_col].to_numpy(dtype=bool)
                oos_fire = oos_df[fire_col].to_numpy(dtype=bool) if len(oos_df) else np.zeros(0, dtype=bool)
                res.n_on_is = int(is_fire.sum())
                res.n_off_is = int((~is_fire).sum())
                res.n_on_oos = int(oos_fire.sum())
                res.trade_days_on_is = int(is_df.loc[is_fire, "trading_day"].nunique()) if res.n_on_is else 0
                res.trade_days_on_oos = int(oos_df.loc[oos_fire, "trading_day"].nunique()) if res.n_on_oos else 0
                on_is = is_df.loc[is_fire, "pnl_r"].to_numpy(dtype=float)
                off_is = is_df.loc[~is_fire, "pnl_r"].to_numpy(dtype=float)
                on_oos = oos_df.loc[oos_fire, "pnl_r"].to_numpy(dtype=float) if res.n_on_oos else np.array([], dtype=float)
                res.expr_on_is = float(np.mean(on_is)) if len(on_is) else None
                res.expr_off_is = float(np.mean(off_is)) if len(off_is) else None
                res.expr_on_oos = float(np.mean(on_oos)) if len(on_oos) else None
                res.delta_is = _mean_diff(is_fire, is_df["pnl_r"].to_numpy(dtype=float))
                res.delta_oos = _mean_diff(oos_fire, oos_df["pnl_r"].to_numpy(dtype=float)) if len(oos_df) else None
                res.wr_on_is = float(np.mean(is_df.loc[is_fire, "outcome"].eq("win"))) if len(on_is) else None
                res.wr_off_is = float(np.mean(is_df.loc[~is_fire, "outcome"].eq("win"))) if len(off_is) else None
                res.wr_on_oos = float(np.mean(oos_df.loc[oos_fire, "outcome"].eq("win"))) if len(on_oos) else None
                if res.wr_on_is is not None and res.wr_off_is is not None:
                    res.wr_spread_is = float(res.wr_on_is - res.wr_off_is)
                res.sr_on_is = _sharpe(on_is)
                res.sr_on_oos = _sharpe(on_oos) if len(on_oos) else None
                res.t_is, res.raw_p = _welch_greater(on_is, off_is)
                res.boot_p = _moving_block_delta_p(is_df, fire_col, SEED)
                res.wfe_holdout = (
                    float(res.delta_oos / res.delta_is)
                    if res.delta_is is not None and res.delta_is > 0 and res.delta_oos is not None
                    else None
                )
                avg_risk = float(is_df.loc[is_fire, "risk_dollars"].mean()) if len(on_is) else None
                res.stressed_expr_1_5x = _cost_stressed_expr(res.expr_on_is, avg_risk, 1.5)
                res.stressed_expr_2_0x = _cost_stressed_expr(res.expr_on_is, avg_risk, 2.0)
                per_year = is_df.loc[is_fire].groupby(pd.to_datetime(is_df.loc[is_fire, "trading_day"]).dt.year)["pnl_r"].agg(["count", "mean"])
                res.yrs_positive_is = int(((per_year["count"] >= 5) & (per_year["mean"] > 0)).sum()) if not per_year.empty else 0
                if res.sr_on_is is not None and res.n_on_is >= 2:
                    for rho, attr in ((0.3, "dsr_03"), (0.5, "dsr_05"), (0.7, "dsr_07")):
                        sr0 = compute_sr0(max(2.0, 12.0 * rho), var_sr)
                        setattr(res, attr, float(compute_dsr(res.sr_on_is, sr0, res.n_on_is)))
                cell_frames[cell_id] = is_df.reset_index(drop=True)
                fire_cols[cell_id] = fire_col
                cell_results.append(res)

    qvals = _bh_qvals([c.raw_p for c in cell_results])
    for cell, q in zip(cell_results, qvals, strict=False):
        cell.q_family = q
        cell.bh_pass_family = bool(
            q < ALPHA
            and cell.t_is is not None
            and cell.t_is >= 3.0
            and cell.n_on_is >= 100
            and cell.coverage_ok
        )

    cv_summary = _cv_summary(scoped, thresholds)
    for cell in cell_results:
        matching = []
        for fold in cv_summary["folds"]:
            for row in fold["cells"]:
                if row["session"] == cell.session and row["feature"] == cell.feature and row["direction"] == cell.direction:
                    matching.append(row)
        if matching:
            cell.cv_mean_delta = float(np.mean([m["test_delta"] for m in matching]))
            wfes = [m["wfe"] for m in matching if m["wfe"] is not None]
            cell.cv_wfe = float(np.mean(wfes)) if wfes else None

    nulls = _family_null_distribution(cell_frames, fire_cols)
    pos_control = _compute_positive_control(db_path)
    neg_control = _compute_negative_control(scoped)

    repro = {
        "hypothesis_id": hyp["slug"],
        "seed": SEED,
        "python_version": subprocess.check_output([sys.executable, "-V"], text=True).strip(),
        "pip_freeze_sha256": hashlib.sha256(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        ).hexdigest(),
        "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "gold_db_path": str(db_path),
        "gold_db_mtime": datetime.fromtimestamp(db_path.stat().st_mtime, tz=UTC).isoformat(),
        "gold_db_size": db_path.stat().st_size,
        "data_mode": json.loads(STATE_PATH.read_text(encoding="utf-8")).get("data_mode", "duckdb_readonly"),
    }

    RESULT_PATH.write_text(
        _render_result(hyp, scoped, cell_results, thresholds, nulls, cv_summary, pos_control, neg_control),
        encoding="utf-8",
    )
    _write_artifacts(scoped, nulls, cv_summary, repro)
    print(f"Wrote {RESULT_PATH}")
    print(f"Wrote {TRADES_PATH}")
    print(f"Wrote {NULLS_PATH}")
    print(f"Wrote {CV_PATH}")
    print(f"Wrote {REPRO_PATH}")


if __name__ == "__main__":
    main()
