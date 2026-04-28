# research/audit_sizing_substrate_diagnostic.py
"""Sizing-substrate Stage-1 diagnostic.

Per docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md v0.2 and the
locked pre-reg YAML at docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml.

Read-only over gold.db. Raises on any 2026 row.

# e2-lookahead-policy: cleared
# This script implements the E2 lookahead gate via feature_temporal_validity() at line ~356.
# Break-bar suffixes (_break_delay_min, _break_bar_volume, _break_bar_continues, _break_ts,
# _break_dir) are explicitly rejected for E2 lanes in _E2_LOOKAHEAD_BREAK_BAR_SUFFIXES.
# The script references these suffixes to enforce the ban, not to use them as predictors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from pipeline.paths import GOLD_DB_PATH


PREREG_PATH = Path("docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml")
RESULT_MD = Path("docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md")
RESULT_JSON = Path("docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json")


def is_holdout_clean(df: pd.DataFrame, holdout: str) -> bool:
    """Return True iff no row's trading_day >= holdout. Raise RuntimeError otherwise."""
    holdout_ts = pd.Timestamp(holdout)
    if (pd.to_datetime(df["trading_day"]) >= holdout_ts).any():
        raise RuntimeError(f"holdout row detected (>= {holdout})")
    return True


def null_coverage_mark(f: pd.Series, threshold: float) -> tuple[str, float]:
    """Return ('OK', drop_frac) if drop_frac<=threshold, else ('INVALID', drop_frac)."""
    drop_frac = float(f.isna().mean())
    return ("OK" if drop_frac <= threshold else "INVALID", drop_frac)


def power_floor_mark(n: int, min_n: int) -> str:
    return "OK" if n >= min_n else "UNDERPOWERED"


def compute_quintile_lift(df: pd.DataFrame, feature_col: str, outcome_col: str) -> dict[str, Any]:
    """Bin into quintiles by feature_col, return per-quintile mean outcome + monotonic flag."""
    quintiles = pd.qcut(df[feature_col], q=5, labels=False, duplicates="drop")
    means = df.groupby(quintiles)[outcome_col].mean().sort_index().tolist()
    if len(means) < 5:
        return {"q_means": means, "monotonic": False, "q1_mean_r": float("nan"),
                "q5_mean_r": float("nan"), "q5_minus_q1": 0.0}
    inc = all(means[i] <= means[i + 1] for i in range(4))
    dec = all(means[i] >= means[i + 1] for i in range(4))
    return {"q_means": means, "monotonic": inc or dec,
            "q1_mean_r": means[0], "q5_mean_r": means[4],
            "q5_minus_q1": means[4] - means[0]}


def bootstrap_sized_vs_flat_ci(
    df: pd.DataFrame, feature_col: str, outcome_col: str,
    weights: tuple[float, float, float, float, float],
    predicted_sign: str, B: int, seed: int, alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap mean(sized) - mean(flat). Weights applied per-quintile in predicted-sign direction."""
    quintiles = pd.qcut(df[feature_col], q=5, labels=False, duplicates="drop")
    pnl = df[outcome_col].to_numpy()
    qarr = quintiles.to_numpy()
    w_pos = np.array(weights)
    w_neg = w_pos[::-1]
    w = w_pos if predicted_sign == "+" else w_neg
    weight_per_trade = w[qarr]  # mean is 1.0 by construction since qcut bins are equi-count
    sized_pnl = pnl * weight_per_trade
    delta_obs = sized_pnl.mean() - pnl.mean()

    rng = np.random.default_rng(seed)
    n = len(pnl)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        deltas[b] = sized_pnl[idx].mean() - pnl[idx].mean()
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return {"observed": float(delta_obs), "lo": lo, "hi": hi}


def apply_bh_fdr(pvals: list[float], q: float) -> list[bool]:
    """Benjamini-Hochberg FDR control. Returns list of pass/fail in original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * (np.arange(1, m + 1)) / m
    passes_sorted = ranked <= thresholds
    if passes_sorted.any():
        max_k = int(np.max(np.where(passes_sorted)[0]))
        passes_sorted = np.arange(m) <= max_k
    out = np.zeros(m, dtype=bool)
    out[order] = passes_sorted
    return out.tolist()


def check_forecast_stability(
    df: pd.DataFrame, feature_col: str, window: int, max_rel_var: float,
) -> str:
    """STABLE if rolling SD relative-variation <= max_rel_var, else UNSTABLE."""
    rolling_sd = df[feature_col].rolling(window=window, min_periods=window // 2).std().dropna()
    if len(rolling_sd) < 2:
        return "STABLE"  # too short to flag instability
    sd_med = rolling_sd.median()
    if sd_med <= 0:
        return "STABLE"
    rel_var = (rolling_sd.max() - rolling_sd.min()) / sd_med
    return "STABLE" if rel_var <= max_rel_var else "UNSTABLE"


def sign_match_split_half(df: pd.DataFrame, feature_col: str, outcome_col: str) -> bool:
    """Split by median trading_day; require Spearman rho sign match in both halves."""
    df_sorted = df.sort_values("trading_day").reset_index(drop=True)
    median_day = df_sorted["trading_day"].iloc[len(df_sorted) // 2]
    h1 = df_sorted[df_sorted["trading_day"] < median_day]
    h2 = df_sorted[df_sorted["trading_day"] >= median_day]
    if len(h1) < 30 or len(h2) < 30:
        return False
    r1, _ = stats.spearmanr(h1[feature_col], h1[outcome_col])
    r2, _ = stats.spearmanr(h2[feature_col], h2[outcome_col])
    return bool(np.sign(r1) == np.sign(r2) and r1 != 0 and r2 != 0)


def classify_cell(cell: dict[str, Any]) -> str:
    """Return PASS / FAIL / INVALID per spec §5.3 + §5.4a.

    Spec §5.3 final bullet: UNSTABLE cells may PASS numerically but cannot be
    promoted to Stage 2 without an explicit forecast-normalization pre-reg.
    The implementation here returns PASS for numerically-passing UNSTABLE cells
    and sets stage2_eligible=False on them (set by stage2_eligible_flag below).
    """
    if cell.get("null_status") == "INVALID":
        return "INVALID"
    if cell.get("power_status") == "UNDERPOWERED":
        return "FAIL"
    required = [
        abs(cell.get("rho", 0.0)) >= 0.10,
        cell.get("bh_fdr_pass") is True,
        cell.get("monotonic") is True,
        abs(cell.get("q5_minus_q1", 0.0)) >= 0.20,
        cell.get("sized_flat_delta_lo", 0.0) > 0,
        cell.get("split_half_rho_match") is True,
        cell.get("split_half_delta_match") is True,
        cell.get("predicted_sign") == cell.get("realized_sign"),
    ]
    return "PASS" if all(required) else "FAIL"


def stage2_eligible_flag(cell: dict[str, Any]) -> bool:
    """A PASS cell is Stage-2 eligible only if forecast-stability is STABLE.

    Per spec §5.3 final bullet — UNSTABLE numerically-passing cells require a
    fresh Stage-2 pre-reg with explicit forecast-normalization, so they are
    NOT eligible for direct Stage-2 promotion under this diagnostic.
    """
    if cell.get("status") != "PASS":
        return False
    return cell.get("stability_status") == "STABLE"


def load_prereg() -> dict:
    return yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))


def load_lane_tape(con: duckdb.DuckDBPyConnection, lane: dict, holdout: str) -> pd.DataFrame:
    """Load IS trade tape for one lane joined with daily_features. Raise on holdout row."""
    q = """
    SELECT o.trading_day, o.symbol, o.pnl_r, d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.rr_target = ?
      AND o.confirm_bars = ?
      AND o.entry_model = ?
      AND o.pnl_r IS NOT NULL
      AND o.trading_day < CAST(? AS DATE)
    ORDER BY o.trading_day
    """
    df = con.execute(q, [
        lane["instrument"], lane["orb_label"], lane["orb_minutes"],
        lane["rr_target"], lane["confirm_bars"], lane["entry_model"],
        holdout,
    ]).fetchdf()
    is_holdout_clean(df, holdout=holdout)  # raises if leak
    return df


def resolve_substrate_column(lane: dict, form_id: str) -> str | None:
    """Map (lane.deployed_filter, form_id) to a daily_features column name OR a derivation key."""
    f = lane["deployed_filter"]
    sess = lane["orb_label"]
    if f == "ORB_G5":
        if form_id == "raw":         return f"orb_{sess}_size"
        if form_id == "vol_norm":    return f"orb_{sess}_size__div__atr_20_pct"
        if form_id == "rank_252d":   return f"orb_{sess}_size__rank252"
    if f == "ATR_P50":
        if form_id == "raw":         return "atr_20_pct"
        if form_id == "vol_norm":    return "atr_20_pct"  # already vol-normalized
        if form_id == "rank_252d":   return "atr_20_pct__rank252"
    if f == "COST_LT12":
        if form_id == "raw":         return "_cost_ratio"  # derived in derive_features()
        if form_id == "vol_norm":    return "_cost_ratio__div__atr_20_pct"
        if form_id == "rank_252d":   return "_cost_ratio__rank252"
    return None


def derive_features(df: pd.DataFrame, lane: dict) -> pd.DataFrame:
    """Add derived columns (vol-norm, 252d rank, cost ratio) to the lane tape."""
    sess = lane["orb_label"]
    out = df.copy()
    # Tier-A vol-normalized + rank forms (ORB_G5 substrate)
    if lane["deployed_filter"] == "ORB_G5":
        size_col = f"orb_{sess}_size"
        if size_col in out.columns:
            out[f"{size_col}__div__atr_20_pct"] = out[size_col] / out["atr_20_pct"].replace(0, np.nan)
            out[f"{size_col}__rank252"] = out[size_col].rolling(252, min_periods=63).rank(pct=True)
    # Tier-A vol-normalized + rank forms (ATR_P50 substrate is already atr_20_pct, no vol-norm needed)
    if lane["deployed_filter"] == "ATR_P50":
        out["atr_20_pct__rank252"] = out["atr_20_pct"].rolling(252, min_periods=63).rank(pct=True)
    # Tier-A COST_LT12 substrate — cost_ratio_pct per trade.
    # Canonical formula from trading_app/config.py:593 CostRatioFilter.matches_row:
    #   raw_risk = orb_size * cost_spec.point_value
    #   cost_ratio_pct = 100 * total_friction / (raw_risk + total_friction)
    if lane["deployed_filter"] == "COST_LT12":
        from pipeline.cost_model import get_cost_spec
        cs = get_cost_spec(lane["instrument"])
        size_col = f"orb_{sess}_size"
        raw_risk = out[size_col] * cs.point_value
        out["_cost_ratio"] = 100.0 * cs.total_friction / (raw_risk + cs.total_friction)
        out["_cost_ratio__div__atr_20_pct"] = out["_cost_ratio"] / out["atr_20_pct"].replace(0, np.nan)
        out["_cost_ratio__rank252"] = out["_cost_ratio"].rolling(252, min_periods=63).rank(pct=True)
    return out


def run_diagnostic() -> dict:
    """Top-level. Returns the structured result dict."""
    prereg = load_prereg()
    holdout = prereg["metadata"]["holdout_date"]
    rho_min = prereg["metadata"]["rho_min"]
    n_min = prereg["metadata"]["power_floor_N"]
    null_max = prereg["metadata"]["null_coverage_max"]
    sd_max = prereg["metadata"]["stability_sd_variation_max"]
    seed = prereg["metadata"]["bootstrap_seed"]
    B = prereg["metadata"]["bootstrap_B"]
    weights = (0.6, 0.8, 1.0, 1.2, 1.4)

    cells: list[dict] = []
    pvals: list[float] = []

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    for lane in prereg["lanes"]:
        df = load_lane_tape(con, lane, holdout=holdout)
        df = derive_features(df, lane)

        # --- Tier-A: 3 forms of the deployed-filter substrate ---
        for form in prereg["features"]["tier_a"]:
            col = resolve_substrate_column(lane, form["form_id"])
            pred_sign = prereg["ex_ante_directions_tier_a"][f"{lane['deployed_filter']}_{form['form_id']}"]
            cells.append(_compute_cell(df, lane, col, pred_sign, "tier_a", form["form_id"],
                                        rho_min, n_min, null_max, sd_max, seed, B, weights))
        # --- Tier-B: 5 orthogonal features ---
        for feat in prereg["features"]["tier_b"]:
            col = feat.get("column") or feat["column_template"].format(ORB_LABEL=lane["orb_label"])
            pred_sign = feat["ex_ante_direction"]
            cells.append(_compute_cell(df, lane, col, pred_sign, "tier_b", feat["feature_id"],
                                        rho_min, n_min, null_max, sd_max, seed, B, weights))

    # BH-FDR over the full K=48 family
    pvals = [c["rho_p"] if c.get("rho_p") is not None else 1.0 for c in cells]
    bh = apply_bh_fdr(pvals, q=prereg["metadata"]["bh_fdr_q"])
    for c, p in zip(cells, bh):
        c["bh_fdr_pass"] = bool(p)
        c["status"] = classify_cell(c)
        c["stage2_eligible"] = stage2_eligible_flag(c)

    # Lane-level + global verdict
    by_lane: dict[str, list[dict]] = {}
    for c in cells:
        by_lane.setdefault(c["lane_id"], []).append(c)
    lanes_with_substrate = [lid for lid, lc in by_lane.items() if any(x["status"] == "PASS" for x in lc)]
    n_passing_lanes = len(lanes_with_substrate)
    if n_passing_lanes >= 3:
        verdict = "SUBSTRATE_CONFIRMED"
    elif n_passing_lanes in (1, 2):
        verdict = "SUBSTRATE_WEAK"
    else:
        verdict = "THESIS_KILLED"
    # Tier-level inconclusive guard
    tier_a_cells = [c for c in cells if c["tier"] == "tier_a"]
    tier_b_cells = [c for c in cells if c["tier"] == "tier_b"]
    inv_a = sum(1 for c in tier_a_cells if c["status"] in ("INVALID",) or c.get("power_status") == "UNDERPOWERED")
    inv_b = sum(1 for c in tier_b_cells if c["status"] in ("INVALID",) or c.get("power_status") == "UNDERPOWERED")
    if inv_a / max(1, len(tier_a_cells)) >= 0.5 or inv_b / max(1, len(tier_b_cells)) >= 0.5:
        verdict = "INCONCLUSIVE"

    db_sha = con.execute("SELECT md5(string_agg(table_name, ',')) FROM information_schema.tables").fetchone()[0]
    git_sha = _git_head_sha()

    result = {
        "design_doc": "docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md",
        "prereg": str(PREREG_PATH),
        "git_head_sha": git_sha,
        "db_sha_proxy": db_sha,
        "bootstrap_seed": seed,
        "bootstrap_B": B,
        "K": len(cells),
        "verdict": verdict,
        "lanes_with_substrate": lanes_with_substrate,
        "cells": cells,
    }
    return result


_OVERNIGHT_VALID_SESSIONS = frozenset({
    "LONDON_METALS", "EUROPE_FLOW", "BRISBANE_1955", "US_DATA_830",
    "NYSE_OPEN", "US_DATA_1000", "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
})

# RULE 1.1 hard-banned look-ahead features per .claude/rules/backtesting-methodology.md.
# These are post-trade or full-session-scan columns that are NEVER decision-time-knowable.
_BANNED_LOOKAHEAD_COLUMNS = frozenset({
    "double_break",   # RULE 1.1: scans full session AFTER entry
    "mae_r",          # intra-trade retrospective
    "mfe_r",          # intra-trade retrospective
    "outcome",        # post-trade
    "pnl_r",          # post-trade — only valid as response variable
})

# RULE 6.3 E2-banned break-bar feature suffixes — break_ts/break_bar_* are
# post-entry for ~41% of E2 trades because E2 fires on first range-touch
# (wick), but daily_features defines "break bar" by close-outside-ORB.
# Canonical authority: trading_app/config.py:3540-3568 E2_EXCLUDED_FILTER_*.
_E2_LOOKAHEAD_BREAK_BAR_SUFFIXES = (
    "_break_ts",
    "_break_delay_min",
    "_break_bar_volume",
    "_break_bar_continues",
    "_break_dir",
)


def feature_temporal_validity(lane: dict, col: str) -> tuple[str, str]:
    """Return ('OK', '') or ('INVALID', reason).

    Enforces .claude/rules/backtesting-methodology.md:
    - RULE 1.1: hard-banned look-ahead features (double_break, mae_r, mfe_r,
      outcome, pnl_r) are NEVER decision-time-knowable and are rejected at
      every session.
    - RULE 1.2: overnight_* features are valid only for ORB sessions starting
      >=17:00 Brisbane (TOKYO_OPEN/SINGAPORE_OPEN/early sessions reject).
    - RULE 6.3: E2-look-ahead break-bar features (*_break_ts, *_break_delay_min,
      *_break_bar_*, *_break_dir-as-predictor) are rejected for entry_model=E2
      because E2 fires on first range-touch (wick) but daily_features defines
      break by close-outside-ORB — post-entry for ~41% of E2 trades. Canonical
      authority: trading_app/config.py:3540-3568.

    This is the RULE 13 pressure-test gate: any future pre-reg attempting to
    introduce a banned feature MUST be flagged here, fail-closed, before
    entering the cell-level statistics.
    """
    if col is None:
        return ("OK", "")
    if col in _BANNED_LOOKAHEAD_COLUMNS:
        return ("INVALID", f"RULE 1.1 hard-banned look-ahead column: {col}")
    if col.startswith("overnight_") and lane["orb_label"] not in _OVERNIGHT_VALID_SESSIONS:
        return ("INVALID", f"overnight_* lookahead on session {lane['orb_label']} (<17:00 Brisbane)")
    if lane.get("entry_model") == "E2":
        for sfx in _E2_LOOKAHEAD_BREAK_BAR_SUFFIXES:
            if col.endswith(sfx):
                return ("INVALID", f"RULE 6.3 E2 break-bar lookahead: column ends with {sfx}")
    return ("OK", "")


def _compute_cell(df, lane, col, pred_sign, tier, form_id,
                  rho_min, n_min, null_max, sd_max, seed, B, weights) -> dict:
    cell = {
        "lane_id": lane["id"], "tier": tier, "form_or_feature": form_id, "column": col,
        "predicted_sign": pred_sign,
    }
    # Pre-NULL temporal-validity gate (RULE 1.2 lookahead enforcement)
    validity_status, validity_reason = feature_temporal_validity(lane, col)
    if validity_status == "INVALID":
        cell.update({"null_status": "INVALID", "power_status": "n/a", "rho_p": 1.0,
                     "rho": 0.0, "monotonic": False, "q5_minus_q1": 0.0,
                     "sized_flat_delta_lo": 0.0, "sized_flat_delta_hi": 0.0,
                     "split_half_rho_match": False, "split_half_delta_match": False,
                     "stability_status": "n/a", "realized_sign": "?", "n": 0,
                     "drop_frac": 0.0, "note": f"lookahead_invalid: {validity_reason}"})
        return cell

    if col is None or col not in df.columns:
        cell.update({"null_status": "INVALID", "power_status": "n/a", "rho_p": 1.0,
                     "rho": 0.0, "monotonic": False, "q5_minus_q1": 0.0,
                     "sized_flat_delta_lo": 0.0, "sized_flat_delta_hi": 0.0,
                     "split_half_rho_match": False, "split_half_delta_match": False,
                     "stability_status": "n/a", "realized_sign": "?", "n": 0,
                     "drop_frac": 1.0, "note": "column missing"})
        return cell

    # NULL coverage
    null_status, drop_frac = null_coverage_mark(df[col], threshold=null_max)
    cell["null_status"] = null_status
    cell["drop_frac"] = drop_frac
    df2 = df.dropna(subset=[col, "pnl_r"]).copy()
    cell["n"] = len(df2)
    cell["power_status"] = power_floor_mark(len(df2), min_n=n_min)
    if cell["null_status"] == "INVALID" or cell["power_status"] == "UNDERPOWERED":
        cell["rho_p"] = 1.0
        cell.update({"rho": 0.0, "monotonic": False, "q5_minus_q1": 0.0,
                     "sized_flat_delta_lo": 0.0, "sized_flat_delta_hi": 0.0,
                     "split_half_rho_match": False, "split_half_delta_match": False,
                     "stability_status": "n/a", "realized_sign": "?"})
        return cell

    # Spearman rho
    rho, p = stats.spearmanr(df2[col], df2["pnl_r"])
    cell["rho"] = float(rho)
    cell["rho_p"] = float(p)
    cell["realized_sign"] = "+" if rho >= 0 else "-"
    # Quintile lift
    ql = compute_quintile_lift(df2, feature_col=col, outcome_col="pnl_r")
    cell.update({"q1_mean_r": ql["q1_mean_r"], "q5_mean_r": ql["q5_mean_r"],
                 "q5_minus_q1": ql["q5_minus_q1"], "monotonic": ql["monotonic"]})
    # Sized vs flat
    ci = bootstrap_sized_vs_flat_ci(df2, feature_col=col, outcome_col="pnl_r",
                                    weights=weights, predicted_sign=pred_sign, B=B, seed=seed)
    cell.update({"sized_flat_delta_obs": ci["observed"],
                 "sized_flat_delta_lo": ci["lo"], "sized_flat_delta_hi": ci["hi"]})
    # Split-half
    cell["split_half_rho_match"] = sign_match_split_half(df2, feature_col=col, outcome_col="pnl_r")
    df_sorted = df2.sort_values("trading_day").reset_index(drop=True)
    median_day = df_sorted["trading_day"].iloc[len(df_sorted) // 2]
    h1 = df_sorted[df_sorted["trading_day"] < median_day]
    h2 = df_sorted[df_sorted["trading_day"] >= median_day]
    if len(h1) > 30 and len(h2) > 30:
        ci1 = bootstrap_sized_vs_flat_ci(h1, col, "pnl_r", weights, pred_sign, B=B, seed=seed)
        ci2 = bootstrap_sized_vs_flat_ci(h2, col, "pnl_r", weights, pred_sign, B=B, seed=seed)
        cell["split_half_delta_match"] = bool(np.sign(ci1["observed"]) == np.sign(ci2["observed"])
                                              and ci1["observed"] != 0 and ci2["observed"] != 0)
    else:
        cell["split_half_delta_match"] = False
    # Forecast stability
    cell["stability_status"] = check_forecast_stability(df2, feature_col=col, window=252, max_rel_var=sd_max)
    return cell


def _git_head_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def render_markdown(result: dict) -> str:
    lines: list[str] = []
    lines.append("# Sizing-Substrate Diagnostic — Result")
    lines.append("")
    lines.append(f"- Design doc: `{result['design_doc']}`")
    lines.append(f"- Pre-reg: `{result['prereg']}`")
    lines.append(f"- Git HEAD: `{result['git_head_sha']}`")
    lines.append(f"- DB schema fingerprint: `{result['db_sha_proxy']}`")
    lines.append(f"- Bootstrap seed: {result['bootstrap_seed']}; B={result['bootstrap_B']}")
    lines.append(f"- K = {result['K']}")
    lines.append(f"- **VERDICT: {result['verdict']}**")
    lines.append(f"- Lanes with substrate: {result['lanes_with_substrate']}")
    lines.append("")
    lines.append("## Scope / Question")
    lines.append("")
    lines.append("Stage-1 falsifier of the thesis that deployed binary filters in the live")
    lines.append("`topstep_50k_mnq_auto` 6-lane portfolio have continuous substrate justifying")
    lines.append("a Carver-style forecast→sizing layer. K=48 cells = 6 deployed lanes ×")
    lines.append("(3 Tier-A substrate forms + 5 Tier-B orthogonal continuous features).")
    lines.append("Pass criterion (per cell, ALL gates): |ρ|≥0.10, monotonic Q1→Q5, |Q5−Q1|≥0.20R,")
    lines.append("sized-vs-flat 95% CI > 0, BH-FDR survives at q=0.05, split-half sign match,")
    lines.append("ex-ante prediction sign match. Lane has substrate iff ≥1 cell passes.")
    lines.append("Substrate confirmed globally iff ≥3 lanes pass.")
    lines.append("")
    lines.append("## Verdict / Decision")
    lines.append("")
    lines.append(f"**{result['verdict']}.** {len(result['lanes_with_substrate'])} of 6 lanes have substrate.")
    if result["verdict"] == "SUBSTRATE_CONFIRMED":
        lines.append("Decision: pre-register Stage-2 sizing model on passing cells.")
    elif result["verdict"] == "SUBSTRATE_WEAK":
        lines.append("Decision: park sizing thesis. Pre-reg requires ≥3 lanes for global confirmation.")
        lines.append("Possible single-lane Stage-2 study only under a fresh pre-reg with")
        lines.append("explicit mechanism citation per lane.")
    elif result["verdict"] == "THESIS_KILLED":
        lines.append("Decision: NO-GO entry to docs/STRATEGY_BLUEPRINT.md §5. Reopen requires")
        lines.append("new mechanism citation.")
    else:
        lines.append("Decision: diagnose tier failure (≥50% INVALID/UNDERPOWERED). Do NOT")
        lines.append("re-run blindly — fix substrate-feature definition and re-pre-register.")
    lines.append("")
    lines.append("## Reproduction / Outputs")
    lines.append("")
    lines.append("- Script: `research/audit_sizing_substrate_diagnostic.py` at git HEAD above")
    lines.append("- Pre-reg: `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml`")
    lines.append(f"- Bootstrap: numpy default_rng seed=42, B={result['bootstrap_B']}")
    lines.append(f"- DB: `pipeline.paths.GOLD_DB_PATH`, schema fingerprint `{result['db_sha_proxy']}`")
    lines.append("- Re-run: `python research/audit_sizing_substrate_diagnostic.py` (read-only;")
    lines.append("  raises RuntimeError on any trading_day ≥ 2026-01-01)")
    lines.append("- Tests: `pytest tests/test_research/test_audit_sizing_substrate_diagnostic.py`")
    lines.append("- Output JSON twin: `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json`")
    lines.append("")
    lines.append("## Caveats / Limitations / Disconfirming Considerations")
    lines.append("")
    lines.append("- **Selection-bias compounding (per spec §6):** the 6 lanes are themselves")
    lines.append("  survivors of an earlier, larger trial space. K=48 BH-FDR controls only the")
    lines.append("  new diagnostic family; it does NOT deflate prior lane-discovery multiplicity.")
    lines.append("  Stage-2 deployment must apply DSR with cumulative trial count.")
    lines.append("- **Lookahead-validity gate enforcement:** cells using `overnight_*` features")
    lines.append("  on TOKYO_OPEN/SINGAPORE_OPEN (sessions starting <17:00 Brisbane) are marked")
    lines.append("  INVALID per `.claude/rules/backtesting-methodology.md` RULE 1.2. The")
    lines.append("  unguarded run had additional apparent passes that disappeared under the gate.")
    n_invalid_la = sum(1 for c in result["cells"]
                       if c.get("status") == "INVALID"
                       and "lookahead" in str(c.get("note", "")))
    lines.append(f"- **Effective tested K = {result['K'] - n_invalid_la}** ({n_invalid_la} cells gated INVALID by lookahead;")
    lines.append("  pre-reg K=48 unchanged but BH-FDR denominator includes the gated cells, making")
    lines.append("  the family-wise correction conservative — survivors are if anything stronger")
    lines.append("  evidence than the q value suggests.)")
    lines.append("- **Stage-2 eligibility:** PASS cells are stage-2 eligible only if their")
    lines.append("  `stability_status == STABLE`. UNSTABLE PASS cells require a fresh Stage-2")
    lines.append("  pre-reg with explicit forecast-normalization per Carver Ch.7 fn 78. Check")
    lines.append("  the `stage2_eligible` field in the JSON twin.")
    lines.append("- **Linear-rank weights {0.6, 0.8, 1.0, 1.2, 1.4} are diagnostic-only,**")
    lines.append("  NOT Carver's actual recipe (Ch. 7 forecast scalar to abs-mean=10, cap=±20).")
    lines.append("  Stage-2 must implement the canonical Carver scaling, not the rank proxy.")
    lines.append("- **Per-cell bootstrap underestimates serial dependence.** Acceptable for Stage 1;")
    lines.append("  Stage 2 must consider block bootstrap if substrate were confirmed.")
    lines.append("- **AFML Ch. 19 sigmoid bet-sizer is NOT in `resources/`.** Sigmoid functional")
    lines.append("  form deferred. Stage 2 (if any) defaults to Carver-only sizing.")
    lines.append("- **Single-pass discipline.** Re-running with different feature lists, weight")
    lines.append("  schemas, thresholds, or expanded K is a NEW pre-reg, not a re-run.")
    lines.append("")
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| lane | tier | feature/form | n | rho | p | bh-fdr | Q5-Q1 R | mono | delta CI | split | stable | pred | real | status |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in result["cells"]:
        ci_str = f"[{c.get('sized_flat_delta_lo', 0):+.3f}, {c.get('sized_flat_delta_hi', 0):+.3f}]"
        split = "Y" if c.get("split_half_rho_match") and c.get("split_half_delta_match") else "N"
        lines.append(f"| {c['lane_id']} | {c['tier']} | {c.get('form_or_feature')} | "
                     f"{c.get('n', 0)} | {c.get('rho', 0):+.3f} | {c.get('rho_p', 1):.4f} | "
                     f"{'Y' if c.get('bh_fdr_pass') else 'N'} | {c.get('q5_minus_q1', 0):+.3f} | "
                     f"{'Y' if c.get('monotonic') else 'N'} | {ci_str} | {split} | "
                     f"{c.get('stability_status', '?')[:1]} | {c.get('predicted_sign', '?')} | "
                     f"{c.get('realized_sign', '?')} | **{c.get('status', '?')}** |")
    return "\n".join(lines) + "\n"


def main() -> int:
    result = run_diagnostic()
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    RESULT_MD.write_text(render_markdown(result), encoding="utf-8")
    print(f"VERDICT: {result['verdict']}")
    print(f"Lanes with substrate: {result['lanes_with_substrate']}")
    print(f"Wrote {RESULT_MD}")
    print(f"Wrote {RESULT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
