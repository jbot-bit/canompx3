#!/usr/bin/env python3
"""Phase 2.9 — Comprehensive 7-year x 38-lane multi-year stratification.

Supersedes the framing of Phase 2.8 v1 per the reframe addendum
(`docs/audit/results/2026-04-19-phase-2-8-reframe-addendum.md` sec 7).

v1 used only 3 high-vol years (2020/2022/2024), a bare 0.03 delta threshold,
and an asymmetric DRAG-only label with no per-cell significance testing.
v2 corrects all four:
  - 7 years (2019-2025, all Mode A)
  - per-cell subset-t + two-sided t-distribution p
  - BH-FDR at three K framings (K_global=266, K_session variable, K_year=38)
  - symmetric DRAG / BOOST labels conditioned on BH survival
  - session x year heat map
  - per-lane fragility disclosure for Phase 2.5 Tier-1 / Phase 2.7 GOLD

Pre-reg: docs/audit/hypotheses/2026-04-19-phase-2-9-comprehensive-multi-year-stratification.yaml
Stage: docs/runtime/stages/phase-2-9-comprehensive-multi-year-stratification.md

Canonical delegations (per research-truth-protocol):
  - compute_mode_a, load_active_setups, direction_from_execution_spec,
    subset_t, _window_stats  -> research.phase_2_8_multi_year_regime_stratification
    (which itself delegates to research.mode_a_revalidation_active_setups and
    research.filter_utils.filter_signal)
  - HOLDOUT_SACRED_FROM -> trading_app.holdout_policy
  - GOLD_DB_PATH -> pipeline.paths
  - SESSION_CATALOG -> pipeline.dst

Outputs (CSV + result doc written separately):
  - research/output/phase_2_9_main.csv             (38 * 7 = 266 lane-year rows)
  - research/output/phase_2_9_session_year_heat.csv (7 sessions x 7 years grid)
  - research/output/phase_2_9_gold_fragility.csv   (Tier-1 + GOLD lanes, ex-each-year)

Usage (from repo root):
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db \\
      uv run --frozen python research/phase_2_9_comprehensive_multi_year_stratification.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dst import SESSION_CATALOG  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.mode_a_revalidation_active_setups import (  # noqa: E402
    compute_mode_a,
    direction_from_execution_spec,
    load_active_setups,
)
from research.phase_2_8_multi_year_regime_stratification import (  # noqa: E402
    _window_stats,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"

# All stratified years within Mode A IS (trading_day < 2026-01-01).
YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)

# Per-cell significance / labelling thresholds.
BH_Q: float = 0.10
DELTA_LABEL_THRESHOLD: float = 0.03   # symmetric DRAG/BOOST
MIN_N_YEAR: int = 30                   # below this, cell is UNEVALUABLE

# Phase 2.7 "truly deploy-safe GOLD" lanes per handoff commit f3b3b72b.
GOLD_LANES: frozenset[str] = frozenset({
    "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30",
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
})

# MinBTL budget (Bailey et al 2013 Eq in `docs/institutional/literature/`).
MIN_N_FOR_MINBTL_PASS: int = 50


def _two_sided_p_from_t(t: float | None, n: int) -> float | None:
    """Two-sided p-value from a one-sample t-statistic on n observations."""
    if t is None or n < 2:
        return None
    if math.isnan(t) or math.isinf(t):
        return None
    df = n - 1
    return float(2.0 * stats.t.sf(abs(t), df))


def bh_fdr(pvalues: list[float | None], q: float) -> list[bool]:
    """Benjamini-Hochberg step-up at level q. None p-values never survive."""
    n = len(pvalues)
    survivors = [False] * n
    indexed = [(p, i) for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    indexed.sort(key=lambda x: x[0])
    k = 0
    for rank, (p, _) in enumerate(indexed, start=1):
        crit = (rank / n) * q
        if p <= crit:
            k = rank
    if k > 0:
        for rank, (_, orig) in enumerate(indexed, start=1):
            if rank <= k:
                survivors[orig] = True
    return survivors


def build_main_rows(con: duckdb.DuckDBPyConnection, specs: list[dict[str, Any]]) -> pd.DataFrame:
    """For each (lane, year): compute only-year + ex-year stats, subset_t, p, delta."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        sid = spec["strategy_id"]
        direction = direction_from_execution_spec(spec.get("execution_spec"))
        full = _window_stats(con, spec)
        full_n = full["n_on"]
        full_expr = full["expr"]
        full_t = full["t"]
        if full_expr is None:
            for y in YEARS:
                rows.append({
                    "strategy_id": sid,
                    "instrument": spec["instrument"],
                    "session": spec["orb_label"],
                    "orb_minutes": spec["orb_minutes"],
                    "rr_target": spec["rr_target"],
                    "filter_type": spec.get("filter_type"),
                    "direction": direction,
                    "year": y,
                    "n_year": 0,
                    "year_expr": None,
                    "ex_year_n": 0,
                    "ex_year_expr": None,
                    "ex_year_t": None,
                    "delta": None,
                    "year_t": None,
                    "year_p": None,
                    "full_n": full_n,
                    "full_expr": None,
                    "full_t": None,
                    "label_raw": "EMPTY_FULL_WINDOW",
                })
            continue
        for y in YEARS:
            only = _window_stats(con, spec, only_year=y)
            ex = _window_stats(con, spec, exclude_year=y)
            n_y = only["n_on"]
            year_expr = only["expr"]
            year_t = only["t"]
            year_p = _two_sided_p_from_t(year_t, n_y)
            delta = (
                (year_expr - ex["expr"]) if (year_expr is not None and ex["expr"] is not None) else None
            )
            if n_y < MIN_N_YEAR or delta is None:
                label_raw = "UNEVALUABLE"
            elif delta > DELTA_LABEL_THRESHOLD:
                label_raw = "BOOST_candidate"
            elif delta < -DELTA_LABEL_THRESHOLD:
                label_raw = "DRAG_candidate"
            else:
                label_raw = "NEUTRAL"
            rows.append({
                "strategy_id": sid,
                "instrument": spec["instrument"],
                "session": spec["orb_label"],
                "orb_minutes": spec["orb_minutes"],
                "rr_target": spec["rr_target"],
                "filter_type": spec.get("filter_type"),
                "direction": direction,
                "year": y,
                "n_year": n_y,
                "year_expr": year_expr,
                "ex_year_n": ex["n_on"],
                "ex_year_expr": ex["expr"],
                "ex_year_t": ex["t"],
                "delta": delta,
                "year_t": year_t,
                "year_p": year_p,
                "full_n": full_n,
                "full_expr": full_expr,
                "full_t": full_t,
                "label_raw": label_raw,
            })
    return pd.DataFrame(rows)


def add_bh_flags(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Add K_global, K_session, K_year BH survival flags in-place and return."""
    df = df.copy()
    # K_global over whole 266-cell family
    p_global = df["year_p"].tolist()
    df["bh_global"] = bh_fdr(p_global, q)
    # K_session: per session across years & lanes-in-session
    df["bh_session"] = False
    for _, sub in df.groupby("session"):
        sv = bh_fdr(sub["year_p"].tolist(), q)
        df.loc[sub.index, "bh_session"] = sv
    # K_year: per year across all 38 lanes
    df["bh_year"] = False
    for _, sub in df.groupby("year"):
        sv = bh_fdr(sub["year_p"].tolist(), q)
        df.loc[sub.index, "bh_year"] = sv
    return df


def assign_v2_pattern(row: pd.Series) -> str:
    """Symmetric DRAG/BOOST, BH-conditioned.

    Any survivor of bh_session OR bh_year counts. Bare label_raw without BH
    support is downgraded to NEUTRAL_LABELED. UNEVALUABLE preserved.
    """
    if row["label_raw"] == "UNEVALUABLE":
        return "UNEVALUABLE"
    if row["label_raw"] == "EMPTY_FULL_WINDOW":
        return "EMPTY_FULL_WINDOW"
    if row["label_raw"] == "NEUTRAL":
        return "NEUTRAL"
    survived = bool(row["bh_session"]) or bool(row["bh_year"])
    if row["label_raw"] == "DRAG_candidate":
        return "DRAG_confirmed" if survived else "DRAG_unconfirmed"
    if row["label_raw"] == "BOOST_candidate":
        return "BOOST_confirmed" if survived else "BOOST_unconfirmed"
    return row["label_raw"]


def build_heat_map(df_main: pd.DataFrame) -> pd.DataFrame:
    """Session x Year aggregate grid. Uses n_on-weighted mean ExpR across lanes."""
    eligible = df_main[df_main["n_year"].fillna(0) >= 1].copy()
    rows: list[dict[str, Any]] = []
    for key, sub in eligible.groupby(["session", "year"]):
        if isinstance(key, tuple):
            sess = str(key[0])
            yr = int(float(key[1]))  # type: ignore[arg-type]
        else:
            sess = str(key)
            yr = 0
        n_lanes = int(sub["strategy_id"].nunique())
        weights = sub["n_year"].astype(float).to_numpy()
        vals_expr = sub["year_expr"].astype(float).to_numpy()
        vals_delta = sub["delta"].astype(float).to_numpy()
        wsum = float(weights.sum())
        w_expr = (
            float((vals_expr * weights).sum() / wsum)
            if wsum > 0 and not np.isnan(vals_expr).all()
            else None
        )
        w_delta = (
            float((vals_delta * weights).sum() / wsum)
            if wsum > 0 and not np.isnan(vals_delta).all()
            else None
        )
        survivors = int(sub["bh_session"].astype(int).sum())
        rows.append({
            "session": sess,
            "year": yr,
            "n_lanes": n_lanes,
            "total_n_trades": int(wsum),
            "weighted_mean_year_expr": w_expr,
            "weighted_mean_delta": w_delta,
            "bh_session_survivor_count": survivors,
        })
    return pd.DataFrame(rows).sort_values(["session", "year"]).reset_index(drop=True)


def build_gold_fragility(df_main: pd.DataFrame, tier1_t_floor: float = 3.00) -> pd.DataFrame:
    """Per-lane fragility: compare full_t vs min ex_year_t across the 7 years.

    A lane is FRAGILE if full_t >= 3.0 (Tier-1 per Phase 2.5) OR lane in GOLD
    AND the worst ex_year_t drops below 1.96. Reports the driver year too.
    """
    lanes = df_main[
        (df_main["full_t"].fillna(0) >= tier1_t_floor)
        | (df_main["strategy_id"].isin(GOLD_LANES))
    ]
    rows: list[dict[str, Any]] = []
    for sid, sub in lanes.groupby("strategy_id"):
        full_t = float(sub["full_t"].iloc[0]) if not pd.isna(sub["full_t"].iloc[0]) else None
        # Use ex_year_t: how strong is the lane EXCLUDING this year?
        ex_t = sub[["year", "ex_year_t"]].dropna()
        if ex_t.empty:
            continue
        worst_idx = ex_t["ex_year_t"].abs().idxmin()
        worst_year = int(float(ex_t.loc[worst_idx, "year"]))  # type: ignore[arg-type]
        worst_ex_t = float(ex_t.loc[worst_idx, "ex_year_t"])  # type: ignore[arg-type]
        fragile = full_t is not None and full_t >= tier1_t_floor and abs(worst_ex_t) < 1.96
        rows.append({
            "strategy_id": sid,
            "in_gold_pool": bool(sid in GOLD_LANES),
            "full_t": full_t,
            "worst_ex_year": worst_year,
            "worst_ex_year_t": worst_ex_t,
            "t_drop": (full_t - worst_ex_t) if full_t is not None else None,
            "fragility_flag": "FRAGILE" if fragile else "STABLE",
        })
    if not rows:
        return pd.DataFrame(
            columns=["strategy_id", "in_gold_pool", "full_t", "worst_ex_year",
                     "worst_ex_year_t", "t_drop", "fragility_flag"]
        )
    return pd.DataFrame(rows).sort_values("full_t", ascending=False).reset_index(drop=True)


def minbtl_check(k: int, max_n: int) -> tuple[bool, float]:
    """Bailey 2013 MinBTL = 2 ln(K) / max_n^2. Returns (pass, computed)."""
    if max_n <= 0:
        return False, float("inf")
    val = 2.0 * math.log(k) / (max_n ** 2)
    return val < 1.0, val


def main() -> int:
    print("Phase 2.9 — comprehensive multi-year stratification")
    print(f"  years         : {YEARS}")
    print(f"  HOLDOUT_SACRED_FROM: {HOLDOUT_SACRED_FROM}")
    print(f"  GOLD_DB_PATH  : {GOLD_DB_PATH}")
    print(f"  BH q          : {BH_Q}")
    print()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        specs = load_active_setups(con)
        expected_lanes = len(specs)
        print(f"Loaded {expected_lanes} active validated_setups.")

        # Sanity: session catalog overlap
        missing = [s for s in specs if s["orb_label"] not in SESSION_CATALOG]
        if missing:
            print(f"ERROR: {len(missing)} lanes have orb_label not in SESSION_CATALOG", file=sys.stderr)
            return 2

        # MinBTL pre-check
        k_global = expected_lanes * len(YEARS)
        est_max_n = max((s.get("sample_size") or 0) for s in specs) or 1
        ok, val = minbtl_check(k_global, est_max_n)
        print(f"MinBTL: K={k_global}, est max_N={est_max_n}, computed={val:.6f}, pass={ok}")

        # Canonical sanity: compute_mode_a for a single spec (first) must agree
        # with _window_stats(full-window). This is the same check Phase 2.8 runs.
        first = specs[0]
        try:
            _, expr_canon, *_ = compute_mode_a(con, first)
            full0 = _window_stats(con, first)
            if expr_canon is not None and full0["expr"] is not None:
                div = abs(expr_canon - full0["expr"])
                assert div < 1e-4, f"canonical divergence on {first['strategy_id']}: {div}"
                print(f"Canonical-sanity PASS on {first['strategy_id']} (div={div:.2e})")
        except AssertionError as e:
            print(f"FATAL canonical divergence: {e}", file=sys.stderr)
            return 3

        # Main sweep
        print()
        print(f"Computing {k_global} cells...")
        df_main = build_main_rows(con, specs)
        print(f"Main rows: {len(df_main)} (expected {k_global})")
        if len(df_main) != k_global:
            print("ERROR: unexpected row count", file=sys.stderr)
            return 4

        # BH at 3 framings
        df_main = add_bh_flags(df_main, BH_Q)
        df_main["pattern_v2"] = df_main.apply(assign_v2_pattern, axis=1)

        # Per-framing survivor counts
        n_global = int(df_main["bh_global"].sum())
        n_session = int(df_main["bh_session"].sum())
        n_year = int(df_main["bh_year"].sum())
        print(f"BH survivors: K_global={n_global}, K_session={n_session}, K_year={n_year}")

        # Heat map
        df_heat = build_heat_map(df_main)
        # Fragility
        df_frag = build_gold_fragility(df_main)

        # Write outputs
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        main_path = OUTPUT_DIR / "phase_2_9_main.csv"
        heat_path = OUTPUT_DIR / "phase_2_9_session_year_heat.csv"
        frag_path = OUTPUT_DIR / "phase_2_9_gold_fragility.csv"
        df_main.to_csv(main_path, index=False)
        df_heat.to_csv(heat_path, index=False)
        df_frag.to_csv(frag_path, index=False)
        print()
        print(f"Wrote {main_path.relative_to(PROJECT_ROOT)} ({len(df_main)} rows)")
        print(f"Wrote {heat_path.relative_to(PROJECT_ROOT)} ({len(df_heat)} rows)")
        print(f"Wrote {frag_path.relative_to(PROJECT_ROOT)} ({len(df_frag)} rows)")

        # Compact verdict print
        print()
        print("=== H1 (session-level regime asymmetry) ===")
        session_survivors = (
            df_main[df_main["bh_session"]]
            .groupby(["session", "year"])["label_raw"]
            .agg(lambda s: s.mode().iloc[0] if len(s) else None)
            .reset_index()
        )
        print(session_survivors.to_string(index=False))
        if len(session_survivors) == 0:
            print("  H1: no BH-survivors at K_session -> REFUTED")

        print()
        print("=== H2 (year-level regime alignment) ===")
        year_survivors = df_main[df_main["bh_year"]].groupby("year").agg(
            n_survivors=("strategy_id", "size"),
            n_drag=("label_raw", lambda s: int((s == "DRAG_candidate").sum())),
            n_boost=("label_raw", lambda s: int((s == "BOOST_candidate").sum())),
        )
        print(year_survivors.to_string())
        if year_survivors.empty:
            print("  H2: no BH-survivors at K_year -> REFUTED")

        print()
        print("=== H3 (GOLD-pool fragility) ===")
        print(df_frag.to_string(index=False))
        n_fragile = int((df_frag["fragility_flag"] == "FRAGILE").sum())
        print(f"  Fragile lanes: {n_fragile}")

        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
