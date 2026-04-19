#!/usr/bin/env python3
"""Phase 2.7 — 2024 regime-break systemic audit across 38 active lanes.

3-layer analysis (literature-grounded):
  Layer 1: per-lane year stratification — full Mode A / ex-2024 / 2024-only
           (N, ExpR, subset-t, WR). Stratification is NOT re-optimization —
           the 2026+ holdout (HOLDOUT_SACRED_FROM) remains sacred.
  Layer 2: 2024 vol-regime characterization — avg atr_20_pct and
           garch_forecast_vol_pct on each lane's universe, 2024 vs rest.
           Grounded in Chan 2008 Ch 7 § volatility regime is most amenable
           (docs/institutional/literature/chan_2008_ch7_regime_switching.md).
  Layer 3: cross-lane pattern — classify each lane into 2024 flag bucket;
           aggregate by filter_type + session to find systemic patterns.

Canonical delegations:
  - load_active_setups, compute_mode_a, direction_from_execution_spec,
    C4_T_WITH_THEORY, C7_MIN_N, C9_ERA_THRESHOLD, C9_MIN_N_PER_ERA from
    research.mode_a_revalidation_active_setups
  - filter_signal via compute_mode_a (canonical ALL_FILTERS)
  - HOLDOUT_SACRED_FROM from trading_app.holdout_policy (SACRED, unchanged)
  - GOLD_DB_PATH from pipeline.paths
  - SESSION_CATALOG whitelist from pipeline.dst

Outputs:
  - research/output/phase_2_7_regime_break_2024_audit.csv — one row per lane
  - research/output/phase_2_7_regime_break_2024_audit_vol_regime.csv — vol stats
  - stdout summary + flag breakdown + filter-class aggregation
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

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
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"

_VALID_SESSIONS: frozenset[str] = frozenset(SESSION_CATALOG.keys())

YEAR_2024_START = date(2024, 1, 1)
YEAR_2024_END = date(2025, 1, 1)  # exclusive upper bound

# Flag thresholds (numeric, no cherry-picking)
FLAG_DRAG_YR_EXPR: float = -0.05       # per pre_registered_criteria § C9
FLAG_DELTA_THRESHOLD: float = 0.03      # 3% R per trade as the "material shift" bar
FLAG_UNEVALUABLE_MIN_N: int = 30        # backtesting-methodology RULE 3.2


def subset_t(expr: float | None, sd: float | None, n: int) -> float:
    if n < 2 or expr is None or sd is None or sd == 0:
        return float("nan")
    return float(expr) / (float(sd) / math.sqrt(n))


def _compute_window_stats(
    con: duckdb.DuckDBPyConnection,
    spec: dict[str, Any],
    *,
    window_sql_clause: str,
    window_params: list[Any],
) -> dict[str, Any]:
    """Generic window-stats helper: applies lane's filter under an arbitrary
    trading_day window clause. Mirrors compute_mode_a SQL structure but with
    a custom window constraint replacing the Mode A cutoff.
    """
    sess = spec["orb_label"]
    if sess not in _VALID_SESSIONS:
        raise ValueError(f"orb_label {sess!r} not in SESSION_CATALOG")
    direction = direction_from_execution_spec(spec.get("execution_spec"))

    sql = f"""
        SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND d.orb_{sess}_break_dir = ?
          AND o.pnl_r IS NOT NULL
          AND {window_sql_clause}
        ORDER BY o.trading_day
    """
    params = [
        spec["instrument"], sess, spec["orb_minutes"],
        spec["entry_model"], spec["confirm_bars"], spec["rr_target"],
        direction,
    ] + window_params
    df = con.execute(sql, params).df()

    n_universe = len(df)
    if n_universe == 0:
        return {"n_universe": 0, "n_on": 0, "expr": None, "sd": None,
                "wr": None, "t": None}

    # Apply lane's filter via canonical filter_signal (requires X_MES_ATR60
    # injection if filter is CrossAssetATRFilter — mirror mode_a_revalidation
    # pattern).
    from trading_app.config import ALL_FILTERS
    from research.filter_utils import filter_signal

    filter_type = spec.get("filter_type")
    if not filter_type or filter_type == "UNFILTERED":
        fire = np.ones(n_universe, dtype=bool)
    else:
        # Cross-asset ATR injection mirror
        filt_obj = ALL_FILTERS.get(filter_type)
        if filt_obj is not None:
            # CrossAssetATRFilter needs cross_atr_{source}_pct column
            from trading_app.config import CrossAssetATRFilter
            if isinstance(filt_obj, CrossAssetATRFilter):
                source = filt_obj.source_instrument
                if source != spec["instrument"]:
                    src_rows = con.execute(
                        "SELECT trading_day, atr_20_pct FROM daily_features "
                        "WHERE symbol = ? AND orb_minutes = 5 "
                        "AND atr_20_pct IS NOT NULL",
                        [source],
                    ).fetchall()
                    src_map = {
                        (td.date() if hasattr(td, "date") else td): float(pct)
                        for td, pct in src_rows
                    }
                    df[f"cross_atr_{source}_pct"] = df["trading_day"].apply(
                        lambda d: src_map.get(d.date() if hasattr(d, "date") else d)
                    )
        try:
            fire = np.asarray(filter_signal(df, filter_type, sess)).astype(bool)
        except Exception as e:  # noqa: BLE001
            return {"n_universe": n_universe, "n_on": 0, "expr": None,
                    "sd": None, "wr": None, "t": None,
                    "error": f"filter_signal: {e}"}

    df_on = df[fire].reset_index(drop=True)
    n_on = len(df_on)
    if n_on == 0:
        return {"n_universe": n_universe, "n_on": 0, "expr": None, "sd": None,
                "wr": None, "t": None}
    pnl = df_on["pnl_r"].astype(float).to_numpy()
    expr = float(pnl.mean())
    sd = float(pnl.std(ddof=1)) if n_on > 1 else None
    wr = float((df_on["outcome"].astype(str) == "win").mean())
    t = subset_t(expr, sd, n_on)

    return {
        "n_universe": n_universe, "n_on": n_on, "expr": expr, "sd": sd,
        "wr": wr, "t": t if not math.isnan(t) else None,
    }


def _vol_regime_stats(
    con: duckdb.DuckDBPyConnection, spec: dict[str, Any]
) -> dict[str, Any]:
    """Layer 2: avg atr_20_pct + garch_forecast_vol_pct on the lane's
    break-day universe — 2024 vs rest-of-Mode-A. Grounded in Chan 2008
    Ch 7 § volatility regime is most tractable classification.
    """
    sess = spec["orb_label"]
    if sess not in _VALID_SESSIONS:
        return {}
    direction = direction_from_execution_spec(spec.get("execution_spec"))

    # Mode A window, 2 cohorts: IN 2024 vs NOT 2024
    sql = f"""
        SELECT
            o.trading_day,
            d.atr_20_pct,
            d.garch_forecast_vol_pct
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND d.orb_{sess}_break_dir = ?
          AND o.pnl_r IS NOT NULL
          AND o.trading_day < ?
        ORDER BY o.trading_day
    """
    df = con.execute(
        sql,
        [
            spec["instrument"], sess, spec["orb_minutes"],
            spec["entry_model"], spec["confirm_bars"], spec["rr_target"],
            direction, HOLDOUT_SACRED_FROM,
        ],
    ).df()
    if len(df) == 0:
        return {}
    df["_td"] = pd.to_datetime(df["trading_day"]).dt.date
    in_2024 = (df["_td"] >= YEAR_2024_START) & (df["_td"] < YEAR_2024_END)
    yr2024 = df[in_2024]
    rest = df[~in_2024]

    def _mean_ignore_nan(s: pd.Series) -> float | None:
        arr = s.dropna().astype(float)
        return float(arr.mean()) if len(arr) else None

    return {
        "atr20_pct_2024": _mean_ignore_nan(yr2024["atr_20_pct"]),
        "atr20_pct_rest": _mean_ignore_nan(rest["atr_20_pct"]),
        "garch_vol_pct_2024": _mean_ignore_nan(yr2024["garch_forecast_vol_pct"]),
        "garch_vol_pct_rest": _mean_ignore_nan(rest["garch_forecast_vol_pct"]),
    }


def classify_2024_flag(
    *,
    expr_full: float | None,
    expr_ex2024: float | None,
    expr_2024: float | None,
    n_2024: int,
) -> str:
    """Hierarchical flag per lane. Numeric thresholds only — no cherry-picking."""
    if n_2024 < FLAG_UNEVALUABLE_MIN_N:
        return "2024_UNEVALUABLE"
    if expr_full is None or expr_ex2024 is None:
        return "2024_UNEVALUABLE"
    delta = expr_ex2024 - expr_full
    # Pure drag: 2024 itself negative AND ex-2024 lifts the lane by ≥0.03
    if expr_2024 is not None and expr_2024 <= FLAG_DRAG_YR_EXPR and delta > FLAG_DELTA_THRESHOLD:
        return "2024_PURE_DRAG"
    # Critical: 2024 CARRIES the lane (dropping it HURTS)
    if delta < -FLAG_DELTA_THRESHOLD:
        return "2024_CRITICAL"
    # Neutral: small delta
    if abs(delta) < FLAG_DELTA_THRESHOLD:
        return "2024_NEUTRAL"
    # Residual: positive lift below threshold OR 2024 mildly bad
    return "2024_MIXED"


def audit_lane(con: duckdb.DuckDBPyConnection, spec: dict[str, Any]) -> dict[str, Any]:
    """Run the 3-layer audit for one lane."""
    try:
        # Layer 1a: full Mode A (canonical compute_mode_a — reuse for validation)
        _, expr_full_canonical, _, _, _, _ = compute_mode_a(con, spec)
    except Exception as e:  # noqa: BLE001
        return {
            "strategy_id": spec["strategy_id"],
            "error": f"compute_mode_a: {e}",
            "flag_2024": "ERROR",
        }

    # Layer 1b: full Mode A via generic window helper (must agree with canonical)
    full = _compute_window_stats(
        con, spec,
        window_sql_clause="o.trading_day < ?",
        window_params=[HOLDOUT_SACRED_FROM],
    )

    # Sanity: compute_mode_a and our window-helper should agree on ExpR to
    # 4 decimals. If not, something's wrong with the mirror.
    if (full.get("expr") is not None and expr_full_canonical is not None
            and abs(full["expr"] - expr_full_canonical) > 1e-4):
        return {
            "strategy_id": spec["strategy_id"],
            "error": f"window-helper/compute_mode_a mismatch: {full['expr']} vs {expr_full_canonical}",
            "flag_2024": "ERROR",
        }

    # Layer 1c: ex-2024 Mode A
    ex2024 = _compute_window_stats(
        con, spec,
        window_sql_clause=(
            "o.trading_day < ? AND "
            "NOT (o.trading_day >= ? AND o.trading_day < ?)"
        ),
        window_params=[HOLDOUT_SACRED_FROM, YEAR_2024_START, YEAR_2024_END],
    )

    # Layer 1d: 2024-only
    yr2024 = _compute_window_stats(
        con, spec,
        window_sql_clause="o.trading_day >= ? AND o.trading_day < ?",
        window_params=[YEAR_2024_START, YEAR_2024_END],
    )

    # Layer 2: 2024 vol-regime characterization
    vol = _vol_regime_stats(con, spec)

    # Layer 3: flag
    flag = classify_2024_flag(
        expr_full=full.get("expr"),
        expr_ex2024=ex2024.get("expr"),
        expr_2024=yr2024.get("expr"),
        n_2024=yr2024.get("n_on", 0),
    )

    return {
        "strategy_id": spec["strategy_id"],
        "instrument": spec["instrument"],
        "session": spec["orb_label"],
        "orb_minutes": spec["orb_minutes"],
        "rr_target": spec["rr_target"],
        "filter_type": spec.get("filter_type"),
        "direction": direction_from_execution_spec(spec.get("execution_spec")),
        # Full Mode A
        "full_n_on": full.get("n_on"),
        "full_expr": full.get("expr"),
        "full_t": full.get("t"),
        "full_wr": full.get("wr"),
        # Ex-2024
        "ex2024_n_on": ex2024.get("n_on"),
        "ex2024_expr": ex2024.get("expr"),
        "ex2024_t": ex2024.get("t"),
        "ex2024_wr": ex2024.get("wr"),
        # 2024-only
        "y2024_n_on": yr2024.get("n_on"),
        "y2024_expr": yr2024.get("expr"),
        "y2024_t": yr2024.get("t"),
        "y2024_wr": yr2024.get("wr"),
        # Deltas
        "delta_expr_ex2024_minus_full": (
            (ex2024.get("expr") - full.get("expr"))
            if (ex2024.get("expr") is not None and full.get("expr") is not None)
            else None
        ),
        "delta_t_ex2024_minus_full": (
            (ex2024.get("t") - full.get("t"))
            if (ex2024.get("t") is not None and full.get("t") is not None)
            else None
        ),
        # Vol regime
        "atr20_pct_2024": vol.get("atr20_pct_2024"),
        "atr20_pct_rest": vol.get("atr20_pct_rest"),
        "garch_vol_pct_2024": vol.get("garch_vol_pct_2024"),
        "garch_vol_pct_rest": vol.get("garch_vol_pct_rest"),
        # Flag
        "flag_2024": flag,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})
    try:
        setups = load_active_setups(con)
        print(f"Loaded {len(setups)} active validated_setups lanes")
        rows = []
        for i, spec in enumerate(setups, 1):
            print(f"  [{i}/{len(setups)}] {spec['strategy_id']}")
            rows.append(audit_lane(con, spec))
    finally:
        con.close()

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "phase_2_7_regime_break_2024_audit.csv"
    df.to_csv(csv_path, index=False)

    # Stdout summary
    print()
    print("=" * 70)
    print("PHASE 2.7 — 2024 REGIME-BREAK SYSTEMIC AUDIT")
    print(f"Mode A holdout: trading_day < {HOLDOUT_SACRED_FROM} (unchanged)")
    print(f"2024 window:   {YEAR_2024_START} <= day < {YEAR_2024_END}")
    print(f"Flag delta threshold: |ex2024 - full| = {FLAG_DELTA_THRESHOLD}")
    print("=" * 70)
    print()

    flag_counts = df["flag_2024"].value_counts().sort_index()
    print("Flag breakdown:")
    for flag, cnt in flag_counts.items():
        print(f"  {flag:<25} {cnt:>3}")
    print()

    # Per filter_type
    print("By filter_type:")
    print(
        df.groupby("filter_type")["flag_2024"].value_counts().unstack(fill_value=0)
          .to_string()
    )
    print()

    # Per session
    print("By session:")
    print(
        df.groupby("session")["flag_2024"].value_counts().unstack(fill_value=0)
          .to_string()
    )
    print()

    # Vol-regime contrast: 2024 vs rest medians
    if df[["atr20_pct_2024", "atr20_pct_rest"]].notna().any().any():
        atr_2024 = df["atr20_pct_2024"].dropna().median()
        atr_rest = df["atr20_pct_rest"].dropna().median()
        print(f"ATR_20_pct regime contrast (median across lanes):")
        print(f"  2024:      {atr_2024:.4f}")
        print(f"  rest-of-Mode-A: {atr_rest:.4f}")
        if atr_rest and atr_rest != 0:
            print(f"  ratio 2024/rest: {atr_2024/atr_rest:.3f}")
        print()

    if df[["garch_vol_pct_2024", "garch_vol_pct_rest"]].notna().any().any():
        g_2024 = df["garch_vol_pct_2024"].dropna().median()
        g_rest = df["garch_vol_pct_rest"].dropna().median()
        print(f"GARCH_vol_pct regime contrast (median across lanes):")
        print(f"  2024:      {g_2024:.2f}")
        print(f"  rest-of-Mode-A: {g_rest:.2f}")
        print()

    # Biggest 2024-PURE_DRAG lanes (ranked by expr improvement)
    pure_drag = df[df["flag_2024"] == "2024_PURE_DRAG"].copy()
    if len(pure_drag):
        pure_drag = pure_drag.sort_values(
            "delta_expr_ex2024_minus_full", ascending=False
        )
        print(f"2024_PURE_DRAG lanes (n={len(pure_drag)}), ranked by ex-2024 lift:")
        cols = ["strategy_id", "full_expr", "ex2024_expr", "y2024_expr",
                "delta_expr_ex2024_minus_full", "full_t", "ex2024_t"]
        print(pure_drag[cols].to_string(index=False))
        print()

    # 2024_CRITICAL lanes (2024 carries)
    critical = df[df["flag_2024"] == "2024_CRITICAL"]
    if len(critical):
        print(f"2024_CRITICAL lanes (n={len(critical)}) — 2024 performance CARRIES the lane:")
        cols = ["strategy_id", "full_expr", "ex2024_expr", "y2024_expr",
                "delta_expr_ex2024_minus_full"]
        print(critical[cols].to_string(index=False))
        print()

    try:
        print(f"Written: {csv_path.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"Written: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
