#!/usr/bin/env python3
"""Mode A canonical re-validation of every active validated_setups row.

Phase 3 deliverable. Recomputes N, ExpR, WR, Sharpe, per-year positivity
under STRICT Mode A IS (trading_day < HOLDOUT_SACRED_FROM) from canonical
orb_outcomes + daily_features, with filter applied via canonical
research.filter_utils.filter_signal. Flags any cell where the canonical
Mode A numbers diverge from the stored validated_setups values by more
than the tolerance thresholds (ΔN > 10% relative OR |ΔExpR| > 0.03
absolute).

Motivation: per research-truth-protocol.md § "Mode B grandfathered
validated_setups baselines" (2026-04-18), rows with last_trade_day in
[2026-01-01, 2026-04-08] were computed under the prior Mode B holdout
policy and include 2026 Q1 data which is now sacred Mode A OOS. The
stored expectancy_r values are therefore sample-inflated (larger N) and
may be stat-inflated (Mode A OOS included in Mode B IS).

This script writes NOTHING to validated_setups. It is a read-only audit
producing a canonical errata document.

Output: docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mode_a_revalidation_active_setups.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from research.filter_utils import filter_signal

RESULT_PATH = (
    PROJECT_ROOT
    / "docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md"
)

# Divergence flagging thresholds
N_RATIO_TOLERANCE = 0.10        # flag if |delta_N / stored_N| > 0.10
EXPR_ABS_TOLERANCE = 0.03       # flag if |delta_ExpR| > 0.03
SHARPE_ABS_TOLERANCE = 0.20     # flag if |delta_Sharpe| > 0.20


@dataclass
class LaneRevalidation:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    entry_model: str
    confirm_bars: int
    filter_type: str | None
    direction: str

    # Stored (from validated_setups, potentially Mode-B grandfathered)
    stored_n: int = 0
    stored_expr: float | None = None
    stored_sharpe: float | None = None
    stored_wr: float | None = None
    stored_last_trade_day: date | None = None

    # Mode A canonical (recomputed)
    mode_a_n: int = 0
    mode_a_expr: float | None = None
    mode_a_sharpe: float | None = None
    mode_a_wr: float | None = None

    # Divergence flags
    delta_n: int | None = None
    delta_n_ratio: float | None = None
    delta_expr: float | None = None
    delta_sharpe: float | None = None
    mode_b_contaminated: bool = False
    material_drift: bool = False
    drift_reasons: list[str] = field(default_factory=list)

    # Per-year Mode A breakdown (for WFE / sanity)
    years_positive: int = 0
    years_total: int = 0
    years_breakdown: dict[int, dict[str, Any]] = field(default_factory=dict)


def direction_from_execution_spec(spec: str | None) -> str:
    """Resolve long/short from execution_spec JSON or default to 'long'."""
    if not spec:
        return "long"
    if "short" in str(spec).lower():
        return "short"
    return "long"


def load_active_setups(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT strategy_id, instrument, orb_label, orb_minutes, rr_target,
               entry_model, confirm_bars, filter_type, sample_size,
               expectancy_r, sharpe_ann, win_rate, last_trade_day,
               execution_spec
        FROM validated_setups
        WHERE LOWER(status) = 'active'
        ORDER BY instrument, orb_label, orb_minutes, rr_target, filter_type
        """
    ).fetchall()
    cols = [
        "strategy_id", "instrument", "orb_label", "orb_minutes", "rr_target",
        "entry_model", "confirm_bars", "filter_type", "sample_size",
        "expectancy_r", "sharpe_ann", "win_rate", "last_trade_day",
        "execution_spec",
    ]
    return [dict(zip(cols, r)) for r in rows]


def compute_mode_a(
    con: duckdb.DuckDBPyConnection, spec: dict[str, Any]
) -> tuple[int, float | None, float | None, float | None, dict[int, dict[str, Any]]]:
    """Recompute (N, ExpR, Sharpe_ann, WR, year_breakdown) under Mode A IS
    (trading_day < HOLDOUT_SACRED_FROM) using canonical filter delegation.
    """
    sess = spec["orb_label"]
    direction = direction_from_execution_spec(spec.get("execution_spec"))
    # SELECT d.* because different canonical filters look up different columns
    # by canonical name (e.g., CostRatioFilter needs orb_{sess}_size + symbol;
    # VWAPBreakDirectionFilter needs orb_{sess}_vwap + orb_{sess}_high/low;
    # OvernightRangeAbsFilter needs overnight_range). Loading all daily_features
    # columns is the only way to remain filter-agnostic without hand-coding a
    # column-needs map per filter. orb_outcomes columns pnl_r/outcome/symbol
    # are selected explicitly since the JOIN aliases could otherwise collide.
    sql = f"""
        SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol,
               d.*
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
        return 0, None, None, None, {}

    filter_type = spec.get("filter_type")

    # CrossAssetATRFilter requires cross_atr_{source}_pct which is NOT in
    # daily_features schema — it is injected at discovery/fitness time by
    # _inject_cross_asset_atrs (canonical impl at trading_app/strategy_discovery.py:978).
    # Mirror that injection here so X_MES_ATR60 et al. evaluate correctly.
    if filter_type and filter_type in ALL_FILTERS:
        filt_obj = ALL_FILTERS[filter_type]
        if isinstance(filt_obj, CrossAssetATRFilter):
            source = filt_obj.source_instrument
            if source != spec["instrument"]:
                src_rows = con.execute(
                    """SELECT trading_day, atr_20_pct FROM daily_features
                       WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL""",
                    [source],
                ).fetchall()
                src_map: dict[date, float] = {}
                for td, pct in src_rows:
                    key = td.date() if hasattr(td, "date") else td
                    src_map[key] = float(pct)
                col = f"cross_atr_{source}_pct"
                df[col] = df["trading_day"].apply(
                    lambda d: src_map.get(d.date() if hasattr(d, "date") else d)
                )

    if filter_type and filter_type != "UNFILTERED":
        try:
            fire = np.asarray(filter_signal(df, filter_type, sess)).astype(bool)
        except Exception as e:
            print(f"  [warn] filter_signal failed for {filter_type} on {sess}: {e}")
            return 0, None, None, None, {}
        df_on = df[fire].reset_index(drop=True)
    else:
        df_on = df

    if len(df_on) == 0:
        return 0, None, None, None, {}

    pnl = df_on["pnl_r"].astype(float).to_numpy()
    n = len(pnl)
    expr = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1)) if n > 1 else None
    sharpe_per_trade = expr / std if std and std > 0 else None
    wr = float(np.mean(df_on["outcome"].astype(str) == "win"))

    # Annualize Sharpe — conservative: assume ~250 trading days × (trades/day ≈ 1 per eligible day)
    # Per-cell annualization: use observed trades/year estimate
    df_on["_year"] = pd.to_datetime(df_on["trading_day"]).dt.year
    years_sorted = sorted(df_on["_year"].unique())
    if sharpe_per_trade is not None and len(years_sorted) > 0:
        trades_per_year = n / len(years_sorted)
        sharpe_ann = sharpe_per_trade * math.sqrt(trades_per_year)
    else:
        sharpe_ann = None

    year_break: dict[int, dict[str, Any]] = {}
    for yr in years_sorted:
        yr_pnl = df_on.loc[df_on["_year"] == yr, "pnl_r"].astype(float).to_numpy()
        if len(yr_pnl) == 0:
            continue
        yr_expr = float(np.mean(yr_pnl))
        year_break[int(yr)] = {
            "n": len(yr_pnl),
            "expr": yr_expr,
            "positive": yr_expr > 0,
        }

    return n, expr, sharpe_ann, wr, year_break


def classify_divergence(rv: LaneRevalidation) -> None:
    reasons: list[str] = []
    if rv.stored_n > 0 and rv.mode_a_n > 0:
        rv.delta_n = rv.mode_a_n - rv.stored_n
        rv.delta_n_ratio = (rv.mode_a_n - rv.stored_n) / rv.stored_n
        if abs(rv.delta_n_ratio) > N_RATIO_TOLERANCE:
            reasons.append(f"|ΔN/N|={abs(rv.delta_n_ratio):.2f}>{N_RATIO_TOLERANCE}")
    if rv.stored_expr is not None and rv.mode_a_expr is not None:
        rv.delta_expr = rv.mode_a_expr - rv.stored_expr
        if abs(rv.delta_expr) > EXPR_ABS_TOLERANCE:
            reasons.append(f"|ΔExpR|={abs(rv.delta_expr):.3f}>{EXPR_ABS_TOLERANCE}")
    if rv.stored_sharpe is not None and rv.mode_a_sharpe is not None:
        rv.delta_sharpe = rv.mode_a_sharpe - rv.stored_sharpe
        if abs(rv.delta_sharpe) > SHARPE_ABS_TOLERANCE:
            reasons.append(f"|ΔSharpe|={abs(rv.delta_sharpe):.2f}>{SHARPE_ABS_TOLERANCE}")

    # Mode-B contamination indicator: last_trade_day after holdout boundary
    if rv.stored_last_trade_day is not None:
        if rv.stored_last_trade_day >= HOLDOUT_SACRED_FROM:
            rv.mode_b_contaminated = True
            reasons.append(
                f"last_trade_day={rv.stored_last_trade_day} >= {HOLDOUT_SACRED_FROM} "
                "(Mode-B grandfathered)"
            )

    rv.drift_reasons = reasons
    rv.material_drift = bool(reasons)


def _fmt(x: Any, places: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{places}f}"
    return str(x)


def render(results: list[LaneRevalidation]) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_total = len(results)
    n_drift = sum(1 for r in results if r.material_drift)
    n_mode_b = sum(1 for r in results if r.mode_b_contaminated)

    lines: list[str] = []
    lines.append("# Mode A canonical re-validation of active validated_setups")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Script:** `research/mode_a_revalidation_active_setups.py`")
    lines.append(f"**IS boundary:** `trading_day < {HOLDOUT_SACRED_FROM}` (Mode A)")
    lines.append(f"**Canonical filter source:** `research.filter_utils.filter_signal` → `trading_app.config.ALL_FILTERS`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total active lanes re-validated: **{n_total}**")
    lines.append(f"- Lanes with material Mode A drift (|ΔN/N|>10% OR |ΔExpR|>0.03 OR |ΔSharpe|>0.20 OR Mode-B contaminated): **{n_drift}**")
    lines.append(f"- Lanes with last_trade_day >= 2026-01-01 (Mode-B grandfathered): **{n_mode_b}**")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append(f"- N ratio: |ΔN / stored_N| > {N_RATIO_TOLERANCE} → flag")
    lines.append(f"- ExpR absolute: |ΔExpR| > {EXPR_ABS_TOLERANCE} → flag")
    lines.append(f"- Sharpe absolute: |ΔSharpe_ann| > {SHARPE_ABS_TOLERANCE} → flag")
    lines.append(f"- Mode-B contaminated: `last_trade_day >= {HOLDOUT_SACRED_FROM}` → flag")
    lines.append("")
    lines.append("A flagged lane does NOT mean the lane is wrong — it means the stored")
    lines.append("validated_setups values are computed on a different IS window than strict")
    lines.append("Mode A, and downstream decisions that cited those values are partially")
    lines.append("built on Mode-B baseline data. Treat the Mode A column as the canonical")
    lines.append("truth going forward; the validated_setups rows themselves are NOT mutated")
    lines.append("by this audit.")
    lines.append("")
    lines.append("## Per-lane re-validation")
    lines.append("")
    lines.append("| Instr | Session | Om | RR | Filter | Dir | Stored N / Mode-A N | ΔN/N | Stored ExpR / Mode-A ExpR | ΔExpR | Stored Sh / Mode-A Sh | ΔSh | Yrs+ | Mode-B | Flag |")
    lines.append("|---|---|---:|---:|---|---|---|---:|---|---:|---|---:|---:|---|---|")
    for r in results:
        flag = "DRIFT" if r.material_drift else ""
        mb = "Y" if r.mode_b_contaminated else "N"
        yrs = f"{r.years_positive}/{r.years_total}" if r.years_total else "—"
        lines.append(
            f"| {r.instrument} | {r.orb_label} | {r.orb_minutes} | {r.rr_target} | "
            f"{r.filter_type or 'UNFILTERED'} | {r.direction} | "
            f"{r.stored_n} / {r.mode_a_n} | {_fmt(r.delta_n_ratio, 2)} | "
            f"{_fmt(r.stored_expr)} / {_fmt(r.mode_a_expr)} | {_fmt(r.delta_expr)} | "
            f"{_fmt(r.stored_sharpe, 2)} / {_fmt(r.mode_a_sharpe, 2)} | {_fmt(r.delta_sharpe, 2)} | "
            f"{yrs} | {mb} | {flag} |"
        )
    lines.append("")
    lines.append("## Materially-drifted lanes — detail")
    lines.append("")
    for r in [x for x in results if x.material_drift]:
        lines.append(f"### {r.instrument} {r.orb_label} O{r.orb_minutes} RR{r.rr_target} {r.filter_type or 'UNFILTERED'} {r.direction}")
        lines.append(f"- `strategy_id`: `{r.strategy_id}`")
        lines.append(f"- Stored: N={r.stored_n} ExpR={_fmt(r.stored_expr)} Sharpe_ann={_fmt(r.stored_sharpe, 2)} WR={_fmt(r.stored_wr, 3)} last_trade_day={r.stored_last_trade_day}")
        lines.append(f"- Mode A: N={r.mode_a_n} ExpR={_fmt(r.mode_a_expr)} Sharpe_ann={_fmt(r.mode_a_sharpe, 2)} WR={_fmt(r.mode_a_wr, 3)}")
        lines.append(f"- Drift reasons: {', '.join(r.drift_reasons)}")
        if r.years_total:
            yr_str = " ".join(
                f"{yr}:{'+' if b['positive'] else '-'}{_fmt(b['expr'], 3)}(N={b['n']})"
                for yr, b in sorted(r.years_breakdown.items())
            )
            lines.append(f"- Mode-A per-year: {yr_str}")
        lines.append("")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mode_a_revalidation_active_setups.py")
    lines.append("```")
    lines.append("")
    lines.append("No writes to validated_setups or experimental_strategies. Output is this")
    lines.append("markdown document only. Numbers reproduce exactly on the same DB state.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        active = load_active_setups(con)
        print(f"Loaded {len(active)} active validated_setups")

        results: list[LaneRevalidation] = []
        for i, spec in enumerate(active, 1):
            direction = direction_from_execution_spec(spec.get("execution_spec"))
            n, expr, sharpe_ann, wr, year_break = compute_mode_a(con, spec)
            yrs_pos = sum(1 for b in year_break.values() if b["positive"] and b["n"] >= 10)
            yrs_tot = sum(1 for b in year_break.values() if b["n"] >= 10)

            rv = LaneRevalidation(
                strategy_id=spec["strategy_id"],
                instrument=spec["instrument"],
                orb_label=spec["orb_label"],
                orb_minutes=spec["orb_minutes"],
                rr_target=spec["rr_target"],
                entry_model=spec["entry_model"],
                confirm_bars=spec["confirm_bars"],
                filter_type=spec["filter_type"],
                direction=direction,
                stored_n=spec["sample_size"] or 0,
                stored_expr=spec["expectancy_r"],
                stored_sharpe=spec["sharpe_ann"],
                stored_wr=spec["win_rate"],
                stored_last_trade_day=spec["last_trade_day"],
                mode_a_n=n,
                mode_a_expr=expr,
                mode_a_sharpe=sharpe_ann,
                mode_a_wr=wr,
                years_positive=yrs_pos,
                years_total=yrs_tot,
                years_breakdown=year_break,
            )
            classify_divergence(rv)
            results.append(rv)

            flag_str = "!!" if rv.material_drift else "  "
            print(
                f"  {flag_str} {i:2d}/{len(active)} {rv.instrument} {rv.orb_label} "
                f"O{rv.orb_minutes} RR{rv.rr_target} {rv.filter_type or 'UNF':<22} "
                f"N={rv.stored_n:>4}/{rv.mode_a_n:<4} "
                f"ExpR={_fmt(rv.stored_expr, 3)}/{_fmt(rv.mode_a_expr, 3)}"
            )
    finally:
        con.close()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(render(results), encoding="utf-8")
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    drift_count = sum(1 for r in results if r.material_drift)
    print(f"Material drift lanes: {drift_count} / {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
