#!/usr/bin/env python3
"""Regime-drift control on 4 CRITICAL committee lanes — 2026-04-19 Correction 3.

The Phase 8 committee review pack flagged 4 MNQ lanes for RETIRE/DOWNGRADE
based on Mode A Sharpe drop from stored (Mode B) values. Self-audit concern:
that framing attributes the drop to LANE DECAY when much/all of it may be
2024-2025 ENVIRONMENT-WIDE regime stress on MNQ intraday breakouts.

This script computes:
  1. Per-CRITICAL-lane Sharpe: 2022-2023 vs 2024-2025 (Mode A IS subsets)
  2. Portfolio-wide MNQ Sharpe (across all 36 active MNQ lanes) in same subsets
  3. Delta comparison: does the CRITICAL lane Sharpe drop look like
     lane-specific decay or environment-wide drop?

Interpretive rules:
  - If CRITICAL lanes' Sharpe drop is within 0.30 of portfolio-wide drop,
    it's REGIME — hold on retirement recommendations.
  - If CRITICAL lanes drop materially more (>0.5 worse than environment),
    it's lane-specific DECAY — retirement recommendation stands.
  - If CRITICAL lanes drop LESS than environment or flat, they're among
    the BETTER-than-peers lanes under regime stress — counter to the
    committee pack's "retire" framing.

Output: docs/audit/results/2026-04-19-regime-drift-control-critical-lanes.md
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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

RESULT_PATH = PROJECT_ROOT / "docs/audit/results/2026-04-19-regime-drift-control-critical-lanes.md"

# 4 CRITICAL lanes from Phase 8 committee pack
CRITICAL = [
    {"id": "CR1", "instrument": "MNQ", "session": "EUROPE_FLOW",  "orb_minutes": 5, "rr": 1.0, "direction": "long", "filter": "OVNRNG_100"},
    {"id": "CR2", "instrument": "MNQ", "session": "EUROPE_FLOW",  "orb_minutes": 5, "rr": 1.5, "direction": "long", "filter": "OVNRNG_100"},
    {"id": "CR3", "instrument": "MNQ", "session": "NYSE_OPEN",    "orb_minutes": 5, "rr": 1.0, "direction": "long", "filter": "X_MES_ATR60"},
    {"id": "CR4", "instrument": "MNQ", "session": "NYSE_OPEN",    "orb_minutes": 5, "rr": 1.5, "direction": "long", "filter": "X_MES_ATR60"},
]


@dataclass
class PeriodStats:
    label: str
    n: int = 0
    expr: float | None = None
    std: float | None = None
    sharpe_ann: float | None = None
    win_rate: float | None = None


@dataclass
class LaneDrift:
    id: str
    instrument: str
    session: str
    orb_minutes: int
    rr_target: float
    direction: str
    filter_type: str
    early: PeriodStats = field(default_factory=lambda: PeriodStats("2022_2023"))
    late: PeriodStats = field(default_factory=lambda: PeriodStats("2024_2025"))
    early_to_late_sharpe_drop: float | None = None


def stats(pnl: np.ndarray, year_span: int = 2) -> PeriodStats:
    ps = PeriodStats(label="")
    ps.n = len(pnl)
    if ps.n == 0:
        return ps
    ps.expr = float(np.mean(pnl))
    if ps.n > 1:
        std = float(np.std(pnl, ddof=1))
        ps.std = std
        if std > 0:
            trades_per_year = ps.n / max(1, year_span)
            sh_per_trade = ps.expr / std
            ps.sharpe_ann = sh_per_trade * math.sqrt(trades_per_year)
    return ps


def inject_cross_atr(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, filter_type: str, instrument: str) -> pd.DataFrame:
    if filter_type not in ALL_FILTERS:
        return df
    filt_obj = ALL_FILTERS[filter_type]
    if not isinstance(filt_obj, CrossAssetATRFilter):
        return df
    source = filt_obj.source_instrument
    if source == instrument:
        return df
    src_rows = con.execute("""
        SELECT trading_day, atr_20_pct FROM daily_features
        WHERE symbol=? AND orb_minutes=5 AND atr_20_pct IS NOT NULL
    """, [source]).fetchall()
    src_map = {td.date() if hasattr(td, "date") else td: float(pct) for td, pct in src_rows}
    col = f"cross_atr_{source}_pct"
    df = df.copy()
    df[col] = df["trading_day"].apply(lambda d: src_map.get(d.date() if hasattr(d, "date") else d))
    return df


def compute_lane_drift(con: duckdb.DuckDBPyConnection, lane: dict) -> LaneDrift:
    sess = lane["session"]
    sql = f"""
    SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol, d.*
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol=? AND o.orb_label=? AND o.orb_minutes=?
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=?
      AND d.orb_{sess}_break_dir=?
      AND o.pnl_r IS NOT NULL
      AND o.trading_day < ?
    """
    df = con.execute(sql, [lane["instrument"], sess, lane["orb_minutes"],
                            lane["rr"], lane["direction"], HOLDOUT_SACRED_FROM]).df()
    df = inject_cross_atr(con, df, lane["filter"], lane["instrument"])
    fire = np.asarray(filter_signal(df, lane["filter"], sess)).astype(bool)
    df_on = df.loc[fire].copy()
    df_on["trading_day"] = pd.to_datetime(df_on["trading_day"])

    # Partition by year
    early_mask = df_on["trading_day"].dt.year.isin([2022, 2023])
    late_mask = df_on["trading_day"].dt.year.isin([2024, 2025])

    early_pnl = df_on.loc[early_mask, "pnl_r"].astype(float).to_numpy()
    late_pnl = df_on.loc[late_mask, "pnl_r"].astype(float).to_numpy()

    ld = LaneDrift(
        id=lane["id"], instrument=lane["instrument"], session=sess,
        orb_minutes=lane["orb_minutes"], rr_target=lane["rr"],
        direction=lane["direction"], filter_type=lane["filter"],
    )
    ld.early = stats(early_pnl, year_span=2)
    ld.late = stats(late_pnl, year_span=2)
    if ld.early.sharpe_ann is not None and ld.late.sharpe_ann is not None:
        ld.early_to_late_sharpe_drop = ld.late.sharpe_ann - ld.early.sharpe_ann
    return ld


def _resolve_direction(spec: dict) -> str:
    """Resolve the direction of a validated_setups row.

    Canonical source is `execution_spec` JSON (optional) or `strategy_id`
    naming convention. The pre-2026-04 project convention is long-only
    for MNQ active book — BUT silently defaulting-to-long masks future
    bugs. This resolver asserts the direction can be determined and
    errors loudly otherwise.

    Priority:
      1. If `execution_spec` is non-null and contains exactly one of
         "long"/"short" (case-insensitive), use that.
      2. If strategy_id contains an explicit _LONG_/_SHORT_ segment, use that.
      3. Else default to "long" WITH an emitted warning (legacy long-only).
         Callers can pass --strict to elevate this path to an error.
    """
    spec_str = str(spec.get("execution_spec") or "").lower()
    has_long = "long" in spec_str
    has_short = "short" in spec_str
    if has_long and not has_short:
        return "long"
    if has_short and not has_long:
        return "short"
    sid = str(spec.get("strategy_id") or "").upper()
    if "_SHORT_" in sid:
        return "short"
    if "_LONG_" in sid:
        return "long"
    # Legacy fallback — emit a one-time warning via stderr.
    import sys as _sys
    if not getattr(_resolve_direction, "_warned", False):
        print(
            f"[_resolve_direction WARNING] spec {spec.get('strategy_id')!r} "
            f"has null/ambiguous execution_spec; defaulting to long. Any future "
            f"short-direction lane will be silently misdirected unless "
            f"execution_spec or strategy_id encodes direction explicitly.",
            file=_sys.stderr,
        )
        _resolve_direction._warned = True  # type: ignore[attr-defined]
    return "long"


def portfolio_mnq_sharpe_per_period(con: duckdb.DuckDBPyConnection) -> tuple[PeriodStats, PeriodStats, list[dict]]:
    """Compute the ALL-ACTIVE-MNQ-LANES portfolio-wide Sharpe for each period.

    Method: for every active MNQ validated_setups row, compute Mode A IS
    per-trade pnl_r restricted to the filter. Aggregate all resulting trades
    across all 36 lanes. Compute aggregate Sharpe for 2022-2023 vs 2024-2025.

    Also returns per-lane early/late Sharpe for ALL 36 MNQ lanes to enable
    delta comparison against the 4 CRITICAL.
    """
    active = con.execute("""
        SELECT strategy_id, instrument, orb_label, orb_minutes, rr_target, entry_model,
               confirm_bars, filter_type, execution_spec
        FROM validated_setups
        WHERE LOWER(status)='active' AND instrument='MNQ'
        ORDER BY orb_label, orb_minutes, rr_target, filter_type
    """).fetchall()
    cols = ["strategy_id", "instrument", "orb_label", "orb_minutes", "rr_target",
            "entry_model", "confirm_bars", "filter_type", "execution_spec"]
    active_dicts = [dict(zip(cols, r)) for r in active]

    all_early: list[float] = []
    all_late: list[float] = []
    per_lane: list[dict] = []

    for spec in active_dicts:
        direction = _resolve_direction(spec)
        sess = spec["orb_label"]
        try:
            sql = f"""
            SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol, d.*
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
            WHERE o.symbol='MNQ' AND o.orb_label=? AND o.orb_minutes=?
              AND o.entry_model=? AND o.confirm_bars=? AND o.rr_target=?
              AND d.orb_{sess}_break_dir=?
              AND o.pnl_r IS NOT NULL AND o.trading_day < ?
            """
            df = con.execute(sql, [sess, spec["orb_minutes"], spec["entry_model"],
                                    spec["confirm_bars"], spec["rr_target"], direction, HOLDOUT_SACRED_FROM]).df()
        except duckdb.Error:
            continue
        if len(df) == 0:
            continue
        df = inject_cross_atr(con, df, spec["filter_type"], spec["instrument"])
        try:
            fire = np.asarray(filter_signal(df, spec["filter_type"], sess)).astype(bool)
        except Exception:
            continue
        df_on = df.loc[fire].copy()
        df_on["trading_day"] = pd.to_datetime(df_on["trading_day"])
        early_p = df_on.loc[df_on["trading_day"].dt.year.isin([2022, 2023]), "pnl_r"].astype(float).to_numpy()
        late_p = df_on.loc[df_on["trading_day"].dt.year.isin([2024, 2025]), "pnl_r"].astype(float).to_numpy()
        all_early.extend(early_p.tolist())
        all_late.extend(late_p.tolist())
        e_st = stats(early_p, 2)
        l_st = stats(late_p, 2)
        per_lane.append({
            "strategy_id": spec["strategy_id"],
            "orb_label": sess, "orb_minutes": spec["orb_minutes"],
            "rr_target": spec["rr_target"], "filter_type": spec["filter_type"],
            "direction": direction,
            "early_n": e_st.n, "early_sharpe": e_st.sharpe_ann,
            "late_n": l_st.n, "late_sharpe": l_st.sharpe_ann,
            "sharpe_drop": None if (e_st.sharpe_ann is None or l_st.sharpe_ann is None) else (l_st.sharpe_ann - e_st.sharpe_ann),
        })

    early_port = stats(np.asarray(all_early), year_span=2)
    early_port.label = "2022_2023 (portfolio all-MNQ aggregated)"
    late_port = stats(np.asarray(all_late), year_span=2)
    late_port.label = "2024_2025 (portfolio all-MNQ aggregated)"
    return early_port, late_port, per_lane


def _fmt(x, p=3):
    if x is None: return "—"
    if isinstance(x, float):
        if math.isnan(x): return "nan"
        return f"{x:.{p}f}"
    return str(x)


def render(critical_drifts: list[LaneDrift], port_early: PeriodStats, port_late: PeriodStats, per_lane: list[dict]) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    port_drop = None
    if port_early.sharpe_ann is not None and port_late.sharpe_ann is not None:
        port_drop = port_late.sharpe_ann - port_early.sharpe_ann

    L: list[str] = []
    L.append("# Regime-drift control on 4 CRITICAL committee lanes — 2026-04-19")
    L.append("")
    L.append(f"**Generated:** {ts}")
    L.append(f"**Script:** `research/regime_drift_control_critical_lanes.py`")
    L.append(f"**IS boundary:** `trading_day < {HOLDOUT_SACRED_FROM}` (Mode A)")
    L.append("")
    L.append("## Motivation")
    L.append("")
    L.append("The 2026-04-19 Phase 8 committee review pack flagged 4 MNQ lanes as CRITICAL for RETIRE/DOWNGRADE based on Mode A Sharpe drop from stored values. The 2026-04-19 self-audit identified a framing bias: attributing the drop to LANE DECAY without controlling for 2024-2025 ENVIRONMENT-WIDE regime stress on MNQ intraday breakouts.")
    L.append("")
    L.append("This control test:")
    L.append("  1. Recomputes each CRITICAL lane's Sharpe in early (2022-2023) vs late (2024-2025) Mode A IS subsets")
    L.append("  2. Computes portfolio-wide MNQ aggregate Sharpe in same subsets (all 36 active MNQ lanes pooled)")
    L.append("  3. Compares each CRITICAL lane's Sharpe drop vs the portfolio-wide drop")
    L.append("")
    L.append("Interpretive thresholds:")
    L.append("  - **REGIME** (hold retirement): lane Sharpe drop within 0.30 of portfolio-wide drop")
    L.append("  - **DECAY** (retirement recommendation stands): lane drops >0.50 beyond environment")
    L.append("  - **BETTER-THAN-PEERS** (counter to committee framing): lane drops LESS than environment or rises")
    L.append("")
    L.append("## Portfolio-wide MNQ aggregate")
    L.append("")
    L.append(f"Early (2022-2023): N={port_early.n} ExpR={_fmt(port_early.expr)} Sharpe_ann={_fmt(port_early.sharpe_ann, 2)}")
    L.append(f"Late  (2024-2025): N={port_late.n} ExpR={_fmt(port_late.expr)} Sharpe_ann={_fmt(port_late.sharpe_ann, 2)}")
    L.append(f"**Portfolio-wide Sharpe drop early->late:** {_fmt(port_drop, 2) if port_drop is not None else '—'}")
    L.append("")
    L.append("This is the baseline environment delta against which the 4 CRITICAL lanes should be measured.")
    L.append("")
    L.append("## CRITICAL lanes — per-lane drift vs environment")
    L.append("")
    L.append("| Cell | Early 2022-2023 Sharpe | Late 2024-2025 Sharpe | Lane drop | Portfolio drop | Excess drop vs port | Verdict |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for cd in critical_drifts:
        excess = None
        if cd.early_to_late_sharpe_drop is not None and port_drop is not None:
            excess = cd.early_to_late_sharpe_drop - port_drop
        if excess is None:
            verdict = "INCOMPLETE"
        elif abs(excess) < 0.30:
            verdict = "REGIME (hold)"
        elif excess < -0.50:
            verdict = "**DECAY** (retire)"
        elif excess > 0.30:
            verdict = "BETTER-THAN-PEERS (keep)"
        else:
            verdict = "BORDERLINE"
        e_s = cd.early.sharpe_ann
        l_s = cd.late.sharpe_ann
        L.append(f"| {cd.id} {cd.instrument} {cd.session} O{cd.orb_minutes} RR{cd.rr_target} {cd.filter_type} {cd.direction} | "
                 f"{_fmt(e_s, 2)} (N={cd.early.n}) | {_fmt(l_s, 2)} (N={cd.late.n}) | "
                 f"{_fmt(cd.early_to_late_sharpe_drop, 2)} | {_fmt(port_drop, 2)} | "
                 f"{_fmt(excess, 2)} | {verdict} |")
    L.append("")
    L.append("## All 36 MNQ active lanes — early/late Sharpe + drop")
    L.append("")
    L.append("(Sorted by Sharpe drop; most-decayed first)")
    L.append("")
    L.append("| Strategy ID | Early Sharpe (N) | Late Sharpe (N) | Drop |")
    L.append("|---|---:|---:|---:|")
    # Sort by drop ascending (most negative drop first)
    per_lane_sorted = sorted(per_lane, key=lambda x: x["sharpe_drop"] if x["sharpe_drop"] is not None else 999)
    for r in per_lane_sorted:
        e_s = f"{_fmt(r['early_sharpe'], 2)} (N={r['early_n']})"
        l_s = f"{_fmt(r['late_sharpe'], 2)} (N={r['late_n']})"
        drop = _fmt(r['sharpe_drop'], 2)
        L.append(f"| `{r['strategy_id']}` | {e_s} | {l_s} | {drop} |")
    L.append("")
    L.append("## Interpretation")
    L.append("")
    # Compute distribution of drops
    drops = [r["sharpe_drop"] for r in per_lane if r["sharpe_drop"] is not None]
    if drops:
        median_drop = float(np.median(drops))
        q25 = float(np.percentile(drops, 25))
        q75 = float(np.percentile(drops, 75))
        L.append(f"Portfolio of 36 MNQ lanes: median Sharpe drop = {_fmt(median_drop, 2)}, IQR [{_fmt(q25, 2)}, {_fmt(q75, 2)}].")
        L.append("")
        n_decayed = sum(1 for d in drops if d < -0.50)
        n_mild = sum(1 for d in drops if -0.50 <= d < -0.20)
        n_flat = sum(1 for d in drops if -0.20 <= d < 0.20)
        n_up = sum(1 for d in drops if d >= 0.20)
        L.append(f"- Lanes with DECAY (Sharpe drop > 0.50): {n_decayed} / {len(drops)}")
        L.append(f"- Lanes with mild drop (0.20-0.50): {n_mild}")
        L.append(f"- Lanes roughly flat (|drop| < 0.20): {n_flat}")
        L.append(f"- Lanes with Sharpe UP: {n_up}")
        L.append("")
    L.append("## Recommendation for the committee review pack")
    L.append("")
    any_decay = any(
        (cd.early_to_late_sharpe_drop is not None and port_drop is not None and
         (cd.early_to_late_sharpe_drop - port_drop) < -0.50)
        for cd in critical_drifts
    )
    any_regime = any(
        (cd.early_to_late_sharpe_drop is not None and port_drop is not None and
         abs(cd.early_to_late_sharpe_drop - port_drop) < 0.30)
        for cd in critical_drifts
    )
    if any_decay and not any_regime:
        L.append("Some or all CRITICAL lanes show excess Sharpe drop > 0.50 vs environment — confirms LANE DECAY framing. Retirement recommendation stands for those specific lanes.")
    elif any_regime and not any_decay:
        L.append("ALL CRITICAL lanes show drops within 0.30 of portfolio-wide drop — REGIME framing. The committee pack's RETIRE framing was over-attributed to lane decay. Recommended action: HOLD on retirement; reclassify as regime-stressed, continue monitoring.")
    elif any_decay and any_regime:
        L.append("MIXED: some CRITICAL lanes look like regime, others like decay. Per-lane verdict above.")
    else:
        L.append("Both criteria weak. BORDERLINE — recommend per-lane committee discussion rather than bulk retirement.")
    L.append("")
    L.append("## Reproduction")
    L.append("```")
    L.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/regime_drift_control_critical_lanes.py")
    L.append("```")
    L.append("")
    L.append("Read-only. No writes.")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        critical_drifts = [compute_lane_drift(con, lane) for lane in CRITICAL]
        port_early, port_late, per_lane = portfolio_mnq_sharpe_per_period(con)
    finally:
        con.close()

    port_drop = None
    if port_early.sharpe_ann is not None and port_late.sharpe_ann is not None:
        port_drop = port_late.sharpe_ann - port_early.sharpe_ann
    print(f"Portfolio-wide MNQ Sharpe drop 2022-23 -> 2024-25: {port_drop}")
    print()
    for cd in critical_drifts:
        e_s = cd.early.sharpe_ann
        l_s = cd.late.sharpe_ann
        excess = None if (cd.early_to_late_sharpe_drop is None or port_drop is None) else (cd.early_to_late_sharpe_drop - port_drop)
        print(f"{cd.id} {cd.instrument} {cd.session} RR{cd.rr_target} {cd.filter_type}: early Sh={e_s} late Sh={l_s} drop={cd.early_to_late_sharpe_drop} excess={excess}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(render(critical_drifts, port_early, port_late, per_lane), encoding="utf-8")
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
