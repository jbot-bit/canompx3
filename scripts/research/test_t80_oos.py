"""
T80 Time-Stop OOS Validation — Paired Portfolio Replay

Tests whether the T80 early-exit rule improves walk-forward OOS performance
by comparing raw pnl_r vs COALESCE(ts_pnl_r, pnl_r) on the SAME strategies,
SAME trade days, SAME filters. Only the exit rule differs.

Design: docs/plans/2026-03-18-t80-oos-validation-design.md

Usage:
    python scripts/research/test_t80_oos.py
    python scripts/research/test_t80_oos.py --instrument MGC
    python scripts/research/test_t80_oos.py --instrument MNQ --csv output.csv
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import EARLY_EXIT_MINUTES

# WF parameters (mirror walkforward.py defaults)
WF_TEST_MONTHS = 6
WF_MIN_TRAIN_MONTHS = 12
WF_START_OVERRIDE = {"MGC": date(2022, 1, 1)}
MIN_TRADES_PER_WINDOW = 15


@dataclass
class WindowResult:
    window_idx: int
    test_start: date
    test_end: date
    n_trades: int
    raw_exp_r: float
    ts_exp_r: float
    raw_sharpe: float
    ts_sharpe: float
    raw_max_dd: float
    ts_max_dd: float
    delta_exp_r: float  # ts minus raw


@dataclass
class SessionResult:
    session: str
    n_trades_raw: int
    n_trades_ts: int  # should equal n_trades_raw
    raw_total_r: float
    ts_total_r: float
    delta_r: float
    delta_per_trade: float
    p_value: float  # paired t-test on per-trade delta
    n_time_stop: int  # trades where ts_outcome != outcome
    verdict: str  # HELPS / HURTS / NEUTRAL / INCONCLUSIVE


@dataclass
class InstrumentResult:
    instrument: str
    windows: list[WindowResult] = field(default_factory=list)
    sessions: list[SessionResult] = field(default_factory=list)
    n_strategies: int = 0
    n_total_trades: int = 0
    n_time_stop_trades: int = 0
    agg_raw_exp_r: float = 0.0
    agg_ts_exp_r: float = 0.0
    agg_delta: float = 0.0
    paired_t_p: float = 1.0
    wilcoxon_p: float = 1.0
    n_windows_ts_better: int = 0
    n_windows_total: int = 0
    verdict: str = "INCONCLUSIVE"


def _add_months(d: date, months: int) -> date:
    """Add calendar months to a date."""
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return date(y, m, day)


def _sharpe(r_values: list[float]) -> float:
    """Per-trade Sharpe: mean / std."""
    if len(r_values) < 2:
        return 0.0
    arr = np.array(r_values)
    s = float(np.std(arr, ddof=1))
    return float(np.mean(arr) / s) if s > 0 else 0.0


def _max_drawdown(r_values: list[float]) -> float:
    """Max drawdown of cumulative R series."""
    if not r_values:
        return 0.0
    cum = np.cumsum(r_values)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def load_paired_outcomes(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> list[dict]:
    """Load all validated strategy outcomes with both raw and ts P&L.

    Returns list of dicts with: strategy_id, orb_label, trading_day, pnl_r, ts_pnl_r, outcome, ts_outcome
    """
    rows = con.execute(
        """
        SELECT vs.strategy_id, vs.orb_label, vs.orb_minutes,
               vs.entry_model, vs.rr_target, vs.confirm_bars, vs.filter_type,
               oo.trading_day,
               oo.outcome, oo.pnl_r,
               oo.ts_outcome, oo.ts_pnl_r
        FROM validated_setups vs
        JOIN orb_outcomes oo
          ON oo.symbol = vs.instrument
          AND oo.orb_label = vs.orb_label
          AND oo.orb_minutes = vs.orb_minutes
          AND oo.entry_model = vs.entry_model
          AND oo.rr_target = vs.rr_target
          AND oo.confirm_bars = vs.confirm_bars
        WHERE vs.instrument = ?
          AND oo.outcome IS NOT NULL
          AND oo.pnl_r IS NOT NULL
        ORDER BY oo.trading_day
        """,
        [instrument],
    ).fetchall()

    cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, r, strict=False)) for r in rows]


def apply_filter_eligibility(
    con: duckdb.DuckDBPyConnection,
    outcomes: list[dict],
    instrument: str,
) -> list[dict]:
    """Filter outcomes to eligible days only (ORB size filter).

    Simplified: checks ORB size >= G threshold from filter_type.
    For volume/composite filters, accepts all days (conservative — may
    slightly overcount but won't bias the A/B comparison since both arms
    use the same eligibility).
    """
    from trading_app.config import ALL_FILTERS

    # Group by (orb_label, orb_minutes, filter_type)
    groups = defaultdict(list)
    for o in outcomes:
        key = (o["orb_label"], o["orb_minutes"], o["filter_type"])
        groups[key].append(o)

    # Load daily features once
    df_rows = con.execute(
        """SELECT trading_day, orb_minutes, *
           FROM daily_features
           WHERE symbol = ?""",
        [instrument],
    ).fetchall()
    df_cols = [desc[0] for desc in con.description]
    df_by_key = {}
    for r in df_rows:
        d = dict(zip(df_cols, r, strict=False))
        df_by_key[(d["trading_day"], d["orb_minutes"])] = d

    eligible = []
    for (orb_label, orb_minutes, filter_type), group_outcomes in groups.items():
        filt = ALL_FILTERS.get(filter_type)
        if filt is None:
            continue  # fail-closed: unknown filter

        for o in group_outcomes:
            td = o["trading_day"]
            df_row = df_by_key.get((td, orb_minutes))
            if df_row is None:
                continue

            # For simple ORB size filters, check eligibility
            # For complex filters (VOL, FAST, etc.), accept all — both arms identical
            if filter_type == "NO_FILTER" or filt.matches_row(df_row, orb_label):
                eligible.append(o)

    return eligible


def generate_wf_windows(
    trading_days: list[date],
    instrument: str,
) -> list[tuple[date, date]]:
    """Generate non-overlapping WF test windows (calendar mode).

    Returns list of (test_start, test_end) tuples.
    """
    if not trading_days:
        return []

    earliest = min(trading_days)
    latest = max(trading_days)

    anchor = WF_START_OVERRIDE.get(instrument, earliest)
    if anchor < earliest:
        anchor = earliest

    windows = []
    window_start = _add_months(anchor, WF_MIN_TRAIN_MONTHS)

    while window_start <= latest:
        window_end = _add_months(window_start, WF_TEST_MONTHS)
        windows.append((window_start, window_end))
        window_start = window_end

    return windows


def run_instrument_test(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> InstrumentResult:
    """Run the paired T80 OOS test for one instrument."""
    result = InstrumentResult(instrument=instrument)

    # Load paired outcomes
    all_outcomes = load_paired_outcomes(con, instrument)
    if not all_outcomes:
        result.verdict = "NO DATA"
        return result

    # Apply filter eligibility
    eligible = apply_filter_eligibility(con, all_outcomes, instrument)
    if not eligible:
        result.verdict = "NO ELIGIBLE TRADES"
        return result

    # Count strategies
    strategy_ids = {o["strategy_id"] for o in eligible}
    result.n_strategies = len(strategy_ids)

    # Resolve ts_pnl_r (COALESCE)
    for o in eligible:
        o["ts_pnl_r_resolved"] = o["ts_pnl_r"] if o["ts_pnl_r"] is not None else o["pnl_r"]
        o["is_time_stop"] = o["ts_outcome"] is not None and o["ts_outcome"] != o["outcome"]

    result.n_total_trades = len(eligible)
    result.n_time_stop_trades = sum(1 for o in eligible if o["is_time_stop"])

    # Get unique trading days for WF windows
    all_trading_days = sorted({o["trading_day"] for o in eligible})
    windows = generate_wf_windows(all_trading_days, instrument)

    if len(windows) < 3:
        result.verdict = f"TOO FEW WINDOWS ({len(windows)})"
        return result

    # --- Per-window portfolio metrics ---
    window_results = []
    for i, (ws, we) in enumerate(windows):
        # Get trades in this window
        window_trades = [o for o in eligible if ws <= o["trading_day"] < we]
        if len(window_trades) < MIN_TRADES_PER_WINDOW:
            continue

        raw_rs = [o["pnl_r"] for o in window_trades]
        ts_rs = [o["ts_pnl_r_resolved"] for o in window_trades]

        raw_exp = float(np.mean(raw_rs))
        ts_exp = float(np.mean(ts_rs))

        wr = WindowResult(
            window_idx=i,
            test_start=ws,
            test_end=we,
            n_trades=len(window_trades),
            raw_exp_r=round(raw_exp, 4),
            ts_exp_r=round(ts_exp, 4),
            raw_sharpe=round(_sharpe(raw_rs), 4),
            ts_sharpe=round(_sharpe(ts_rs), 4),
            raw_max_dd=round(_max_drawdown(raw_rs), 4),
            ts_max_dd=round(_max_drawdown(ts_rs), 4),
            delta_exp_r=round(ts_exp - raw_exp, 4),
        )
        window_results.append(wr)

    result.windows = window_results
    result.n_windows_total = len(window_results)

    if result.n_windows_total < 3:
        result.verdict = f"TOO FEW VALID WINDOWS ({result.n_windows_total})"
        return result

    # --- Aggregate metrics ---
    total_raw_r = sum(w.raw_exp_r * w.n_trades for w in window_results)
    total_ts_r = sum(w.ts_exp_r * w.n_trades for w in window_results)
    total_trades = sum(w.n_trades for w in window_results)

    result.agg_raw_exp_r = round(total_raw_r / total_trades, 4) if total_trades > 0 else 0.0
    result.agg_ts_exp_r = round(total_ts_r / total_trades, 4) if total_trades > 0 else 0.0
    result.agg_delta = round(result.agg_ts_exp_r - result.agg_raw_exp_r, 4)
    result.n_windows_ts_better = sum(1 for w in window_results if w.delta_exp_r > 0)

    # --- Statistical tests on per-window deltas ---
    deltas = [w.delta_exp_r for w in window_results]

    if len(deltas) >= 3:
        t_stat, p_val = stats.ttest_rel(
            [w.ts_exp_r for w in window_results],
            [w.raw_exp_r for w in window_results],
        )
        result.paired_t_p = round(float(p_val), 4)

        # Wilcoxon signed-rank (non-parametric)
        try:
            _, wilc_p = stats.wilcoxon(deltas)
            result.wilcoxon_p = round(float(wilc_p), 4)
        except ValueError:
            result.wilcoxon_p = 1.0  # all zeros or too few

    # --- Per-session breakdown ---
    session_trades = defaultdict(list)
    for o in eligible:
        session_trades[o["orb_label"]].append(o)

    session_results = []
    for session, trades in sorted(session_trades.items()):
        has_t80 = EARLY_EXIT_MINUTES.get(session) is not None
        n_ts = sum(1 for t in trades if t["is_time_stop"])

        raw_rs = [t["pnl_r"] for t in trades]
        ts_rs = [t["ts_pnl_r_resolved"] for t in trades]

        raw_total = sum(raw_rs)
        ts_total = sum(ts_rs)
        delta_total = ts_total - raw_total
        delta_per = delta_total / len(trades) if trades else 0.0

        # Paired t-test on per-trade deltas
        per_trade_deltas = [ts - raw for ts, raw in zip(ts_rs, raw_rs, strict=True)]
        if len(per_trade_deltas) >= 20 and any(d != 0 for d in per_trade_deltas):
            _, p_val = stats.ttest_1samp(per_trade_deltas, 0)
            p_val = float(p_val)
        else:
            p_val = 1.0

        # Verdict
        if not has_t80:
            verdict = "NO T80"
        elif n_ts == 0:
            verdict = "NO EFFECT"
        elif p_val < 0.05 and delta_per > 0:
            verdict = "HELPS"
        elif p_val < 0.05 and delta_per < 0:
            verdict = "HURTS"
        elif abs(delta_per) < 0.01:
            verdict = "NEUTRAL"
        else:
            verdict = "INCONCLUSIVE"

        session_results.append(
            SessionResult(
                session=session,
                n_trades_raw=len(raw_rs),
                n_trades_ts=len(ts_rs),
                raw_total_r=round(raw_total, 2),
                ts_total_r=round(ts_total, 2),
                delta_r=round(delta_total, 2),
                delta_per_trade=round(delta_per, 4),
                p_value=round(p_val, 4),
                n_time_stop=n_ts,
                verdict=verdict,
            )
        )

    result.sessions = session_results

    # --- Apply BH FDR to session p-values ---
    session_pvals = [(s.session, s.p_value) for s in session_results if s.p_value < 1.0]
    if session_pvals:
        session_pvals.sort(key=lambda x: x[1])
        m = len(session_pvals)
        for rank, (sess_name, pval) in enumerate(session_pvals, 1):
            bh_threshold = 0.10 * rank / m
            # Find the session result and annotate
            for sr in session_results:
                if sr.session == sess_name:
                    if pval > bh_threshold:
                        # Not BH-significant — downgrade verdict
                        if sr.verdict in ("HELPS", "HURTS"):
                            sr.verdict += " (not BH-sig)"

    # --- Overall verdict ---
    if result.paired_t_p < 0.05 and result.agg_delta > 0:
        result.verdict = "T80 HELPS (p<0.05)"
    elif result.paired_t_p < 0.05 and result.agg_delta < 0:
        result.verdict = "T80 HURTS (p<0.05)"
    elif result.n_windows_ts_better > result.n_windows_total * 0.7:
        result.verdict = "T80 LIKELY HELPS (>70% windows)"
    elif result.n_windows_ts_better < result.n_windows_total * 0.3:
        result.verdict = "T80 LIKELY HURTS (<30% windows)"
    else:
        result.verdict = "INCONCLUSIVE"

    return result


def print_result(r: InstrumentResult) -> None:
    """Print formatted results."""
    print(f"\n{'=' * 70}")
    print(f"  {r.instrument} — T80 OOS VALIDATION")
    print(f"{'=' * 70}")
    print(f"  Strategies: {r.n_strategies}")
    print(f"  Total trades: {r.n_total_trades}")
    print(
        f"  Time-stop trades: {r.n_time_stop_trades} ({r.n_time_stop_trades / r.n_total_trades * 100:.1f}%)"
        if r.n_total_trades > 0
        else ""
    )
    print(f"  WF windows: {r.n_windows_total}")
    print()

    # Portfolio-level
    print(f"  {'Metric':<25} {'Raw':>10} {'T80':>10} {'Delta':>10}")
    print(f"  {'-' * 55}")
    print(f"  {'Agg OOS ExpR':<25} {r.agg_raw_exp_r:>10.4f} {r.agg_ts_exp_r:>10.4f} {r.agg_delta:>+10.4f}")
    print()
    print(f"  Paired t-test p-value: {r.paired_t_p}")
    print(f"  Wilcoxon p-value:      {r.wilcoxon_p}")
    print(f"  Windows where T80 better: {r.n_windows_ts_better}/{r.n_windows_total}")
    print()

    # Per-window
    if r.windows:
        print(f"  {'Window':<6} {'Period':<25} {'N':>5} {'Raw ExpR':>10} {'T80 ExpR':>10} {'Delta':>10}")
        print(f"  {'-' * 70}")
        for w in r.windows:
            print(
                f"  {w.window_idx:<6} "
                f"{str(w.test_start)}..{str(w.test_end):<13} "
                f"{w.n_trades:>5} "
                f"{w.raw_exp_r:>10.4f} "
                f"{w.ts_exp_r:>10.4f} "
                f"{w.delta_exp_r:>+10.4f}"
            )
        print()

    # Per-session
    if r.sessions:
        print(
            f"  {'Session':<20} {'N':>6} {'T80#':>5} {'Raw R':>8} {'T80 R':>8} {'Δ/trade':>8} {'p':>8} {'Verdict':<20}"
        )
        print(f"  {'-' * 95}")
        for s in r.sessions:
            print(
                f"  {s.session:<20} "
                f"{s.n_trades_raw:>6} "
                f"{s.n_time_stop:>5} "
                f"{s.raw_total_r:>8.1f} "
                f"{s.ts_total_r:>8.1f} "
                f"{s.delta_per_trade:>+8.4f} "
                f"{s.p_value:>8.4f} "
                f"{s.verdict:<20}"
            )
        print()

    print(f"  VERDICT: {r.verdict}")
    print(f"{'=' * 70}")


def write_csv(results: list[InstrumentResult], path: str) -> None:
    """Write per-window results to CSV."""
    with open(path, "w") as f:
        f.write("instrument,window,test_start,test_end,n_trades,raw_exp_r,ts_exp_r,delta_exp_r,raw_sharpe,ts_sharpe\n")
        for r in results:
            for w in r.windows:
                f.write(
                    f"{r.instrument},{w.window_idx},{w.test_start},{w.test_end},"
                    f"{w.n_trades},{w.raw_exp_r},{w.ts_exp_r},{w.delta_exp_r},"
                    f"{w.raw_sharpe},{w.ts_sharpe}\n"
                )
    print(f"\nCSV written to {path}")


def main():
    parser = argparse.ArgumentParser(description="T80 Time-Stop OOS Validation")
    parser.add_argument("--instrument", type=str, default=None, help="Single instrument (default: all active)")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--db", type=str, default=None, help="Database path override")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    con = duckdb.connect(str(db_path), read_only=True)

    instruments = [args.instrument] if args.instrument else ["MGC", "MNQ", "MES"]

    results = []
    for inst in instruments:
        print(f"\nRunning T80 OOS test for {inst}...")
        r = run_instrument_test(con, inst)
        print_result(r)
        results.append(r)

    con.close()

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        pct = f"{r.n_time_stop_trades / r.n_total_trades * 100:.1f}%" if r.n_total_trades > 0 else "N/A"
        print(f"  {r.instrument}: {r.verdict} (delta={r.agg_delta:+.4f}, p={r.paired_t_p}, time-stops={pct})")

    # Decision recommendation
    print(f"\n  {'─' * 50}")
    all_help = all(r.verdict.startswith("T80 HELPS") for r in results if r.n_total_trades > 0)
    any_hurt = any(r.verdict.startswith("T80 HURTS") for r in results if r.n_total_trades > 0)

    if all_help:
        print("  RECOMMENDATION: T80 improves OOS across all instruments.")
        print("  → Keep ts_outcome in discovery. Proceed with full rebuild.")
    elif any_hurt:
        hurts = [r.instrument for r in results if r.verdict.startswith("T80 HURTS")]
        helps = [r.instrument for r in results if r.verdict.startswith("T80 HELPS")]
        print(f"  RECOMMENDATION: T80 HURTS {hurts}, HELPS {helps}.")
        print("  → Apply T80 selectively. Remove from execution for instruments where it hurts.")
    else:
        print("  RECOMMENDATION: T80 effect is inconclusive.")
        print("  → Run sensitivity analysis before deciding. Do NOT rebuild yet.")
    print(f"{'=' * 70}")

    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
