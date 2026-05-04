"""
LONDON_ADJACENT HYPOTHESIS TEST

Tests whether an ORB at the hour adjacent to LONDON_METALS
(17:00 AEST winter / 18:00 AEST summer -- the DST audit's "wrong time")
has a tradeable edge with E1/E2 entries.

Hypothesis: The DST wrong-time audit (Mar 2026) found LONDON_METALS
wrong > right in both R and $, the strongest signal in the audit. But it
was dismissed on structural arguments (81% double-break, E3-only, G8+)
without testing whether the adjacent slot itself has an edge with E1/E2.

The adjacent slot is a DIFFERENT market event:
  Winter (7AM London/GMT): European pre-open positioning flow
  Summer (9AM London/BST): Post-metals, FTSE already trading

@research-source scripts/tmp_dst_wrong_time_audit.py
@research-source research/output/dst_wrong_time_audit_summary.csv

Usage:
    python research/research_london_adjacent.py

Date: 2026-03-11
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
from scipy.stats import ttest_1samp

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import COST_SPECS
from pipeline.dst import SESSION_CATALOG, is_uk_dst
from pipeline.paths import GOLD_DB_PATH

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

BRIS = ZoneInfo("Australia/Brisbane")
LONDON_METALS_RESOLVER = SESSION_CATALOG["LONDON_METALS"]["resolver"]

APERTURES = [5, 15, 30]
RR_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

# E1: confirm-based (CB1, CB3). E2: touch-based (always CB1 per production grid).
ENTRY_SPECS: list[tuple[str, int]] = [
    ("E1", 1),
    ("E1", 3),
    ("E2", 1),
]

G_FILTER_THRESHOLDS: dict[str, float] = {
    "NO_FILTER": 0.0,
    "ORB_G4": 4.0,
    "ORB_G5": 5.0,
    "ORB_G6": 6.0,
    "ORB_G8": 8.0,
}

MAX_BARS_POST_ORB = 120

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "output"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TradeResult:
    trading_day: date
    instrument: str
    aperture: int
    entry_model: str
    confirm_bars: int
    rr_target: float
    pnl_r_gross: float
    pnl_r_net: float
    pnl_dollars: float
    orb_range: float
    orb_volume: int
    break_dir: str
    double_break: bool
    season: str


# ═══════════════════════════════════════════════════════════════════════════
# ADJACENT TIME RESOLVER
# ═══════════════════════════════════════════════════════════════════════════


def get_adjacent_time_brisbane(trading_day: date) -> tuple[int, int]:
    """Compute the adjacent slot time in Brisbane hours.

    When LONDON_METALS is at 18:00 (winter/GMT), adjacent = 17:00.
    When LONDON_METALS is at 17:00 (summer/BST), adjacent = 18:00.
    """
    h_lm, m_lm = LONDON_METALS_RESOLVER(trading_day)
    if is_uk_dst(trading_day):
        return (h_lm + 1, m_lm)
    else:
        return (h_lm - 1, m_lm)


# ═══════════════════════════════════════════════════════════════════════════
# ORB + TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════


def compute_orb(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    bris_start: datetime,
    aperture: int,
) -> tuple[float | None, float | None, int]:
    """Compute ORB high/low/volume from 1m bars in the aperture window."""
    bris_end = bris_start + timedelta(minutes=aperture)
    rows = con.execute(
        "SELECT high, low, volume FROM bars_1m WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ? ORDER BY ts_utc",
        [symbol, bris_start, bris_end],
    ).fetchall()
    if not rows:
        return None, None, 0
    return max(r[0] for r in rows), min(r[1] for r in rows), sum(r[2] for r in rows)


def fetch_post_orb_bars(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    bris_orb_end: datetime,
    max_bars: int,
) -> list[tuple]:
    """Fetch 1m bars after ORB window for trade simulation."""
    end_time = bris_orb_end + timedelta(minutes=max_bars)
    return con.execute(
        "SELECT high, low, close FROM bars_1m WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ? ORDER BY ts_utc",
        [symbol, bris_orb_end, end_time],
    ).fetchall()


def detect_double_break(bars: list[tuple], orb_high: float, orb_low: float) -> bool:
    """Check if both ORB levels are breached (close-based) within scan window."""
    broke_high = any(c > orb_high for _, _, c in bars)
    broke_low = any(c < orb_low for _, _, c in bars)
    return broke_high and broke_low


def simulate_trade(
    bars: list[tuple],
    orb_high: float,
    orb_low: float,
    rr_target: float,
    entry_type: str,
    confirm_bars: int = 1,
) -> tuple[float | None, str | None]:
    """Simulate ORB breakout trade from post-ORB 1m bars.

    entry_type:
        "E1" -- N consecutive closes outside ORB -> entry at ORB level
        "E2" -- First bar whose range crosses ORB level (no close needed)

    Returns (pnl_r_gross, break_dir) or (None, None).
    """
    risk = orb_high - orb_low
    if risk <= 0 or not bars:
        return None, None

    # ── Phase 1: detect break ──
    break_dir: str | None = None
    break_idx: int | None = None

    if entry_type == "E1":
        consec_high = 0
        consec_low = 0
        for i, (h, l, c) in enumerate(bars):
            if c > orb_high:
                consec_high += 1
                consec_low = 0
                if consec_high >= confirm_bars:
                    break_dir = "long"
                    break_idx = i
                    break
            elif c < orb_low:
                consec_low += 1
                consec_high = 0
                if consec_low >= confirm_bars:
                    break_dir = "short"
                    break_idx = i
                    break
            else:
                consec_high = 0
                consec_low = 0
    else:  # E2: first bar whose range crosses ORB (touch-based)
        for i, (h, l, c) in enumerate(bars):
            crosses_high = h > orb_high
            crosses_low = l < orb_low
            if crosses_high and crosses_low:
                continue  # Ambiguous bar -- skip
            if crosses_high:
                break_dir = "long"
                break_idx = i
                break
            if crosses_low:
                break_dir = "short"
                break_idx = i
                break

    if break_dir is None or break_idx is None:
        return None, None

    # ── Phase 2: resolve trade (target or stop) ──
    entry = orb_high if break_dir == "long" else orb_low

    for h, l, _c in bars[break_idx:]:
        if break_dir == "long":
            if l <= orb_low:
                return -1.0, break_dir
            if h >= orb_high + risk * rr_target:
                return rr_target, break_dir
        else:
            if h >= orb_high:
                return -1.0, break_dir
            if l <= orb_low - risk * rr_target:
                return rr_target, break_dir

    # No resolution within window: pro-rate at last close
    last_close = bars[-1][2]
    if break_dir == "long":
        return (last_close - entry) / risk, break_dir
    else:
        return (entry - last_close) / risk, break_dir


# ═══════════════════════════════════════════════════════════════════════════
# BH FDR
# ═══════════════════════════════════════════════════════════════════════════


def benjamini_hochberg(p_values: list[float], q: float = 0.10) -> set[int]:
    """BH FDR correction. Returns indices of rejected (significant) hypotheses."""
    n = len(p_values)
    if n == 0:
        return set()
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    max_sig = -1
    for rank, (_, p) in enumerate(indexed):
        if p <= q * (rank + 1) / n:
            max_sig = rank
    if max_sig < 0:
        return set()
    return {orig_idx for _, (orig_idx, _) in enumerate(indexed[: max_sig + 1])}


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════


def compute_metrics(pnls_r: np.ndarray, pnls_dollars: np.ndarray) -> dict:
    """Compute strategy metrics from arrays of trade P&Ls."""
    n = len(pnls_r)
    if n == 0:
        return {}
    wr = float((pnls_r > 0).sum() / n)
    avg_r = float(pnls_r.mean())
    std_r = float(pnls_r.std())
    sharpe = avg_r / std_r if std_r > 0 else 0.0
    total_r = float(pnls_r.sum())
    total_dollars = float(pnls_dollars.sum())
    avg_dollars = float(pnls_dollars.mean())

    if n >= 3 and std_r > 0:
        _, p_value = ttest_1samp(pnls_r, 0)
        p_value = float(p_value)
    else:
        p_value = 1.0

    return {
        "n": n,
        "wr": wr,
        "avg_r": avg_r,
        "sharpe": sharpe,
        "total_r": total_r,
        "total_dollars": total_dollars,
        "avg_dollars": avg_dollars,
        "p_value": p_value,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("=" * 70)
    print("LONDON_ADJACENT HYPOTHESIS TEST")
    print("Testing ORB edge at the hour adjacent to LONDON_METALS")
    print("  Winter: 17:00 AEST (7AM London/GMT) -- pre-metals")
    print("  Summer: 18:00 AEST (9AM London/BST) -- post-metals/FTSE")
    print("=" * 70)
    print()

    # ── Collect all trade outcomes ──
    all_results: list[TradeResult] = []
    orb_diagnostics: list[dict] = []

    for instrument in ACTIVE_ORB_INSTRUMENTS:
        cost_spec = COST_SPECS[instrument]
        friction_dollars = cost_spec.commission_rt + cost_spec.spread_doubled + cost_spec.slippage

        trading_days_raw = con.execute(
            "SELECT DISTINCT trading_day FROM daily_features WHERE symbol = ? ORDER BY trading_day",
            [instrument],
        ).fetchall()
        trading_days = []
        for row in trading_days_raw:
            td = row[0]
            if isinstance(td, date) and not isinstance(td, datetime):
                trading_days.append(td)
            elif hasattr(td, "date"):
                trading_days.append(td.date())
            else:
                trading_days.append(date.fromisoformat(str(td)[:10]))

        print(f"{instrument}: {len(trading_days)} trading days ({trading_days[0]} to {trading_days[-1]})")

        inst_trades = 0
        for td_idx, trading_day in enumerate(trading_days):
            if td_idx % 500 == 0 and td_idx > 0:
                print(f"  {instrument}: {td_idx}/{len(trading_days)} days, {inst_trades} trades")

            h_adj, m_adj = get_adjacent_time_brisbane(trading_day)
            season = "summer" if is_uk_dst(trading_day) else "winter"

            for aperture in APERTURES:
                bris_start = datetime(
                    trading_day.year,
                    trading_day.month,
                    trading_day.day,
                    h_adj,
                    m_adj,
                    tzinfo=BRIS,
                )

                orb_high, orb_low, orb_vol = compute_orb(
                    con,
                    instrument,
                    bris_start,
                    aperture,
                )
                if orb_high is None or orb_low is None:
                    continue
                orb_range = orb_high - orb_low
                if orb_range <= 0:
                    continue

                bris_orb_end = bris_start + timedelta(minutes=aperture)
                post_bars = fetch_post_orb_bars(
                    con,
                    instrument,
                    bris_orb_end,
                    MAX_BARS_POST_ORB,
                )
                if not post_bars:
                    continue

                dbl_break = detect_double_break(post_bars, orb_high, orb_low)

                orb_diagnostics.append(
                    {
                        "instrument": instrument,
                        "aperture": aperture,
                        "orb_range": orb_range,
                        "orb_volume": orb_vol,
                        "double_break": dbl_break,
                        "season": season,
                    }
                )

                for entry_model, cb in ENTRY_SPECS:
                    for rr in RR_TARGETS:
                        pnl_r_gross, break_dir = simulate_trade(
                            post_bars,
                            orb_high,
                            orb_low,
                            rr,
                            entry_model,
                            cb,
                        )
                        if pnl_r_gross is None or break_dir is None:
                            continue

                        risk_dollars = orb_range * cost_spec.point_value
                        friction_r = friction_dollars / risk_dollars
                        pnl_r_net = pnl_r_gross - friction_r
                        pnl_dollars = pnl_r_net * risk_dollars

                        all_results.append(
                            TradeResult(
                                trading_day=trading_day,
                                instrument=instrument,
                                aperture=aperture,
                                entry_model=entry_model,
                                confirm_bars=cb,
                                rr_target=rr,
                                pnl_r_gross=pnl_r_gross,
                                pnl_r_net=pnl_r_net,
                                pnl_dollars=pnl_dollars,
                                orb_range=orb_range,
                                orb_volume=orb_vol,
                                break_dir=break_dir,
                                double_break=dbl_break,
                                season=season,
                            )
                        )
                        inst_trades += 1

        print(f"  {instrument}: DONE -- {inst_trades} trade outcomes")

    print(f"\nTotal trade outcomes: {len(all_results)}")

    # ── Structural Diagnostics ──
    print("\n" + "=" * 70)
    print("STRUCTURAL DIAGNOSTICS")
    print("=" * 70)

    diag_groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in orb_diagnostics:
        key = (d["instrument"], d["aperture"], d["season"])
        diag_groups[key].append(d)

    print(
        f"\n{'Instrument':<10} {'Apt':<5} {'Season':<8} {'Total':<7} "
        f"{'DblBrk':<7} {'Rate':<7} {'AvgVol':<8} {'AvgRng':<8}"
    )
    print("-" * 70)
    for key in sorted(diag_groups.keys()):
        items = diag_groups[key]
        total = len(items)
        dbl = sum(1 for d in items if d["double_break"])
        rate = dbl / total if total > 0 else 0
        avg_vol = np.mean([d["orb_volume"] for d in items])
        avg_range = np.mean([d["orb_range"] for d in items])
        print(
            f"{key[0]:<10} O{key[1]:<4} {key[2]:<8} {total:<7} {dbl:<7} {rate:<7.1%} {avg_vol:<8.0f} {avg_range:<8.2f}"
        )

    # Compare to LONDON_METALS 81% double-break
    all_dbl = sum(1 for d in orb_diagnostics if d["double_break"])
    all_total = len(orb_diagnostics)
    overall_dbl_rate = all_dbl / all_total if all_total > 0 else 0
    print(f"\nOverall adjacent double-break rate: {overall_dbl_rate:.1%} (vs LONDON_METALS ~81%)")

    # ── Grid Metrics + BH FDR ──
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)

    grid_cells: dict[tuple, list[TradeResult]] = defaultdict(list)
    for r in all_results:
        for filter_name, min_size in G_FILTER_THRESHOLDS.items():
            if r.orb_range >= min_size:
                key = (r.instrument, r.aperture, r.entry_model, r.confirm_bars, r.rr_target, filter_name)
                grid_cells[key].append(r)

    cell_metrics: list[tuple[tuple, dict]] = []
    for key, trades in sorted(grid_cells.items()):
        pnls_r = np.array([t.pnl_r_net for t in trades])
        pnls_dollars = np.array([t.pnl_dollars for t in trades])
        metrics = compute_metrics(pnls_r, pnls_dollars)
        if metrics and metrics["n"] >= 10:
            cell_metrics.append((key, metrics))

    all_p = [m["p_value"] for _, m in cell_metrics]
    significant = benjamini_hochberg(all_p, q=0.10)

    n_tested = len(cell_metrics)
    n_survivors = len(significant)
    print(f"\nTested {n_tested} grid cells (min N=10 for inclusion)")
    print(f"BH FDR survivors (q=0.10): {n_survivors}/{n_tested}")

    if n_survivors > 0:
        print(f"\n{'Cell':<50} {'N':>5} {'WR':>6} {'AvgR':>7} {'$Avg':>8} {'$Tot':>9} {'Shrp':>6} {'p':>8}")
        print("-" * 105)
        for i, (key, metrics) in enumerate(cell_metrics):
            if i in significant:
                inst, ap, em, cb, rr, filt = key
                label = f"{inst} O{ap} {em}_CB{cb} RR{rr} {filt}"
                m = metrics
                print(
                    f"{label:<50} {m['n']:>5} {m['wr']:>6.1%} "
                    f"{m['avg_r']:>+7.3f} ${m['avg_dollars']:>7.2f} "
                    f"${m['total_dollars']:>8.0f} {m['sharpe']:>6.3f} "
                    f"{m['p_value']:>8.4f}"
                )

    # Top 20 by avg $ (N>=30)
    print(f"\n--- Top 20 by avg $ (N>=30, for context) ---")
    top_cells = [(k, m) for k, m in cell_metrics if m["n"] >= 30]
    top_cells.sort(key=lambda x: x[1]["avg_dollars"], reverse=True)
    print(f"{'Cell':<50} {'N':>5} {'WR':>6} {'AvgR':>7} {'$Avg':>8} {'Shrp':>6} {'p':>8} {'BH':>3}")
    print("-" * 100)
    sig_keys = {cell_metrics[i][0] for i in significant} if significant else set()
    for key, metrics in top_cells[:20]:
        inst, ap, em, cb, rr, filt = key
        label = f"{inst} O{ap} {em}_CB{cb} RR{rr} {filt}"
        m = metrics
        bh = "Y" if key in sig_keys else ""
        print(
            f"{label:<50} {m['n']:>5} {m['wr']:>6.1%} "
            f"{m['avg_r']:>+7.3f} ${m['avg_dollars']:>7.2f} "
            f"{m['sharpe']:>6.3f} {m['p_value']:>8.4f} {bh:>3}"
        )

    # Seasonal decomposition for survivors
    if n_survivors > 0:
        print(f"\n--- Seasonal Decomposition (BH survivors) ---")
        print(f"{'Cell':<50} {'W_N':>5} {'W_AvgR':>7} {'S_N':>5} {'S_AvgR':>7}")
        print("-" * 80)
        for i, (key, _metrics) in enumerate(cell_metrics):
            if i in significant:
                trades = grid_cells[key]
                winter = [t.pnl_r_net for t in trades if t.season == "winter"]
                summer = [t.pnl_r_net for t in trades if t.season == "summer"]
                inst, ap, em, cb, rr, filt = key
                label = f"{inst} O{ap} {em}_CB{cb} RR{rr} {filt}"
                w_avg = np.mean(winter) if winter else float("nan")
                s_avg = np.mean(summer) if summer else float("nan")
                print(f"{label:<50} {len(winter):>5} {w_avg:>+7.3f} {len(summer):>5} {s_avg:>+7.3f}")

    # ── Save CSV ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "london_adjacent_grid.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "instrument",
                "aperture",
                "entry_model",
                "confirm_bars",
                "rr_target",
                "filter",
                "n",
                "win_rate",
                "avg_r_net",
                "total_r",
                "avg_dollars",
                "total_dollars",
                "sharpe",
                "p_value",
                "bh_significant",
            ]
        )
        for i, (key, metrics) in enumerate(cell_metrics):
            inst, ap, em, cb, rr, filt = key
            m = metrics
            writer.writerow(
                [
                    inst,
                    ap,
                    em,
                    cb,
                    rr,
                    filt,
                    m["n"],
                    f"{m['wr']:.4f}",
                    f"{m['avg_r']:.6f}",
                    f"{m['total_r']:.2f}",
                    f"{m['avg_dollars']:.2f}",
                    f"{m['total_dollars']:.2f}",
                    f"{m['sharpe']:.4f}",
                    f"{m['p_value']:.6f}",
                    "Y" if i in significant else "N",
                ]
            )
    print(f"\nGrid saved: {csv_path}")

    trades_csv = OUTPUT_DIR / "london_adjacent_trades.csv"
    with open(trades_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trading_day",
                "instrument",
                "aperture",
                "entry_model",
                "confirm_bars",
                "rr_target",
                "pnl_r_gross",
                "pnl_r_net",
                "pnl_dollars",
                "orb_range",
                "break_dir",
                "double_break",
                "season",
            ]
        )
        for r in all_results:
            writer.writerow(
                [
                    r.trading_day,
                    r.instrument,
                    r.aperture,
                    r.entry_model,
                    r.confirm_bars,
                    r.rr_target,
                    f"{r.pnl_r_gross:.6f}",
                    f"{r.pnl_r_net:.6f}",
                    f"{r.pnl_dollars:.2f}",
                    f"{r.orb_range:.2f}",
                    r.break_dir,
                    r.double_break,
                    r.season,
                ]
            )
    print(f"Trades saved: {trades_csv}")

    # ── Honest Summary ──
    print("\n" + "=" * 70)
    print("HONEST SUMMARY")
    print("=" * 70)

    print(f"\nSURVIVED SCRUTINY:")
    if n_survivors > 0:
        for i, (key, metrics) in enumerate(cell_metrics):
            if i in significant:
                inst, ap, em, cb, rr, filt = key
                m = metrics
                if m["n"] >= 100:
                    cls = "CORE"
                elif m["n"] >= 30:
                    cls = "REGIME"
                else:
                    cls = "INVALID"
                print(
                    f"  {inst} O{ap} {em}_CB{cb} RR{rr} {filt}: "
                    f"N={m['n']}, avgR={m['avg_r']:+.3f}, "
                    f"avg$={m['avg_dollars']:+.2f}, p={m['p_value']:.4f} "
                    f"[{cls}]"
                )
    else:
        print("  NONE -- zero cells survived BH FDR at q=0.10")

    print(f"\nDID NOT SURVIVE:")
    print(f"  {n_tested - n_survivors}/{n_tested} cells failed BH FDR")

    print(f"\nMECHANISM:")
    print(f"  Winter adjacent (7AM London): European pre-open positioning")
    print(f"  Summer adjacent (9AM London): Post-metals, FTSE flow")
    print(f"  Adjacent double-break rate: {overall_dbl_rate:.1%} (LONDON_METALS correct time: ~81%)")

    print(f"\nCAVEATS:")
    print(f"  - Adjacent slot = TWO different market events (7AM/9AM London)")
    print(f"  - Winter adjacent has ~50% of LONDON_METALS volume")
    print(f"  - Simplified entry (ORB-level, no E2 slippage differential)")
    print(f"  - {n_tested} cells tested -- BH correction is strict")
    print(f"  - IN-SAMPLE ONLY -- no walk-forward validation")

    print(f"\nNEXT STEPS:")
    if n_survivors > 0:
        print(f"  1. Decompose survivors by season -- which half drives the edge?")
        print(f"  2. Walk-forward validation on survivors")
        print(f"  3. If one season dominant -> test as fixed-event session")
        print(f"  4. If viable -> add LONDON_ADJACENT resolver to dst.py")
    else:
        print(f"  NO-GO. Adjacent slot has no BH FDR-surviving edge.")
        print(f"  The DST audit 'wrong > right' signal was structural")
        print(f"  (different trades, not better timing).")
        print(f"  Do not revisit without new entry model or approach.")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
