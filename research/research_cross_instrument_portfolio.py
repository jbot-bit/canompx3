#!/usr/bin/env python3
"""
Cross-Instrument Portfolio Analysis (MGC x MNQ x MES).

Read-only analysis: queries pre-computed orb_outcomes + daily_features to measure
correlation structure across instruments and simulate combined portfolios.

Key question: Are confirmed edges across MGC/MNQ/MES correlated (shared drawdowns)
or independent (free diversification)?

Four analysis sections:
  1. Same-Day Cross-Instrument Correlation (pnl_r Pearson, co-loss, co-win)
  2. Combined Portfolio Equity Curves (equal-weight, edge-weighted, MGC-only)
  3. Marginal Value of Each Instrument (delta Sharpe, delta MaxDD)
  4. Regime Analysis by Calendar Year (per-year stats + pairwise correlation)

Usage:
    python research/research_cross_instrument_portfolio.py
    python research/research_cross_instrument_portfolio.py --db-path C:/db/gold.db
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
from trading_app.config import ORB_DURATION_MINUTES
from research._alt_strategy_utils import compute_strategy_metrics

# =========================================================================
# Edge specifications (from TRADING_RULES.md confirmed edges)
# =========================================================================

INSTRUMENTS = ["MGC", "MNQ", "MES"]

# Each edge: (session, instrument, gate_label, min_size, direction_filter, orb_minutes)
# direction_filter: "long" = LONG-ONLY, None = BOTH directions
EDGES_1000 = [
    {"session": "1000", "instrument": "MGC", "gate": "G5+", "min_size": 5.0,
     "direction": "long", "orb_minutes": 5},
    {"session": "1000", "instrument": "MNQ", "gate": "G4+", "min_size": 4.0,
     "direction": "long", "orb_minutes": 5},
    {"session": "1000", "instrument": "MES", "gate": "G4+", "min_size": 4.0,
     "direction": "long", "orb_minutes": 5},
]

EDGES_0900 = [
    {"session": "0900", "instrument": "MGC", "gate": "G5+", "min_size": 5.0,
     "direction": None, "orb_minutes": 5},
    {"session": "0900", "instrument": "MNQ", "gate": "G4+", "min_size": 4.0,
     "direction": None, "orb_minutes": 5},
    {"session": "0900", "instrument": "MES", "gate": "G3+", "min_size": 3.0,
     "direction": None, "orb_minutes": 5},
]

# Fixed query parameters
ENTRY_MODEL = "E1"
CONFIRM_BARS = 2
RR_TARGET = 2.5


# =========================================================================
# Data Loading
# =========================================================================

def load_edge_data(con: duckdb.DuckDBPyConnection, edges: list[dict]) -> pd.DataFrame:
    """Load pnl_r for a set of edge specs, return wide DataFrame keyed on trading_day.

    Each instrument gets a column like 'MGC_pnl_r'. Days with no trade for an
    instrument have NaN in that column.
    """
    frames = {}
    for edge in edges:
        inst = edge["instrument"]
        session = edge["session"]
        orb_min = edge["orb_minutes"]
        min_size = edge["min_size"]
        direction = edge["direction"]

        size_col = f"orb_{session}_size"
        dir_col = f"orb_{session}_break_dir"

        params = [ENTRY_MODEL, CONFIRM_BARS, RR_TARGET, session, orb_min, inst, min_size]

        direction_clause = ""
        if direction is not None:
            direction_clause = f"AND df.{dir_col} = ?"
            params.append(direction)

        query = f"""
            SELECT oo.trading_day, oo.pnl_r
            FROM orb_outcomes oo
            JOIN daily_features df
              ON oo.symbol = df.symbol
              AND oo.trading_day = df.trading_day
              AND oo.orb_minutes = df.orb_minutes
            WHERE oo.entry_model = ?
              AND oo.confirm_bars = ?
              AND oo.rr_target = ?
              AND oo.orb_label = ?
              AND oo.orb_minutes = ?
              AND oo.symbol = ?
              AND oo.pnl_r IS NOT NULL
              AND df.{size_col} >= ?
              {direction_clause}
            ORDER BY oo.trading_day
        """
        df = con.execute(query, params).fetchdf()
        df = df.rename(columns={"pnl_r": f"{inst}_pnl_r"})
        frames[inst] = df.set_index("trading_day")

    # Merge all instruments on trading_day (outer join — keep all days)
    result = None
    for inst in INSTRUMENTS:
        if inst not in frames:
            continue
        if result is None:
            result = frames[inst]
        else:
            result = result.join(frames[inst], how="outer")

    if result is None:
        return pd.DataFrame()

    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# =========================================================================
# Section 1: Same-Day Cross-Instrument Correlation
# =========================================================================

def analyze_correlation(wide_df: pd.DataFrame, label: str) -> list[dict]:
    """Compute pairwise correlation stats on overlapping days."""
    results = []
    pnl_cols = [c for c in wide_df.columns if c.endswith("_pnl_r")]
    instruments = [c.replace("_pnl_r", "") for c in pnl_cols]

    for i, j in combinations(range(len(instruments)), 2):
        inst_a, inst_b = instruments[i], instruments[j]
        col_a, col_b = pnl_cols[i], pnl_cols[j]

        # Only days where BOTH instruments traded
        overlap = wide_df[[col_a, col_b]].dropna()
        n = len(overlap)

        if n < 15:
            results.append({
                "pair": f"{inst_a}/{inst_b}", "corr": None, "p_value": None,
                "co_loss_pct": None, "co_win_pct": None, "n_overlap": n,
                "note": "INSUFFICIENT (N < 15)",
            })
            continue

        a_vals = overlap[col_a].values
        b_vals = overlap[col_b].values

        corr, p_val = scipy_stats.pearsonr(a_vals, b_vals)
        co_loss = float(((a_vals < 0) & (b_vals < 0)).sum() / n)
        co_win = float(((a_vals > 0) & (b_vals > 0)).sum() / n)

        results.append({
            "pair": f"{inst_a}/{inst_b}", "corr": corr, "p_value": p_val,
            "co_loss_pct": co_loss, "co_win_pct": co_win, "n_overlap": n,
            "note": "",
        })

    return results


def print_correlation_section(results: list[dict], section_label: str):
    """Print Section 1 correlation table."""
    print(f"\n--- {section_label} ---")
    print(f"  {'Pair':<12} {'Corr':>8} {'p-value':>10} {'Co-Loss%':>10} {'Co-Win%':>10} {'N-overlap':>10}")

    for r in results:
        if r["corr"] is None:
            print(f"  {r['pair']:<12} {'---':>8} {'---':>10} {'---':>10} {'---':>10} {r['n_overlap']:>10}  {r['note']}")
        else:
            print(f"  {r['pair']:<12} {r['corr']:>+8.3f} {r['p_value']:>10.4f}"
                  f" {r['co_loss_pct']:>9.1%} {r['co_win_pct']:>9.1%} {r['n_overlap']:>10}")


# =========================================================================
# Section 2: Combined Portfolio Equity Curves
# =========================================================================

def build_portfolio_variants(wide_df: pd.DataFrame) -> dict[str, dict]:
    """Build portfolio variants and compute metrics for each.

    Returns dict of variant_name -> metrics dict (from compute_strategy_metrics).
    """
    pnl_cols = [c for c in wide_df.columns if c.endswith("_pnl_r")]
    instruments = [c.replace("_pnl_r", "") for c in pnl_cols]
    n_inst = len(instruments)

    if n_inst == 0:
        return {}

    # Date range for trades/year calculation
    date_range = wide_df.index
    years = max((date_range.max() - date_range.min()).days / 365.25, 0.1)

    variants = {}

    # --- MGC-only baseline ---
    mgc_col = "MGC_pnl_r"
    if mgc_col in wide_df.columns:
        mgc_series = wide_df[mgc_col].dropna().values
        m = compute_strategy_metrics(mgc_series)
        if m:
            m["trades_per_yr"] = m["n"] / years
            variants["MGC-only"] = m

    # --- Equal weight (1/N per instrument, per day) ---
    daily_pnl = []
    total_trades = 0
    for _, row in wide_df.iterrows():
        vals = [row[c] for c in pnl_cols if pd.notna(row[c])]
        if vals:
            daily_pnl.append(np.mean(vals))
            total_trades += len(vals)
        # Skip days where no instrument traded (don't append 0)

    if daily_pnl:
        eq_arr = np.array(daily_pnl)
        m = compute_strategy_metrics(eq_arr)
        if m:
            m["trades_per_yr"] = total_trades / years
            m["n_trade_events"] = total_trades
            variants["Equal-weight"] = m

    # --- Edge-weighted (proportional to per-instrument avgR) ---
    avg_rs = {}
    for col in pnl_cols:
        inst = col.replace("_pnl_r", "")
        vals = wide_df[col].dropna().values
        if len(vals) > 0:
            avg_rs[inst] = float(vals.mean())

    total_avgr = sum(max(v, 0) for v in avg_rs.values())
    if total_avgr > 0:
        weights = {inst: max(avg_rs.get(inst, 0), 0) / total_avgr for inst in instruments}
    else:
        weights = {inst: 1.0 / n_inst for inst in instruments}

    daily_pnl_ew = []
    total_trades_ew = 0
    for _, row in wide_df.iterrows():
        day_pnl = 0.0
        day_count = 0
        for col in pnl_cols:
            inst = col.replace("_pnl_r", "")
            if pd.notna(row[col]):
                day_pnl += row[col] * weights[inst] * n_inst  # scale so total weight sums to ~1
                day_count += 1
        if day_count > 0:
            daily_pnl_ew.append(day_pnl)
            total_trades_ew += day_count

    if daily_pnl_ew:
        ew_arr = np.array(daily_pnl_ew)
        m = compute_strategy_metrics(ew_arr)
        if m:
            m["trades_per_yr"] = total_trades_ew / years
            m["n_trade_events"] = total_trades_ew
            m["weights"] = weights
            variants["Edge-weighted"] = m

    return variants


def print_portfolio_section(variants: dict[str, dict], section_label: str):
    """Print Section 2 portfolio comparison table."""
    print(f"\n--- {section_label} ---")
    print(f"  {'Variant':<18} {'TotalR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades/yr':>10} {'WR':>6}")

    for name, m in variants.items():
        tpy = m.get("trades_per_yr", 0)
        print(f"  {name:<18} {m['total']:>+8.1f} {m['sharpe']:>8.3f} {m['maxdd']:>7.1f}R {tpy:>10.0f} {m['wr']:>5.0%}")

    # Print edge weights if available
    ew = variants.get("Edge-weighted")
    if ew and "weights" in ew:
        w = ew["weights"]
        parts = [f"{inst}={wt:.1%}" for inst, wt in sorted(w.items())]
        print(f"  Edge weights: {', '.join(parts)}")


# =========================================================================
# Section 3: Marginal Value of Each Instrument
# =========================================================================

def analyze_marginal_value(wide_df: pd.DataFrame) -> list[dict]:
    """Compute marginal Sharpe / MaxDD improvement of adding instruments to MGC."""
    pnl_cols = [c for c in wide_df.columns if c.endswith("_pnl_r")]
    instruments = [c.replace("_pnl_r", "") for c in pnl_cols]

    if "MGC" not in instruments:
        return []

    date_range = wide_df.index
    years = max((date_range.max() - date_range.min()).days / 365.25, 0.1)

    def portfolio_metrics(cols: list[str]) -> dict | None:
        daily = []
        for _, row in wide_df.iterrows():
            vals = [row[c] for c in cols if pd.notna(row[c])]
            if vals:
                daily.append(np.mean(vals))
        if not daily:
            return None
        m = compute_strategy_metrics(np.array(daily))
        if m:
            m["trades_per_yr"] = m["n"] / years
        return m

    mgc_only = portfolio_metrics(["MGC_pnl_r"])
    if mgc_only is None:
        return []

    results = []
    other_instruments = [i for i in instruments if i != "MGC"]

    # Each individual addition
    for inst in other_instruments:
        combined = portfolio_metrics(["MGC_pnl_r", f"{inst}_pnl_r"])
        if combined:
            results.append({
                "label": f"Adding {inst} to MGC",
                "delta_sharpe": combined["sharpe"] - mgc_only["sharpe"],
                "delta_maxdd": combined["maxdd"] - mgc_only["maxdd"],
                "combined_sharpe": combined["sharpe"],
                "combined_maxdd": combined["maxdd"],
                "flag": "WORSE" if combined["sharpe"] < mgc_only["sharpe"] else "",
            })

    # All three together
    if len(other_instruments) >= 1:
        all_cols = [f"{i}_pnl_r" for i in instruments]
        combined = portfolio_metrics(all_cols)
        if combined:
            results.append({
                "label": "All three",
                "delta_sharpe": combined["sharpe"] - mgc_only["sharpe"],
                "delta_maxdd": combined["maxdd"] - mgc_only["maxdd"],
                "combined_sharpe": combined["sharpe"],
                "combined_maxdd": combined["maxdd"],
                "flag": "WORSE" if combined["sharpe"] < mgc_only["sharpe"] else "",
            })

    return results


def print_marginal_section(results: list[dict], baseline_maxdd: float):
    """Print Section 3 marginal value table."""
    print("\n--- SECTION 3: MARGINAL VALUE ---")

    for r in results:
        dd_dir = "improved" if r["delta_maxdd"] > 0 else "worsened"
        flag = f"  ** {r['flag']} **" if r["flag"] else ""
        print(f"  {r['label']:<25} Sharpe {r['delta_sharpe']:>+.3f},"
              f" MaxDD {dd_dir} {abs(r['delta_maxdd']):.1f}R{flag}")


# =========================================================================
# Section 4: Regime Analysis (by Calendar Year)
# =========================================================================

def analyze_regimes(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Per-year stats and pairwise correlation."""
    pnl_cols = [c for c in wide_df.columns if c.endswith("_pnl_r")]
    instruments = [c.replace("_pnl_r", "") for c in pnl_cols]

    wide_df = wide_df.copy()
    wide_df["year"] = wide_df.index.year

    rows = []
    for year, group in wide_df.groupby("year"):
        row = {"year": year}

        # Per-instrument stats
        for inst in instruments:
            col = f"{inst}_pnl_r"
            vals = group[col].dropna().values
            n = len(vals)
            avg_r = float(vals.mean()) if n > 0 else None
            sharpe = float(vals.mean() / vals.std()) if n > 1 and vals.std() > 0 else None
            row[f"{inst}_n"] = n
            row[f"{inst}_avgR"] = avg_r
            row[f"{inst}_sharpe"] = sharpe

        # Pairwise correlation
        for i, j in combinations(range(len(instruments)), 2):
            inst_a, inst_b = instruments[i], instruments[j]
            col_a, col_b = pnl_cols[i], pnl_cols[j]
            overlap = group[[col_a, col_b]].dropna()
            n_overlap = len(overlap)
            pair_key = f"{inst_a}/{inst_b}_r"

            if n_overlap < 15:
                row[pair_key] = None
                row[f"{pair_key}_note"] = "INSUFF"
            else:
                corr, _ = scipy_stats.pearsonr(overlap[col_a].values, overlap[col_b].values)
                row[pair_key] = corr
                row[f"{pair_key}_note"] = "HIGH" if abs(corr) > 0.3 else ""

        rows.append(row)

    return pd.DataFrame(rows)


def print_regime_section(regime_df: pd.DataFrame, instruments: list[str]):
    """Print Section 4 regime table."""
    print("\n--- SECTION 4: REGIME ANALYSIS (BY YEAR) ---")

    # Header
    inst_headers = "  ".join(f"{i}(N,avgR)" for i in instruments)
    pair_headers = []
    for i, j in combinations(range(len(instruments)), 2):
        pair_headers.append(f"{instruments[i]}/{instruments[j]}-r")
    header = f"  {'Year':<6} {inst_headers}  {'  '.join(pair_headers)}"
    print(header)

    for _, row in regime_df.iterrows():
        parts = [f"  {int(row['year']):<6}"]

        for inst in instruments:
            n = row.get(f"{inst}_n", 0)
            avg_r = row.get(f"{inst}_avgR")
            if n == 0 or avg_r is None:
                parts.append(f"{'---':>14}")
            else:
                parts.append(f"({n},{avg_r:>+.2f})")

        for i, j in combinations(range(len(instruments)), 2):
            pair_key = f"{instruments[i]}/{instruments[j]}_r"
            corr = row.get(pair_key)
            note = row.get(f"{pair_key}_note", "")
            if corr is None or (isinstance(corr, float) and np.isnan(corr)):
                parts.append(f"{'INSUFF':>10}")
            else:
                flag = " !" if note == "HIGH" else ""
                parts.append(f"{corr:>+.2f}{flag}")

        print("  ".join(parts))


# =========================================================================
# Recommendations
# =========================================================================

def generate_recommendations(
    marginal_results: list[dict],
    corr_results_1000: list[dict],
    wide_1000: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    """Generate ADD / MONITOR / DO NOT ADD per instrument."""
    recommendations = {}
    recommendations["MGC"] = ("BASELINE", "Already deployed")

    pnl_cols = [c for c in wide_1000.columns if c.endswith("_pnl_r")]

    for inst in ["MNQ", "MES"]:
        col = f"{inst}_pnl_r"
        n_trades = int(wide_1000[col].dropna().shape[0]) if col in wide_1000.columns else 0

        # Find marginal result for this instrument
        marginal = None
        for m in marginal_results:
            if inst in m["label"] and "All" not in m["label"]:
                marginal = m
                break

        # Find correlation with MGC
        corr_val = None
        for c in corr_results_1000:
            if "MGC" in c["pair"] and inst in c["pair"] and c["corr"] is not None:
                corr_val = abs(c["corr"])
                break

        # Decision logic
        if n_trades < 30:
            recommendations[inst] = ("DO NOT ADD", f"N={n_trades} < 30 (INVALID sample)")
        elif marginal is None:
            recommendations[inst] = ("MONITOR", "No marginal data available")
        elif marginal["delta_sharpe"] <= 0:
            recommendations[inst] = ("DO NOT ADD",
                                     f"Marginal Sharpe {marginal['delta_sharpe']:+.3f} (no improvement)")
        elif corr_val is not None and corr_val > 0.4:
            recommendations[inst] = ("DO NOT ADD",
                                     f"Correlation {corr_val:.2f} > 0.4 (correlated drawdown risk)")
        elif n_trades < 100 or (corr_val is not None and corr_val > 0.2):
            reason_parts = []
            if n_trades < 100:
                reason_parts.append(f"N={n_trades} < 100")
            if corr_val is not None and corr_val > 0.2:
                reason_parts.append(f"corr={corr_val:.2f}")
            recommendations[inst] = ("MONITOR", "; ".join(reason_parts))
        else:
            recommendations[inst] = ("ADD",
                                     f"Sharpe +{marginal['delta_sharpe']:.3f}, corr={corr_val:.2f}, N={n_trades}")

    return recommendations


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Instrument Portfolio Analysis")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to gold.db (default: auto-detect)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        # Verify data availability
        counts = con.execute("""
            SELECT symbol, COUNT(*) as n
            FROM orb_outcomes
            WHERE entry_model = ? AND confirm_bars = ? AND rr_target = ?
              AND pnl_r IS NOT NULL
            GROUP BY symbol
            ORDER BY symbol
        """, [ENTRY_MODEL, CONFIRM_BARS, RR_TARGET]).fetchdf()

        if counts.empty:
            print("ERROR: No orb_outcomes data found for E1/CB2/RR2.5", file=sys.stderr)
            sys.exit(1)

        # Date range
        date_range = con.execute("""
            SELECT MIN(trading_day) as min_day, MAX(trading_day) as max_day
            FROM orb_outcomes
            WHERE entry_model = ? AND pnl_r IS NOT NULL
        """, [ENTRY_MODEL]).fetchdf()
        min_day = date_range["min_day"].iloc[0]
        max_day = date_range["max_day"].iloc[0]

        # Load data for both edge groups
        wide_1000 = load_edge_data(con, EDGES_1000)
        wide_0900 = load_edge_data(con, EDGES_0900)

        # Per-edge data check
        edge_counts = {}
        for edge in EDGES_1000 + EDGES_0900:
            inst = edge["instrument"]
            session = edge["session"]
            orb_min = edge["orb_minutes"]
            n = con.execute("""
                SELECT COUNT(*) FROM orb_outcomes
                WHERE entry_model = ? AND confirm_bars = ? AND rr_target = ?
                  AND orb_label = ? AND orb_minutes = ? AND symbol = ?
                  AND pnl_r IS NOT NULL
            """, [ENTRY_MODEL, CONFIRM_BARS, RR_TARGET, session, orb_min, inst]).fetchone()[0]
            edge_counts[(session, inst, orb_min)] = n

    finally:
        con.close()

    # ===================================================================
    # Header
    # ===================================================================
    print("=" * 64)
    print("CROSS-INSTRUMENT PORTFOLIO ANALYSIS")
    print("=" * 64)
    print(f"Instruments: {', '.join(INSTRUMENTS)}")
    print(f"Params: {ENTRY_MODEL}, CB{CONFIRM_BARS}, RR{RR_TARGET}")
    print(f"Period: {min_day} to {max_day}")
    print(f"\nData availability (E1/CB2/RR2.5, all sessions):")
    for _, row in counts.iterrows():
        print(f"  {row['symbol']}: {int(row['n']):,} outcomes")

    # Edge-specific data check
    print(f"\nPer-edge outcome counts:")
    for edge in EDGES_1000 + EDGES_0900:
        key = (edge["session"], edge["instrument"], edge["orb_minutes"])
        n = edge_counts.get(key, 0)
        dir_label = f" {edge['direction'].upper()}-ONLY" if edge["direction"] else " BOTH"
        status = "" if n > 0 else " <-- NO DATA (backfill needed?)"
        print(f"  {edge['session']} {edge['instrument']} {edge['gate']}"
              f" {edge['orb_minutes']}m{dir_label}: {n}{status}")

    # Friction reference
    print("\nFriction (embedded in pnl_r, for reference):")
    for inst in INSTRUMENTS:
        try:
            spec = get_cost_spec(inst)
            print(f"  {inst}: {spec.friction_in_points:.2f} pt"
                  f" (${spec.total_friction:.2f} RT)")
        except ValueError:
            pass

    # ===================================================================
    # Section 1: Correlation
    # ===================================================================
    has_1000_data = len(wide_1000) > 0 and any(
        wide_1000[c].notna().any() for c in wide_1000.columns if c.endswith("_pnl_r")
    ) if len(wide_1000.columns) > 0 else False

    if has_1000_data:
        corr_1000 = analyze_correlation(wide_1000, "1000 LONG-ONLY")
        print_correlation_section(corr_1000, "SECTION 1: SAME-DAY CORRELATION (1000 LONG-ONLY)")
    else:
        corr_1000 = []
        print("\n--- SECTION 1: SAME-DAY CORRELATION (1000 LONG-ONLY) ---")
        print("  NO DATA: 1000 session 15m outcomes not yet backfilled.")
        print("  Run: python trading_app/outcome_builder.py --instrument <INST>")

    corr_0900 = analyze_correlation(wide_0900, "0900 BIDIRECTIONAL")
    print_correlation_section(corr_0900, "SECTION 1B: SAME-DAY CORRELATION (0900 BIDIRECTIONAL)")

    # ===================================================================
    # Section 2: Portfolio Equity Curves (1000 LONG-ONLY)
    # ===================================================================
    if has_1000_data:
        variants = build_portfolio_variants(wide_1000)
        print_portfolio_section(variants, "SECTION 2: PORTFOLIO EQUITY CURVES (1000 LONG-ONLY)")
    else:
        variants = {}
        print("\n--- SECTION 2: PORTFOLIO EQUITY CURVES (1000 LONG-ONLY) ---")
        print("  NO DATA: skipped (see Section 1 note)")

    # ===================================================================
    # Section 3: Marginal Value
    # ===================================================================
    if has_1000_data:
        marginal = analyze_marginal_value(wide_1000)
        baseline_maxdd = variants.get("MGC-only", {}).get("maxdd", 0)
        print_marginal_section(marginal, baseline_maxdd)
    else:
        marginal = []
        print("\n--- SECTION 3: MARGINAL VALUE ---")
        print("  NO DATA: skipped (see Section 1 note)")

    # ===================================================================
    # Section 4: Regime Analysis
    # ===================================================================
    if has_1000_data:
        active_instruments = [c.replace("_pnl_r", "")
                              for c in wide_1000.columns if c.endswith("_pnl_r")]
        regime_df = analyze_regimes(wide_1000)
        print_regime_section(regime_df, active_instruments)
    else:
        active_instruments = []
        print("\n--- SECTION 4: REGIME ANALYSIS (BY YEAR) ---")
        print("  NO DATA: skipped (see Section 1 note)")

    # ===================================================================
    # Recommendations
    # ===================================================================
    recs = generate_recommendations(marginal, corr_1000, wide_1000)

    print("\n" + "=" * 64)
    print("RECOMMENDATIONS")
    print("=" * 64)

    for inst in INSTRUMENTS:
        if inst in recs:
            action, reason = recs[inst]
            print(f"  {inst}: {action} -- {reason}")

    # Caveats
    print("\nCAVEATS:")
    for inst in active_instruments:
        col = f"{inst}_pnl_r"
        if col in wide_1000.columns:
            n = int(wide_1000[col].dropna().shape[0])
            flag = " (< 100 trades)" if n < 100 else ""
            print(f"  - {inst}: N={n} trades (1000 LONG-ONLY){flag}")

    years = (wide_1000.index.max() - wide_1000.index.min()).days / 365.25 if len(wide_1000) > 0 else 0
    print(f"  - Period: {years:.1f} years")
    print(f"  - pnl_r includes friction (no double-counting)")
    print(f"  - This is NOT walk-forward -- full-period stats only")
    print("=" * 64)


if __name__ == "__main__":
    main()
