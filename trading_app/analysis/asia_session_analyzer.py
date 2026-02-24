"""
Deep CME_REOPEN Asia Session Analysis + Cross-Session Intelligence.

Standalone research script -- queries gold.db read-only and outputs
a structured report to stdout + artifacts/asia_session_analysis_2025.txt.

Sections:
  A) 0900 2025 Regime Deep Dive
  B) Cross-Session Cascade (0900 -> 1000 -> 1100)
  C) MFE/MAE Path Analysis on 0900 Losses
  D) Direction-Filtered Grid
  E) 1000/1100 as Reversal Trades

Usage:
    python trading_app/analysis/asia_session_analyzer.py
    python trading_app/analysis/asia_session_analyzer.py --year 2025
"""

import sys
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INSTRUMENT = "MGC"
DEFAULT_YEAR = 2025
ORB_MINUTES = 5  # baseline 5m ORB
COST_SPEC = get_cost_spec(INSTRUMENT)

OUTPUT_DIR = PROJECT_ROOT / "artifacts"

def _connect(db_path: Path | None = None) -> "duckdb.DuckDBPyConnection":
    p = db_path or GOLD_DB_PATH
    return duckdb.connect(str(p), read_only=True)

def _fmt(val, decimals=2):
    if val is None:
        return "N/A"
    return f"{val:+.{decimals}f}" if isinstance(val, float) else str(val)

def _pct(val, decimals=1):
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"

# =========================================================================
# SECTION A: CME_REOPEN 2025 Regime Deep Dive
# =========================================================================
def section_a_regime_deep_dive(con, year: int) -> list[str]:
    lines = []
    lines.append("=" * 72)
    lines.append(f"SECTION A: CME_REOPEN {year} Regime Deep Dive")
    lines.append("=" * 72)

    # A1: Top regime strategies for 0900
    lines.append(f"\nA1) Top 0900 regime_strategies ({year})")
    lines.append("-" * 60)
    try:
        rows = con.execute("""
            SELECT strategy_id, entry_model, confirm_bars, rr_target,
                   filter_type, sample_size, win_rate,
                   expectancy_r, sharpe_ratio, max_drawdown_r
            FROM regime_strategies
            WHERE orb_label = 'CME_REOPEN'
              AND expectancy_r IS NOT NULL
              AND sample_size >= 20
            ORDER BY expectancy_r DESC
            LIMIT 20
        """).fetchall()
        lines.append(f"{'ID':<32} {'EM':>3} {'CB':>3} {'RR':>4} {'Filter':<16} "
                     f"{'N':>4} {'WR':>6} {'ExpR':>7} {'Sharpe':>7} {'MaxDD':>7}")
        for r in rows:
            sid, em, cb, rr, ft, n, wr, expr, sh, mdd = r
            lines.append(
                f"{sid:<32} {em:>3} {cb:>3} {rr:>4.1f} {(ft or 'NO_FILTER'):<16} "
                f"{n:>4} {_pct(wr):>6} {_fmt(expr):>7} "
                f"{_fmt(sh):>7} {_fmt(mdd):>7}"
            )
    except duckdb.CatalogException:
        lines.append("  [regime_strategies table not found -- run regime discovery first]")

    # A2: Direction split for 0900 E1 CB2 RR2.5 G4+ (year vs full)
    lines.append(f"\nA2) 0900 Direction Split -- E1 CB2 RR2.5 G4+ ({year} vs Full)")
    lines.append("-" * 60)
    for label, date_filter in [(str(year), f"AND o.trading_day >= '{year}-01-01' AND o.trading_day <= '{year}-12-31'"),
                                ("Full", "")]:
        for direction in ["LONG", "SHORT"]:
            outcomes = con.execute(f"""
                SELECT o.trading_day, o.outcome, o.pnl_r, o.entry_price,
                       o.stop_price, o.mfe_r, o.mae_r
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
                  AND d.orb_minutes = {ORB_MINUTES}
                WHERE o.symbol = '{INSTRUMENT}'
                  AND o.orb_label = 'CME_REOPEN'
                  AND o.entry_model = 'E1'
                  AND o.confirm_bars = 2
                  AND o.rr_target = 2.5
                  AND d.orb_CME_REOPEN_size >= 4.0
                  AND UPPER(d.orb_CME_REOPEN_break_dir) = '{direction}'
                  {date_filter}
                ORDER BY o.trading_day
            """).fetchall()
            n = len(outcomes)
            if n > 0:
                wins = sum(1 for o in outcomes if o[1] == "win")
                total_r = sum(o[2] for o in outcomes if o[2] is not None)
                avg_r = total_r / n
                lines.append(f"  {label:>5} {direction:<6}: N={n:>4}, "
                             f"WR={_pct(wins / n)}, AvgR={_fmt(avg_r)}, "
                             f"TotalR={_fmt(total_r)}")
            else:
                lines.append(f"  {label:>5} {direction:<6}: N=0")

    # A3: ORB size distribution comparison
    lines.append(f"\nA3) 0900 ORB Size Distribution ({year} vs 2022-2024)")
    lines.append("-" * 60)
    buckets = [(0, 2, "<2"), (2, 4, "2-4"), (4, 6, "4-6"),
               (6, 8, "6-8"), (8, 10, "8-10"), (10, 999, "10+")]
    for label, date_filter in [(str(year), f"AND trading_day >= '{year}-01-01' AND trading_day <= '{year}-12-31'"),
                                ("2022-24", "AND trading_day >= '2022-01-01' AND trading_day <= '2024-12-31'")]:
        counts = []
        for lo, hi, bkt in buckets:
            row = con.execute(f"""
                SELECT COUNT(*) FROM daily_features
                WHERE symbol = '{INSTRUMENT}' AND orb_minutes = {ORB_MINUTES}
                  AND orb_CME_REOPEN_size >= {lo} AND orb_CME_REOPEN_size < {hi}
                  {date_filter}
            """).fetchone()
            counts.append((bkt, row[0]))
        total = sum(c for _, c in counts)
        dist = "  ".join(f"{bkt}:{cnt}({cnt*100//max(total,1)}%)" for bkt, cnt in counts)
        lines.append(f"  {label:>7}: {dist}  [total={total}]")

    return lines

# =========================================================================
# SECTION B: Cross-Session Cascade (0900 -> 1000 -> 1100)
# =========================================================================
def section_b_cross_session(con, year: int) -> list[str]:
    lines = []
    lines.append("\n" + "=" * 72)
    lines.append(f"SECTION B: Cross-Session Cascade 0900 -> 1000 -> 1100 ({year})")
    lines.append("=" * 72)

    date_filter = f"AND o9.trading_day >= '{year}-01-01' AND o9.trading_day <= '{year}-12-31'" if year else ""

    # B1: Conditional WR: P(1000 outcome | 0900 outcome)
    lines.append("\nB1) Conditional WR: P(1000 outcome | 0900 outcome)")
    lines.append("-" * 60)
    cascade = con.execute(f"""
        SELECT
            o9.outcome AS o900_outcome,
            o10.outcome AS o1000_outcome,
            CASE WHEN UPPER(d.orb_CME_REOPEN_break_dir) = UPPER(d.orb_TOKYO_OPEN_break_dir)
                 THEN 'same' ELSE 'opposite' END AS dir_rel,
            COUNT(*) AS cnt
        FROM orb_outcomes o9
        JOIN orb_outcomes o10
          ON o9.trading_day = o10.trading_day AND o10.symbol = o9.symbol
          AND o10.orb_label = 'TOKYO_OPEN' AND o10.entry_model = 'E1'
          AND o10.confirm_bars = 2 AND o10.rr_target = 2.5
        JOIN daily_features d
          ON o9.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
          AND d.orb_minutes = {ORB_MINUTES}
        WHERE o9.symbol = '{INSTRUMENT}'
          AND o9.orb_label = 'CME_REOPEN'
          AND o9.entry_model = 'E1'
          AND o9.confirm_bars = 2
          AND o9.rr_target = 2.5
          AND d.orb_CME_REOPEN_size >= 4.0
          AND d.orb_TOKYO_OPEN_size >= 4.0
          {date_filter}
        GROUP BY o9.outcome, o10.outcome, dir_rel
        ORDER BY o9.outcome, o10.outcome, dir_rel
    """).fetchall()
    lines.append(f"  {'0900':>8} {'1000':>8} {'DirRel':>10} {'Count':>6}")
    for r in cascade:
        o900 = r[0] or "N/A"
        o1000 = r[1] or "N/A"
        drel = r[2] or "N/A"
        lines.append(f"  {o900:>8} {o1000:>8} {drel:>10} {r[3]:>6}")

    # B2: Reversal signal: 0900 LOSS + 1000 opposite-dir G4+
    lines.append("\nB2) Reversal Signal: 0900 LOSS + 1000 Opposite-Dir G4+")
    lines.append("-" * 60)
    for period_label, period_filter in [(str(year), date_filter), ("Full", "")]:
        rev = con.execute(f"""
            SELECT o10.outcome, COUNT(*) AS cnt
            FROM orb_outcomes o9
            JOIN orb_outcomes o10
              ON o9.trading_day = o10.trading_day AND o10.symbol = o9.symbol
              AND o10.orb_label = 'TOKYO_OPEN' AND o10.entry_model = 'E1'
              AND o10.confirm_bars = 2 AND o10.rr_target = 2.5
            JOIN daily_features d
              ON o9.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
              AND d.orb_minutes = {ORB_MINUTES}
            WHERE o9.symbol = '{INSTRUMENT}'
              AND o9.orb_label = 'CME_REOPEN'
              AND o9.entry_model = 'E1'
              AND o9.confirm_bars = 2
              AND o9.rr_target = 2.5
              AND o9.outcome = 'loss'
              AND UPPER(d.orb_CME_REOPEN_break_dir) != UPPER(d.orb_TOKYO_OPEN_break_dir)
              AND d.orb_CME_REOPEN_size >= 4.0
              AND d.orb_TOKYO_OPEN_size >= 4.0
              {period_filter}
            GROUP BY o10.outcome
        """).fetchall()
        total = sum(r[1] for r in rev)
        wins = sum(r[1] for r in rev if r[0] == "win")
        wr = wins / total if total > 0 else None
        lines.append(f"  {period_label:>7}: N={total}, Wins={wins}, WR={_pct(wr)}")

    # B3: Chop detector: 0900+1000 double loss -> 1100 outcome
    lines.append("\nB3) Chop Detector: 0900+1000 Double Loss -> 1100 Outcome")
    lines.append("-" * 60)
    chop = con.execute(f"""
        SELECT o11.outcome, COUNT(*) AS cnt
        FROM orb_outcomes o9
        JOIN orb_outcomes o10
          ON o9.trading_day = o10.trading_day AND o10.symbol = o9.symbol
          AND o10.orb_label = 'TOKYO_OPEN' AND o10.entry_model = 'E1'
          AND o10.confirm_bars = 2 AND o10.rr_target = 2.5
        JOIN orb_outcomes o11
          ON o9.trading_day = o11.trading_day AND o11.symbol = o9.symbol
          AND o11.orb_label = 'SINGAPORE_OPEN' AND o11.entry_model = 'E1'
          AND o11.confirm_bars = 2 AND o11.rr_target = 2.5
        JOIN daily_features d
          ON o9.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
          AND d.orb_minutes = {ORB_MINUTES}
        WHERE o9.symbol = '{INSTRUMENT}'
          AND o9.orb_label = 'CME_REOPEN'
          AND o9.entry_model = 'E1'
          AND o9.confirm_bars = 2
          AND o9.rr_target = 2.5
          AND o9.outcome = 'loss'
          AND o10.outcome = 'loss'
          AND d.orb_CME_REOPEN_size >= 4.0
          AND d.orb_TOKYO_OPEN_size >= 4.0
          {date_filter}
        GROUP BY o11.outcome
    """).fetchall()
    total = sum(r[1] for r in chop)
    for r in chop:
        lines.append(f"  1100 {r[0]}: {r[1]} ({r[1]*100//max(total,1)}%)")
    lines.append(f"  Total double-loss days: {total}")

    return lines

# =========================================================================
# SECTION C: MFE/MAE Path Analysis on 0900 Losses
# =========================================================================
def section_c_mfe_analysis(con, year: int) -> list[str]:
    lines = []
    lines.append("\n" + "=" * 72)
    lines.append(f"SECTION C: MFE/MAE Path Analysis on 0900 Losses ({year})")
    lines.append("=" * 72)

    date_filter = f"AND o.trading_day >= '{year}-01-01' AND o.trading_day <= '{year}-12-31'" if year else ""

    # C1: MFE distribution on losses by RR target
    lines.append("\nC1) MFE Distribution on 0900 Losses (E1 CB2 G4+)")
    lines.append("-" * 60)
    for rr in [1.5, 2.0, 2.5, 3.0]:
        mfe_vals = con.execute(f"""
            SELECT o.mfe_r
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
              AND d.orb_minutes = {ORB_MINUTES}
            WHERE o.symbol = '{INSTRUMENT}'
              AND o.orb_label = 'CME_REOPEN'
              AND o.entry_model = 'E1'
              AND o.confirm_bars = 2
              AND o.rr_target = {rr}
              AND o.outcome = 'loss'
              AND d.orb_CME_REOPEN_size >= 4.0
              AND o.mfe_r IS NOT NULL
              {date_filter}
        """).fetchall()
        vals = [v[0] for v in mfe_vals]
        if vals:
            p25 = sorted(vals)[len(vals) // 4]
            p50 = statistics.median(vals)
            p75 = sorted(vals)[3 * len(vals) // 4]
            near_miss = sum(1 for v in vals if v >= 1.0)
            lines.append(f"  RR{rr}: N={len(vals)}, P25={_fmt(p25)}, "
                         f"Med={_fmt(p50)}, P75={_fmt(p75)}, "
                         f">=1.0R: {near_miss} ({near_miss*100//len(vals)}%)")
        else:
            lines.append(f"  RR{rr}: N=0")

    # C2: LONG vs SHORT loss MFE profiles
    lines.append("\nC2) LONG vs SHORT Loss MFE Profiles (E1 CB2 RR2.5 G4+)")
    lines.append("-" * 60)
    for direction in ["LONG", "SHORT"]:
        mfe_vals = con.execute(f"""
            SELECT o.mfe_r
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
              AND d.orb_minutes = {ORB_MINUTES}
            WHERE o.symbol = '{INSTRUMENT}'
              AND o.orb_label = 'CME_REOPEN'
              AND o.entry_model = 'E1'
              AND o.confirm_bars = 2
              AND o.rr_target = 2.5
              AND o.outcome = 'loss'
              AND d.orb_CME_REOPEN_size >= 4.0
              AND UPPER(d.orb_CME_REOPEN_break_dir) = '{direction}'
              AND o.mfe_r IS NOT NULL
              {date_filter}
        """).fetchall()
        vals = [v[0] for v in mfe_vals]
        if vals:
            med = statistics.median(vals)
            near_miss = sum(1 for v in vals if v >= 1.0)
            be_sim = sum(1 for v in vals if v >= 0.5)
            lines.append(f"  {direction:<6}: N={len(vals)}, MedMFE={_fmt(med)}, "
                         f">=1.0R: {near_miss} ({near_miss*100//len(vals)}%), "
                         f">=0.5R (BE trail): {be_sim} ({be_sim*100//len(vals)}%)")
        else:
            lines.append(f"  {direction:<6}: N=0")

    # C3: Trailing stop simulation: if stop moved to BE at +0.5R
    lines.append("\nC3) Trailing Stop Simulation: Move to BE at +0.5R MFE")
    lines.append("-" * 60)
    lines.append("  (Losses that would convert to scratch if stop moved to BE at +0.5R MFE)")
    for rr in [2.0, 2.5, 3.0]:
        loss_rows = con.execute(f"""
            SELECT o.mfe_r, o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
              AND d.orb_minutes = {ORB_MINUTES}
            WHERE o.symbol = '{INSTRUMENT}'
              AND o.orb_label = 'CME_REOPEN'
              AND o.entry_model = 'E1'
              AND o.confirm_bars = 2
              AND o.rr_target = {rr}
              AND o.outcome = 'loss'
              AND d.orb_CME_REOPEN_size >= 4.0
              AND o.mfe_r IS NOT NULL
              {date_filter}
        """).fetchall()
        if loss_rows:
            total_losses = len(loss_rows)
            # Losses that reached +0.5R MFE would become scratches (0R)
            converted = sum(1 for r in loss_rows if r[0] >= 0.5)
            saved_r = sum(abs(r[1]) for r in loss_rows if r[0] >= 0.5)
            lines.append(f"  RR{rr}: {converted}/{total_losses} losses convert "
                         f"({converted*100//total_losses}%), "
                         f"saves {_fmt(saved_r)}R total")
        else:
            lines.append(f"  RR{rr}: no losses found")

    return lines

# =========================================================================
# SECTION D: Direction-Filtered Grid
# =========================================================================
def section_d_direction_grid(con, year: int) -> list[str]:
    lines = []
    lines.append("\n" + "=" * 72)
    lines.append(f"SECTION D: Direction-Filtered Grid -- 0900 ({year})")
    lines.append("=" * 72)

    date_filter = f"AND o.trading_day >= '{year}-01-01' AND o.trading_day <= '{year}-12-31'" if year else ""

    lines.append("\nD1) LONG-only vs Bidirectional for 0900 (E1 CB2, G4+/G6+)")
    lines.append("-" * 60)
    lines.append(f"  {'RR':>4} {'Filter':<8} {'Dir':<8} {'N':>4} {'WR':>6} "
                 f"{'ExpR':>7} {'TotalR':>8} {'Sharpe':>7}")

    for rr in [2.0, 2.5, 3.0]:
        for min_size, flabel in [(4.0, "G4+"), (6.0, "G6+")]:
            for direction, dlabel in [(None, "both"), ("LONG", "LONG"), ("SHORT", "SHORT")]:
                dir_clause = f"AND UPPER(d.orb_CME_REOPEN_break_dir) = '{direction}'" if direction else ""
                outcomes = con.execute(f"""
                    SELECT o.trading_day, o.outcome, o.pnl_r,
                           o.entry_price, o.stop_price
                    FROM orb_outcomes o
                    JOIN daily_features d
                      ON o.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
                      AND d.orb_minutes = {ORB_MINUTES}
                    WHERE o.symbol = '{INSTRUMENT}'
                      AND o.orb_label = 'CME_REOPEN'
                      AND o.entry_model = 'E1'
                      AND o.confirm_bars = 2
                      AND o.rr_target = {rr}
                      AND d.orb_CME_REOPEN_size >= {min_size}
                      {dir_clause}
                      {date_filter}
                    ORDER BY o.trading_day
                """).fetchall()
                n = len(outcomes)
                if n > 0:
                    wins = sum(1 for o in outcomes if o[1] == "win")
                    total_r = sum(o[2] for o in outcomes if o[2] is not None)
                    wr = wins / n
                    avg_r = total_r / n
                    # Sharpe
                    r_vals = [o[2] for o in outcomes if o[2] is not None]
                    if len(r_vals) > 1:
                        mean_r = sum(r_vals) / len(r_vals)
                        var = sum((r - mean_r) ** 2 for r in r_vals) / (len(r_vals) - 1)
                        sharpe = mean_r / (var ** 0.5) if var > 0 else None
                    else:
                        sharpe = None
                    lines.append(f"  {rr:>4.1f} {flabel:<8} {dlabel:<8} {n:>4} "
                                 f"{_pct(wr):>6} {_fmt(avg_r):>7} {_fmt(total_r):>8} "
                                 f"{_fmt(sharpe):>7}")
                else:
                    lines.append(f"  {rr:>4.1f} {flabel:<8} {dlabel:<8}    0")

    # D2: E3 retrace LONG-only (strongest single edge found)
    lines.append(f"\nD2) E3 Retrace LONG-only -- 0900 G6+ ({year})")
    lines.append("-" * 60)
    lines.append(f"  {'RR':>4} {'N':>4} {'WR':>6} {'ExpR':>7} {'TotalR':>8}")
    for rr in [1.5, 2.0, 2.5, 3.0, 4.0]:
        outcomes = con.execute(f"""
            SELECT o.outcome, o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
              AND d.orb_minutes = {ORB_MINUTES}
            WHERE o.symbol = '{INSTRUMENT}'
              AND o.orb_label = 'CME_REOPEN'
              AND o.entry_model = 'E3'
              AND o.confirm_bars = 2
              AND o.rr_target = {rr}
              AND d.orb_CME_REOPEN_size >= 6.0
              AND UPPER(d.orb_CME_REOPEN_break_dir) = 'LONG'
              {date_filter}
        """).fetchall()
        n = len(outcomes)
        if n > 0:
            wins = sum(1 for o in outcomes if o[0] == "win")
            total_r = sum(o[1] for o in outcomes if o[1] is not None)
            avg_r = total_r / n
            lines.append(f"  {rr:>4.1f} {n:>4} {_pct(wins/n):>6} "
                         f"{_fmt(avg_r):>7} {_fmt(total_r):>8}")
        else:
            lines.append(f"  {rr:>4.1f}    0")

    return lines

# =========================================================================
# SECTION E: 1000/1100 as Reversal Trades
# =========================================================================
def section_e_reversal_trades(con, year: int) -> list[str]:
    lines = []
    lines.append("\n" + "=" * 72)
    lines.append(f"SECTION E: 1000/1100 as Reversal Trades ({year})")
    lines.append("=" * 72)

    date_filter = f"AND o9.trading_day >= '{year}-01-01' AND o9.trading_day <= '{year}-12-31'" if year else ""

    # E1: 1000 ORB performance conditioned on 0900 outcome
    lines.append("\nE1) 1000 Performance Conditioned on 0900 Outcome (E1 CB2 RR2.5 G4+)")
    lines.append("-" * 60)
    for o900_cond in ["win", "loss"]:
        for rr_1000 in [2.0, 2.5, 3.0]:
            cond_rows = con.execute(f"""
                SELECT o10.outcome, o10.pnl_r
                FROM orb_outcomes o9
                JOIN orb_outcomes o10
                  ON o9.trading_day = o10.trading_day AND o10.symbol = o9.symbol
                  AND o10.orb_label = 'TOKYO_OPEN' AND o10.entry_model = 'E1'
                  AND o10.confirm_bars = 2 AND o10.rr_target = {rr_1000}
                JOIN daily_features d
                  ON o9.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
                  AND d.orb_minutes = {ORB_MINUTES}
                WHERE o9.symbol = '{INSTRUMENT}'
                  AND o9.orb_label = 'CME_REOPEN'
                  AND o9.entry_model = 'E1'
                  AND o9.confirm_bars = 2
                  AND o9.rr_target = 2.5
                  AND o9.outcome = '{o900_cond}'
                  AND d.orb_CME_REOPEN_size >= 4.0
                  AND d.orb_TOKYO_OPEN_size >= 4.0
                  {date_filter}
            """).fetchall()
            n = len(cond_rows)
            if n > 0:
                wins = sum(1 for r in cond_rows if r[0] == "win")
                total_r = sum(r[1] for r in cond_rows if r[1] is not None)
                lines.append(f"  0900={o900_cond:<5} 1000 RR{rr_1000}: N={n:>3}, "
                             f"WR={_pct(wins/n)}, TotalR={_fmt(total_r)}")
            else:
                lines.append(f"  0900={o900_cond:<5} 1000 RR{rr_1000}: N=0")

    # E2: Reversal spec with direction constraint
    lines.append("\nE2) Full Reversal Spec: 0900 Loss + 1000 Opposite-Dir")
    lines.append("-" * 60)
    lines.append("  Scanning RR x CB combinations for 1000 opposite-dir after 0900 loss...")
    lines.append(f"  {'RR':>4} {'CB':>3} {'N':>4} {'WR':>6} {'ExpR':>7} {'TotalR':>8}")
    for rr in [2.0, 2.5, 3.0]:
        for cb in [1, 2, 3]:
            rev_rows = con.execute(f"""
                SELECT o10.outcome, o10.pnl_r, o10.entry_price, o10.stop_price
                FROM orb_outcomes o9
                JOIN orb_outcomes o10
                  ON o9.trading_day = o10.trading_day AND o10.symbol = o9.symbol
                  AND o10.orb_label = 'TOKYO_OPEN' AND o10.entry_model = 'E1'
                  AND o10.confirm_bars = {cb} AND o10.rr_target = {rr}
                JOIN daily_features d
                  ON o9.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
                  AND d.orb_minutes = {ORB_MINUTES}
                WHERE o9.symbol = '{INSTRUMENT}'
                  AND o9.orb_label = 'CME_REOPEN'
                  AND o9.entry_model = 'E1'
                  AND o9.confirm_bars = 2
                  AND o9.rr_target = 2.5
                  AND o9.outcome = 'loss'
                  AND UPPER(d.orb_CME_REOPEN_break_dir) != UPPER(d.orb_TOKYO_OPEN_break_dir)
                  AND d.orb_CME_REOPEN_size >= 4.0
                  AND d.orb_TOKYO_OPEN_size >= 4.0
                  {date_filter}
            """).fetchall()
            n = len(rev_rows)
            if n > 0:
                wins = sum(1 for r in rev_rows if r[0] == "win")
                total_r = sum(r[1] for r in rev_rows if r[1] is not None)
                avg_r = total_r / n
                lines.append(f"  {rr:>4.1f} {cb:>3} {n:>4} "
                             f"{_pct(wins/n):>6} {_fmt(avg_r):>7} {_fmt(total_r):>8}")

    # E3: 1100 as chop-cleared signal
    lines.append("\nE3) 1100 After Double 0900+1000 Loss (Chop-Cleared Signal)")
    lines.append("-" * 60)
    for period_label, pf in [(str(year), date_filter), ("Full", "")]:
        for rr in [2.0, 2.5]:
            chop_rows = con.execute(f"""
                SELECT o11.outcome, o11.pnl_r
                FROM orb_outcomes o9
                JOIN orb_outcomes o10
                  ON o9.trading_day = o10.trading_day AND o10.symbol = o9.symbol
                  AND o10.orb_label = 'TOKYO_OPEN' AND o10.entry_model = 'E1'
                  AND o10.confirm_bars = 2 AND o10.rr_target = 2.5
                JOIN orb_outcomes o11
                  ON o9.trading_day = o11.trading_day AND o11.symbol = o9.symbol
                  AND o11.orb_label = 'SINGAPORE_OPEN' AND o11.entry_model = 'E1'
                  AND o11.confirm_bars = 2 AND o11.rr_target = {rr}
                JOIN daily_features d
                  ON o9.trading_day = d.trading_day AND d.symbol = '{INSTRUMENT}'
                  AND d.orb_minutes = {ORB_MINUTES}
                WHERE o9.symbol = '{INSTRUMENT}'
                  AND o9.orb_label = 'CME_REOPEN'
                  AND o9.entry_model = 'E1'
                  AND o9.confirm_bars = 2
                  AND o9.rr_target = 2.5
                  AND o9.outcome = 'loss'
                  AND o10.outcome = 'loss'
                  AND d.orb_CME_REOPEN_size >= 4.0
                  AND d.orb_TOKYO_OPEN_size >= 4.0
                  {pf}
            """).fetchall()
            n = len(chop_rows)
            if n > 0:
                wins = sum(1 for r in chop_rows if r[0] == "win")
                total_r = sum(r[1] for r in chop_rows if r[1] is not None)
                lines.append(f"  {period_label:>7} 1100 RR{rr}: N={n:>3}, "
                             f"WR={_pct(wins/n)}, TotalR={_fmt(total_r)}")
            else:
                lines.append(f"  {period_label:>7} 1100 RR{rr}: N=0")

    return lines

# =========================================================================
# MAIN
# =========================================================================
def run_analysis(year: int = DEFAULT_YEAR, db_path: Path | None = None) -> str:
    con = _connect(db_path)
    try:
        all_lines = []
        all_lines.append("=" * 72)
        all_lines.append("  DEEP 0900 ASIA SESSION ANALYSIS + CROSS-SESSION INTELLIGENCE")
        all_lines.append(f"  Year focus: {year}  |  Instrument: {INSTRUMENT}  |  ORB: {ORB_MINUTES}m")
        all_lines.append("  Generated by asia_session_analyzer.py")
        all_lines.append("=" * 72)

        all_lines.extend(section_a_regime_deep_dive(con, year))
        all_lines.extend(section_b_cross_session(con, year))
        all_lines.extend(section_c_mfe_analysis(con, year))
        all_lines.extend(section_d_direction_grid(con, year))
        all_lines.extend(section_e_reversal_trades(con, year))

        # Summary
        all_lines.append("\n" + "=" * 72)
        all_lines.append("ACTIONABLE SUMMARY")
        all_lines.append("=" * 72)
        all_lines.append("1. 0900 E3 LONG G6+ RR3.0 -- strongest single edge (check full-period)")
        all_lines.append("2. Reversal rule: 0900 stop-out + 1000 opposite-dir G4+ = take 1000")
        all_lines.append("3. Chop detector: 0900+1000 both loss = reduce 1100 size or skip")
        all_lines.append("4. Direction: LONG-only filters for 0900 in current regime")
        all_lines.append("5. Trailing stop: check C3 for BE-at-0.5R conversion rate")
        all_lines.append("")

        return "\n".join(all_lines)
    finally:
        con.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep CME_REOPEN Asia Session Analysis")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR,
                        help=f"Focus year (default: {DEFAULT_YEAR})")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to gold.db (default: auto-detect)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else None
    report = run_analysis(year=args.year, db_path=db_path)

    print(report)

    # Save to artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"asia_session_analysis_{args.year}.txt"
    out_file.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_file}")

if __name__ == "__main__":
    main()
