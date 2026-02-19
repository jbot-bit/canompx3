#!/usr/bin/env python3
"""P0: Session Time vs Market Event Analysis.

Determines whether validated trading edges follow clock time (fixed Brisbane
session) or market events (DST-adjusted dynamic session).

Computes its own DST splits from filtered orb_outcomes.  NEVER reads
pre-computed dst_verdict / dst_winter_avg_r / dst_summer_avg_r columns.

Output: research/output/session_event_analysis.md
"""

import duckdb
import csv
import statistics
import os
from pathlib import Path
from pipeline.dst import DST_AFFECTED_SESSIONS, classify_dst_verdict

# ── Configuration ──────────────────────────────────────────────────────

DB_PATH = os.environ.get(
    "DUCKDB_PATH",
    str(Path(__file__).resolve().parent.parent / "gold.db"),
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "session_event_analysis.md"
TIME_SCAN_CSV = OUTPUT_DIR / "orb_time_scan_full.csv"

# Fixed ↔ Dynamic pairs: (fixed_label, dynamic_label, dst_driver)
SESSION_PAIRS = [
    ("0900", "CME_OPEN", "US"),
    ("1800", "LONDON_OPEN", "UK"),
    ("0030", "US_EQUITY_OPEN", "US"),
    ("2300", "US_DATA_OPEN", "US"),
]

# ORB size filter thresholds (from trading_app/config.py)
FILTER_MIN_SIZE = {
    "ORB_G4": 4.0,
    "ORB_G5": 5.0,
    "ORB_G6": 6.0,
    "ORB_G8": 8.0,
}

# Brisbane-hour locations of each event in winter vs summer
ADJACENT = {
    "0900": {"winter_h": 9, "winter_m": 0, "summer_h": 8, "summer_m": 0},
    "1800": {"winter_h": 18, "winter_m": 0, "summer_h": 17, "summer_m": 0},
    "0030": {"winter_h": 0, "winter_m": 30, "summer_h": 23, "summer_m": 30},
    "2300": {"winter_h": 23, "winter_m": 0, "summer_h": 22, "summer_m": 30},
}


# ── Helpers ────────────────────────────────────────────────────────────

def sharpe(rs):
    """Sharpe ratio from list of R-multiples."""
    if len(rs) < 3:
        return 0.0
    m = statistics.mean(rs)
    s = statistics.stdev(rs)
    return m / s if s > 0 else 0.0


def compute_filtered_dst_split(con, instrument, orb_label, entry_model,
                                rr_target, confirm_bars, filter_type,
                                orb_minutes=5, dst_driver="US"):
    """Compute winter/summer split from filtered orb_outcomes.

    Applies the same filter logic as strategy_discovery:
    - ORB size filter (G4/G5/G6/G8 percentile thresholds)
    - Double-break exclusion
    - Only win/loss outcomes (exclude scratch/NULL)

    Returns dict with winter/summer/combined metrics, or None if no data.
    """
    size_col = f"orb_{orb_label}_size"
    dbl_col = f"orb_{orb_label}_double_break"
    dst_col = "us_dst" if dst_driver == "US" else "uk_dst"

    sql = f"""
        SELECT o.pnl_r,
               df.{dst_col}     AS is_summer,
               df.{size_col}    AS orb_size,
               df.{dbl_col}     AS double_break
        FROM orb_outcomes o
        JOIN daily_features df
            ON o.trading_day = df.trading_day
            AND o.symbol    = df.symbol
        WHERE o.symbol      = $1
          AND o.orb_label   = $2
          AND o.entry_model = $3
          AND o.rr_target   = $4
          AND o.confirm_bars = $5
          AND o.orb_minutes = $6
          AND df.orb_minutes = $6
          AND o.outcome IN ('win', 'loss')
    """
    rows = con.execute(
        sql, [instrument, orb_label, entry_model,
              rr_target, confirm_bars, orb_minutes]
    ).fetchall()

    if not rows:
        return None

    min_size = FILTER_MIN_SIZE.get(filter_type)
    winter_rs, summer_rs = [], []

    for pnl_r, is_summer, orb_size, double_break in rows:
        # Exclude double-break days
        if double_break:
            continue
        # Apply ORB size filter
        if min_size is not None:
            if orb_size is None or orb_size < min_size:
                continue
        # Skip VOL / DIR filters (can't apply without extra data)
        if filter_type and (filter_type.startswith("VOL_")
                            or filter_type.startswith("DIR_")):
            continue

        if is_summer:
            summer_rs.append(pnl_r)
        else:
            winter_rs.append(pnl_r)

    combined_rs = winter_rs + summer_rs

    def metrics(rs):
        if not rs:
            return {"n": 0, "avg_r": None, "total_r": 0.0,
                    "wr": None, "sharpe": 0.0}
        wins = sum(1 for r in rs if r > 0)
        return {
            "n": len(rs),
            "avg_r": statistics.mean(rs),
            "total_r": sum(rs),
            "wr": wins / len(rs),
            "sharpe": sharpe(rs),
        }

    return {
        "winter": metrics(winter_rs),
        "summer": metrics(summer_rs),
        "combined": metrics(combined_rs),
    }


def load_time_scan(csv_path):
    """Load time scan CSV keyed by (instrument, bris_h, bris_m)."""
    def _float(val):
        if val is None or val.strip() == "":
            return None
        return float(val)

    def _int(val):
        if val is None or val.strip() == "":
            return 0
        return int(val)

    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["instrument"], int(row["bris_h"]), int(row["bris_m"]))
            data[key] = {
                "n_trades": _int(row["n_trades"]),
                "avg_r": _float(row["avg_r"]),
                "n_winter": _int(row["n_winter"]),
                "avg_r_winter": _float(row["avg_r_winter"]),
                "wr_winter": _float(row["wr_winter"]),
                "n_summer": _int(row["n_summer"]),
                "avg_r_summer": _float(row["avg_r_summer"]),
                "wr_summer": _float(row["wr_summer"]),
                "current_session": row.get("current_session", ""),
            }
    return data


def fmt_r(val):
    if val is None:
        return "---"
    return f"{val:+.4f}"


def fmt_pct(val):
    if val is None:
        return "---"
    return f"{val:.1%}"


def _empty_metrics():
    return {"n": 0, "avg_r": None, "total_r": 0.0, "wr": None, "sharpe": 0.0}


# ── Sections ───────────────────────────────────────────────────────────

def section1_aggregate(con, out):
    """Section 1: Fixed vs Dynamic Aggregate Comparison."""
    out.append("## Section 1: Fixed vs Dynamic Aggregate Comparison")
    out.append("")
    out.append("Top 20 positive-expectancy MGC strategies by Sharpe from "
               "`experimental_strategies`, with filtered DST splits "
               "computed from scratch.")
    out.append("")

    for fixed, dynamic, dst_driver in SESSION_PAIRS:
        out.append(f"### {fixed} (fixed) vs {dynamic} (dynamic) "
                   f"--- DST driver: {dst_driver}")
        out.append("")

        for label in [fixed, dynamic]:
            strats = con.execute("""
                SELECT entry_model, rr_target, confirm_bars, filter_type,
                       sharpe_ratio, expectancy_r, sample_size
                FROM experimental_strategies
                WHERE instrument = 'MGC' AND orb_label = $1
                  AND expectancy_r > 0 AND sample_size >= 20
                ORDER BY sharpe_ratio DESC
                LIMIT 20
            """, [label]).fetchall()

            if not strats:
                out.append(f"**{label}:** No positive-expectancy strategies "
                           f"with N>=20 found.")
                out.append("")
                continue

            winter_rs, summer_rs, combined_rs = [], [], []
            winter_ns, summer_ns = [], []
            winter_sharpes, summer_sharpes = [], []
            n_computed = 0

            for em, rr, cb, ft, sr, expr, ss in strats:
                split = compute_filtered_dst_split(
                    con, "MGC", label, em, rr, cb, ft,
                    orb_minutes=5, dst_driver=dst_driver)
                if split is None:
                    continue
                w, s, c = split["winter"], split["summer"], split["combined"]
                n_computed += 1
                if w["avg_r"] is not None and w["n"] > 0:
                    winter_rs.append(w["avg_r"])
                    winter_ns.append(w["n"])
                    winter_sharpes.append(w["sharpe"])
                if s["avg_r"] is not None and s["n"] > 0:
                    summer_rs.append(s["avg_r"])
                    summer_ns.append(s["n"])
                    summer_sharpes.append(s["sharpe"])
                if c["avg_r"] is not None:
                    combined_rs.append(c["avg_r"])

            out.append(f"**{label}** ({n_computed} strategies analyzed):")
            out.append("")
            if winter_rs:
                out.append(
                    f"- Winter: mean avgR={fmt_r(statistics.mean(winter_rs))}, "
                    f"median={fmt_r(statistics.median(winter_rs))}, "
                    f"mean Sharpe={statistics.mean(winter_sharpes):.4f}, "
                    f"mean N={sum(winter_ns)/len(winter_ns):.0f}")
            else:
                out.append("- Winter: no data")
            if summer_rs:
                out.append(
                    f"- Summer: mean avgR={fmt_r(statistics.mean(summer_rs))}, "
                    f"median={fmt_r(statistics.median(summer_rs))}, "
                    f"mean Sharpe={statistics.mean(summer_sharpes):.4f}, "
                    f"mean N={sum(summer_ns)/len(summer_ns):.0f}")
            else:
                out.append("- Summer: no data")
            if combined_rs:
                out.append(
                    f"- Combined: mean avgR="
                    f"{fmt_r(statistics.mean(combined_rs))}, "
                    f"median={fmt_r(statistics.median(combined_rs))}")
            out.append("")

        out.append("---")
        out.append("")


def section2_matched_pair(con, out):
    """Section 2: Matched Pair Deep Dive for WINTER-DOM 0900 strategies."""
    out.append("## Section 2: Matched Pair Deep Dive "
               "--- WINTER-DOM 0900 Strategies")
    out.append("")
    out.append("The 4 validated WINTER-DOM MGC 0900 strategies "
               "(all E3/CB1/ORB_G5) compared with CME_OPEN using "
               "identical parameters.  Splits computed from filtered "
               "outcomes (double-break excluded, G5 size filter applied).")
    out.append("")

    validated_0900 = con.execute("""
        SELECT strategy_id, entry_model, rr_target, confirm_bars,
               filter_type, sample_size, expectancy_r, sharpe_ratio
        FROM validated_setups
        WHERE instrument = 'MGC'
          AND orb_label  = '0900'
          AND dst_verdict = 'WINTER-DOM'
        ORDER BY rr_target
    """).fetchall()

    if not validated_0900:
        out.append("*No WINTER-DOM 0900 strategies found in validated_setups.*")
        out.append("")
        return validated_0900

    out.append("| Strategy | RR | 0900 W avgR (N) | 0900 S avgR (N) "
               "| CME W avgR (N) | CME S avgR (N) | CME Comb avgR (N) "
               "| Signal |")
    out.append("|----------|----|-----------------|-----------------"
               "|----------------|----------------|-------------------"
               "|--------|")

    for sid, em, rr, cb, ft, ss, expr, sr in validated_0900:
        fixed_split = compute_filtered_dst_split(
            con, "MGC", "0900", em, rr, cb, ft,
            orb_minutes=5, dst_driver="US")
        dynamic_split = compute_filtered_dst_split(
            con, "MGC", "CME_OPEN", em, rr, cb, ft,
            orb_minutes=5, dst_driver="US")

        fw = fixed_split["winter"] if fixed_split else _empty_metrics()
        fs = fixed_split["summer"] if fixed_split else _empty_metrics()
        dw = dynamic_split["winter"] if dynamic_split else _empty_metrics()
        ds = dynamic_split["summer"] if dynamic_split else _empty_metrics()
        dc = dynamic_split["combined"] if dynamic_split else _empty_metrics()

        # Signal logic
        signal = "UNCLEAR"
        if dc["avg_r"] is not None and dc["avg_r"] > 0:
            if (dw["avg_r"] is not None and dw["avg_r"] > 0
                    and ds["avg_r"] is not None and ds["avg_r"] > 0):
                signal = "EVENT"
            else:
                signal = "EVENT (partial)"
        elif fw["avg_r"] is not None and fw["avg_r"] > 0:
            if fs["avg_r"] is not None and fs["avg_r"] <= 0:
                signal = "CLOCK (W-only)"
            else:
                signal = "CLOCK"

        out.append(
            f"| {sid} | {rr} "
            f"| {fmt_r(fw['avg_r'])} ({fw['n']}) "
            f"| {fmt_r(fs['avg_r'])} ({fs['n']}) "
            f"| {fmt_r(dw['avg_r'])} ({dw['n']}) "
            f"| {fmt_r(ds['avg_r'])} ({ds['n']}) "
            f"| {fmt_r(dc['avg_r'])} ({dc['n']}) "
            f"| {signal} |"
        )

    out.append("")

    # ── With vs Without Filter ─────────────────────────────────────
    out.append("### Filter Effect: CME_OPEN with G5 vs NO_FILTER")
    out.append("")
    out.append("Isolates whether the G5 filter is the reason CME_OPEN "
               "diverges from the time-scan 0800 signal.")
    out.append("")
    out.append("| RR | CME G5 avgR (N) | CME NoFilter avgR (N) | Delta |")
    out.append("|----|-----------------|----------------------|-------|")

    for sid, em, rr, cb, ft, ss, expr, sr in validated_0900:
        with_f = compute_filtered_dst_split(
            con, "MGC", "CME_OPEN", em, rr, cb, ft,
            orb_minutes=5, dst_driver="US")
        no_f = compute_filtered_dst_split(
            con, "MGC", "CME_OPEN", em, rr, cb, "NO_FILTER",
            orb_minutes=5, dst_driver="US")

        wf_c = with_f["combined"] if with_f else _empty_metrics()
        nf_c = no_f["combined"] if no_f else _empty_metrics()
        delta = None
        if wf_c["avg_r"] is not None and nf_c["avg_r"] is not None:
            delta = wf_c["avg_r"] - nf_c["avg_r"]

        out.append(
            f"| {rr} "
            f"| {fmt_r(wf_c['avg_r'])} ({wf_c['n']}) "
            f"| {fmt_r(nf_c['avg_r'])} ({nf_c['n']}) "
            f"| {fmt_r(delta)} |"
        )

    out.append("")

    # ── Entry model comparison ─────────────────────────────────────
    out.append("### Entry Model Effect: CME_OPEN E3 vs E1")
    out.append("")
    out.append("Time scan uses E1; validated 0900 strategies use E3. "
               "Does E1 at CME_OPEN show the edge the time scan sees?")
    out.append("")
    out.append("| RR | CME E3/G5 avgR (N) | CME E1/G5 avgR (N) "
               "| CME E1/G4 avgR (N) |")
    out.append("|----|--------------------|--------------------|"
               "--------------------|")

    for sid, em, rr, cb, ft, ss, expr, sr in validated_0900:
        e3_split = compute_filtered_dst_split(
            con, "MGC", "CME_OPEN", "E3", rr, 1, ft,
            orb_minutes=5, dst_driver="US")
        e1_g5 = compute_filtered_dst_split(
            con, "MGC", "CME_OPEN", "E1", rr, 2, "ORB_G5",
            orb_minutes=5, dst_driver="US")
        e1_g4 = compute_filtered_dst_split(
            con, "MGC", "CME_OPEN", "E1", rr, 2, "ORB_G4",
            orb_minutes=5, dst_driver="US")

        e3c = e3_split["combined"] if e3_split else _empty_metrics()
        e1g5c = e1_g5["combined"] if e1_g5 else _empty_metrics()
        e1g4c = e1_g4["combined"] if e1_g4 else _empty_metrics()

        out.append(
            f"| {rr} "
            f"| {fmt_r(e3c['avg_r'])} ({e3c['n']}) "
            f"| {fmt_r(e1g5c['avg_r'])} ({e1g5c['n']}) "
            f"| {fmt_r(e1g4c['avg_r'])} ({e1g4c['n']}) |"
        )

    out.append("")
    out.append("---")
    out.append("")

    return validated_0900


def section3_stable(con, out):
    """Section 3: STABLE Session Analysis --- 1800 vs LONDON_OPEN."""
    out.append("## Section 3: STABLE Session Analysis "
               "--- 1800 vs LONDON_OPEN")
    out.append("")
    out.append("Note: In winter (GMT), LONDON_OPEN resolves to 18:00 "
               "Brisbane --- identical to the 1800 fixed session. "
               "The winter columns should match.  The comparison is "
               "purely about summer behavior (17:00 vs 18:00 Brisbane).")
    out.append("")

    strats_1800 = con.execute("""
        SELECT entry_model, rr_target, confirm_bars, filter_type,
               sharpe_ratio, expectancy_r, sample_size
        FROM experimental_strategies
        WHERE instrument = 'MGC' AND orb_label = '1800'
          AND expectancy_r > 0 AND sample_size >= 20
        ORDER BY sharpe_ratio DESC
        LIMIT 10
    """).fetchall()

    if not strats_1800:
        out.append("*No positive-expectancy 1800 strategies found.*")
        out.append("")
        return

    out.append("| Params | 1800 W avgR (N) | 1800 S avgR (N) "
               "| 1800 Verdict | LDN W avgR (N) | LDN S avgR (N) "
               "| LDN Verdict |")
    out.append("|--------|-----------------|-----------------|"
               "-------------|----------------|----------------|"
               "-------------|")

    for em, rr, cb, ft, sr, expr, ss in strats_1800:
        fsplit = compute_filtered_dst_split(
            con, "MGC", "1800", em, rr, cb, ft,
            orb_minutes=5, dst_driver="UK")
        dsplit = compute_filtered_dst_split(
            con, "MGC", "LONDON_OPEN", em, rr, cb, ft,
            orb_minutes=5, dst_driver="UK")

        fw = fsplit["winter"] if fsplit else _empty_metrics()
        fs = fsplit["summer"] if fsplit else _empty_metrics()
        dw = dsplit["winter"] if dsplit else _empty_metrics()
        ds = dsplit["summer"] if dsplit else _empty_metrics()

        fv = classify_dst_verdict(
            fw["avg_r"], fs["avg_r"], fw["n"], fs["n"]
        ) if fsplit else "NO-DATA"
        dv = classify_dst_verdict(
            dw["avg_r"], ds["avg_r"], dw["n"], ds["n"]
        ) if dsplit else "NO-DATA"

        params = f"{em}/RR{rr}/CB{cb}/{ft}"
        out.append(
            f"| {params} "
            f"| {fmt_r(fw['avg_r'])} ({fw['n']}) "
            f"| {fmt_r(fs['avg_r'])} ({fs['n']}) "
            f"| {fv} "
            f"| {fmt_r(dw['avg_r'])} ({dw['n']}) "
            f"| {fmt_r(ds['avg_r'])} ({ds['n']}) "
            f"| {dv} |"
        )

    out.append("")

    # Also check 0030/US_EQUITY_OPEN briefly
    out.append("### 0030 vs US_EQUITY_OPEN (brief)")
    out.append("")

    strats_0030 = con.execute("""
        SELECT entry_model, rr_target, confirm_bars, filter_type,
               sharpe_ratio, expectancy_r, sample_size
        FROM experimental_strategies
        WHERE instrument = 'MGC' AND orb_label = '0030'
          AND expectancy_r > 0 AND sample_size >= 20
        ORDER BY sharpe_ratio DESC
        LIMIT 5
    """).fetchall()

    if not strats_0030:
        out.append("*No positive-expectancy 0030 strategies with N>=20.*")
    else:
        out.append("| Params | 0030 W (N) | 0030 S (N) "
                   "| US_EQ W (N) | US_EQ S (N) |")
        out.append("|--------|-----------|-----------|"
                   "------------|------------|")
        for em, rr, cb, ft, sr, expr, ss in strats_0030:
            fsplit = compute_filtered_dst_split(
                con, "MGC", "0030", em, rr, cb, ft,
                orb_minutes=5, dst_driver="US")
            dsplit = compute_filtered_dst_split(
                con, "MGC", "US_EQUITY_OPEN", em, rr, cb, ft,
                orb_minutes=5, dst_driver="US")
            fw = fsplit["winter"] if fsplit else _empty_metrics()
            fs = fsplit["summer"] if fsplit else _empty_metrics()
            dw = dsplit["winter"] if dsplit else _empty_metrics()
            ds = dsplit["summer"] if dsplit else _empty_metrics()
            params = f"{em}/RR{rr}/CB{cb}/{ft}"
            out.append(
                f"| {params} "
                f"| {fmt_r(fw['avg_r'])} ({fw['n']}) "
                f"| {fmt_r(fs['avg_r'])} ({fs['n']}) "
                f"| {fmt_r(dw['avg_r'])} ({dw['n']}) "
                f"| {fmt_r(ds['avg_r'])} ({ds['n']}) |"
            )

    out.append("")
    out.append("---")
    out.append("")


def section4_adjacent_times(out):
    """Section 4: +/-1hr Adjacent Time Analysis from time scan CSV."""
    out.append("## Section 4: Adjacent Time Analysis "
               "(from time scan CSV)")
    out.append("")
    out.append("Data from `orb_time_scan_full.csv` (RR=2.0, G4+, E1 entry, "
               "US DST classification for winter/summer).")
    out.append("")
    out.append("**Caveat:** The time scan uses US DST for ALL sessions. "
               "For 1800 (UK DST driver), the winter/summer split here "
               "uses US DST, not UK DST. Interpret with care.")
    out.append("")

    if not TIME_SCAN_CSV.exists():
        out.append("*Time scan CSV not found. Skipping.*")
        out.append("")
        return

    ts = load_time_scan(TIME_SCAN_CSV)

    # ── Q1: Does the edge follow the event into summer? ────────────
    out.append("### Does the edge follow the event into summer?")
    out.append("")
    out.append("Compare: fixed-time winter avgR vs event-shifted summer "
               "avgR.  If both positive, edge may follow the event.")
    out.append("")
    out.append("| Fixed | Fixed-Winter avgR (N) "
               "| Event-Shifted Summer avgR (N) | Follows? |")
    out.append("|-------|----------------------|------------------------------|"
               "----------|")

    for sess, adj in ADJACENT.items():
        fk = ("MGC", adj["winter_h"], adj["winter_m"])
        sk = ("MGC", adj["summer_h"], adj["summer_m"])
        fd = ts.get(fk, {})
        sd = ts.get(sk, {})

        fw = fd.get("avg_r_winter")
        ss_avgr = sd.get("avg_r_summer")

        follows = "---"
        if fw is not None and ss_avgr is not None:
            if fw > 0.05 and ss_avgr > 0.05:
                follows = "YES"
            elif fw > 0.05 and ss_avgr <= 0:
                follows = "NO"
            elif fw <= 0:
                follows = "N/A (no winter edge)"
            else:
                follows = "WEAK"

        out.append(
            f"| {sess} "
            f"| {fmt_r(fw)} (N={fd.get('n_winter', '?')}) "
            f"| {fmt_r(ss_avgr)} (N={sd.get('n_summer', '?')}) "
            f"| {follows} |"
        )

    out.append("")

    # ── Q2: Does the fixed time retain value in summer? ────────────
    out.append("### Does the fixed time retain value when the event "
               "has moved away?")
    out.append("")
    out.append("| Fixed | Fixed-Summer avgR (N) | Retains? |")
    out.append("|-------|-----------------------|----------|")

    for sess, adj in ADJACENT.items():
        fk = ("MGC", adj["winter_h"], adj["winter_m"])
        fd = ts.get(fk, {})
        fs_avgr = fd.get("avg_r_summer")

        retains = "---"
        if fs_avgr is not None:
            if fs_avgr > 0.05:
                retains = "YES"
            elif fs_avgr > 0:
                retains = "MARGINAL"
            else:
                retains = "NO"

        out.append(
            f"| {sess} "
            f"| {fmt_r(fs_avgr)} (N={fd.get('n_summer', '?')}) "
            f"| {retains} |"
        )

    out.append("")

    # ── Full 4-point grid ──────────────────────────────────────────
    out.append("### Full 4-Point Comparison")
    out.append("")
    out.append("Each cell: avgR at that (Brisbane time, DST regime) "
               "from the time scan.")
    out.append("")
    out.append("| Session | Fixed-Winter | Fixed-Summer "
               "| Shifted-Summer | Shifted-Winter |")
    out.append("|---------|-------------|-------------|"
               "---------------|---------------|")

    for sess, adj in ADJACENT.items():
        fk = ("MGC", adj["winter_h"], adj["winter_m"])
        sk = ("MGC", adj["summer_h"], adj["summer_m"])
        fd = ts.get(fk, {})
        sd = ts.get(sk, {})

        out.append(
            f"| {sess} "
            f"| {fmt_r(fd.get('avg_r_winter'))} "
            f"(N={fd.get('n_winter', '?')}) "
            f"| {fmt_r(fd.get('avg_r_summer'))} "
            f"(N={fd.get('n_summer', '?')}) "
            f"| {fmt_r(sd.get('avg_r_summer'))} "
            f"(N={sd.get('n_summer', '?')}) "
            f"| {fmt_r(sd.get('avg_r_winter'))} "
            f"(N={sd.get('n_winter', '?')}) |"
        )

    out.append("")
    out.append("---")
    out.append("")


def section5_verdict(con, out, validated_0900):
    """Section 5: Verdict Table + Contradictions."""
    out.append("## Section 5: Verdict Table")
    out.append("")
    out.append("| Session Pair | Edge Follows | Key Evidence "
               "| Recommendation |")
    out.append("|-------------|-------------|-------------|"
               "----------------|")

    for fixed, dynamic, dst_driver in SESSION_PAIRS:
        if fixed == "0900" and validated_0900:
            # Aggregate across 4 validated strategies
            dyn_combined, fix_w, fix_s = [], [], []
            dyn_w, dyn_s = [], []
            for sid, em, rr, cb, ft, ss, expr, sr in validated_0900:
                fs = compute_filtered_dst_split(
                    con, "MGC", fixed, em, rr, cb, ft,
                    orb_minutes=5, dst_driver=dst_driver)
                ds = compute_filtered_dst_split(
                    con, "MGC", dynamic, em, rr, cb, ft,
                    orb_minutes=5, dst_driver=dst_driver)
                if fs:
                    if fs["winter"]["avg_r"] is not None:
                        fix_w.append(fs["winter"]["avg_r"])
                    if fs["summer"]["avg_r"] is not None:
                        fix_s.append(fs["summer"]["avg_r"])
                if ds:
                    if ds["combined"]["avg_r"] is not None:
                        dyn_combined.append(ds["combined"]["avg_r"])
                    if ds["winter"]["avg_r"] is not None:
                        dyn_w.append(ds["winter"]["avg_r"])
                    if ds["summer"]["avg_r"] is not None:
                        dyn_s.append(ds["summer"]["avg_r"])

            m_dc = statistics.mean(dyn_combined) if dyn_combined else 0
            m_fw = statistics.mean(fix_w) if fix_w else 0
            m_fs = statistics.mean(fix_s) if fix_s else 0
            m_dw = statistics.mean(dyn_w) if dyn_w else 0
            m_ds = statistics.mean(dyn_s) if dyn_s else 0

            if m_dc > 0 and m_dw > 0 and m_ds > 0:
                verdict = "EVENT"
                evidence = (f"CME_OPEN +avgR both regimes "
                            f"(W:{fmt_r(m_dw)}, S:{fmt_r(m_ds)})")
            elif m_fw > 0 and m_fs <= 0 and m_dc <= 0:
                verdict = "CLOCK (winter)"
                evidence = (f"0900 winter edge ({fmt_r(m_fw)}), "
                            f"CME_OPEN combined {fmt_r(m_dc)}")
            elif m_fw > 0 and m_dc > 0:
                verdict = "UNCLEAR"
                evidence = (f"Both positive: fixed W={fmt_r(m_fw)}, "
                            f"dynamic comb={fmt_r(m_dc)}")
            elif m_fw > 0 and m_fs > 0:
                verdict = "BOTH"
                evidence = (f"Fixed works both regimes "
                            f"(W:{fmt_r(m_fw)}, S:{fmt_r(m_fs)})")
            else:
                verdict = "UNCLEAR"
                evidence = (f"F_W={fmt_r(m_fw)}, F_S={fmt_r(m_fs)}, "
                            f"D={fmt_r(m_dc)}")
            rec = "See Section 2 deep dive"
        else:
            # Use top experimental strategy for this fixed session
            top = con.execute("""
                SELECT entry_model, rr_target, confirm_bars, filter_type
                FROM experimental_strategies
                WHERE instrument = 'MGC' AND orb_label = $1
                  AND expectancy_r > 0 AND sample_size >= 20
                ORDER BY sharpe_ratio DESC
                LIMIT 1
            """, [fixed]).fetchone()

            if top:
                em, rr, cb, ft = top
                fs = compute_filtered_dst_split(
                    con, "MGC", fixed, em, rr, cb, ft,
                    orb_minutes=5, dst_driver=dst_driver)
                ds = compute_filtered_dst_split(
                    con, "MGC", dynamic, em, rr, cb, ft,
                    orb_minutes=5, dst_driver=dst_driver)

                if fs and ds:
                    fw_r = fs["winter"]["avg_r"] or 0
                    fs_r = fs["summer"]["avg_r"] or 0
                    dc_r = ds["combined"]["avg_r"] or 0
                    dw_r = ds["winter"]["avg_r"] or 0
                    ds_r = ds["summer"]["avg_r"] or 0

                    if dc_r > 0 and dw_r > 0 and ds_r > 0:
                        verdict = "EVENT"
                        evidence = f"{dynamic} positive both regimes"
                    elif fw_r > 0 and fs_r > 0 and abs(fw_r - fs_r) < 0.10:
                        verdict = "BOTH"
                        evidence = "Fixed stable across regimes"
                    elif fw_r > 0 and dc_r <= 0:
                        verdict = "CLOCK"
                        evidence = "Fixed winter works, dynamic fails"
                    else:
                        verdict = "UNCLEAR"
                        evidence = (f"F_W={fmt_r(fw_r)}, "
                                    f"F_S={fmt_r(fs_r)}, "
                                    f"D_comb={fmt_r(dc_r)}")
                    rec = (f"Params: {em}/RR{rr}/CB{cb}/{ft}")
                else:
                    verdict = "NO-DATA"
                    evidence = "Missing split data"
                    rec = "---"
            else:
                verdict = "NO-DATA"
                evidence = "No positive-expectancy strategies"
                rec = "---"

        out.append(f"| {fixed}/{dynamic} | {verdict} | {evidence} | {rec} |")

    out.append("")

    # ── Contradictions & Open Questions ────────────────────────────
    out.append("### Contradictions & Open Questions")
    out.append("")

    # 0800/CME_OPEN divergence
    cme_avg = con.execute("""
        SELECT AVG(expectancy_r), COUNT(*)
        FROM experimental_strategies
        WHERE instrument = 'MGC' AND orb_label = 'CME_OPEN'
          AND expectancy_r IS NOT NULL
    """).fetchone()

    cme_pos = con.execute("""
        SELECT AVG(expectancy_r), COUNT(*)
        FROM experimental_strategies
        WHERE instrument = 'MGC' AND orb_label = 'CME_OPEN'
          AND expectancy_r > 0 AND sample_size >= 20
    """).fetchone()

    out.append("1. **0800 Summer / CME_OPEN Divergence:**")
    if TIME_SCAN_CSV.exists():
        ts = load_time_scan(TIME_SCAN_CSV)
        ts_0800 = ts.get(("MGC", 8, 0), {})
        out.append(f"   - Time scan at 08:00 summer: "
                   f"avgR={fmt_r(ts_0800.get('avg_r_summer'))} "
                   f"(N={ts_0800.get('n_summer', '?')})")
    out.append(f"   - CME_OPEN all strategies: "
               f"mean expR={fmt_r(cme_avg[0])} "
               f"(N={cme_avg[1]} strategies)")
    out.append(f"   - CME_OPEN positive strategies: "
               f"mean expR={fmt_r(cme_pos[0])} "
               f"(N={cme_pos[1]} strategies)")
    out.append("   - Explanation candidates:")
    out.append("     - Time scan = RR2.0/E1/G4+; "
               "validated 0900 = E3/various RR/G5")
    out.append("     - Entry model (E1 vs E3) may capture different edge")
    out.append("     - CME_OPEN average pulled down by many negative "
               "strategy combos (grid of 2376)")
    out.append("")

    out.append("2. **LONDON_OPEN winter = 1800 winter:** "
               "In winter, LONDON_OPEN resolves to 18:00 Brisbane "
               "(same as 1800). Winter metrics should be nearly identical "
               "between the two. Any difference indicates data coverage "
               "or rounding variance.")
    out.append("")

    out.append("3. **Sample sizes:** Dynamic sessions have fewer "
               "outcome rows (~60-80% of fixed) because they were "
               "added to the pipeline later. This reduces statistical "
               "power for regime comparisons.")
    out.append("")

    out.append("4. **Time scan uses US DST universally:** The "
               "winter/summer split in the time scan CSV always uses "
               "US DST (EDT/EST). For the 1800/LONDON_OPEN pair "
               "(UK DST driver), Section 4 numbers are classified by "
               "US DST, not UK DST. This matters during the ~3-week "
               "gap when US and UK DST don't overlap.")
    out.append("")

    out.append("---")
    out.append("")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print(f"Database: {DB_PATH}")
    con = duckdb.connect(DB_PATH, read_only=True)

    # Sanity check
    orb_min = con.execute("""
        SELECT DISTINCT orb_minutes
        FROM daily_features WHERE symbol = 'MGC'
        ORDER BY orb_minutes
    """).fetchall()
    print(f"daily_features orb_minutes: {[r[0] for r in orb_min]}")

    n_validated = con.execute("""
        SELECT COUNT(*) FROM validated_setups WHERE instrument = 'MGC'
    """).fetchone()[0]
    print(f"Validated MGC strategies: {n_validated}")

    out = []
    out.append("# Session Time vs Market Event Analysis")
    out.append("")
    out.append("**P0 Research:** Does the trading edge follow the "
               "**clock time** (fixed Brisbane session) or the "
               "**market event** (DST-adjusted dynamic session)?")
    out.append("")
    out.append("**Method:** All DST splits computed from scratch using "
               "filtered `orb_outcomes` joined with `daily_features`. "
               "Applies ORB size filter + double-break exclusion. "
               "Never reads pre-computed `dst_verdict` / "
               "`dst_winter_avg_r` / `dst_summer_avg_r` columns.")
    out.append("")
    out.append("**Fixed/Dynamic Pairs:**")
    out.append("")
    out.append("| Fixed | Dynamic | DST Driver | Summer Divergence |")
    out.append("|-------|---------|-----------|-------------------|")
    out.append("| 0900 | CME_OPEN | US | CME opens at 0800 in summer |")
    out.append("| 1800 | LONDON_OPEN | UK | London opens at 1700 "
               "in summer |")
    out.append("| 0030 | US_EQUITY_OPEN | US | NYSE opens at 2330 "
               "in summer |")
    out.append("| 2300 | US_DATA_OPEN | US | 30min off both regimes "
               "(special) |")
    out.append("")
    out.append("---")
    out.append("")

    section1_aggregate(con, out)
    validated_0900 = section2_matched_pair(con, out)
    section3_stable(con, out)
    section4_adjacent_times(out)
    section5_verdict(con, out, validated_0900)

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Lines:  {len(out)}")
    con.close()


if __name__ == "__main__":
    main()
