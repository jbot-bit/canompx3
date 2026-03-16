"""
Walk-Forward Validation: ATR70+VOL Combined Filter & Cross-Asset ATR
=====================================================================

Validates the top candidates from research_vol_regime_filter.py using
12-window anchored walk-forward (matches pipeline walkforward.py logic).

For each candidate:
1. Load raw E2 CB1 5m outcomes
2. Apply the combined filter (ATR pct >= 70 AND rel_vol >= 1.2)
   or cross-asset filter (source ATR pct >= threshold)
3. Run anchored WF: 12-month min training, 6-month test windows
4. Report IS vs OOS Sharpe haircut, pass/fail

Zero-lookahead guaranteed: ATR percentile and rel_vol are both computed
from prior data only (trailing 252d for ATR, trailing 20 breaks for vol).

@research-source research_vol_regime_wf.py
"""

import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

# ── WF Parameters (match pipeline defaults) ────────────────────────────
TEST_WINDOW_MONTHS = 6
MIN_TRAIN_MONTHS = 12
MIN_TRADES_PER_WINDOW = 15
MIN_VALID_WINDOWS = 3
MIN_PCT_POSITIVE = 0.60
ATR_TRAILING = 252

# ── Candidates to test ─────────────────────────────────────────────────
# Format: (instrument, session, rr_target, filter_type, filter_params)
# filter_params: dict with keys depending on filter_type

CANDIDATES = [
    # ATR70+VOL combined (JK-significant from research)
    ("MES", "CME_PRECLOSE", 1.0, "ATR70+VOL", {}),
    ("MES", "CME_PRECLOSE", 1.5, "ATR70+VOL", {}),
    ("MES", "CME_PRECLOSE", 2.0, "ATR70+VOL", {}),
    ("MES", "SINGAPORE_OPEN", 1.0, "ATR70+VOL", {}),
    ("MES", "SINGAPORE_OPEN", 1.5, "ATR70+VOL", {}),
    ("MES", "COMEX_SETTLE", 1.0, "ATR70+VOL", {}),
    ("MNQ", "SINGAPORE_OPEN", 1.0, "ATR70+VOL", {}),
    ("MNQ", "SINGAPORE_OPEN", 1.5, "ATR70+VOL", {}),
    ("MNQ", "SINGAPORE_OPEN", 2.0, "ATR70+VOL", {}),
    ("MNQ", "COMEX_SETTLE", 1.0, "ATR70+VOL", {}),
    ("MNQ", "LONDON_METALS", 1.0, "ATR70+VOL", {}),
    ("MNQ", "LONDON_METALS", 1.5, "ATR70+VOL", {}),
    ("MNQ", "NYSE_CLOSE", 1.0, "ATR70+VOL", {}),

    # ATR70|VOL union (new opportunities)
    ("MNQ", "US_DATA_830", 1.0, "ATR70|VOL", {}),
    ("M2K", "CME_PRECLOSE", 1.0, "ATR70|VOL", {}),

    # Cross-asset MES ATR -> MNQ
    ("MNQ", "CME_PRECLOSE", 1.0, "X_MES_P70", {"source": "MES", "pct": 70}),
    ("MNQ", "CME_PRECLOSE", 1.0, "X_MES_P60", {"source": "MES", "pct": 60}),
    ("MNQ", "COMEX_SETTLE", 1.0, "X_MES_P70", {"source": "MES", "pct": 70}),
    ("MNQ", "COMEX_SETTLE", 1.0, "X_MES_P60", {"source": "MES", "pct": 60}),
    ("MNQ", "US_DATA_1000", 1.0, "X_MES_P70", {"source": "MES", "pct": 70}),
    ("MNQ", "NYSE_OPEN", 1.0, "X_MES_P60", {"source": "MES", "pct": 60}),

    # Cross-asset MGC ATR -> MNQ
    ("MNQ", "CME_PRECLOSE", 1.0, "X_MGC_P70", {"source": "MGC", "pct": 70}),
    ("MNQ", "COMEX_SETTLE", 1.0, "X_MGC_P70", {"source": "MGC", "pct": 70}),
]


def _add_months(d: date, months: int) -> date:
    """Add calendar months to a date, clamping day to month end."""
    import calendar
    total_months = d.year * 12 + (d.month - 1) + months
    year = total_months // 12
    month = total_months % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)


def load_all_data(db_path):
    """Load daily_features and outcomes for all instruments we need."""
    instruments = list(set(
        c[0] for c in CANDIDATES
    ) | set(
        c[4].get("source", "") for c in CANDIDATES if c[4].get("source")
    ))
    instruments = [i for i in instruments if i]

    con = duckdb.connect(str(db_path), read_only=True)

    features = con.execute("""
        SELECT symbol, trading_day, atr_20, garch_forecast_vol,
               rel_vol_CME_PRECLOSE, rel_vol_SINGAPORE_OPEN, rel_vol_COMEX_SETTLE,
               rel_vol_LONDON_METALS, rel_vol_NYSE_CLOSE, rel_vol_NYSE_OPEN,
               rel_vol_US_DATA_830, rel_vol_US_DATA_1000, rel_vol_TOKYO_OPEN,
               rel_vol_CME_REOPEN, rel_vol_BRISBANE_1025, rel_vol_EUROPE_FLOW
        FROM daily_features
        WHERE orb_minutes = 5
          AND symbol IN (SELECT UNNEST(?::VARCHAR[]))
        ORDER BY symbol, trading_day
    """, [instruments]).fetchdf()

    outcomes = con.execute("""
        SELECT trading_day, symbol, orb_label, rr_target, pnl_r
        FROM orb_outcomes
        WHERE orb_minutes = 5
          AND entry_model = 'E2'
          AND confirm_bars = 1
          AND pnl_r IS NOT NULL
          AND symbol IN (SELECT UNNEST(?::VARCHAR[]))
        ORDER BY symbol, orb_label, trading_day
    """, [instruments]).fetchdf()

    con.close()
    return features, outcomes


def compute_atr_percentiles(features, instruments):
    """Compute rolling ATR percentile for each instrument."""
    atr_pct = {}
    for inst in instruments:
        inst_df = features[
            (features["symbol"] == inst) & (features["atr_20"].notna())
        ].sort_values("trading_day")

        atr_vals = inst_df["atr_20"].values
        # Convert to list of datetime.date for consistent key types
        days = [d.date() if hasattr(d, "date") else d for d in inst_df["trading_day"]]

        for i in range(min(ATR_TRAILING, len(atr_vals)), len(atr_vals)):
            window = atr_vals[max(0, i - ATR_TRAILING):i]
            today_val = atr_vals[i]
            pct = np.searchsorted(np.sort(window), today_val) / len(window) * 100
            atr_pct[(inst, days[i])] = pct

    return atr_pct


def build_rel_vol_lookup(features):
    """Build (instrument, session, day) -> rel_vol lookup."""
    rv = {}
    for _, row in features.iterrows():
        inst = row["symbol"]
        day = row["trading_day"]
        if hasattr(day, "date"):
            day = day.date()
        for col in features.columns:
            if col.startswith("rel_vol_"):
                val = row[col]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    session = col.replace("rel_vol_", "")
                    rv[(inst, session, day)] = val
    return rv


def apply_filter(days_pnl, instrument, session, filter_type, filter_params,
                 atr_pct, rel_vol_lookup):
    """Apply filter and return list of (day, pnl_r) tuples that pass."""
    filtered = []

    for day, pnl_r in days_pnl:
        atr_p = atr_pct.get((instrument, day), 0)
        rv = rel_vol_lookup.get((instrument, session, day), 0)

        if filter_type == "ATR70+VOL":
            if atr_p >= 70 and rv >= 1.2:
                filtered.append((day, pnl_r))

        elif filter_type == "ATR70|VOL":
            if atr_p >= 70 or rv >= 1.2:
                filtered.append((day, pnl_r))

        elif filter_type.startswith("X_"):
            source = filter_params["source"]
            pct_thresh = filter_params["pct"]
            source_atr = atr_pct.get((source, day), 0)
            if source_atr >= pct_thresh:
                filtered.append((day, pnl_r))

        else:
            filtered.append((day, pnl_r))

    return filtered


def run_wf(filtered_outcomes):
    """Run anchored walk-forward on filtered (day, pnl_r) list.

    Returns dict with WF results.
    """
    if len(filtered_outcomes) < MIN_TRADES_PER_WINDOW * MIN_VALID_WINDOWS:
        return {
            "passed": False,
            "reason": f"Too few trades ({len(filtered_outcomes)})",
            "n_windows": 0,
            "n_positive": 0,
            "oos_avg_r": 0,
            "oos_total_r": 0,
            "oos_n": 0,
            "is_sharpe": 0,
            "oos_sharpe": 0,
            "haircut": None,
            "windows": [],
        }

    filtered_outcomes.sort(key=lambda x: x[0])
    # Normalize to date objects (pandas may return Timestamps)
    filtered_outcomes = [
        (d.date() if hasattr(d, "date") else d, pnl)
        for d, pnl in filtered_outcomes
    ]
    earliest = filtered_outcomes[0][0]
    latest = filtered_outcomes[-1][0]

    # Generate windows
    window_start = _add_months(earliest, MIN_TRAIN_MONTHS)
    windows = []
    all_oos_pnls = []

    while window_start <= latest:
        window_end = _add_months(window_start, TEST_WINDOW_MONTHS)

        # IS: everything before window_start
        is_trades = [pnl for d, pnl in filtered_outcomes if d < window_start]
        # OOS: window_start <= d < window_end
        oos_trades = [pnl for d, pnl in filtered_outcomes
                      if window_start <= d < window_end]

        if len(oos_trades) >= MIN_TRADES_PER_WINDOW:
            is_avg = np.mean(is_trades) if is_trades else 0
            oos_avg = np.mean(oos_trades)
            windows.append({
                "start": str(window_start),
                "end": str(window_end),
                "is_n": len(is_trades),
                "oos_n": len(oos_trades),
                "is_avg_r": round(is_avg, 4),
                "oos_avg_r": round(oos_avg, 4),
                "oos_positive": oos_avg > 0,
            })
            all_oos_pnls.extend(oos_trades)

        window_start = window_end

    n_valid = len(windows)
    n_positive = sum(1 for w in windows if w["oos_positive"])
    pct_positive = n_positive / n_valid if n_valid > 0 else 0

    # Aggregate OOS
    oos_n = len(all_oos_pnls)
    oos_avg_r = np.mean(all_oos_pnls) if all_oos_pnls else 0
    oos_total_r = np.sum(all_oos_pnls) if all_oos_pnls else 0

    # IS and OOS Sharpe for haircut
    is_pnls = [pnl for d, pnl in filtered_outcomes]
    is_std = np.std(is_pnls, ddof=1) if len(is_pnls) > 1 else 1
    is_sharpe = np.mean(is_pnls) / is_std if is_std > 0 else 0

    oos_std = np.std(all_oos_pnls, ddof=1) if len(all_oos_pnls) > 1 else 1
    oos_sharpe = oos_avg_r / oos_std if oos_std > 0 else 0

    # Annualize both using actual trades/year
    span_days = (latest - earliest).days
    years = span_days / 365.25 if span_days > 0 else 1
    tpy_is = len(is_pnls) / years
    tpy_oos = oos_n / years if years > 0 else 0

    is_sharpe_ann = is_sharpe * np.sqrt(tpy_is) if tpy_is > 0 else 0
    oos_sharpe_ann = oos_sharpe * np.sqrt(tpy_oos) if tpy_oos > 0 else 0

    haircut = oos_sharpe_ann - is_sharpe_ann

    # Pass/fail (4 rules from pipeline)
    passed = (
        n_valid >= MIN_VALID_WINDOWS
        and pct_positive >= MIN_PCT_POSITIVE
        and oos_avg_r > 0
        and oos_n >= MIN_TRADES_PER_WINDOW * MIN_VALID_WINDOWS
    )

    reason = None
    if not passed:
        if n_valid < MIN_VALID_WINDOWS:
            reason = f"Too few valid windows ({n_valid})"
        elif pct_positive < MIN_PCT_POSITIVE:
            reason = f"Pct positive too low ({pct_positive:.1%})"
        elif oos_avg_r <= 0:
            reason = f"OOS avg_r <= 0 ({oos_avg_r:.4f})"
        elif oos_n < MIN_TRADES_PER_WINDOW * MIN_VALID_WINDOWS:
            reason = f"Too few OOS trades ({oos_n})"

    return {
        "passed": passed,
        "reason": reason,
        "n_windows": n_valid,
        "n_positive": n_positive,
        "pct_positive": pct_positive,
        "oos_avg_r": round(oos_avg_r, 4),
        "oos_total_r": round(oos_total_r, 2),
        "oos_n": oos_n,
        "is_sharpe_ann": round(is_sharpe_ann, 3),
        "oos_sharpe_ann": round(oos_sharpe_ann, 3),
        "haircut": round(haircut, 3),
        "windows": windows,
    }


def main():
    print("=" * 90)
    print("WALK-FORWARD VALIDATION: ATR70+VOL & CROSS-ASSET ATR")
    print("=" * 90)
    print(f"WF params: {TEST_WINDOW_MONTHS}mo windows, {MIN_TRAIN_MONTHS}mo min train")
    print(f"Pass rules: {MIN_VALID_WINDOWS} valid windows, {MIN_PCT_POSITIVE:.0%} positive, OOS avg_r > 0")
    print(f"Candidates: {len(CANDIDATES)}")
    print()

    # Load data
    print("Loading data...")
    features, outcomes = load_all_data(GOLD_DB_PATH)
    print(f"  Features: {len(features)} rows, Outcomes: {len(outcomes)} rows")

    # Compute ATR percentiles for all needed instruments
    all_instruments = list(set(
        c[0] for c in CANDIDATES
    ) | set(
        c[4].get("source", "") for c in CANDIDATES if c[4].get("source")
    ))
    all_instruments = [i for i in all_instruments if i]

    print("Computing ATR percentiles...")
    atr_pct = compute_atr_percentiles(features, all_instruments)
    print(f"  Entries: {len(atr_pct)}")

    print("Building rel_vol lookup...")
    rel_vol_lookup = build_rel_vol_lookup(features)
    print(f"  Entries: {len(rel_vol_lookup)}")
    print()

    # Run WF for each candidate
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"\n{'Inst':4s} {'Session':20s} {'RR':>4s} {'Filter':14s} {'Pass':>5s} "
          f"{'Win':>5s} {'Pos':>5s} {'OOS_N':>6s} {'OOS_R':>8s} {'IS_Sh':>7s} "
          f"{'OOS_Sh':>7s} {'Hcut':>7s} {'Reason'}")
    print("-" * 120)

    passed_list = []
    failed_list = []

    for inst, session, rr, filt, params in CANDIDATES:
        # Get outcomes for this combo
        combo_outcomes = outcomes[
            (outcomes["symbol"] == inst) &
            (outcomes["orb_label"] == session) &
            (outcomes["rr_target"] == rr)
        ]

        days_pnl = [
            (row["trading_day"].date() if hasattr(row["trading_day"], "date") else row["trading_day"],
             row["pnl_r"])
            for _, row in combo_outcomes.iterrows()
        ]

        # Apply filter
        filtered = apply_filter(
            days_pnl, inst, session, filt, params,
            atr_pct, rel_vol_lookup
        )

        # Run WF
        result = run_wf(filtered)

        status = "PASS" if result["passed"] else "FAIL"
        reason = result.get("reason", "") or ""

        print(
            f"{inst:4s} {session:20s} {rr:4.1f} {filt:14s} {status:>5s} "
            f"{result['n_windows']:5d} {result.get('n_positive', 0):5d} "
            f"{result['oos_n']:6d} {result['oos_avg_r']:+8.4f} "
            f"{result.get('is_sharpe_ann', 0):7.3f} "
            f"{result.get('oos_sharpe_ann', 0):7.3f} "
            f"{(result.get('haircut') or 0):+7.3f} {reason}"
        )

        entry = {
            "instrument": inst,
            "session": session,
            "rr": rr,
            "filter": filt,
            **result,
        }
        if result["passed"]:
            passed_list.append(entry)
        else:
            failed_list.append(entry)

    # Summary
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Total candidates: {len(CANDIDATES)}")
    print(f"WF PASSED: {len(passed_list)}")
    print(f"WF FAILED: {len(failed_list)}")

    if passed_list:
        print(f"\nPASSED candidates (sorted by OOS Sharpe):")
        passed_list.sort(key=lambda x: x.get("oos_sharpe_ann", 0), reverse=True)
        for p in passed_list:
            print(
                f"  {p['instrument']:4s} {p['session']:20s} RR{p['rr']:.1f} "
                f"{p['filter']:14s}  OOS_Sharpe={p.get('oos_sharpe_ann', 0):.3f}  "
                f"Haircut={p.get('haircut', 0):+.3f}  "
                f"OOS_R={p['oos_avg_r']:+.4f}  N={p['oos_n']}"
            )

    if failed_list:
        print(f"\nFAILED candidates:")
        for f in failed_list:
            print(
                f"  {f['instrument']:4s} {f['session']:20s} RR{f['rr']:.1f} "
                f"{f['filter']:14s}  Reason: {f.get('reason', 'unknown')}"
            )

    # Window details for top 5 passed
    if passed_list:
        print()
        print("=" * 90)
        print("WINDOW DETAILS — TOP 5 PASSED")
        print("=" * 90)
        for p in passed_list[:5]:
            print(f"\n{p['instrument']} {p['session']} RR{p['rr']} {p['filter']}:")
            for w in p["windows"]:
                flag = "+" if w["oos_positive"] else "-"
                print(
                    f"  {w['start']} - {w['end']}  "
                    f"IS: N={w['is_n']:4d} avgR={w['is_avg_r']:+.4f}  "
                    f"OOS: N={w['oos_n']:3d} avgR={w['oos_avg_r']:+.4f} {flag}"
                )


if __name__ == "__main__":
    main()
