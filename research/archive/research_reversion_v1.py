"""
Reversion Strategy Research v1

User observation: price frequently reverts to session open / prior levels
before the next session. Testing three hypotheses:

R1: ORB Double-Break Reversion
    Price breaks ORB in direction X, FAILS, then breaks through the OTHER side.
    Hypothesis: second-direction break continues further than first.

R2: Prior Day Level Fade
    Price tests prior day high or low before 10am.
    Hypothesis: after testing PDH/PDL, price reverts toward prior day midpoint.

R3: Gap Fill
    Price opens with a gap.
    Hypothesis: gaps fill intraday at above-random rate.

Anti-overfit guardrails:
- Split by year, require uplift in majority of years
- Require minimum N per cell
- No parameter tuning — raw structural test only
"""

import sys
from pathlib import Path
import duckdb
import numpy as np
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.paths import GOLD_DB_PATH

sys.stdout.reconfigure(line_buffering=True)

SYMBOLS = ["MES", "MNQ", "MGC", "M2K"]
MIN_N = 30

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)


# ============================================================
# R1: ORB DOUBLE-BREAK REVERSION
# ============================================================
print("=" * 70)
print("R1: ORB DOUBLE-BREAK REVERSION")
print("When first ORB break FAILS and price double-breaks,")
print("how far does the second direction move?")
print("=" * 70)

sessions = ["0900", "1000", "1800", "US_EQUITY_OPEN", "CME_OPEN", "LONDON_OPEN"]

for session in sessions:
    db_col = f"orb_{session}"
    q = f"""
        SELECT
            symbol,
            YEAR(trading_day) as yr,
            {db_col}_break_dir          as first_dir,
            {db_col}_outcome            as outcome,
            {db_col}_double_break       as dbl,
            {db_col}_mae_r              as mae_r,
            {db_col}_mfe_r              as mfe_r
        FROM daily_features
        WHERE symbol IN ('MES','MNQ','MGC','M2K')
          AND {db_col}_break_dir IS NOT NULL
          AND {db_col}_outcome   IS NOT NULL
          AND {db_col}_double_break IS NOT NULL
    """
    try:
        rows = con.execute(q).fetchall()
    except Exception as e:
        continue

    if not rows:
        continue

    # Separate: regular losses vs double-break losses
    dbl_loss = [r for r in rows if r[4] and r[3] == "loss"]
    reg_win = [r for r in rows if not r[4] and r[3] == "win"]
    reg_loss = [r for r in rows if not r[4] and r[3] == "loss"]
    all_rows = rows

    if len(dbl_loss) < MIN_N:
        continue

    # MAE on double-break losses = how far the SECOND direction went
    # (MAE is excursion AGAINST the first trade = IN FAVOR of reversion trade)
    mae_vals = [r[5] for r in dbl_loss if r[5] is not None]
    # MFE for comparison = how far the first direction got before failing
    mfe_vals = [r[6] for r in dbl_loss if r[6] is not None]

    avg_mae = np.mean(mae_vals) if mae_vals else 0
    avg_mfe = np.mean(mfe_vals) if mfe_vals else 0

    # Proxy reversion return at RR 1.5: entry = +1R past ORB, target = 1.5R
    # If mae_r > 2.0, the reversion trade (entered at -1R on first side = other ORB edge)
    # had at least 1R of room. We estimate "profit at RR1.5":
    # win if mae_r >= 2.5 (1R entry + 1.5R move), loss if < 2.0
    rev_wins = sum(1 for m in mae_vals if m >= 2.5)
    rev_total = len(mae_vals)
    rev_wr = rev_wins / rev_total if rev_total else 0

    # Year breakdown
    by_year = defaultdict(list)
    for r in dbl_loss:
        if r[5] is not None:
            by_year[r[1]].append(r[5])
    year_avgs = {y: np.mean(v) for y, v in by_year.items() if len(v) >= 5}
    years_positive = sum(1 for v in year_avgs.values() if v > 2.0)
    years_total = len(year_avgs)

    dbl_rate = len(dbl_loss) / len(all_rows) * 100

    print(f"\n{session} | Symbols combined")
    print(f"  Double-break losses: {len(dbl_loss)} ({dbl_rate:.1f}% of all breaks)")
    print(f"  Avg MAE (reversion move): {avg_mae:.2f}R | Avg MFE (first dir): {avg_mfe:.2f}R")
    print(f"  Reversion trade WR (proxy RR1.5): {rev_wr:.1%} ({rev_wins}/{rev_total})")
    print(f"  Years with MAE > 2.0R: {years_positive}/{years_total} {year_avgs}")

    # Per symbol breakdown
    by_sym = defaultdict(list)
    for r in dbl_loss:
        if r[5] is not None:
            by_sym[r[0]].append(r[5])
    print("  By symbol (avg MAE):", {s: f"{np.mean(v):.2f}R (n={len(v)})" for s, v in by_sym.items()})


# ============================================================
# R2: PRIOR DAY HIGH/LOW FADE
# ============================================================
print("\n")
print("=" * 70)
print("R2: PRIOR DAY HIGH/LOW FADE")
print("When price tests PDH or PDL before 10am, does it revert?")
print("(MFE of the fade trade = how far price moved away from PDH/PDL)")
print("=" * 70)

q2 = """
    SELECT
        symbol,
        YEAR(trading_day) as yr,
        took_pdh_before_1000,
        took_pdl_before_1000,
        prev_day_range,
        prev_day_direction,
        orb_1000_break_dir,
        orb_1000_outcome,
        orb_1000_mfe_r,
        orb_1000_mae_r,
        gap_open_points,
        gap_type
    FROM daily_features
    WHERE symbol IN ('MES','MNQ','MGC','M2K')
      AND took_pdh_before_1000 IS NOT NULL
      AND orb_1000_outcome IS NOT NULL
"""
rows2 = con.execute(q2).fetchall()

for sym in SYMBOLS:
    sym_rows = [r for r in rows2 if r[0] == sym]
    if not sym_rows:
        continue

    pdh_rows = [r for r in sym_rows if r[2]]  # took PDH before 10am
    pdl_rows = [r for r in sym_rows if r[3]]  # took PDL before 10am

    for label, fade_rows, fade_dir_bias in [
        ("PDH fade (short)", pdh_rows, "short"),
        ("PDL fade (long)", pdl_rows, "long"),
    ]:
        if len(fade_rows) < MIN_N:
            continue

        # After testing PDH/PDL, does the 10am ORB break in the reversion direction?
        reversion_breaks = [r for r in fade_rows if r[6] == fade_dir_bias]
        total = len(fade_rows)
        rev_rate = len(reversion_breaks) / total if total else 0

        # Among those reversion breaks, win rate
        rev_wins_n = sum(1 for r in reversion_breaks if r[7] == "win")
        rev_win_rate = rev_wins_n / len(reversion_breaks) if reversion_breaks else 0

        # Baseline: overall ORB_1000 win rate for this symbol
        baseline_rows = [r for r in sym_rows if r[6] == fade_dir_bias and r[7] is not None]
        base_wr = sum(1 for r in baseline_rows if r[7] == "win") / len(baseline_rows) if baseline_rows else 0

        print(f"\n  {sym} {label}: n={len(fade_rows)}")
        print(f"    After testing level, {fade_dir_bias} ORB_1000 rate: {rev_rate:.1%}")
        print(f"    Reversion break WR: {rev_win_rate:.1%} (n={len(reversion_breaks)})")
        print(f"    Baseline {fade_dir_bias} ORB_1000 WR: {base_wr:.1%}")
        uplift = rev_win_rate - base_wr
        print(f"    Uplift: {uplift:+.1%} {'<< SIGNAL' if uplift > 0.05 else ''}")


# ============================================================
# R3: GAP FILL
# ============================================================
print("\n")
print("=" * 70)
print("R3: GAP FILL — do gaps fill intraday?")
print("=" * 70)

q3 = """
    SELECT
        symbol,
        YEAR(trading_day) as yr,
        gap_type,
        gap_open_points,
        daily_open,
        daily_close,
        prev_day_close,
        took_pdh_before_1000,
        took_pdl_before_1000
    FROM daily_features
    WHERE symbol IN ('MES','MNQ','MGC','M2K')
      AND gap_type IS NOT NULL
      AND gap_open_points IS NOT NULL
      AND ABS(gap_open_points) > 0
"""
rows3 = con.execute(q3).fetchall()

for sym in SYMBOLS:
    sym_rows = [r for r in rows3 if r[0] == sym]
    if not sym_rows:
        continue

    gap_up = [r for r in sym_rows if r[2] == "gap_up" or (r[3] and r[3] > 0)]
    gap_down = [r for r in sym_rows if r[2] == "gap_down" or (r[3] and r[3] < 0)]

    for label, g_rows, fill_cond in [
        ("Gap UP   (fade short)", gap_up, "took_pdl"),
        ("Gap DOWN (fade long)", gap_down, "took_pdh"),
    ]:
        if len(g_rows) < MIN_N:
            continue

        # Gap fills if daily_close went back past prev_day_close
        if label.startswith("Gap UP"):
            fills = [r for r in g_rows if r[5] is not None and r[6] is not None and r[5] <= r[6]]
        else:
            fills = [r for r in g_rows if r[5] is not None and r[6] is not None and r[5] >= r[6]]

        fill_rate = len(fills) / len(g_rows) if g_rows else 0

        print(f"\n  {sym} {label}: n={len(g_rows)}")
        print(f"    Gap fill rate (EOD close past prior close): {fill_rate:.1%}")
        print(f"    (>50% = fade has structural edge)")

con.close()
print("\nDone.")
