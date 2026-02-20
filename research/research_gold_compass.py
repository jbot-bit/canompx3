#!/usr/bin/env python3
"""
research_gold_compass.py — "The Gold Compass" hypothesis

HYPOTHESIS:
    MGC 0900 (CME gold open, 9:00 AM Brisbane) resolves overnight gold
    positioning. The break direction and quality — observable by 9:05 AM —
    reveal which macro regime is dominating:
        - Risk-on:   gold steady/down, equities bid
        - Risk-off:  gold UP (safe haven), equities sold
        - Inflation: gold UP, equities mixed

    CLAIM: MGC 0900 break direction is a leading indicator for MES 1000
    break quality 55+ minutes later.

MECHANISM:
    Not technical analysis. MGC 0900 is the first price-discovery event of
    the trading day for futures markets (CME open). It reveals risk appetite
    that then propagates to equity session opens (MES 1000 = US data release
    context). Same macro conditions drive both; gold resolves first.

NO LOOK-AHEAD:
    - MGC 0900 ORB: 09:00–09:05 Brisbane
    - Break direction known: by 09:05–09:10 Brisbane
    - MES 1000 ORB opens: 10:00 Brisbane
    - Signal lead time: 50-55 minutes (clean)

    NOTE: mgc_outcome (WIN/LOSS) uses FINAL session outcome — partially
    look-ahead for slow-resolving trades. Flagged explicitly below.
    Pure direction (L1) is always clean.

SECONDARY HYPOTHESIS — "ATR Exhaustion":
    ORB_size / ATR_20 = fraction of daily volatility budget consumed during
    range formation. A tight gold ORB (low ratio) means the market has
    more energy for the directional move to carry into equities.
    Prediction: low MGC_ORB/ATR → higher MES 1000 quality.

EXPECTED FINDINGS (pre-registration):
    - Gold LONG at CME open → MES 1000 may favour LONG (risk-on) or SHORT
      (risk-off safe-haven → equities sold). Direction unclear; quality
      (mfe_r) signal is the primary test.
    - MGC LONG WIN + LOW ATR_RATIO → highest MES 1000 mfe quality.
    - 2022 (risk-off + inflation) likely shows ANTI-correlation between
      gold and equity direction. 2023–2024 growth regime may show different.
"""

import os
import sys
from pathlib import Path
from scipy import stats
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

DB_PATH = os.environ.get("DUCKDB_PATH", str(PROJECT_ROOT / "gold.db"))
OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── BH FDR correction ─────────────────────────────────────────────────────────
def benjamini_hochberg(p_values: list[float], q: float = 0.10) -> set[int]:
    n = len(p_values)
    if n == 0:
        return set()
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    thresholds = [q * (rank + 1) / n for rank in range(n)]
    max_sig = -1
    for rank, (_, p) in enumerate(indexed):
        if p <= thresholds[rank]:
            max_sig = rank
    rejected = set()
    if max_sig >= 0:
        for rank, (orig_idx, _) in enumerate(indexed[: max_sig + 1]):
            rejected.add(orig_idx)
    return rejected


def ttest(values, mu=0.0):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return float("nan"), float("nan")
    t, p = stats.ttest_1samp(arr, mu)
    return float(t), float(p)


def ttest_ind(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b)
    return float(t), float(p)


# ── main ──────────────────────────────────────────────────────────────────────
def run():
    con = duckdb.connect(DB_PATH, read_only=True)

    # ── Load data ────────────────────────────────────────────────────────────
    # MGC signal: 0900 break direction, outcome, size, ATR, DST
    # MES outcome: 1000 mfe_r (max-favorable-excursion), outcome, direction
    # JOIN on trading_day — only days where BOTH had a 0900/1000 break.
    #
    # orb_minutes=5 required to avoid the 3-row-per-day trap.
    df_raw = con.execute("""
        WITH mgc AS (
            SELECT
                trading_day,
                orb_0900_break_dir   AS mgc_dir,
                orb_0900_outcome     AS mgc_outcome,
                orb_0900_size        AS mgc_orb_size,
                CASE WHEN atr_20 > 0 THEN orb_0900_size / atr_20 ELSE NULL END
                                     AS mgc_atr_ratio,
                us_dst               AS mgc_us_dst
            FROM daily_features
            WHERE symbol = 'MGC'
              AND orb_minutes = 5
              AND orb_0900_break_dir IS NOT NULL
        ),
        mes AS (
            SELECT
                trading_day,
                orb_1000_break_dir   AS mes_dir,
                orb_1000_outcome     AS mes_outcome,
                orb_1000_mfe_r       AS mes_mfe_r,
                orb_1000_size        AS mes_orb_size
            FROM daily_features
            WHERE symbol = 'MES'
              AND orb_minutes = 5
              AND orb_1000_break_dir IS NOT NULL
        )
        SELECT
            mgc.trading_day,
            YEAR(mgc.trading_day)   AS year,
            mgc.mgc_dir,
            mgc.mgc_outcome,
            mgc.mgc_orb_size,
            mgc.mgc_atr_ratio,
            mgc.mgc_us_dst,
            mes.mes_dir,
            mes.mes_outcome,
            mes.mes_mfe_r,
            mes.mes_orb_size
        FROM mgc
        JOIN mes ON mgc.trading_day = mes.trading_day
        ORDER BY mgc.trading_day
    """).df()
    con.close()

    N = len(df_raw)
    if N == 0:
        print("NO DATA: MES not in daily_features. Check DB path and MES build.")
        return

    print()
    print("=" * 65)
    print("THE GOLD COMPASS — MGC 0900 → MES 1000 Cross-Instrument Signal")
    print("=" * 65)
    print(f"Co-days (both had breaks): {N}")
    print(f"Date range: {df_raw['trading_day'].min()} → {df_raw['trading_day'].max()}")
    years = sorted(df_raw["year"].unique())
    print(f"Years: {years}")
    print()

    # Baseline MES mfe_r
    base_mfe = df_raw["mes_mfe_r"].dropna()
    t_base, p_base = ttest(base_mfe)
    print(f"MES 1000 baseline: N={len(base_mfe)}, avg_mfe={base_mfe.mean():.3f}R "
          f"(t={t_base:.2f}, p={p_base:.4f})")
    print()

    # Normalise direction/outcome to uppercase
    df_raw["mgc_dir"] = df_raw["mgc_dir"].str.upper()
    df_raw["mgc_outcome"] = df_raw["mgc_outcome"].str.upper()
    df_raw["mes_dir"] = df_raw["mes_dir"].str.upper()
    df_raw["mes_outcome"] = df_raw["mes_outcome"].str.upper()

    # Construct signal states
    df_raw["mgc_signal"] = df_raw["mgc_dir"] + "_" + df_raw["mgc_outcome"]
    df_raw["concordant"] = df_raw["mgc_dir"] == df_raw["mes_dir"]
    df_raw["mgc_won"] = df_raw["mgc_outcome"] == "WIN"

    # Better outcome metric: directional mfe — positive if MES moved in break direction,
    # but we can also derive a "signed" quality:
    # mes_outcome WIN = positive signal, LOSS = negative, use mfe_r for magnitude.
    # Also compute win_flag for win-rate analysis.
    df_raw["mes_win"] = (df_raw["mes_outcome"] == "WIN").astype(float)

    # ATR ratio terciles (on valid rows only)
    atr_valid = df_raw["mgc_atr_ratio"].dropna()
    q33 = atr_valid.quantile(0.33)
    q67 = atr_valid.quantile(0.67)

    def atr_tier(v):
        if v is None or np.isnan(v):
            return None
        if v <= q33:
            return "TIGHT"    # small fraction of ATR = energy to spare
        if v <= q67:
            return "MEDIUM"
        return "WIDE"         # large fraction of ATR = volatile/tired

    df_raw["atr_tier"] = df_raw["mgc_atr_ratio"].apply(atr_tier)

    # Collect all (p_val, label) for BH at the end
    all_results = []

    # ── L1: MGC direction alone → MES mfe_r [CLEAN — no look-ahead] ──────────
    print("─" * 65)
    print("L1: MGC 0900 direction alone → MES 1000 mfe_r  [CLEAN ✓]")
    print("─" * 65)
    print(f"{'MGC dir':<12} {'N':>5} {'MES avg_mfe':>12} {'t':>7} {'p':>9}")
    print(f"{'MGC dir':<12} {'N':>5} {'MES avg_mfe':>12} {'MES WR':>8} {'t(mfe)':>8} {'p(mfe)':>9} {'p(WR)':>9}")
    for mgc_dir in ["LONG", "SHORT"]:
        sub = df_raw[df_raw["mgc_dir"] == mgc_dir]
        mfe = sub["mes_mfe_r"].dropna()
        wr = sub["mes_win"].dropna()
        t, p = ttest(mfe)
        # WR vs 50%
        t_wr, p_wr = ttest(wr, mu=0.5)
        label = f"L1_{mgc_dir}_mfe"
        all_results.append((label, len(mfe), mfe.mean(), t, p))
        all_results.append((f"L1_{mgc_dir}_wr", len(wr), wr.mean(), t_wr, p_wr))
        print(f"  {mgc_dir:<10} {len(mfe):>5} {mfe.mean():>12.3f} {wr.mean():>8.1%} {t:>8.2f} {p:>9.4f} {p_wr:>9.4f}")

    # LONG vs SHORT mfe_r difference
    long_mfe = df_raw[df_raw["mgc_dir"] == "LONG"]["mes_mfe_r"].dropna()
    short_mfe = df_raw[df_raw["mgc_dir"] == "SHORT"]["mes_mfe_r"].dropna()
    t_diff, p_diff = ttest_ind(long_mfe, short_mfe)
    all_results.append(("L1_LONG_vs_SHORT", len(long_mfe) + len(short_mfe), float("nan"), t_diff, p_diff))
    print(f"  LONG vs SHORT mfe_r:  t={t_diff:.2f}, p={p_diff:.4f}")
    print()

    # ── L2: MGC signal state (dir × outcome) → MES mfe_r [PARTIAL look-ahead] ─
    print("─" * 65)
    print("L2: MGC signal state (dir × outcome) → MES mfe_r")
    print("    ⚠ NOTE: mgc_outcome = FINAL result; partially look-ahead")
    print("    for trades resolving after 10:00 AM. Treat as correlation,")
    print("    not necessarily a real-time decision rule.")
    print("─" * 65)
    print(f"{'MGC signal':<18} {'N':>5} {'MES avg_mfe':>12} {'t':>7} {'p':>9}")
    states = sorted(df_raw["mgc_signal"].unique())
    for state in states:
        sub = df_raw[df_raw["mgc_signal"] == state]["mes_mfe_r"].dropna()
        if len(sub) < 20:
            print(f"  {state:<16} {len(sub):>5}  (skip — N<20)")
            continue
        t, p = ttest(sub)
        label = f"L2_{state}"
        all_results.append((label, len(sub), sub.mean(), t, p))
        print(f"  {state:<16} {len(sub):>5} {sub.mean():>12.3f} {t:>7.2f} {p:>9.4f}")
    print()

    # ── L3: Direction concordance → does gold direction carry to equity? ───────
    print("─" * 65)
    print("L3: Direction concordance (MGC 0900 dir == MES 1000 dir?)  [CLEAN ✓]")
    print("─" * 65)
    conc = df_raw[df_raw["concordant"]]["mes_mfe_r"].dropna()
    disc = df_raw[~df_raw["concordant"]]["mes_mfe_r"].dropna()
    t_c, p_c = ttest(conc)
    t_d, p_d = ttest(disc)
    t_cd, p_cd = ttest_ind(conc, disc)
    all_results.append(("L3_concordant", len(conc), conc.mean(), t_c, p_c))
    all_results.append(("L3_discordant", len(disc), disc.mean(), t_d, p_d))
    all_results.append(("L3_conc_vs_disc", len(conc) + len(disc), float("nan"), t_cd, p_cd))

    conc_pct = df_raw["concordant"].mean() * 100
    print(f"  Direction concordance rate: {conc_pct:.1f}%  (50% = pure noise)")
    print(f"  Concordant  (same dir): N={len(conc)}, avg_mfe={conc.mean():.3f}R, "
          f"t={t_c:.2f}, p={p_c:.4f}")
    print(f"  Discordant  (opp dir):  N={len(disc)}, avg_mfe={disc.mean():.3f}R, "
          f"t={t_d:.2f}, p={p_d:.4f}")
    print(f"  Conc vs Disc difference: t={t_cd:.2f}, p={p_cd:.4f}")
    print()

    # Win-rate comparison: concordant vs discordant days
    conc_wr = (df_raw[df_raw["concordant"]]["mes_outcome"] == "win").mean()
    disc_wr = (df_raw[~df_raw["concordant"]]["mes_outcome"] == "win").mean()
    print(f"  MES win rate — concordant: {conc_wr:.1%}  |  discordant: {disc_wr:.1%}")
    print()

    # ── L4: ATR Exhaustion — MGC ORB/ATR → MES quality ───────────────────────
    print("─" * 65)
    print("L4: ATR Exhaustion — MGC ORB size / ATR_20 → MES 1000 mfe_r  [CLEAN ✓]")
    print("    Tight=ORB<33rd pctile of ATR, Wide=ORB>67th pctile")
    print("─" * 65)
    print(f"  MGC ATR ratio terciles:  TIGHT≤{q33:.3f}  MEDIUM≤{q67:.3f}  WIDE>{q67:.3f}")
    print()
    print(f"{'ATR tier':<12} {'N':>5} {'MES avg_mfe':>12} {'t':>7} {'p':>9}")
    for tier in ["TIGHT", "MEDIUM", "WIDE"]:
        sub = df_raw[df_raw["atr_tier"] == tier]["mes_mfe_r"].dropna()
        if len(sub) < 10:
            print(f"  {tier:<10} {len(sub):>5}  (skip — N<10)")
            continue
        t, p = ttest(sub)
        all_results.append((f"L4_{tier}", len(sub), sub.mean(), t, p))
        print(f"  {tier:<10} {len(sub):>5} {sub.mean():>12.3f} {t:>7.2f} {p:>9.4f}")

    # Tight vs Wide comparison
    tight_mfe = df_raw[df_raw["atr_tier"] == "TIGHT"]["mes_mfe_r"].dropna()
    wide_mfe = df_raw[df_raw["atr_tier"] == "WIDE"]["mes_mfe_r"].dropna()
    t_tw, p_tw = ttest_ind(tight_mfe, wide_mfe)
    all_results.append(("L4_TIGHT_vs_WIDE", len(tight_mfe) + len(wide_mfe), float("nan"), t_tw, p_tw))
    print(f"  TIGHT vs WIDE: t={t_tw:.2f}, p={p_tw:.4f}")
    print()

    # ── L5: Combined signal: MGC win + tight ATR + concordant ─────────────────
    print("─" * 65)
    print("L5: Combined signal — MGC WIN + TIGHT ORB + concordant direction")
    print("    ⚠ mgc_won uses final outcome (partial look-ahead)")
    print("─" * 65)
    combined = df_raw[
        df_raw["mgc_won"]
        & (df_raw["atr_tier"] == "TIGHT")
        & df_raw["concordant"]
    ]["mes_mfe_r"].dropna()
    baseline_match = df_raw[
        ~(df_raw["mgc_won"] & (df_raw["atr_tier"] == "TIGHT") & df_raw["concordant"])
    ]["mes_mfe_r"].dropna()
    t5, p5 = ttest(combined)
    t5_vs, p5_vs = ttest_ind(combined, baseline_match)
    all_results.append(("L5_combined", len(combined), combined.mean(), t5, p5))
    all_results.append(("L5_combined_vs_rest", len(combined) + len(baseline_match),
                        float("nan"), t5_vs, p5_vs))
    print(f"  Combined signal: N={len(combined)}, avg_mfe={combined.mean():.3f}R, "
          f"t={t5:.2f}, p={p5:.4f}")
    print(f"  Rest:            N={len(baseline_match)}, avg_mfe={baseline_match.mean():.3f}R")
    print(f"  Combined vs Rest: t={t5_vs:.2f}, p={p5_vs:.4f}")
    print()

    # ── L6: DST split — is signal cleaner when MGC 0900 = actual CME open? ────
    print("─" * 65)
    print("L6: DST split — winter (MGC 0900 = CME open) vs summer (MGC 0900 = 1hr post-open)")
    print("─" * 65)
    for dst_label, dst_val in [("Winter (MGC 0900 = CME open)", False),
                                ("Summer (MGC 0900 = post-open)", True)]:
        sub_dst = df_raw[df_raw["mgc_us_dst"] == dst_val]
        # Direction signal within DST regime
        for mgc_dir in ["LONG", "SHORT"]:
            sub = sub_dst[sub_dst["mgc_dir"] == mgc_dir]["mes_mfe_r"].dropna()
            if len(sub) >= 20:
                t, p = ttest(sub)
                print(f"  {dst_label[:28]} | MGC {mgc_dir}: N={len(sub)}, "
                      f"avg_mfe={sub.mean():.3f}R, p={p:.4f}")
    print()

    # ── L7: Year-by-year for the direction signal ─────────────────────────────
    print("─" * 65)
    print("L7: Year-by-year — does MGC direction→MES quality hold across regimes?")
    print("─" * 65)
    print(f"{'Year':<6} {'Dir':<7} {'N':>4} {'MES avg_mfe':>12} {'p':>9}")
    for yr in years:
        yr_df = df_raw[df_raw["year"] == yr]
        for mgc_dir in ["LONG", "SHORT"]:
            sub = yr_df[yr_df["mgc_dir"] == mgc_dir]["mes_mfe_r"].dropna()
            if len(sub) < 10:
                continue
            t, p = ttest(sub)
            print(f"  {yr}  {mgc_dir:<5}  {len(sub):>4} {sub.mean():>12.3f} {p:>9.4f}")
    print()

    # ── BH FDR correction across all tests ────────────────────────────────────
    print("─" * 65)
    print(f"BH FDR correction (q=0.10) across {len(all_results)} tests")
    print("─" * 65)
    p_vals = [r[4] for r in all_results]
    rejected = benjamini_hochberg(p_vals, q=0.10)

    survivors = []
    for i, (label, n, avg, t, p) in enumerate(all_results):
        if np.isnan(p):
            continue
        sig = "*** BH-SIG ***" if i in rejected else ""
        if sig or p < 0.10:
            avg_str = f"{avg:.3f}" if not np.isnan(avg) else "   —"
            print(f"  {label:<28} N={n:>4}, avg={avg_str}, p={p:.4f}  {sig}")
            if sig:
                survivors.append((label, n, avg, p))
    if not survivors:
        print("  No tests survive BH correction at q=0.10")
    print()

    # ── Honest summary ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("HONEST SUMMARY")
    print("=" * 65)
    print()
    if survivors:
        print("SURVIVED BH FDR (q=0.10):")
        for label, n, avg, p in survivors:
            print(f"  [{label}] N={n}, avg_mfe={avg:.3f}R, p={p:.4f}")
        print()
        print("CAVEATS:")
        print("  - Any L2/L5 survivor uses final mgc_outcome (partial look-ahead)")
        print("  - L1/L3/L4 survivors are fully clean (no look-ahead)")
        print("  - mfe_r = max excursion, not realised P&L at a specific RR target")
        print("  - MES data coverage may be shorter than MGC (check years)")
    else:
        print("DID NOT SURVIVE: No Gold Compass signal passes BH FDR at q=0.10.")
        print()
        print("INTERPRETATION:")
        print("  Gold direction does not reliably predict equity ORB quality.")
        print("  Possible reasons:")
        print("  1. Macro correlation is regime-dependent (averages to noise)")
        print("  2. mfe_r measures direction quality but not directional bias")
        print("  3. MES 1000 is sufficiently independent of gold's CME open")
        print("  4. The hypothesis is wrong — gold and equity ORBs are decorrelated")
        print()
        print("NEXT STEPS if signal exists in a sub-period:")
        print("  - Check year-by-year (L7) for regime windows")
        print("  - Test with actual pnl_r at RR2.5 from orb_outcomes (not mfe_r)")
        print("  - Test concordance WIN-RATE (L3 win rate) not just mfe quality")

    print()
    print("WHAT WOULD KILL THIS EDGE:")
    print("  - Regime change (inflation→deflation flips gold/equity correlation)")
    print("  - Arbitrage: if the signal is strong, algos would trade it away")
    print("  - DST transitions cause 0900 to miss the actual CME open by 1hr")
    print()

    # Save raw data
    out_path = OUTPUT_DIR / "gold_compass_codays.csv"
    df_raw.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df_raw)} rows)")


if __name__ == "__main__":
    run()
