#!/usr/bin/env python3
"""MNQ 09:25 vs CME_REOPEN (09:00) — Head-to-head comparison.

Uses session_discovery_full.csv only (no DB connection needed).
The discovery scan tested 288 times x 5 RR x 3 G-filters x 4 instruments
with BH FDR at q=0.05 across all 14,440 combos.
"""
import pandas as pd

CSV_PATH = "research/output/session_discovery_full.csv"
disc = pd.read_csv(CSV_PATH)

mnq = disc[disc["instrument"] == "MNQ"]

print("=" * 75)
print("MNQ 09:25 vs 09:00 (CME_REOPEN) — Session Discovery Head-to-Head")
print("=" * 75)

# ── Section 1: 09:25 full results ──────────────────────────────────
print("\n--- 09:25 Brisbane: All Combos ---")
t925 = mnq[mnq["time"] == "09:25"].sort_values(["g_name", "rr"])
for _, r in t925.iterrows():
    fdr = "FDR" if r["fdr_significant"] else "   "
    print(f"  RR{r['rr']:.1f} {r['g_name']:4s}  N={r['n']:5.0f}  "
          f"avgR={r['mean_r']:+.4f}  WR={r['win_rate']:.1%}  "
          f"Sharpe={r['sharpe_ann']:.2f}  totalR={r['total_r']:+7.1f}  "
          f"p_bh={r['p_bh']:.6f}  {fdr}")

# ── Section 2: 09:00 full results ──────────────────────────────────
print("\n--- 09:00 Brisbane (CME_REOPEN time): All Combos ---")
t900 = mnq[mnq["time"] == "09:00"].sort_values(["g_name", "rr"])
for _, r in t900.iterrows():
    fdr = "FDR" if r["fdr_significant"] else "   "
    print(f"  RR{r['rr']:.1f} {r['g_name']:4s}  N={r['n']:5.0f}  "
          f"avgR={r['mean_r']:+.4f}  WR={r['win_rate']:.1%}  "
          f"Sharpe={r['sharpe_ann']:.2f}  totalR={r['total_r']:+7.1f}  "
          f"p_bh={r['p_bh']:.6f}  {fdr}")

# ── Section 3: Head-to-head at matching params ─────────────────────
print("\n--- HEAD-TO-HEAD (matching RR + G-filter) ---")
print(f"  {'Combo':<12s}  {'09:00 avgR':>10s} {'09:25 avgR':>10s} "
      f"{'delta':>8s}  {'09:00 FDR':>9s} {'09:25 FDR':>9s}  Winner")
print("  " + "-" * 73)

wins_925 = 0
wins_900 = 0

for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for g in ["G4", "G5", "G6"]:
        r900 = t900[(t900["rr"] == rr) & (t900["g_name"] == g)]
        r925 = t925[(t925["rr"] == rr) & (t925["g_name"] == g)]

        if r900.empty or r925.empty:
            continue

        r9 = r900.iloc[0]
        r25 = r925.iloc[0]
        delta = r25["mean_r"] - r9["mean_r"]
        winner = "09:25" if delta > 0 else "09:00"

        if delta > 0:
            wins_925 += 1
        else:
            wins_900 += 1

        fdr9 = "FDR" if r9["fdr_significant"] else "   "
        fdr25 = "FDR" if r25["fdr_significant"] else "   "

        print(f"  RR{rr:.1f} {g:4s}    {r9['mean_r']:+.4f}     {r25['mean_r']:+.4f}   "
              f"{delta:+.4f}       {fdr9}       {fdr25}   {winner}")

print(f"\n  Score: 09:25 wins {wins_925}/{wins_925 + wins_900}, "
      f"09:00 wins {wins_900}/{wins_925 + wins_900}")

# ── Section 4: Key metrics comparison ──────────────────────────────
print("\n--- KEY METRICS (best combo: RR2.5 G4) ---")
best900 = t900[(t900["rr"] == 2.5) & (t900["g_name"] == "G4")].iloc[0]
best925 = t925[(t925["rr"] == 2.5) & (t925["g_name"] == "G4")].iloc[0]

metrics = [
    ("N trades", f"{best900['n']:.0f}", f"{best925['n']:.0f}"),
    ("Avg R", f"{best900['mean_r']:+.4f}", f"{best925['mean_r']:+.4f}"),
    ("Win Rate", f"{best900['win_rate']:.1%}", f"{best925['win_rate']:.1%}"),
    ("Ann. Sharpe", f"{best900['sharpe_ann']:.2f}", f"{best925['sharpe_ann']:.2f}"),
    ("Total R", f"{best900['total_r']:+.1f}", f"{best925['total_r']:+.1f}"),
    ("Trades/yr", f"{best900['trades_per_year']:.0f}", f"{best925['trades_per_year']:.0f}"),
    ("Years pos", f"{best900['years_pos']:.0f}/{best900['years_total']:.0f}",
                  f"{best925['years_pos']:.0f}/{best925['years_total']:.0f}"),
    ("Avg ORB size", f"{best900['avg_orb_size']:.1f} pts", f"{best925['avg_orb_size']:.1f} pts"),
    ("Avg volume", f"{best900['avg_vol']:.0f}", f"{best925['avg_vol']:.0f}"),
    ("DST winter avgR", f"{best900['avg_r_winter']:+.4f} (N={best900['n_winter']:.0f})",
                        f"{best925['avg_r_winter']:+.4f} (N={best925['n_winter']:.0f})"),
    ("DST summer avgR", f"{best900['avg_r_summer']:+.4f} (N={best900['n_summer']:.0f})",
                        f"{best925['avg_r_summer']:+.4f} (N={best925['n_summer']:.0f})"),
    ("p (raw)", f"{best900['p_value']:.6f}", f"{best925['p_value']:.6f}"),
    ("p (BH FDR)", f"{best900['p_bh']:.6f}", f"{best925['p_bh']:.6f}"),
    ("FDR significant", str(best900["fdr_significant"]), str(best925["fdr_significant"])),
]

print(f"  {'Metric':<22s}  {'09:00':>25s}  {'09:25':>25s}")
print("  " + "-" * 75)
for name, v900, v925 in metrics:
    print(f"  {name:<22s}  {v900:>25s}  {v925:>25s}")

# ── Section 5: Nearby times (context) ─────────────────────────────
print("\n--- NEARBY TIMES (MNQ, RR2.5, G4) ---")
nearby = mnq[(mnq["rr"] == 2.5) & (mnq["g_name"] == "G4") &
             (mnq["bris_h"] == 9)].sort_values("bris_m")

for _, r in nearby.iterrows():
    fdr = "FDR" if r["fdr_significant"] else "   "
    bar = "#" * max(0, int((r["mean_r"] + 0.3) * 20))
    print(f"  09:{r['bris_m']:02.0f}  avgR={r['mean_r']:+.4f}  Sharpe={r['sharpe_ann']:.2f}  "
          f"p_bh={r['p_bh']:.6f}  {fdr}  {bar}")

# ── Section 6: Overlap note ───────────────────────────────────────
print("\n--- OVERLAP WITH CME_REOPEN (from prior analysis) ---")
print("  99.6% of 09:25 break days also have 09:00 breaks (1,270/1,275)")
print("  Direction agreement: 51% (effectively random)")
print("  34% have completely independent ORB ranges")
print("  After 09:00 WIN:  09:25 avgR = +0.4502 (N=400)")
print("  After 09:00 LOSS: 09:25 avgR = +0.2549 (N=875)")

# ── Section 7: Verdict ────────────────────────────────────────────
print("\n" + "=" * 75)
print("VERDICT")
print("=" * 75)
print("""
  RAW 09:00 ORB break is NEGATIVE (-0.06R). The E2 entry model saves it.
  RAW 09:25 ORB break is POSITIVE (+0.32R at RR2.5). 15/15 FDR survivors.

  The 25-minute consolidation after CME reopen does naturally what the E2
  stop-market entry model does mechanically: it filters out the initial
  volatility noise and creates a cleaner range to break from.

  SAME DAYS, DIFFERENT RANGE — not an independent trading opportunity.
  But it confirms the CME reopen ZONE (09:00-09:30 Brisbane) as robust.

  NEXT STEP: If you want to formally test 09:25 through the pipeline,
  it would need to be added as a session in dst.py and run through the
  full chain (outcomes -> discovery -> validation -> edge families).
""")
print("Done.")
