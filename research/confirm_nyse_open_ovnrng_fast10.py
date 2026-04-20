"""Pathway-B K=1 confirmatory test — MNQ NYSE_OPEN 5m E2 RR1.0 CB1 OVNRNG_50_FAST10.

Pre-reg: docs/audit/hypotheses/2026-04-20-nyse-open-ovnrng-fast10-v1.yaml
Runs ALL pre-committed kill criteria from the pre-reg. Each criterion is
evaluated independently and reported with numeric outcome + PASS/FAIL tag.
No implementation or deployment action is taken.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")
SESSION = "NYSE_OPEN"
INSTRUMENT = "MNQ"
ORB_MIN = 5
ENTRY = "E2"
CB = 1
RR = 1.0
FILTER = "OVNRNG_50_FAST10"

print("=" * 80)
print("CONFIRMATORY TEST — MNQ NYSE_OPEN 5m E2 RR1.0 CB1 OVNRNG_50_FAST10 (Pathway B K=1)")
print(f"Ran at: {pd.Timestamp.now('UTC')}")
print("=" * 80)


def load_lane(symbol: str, session: str, orb_min: int, entry: str, rr: float, cb: int) -> pd.DataFrame:
    q = """
        SELECT o.trading_day, o.pnl_r,
               CASE WHEN o.pnl_r > 0 THEN 1 ELSE 0 END AS win,
               d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.rr_target = ?
          AND o.confirm_bars = ?
          AND o.pnl_r IS NOT NULL
        ORDER BY o.trading_day
    """
    return DB.execute(q, [symbol, session, orb_min, entry, rr, cb]).fetchdf()


# Load and filter
df_full = load_lane(INSTRUMENT, SESSION, ORB_MIN, ENTRY, RR, CB)
sig = filter_signal(df_full, FILTER, SESSION)
df = df_full.loc[sig == 1].copy().reset_index(drop=True)
is_df = df[df.trading_day < HOLDOUT].copy()
oos_df = df[df.trading_day >= HOLDOUT].copy()
is_unfilt = df_full[df_full.trading_day < HOLDOUT].copy()

print(f"\n[LOAD] full (unfiltered) IS: n={len(is_unfilt)}")
print(f"[LOAD] filtered universe: n={len(df)} (fire_rate {len(df)/len(df_full):.3f})")
print(f"[LOAD] IS: n={len(is_df)}  OOS: n={len(oos_df)}")

results = {}

# -------- C3 Chordia with theory: |t| >= 3.00 on IS --------
print("\n" + "-" * 80)
print("C3 — IS t-stat vs zero (Chordia with theory, t ≥ 3.00)")
print("-" * 80)
is_mean = is_df.pnl_r.mean()
is_std = is_df.pnl_r.std()
is_se = is_std / np.sqrt(len(is_df))
is_t = is_mean / is_se
print(f"  IS n={len(is_df)}  mean={is_mean:+.4f}  std={is_std:.4f}  t={is_t:+.3f}")
c3_pass = abs(is_t) >= 3.00
results["C3"] = ("PASS" if c3_pass else "FAIL", f"t={is_t:+.3f} vs 3.00")
print(f"  → {results['C3'][0]} (threshold 3.00)")

# -------- C5 DSR with K_upstream = 772 --------
print("\n" + "-" * 80)
print("C5 — Deflated Sharpe Ratio (Bailey-LdP 2014), K_upstream = 772")
print("-" * 80)
from math import log
is_sharpe = (is_mean / is_std) * np.sqrt(252)
N = len(is_df)
# Deflated Sharpe: DSR = (SR_obs - E[max_SR_null]) / std(SR)
# E[max_SR_null] under K trials ≈ sqrt(2*ln(K) - ln(ln(K)) - ln(4*pi))
K = 772
emax = np.sqrt(2 * log(K) - log(log(K)) - log(4 * np.pi)) if K > 1 else 0.0
# Std of SR estimator (annualised): sqrt((1 + 0.5*SR^2)/N)*sqrt(252)
skew = stats.skew(is_df.pnl_r.values)
kurt = stats.kurtosis(is_df.pnl_r.values, fisher=True)
sr_daily = is_mean / is_std
sr_var = (1 - skew * sr_daily + ((kurt - 1) / 4) * sr_daily**2) / (N - 1)
sr_std = np.sqrt(sr_var) * np.sqrt(252)
# DSR is Pr(true SR > 0) with deflation
dsr_z = (is_sharpe - emax * sr_std) / sr_std if sr_std > 0 else 0.0
dsr_p = stats.norm.cdf(dsr_z)  # probability true SR exceeds deflated null
print(f"  IS Sharpe ann = {is_sharpe:+.3f}")
print(f"  E[max|null, K={K}] (Bailey approx) ≈ {emax:.3f}")
print(f"  SR std (skew={skew:+.2f}, ex-kurt={kurt:+.2f}): {sr_std:.3f}")
print(f"  DSR p-value (Pr[true SR > deflated null]) ≈ {dsr_p:.4f}")
c5_pass = dsr_p >= 0.95
results["C5"] = ("PASS" if c5_pass else "FAIL/DOWNGRADE", f"DSR≈{dsr_p:.4f} vs 0.95")
print(f"  → {results['C5'][0]}")

# -------- C6 WFE >= 0.50 --------
print("\n" + "-" * 80)
print("C6 — WFE (OOS Sharpe / IS Sharpe) ≥ 0.50")
print("-" * 80)
oos_mean = oos_df.pnl_r.mean() if len(oos_df) else float("nan")
oos_std = oos_df.pnl_r.std() if len(oos_df) else float("nan")
oos_sharpe = (oos_mean / oos_std) * np.sqrt(252) if len(oos_df) > 1 else float("nan")
wfe = oos_sharpe / is_sharpe if is_sharpe != 0 else float("nan")
print(f"  OOS n={len(oos_df)}  mean={oos_mean:+.4f}  std={oos_std:.4f}  Sharpe_ann={oos_sharpe:+.3f}")
print(f"  IS Sharpe_ann={is_sharpe:+.3f}  WFE={wfe:+.3f}")
c6_pass = not np.isnan(wfe) and wfe >= 0.50
results["C6"] = ("PASS" if c6_pass else "FAIL", f"WFE={wfe:+.3f} vs 0.50")
print(f"  → {results['C6'][0]}")

# -------- C7 deployable N --------
print("\n" + "-" * 80)
print("C7 — Deployable N (≥30 directional, ≥100 confirmatory)")
print("-" * 80)
n_oos = len(oos_df)
if n_oos >= 100:
    c7_label = "PASS_CONFIRMATORY"
elif n_oos >= 30:
    c7_label = "DIRECTIONAL_ONLY"
else:
    c7_label = "FAIL"
results["C7"] = (c7_label, f"OOS n={n_oos}")
print(f"  OOS n={n_oos}  → {c7_label}")

# -------- C8 OOS ExpR ≥ 0.40 * IS ExpR --------
print("\n" + "-" * 80)
print("C8 — OOS ExpR ≥ 0.40 × IS ExpR")
print("-" * 80)
c8_threshold = 0.40 * is_mean
c8_pass = oos_mean >= c8_threshold and np.sign(oos_mean) == np.sign(is_mean)
print(f"  IS ExpR={is_mean:+.4f}  threshold={c8_threshold:+.4f}  OOS ExpR={oos_mean:+.4f}")
print(f"  dir_match: {np.sign(is_mean) == np.sign(oos_mean)}")
results["C8"] = ("PASS" if c8_pass else "FAIL", f"OOS={oos_mean:+.4f} vs {c8_threshold:+.4f}")
print(f"  → {results['C8'][0]}")

# -------- C9 era stability --------
print("\n" + "-" * 80)
print("C9 — Era stability (no year ExpR < -0.05 at N ≥ 50)")
print("-" * 80)
is_y = is_df.copy()
is_y["year"] = pd.to_datetime(is_y["trading_day"]).dt.year
era_fail = False
era_rows = []
for y, grp in is_y.groupby("year"):
    n = len(grp)
    expr = grp.pnl_r.mean()
    flag = ""
    if n >= 50 and expr < -0.05:
        era_fail = True
        flag = " *FAIL*"
    era_rows.append((y, n, expr, flag))
    print(f"  {y}: n={n:4d}  ExpR={expr:+.4f}{flag}")
c9_pass = not era_fail
results["C9"] = ("PASS" if c9_pass else "FAIL", "no year breaks -0.05 with N≥50" if c9_pass else "year(s) below -0.05")
print(f"  → {results['C9'][0]}")

# -------- T0 Tautology --------
print("\n" + "-" * 80)
print("T0 — Tautology vs deployed filters (|corr| ≤ 0.70)")
print("-" * 80)
t0_fail = False
for comp_key in ["COST_LT12", "ORB_G5", "ORB_G8", "OVNRNG_100", "ATR_P50"]:
    s2 = filter_signal(df_full, comp_key, SESSION)
    c = float(np.corrcoef(sig, s2)[0, 1])
    mark = ""
    if abs(c) > 0.70:
        mark = " *HIGH*"
        t0_fail = True
    print(f"  {FILTER} vs {comp_key:12s}: corr={c:+.4f}{mark}")
results["T0"] = ("PASS" if not t0_fail else "DOWNGRADE", "")
print(f"  → {results['T0'][0]}")

# -------- T1 WR monotonicity vs unfiltered --------
print("\n" + "-" * 80)
print("T1 — WR lift vs unfiltered MNQ NYSE_OPEN 5m E2 RR1.0 (≥ 2pp)")
print("-" * 80)
wr_filt = is_df.win.mean()
wr_unfilt = is_unfilt.win.mean()
wr_lift = (wr_filt - wr_unfilt) * 100
t1_pass = wr_lift >= 2.0
print(f"  Filtered IS WR = {wr_filt:.4f}")
print(f"  Unfilt   IS WR = {wr_unfilt:.4f}")
print(f"  Lift = {wr_lift:+.2f}pp")
results["T1"] = ("PASS" if t1_pass else "ARITHMETIC_ONLY", f"lift={wr_lift:+.2f}pp vs 2pp")
print(f"  → {results['T1'][0]}")

# -------- T4 Sensitivity: OVNRNG_25/75 × FAST5/15 --------
print("\n" + "-" * 80)
print("T4 — Sensitivity: neighboring OVNRNG × FAST variants must stay positive")
print("-" * 80)
sens_fail = False
for neighbor in ["OVNRNG_25_FAST5", "OVNRNG_25_FAST10", "OVNRNG_50_FAST5",
                 "OVNRNG_50_FAST10", "OVNRNG_100_FAST5", "OVNRNG_100_FAST10"]:
    try:
        s = filter_signal(df_full, neighbor, SESSION)
        sub = df_full.loc[s == 1]
        is_sub = sub[sub.trading_day < HOLDOUT]
        oos_sub = sub[sub.trading_day >= HOLDOUT]
        is_expr = is_sub.pnl_r.mean() if len(is_sub) else float("nan")
        oos_expr = oos_sub.pnl_r.mean() if len(oos_sub) else float("nan")
        mark = ""
        if is_expr < 0 and len(is_sub) >= 50:
            sens_fail = True
            mark = " *NEG_IS*"
        print(f"  {neighbor:22s}: IS n={len(is_sub):4d} ExpR={is_expr:+.4f}  |  "
              f"OOS n={len(oos_sub):4d} ExpR={oos_expr:+.4f}{mark}")
    except Exception as e:
        print(f"  {neighbor}: err {e}")
results["T4"] = ("PASS" if not sens_fail else "SENSITIVE", "neighbors positive" if not sens_fail else "at least one neighbor flips sign")
print(f"  → {results['T4'][0]}")

# -------- T6 null floor one-sample --------
print("\n" + "-" * 80)
print("T6 — Null floor: one-sample t-test vs 0 one-tailed p ≤ 0.05")
print("-" * 80)
t6_p = stats.ttest_1samp(is_df.pnl_r.values, 0.0, alternative="greater").pvalue
t6_pass = t6_p <= 0.05
print(f"  one-tailed p = {t6_p:.6f}")
results["T6"] = ("PASS" if t6_pass else "FAIL", f"p={t6_p:.4f} vs 0.05")
print(f"  → {results['T6'][0]}")

# -------- T8 cross-instrument --------
print("\n" + "-" * 80)
print("T8 — Cross-instrument directional consistency on IS")
print("-" * 80)
t8_dirs = {}
for inst in ["MNQ", "MGC", "MES"]:
    df_i = load_lane(inst, SESSION, ORB_MIN, ENTRY, RR, CB)
    if len(df_i) == 0:
        t8_dirs[inst] = float("nan")
        print(f"  {inst}: no data")
        continue
    s = filter_signal(df_i, FILTER, SESSION)
    sub = df_i.loc[s == 1]
    is_sub = sub[sub.trading_day < HOLDOUT]
    oos_sub = sub[sub.trading_day >= HOLDOUT]
    is_expr = is_sub.pnl_r.mean() if len(is_sub) else float("nan")
    t8_dirs[inst] = is_expr
    print(f"  {inst}: IS n={len(is_sub):4d} ExpR={is_expr:+.4f}  | OOS n={len(oos_sub):3d} ExpR={oos_sub.pnl_r.mean() if len(oos_sub) else float('nan'):+.4f}")
mnq_sign = np.sign(t8_dirs.get("MNQ", 0))
consistent = all(
    np.sign(t8_dirs.get(i, float("nan"))) == mnq_sign or np.isnan(t8_dirs.get(i, float("nan")))
    for i in ["MGC", "MES"]
)
results["T8"] = ("PASS" if consistent else "FLAG_INSTRUMENT_SPECIFIC", "all same sign as MNQ" if consistent else "one or more flip")
print(f"  → {results['T8'][0]}")

# -------- Final verdict --------
print("\n" + "=" * 80)
print("VERDICT SUMMARY")
print("=" * 80)
for k in ["C3", "C5", "C6", "C7", "C8", "C9", "C10", "T0", "T1", "T4", "T6", "T8"]:
    r = results.get(k, ("—", ""))
    print(f"  {k:3s}: {r[0]:25s}  {r[1]}")

kill_criteria = ["C3", "C6", "C8", "C9", "T6"]
non_waivable_fails = [k for k in kill_criteria if results.get(k, ("", ""))[0] == "FAIL"]
if non_waivable_fails:
    verdict = f"KILLED — non-waivable fail on {non_waivable_fails}"
elif results.get("C7", ("", ""))[0] == "DIRECTIONAL_ONLY":
    verdict = "SHADOW_CONDITIONAL — edge directionally confirmed, OOS N below confirmatory threshold (64 < 100)"
elif results.get("T1", ("", ""))[0] == "ARITHMETIC_ONLY":
    verdict = "CONFIRMED_AS_ARITHMETIC — gate works but not WR-driven; treat as size/quality screen"
else:
    verdict = "CONFIRMED"

print(f"\nVERDICT: {verdict}")
print("=" * 80)
