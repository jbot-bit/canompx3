"""Canonical Bailey-LdP 2014 DSR recomputation for OVNRNG_50_FAST10 finding.

Corrects the DSR number in the PR #44 pre-reg. The original audit used a
simplified E[max|null] approximation sqrt(2 ln K − ln ln K − ln 4π) and
divided by a Mertens single-strategy SR std. That is NOT Bailey's DSR.

Canonical source extract:
    docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md

Bailey-LdP 2014 Eq. 2 (DSR):
    DSR ≡ Φ( (ŜR − ŜR_0) · √(T−1) / √(1 − γ̂₃·ŜR + (γ̂₄−1)/4 · ŜR²) )
    ŜR_0 = √V[{ŜR_n}] · ((1−γ)·Φ⁻¹[1 − 1/N] + γ·Φ⁻¹[1 − 1/(Ne)])

Includes a sanity-check that reproduces Bailey's own numerical example
(page 9-10): SR_ann=2.5, T=1250, V=0.5, skew=-3, kurt=10, N=100 →
DSR ≈ 0.9004 per the paper.
"""

from __future__ import annotations

import sys
from math import e as EULER_E

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EULER_GAMMA = 0.5772156649

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def bailey_expected_max(N: int, trial_sr_variance: float) -> float:
    if N <= 1:
        return 0.0
    term1 = (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1/N)
    term2 = EULER_GAMMA * stats.norm.ppf(1 - 1/(N * EULER_E))
    return float(np.sqrt(trial_sr_variance) * (term1 + term2))


def bailey_dsr(sr_obs: float, sr_0: float, T: int, skew: float, kurt_pearson: float) -> float:
    numerator = (sr_obs - sr_0) * np.sqrt(T - 1)
    denom_sq = 1 - skew * sr_obs + ((kurt_pearson - 1) / 4) * sr_obs ** 2
    if denom_sq <= 0:
        return float("nan")
    dsr_z = numerator / np.sqrt(denom_sq)
    return float(stats.norm.cdf(dsr_z))


print("=" * 80)
print("CANONICAL BAILEY-LDP 2014 DSR — MNQ NYSE_OPEN 5m E2 RR1.0 CB1 OVNRNG_50_FAST10")
print("=" * 80)

q = """
SELECT o.trading_day, o.pnl_r, d.*
FROM orb_outcomes o
JOIN daily_features d
  ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
WHERE o.symbol = 'MNQ' AND o.orb_label = 'NYSE_OPEN' AND o.orb_minutes = 5
  AND o.entry_model = 'E2' AND o.rr_target = 1.0 AND o.confirm_bars = 1
  AND o.pnl_r IS NOT NULL
  AND o.trading_day < '2026-01-01'
ORDER BY o.trading_day
"""
df = DB.execute(q).fetchdf()
sig = filter_signal(df, "OVNRNG_50_FAST10", "NYSE_OPEN")
sel = df.loc[sig == 1].copy()
T = len(sel)
mean_r = sel.pnl_r.mean()
std_r = sel.pnl_r.std(ddof=1)
sr_nonann = mean_r / std_r
skew = float(stats.skew(sel.pnl_r.values))
kurt_pearson = float(stats.kurtosis(sel.pnl_r.values, fisher=False))

print(f"\nSelected strategy (OVNRNG_50_FAST10 IS):")
print(f"  T = {T}  mean={mean_r:+.4f}  std={std_r:.4f}")
print(f"  SR (non-ann) = {sr_nonann:+.4f}  SR (ann) = {sr_nonann * np.sqrt(252):+.3f}")
print(f"  skew γ̂₃ = {skew:+.3f}  kurt γ̂₄ (Pearson) = {kurt_pearson:+.3f}")

# V[{ŜR_n}] from 772-cell scan
from trading_app.config import ALL_FILTERS
sr_samples = []
for orb_m in [5, 15, 30]:
    for rr_t in [1.0, 1.5, 2.0]:
        q2 = """
        SELECT o.trading_day, o.pnl_r, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = 'MNQ' AND o.orb_label = 'NYSE_OPEN' AND o.orb_minutes = ?
          AND o.entry_model = 'E2' AND o.rr_target = ? AND o.confirm_bars = 1
          AND o.pnl_r IS NOT NULL AND o.trading_day < '2026-01-01'
        """
        df_c = DB.execute(q2, [orb_m, rr_t]).fetchdf()
        if len(df_c) < 50:
            continue
        m = df_c.pnl_r.mean(); s = df_c.pnl_r.std(ddof=1)
        if s > 0 and len(df_c) >= 50:
            sr_samples.append(m / s)
        for fk in ALL_FILTERS.keys():
            try:
                s_f = filter_signal(df_c, fk, "NYSE_OPEN")
                sub = df_c.loc[s_f == 1]
                if len(sub) < 50:
                    continue
                m2 = sub.pnl_r.mean(); s2 = sub.pnl_r.std(ddof=1)
                if s2 > 0:
                    sr_samples.append(m2 / s2)
            except Exception:
                continue

sr_arr = np.array(sr_samples)
V_trials = sr_arr.var(ddof=1)
print(f"\nScan cells ≥50: {len(sr_arr)}  V[{{ŜR_n}}]={V_trials:.6f}  √V={np.sqrt(V_trials):.4f}")

print("\n" + "-" * 80)
print("DSR at various N (independent trials)")
print("-" * 80)
print(f"  {'N':>8s} {'E[max] z':>11s} {'SR_0':>8s} {'DSR':>8s} {'Verdict':>8s}")
for N in [46, 100, 772, 2000, 9504, 28000, 50000]:
    z1 = stats.norm.ppf(1 - 1/N)
    z2 = stats.norm.ppf(1 - 1/(N * EULER_E))
    emax_z = (1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2
    sr_0 = np.sqrt(V_trials) * emax_z
    dsr = bailey_dsr(sr_nonann, sr_0, T, skew, kurt_pearson)
    verdict = "PASS" if dsr >= 0.95 else "FAIL"
    print(f"  {N:>8d} {emax_z:>11.4f} {sr_0:>8.4f} {dsr:>8.4f} {verdict:>8s}")

# Bailey's own example as sanity check
print("\n" + "-" * 80)
print("SANITY CHECK — Bailey 2014 numerical example page 9-10")
print("-" * 80)
SR_ann = 2.5
SR_nonann_ex = SR_ann / np.sqrt(250)
T_ex = 1250
V_ex = 0.5
z1 = stats.norm.ppf(1 - 1/100)
z2 = stats.norm.ppf(1 - 1/(100 * EULER_E))
emax_z = (1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2
sr_0_ex = np.sqrt(V_ex / 250) * emax_z
dsr_ex = bailey_dsr(SR_nonann_ex, sr_0_ex, T_ex, -3, 10)
print(f"  Bailey example DSR = {dsr_ex:.4f}")
print(f"  Paper says         = 0.9004")
print(f"  Match: {'YES' if abs(dsr_ex - 0.9004) < 0.01 else 'NO — FORMULA BUG'}")

print("\n" + "=" * 80)
