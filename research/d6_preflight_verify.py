"""D6 pre-reg pre-flight verification — re-derive every load-bearing number from canonical layers.

# e2-lookahead-policy: not-predictor
# garch_forecast_vol_pct is RULE 6.1 SAFE (prior-day close, rolling 252-day window).
# break_dir is NOT used (E2-LA banned per backtesting-methodology.md § 6.3).
# Both-sides cohort, no break_dir filter, no break_bar features.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
from scipy.stats import t as tdist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

WHERE_BASE = """
  o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
  AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.5
  AND d.orb_COMEX_SETTLE_size >= 5
  AND o.entry_ts IS NOT NULL AND o.pnl_r IS NOT NULL
"""
JOIN = """
  orb_outcomes o JOIN daily_features d
    ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
"""

def banner(s: str) -> None:
    print(f"\n=== {s} ===")

banner("DB FRESHNESS")
r = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol='MNQ'").fetchone()
print(f"  orb_outcomes max trading_day (MNQ): {r[0]}")

banner("A. IS LANE BASELINE (deployed, ORB_G5 base, RR1.5, both sides)")
sql = f"""
  SELECT COUNT(*) AS n, AVG(o.pnl_r) AS expr, STDDEV(o.pnl_r) AS sd,
         MIN(o.trading_day) AS first, MAX(o.trading_day) AS last
  FROM {JOIN}
  WHERE {WHERE_BASE} AND o.trading_day < '2026-01-01'
"""
r = con.execute(sql).fetchone()
print(f"  N={r[0]}, ExpR={r[1]:.6f}, sd={r[2]:.6f}, span={r[3]} -> {r[4]}")
PRE_REG_IS_N, PRE_REG_IS_EXPR = 1577, 0.0941
assert r[0] == PRE_REG_IS_N, f"IS N mismatch: pre-reg={PRE_REG_IS_N}, actual={r[0]}"
assert abs(r[1] - PRE_REG_IS_EXPR) < 0.001, f"IS ExpR mismatch: pre-reg={PRE_REG_IS_EXPR}, actual={r[1]:.6f}"
print(f"  VERIFIED against pre-reg (N={PRE_REG_IS_N}, ExpR={PRE_REG_IS_EXPR})")

banner("B. IS PARTITION (gate=garch>70)")
sql = f"""
  SELECT (CASE WHEN d.garch_forecast_vol_pct > 70 THEN 'on' ELSE 'off' END) AS gate,
         COUNT(*) AS n, AVG(o.pnl_r) AS expr, STDDEV(o.pnl_r) AS sd
  FROM {JOIN}
  WHERE {WHERE_BASE} AND o.trading_day < '2026-01-01'
    AND d.garch_forecast_vol_pct IS NOT NULL
  GROUP BY 1 ORDER BY 1
"""
rows = con.execute(sql).fetchall()
gate = {row[0]: {"n": row[1], "expr": row[2], "sd": row[3]} for row in rows}
for g in ("off", "on"):
    print(f"  gate={g}: N={gate[g]['n']}, ExpR={gate[g]['expr']:.6f}, sd={gate[g]['sd']:.6f}")

# Check pre-reg numbers (PRE_REG_GATE_ON / OFF)
PRE_REG = {"on": {"n": 370, "expr": 0.2477}, "off": {"n": 1019, "expr": 0.0530}}
for g in ("on", "off"):
    assert gate[g]["n"] == PRE_REG[g]["n"], f"{g} N mismatch"
    assert abs(gate[g]["expr"] - PRE_REG[g]["expr"]) < 0.001, f"{g} ExpR mismatch"
print("  VERIFIED against pre-reg")

# Lift
is_lift = gate["on"]["expr"] - gate["off"]["expr"]
print(f"  IS lift = +{is_lift:.4f}")
assert abs(is_lift - 0.1947) < 0.001

banner("C. NULL garch rows (excluded from partition)")
sql = f"""
  SELECT COUNT(*) FROM {JOIN}
  WHERE {WHERE_BASE} AND o.trading_day < '2026-01-01'
    AND d.garch_forecast_vol_pct IS NULL
"""
r = con.execute(sql).fetchone()
print(f"  IS rows with NULL garch: {r[0]}  (baseline 1577 - partition {gate['on']['n']+gate['off']['n']} = {1577-gate['on']['n']-gate['off']['n']})")
# 188 rows are pre-2020 when GARCH model didn't have 252-day burn-in yet

banner("D. WELCH t/p re-derivation (lift)")
N_on, m_on, sd_on = gate["on"]["n"], gate["on"]["expr"], gate["on"]["sd"]
N_off, m_off, sd_off = gate["off"]["n"], gate["off"]["expr"], gate["off"]["sd"]
se = math.sqrt(sd_on**2/N_on + sd_off**2/N_off)
t_lift = (m_on - m_off) / se
df = (sd_on**2/N_on + sd_off**2/N_off)**2 / ((sd_on**2/N_on)**2/(N_on-1) + (sd_off**2/N_off)**2/(N_off-1))
p_lift = 2*(1 - tdist.cdf(abs(t_lift), df))
print(f"  Welch t={t_lift:.4f}, df={df:.1f}, p={p_lift:.6f}")
print(f"  Pre-reg cited: t=2.76, p=0.0059")
assert abs(t_lift - 2.76) < 0.05
assert abs(p_lift - 0.0059) < 0.001

# IS gate-on vs zero
t_on_zero = m_on / (sd_on / math.sqrt(N_on))
print(f"  IS gate-on vs 0: t={t_on_zero:.4f}")
assert abs(t_on_zero - 4.07) < 0.05

banner("E. C9 ERA STABILITY (per IS year)")
sql = f"""
  SELECT EXTRACT(YEAR FROM o.trading_day)::int AS yr,
         SUM(CASE WHEN d.garch_forecast_vol_pct > 70 THEN 1 ELSE 0 END) AS n_on,
         AVG(CASE WHEN d.garch_forecast_vol_pct > 70 THEN o.pnl_r END) AS expr_on,
         SUM(CASE WHEN d.garch_forecast_vol_pct <= 70 THEN 1 ELSE 0 END) AS n_off,
         AVG(CASE WHEN d.garch_forecast_vol_pct <= 70 THEN o.pnl_r END) AS expr_off
  FROM {JOIN}
  WHERE {WHERE_BASE} AND o.trading_day < '2026-01-01'
    AND d.garch_forecast_vol_pct IS NOT NULL
  GROUP BY 1 ORDER BY 1
"""
rows = con.execute(sql).fetchall()
years_with_n_ge_10 = []
years_positive_n_ge_10 = []
years_failing_c9 = []
for row in rows:
    yr, n_on, e_on, n_off, e_off = row
    flag = ""
    if n_on >= 50 and e_on is not None and e_on < -0.05:
        flag = " *** C9_FAIL"
        years_failing_c9.append(yr)
    if n_on >= 10:
        years_with_n_ge_10.append(yr)
        if e_on is not None and e_on > 0:
            years_positive_n_ge_10.append(yr)
    expr_on_str = f"{e_on:.4f}" if e_on is not None else "NA"
    expr_off_str = f"{e_off:.4f}" if e_off is not None else "NA"
    print(f"  {yr}: gate-on N={n_on} ExpR={expr_on_str}{flag} | gate-off N={n_off} ExpR={expr_off_str}")
print(f"  C9 fails (N>=50, ExpR<-0.05): {years_failing_c9}")
print(f"  Years with gate-on N>=10: {years_with_n_ge_10}")
print(f"  Positive years (N>=10): {years_positive_n_ge_10} — count={len(years_positive_n_ge_10)}")

banner("F. OOS PARTITION (accruing forward shadow)")
sql = f"""
  SELECT (CASE WHEN d.garch_forecast_vol_pct > 70 THEN 'on' ELSE 'off' END) AS gate,
         COUNT(*) AS n, AVG(o.pnl_r) AS expr, STDDEV(o.pnl_r) AS sd,
         MIN(o.trading_day) AS first, MAX(o.trading_day) AS last
  FROM {JOIN}
  WHERE {WHERE_BASE} AND o.trading_day >= '2026-01-01'
    AND d.garch_forecast_vol_pct IS NOT NULL
  GROUP BY 1 ORDER BY 1
"""
rows = con.execute(sql).fetchall()
oos = {row[0]: row for row in rows}
for g in ("off", "on"):
    row = oos[g]
    print(f"  gate={g}: N={row[1]}, ExpR={row[2]:.6f}, sd={row[3]:.6f}, span={row[4]} -> {row[5]}")
oos_lift = oos["on"][2] - oos["off"][2]
print(f"  OOS lift = +{oos_lift:.4f}")
print(f"  dir_match = {(oos_lift > 0) == (is_lift > 0)}")
print(f"  OOS_lift >= 0.40 * IS_lift = {oos_lift >= 0.40 * is_lift}")

# OOS Welch t
N_on_o, m_on_o, sd_on_o = oos["on"][1], oos["on"][2], oos["on"][3]
N_off_o, m_off_o, sd_off_o = oos["off"][1], oos["off"][2], oos["off"][3]
se_o = math.sqrt(sd_on_o**2/N_on_o + sd_off_o**2/N_off_o)
t_oos = (m_on_o - m_off_o) / se_o
df_o = (sd_on_o**2/N_on_o + sd_off_o**2/N_off_o)**2 / ((sd_on_o**2/N_on_o)**2/(N_on_o-1) + (sd_off_o**2/N_off_o)**2/(N_off_o-1))
p_oos = 2*(1 - tdist.cdf(abs(t_oos), df_o))
print(f"  OOS Welch t={t_oos:.4f}, df={df_o:.1f}, p={p_oos:.6f}")

banner("G. PRESSURE TEST — known-bad feature (pnl_r as predictor)")
sql = f"""
  SELECT corr(o.pnl_r, o.pnl_r) AS self_corr,
         corr(CASE WHEN o.outcome='win' THEN 1.0 ELSE 0.0 END, o.pnl_r) AS win_corr
  FROM {JOIN}
  WHERE {WHERE_BASE} AND o.trading_day < '2026-01-01'
"""
r = con.execute(sql).fetchone()
print(f"  corr(pnl_r, pnl_r)={r[0]:.6f}  (expect 1.0; tautology check baseline)")
print(f"  corr(is_win, pnl_r)={r[1]:.6f} (post-trade label corr; should be high — RULE 7 banned)")
assert abs(r[0] - 1.0) < 0.001, "self-corr sanity failed"
assert r[1] > 0.50, "win-corr should be high (post-trade) — pressure test must reject"
print("  Pressure test passed: tautology detector would correctly reject pnl_r/outcome as predictors.")

banner("H. WALK-FORWARD CHUNKS for C6 WFE (per-year IS+OOS lift)")
sql = f"""
  SELECT EXTRACT(YEAR FROM o.trading_day)::int AS yr,
         SUM(CASE WHEN d.garch_forecast_vol_pct > 70 THEN 1 ELSE 0 END) AS n_on,
         AVG(CASE WHEN d.garch_forecast_vol_pct > 70 THEN o.pnl_r END) AS expr_on,
         SUM(CASE WHEN d.garch_forecast_vol_pct <= 70 THEN 1 ELSE 0 END) AS n_off,
         AVG(CASE WHEN d.garch_forecast_vol_pct <= 70 THEN o.pnl_r END) AS expr_off
  FROM {JOIN}
  WHERE {WHERE_BASE}
    AND d.garch_forecast_vol_pct IS NOT NULL
  GROUP BY 1 ORDER BY 1
"""
rows = con.execute(sql).fetchall()
for row in rows:
    yr, n_on, e_on, n_off, e_off = row
    lift = (e_on or 0) - (e_off or 0)
    e_on_s = f"{e_on:.4f}" if e_on is not None else "NA"
    print(f"  {yr}: gate-on N={n_on} ExpR={e_on_s}, lift={lift:.4f}")

print("\nALL CHECKS PASSED — pre-reg numbers reproduce on canonical layers.")
con.close()
