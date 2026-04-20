"""L4 NYSE_OPEN COST_LT12 decay audit (confirmatory).

Claim under test: "L4 MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 is decaying live
based on SR ALARM status; strategy should be paused / reviewed."

Follows .claude/rules/quant-audit-protocol.md STEP 1-6 plus the user's
NO BIAS / NO TUNNEL / FULL TRUTH AUDIT 8-step overlay.

Reads canonical layers only (orb_outcomes, daily_features, paper_trades).
No derived layers. Idempotent, read-only.
"""

from __future__ import annotations

import sys
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")
L4_SID = "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12"
SESSION = "NYSE_OPEN"
INSTRUMENT = "MNQ"
RR = 1.0
ORB_MIN = 5
ENTRY = "E2"
CB = 1

# -------- Pre-flight --------
print("=" * 80)
print(f"AUDIT: L4 NYSE_OPEN COST_LT12 decay claim  (ran at {pd.Timestamp.now('UTC')})")
print("=" * 80)

preflight = DB.execute("""
    SELECT MAX(trading_day) AS max_day, COUNT(*) AS n
    FROM orb_outcomes WHERE symbol = ?
""", [INSTRUMENT]).fetchone()
print(f"\n[PREFLIGHT] orb_outcomes {INSTRUMENT}: max={preflight[0]}  n={preflight[1]:,}")

pt = DB.execute("""
    SELECT COUNT(*), MIN(trading_day), MAX(trading_day) FROM paper_trades
    WHERE strategy_id = ?
""", [L4_SID]).fetchone()
print(f"[PREFLIGHT] paper_trades L4: n={pt[0]}  range={pt[1]} → {pt[2]}")


# -------- Pull canonical MNQ NYSE_OPEN E2 RR1.0 CB1 5m outcomes joined to features --------
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


df_full = load_lane(INSTRUMENT, SESSION, ORB_MIN, ENTRY, RR, CB)
print(f"\n[LOAD] MNQ {SESSION} 5m E2 RR1.0 CB1 unfiltered: n={len(df_full)}  "
      f"range={df_full.trading_day.min()} → {df_full.trading_day.max()}")

# Apply COST_LT12 via canonical filter
sig = filter_signal(df_full, "COST_LT12", SESSION)
df_l4 = df_full.loc[sig == 1].copy().reset_index(drop=True)
print(f"[LOAD] COST_LT12 fires: {len(df_l4)}  fire_rate={len(df_l4) / len(df_full):.3f}")


# -------- T0: Tautology — COST_LT12 vs ORB_G5 and others at NYSE_OPEN --------
print("\n" + "-" * 80)
print("T0 TAUTOLOGY CHECK — COST_LT12 vs deployed filters at NYSE_OPEN")
print("-" * 80)
candidates = ["ORB_G5", "ORB_G8", "ATR_P50", "COST_LT08", "COST_LT10", "COST_LT15"]
for key in candidates:
    try:
        s2 = filter_signal(df_full, key, SESSION)
        if s2.sum() == 0:
            print(f"  {key:10s}: fires=0 (skip)")
            continue
        c = np.corrcoef(sig, s2)[0, 1]
        print(f"  COST_LT12 vs {key:10s}: corr={c:+.4f}  fires={int(s2.sum())}")
    except Exception as e:
        print(f"  {key}: err {e}")


# -------- T1: WR Monotonicity on cost ratio bins --------
print("\n" + "-" * 80)
print("T1 WR MONOTONICITY — does WR improve or only payoff improve across cost bins?")
print("-" * 80)
# cost_ratio_pct at NYSE_OPEN — find the column
cols = [c for c in df_full.columns if "cost" in c.lower() or "risk" in c.lower()]
print(f"  cost-related cols available: {cols[:15]}")
risk_col = None
for candidate in ["orb_NYSE_OPEN_cost_ratio_pct", "orb_NYSE_OPEN_risk_ratio_pct",
                  "orb_NYSE_OPEN_risk_pct", "risk_pct"]:
    if candidate in df_full.columns:
        risk_col = candidate
        break
print(f"  using col: {risk_col}")
if risk_col:
    is_df = df_full[df_full.trading_day < HOLDOUT].copy()
    is_df = is_df.dropna(subset=[risk_col])
    is_df["bin"] = pd.qcut(is_df[risk_col], 5, labels=["Q1 lo", "Q2", "Q3", "Q4", "Q5 hi"], duplicates="drop")
    for b, grp in is_df.groupby("bin", observed=True):
        wr = grp.win.mean()
        expr = grp.pnl_r.mean()
        print(f"  {b}: n={len(grp):4d}  WR={wr:.3f}  ExpR={expr:+.4f}  avg_risk={grp[risk_col].mean():.3f}")


# -------- T2: IS baseline for L4 (COST_LT12 filtered) --------
print("\n" + "-" * 80)
print("T2 IS BASELINE — L4 on 2019-01-01 to 2025-12-31 (strict Mode A IS)")
print("-" * 80)
is_l4 = df_l4[df_l4.trading_day < HOLDOUT]
oos_l4 = df_l4[df_l4.trading_day >= HOLDOUT]
print(f"  IS:  n={len(is_l4):4d}  WR={is_l4.win.mean():.3f}  ExpR={is_l4.pnl_r.mean():+.4f}  std={is_l4.pnl_r.std():.4f}")
print(f"  OOS: n={len(oos_l4):4d}  WR={oos_l4.win.mean():.3f}  ExpR={oos_l4.pnl_r.mean():+.4f}  std={oos_l4.pnl_r.std():.4f}")
if len(is_l4) > 30 and len(oos_l4) > 5:
    is_sharpe = is_l4.pnl_r.mean() / is_l4.pnl_r.std() * np.sqrt(252)
    oos_sharpe = oos_l4.pnl_r.mean() / oos_l4.pnl_r.std() * np.sqrt(252)
    wfe = oos_sharpe / is_sharpe if is_sharpe != 0 else float("nan")
    print(f"  IS Sharpe={is_sharpe:+.3f}  OOS Sharpe={oos_sharpe:+.3f}  WFE={wfe:+.3f}")
    # dir_match
    dir_match = np.sign(is_l4.pnl_r.mean()) == np.sign(oos_l4.pnl_r.mean())
    print(f"  dir_match: {dir_match}")


# -------- T3/T6: Bootstrap null + IS t-stat --------
print("\n" + "-" * 80)
print("T6 BOOTSTRAP NULL — shuffle pnl_r 2000x, compute ExpR distribution")
print("-" * 80)
if len(is_l4) > 30:
    rng = np.random.default_rng(42)
    pool = is_l4.pnl_r.values
    null_means = rng.choice(pool, size=(2000, len(is_l4)), replace=True).mean(axis=1)
    observed = is_l4.pnl_r.mean()
    p95 = np.percentile(null_means, 95)
    p_val = ((null_means >= observed).sum() + 1) / 2001
    t_stat = observed / (is_l4.pnl_r.std() / np.sqrt(len(is_l4)))
    print(f"  Observed IS ExpR={observed:+.4f}")
    print(f"  Bootstrap P95={p95:+.4f}  p-value={p_val:.4f}  t-stat={t_stat:+.3f}")
    print(f"  Chordia-strict (t≥3.79): {'PASS' if abs(t_stat) >= 3.79 else 'FAIL'}")


# -------- T7: Per-year stability --------
print("\n" + "-" * 80)
print("T7 PER-YEAR STABILITY — L4 (IS only)")
print("-" * 80)
if len(is_l4) > 30:
    is_l4_y = is_l4.copy()
    is_l4_y["year"] = pd.to_datetime(is_l4_y["trading_day"]).dt.year
    for y, grp in is_l4_y.groupby("year"):
        wr = grp.win.mean()
        expr = grp.pnl_r.mean()
        mark = " +" if expr > 0 else " -"
        print(f"  {y}: n={len(grp):4d}  WR={wr:.3f}  ExpR={expr:+.4f} {mark}")


# -------- T4: Sensitivity — COST_LT08/10/12/15 at same lane --------
print("\n" + "-" * 80)
print("T4 SENSITIVITY — COST_LT threshold variants on same MNQ NYSE_OPEN 5m E2 RR1.0 CB1 lane")
print("-" * 80)
for variant in ["COST_LT08", "COST_LT10", "COST_LT12", "COST_LT15"]:
    try:
        s = filter_signal(df_full, variant, SESSION)
        subset = df_full.loc[s == 1].copy()
        is_s = subset[subset.trading_day < HOLDOUT]
        oos_s = subset[subset.trading_day >= HOLDOUT]
        if len(is_s) == 0:
            print(f"  {variant}: n=0")
            continue
        print(f"  {variant}: IS n={len(is_s):3d} WR={is_s.win.mean():.3f} ExpR={is_s.pnl_r.mean():+.4f}  |  "
              f"OOS n={len(oos_s):3d} WR={oos_s.win.mean() if len(oos_s) else float('nan'):.3f} "
              f"ExpR={oos_s.pnl_r.mean() if len(oos_s) else float('nan'):+.4f}")
    except Exception as e:
        print(f"  {variant}: err {e}")


# -------- T5/T8: Family — COST_LT12 at other sessions + other filters at NYSE_OPEN + cross-instrument --------
print("\n" + "-" * 80)
print("T5 FAMILY SCAN — COST_LT12 across sessions (MNQ 5m E2 RR1.0 CB1)")
print("-" * 80)
from pipeline.dst import SESSION_CATALOG
for sess in sorted(SESSION_CATALOG.keys()):
    df_s = load_lane(INSTRUMENT, sess, ORB_MIN, ENTRY, RR, CB)
    if len(df_s) < 30:
        continue
    try:
        s = filter_signal(df_s, "COST_LT12", sess)
        sub = df_s.loc[s == 1]
        is_sub = sub[sub.trading_day < HOLDOUT]
        oos_sub = sub[sub.trading_day >= HOLDOUT]
        if len(is_sub) < 30:
            continue
        print(f"  {sess:15s}: IS n={len(is_sub):4d} WR={is_sub.win.mean():.3f} ExpR={is_sub.pnl_r.mean():+.4f}  "
              f"| OOS n={len(oos_sub):3d} ExpR={oos_sub.pnl_r.mean() if len(oos_sub) else float('nan'):+.4f}")
    except Exception as e:
        print(f"  {sess}: err {e}")


# -------- T8: Cross-instrument --------
print("\n" + "-" * 80)
print("T8 CROSS-INSTRUMENT — NYSE_OPEN 5m E2 RR1.0 CB1 COST_LT12")
print("-" * 80)
for inst in ["MNQ", "MGC", "MES"]:
    df_i = load_lane(inst, SESSION, ORB_MIN, ENTRY, RR, CB)
    if len(df_i) < 30:
        print(f"  {inst}: n<30 skip")
        continue
    try:
        s = filter_signal(df_i, "COST_LT12", SESSION)
        sub = df_i.loc[s == 1]
        is_sub = sub[sub.trading_day < HOLDOUT]
        oos_sub = sub[sub.trading_day >= HOLDOUT]
        print(f"  {inst}: IS n={len(is_sub):3d} ExpR={is_sub.pnl_r.mean() if len(is_sub) else float('nan'):+.4f}  "
              f"| OOS n={len(oos_sub):3d} ExpR={oos_sub.pnl_r.mean() if len(oos_sub) else float('nan'):+.4f}")
    except Exception as e:
        print(f"  {inst}: err {e}")


# -------- LIVE-DECAY specific: two-sample t-test, baseline-50 vs monitored-21 --------
print("\n" + "-" * 80)
print("LIVE-DECAY: Welch two-sample t-test, baseline-50 vs monitored-21 paper_trades")
print("-" * 80)
pt_df = DB.execute("""
    SELECT trading_day, pnl_r FROM paper_trades
    WHERE strategy_id = ?
    ORDER BY trading_day, entry_time
""", [L4_SID]).fetchdf()
baseline = pt_df.iloc[:50].pnl_r.values
monitored = pt_df.iloc[50:].pnl_r.values
from scipy import stats
t, p = stats.ttest_ind(monitored, baseline, equal_var=False)
print(f"  baseline n={len(baseline)} mean={baseline.mean():+.4f} std={baseline.std():.4f}")
print(f"  monitored n={len(monitored)} mean={monitored.mean():+.4f} std={monitored.std():.4f}")
print(f"  Welch t={t:+.3f}  p={p:.4f}  ({'REJECT null — real drift' if p < 0.05 else 'FAIL to reject — no significant drift'})")


# -------- Missed-edge scan: NYSE_OPEN × aperture × RR × filter families (descriptive only) --------
print("\n" + "=" * 80)
print("ADJACENT-CELL REFERENCE FRAME — NYSE_OPEN × aperture × RR × filter (descriptive, no promotion)")
print("=" * 80)
from trading_app.config import ALL_FILTERS
filter_keys = [k for k in ALL_FILTERS.keys() if not k.startswith("X_") and "MANUAL" not in k]
print(f"  filter keys to scan: {len(filter_keys)}")

rows = []
for orb_m in [5, 15, 30]:
    for rr_t in [1.0, 1.5, 2.0]:
        df_c = load_lane(INSTRUMENT, SESSION, orb_m, ENTRY, rr_t, CB)
        if len(df_c) < 50:
            continue
        # Always include the unfiltered baseline
        is_base = df_c[df_c.trading_day < HOLDOUT]
        oos_base = df_c[df_c.trading_day >= HOLDOUT]
        if len(is_base) >= 50:
            rows.append({
                "orb_m": orb_m, "rr": rr_t, "filter": "NONE",
                "is_n": len(is_base), "is_expr": is_base.pnl_r.mean(), "is_wr": is_base.win.mean(),
                "oos_n": len(oos_base), "oos_expr": oos_base.pnl_r.mean() if len(oos_base) else float("nan"),
                "oos_wr": oos_base.win.mean() if len(oos_base) else float("nan"),
            })
        # Filtered variants
        for fk in filter_keys:
            try:
                s = filter_signal(df_c, fk, SESSION)
                sub = df_c.loc[s == 1]
                is_sub = sub[sub.trading_day < HOLDOUT]
                oos_sub = sub[sub.trading_day >= HOLDOUT]
                if len(is_sub) < 50:
                    continue
                rows.append({
                    "orb_m": orb_m, "rr": rr_t, "filter": fk,
                    "is_n": len(is_sub), "is_expr": is_sub.pnl_r.mean(), "is_wr": is_sub.win.mean(),
                    "oos_n": len(oos_sub), "oos_expr": oos_sub.pnl_r.mean() if len(oos_sub) else float("nan"),
                    "oos_wr": oos_sub.win.mean() if len(oos_sub) else float("nan"),
                })
            except Exception:
                continue

scan_df = pd.DataFrame(rows)
if not scan_df.empty:
    # dir_match IS and OOS
    scan_df["dir_match"] = np.sign(scan_df.is_expr) == np.sign(scan_df.oos_expr)
    # Top 25 IS
    print(f"\n  Total cells with N≥50: {len(scan_df)}")
    print("\n  Top 20 IS ExpR (descriptive, no BH-FDR — no promotion):")
    top = scan_df.sort_values("is_expr", ascending=False).head(20)
    for _, r in top.iterrows():
        dm = "Y" if r["dir_match"] else "N"
        print(f"    orb={int(r['orb_m']):2d} RR={r['rr']:.1f} {str(r['filter']):25s}  "
              f"IS n={int(r['is_n']):4d} WR={r['is_wr']:.3f} ExpR={r['is_expr']:+.4f}  |  "
              f"OOS n={int(r['oos_n']):4d} WR={r['oos_wr']:.3f} ExpR={r['oos_expr']:+.4f}  dir_match={dm}")

    # Where does L4 rank?
    l4_row = scan_df[(scan_df.orb_m == 5) & (scan_df.rr == 1.0) & (scan_df["filter"] == "COST_LT12")]
    if not l4_row.empty:
        rank = (scan_df.is_expr > l4_row.is_expr.iloc[0]).sum() + 1
        print(f"\n  L4 (COST_LT12, 5m, RR1.0) IS rank: {rank} / {len(scan_df)}   ExpR={l4_row.is_expr.iloc[0]:+.4f}")
        print(f"  L4 OOS: n={int(l4_row.oos_n.iloc[0])} ExpR={l4_row.oos_expr.iloc[0]:+.4f} dir_match={bool(l4_row.dir_match.iloc[0])}")

print("\n" + "=" * 80)
print("AUDIT END")
print("=" * 80)
