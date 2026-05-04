"""OVNRNG Q3-Q4 sweet-spot cross-session replication check.

Diagnostic: does PR #47's finding (on NYSE_OPEN E2 RR1.0 CB1) replicate
across sessions? If Q3-Q4 > Q5 pattern holds on ≥3 sessions, a
cross-session Pathway-B pre-reg is worth writing.

Rule 1.2 (backtesting-methodology.md): overnight_range is valid ONLY
for ORB sessions starting ≥17:00 Brisbane. Earlier sessions fire
before the overnight window (09:00-17:00 Brisbane) closes, so
overnight_range leaks future data. Those sessions are SKIPPED here.

Canonical truth only: orb_outcomes and daily_features (triple-joined).
Read-only. No production code touched.
"""

from __future__ import annotations

import sys

import duckdb
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")

# Sessions where overnight_range is look-ahead CLEAN (start ≥17:00 Brisbane per
# backtesting-methodology.md Rule 1.2). Sessions starting BEFORE 17:00 exclude
# because overnight_range 09:00-17:00 Brisbane window is still open at their ORB.
# Canonical start times from pipeline/dst.py SESSION_CATALOG are dynamic/event-
# driven; 17:00 cutoff per Rule 1.2 table.
ELIGIBLE_SESSIONS = [
    "LONDON_METALS",     # ~17:00
    "EUROPE_FLOW",       # ~18:00
    "US_DATA_830",       # ~23:30
    "NYSE_OPEN",         # ~00:30 next day
    "US_DATA_1000",      # ~01:00 next day
    "COMEX_SETTLE",      # ~04:30
    "CME_PRECLOSE",      # ~06:00
    "NYSE_CLOSE",        # ~07:00
]

# Sessions EXCLUDED (look-ahead risk): CME_REOPEN (~08:00), TOKYO_OPEN (~10:00),
# SINGAPORE_OPEN (~11:00), BRISBANE_1025 (~10:25).


def _welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a.values, b.values, equal_var=False)
    return float(t), float(p)


def audit_session(session: str) -> dict:
    q = """
    SELECT o.trading_day, o.pnl_r,
           d.overnight_range, d.atr_20
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = ?
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.rr_target = 1.5
      AND o.confirm_bars = 1
      AND o.pnl_r IS NOT NULL
      AND o.trading_day < ?
      AND d.overnight_range IS NOT NULL
      AND d.atr_20 IS NOT NULL
      AND d.atr_20 > 0
    ORDER BY o.trading_day
    """
    df = DB.execute(q, [session, str(HOLDOUT.date())]).fetchdf()
    if len(df) < 100:
        return {"session": session, "n": len(df), "skipped": "insufficient_n"}

    df["ovn_atr"] = df["overnight_range"] / df["atr_20"]
    bins = pd.qcut(df["ovn_atr"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                   duplicates="drop")
    df["bin"] = bins

    stats_rows = []
    for b, g in df.groupby("bin", observed=True):
        stats_rows.append({
            "bin": str(b),
            "n": len(g),
            "ovn_atr_mean": float(g["ovn_atr"].mean()),
            "wr": float((g["pnl_r"] > 0).mean()),
            "expr": float(g["pnl_r"].mean()),
        })
    stats_rows.sort(key=lambda r: r["bin"])

    # Q3+Q4 pooled vs Q5
    q34 = df[df.bin.isin(["Q3", "Q4"])]["pnl_r"]
    q5 = df[df.bin == "Q5"]["pnl_r"]
    t, p = _welch(q34, q5)
    delta = float(q34.mean()) - float(q5.mean())
    q34_mean = float(q34.mean())
    q5_mean = float(q5.mean())

    # Classification
    if delta >= 0.05:
        verdict = "SWEET_SPOT_PRESENT"
    elif delta <= -0.05:
        verdict = "INVERSE"
    else:
        verdict = "NO_PATTERN"

    return {
        "session": session, "n": len(df),
        "per_bin": stats_rows,
        "q34_mean": q34_mean, "q5_mean": q5_mean, "delta_q34_minus_q5": delta,
        "welch_t": t, "welch_p": p,
        "verdict": verdict,
    }


def main() -> None:
    print("=" * 80)
    print("OVNRNG Q3-Q4 SWEET-SPOT CROSS-SESSION REPLICATION CHECK")
    print(f"ran {pd.Timestamp.now('UTC')}")
    print("IS only (trading_day < 2026-01-01).  MNQ E2 RR=1.5 CB=1 orb_minutes=5.")
    print(f"Sessions tested: {len(ELIGIBLE_SESSIONS)} (overnight_range lookahead-clean)")
    print("=" * 80)

    results = []
    for session in ELIGIBLE_SESSIONS:
        print(f"\n--- {session} ---")
        r = audit_session(session)
        if "skipped" in r:
            print(f"  SKIPPED: {r['skipped']} (n={r['n']})")
            results.append(r)
            continue

        print(f"  n={r['n']}")
        print(f"  {'bin':5s} {'n':>4s} {'ovn/atr μ':>10s} {'WR':>6s} {'ExpR':>8s}")
        for row in r["per_bin"]:
            print(f"  {row['bin']:5s} {row['n']:>4d} {row['ovn_atr_mean']:>10.3f} "
                  f"{row['wr']*100:>5.1f}% {row['expr']:>+8.4f}")
        print(f"  Q3+Q4 mean = {r['q34_mean']:+.4f}  Q5 mean = {r['q5_mean']:+.4f}  "
              f"Δ(Q34−Q5) = {r['delta_q34_minus_q5']:+.4f}  "
              f"Welch t={r['welch_t']:+.2f} p={r['welch_p']:.4f}")
        print(f"  Verdict: {r['verdict']}")
        results.append(r)

    # Rollup
    print("\n" + "=" * 80)
    print("ROLLUP")
    print("=" * 80)
    counts: dict[str, int] = {}
    for r in results:
        v = r.get("verdict", "SKIPPED")
        counts[v] = counts.get(v, 0) + 1
    for v, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v}: {n}")

    print(f"\n  {'Session':15s} {'Verdict':20s} {'Δ':>8s} {'Welch p':>8s}")
    for r in results:
        d = r.get("delta_q34_minus_q5", float("nan"))
        p = r.get("welch_p", float("nan"))
        v = r.get("verdict", "SKIPPED")
        print(f"  {r['session']:15s} {v:20s} {d:>+8.4f} {p:>8.4f}")


if __name__ == "__main__":
    main()
