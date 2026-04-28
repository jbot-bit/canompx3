"""Extract golden nuggets from break delay data — O5 only, per methodology rules.

# e2-lookahead-policy: tainted
# break_delay_min (orb_{sess}_break_delay_min) is used as a predictor of pnl_r on E2 entries.
# On E2, ~41% of trades have entry_ts < break_ts, making break_delay_min post-entry on that
# subset. All "golden nuggets" found here for E2 lanes are unreliable. Clean re-derivation
# required using pre-break features before any finding can be cited.
# Registry: docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md
"""

import numpy as np
import duckdb
from scipy import stats
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

print("=" * 100)
print("GOLDEN NUGGETS FROM BREAK DELAY DATA (O5 ONLY, E2/CB1/RR1.0, NO FILTER)")
print("=" * 100)

sessions = [
    "CME_REOPEN",
    "TOKYO_OPEN",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "NYSE_OPEN",
    "NYSE_CLOSE",
    "COMEX_SETTLE",
    "US_DATA_830",
    "US_DATA_1000",
    "CME_PRECLOSE",
]

print()
print(
    f"{'Inst':<5s} {'Session':<20s} {'Fast_mean':>10s} {'Fast_N':>7s} "
    f"{'Slow_mean':>10s} {'Slow_N':>7s} {'d':>7s} {'p':>8s} {'Dir':>10s} {'Note':>6s}"
)
print("-" * 100)

nuggets = []

for inst in ["MNQ", "MGC", "MES"]:
    for sess in sessions:
        delay_col = f"orb_{sess}_break_delay_min"
        try:
            r = con.sql(f"""
                SELECT o.pnl_r, d."{delay_col}" as delay
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day
                  AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = '{inst}'
                  AND d.orb_minutes = 5
                  AND d."{delay_col}" IS NOT NULL
                  AND o.entry_model = 'E2'
                  AND o.confirm_bars = 1
                  AND o.rr_target = 1.0
            """).fetchnumpy()
        except Exception:
            continue

        pnl = np.array(r["pnl_r"])
        delay = np.array(r["delay"])

        fast = pnl[delay <= 5]
        slow = pnl[delay > 15]

        if len(fast) < 30 or len(slow) < 30:
            continue

        n1, n2 = len(fast), len(slow)
        pooled = np.sqrt(((n1 - 1) * fast.std(ddof=1) ** 2 + (n2 - 1) * slow.std(ddof=1) ** 2) / (n1 + n2 - 2))
        d = (fast.mean() - slow.mean()) / pooled if pooled > 0 else 0
        t, p = stats.ttest_ind(fast, slow, equal_var=False)

        direction = "FAST>SLOW" if d > 0 else "SLOW>FAST"
        marker = "***" if abs(d) >= 0.2 and p < 0.05 else ("*" if abs(d) >= 0.1 and p < 0.05 else "")

        print(
            f"{inst:<5s} {sess:<20s} {fast.mean():>+10.4f} {n1:>7d} "
            f"{slow.mean():>+10.4f} {n2:>7d} {d:>+7.3f} {p:>8.4f} {direction:>10s} {marker:>6s}"
        )

        nuggets.append(
            {
                "inst": inst,
                "sess": sess,
                "d": d,
                "p": p,
                "fast_mean": fast.mean(),
                "slow_mean": slow.mean(),
                "fast_n": n1,
                "slow_n": n2,
                "direction": direction,
            }
        )

# Summary
print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

# Sessions where FAST is consistently better across instruments
print()
print("FAST > SLOW across ALL 3 instruments (consistent direction):")
from collections import defaultdict

session_dirs = defaultdict(list)
for n in nuggets:
    session_dirs[n["sess"]].append(n["d"])

for sess, ds in sorted(session_dirs.items()):
    if len(ds) == 3:
        all_pos = all(d > 0 for d in ds)
        all_neg = all(d < 0 for d in ds)
        avg_d = np.mean(ds)
        if all_pos:
            print(f"  {sess:<20s} avg d={avg_d:+.3f}  FAST better on all 3")
        elif all_neg:
            print(f"  {sess:<20s} avg d={avg_d:+.3f}  SLOW better on all 3 (REVERSE)")

# Sessions where direction is inconsistent
print()
print("INCONSISTENT direction across instruments:")
for sess, ds in sorted(session_dirs.items()):
    if len(ds) == 3:
        signs = [d > 0 for d in ds]
        if not all(signs) and any(signs):
            print(f"  {sess:<20s} d values: {[f'{d:+.3f}' for d in ds]}")

# The actual nugget: which sessions show the STRONGEST fast-break advantage?
print()
print("STRONGEST effects (|d| > 0.1, consistent direction, p < 0.01):")
strong = [n for n in nuggets if abs(n["d"]) >= 0.1 and n["p"] < 0.01]
for n in sorted(strong, key=lambda x: abs(x["d"]), reverse=True):
    print(
        f"  {n['inst']} {n['sess']:<20s} d={n['d']:+.3f}  {n['direction']}  "
        f"FAST={n['fast_mean']:+.4f} SLOW={n['slow_mean']:+.4f}"
    )

con.close()
