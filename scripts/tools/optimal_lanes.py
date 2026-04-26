"""Optimal lane selection — best strategy per session x instrument, all variables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import date

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.validated_shelf import deployable_validated_relation

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

# Get #1 strategy per session x instrument, N>=100
rows = con.execute(f"""
    WITH ranked AS (
        SELECT strategy_id, instrument, orb_label, filter_type,
               rr_target, stop_multiplier, sample_size,
               ROUND(win_rate, 4) as wr,
               ROUND(expectancy_r, 4) as expr,
               ROUND(sharpe_ann, 2) as sharpe,
               COALESCE(median_risk_dollars, 0) as med_risk,
               ROUND(COALESCE(median_risk_dollars, 0) * LEAST(COALESCE(stop_multiplier, 1.0), 0.75), 0) as prop_risk,
               ROW_NUMBER() OVER (PARTITION BY orb_label, instrument ORDER BY expectancy_r DESC) as rn
        FROM {deployable_validated_relation(con)}
        WHERE sample_size >= 100
    )
    SELECT strategy_id, instrument, orb_label, prop_risk, expr, sample_size, wr, sharpe
    FROM ranked WHERE rn = 1
    ORDER BY expr DESC
""").fetchall()
con.close()

lanes = {
    (r[2], r[1]): {
        "sid": r[0],
        "inst": r[1],
        "orb": r[2],
        "risk": r[3],
        "expr": r[4],
        "n": r[5],
        "wr": r[6],
        "sharpe": r[7],
    }
    for r in rows
}

# Session times from canonical resolver (DST-aware)
today = date.today()
times = {}
for _name, _cfg in SESSION_CATALOG.items():
    _h, _m = _cfg["resolver"](today)
    times[_name] = _h * 60 + _m

# ═══════════════════════════════════════════════════════════
# UNIVERSE: Every tradeable lane
# ═══════════════════════════════════════════════════════════
print("=" * 85)
print("FULL UNIVERSE — Best strategy per session x instrument (N>=100)")
print("=" * 85)

all_ev = 0
all_risk = 0
sorted_lanes = sorted(lanes.values(), key=lambda x: x["expr"], reverse=True)
print(f"  {'Session':<22s} {'Inst':<4s} {'Risk':>6s} {'ExpR':>7s} {'EV/day':>7s} {'WR':>6s} {'N':>5s}")
print("  " + "-" * 65)
for lane in sorted_lanes:
    ev = lane["risk"] * lane["expr"]
    all_ev += ev
    all_risk += lane["risk"]
    print(
        f"  {lane['orb']:<22s} {lane['inst']:<4s} ${lane['risk']:5.0f} {lane['expr']:7.4f} +${ev:5.1f} {lane['wr'] * 100:5.1f}% {lane['n']:5d}"
    )

print("  " + "-" * 65)
print(f"  UNIVERSE: {len(sorted_lanes)} lanes | risk=${all_risk} | ev=+${all_ev:.1f}/day | ${all_ev * 250:,.0f}/yr")

# ═══════════════════════════════════════════════════════════
# CONFLICT-FREE PROFILES
# ═══════════════════════════════════════════════════════════
# TYPE-A: CME_PRECLOSE side
type_a = [
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "CME_REOPEN",
    "TOKYO_OPEN",
    "LONDON_METALS",
]

# TYPE-B: NYSE_CLOSE side
type_b = [
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "NYSE_CLOSE",
    "CME_REOPEN",
    "SINGAPORE_OPEN",
    "EUROPE_FLOW",
]


def calc_profile(name, sessions):
    profile = []
    for sess in sessions:
        for inst in ACTIVE_ORB_INSTRUMENTS:
            key = (sess, inst)
            if key in lanes:
                profile.append(lanes[key])

    profile.sort(key=lambda x: times.get(x["orb"], 0))
    total_risk = sum(lane["risk"] for lane in profile)
    total_ev = sum(lane["risk"] * lane["expr"] for lane in profile)

    print(f"\n{name}: {len(profile)} lanes across {len(sessions)} sessions")
    print(f"  {'Time':<5s} {'Session':<22s} {'Inst':<4s} {'Risk':>6s} {'ExpR':>7s} {'EV':>7s}")
    print("  " + "-" * 55)
    for lane in profile:
        t = times[lane["orb"]]
        h, m = divmod(t, 60)
        ev = lane["risk"] * lane["expr"]
        print(
            f"  {h:02d}:{m:02d} {lane['orb']:<22s} {lane['inst']:<4s} ${lane['risk']:5.0f} {lane['expr']:7.4f} +${ev:5.1f}"
        )
    print("  " + "-" * 55)
    print(f"  TOTAL: risk=${total_risk} | ev=+${total_ev:.1f}/day | ${total_ev * 250:,.0f}/yr")
    return total_ev, total_risk


print("\n" + "=" * 85)
print("CONFLICT-FREE PROFILES (with multi-instrument stacking)")
print("=" * 85)

ev_a, risk_a = calc_profile("TYPE-A (3 copies)", type_a)
ev_b, risk_b = calc_profile("TYPE-B (2 copies)", type_b)

avg_ev = (ev_a * 3 + ev_b * 2) / 5  # weighted by copy count
avg_risk = (risk_a * 3 + risk_b * 2) / 5

# ═══════════════════════════════════════════════════════════
# FINAL SCALING TABLE
# ═══════════════════════════════════════════════════════════
print("\n\n" + "=" * 85)
print("FINAL SCALING TABLE")
print("=" * 85)

configs = [
    ("TopStep 5x 50K", 5, 2000, 0.85),
    ("TopStep 5x 150K", 5, 4500, 0.85),
    ("TS+TF 10x 50K", 10, 2000, 0.87),
    ("TS+TF 10x 150K", 10, 4500, 0.87),
]

for label, accts, dd, split in configs:
    print(f"\n  {label}:")
    print(f"  {'Cts':>4s} {'DD%':>6s} {'Gross/yr':>12s} {'Net':>12s} {'Real(50%)':>12s} {'Status':>8s}")
    print("  " + "-" * 62)
    for cts in [1, 2, 3, 5, 7, 10]:
        worst = avg_risk * cts
        pct = worst / dd * 100
        gross = avg_ev * cts * 250 * accts
        net = gross * split
        real = net * 0.50
        status = "SAFE" if pct < 50 else ("AGGRO" if pct < 80 else "YOLO")
        print(f"  {cts:4d} {pct:5.0f}%  ${gross:>11,.0f} ${net:>11,.0f} ${real:>11,.0f} {status:>8s}")

# ═══════════════════════════════════════════════════════════
# THE ANSWER
# ═══════════════════════════════════════════════════════════
print("\n\n" + "=" * 85)
print("THE PLAN")
print("=" * 85)

# Phase 1: Start with what we have (TopStep 5x 50K, 2 cts)
p1 = avg_ev * 2 * 250 * 5 * 0.85 * 0.50
# Phase 2: Scale to 5 cts after unlock
p2 = avg_ev * 5 * 250 * 5 * 0.85 * 0.50
# Phase 3: Add Tradeify (10 accounts total)
p3 = avg_ev * 5 * 250 * 10 * 0.87 * 0.50
# Phase 4: Upgrade to 150K (more DD headroom for 7+ cts)
p4 = avg_ev * 7 * 250 * 10 * 0.87 * 0.50
# Phase 5: Max send
p5 = avg_ev * 10 * 250 * 10 * 0.87 * 0.50

print(f"\n  Phase 1 (NOW):     TopStep 5x 50K, 2 cts    -> ${p1:,.0f}/yr")
print(f"  Phase 2 (week 3):  Unlock 50 micro, 5 cts    -> ${p2:,.0f}/yr")
print(f"  Phase 3 (month 2): Add Tradeify 5x, 5 cts    -> ${p3:,.0f}/yr")
print(f"  Phase 4 (month 3): Upgrade to 150K, 7 cts    -> ${p4:,.0f}/yr")
print(f"  Phase 5 (month 4): Max send 10 cts            -> ${p5:,.0f}/yr")
print("\n  All numbers are REALISTIC (50% shrinkage applied)")
print("  + Apex manual daytime sessions on top")
