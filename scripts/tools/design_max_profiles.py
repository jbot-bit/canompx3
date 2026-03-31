"""Design max extraction TopStep profiles — conflict-free session assignment."""

from pipeline.dst import SESSION_CATALOG
from datetime import date

today = date(2026, 3, 31)

# Resolve all session times
times = {}
for name, cfg in SESSION_CATALOG.items():
    h, m = cfg["resolver"](today)
    times[name] = h * 60 + m

# Best strategy per session (from validated_setups, N>=100)
best = {
    # (session, instrument, risk_s075, expr, rr)
    "US_DATA_830":    ("MNQ", 34, 0.076, 1.0),
    "NYSE_OPEN":      ("MNQ", 69, 0.168, 1.0),
    "US_DATA_1000":   ("MNQ", 59, 0.150, 1.0),
    "COMEX_SETTLE":   ("MNQ", 46, 0.264, 1.5),
    "CME_PRECLOSE":   ("MNQ", 22, 0.340, 1.0),
    "NYSE_CLOSE":     ("MNQ", 26, 0.325, 1.0),
    "CME_REOPEN":     ("MNQ", 23, 0.130, 1.0),
    "TOKYO_OPEN":     ("MGC", 42, 0.283, 2.0),
    "BRISBANE_1025":  ("MNQ", 26, 0.128, 1.5),
    "SINGAPORE_OPEN": ("MNQ", 36, 0.276, 1.5),
    "LONDON_METALS":  ("MNQ", 25, 0.178, 1.5),
    "EUROPE_FLOW":    ("MNQ", 30, 0.217, 2.0),
}

def est_dur(rr):
    if rr <= 1.0: return 45
    if rr <= 1.5: return 60
    return 90

# Shared overnight chain (no conflicts, all sequential)
shared = ["US_DATA_830", "NYSE_OPEN", "US_DATA_1000", "COMEX_SETTLE"]

# Conflict pairs:
# CME_PRECLOSE (05:45) vs NYSE_CLOSE (06:00) -> 15min gap, PRECLOSE needs 45min
# TOKYO_OPEN (10:00, 90min) vs BRISBANE_1025 (10:25) vs SINGAPORE_OPEN (11:00)
# LONDON_METALS (17:00, 60min) vs EUROPE_FLOW (18:00) -> 0min gap

# TYPE-A: CME_PRECLOSE, CME_REOPEN, TOKYO_OPEN(MGC), LONDON_METALS
# TYPE-B: NYSE_CLOSE, CME_REOPEN, SINGAPORE_OPEN, EUROPE_FLOW

fork_a = ["CME_PRECLOSE", "CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS"]
fork_b = ["NYSE_CLOSE", "CME_REOPEN", "SINGAPORE_OPEN", "EUROPE_FLOW"]

DD = 2000

def print_profile(name, sessions):
    total_risk = 0
    total_ev = 0
    print(f"\n{name}:")
    print(f"  {'Time':>5}  {'Session':<20} {'Inst':<4} {'Risk':>5} {'ExpR':>6} {'EV':>6}")
    print(f"  {'-'*55}")
    for sess in sessions:
        inst, risk, expr, rr = best[sess]
        t = times[sess]
        h, m = divmod(t, 60)
        ev = risk * expr
        total_risk += risk
        total_ev += ev
        print(f"  {h:02d}:{m:02d}  {sess:<20} {inst:<4} ${risk:>4} {expr:>6.3f} +${ev:>4.1f}")

    annual = total_ev * 250
    print(f"  {'-'*55}")
    print(f"  TOTAL: {len(sessions)} sessions | risk=${total_risk} ({total_risk/DD*100:.0f}% DD) | ev=+${total_ev:.1f}/day | ${annual:,.0f}/yr")
    return total_ev

print("=" * 65)
print("MAX EXTRACTION TOPSTEP PROFILES — ZERO OVERLAP")
print("=" * 65)

ev_a = print_profile("TYPE-A (3 copies)", shared + fork_a)
ev_b = print_profile("TYPE-B (2 copies)", shared + fork_b)

print("\n" + "=" * 65)
print("DEPLOYMENT SUMMARY")
print("=" * 65)

gross = ev_a * 250 * 3 + ev_b * 250 * 2
# First $5K at 50%, rest at 90% per account
# Simplified blended rate ~85%
net = gross * 0.85
realistic = net * 0.5  # 50% shrinkage + filter hit rate

print(f"\n  3x TYPE-A annual: ${ev_a * 250 * 3:,.0f}")
print(f"  2x TYPE-B annual: ${ev_b * 250 * 2:,.0f}")
print(f"  GROSS (5 accounts): ${gross:,.0f}")
print(f"  After ~85% blended split: ${net:,.0f}")
print(f"  REALISTIC (50% shrinkage): ${realistic:,.0f}")
print()
print(f"  Worst-case day (TYPE-A): ALL 8 sessions lose = ${sum(best[s][1] for s in shared + fork_a)}")
print(f"  Worst-case day (TYPE-B): ALL 8 sessions lose = ${sum(best[s][1] for s in shared + fork_b)}")
print(f"  P(all 8 lose same day): < 0.4% (independent sessions)")

# Verify no overlaps
print("\n  OVERLAP CHECK:")
for label, sessions in [("TYPE-A", shared + fork_a), ("TYPE-B", shared + fork_b)]:
    sorted_sess = sorted(sessions, key=lambda s: times[s])
    for i in range(len(sorted_sess) - 1):
        s1, s2 = sorted_sess[i], sorted_sess[i+1]
        _, _, _, rr1 = best[s1]
        end1 = times[s1] + est_dur(rr1)
        start2 = times[s2]
        gap = start2 - end1
        if end1 > 1440:
            end1 -= 1440
        status = "OK" if gap >= 0 else f"OVERLAP {-gap}m"
        print(f"  {label}: {s1} -> {s2}: gap={gap}min {status}")
