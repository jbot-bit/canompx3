"""Max extraction model — real numbers from validated_setups.

Calculates realistic annual profit across all firms, account sizes,
sessions, instruments, and contract scaling tiers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import date

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH


def get_best_strategies():
    """Get best strategy per session × instrument from DB."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    rows = con.execute("""
        WITH ranked AS (
            SELECT strategy_id, instrument, orb_label, filter_type,
                   rr_target, stop_multiplier, sample_size,
                   win_rate, expectancy_r, sharpe_ann,
                   COALESCE(median_risk_dollars, 0) as med_risk,
                   json_extract_string(yearly_results, '$.2025') as yr2025,
                   ROW_NUMBER() OVER (
                       PARTITION BY orb_label, instrument
                       ORDER BY expectancy_r DESC
                   ) as rn
            FROM validated_setups
            WHERE status = 'active' AND sample_size >= 100
        )
        SELECT * FROM ranked WHERE rn = 1
        ORDER BY expectancy_r DESC
    """).fetchall()
    con.close()

    strategies = []
    for r in rows:
        sid, inst, orb, filt, rr, sm, n, wr, expr, sharpe, med_risk, yr2025, _ = r
        sm = sm or 1.0
        prop_risk = med_risk * min(sm, 0.75)

        # Parse 2025 forward
        fwd_r, fwd_n = 0, 0
        if yr2025:
            import json

            try:
                yr = json.loads(yr2025)
                fwd_r = yr.get("pnl_r", 0) or 0
                fwd_n = yr.get("n", 0) or 0
            except (json.JSONDecodeError, TypeError):
                pass

        strategies.append(
            {
                "strategy_id": sid,
                "instrument": inst,
                "orb_label": orb,
                "filter_type": filt,
                "rr_target": rr,
                "sample_size": n,
                "win_rate": wr,
                "expr": expr,
                "sharpe": sharpe,
                "med_risk": med_risk,
                "prop_risk": prop_risk,
                "fwd_r": fwd_r,
                "fwd_n": fwd_n,
            }
        )
    return strategies


def get_session_times():
    """Get session start times in Brisbane minutes-from-midnight."""
    today = date(2026, 3, 31)
    times = {}
    for name, cfg in SESSION_CATALOG.items():
        h, m = cfg["resolver"](today)
        times[name] = h * 60 + m
    return times


def est_duration(rr):
    """Estimated trade duration in minutes by RR target."""
    if rr <= 1.0:
        return 45
    if rr <= 1.5:
        return 60
    return 90


def check_overlap(sessions, times, strategies_by_key):
    """Check if session list has any time overlaps. Returns list of conflicts."""
    entries = []
    for sess, inst in sessions:
        key = (sess, inst)
        strat = strategies_by_key.get(key)
        if not strat:
            continue
        start = times.get(sess, 0)
        dur = est_duration(strat["rr_target"])
        entries.append((start, start + dur, sess, inst))

    entries.sort()
    conflicts = []
    for i in range(len(entries) - 1):
        gap = entries[i + 1][0] - entries[i][1]
        if gap < 0:
            conflicts.append((entries[i][2], entries[i + 1][2], -gap))
    return conflicts


def build_profiles(strategies, times):
    """Build TYPE-A and TYPE-B non-overlapping profiles with multi-instrument."""
    by_key = {(s["orb_label"], s["instrument"]): s for s in strategies}

    # Sessions available per instrument
    available = {}
    for s in strategies:
        key = (s["orb_label"], s["instrument"])
        available[key] = s

    # Shared overnight (no conflicts between these — verified)
    shared = [
        ("US_DATA_830", "MNQ"),
        ("NYSE_OPEN", "MNQ"),
        ("US_DATA_1000", "MNQ"),
        ("COMEX_SETTLE", "MNQ"),
    ]

    # TYPE-A daytime fork
    fork_a = [
        ("CME_PRECLOSE", "MNQ"),
        ("CME_REOPEN", "MNQ"),
        ("TOKYO_OPEN", "MGC"),
        ("LONDON_METALS", "MNQ"),
    ]

    # TYPE-B daytime fork
    fork_b = [
        ("NYSE_CLOSE", "MNQ"),
        ("CME_REOPEN", "MNQ"),
        ("SINGAPORE_OPEN", "MNQ"),
        ("EUROPE_FLOW", "MNQ"),
    ]

    # Add MES stacking where available (non-overlapping with MNQ in same session)
    # MES fires in the SAME session window as MNQ — no extra time needed
    # Just an additional contract in the same window
    mes_stack = []
    for s in strategies:
        if s["instrument"] == "MES" and s["sample_size"] >= 100:
            mes_stack.append((s["orb_label"], "MES"))

    type_a_sessions = shared + fork_a
    type_b_sessions = shared + fork_b

    # Add MES where it doesn't create NEW time conflicts
    # MES in same session = same time window, just 2 instruments
    type_a_mes = [(orb, "MES") for orb, _ in type_a_sessions if (orb, "MES") in available]
    type_b_mes = [(orb, "MES") for orb, _ in type_b_sessions if (orb, "MES") in available]

    return {
        "TYPE-A": {"mnq_mgc": type_a_sessions, "mes_stack": type_a_mes},
        "TYPE-B": {"mnq_mgc": type_b_sessions, "mes_stack": type_b_mes},
    }, by_key


def main():
    strategies = get_best_strategies()
    times = get_session_times()
    profiles, by_key = build_profiles(strategies, times)

    # TopStep scaling tiers
    ts_tiers = [
        ("50K", 2000, 50, 50, 2000),  # name, dd, max_micro, reset, unlock_at
        ("100K", 3000, 100, 100, 3000),
        ("150K", 4500, 150, 150, 4500),
    ]

    print("=" * 90)
    print("MAX EXTRACTION MODEL — REAL NUMBERS FROM VALIDATED_SETUPS")
    print("=" * 90)

    for ptype, pdata in profiles.items():
        print(f"\n{'=' * 90}")
        print(f"PROFILE {ptype}")
        print(f"{'=' * 90}")

        all_lanes = pdata["mnq_mgc"] + pdata["mes_stack"]
        total_risk_1ct = 0
        total_ev_1ct = 0
        total_fwd = 0

        print(
            f"\n  {'Time':>5}  {'Session':<20} {'Inst':<4} {'Risk':>5} {'ExpR':>7} {'EV/day':>7} {'Fwd2025':>8} {'N':>5}"
        )
        print(f"  {'-' * 70}")

        # Sort by time
        sorted_lanes = sorted(all_lanes, key=lambda x: times.get(x[0], 0))
        for orb, inst in sorted_lanes:
            s = by_key.get((orb, inst))
            if not s:
                continue
            t = times.get(orb, 0)
            h, m = divmod(t, 60)
            risk = s["prop_risk"]
            ev = risk * s["expr"]
            total_risk_1ct += risk
            total_ev_1ct += ev
            total_fwd += s["fwd_r"]
            print(
                f"  {h:02d}:{m:02d}  {orb:<20} {inst:<4} ${risk:>4.0f} {s['expr']:>7.4f} +${ev:>5.1f} {s['fwd_r']:>+7.1f}R {s['sample_size']:>5}"
            )

        print(f"  {'-' * 70}")
        print(
            f"  TOTAL: {len([x for x in sorted_lanes if (x[0], x[1]) in by_key])} lanes | risk=${total_risk_1ct:.0f} | ev=+${total_ev_1ct:.1f}/day | fwd={total_fwd:+.1f}R"
        )

        # Overlap check
        conflicts = check_overlap(pdata["mnq_mgc"], times, by_key)
        if conflicts:
            print("\n  OVERLAPS (MNQ/MGC):")
            for s1, s2, mins in conflicts:
                print(f"    {s1} <-> {s2}: {mins}min overlap")
        else:
            print("\n  OVERLAP CHECK: ALL CLEAR")

    # Now compute across firms and account sizes
    print(f"\n\n{'=' * 90}")
    print("FIRM × ACCOUNT SIZE × CONTRACT SCALING")
    print(f"{'=' * 90}")

    # Compute per-profile daily EV
    profile_evs = {}
    profile_risks = {}
    for ptype, pdata in profiles.items():
        all_lanes = pdata["mnq_mgc"] + pdata["mes_stack"]
        ev = sum(by_key[(o, i)]["prop_risk"] * by_key[(o, i)]["expr"] for o, i in all_lanes if (o, i) in by_key)
        risk = sum(by_key[(o, i)]["prop_risk"] for o, i in all_lanes if (o, i) in by_key)
        profile_evs[ptype] = ev
        profile_risks[ptype] = risk

    avg_ev = (profile_evs["TYPE-A"] + profile_evs["TYPE-B"]) / 2
    avg_risk = (profile_risks["TYPE-A"] + profile_risks["TYPE-B"]) / 2

    firms = [
        ("TopStep", "full", 5, 0.85),  # firm, auto, max_accounts, blended_split
        ("Tradeify", "full", 5, 0.90),
    ]

    print(f"\n  {'Firm':<12} {'Acct':>6} {'Cts':>4} {'DD%':>5} {'Gross/yr':>12} {'Net':>12} {'Real(50%)':>12}")
    print(f"  {'-' * 75}")

    grand_totals = {}  # (firm, tier_name, cts) -> realistic annual

    for firm_name, _auto, max_accts, split in firms:
        for tier_name, dd, max_micro, _reset, _unlock in ts_tiers:
            for cts in [2, 3, 5, 7, 10]:
                if cts > max_micro:
                    continue
                worst = avg_risk * cts
                pct_dd = worst / dd * 100
                if pct_dd > 150:
                    continue  # skip absurd

                daily = avg_ev * cts
                gross = daily * 250 * max_accts
                net = gross * split
                real = net * 0.50

                flag = ""
                if pct_dd > 80:
                    flag = " !"
                elif pct_dd > 50:
                    flag = " *"

                key = (firm_name, tier_name, cts)
                grand_totals[key] = real

                print(
                    f"  {firm_name:<12} {tier_name:>6} {cts:>4} {pct_dd:>4.0f}% ${gross:>11,.0f} ${net:>11,.0f} ${real:>11,.0f}{flag}"
                )

    # Best combos
    print(f"\n\n{'=' * 90}")
    print("TOP SCENARIOS (sorted by realistic annual)")
    print(f"{'=' * 90}")
    print(f"\n  {'Firm':<12} {'Acct':>6} {'Cts':>4} {'Realistic/yr':>14} {'Notes'}")
    print(f"  {'-' * 60}")

    for key in sorted(grand_totals, key=grand_totals.get, reverse=True)[:15]:
        firm, tier, cts = key
        real = grand_totals[key]
        dd = {"50K": 2000, "100K": 3000, "150K": 4500}[tier]
        worst = avg_risk * cts
        pct = worst / dd * 100
        note = "SAFE" if pct < 50 else ("AGGRO" if pct < 80 else "YOLO")
        print(f"  {firm:<12} {tier:>6} {cts:>4}   ${real:>12,.0f}   {note} ({pct:.0f}% DD worst)")

    # Combined best
    print(f"\n\n{'=' * 90}")
    print("COMBINED FIRM SCENARIOS")
    print(f"{'=' * 90}")

    combos = [
        ("Conservative", [("TopStep", "50K", 3), ("Tradeify", "50K", 3)]),
        ("Moderate", [("TopStep", "100K", 5), ("Tradeify", "100K", 5)]),
        ("Aggressive", [("TopStep", "150K", 7), ("Tradeify", "150K", 7)]),
        ("Max Send", [("TopStep", "150K", 10), ("Tradeify", "150K", 10)]),
    ]

    for label, combo in combos:
        total = sum(grand_totals.get(c, 0) for c in combo)
        details = " + ".join(f"{f} {t} {c}ct" for f, t, c in combo)
        print(f"\n  {label}: ${total:>12,.0f}/yr realistic")
        print(f"    {details}")

    # Add Apex manual on top
    print("\n\n  + APEX MANUAL (daytime sessions, 100% split, 1-2 cts):")
    # Apex gets the daytime sessions you trade by hand
    apex_sessions = ["NYSE_CLOSE", "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS", "EUROPE_FLOW"]
    apex_ev = 0
    for sess in apex_sessions:
        for inst in ACTIVE_ORB_INSTRUMENTS:
            s = by_key.get((sess, inst))
            if s:
                apex_ev += s["prop_risk"] * s["expr"]

    apex_annual = apex_ev * 250 * 1.0 * 0.50  # 100% split, 50% shrinkage, 1 ct
    print(f"    {len(apex_sessions)} sessions, ~${apex_ev:.0f}/day EV, ${apex_annual:,.0f}/yr realistic at 1 ct")


if __name__ == "__main__":
    main()
