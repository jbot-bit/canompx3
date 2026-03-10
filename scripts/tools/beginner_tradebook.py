"""
Beginner Tradebook — Personal daily trading reference.

Shows flagged strategies from the validated book:
  TRADE   = passes all quality gates (ROBUST/WHITELISTED, PBO<0.5, Fitschen 3x at chosen contracts)
  STRETCH = below Fitschen 3x at 1c but viable at 3+ contracts (morning block)
  SKIP    = singleton, high PBO, or below cost threshold

Usage:
    python scripts/tools/beginner_tradebook.py
    python scripts/tools/beginner_tradebook.py --contracts 5
    python scripts/tools/beginner_tradebook.py --contracts 1
"""

import argparse
import sys
from datetime import date

sys.path.insert(0, ".")
import duckdb

from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH

# ============================================================
# Beginner-approved sessions (best family per session, hand-picked)
# Criteria: ROBUST/WHITELISTED, PBO<0.5, best Exp$ per session
# Update this list after each rebuild chain.
# ============================================================

# TIER 1: TRADE — clears Fitschen 3x at 1 contract (or near-threshold with high quality)
TIER1_SESSIONS = [
    # US session — late night Brisbane (22:30-06:00)
    {"session": "NYSE_OPEN", "instrument": "MNQ", "bris_time": "22:30", "label": "TRADE"},
    {"session": "US_DATA_1000", "instrument": "MNQ", "bris_time": "00:00", "label": "TRADE"},
    {"session": "CME_PRECLOSE", "instrument": "MNQ", "bris_time": "05:45", "label": "TRADE"},
    {"session": "COMEX_SETTLE", "instrument": "MNQ", "bris_time": "03:30", "label": "TRADE"},
]

# TIER 2: STRETCH — clears Fitschen 3x at 3+ contracts, morning block
TIER2_SESSIONS = [
    {"session": "SINGAPORE_OPEN", "instrument": "MNQ", "bris_time": "11:00", "label": "STRETCH"},
    {"session": "LONDON_METALS", "instrument": "MNQ", "bris_time": "17:00", "label": "STRETCH"},
]

ALL_FLAGGED = TIER1_SESSIONS + TIER2_SESSIONS

# Quality gates
ROBUST_OK = {"ROBUST", "WHITELISTED"}
PBO_MAX = 0.5
MIN_FITSCHEN_MULT = 3.0  # Fitschen floor

TIER_EMOJI = {
    "TRADE": "[TRADE]  ",
    "STRETCH": "[STRETCH]",
    "SKIP": "[SKIP]   ",
}


def get_session_best(con, instrument: str, session: str, contracts: int) -> dict | None:
    """Get best beginner-eligible strategy for this session/instrument."""
    rows = con.execute(
        """
        SELECT
            vs.strategy_id,
            vs.orb_label,
            vs.orb_minutes,
            vs.entry_model,
            vs.rr_target,
            vs.confirm_bars,
            vs.filter_type,
            vs.sample_size,
            vs.win_rate,
            vs.expectancy_r,
            vs.sharpe_ann,
            ROUND(es.median_risk_dollars, 2) AS risk_1r,
            ROUND(vs.expectancy_r * es.median_risk_dollars, 2) AS exp_1c,
            ef.robustness_status,
            ef.member_count,
            COALESCE(ef.pbo, -1) AS pbo
        FROM validated_setups vs
        LEFT JOIN experimental_strategies es ON vs.strategy_id = es.strategy_id
        LEFT JOIN edge_families ef ON vs.family_hash = ef.family_hash
        WHERE vs.is_family_head = TRUE
          AND LOWER(vs.status) = 'active'
          AND vs.instrument = ?
          AND vs.orb_label = ?
          AND ef.robustness_status IN ('ROBUST', 'WHITELISTED')
          AND (ef.pbo IS NULL OR ef.pbo < 0.5)
          AND es.median_risk_dollars IS NOT NULL
        ORDER BY (vs.expectancy_r * es.median_risk_dollars) DESC
        LIMIT 1
    """,
        [instrument, session],
    ).fetchone()

    if not rows:
        return None

    (sid, orb_label, orb_min, em, rr, cb, ftype, n, wr, exr, sharpe, r1, exp1c, rob, members, pbo) = rows
    spec = COST_SPECS.get(instrument)
    rt = spec.total_friction
    exp_nc = exp1c * contracts

    fitschen_1c = exp1c / rt if rt > 0 else 0
    fitschen_nc = exp_nc / rt if rt > 0 else 0
    passes = fitschen_nc >= MIN_FITSCHEN_MULT

    return {
        "strategy_id": sid,
        "session": orb_label,
        "orb_minutes": orb_min,
        "entry_model": em,
        "rr_target": rr,
        "confirm_bars": cb,
        "filter_type": ftype,
        "sample_size": n,
        "win_rate": wr,
        "expectancy_r": exr,
        "sharpe": sharpe,
        "risk_1r": r1,
        "exp_1c": exp1c,
        "exp_nc": exp_nc,
        "robustness": rob,
        "members": members,
        "pbo": pbo,
        "fitschen_1c": fitschen_1c,
        "fitschen_nc": fitschen_nc,
        "passes_fitschen": passes,
        "rt_cost": rt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=int, default=1, help="Number of contracts per trade (default 1)")
    args = parser.parse_args()

    contracts = args.contracts
    today = date.today()

    print(f"BEGINNER TRADEBOOK  — {today.strftime('%A %d %b %Y')}  |  Contracts: {contracts}")
    print("=" * 100)
    print(f"Gates: ROBUST/WHITELISTED | PBO < {PBO_MAX} | Fitschen >= {MIN_FITSCHEN_MULT}x RT cost at {contracts}c")
    print()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("US SESSION BLOCK  (late night Brisbane — use alerts or trade overnight)")
    print("-" * 100)
    print(
        f"  {'FLAG':<11} {'Time BRIS':<10} {'Session':<22} {'Inst':<5} {'ORB':<4} {'RR':<4} {'CB':<3} {'Filter':<20} {'N':<5} {'WR':<6} {'Exp$' + str(contracts) + 'c':<10} {'Fitschen':<10} {'Family':<14} {'PBO'}"
    )
    print(f"  {'-' * 110}")

    for entry in TIER1_SESSIONS:
        s = get_session_best(con, entry["instrument"], entry["session"], contracts)
        if s is None:
            print(
                f"  {TIER_EMOJI['SKIP']}  {entry['bris_time']:<10} {entry['session']:<22} {entry['instrument']:<5} --- no eligible strategy ---"
            )
            continue
        flag = "TRADE" if s["passes_fitschen"] else "SKIP"
        fam = f"{s['robustness']}x{s['members']}"
        pbo_str = f"{s['pbo']:.2f}" if s["pbo"] >= 0 else "-"
        fitschen_str = f"{s['fitschen_nc']:.1f}x"
        print(
            f"  {TIER_EMOJI[flag]}  {entry['bris_time']:<10} {entry['session']:<22} {entry['instrument']:<5} {s['orb_minutes']:<4} {str(s['rr_target']):<4} {str(s['confirm_bars']):<3} {s['filter_type']:<20} {s['sample_size']:<5} {s['win_rate']:<6.1%} ${s['exp_nc']:<9.2f} {fitschen_str:<10} {fam:<14} {pbo_str}"
        )

    print()
    print("MORNING BLOCK  (Brisbane daytime — live trading hours)")
    print("-" * 100)
    print(
        f"  {'FLAG':<11} {'Time BRIS':<10} {'Session':<22} {'Inst':<5} {'ORB':<4} {'RR':<4} {'CB':<3} {'Filter':<20} {'N':<5} {'WR':<6} {'Exp$' + str(contracts) + 'c':<10} {'Fitschen':<10} {'Family':<14} {'PBO'}"
    )
    print(f"  {'-' * 110}")

    for entry in TIER2_SESSIONS:
        s = get_session_best(con, entry["instrument"], entry["session"], contracts)
        if s is None:
            print(
                f"  {TIER_EMOJI['SKIP']}  {entry['bris_time']:<10} {entry['session']:<22} {entry['instrument']:<5} --- no eligible strategy ---"
            )
            continue
        flag = "TRADE" if s["passes_fitschen"] else "STRETCH"
        fam = f"{s['robustness']}x{s['members']}"
        pbo_str = f"{s['pbo']:.2f}" if s["pbo"] >= 0 else "-"
        fitschen_str = f"{s['fitschen_nc']:.1f}x"
        print(
            f"  {TIER_EMOJI[flag]}  {entry['bris_time']:<10} {entry['session']:<22} {entry['instrument']:<5} {s['orb_minutes']:<4} {str(s['rr_target']):<4} {str(s['confirm_bars']):<3} {s['filter_type']:<20} {s['sample_size']:<5} {s['win_rate']:<6.1%} ${s['exp_nc']:<9.2f} {fitschen_str:<10} {fam:<14} {pbo_str}"
        )

    con.close()

    print()
    print("LEGEND")
    print("  [TRADE]   = take every valid signal. Passes all quality gates at your contract size.")
    print("  [STRETCH] = viable at this contract size only if Fitschen shows >=3x above.")
    print("              If not yet at required size, skip or use 1c to learn the session.")
    print("  [SKIP]    = no eligible strategy for this session. Do not force trades.")
    print()
    print(f"  Exp$ shown = expectancy at {contracts} contract(s) per trade (before prop split)")
    print(f"  Fitschen = Exp$ / round-trip cost. Floor = {MIN_FITSCHEN_MULT}x.")
    print()

    # Daily totals
    print("DAILY EXPECTANCY ESTIMATE")
    print("  Assumes ~65% of days have a valid ORB signal (historical filter activation rate)")
    trade_sessions = TIER1_SESSIONS  # conservative: US only
    total_exp = 0
    count = 0
    _con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    for entry in trade_sessions:
        s = get_session_best(_con, entry["instrument"], entry["session"], contracts)
        if s and s["passes_fitschen"]:
            total_exp += s["exp_nc"]
            count += 1
    _con.close()

    if contracts >= 3:
        # Add morning block
        _con2 = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        for entry in TIER2_SESSIONS:
            s = get_session_best(_con2, entry["instrument"], entry["session"], contracts)
            if s and s["passes_fitschen"]:
                total_exp += s["exp_nc"]
                count += 1
        _con2.close()

    active_rate = 0.65
    daily_exp = total_exp * active_rate
    annual_exp = daily_exp * 200

    print(
        f"  Sessions in plan: {count} | Gross Exp$ per active day: ${total_exp:.2f} × {active_rate:.0%} = ${daily_exp:.2f}/day"
    )
    print(f"  Annual gross (200 trading days): ${annual_exp:,.0f}")
    print(f"  After 80% prop split: ${annual_exp * 0.8:,.0f}/year")
    print()
    print("  NOTE: These are EXPECTED VALUES — actual results vary. Drawdown periods are real.")
    print("  Rule: Never risk more than 2% of account per trade. Stop day at -6R.")


if __name__ == "__main__":
    main()
