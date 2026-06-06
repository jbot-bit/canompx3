"""READ-ONLY C11 clearance-scenario diagnostic for topstep_50k_mnq_auto.

Reuses canonical account_survival loaders (no re-encoding of PnL/cost/DD logic).
Produces:
  - per-lane trade-path history (canonical)
  - combined-book breach-day attribution
  - per-lane standalone C11 verdicts
  - lane-subset scenarios (drop-one, single-lane, pairs)
  - pairwise same-day overlap + loss correlation

NO config mutation. NO live write. Read-only DB. Diagnostic output only.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.account_survival import (
    MIN_SURVIVAL_PROBABILITY,
    STRICT_DD_HORIZON_DAYS,
    _build_rules,
    _historical_daily_loss_breach_days,
    _load_lane_trade_paths,
    _max_observed_rolling_drawdown,
    _scenario_from_trade_paths,
    _with_consistency_rule,
    effective_strict_dd_budget,
    simulate_survival,
)
from trading_app.prop_profiles import (
    get_profile,
    get_profile_lane_definitions,
    load_allocation_lanes,
)

PROFILE_ID = "topstep_50k_mnq_auto"
AS_OF = date(2026, 6, 3)


def _lane_trades(con, lane_defs, eff_stop):
    """Return {strategy_id: list[TradePath]} using canonical loader."""
    out = {}
    for lane in lane_defs:
        sid = lane["strategy_id"]
        out[sid] = _load_lane_trade_paths(con, sid, as_of_date=AS_OF, effective_stop_multiplier=eff_stop.get(sid))
    return out


def _scenarios_for_subset(con, instruments, trades_by_lane, subset_ids):
    """Build combined daily scenarios for a subset of lanes over common support."""
    trades_by_day: dict[date, list] = defaultdict(list)
    lane_first: dict[str, date] = {}
    for sid in subset_ids:
        daily_days = set()
        for tr in trades_by_lane[sid]:
            trades_by_day[tr.trading_day].append(tr)
            daily_days.add(tr.trading_day)
        if not daily_days:
            continue
        lane_first[sid] = min(daily_days)
    if not lane_first:
        return []
    common_start = max(lane_first.values())
    placeholders = ", ".join("?" for _ in instruments)
    cal = [
        r[0]
        for r in con.execute(
            f"""
            SELECT DISTINCT trading_day FROM daily_features
            WHERE symbol IN ({placeholders}) AND trading_day >= ? AND trading_day <= ?
            ORDER BY trading_day
            """,
            [*instruments, common_start, AS_OF],
        ).fetchall()
    ]
    return [
        _scenario_from_trade_paths(d, [t for t in trades_by_day.get(d, []) if t.strategy_id in subset_ids]) for d in cal
    ]


def _gate(scenarios, rules, profile, *, n_paths=10000, seed=0):
    """Return strict + operational gate verdict for a scenario set."""
    if not scenarios:
        return None
    res = simulate_survival(scenarios, rules, horizon_days=90, n_paths=n_paths, seed=seed)
    op = round(res["operational_pass_probability"], 4)
    budget = effective_strict_dd_budget(profile, rules)
    breach_days = _historical_daily_loss_breach_days(scenarios, rules)
    max_dd = _max_observed_rolling_drawdown(scenarios, horizon_days=STRICT_DD_HORIZON_DAYS)
    op_pass = op >= float(MIN_SURVIVAL_PROBABILITY)
    strict_pass = len(breach_days) == 0 and max_dd <= budget
    return {
        "operational_pass": op,
        "op_gate_pass": op_pass,
        "budget": budget,
        "breach_days": breach_days,
        "breach_count": len(breach_days),
        "max_90d_dd": max_dd,
        "strict_pass": strict_pass,
        "gate_pass": op_pass and strict_pass,
        "daily_loss_breach_prob": round(res["daily_loss_breach_probability"], 4),
        "n_days": len(scenarios),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    profile = get_profile(PROFILE_ID)
    lane_defs = get_profile_lane_definitions(PROFILE_ID)
    lane_specs = profile.daily_lanes or load_allocation_lanes(PROFILE_ID)
    eff_stop = {L.strategy_id: float(L.planned_stop_multiplier or profile.stop_multiplier) for L in lane_specs}
    rules = _with_consistency_rule(_build_rules(profile), profile)
    instruments = sorted({lane["instrument"] for lane in lane_defs})
    all_ids = [lane["strategy_id"] for lane in lane_defs]

    print(
        f"PROFILE {PROFILE_ID} | daily_loss_limit=${rules.daily_loss_limit} | "
        f"dd_limit=${rules.dd_limit_dollars} | strict_budget=${effective_strict_dd_budget(profile, rules):.0f} | "
        f"min_surv={MIN_SURVIVAL_PROBABILITY}"
    )
    print(f"LANES ({len(all_ids)}): {all_ids}")
    print(f"INSTRUMENTS: {instruments}")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        trades_by_lane = _lane_trades(con, lane_defs, eff_stop)

        # --- per-lane daily pnl, for attribution + correlation ---
        lane_daily: dict[str, dict[date, float]] = {}
        for sid, trs in trades_by_lane.items():
            dd: dict[date, float] = defaultdict(float)
            for t in trs:
                dd[t.trading_day] += t.pnl_dollars
            lane_daily[sid] = dict(dd)
            print(
                f"  {sid}: trades={len(trs)} trade_days={len(dd)} "
                f"total_pnl=${sum(dd.values()):,.0f} first={min(dd) if dd else None} last={max(dd) if dd else None}"
            )

        # === FULL BOOK ===
        print("\n=== FULL BOOK (all lanes) ===")
        full = _scenarios_for_subset(con, instruments, trades_by_lane, set(all_ids))
        gfull = _gate(full, rules, profile)
        print(gfull)

        # === BREACH-DAY ATTRIBUTION ===
        print("\n=== BREACH-DAY ATTRIBUTION (full book) ===")
        # map trading_day -> combined scenario for breach days
        full_by_day = {s.trading_day: s for s in full}
        for bd in gfull["breach_days"]:
            d = date.fromisoformat(bd)
            sc = full_by_day.get(bd)
            per_lane = {sid: lane_daily[sid].get(d, 0.0) for sid in all_ids}
            traded = {k: round(v, 0) for k, v in per_lane.items() if abs(v) > 1e-6}
            worst = min(per_lane.items(), key=lambda kv: kv[1])
            print(
                f"  {bd}: total_close=${sc.total_pnl_dollars:,.0f} intraday_min=${sc.min_balance_delta_dollars:,.0f} "
                f"limit=-${rules.daily_loss_limit} | lanes={len(traded)} worst={worst[0].split('_')[1]}:${worst[1]:,.0f} | {traded}"
            )

        # === STANDALONE PER-LANE ===
        print("\n=== STANDALONE C11 PER LANE ===")
        standalone = {}
        for sid in all_ids:
            g = _gate(_scenarios_for_subset(con, instruments, trades_by_lane, {sid}), rules, profile)
            standalone[sid] = g
            print(
                f"  {sid}: gate_pass={g['gate_pass']} strict={g['strict_pass']} op={g['operational_pass']} "
                f"breach_days={g['breach_count']} max90dd=${g['max_90d_dd']:,.0f}/${g['budget']:,.0f}"
            )

        # === DROP-ONE ===
        print("\n=== DROP-ONE-LANE ===")
        for drop in all_ids:
            keep = set(all_ids) - {drop}
            g = _gate(_scenarios_for_subset(con, instruments, trades_by_lane, keep), rules, profile)
            print(
                f"  drop {drop.split('_')[1]:13}: gate_pass={g['gate_pass']} strict={g['strict_pass']} "
                f"op={g['operational_pass']} breach={g['breach_count']} max90dd=${g['max_90d_dd']:,.0f}/${g['budget']:,.0f}"
            )

        # === PAIRS ===
        print("\n=== LANE PAIRS ===")
        for pair in combinations(all_ids, 2):
            g = _gate(_scenarios_for_subset(con, instruments, trades_by_lane, set(pair)), rules, profile)
            tags = "+".join(p.split("_")[1] for p in pair)
            print(
                f"  {tags:28}: gate_pass={g['gate_pass']} strict={g['strict_pass']} "
                f"op={g['operational_pass']} breach={g['breach_count']} max90dd=${g['max_90d_dd']:,.0f}/${g['budget']:,.0f}"
            )

        # === PAIRWISE SAME-DAY OVERLAP + LOSS CORRELATION ===
        print("\n=== PAIRWISE SAME-DAY OVERLAP / LOSS-DAY CORRELATION ===")
        for a, b in combinations(all_ids, 2):
            da, db = lane_daily[a], lane_daily[b]
            common = set(da) & set(db)
            both_loss = sum(1 for d in common if da[d] < 0 and db[d] < 0)
            print(f"  {a.split('_')[1]}+{b.split('_')[1]}: common_days={len(common)} both_loss_days={both_loss}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
