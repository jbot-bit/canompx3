"""Optimal lane selector — multi-instrument, multi-firm portfolio optimizer.

Scans ALL validated strategies across MNQ/MES/MGC, scores them, and picks
the best lane per session subject to constraints:
  - DD budget: total portfolio risk <= budget_pct of firm DD
  - Stop multiplier optimization: picks S0.75 vs S1.0 based on DD headroom
  - Filter diversity: max N lanes with same filter family
  - Auto-eligibility: ROBUST/WHITELISTED + FDR + forward > 0
  - 20% switching threshold (Carver Ch 12) vs current deployed

Outputs ranked selection + DailyLaneSpec Python code for prop_profiles.py.

Usage:
    python -m scripts.tools.build_optimal_profiles                        # All firms
    python -m scripts.tools.build_optimal_profiles --firm apex --size 100000
    python -m scripts.tools.build_optimal_profiles --budget-pct 40        # Tighter DD budget
    python -m scripts.tools.build_optimal_profiles --max-slots 6          # More lanes
    python -m scripts.tools.build_optimal_profiles --show-switches        # Flag changes vs current
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.prop_profiles import (
    ACCOUNT_PROFILES,
    ACCOUNT_TIERS,
    PROP_FIRM_SPECS,
)

# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_BUDGET_PCT = 50.0  # Max % of DD consumed by all lanes combined
DEFAULT_MAX_SLOTS = 6  # Max lanes per profile
DEFAULT_MAX_SAME_FILTER = 2  # Max lanes sharing a filter family
DEFAULT_MIN_N = 100  # Min sample size
CARVER_SWITCH_THRESHOLD = 0.20  # 20% score improvement to justify switch


# ── Filter family grouping ──────────────────────────────────────────────────


def _filter_family(filter_type: str) -> str:
    """Group filters into families for diversity check."""
    if filter_type.startswith("ORB_G"):
        return "ORB_SIZE"
    if filter_type.startswith("ORB_VOL"):
        return "ORB_VOLUME"
    if filter_type.startswith("VOL_RV"):
        return "REL_VOL"
    if filter_type.startswith("ATR"):
        return "ATR"
    if filter_type.startswith("COST_LT"):
        return "COST"
    if filter_type.startswith("OVNRNG"):
        return "OVERNIGHT"
    if filter_type.startswith("X_M"):
        return "CROSS_ASSET"
    return filter_type


# ── Brisbane time helpers ───────────────────────────────────────────────────


def _resolve_brisbane_hours(trading_day: date | None = None) -> dict[str, tuple[int, int]]:
    td = trading_day or date.today()
    result: dict[str, tuple[int, int]] = {}
    for name, info in SESSION_CATALOG.items():
        resolver = info.get("resolver")
        if resolver:
            try:
                h, m = resolver(td)
                result[name] = (h, m)
            except Exception:
                pass
    return result


def _brisbane_time_str(session: str, hours_map: dict[str, tuple[int, int]]) -> str:
    hm = hours_map.get(session)
    return f"{hm[0]:02d}:{hm[1]:02d}" if hm else "??:??"


def _slot_label(session: str, hours_map: dict[str, tuple[int, int]]) -> str:
    hm = hours_map.get(session)
    if not hm:
        return "?"
    h = hm[0]
    if h >= 22 or h < 6:
        return "auto"
    if 6 <= h < 10:
        return "either"
    return "manual"


# ── Scoring (same factors as score_lanes.py) ───────────────────────────────


def _sharpe_adj(sharpe_ann: float | None) -> float:
    if not sharpe_ann or sharpe_ann <= 0:
        return 0.1
    return min(sharpe_ann / 1.5, 2.0)


def _ayp_factor(ayp: bool | None) -> float:
    return 1.2 if ayp else 1.0


def _n_confidence(n: int | None) -> float:
    if not n or n <= 0:
        return 0.1
    return min(1.0, n / 300)


def _fitness_factor(robustness: str | None) -> float:
    if robustness == "ROBUST":
        return 1.0
    if robustness == "WHITELISTED":
        return 0.9
    return 0.7


def _rr_adj(rr: float | None) -> float:
    if not rr:
        return 1.0
    if rr <= 1.0:
        return 1.0
    if rr <= 2.0:
        return 0.95
    return 0.85


def _forward_2025(yearly_json: str | None) -> tuple[float, int]:
    if not yearly_json:
        return 0.0, 0
    try:
        data = json.loads(yearly_json) if isinstance(yearly_json, str) else yearly_json
        entry = data.get("2025")
        if entry:
            return entry.get("total_r", 0.0), entry.get("trades", 0)
    except (json.JSONDecodeError, AttributeError):
        pass
    return 0.0, 0


# ── Strategy data class ────────────────────────────────────────────────────


@dataclass
class ScoredStrategy:
    strategy_id: str
    instrument: str
    session: str
    filter_type: str
    filter_family: str
    rr_target: float
    stop_mult: float
    sample_size: int
    win_rate: float
    expectancy_r: float
    sharpe_ann: float
    ayp: bool
    wfe: float
    median_risk_dollars: float
    exec_risk: float  # risk per 1 contract
    fdr_p: float | None
    robustness: str | None
    members: int
    fwd_r: float
    fwd_n: int
    tpy: float
    mdd: float
    score: float
    auto_eligible: bool
    slot: str
    bris_time: str
    base_strategy: str  # strategy_id without _S075 suffix
    contracts: int = 1  # set during lane selection


# ── Load all candidates ────────────────────────────────────────────────────


def _load_all_candidates(
    firm: str,
    account_size: int,
    dd_limit: float,
) -> list[ScoredStrategy]:
    """Load and score ALL eligible strategies across all active instruments."""
    hours_map = _resolve_brisbane_hours()
    spec = PROP_FIRM_SPECS.get(firm)
    split = spec.profit_split_tiers[-1][1] if spec else 0.90

    candidates = []

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        rows = con.execute(
            """
            SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.filter_type,
                   vs.rr_target, vs.sample_size, vs.win_rate, vs.expectancy_r,
                   vs.sharpe_ann, vs.all_years_positive, vs.wfe,
                   vs.median_risk_dollars, vs.stop_multiplier, vs.yearly_results,
                   vs.fdr_adjusted_p, vs.max_drawdown_r, vs.trades_per_year,
                   ef.robustness_status, ef.member_count
            FROM validated_setups vs
            LEFT JOIN edge_families ef ON vs.family_hash = ef.family_hash
            WHERE vs.status = 'active'
              AND vs.entry_model = 'E2'
              AND vs.confirm_bars = 1
              AND vs.orb_minutes = 5
              AND vs.sample_size >= ?
            ORDER BY vs.expectancy_r DESC
            """,
            [DEFAULT_MIN_N],
        ).fetchall()

    for r in rows:
        (
            sid,
            inst,
            session,
            filt,
            rr,
            n,
            wr,
            expr,
            sharpe,
            ayp,
            wfe,
            med_risk,
            sm,
            yearly_json,
            fdr_p,
            mdd,
            tpy,
            robustness,
            members,
        ) = r

        if inst not in ACTIVE_ORB_INSTRUMENTS:
            continue

        sm = sm or 1.0
        exec_risk = (med_risk or 0) * sm
        risk_pct = (exec_risk / dd_limit * 100) if dd_limit > 0 else 0

        fwd_r, fwd_n = _forward_2025(yearly_json)

        score = (
            (expr or 0)
            * _sharpe_adj(sharpe)
            * _ayp_factor(ayp)
            * _n_confidence(n)
            * _fitness_factor(robustness)
            * _rr_adj(rr)
            * split
        )

        auto_ok = (
            robustness in ("ROBUST", "WHITELISTED")
            and fdr_p is not None
            and fdr_p <= 0.05
            and fwd_r > 0
            and risk_pct <= 5.0
        )

        base = sid.replace("_S075", "")

        candidates.append(
            ScoredStrategy(
                strategy_id=sid,
                instrument=inst,
                session=session,
                filter_type=filt,
                filter_family=_filter_family(filt),
                rr_target=rr,
                stop_mult=sm,
                sample_size=n,
                win_rate=wr,
                expectancy_r=expr or 0,
                sharpe_ann=sharpe or 0,
                ayp=ayp or False,
                wfe=wfe or 0,
                median_risk_dollars=med_risk or 0,
                exec_risk=exec_risk,
                fdr_p=fdr_p,
                robustness=robustness,
                members=members or 0,
                fwd_r=fwd_r,
                fwd_n=fwd_n,
                tpy=tpy or 0,
                mdd=mdd or 0,
                score=score,
                auto_eligible=auto_ok,
                slot=_slot_label(session, hours_map),
                bris_time=_brisbane_time_str(session, hours_map),
                base_strategy=base,
            )
        )

    return candidates


# ── Per-lane SM optimization ───────────────────────────────────────────────


def _build_sm_lookup(candidates: list[ScoredStrategy]) -> dict[str, list[ScoredStrategy]]:
    """Group strategies by base name (with/without _S075) for per-lane SM choice."""
    by_base: dict[str, list[ScoredStrategy]] = {}
    for c in candidates:
        by_base.setdefault(c.base_strategy, []).append(c)
    return by_base


def _pick_sm_for_lane(
    variants: list[ScoredStrategy],
    remaining_budget: float,
) -> ScoredStrategy:
    """Pick the best SM variant given remaining DD budget.

    With lots of headroom: prefer S1.0 (higher ExpR, higher WR).
    Tight budget: prefer S0.75 (lower risk per trade).
    Tie-break: higher absolute score wins.
    """
    if len(variants) == 1:
        return variants[0]

    s10 = [v for v in variants if v.stop_mult >= 1.0]
    s075 = [v for v in variants if v.stop_mult < 1.0]

    if s10 and s075:
        best_10 = max(s10, key=lambda v: v.score)
        best_075 = max(s075, key=lambda v: v.score)

        # Can we afford S1.0? Check if risk < 5% of remaining budget
        if best_10.exec_risk <= remaining_budget * 0.05:
            # Plenty of room — pick by raw score (S1.0 usually wins on ExpR)
            return best_10 if best_10.score >= best_075.score else best_075
        elif best_10.exec_risk <= remaining_budget:
            # Fits but tight — pick by score/risk efficiency
            eff_10 = best_10.score / max(best_10.exec_risk, 1)
            eff_075 = best_075.score / max(best_075.exec_risk, 1)
            return best_10 if eff_10 >= eff_075 else best_075
        else:
            # S1.0 doesn't fit — use S0.75
            return best_075

    return max(variants, key=lambda v: v.score)


# ── Greedy lane selection ──────────────────────────────────────────────────


def _select_lanes(
    candidates: list[ScoredStrategy],
    dd_limit: float,
    budget_pct: float = DEFAULT_BUDGET_PCT,
    max_slots: int = DEFAULT_MAX_SLOTS,
    max_same_filter: int = DEFAULT_MAX_SAME_FILTER,
) -> list[ScoredStrategy]:
    """Greedy selection with per-lane SM optimization.

    For each session slot, compares all SM variants of the best strategy
    and picks the one optimal for the REMAINING DD budget at that point.
    Earlier lanes get more headroom -> more likely S1.0.
    Later lanes see tighter budget -> more likely S0.75.
    """
    budget = dd_limit * (budget_pct / 100)
    remaining = budget

    # Build SM lookup for per-lane optimization
    sm_lookup = _build_sm_lookup(candidates)

    # Get unique base strategies, scored by best variant
    best_per_base: dict[str, ScoredStrategy] = {}
    for base, variants in sm_lookup.items():
        best_per_base[base] = max(variants, key=lambda v: v.score)

    # Sort by best possible score
    ranked_bases = sorted(best_per_base.values(), key=lambda x: x.score, reverse=True)

    selected: list[ScoredStrategy] = []
    used_sessions: set[tuple[str, str]] = set()
    filter_counts: Counter = Counter()

    for candidate in ranked_bases:
        if len(selected) >= max_slots:
            break

        key = (candidate.instrument, candidate.session)
        if key in used_sessions:
            continue

        if filter_counts[candidate.filter_family] >= max_same_filter:
            continue

        # Per-lane SM optimization: pick best variant given REMAINING budget
        variants = sm_lookup[candidate.base_strategy]
        best = _pick_sm_for_lane(variants, remaining)

        if best.exec_risk > remaining:
            # Try the other SM variant
            alternatives = [v for v in variants if v.exec_risk <= remaining]
            if alternatives:
                best = max(alternatives, key=lambda v: v.score)
            else:
                continue

        # Position sizing: how many contracts can we afford?
        # Max contracts = floor(remaining / exec_risk), capped at firm max
        max_cts = int(remaining / best.exec_risk) if best.exec_risk > 0 else 1
        max_cts = max(1, min(max_cts, 10))  # 1-10 range

        # Don't blow more than 30% of TOTAL budget on a single lane
        max_single_lane = budget * 0.30
        if max_cts > 1:
            max_cts = min(max_cts, int(max_single_lane / best.exec_risk))
            max_cts = max(1, max_cts)

        best.contracts = max_cts
        total_risk = best.exec_risk * max_cts

        selected.append(best)
        used_sessions.add(key)
        filter_counts[best.filter_family] += 1
        remaining -= total_risk

    return selected


# ── Compare vs current deployed ────────────────────────────────────────────


def _get_current_deployed(firm: str) -> dict[tuple[str, str], str]:
    deployed = {}
    for prof in ACCOUNT_PROFILES.values():
        if prof.firm == firm and prof.active and prof.daily_lanes:
            for lane in prof.daily_lanes:
                deployed[(lane.instrument, lane.orb_label)] = lane.strategy_id
    return deployed


def _check_switches(
    selected: list[ScoredStrategy],
    current: dict[tuple[str, str], str],
    all_candidates: list[ScoredStrategy],
) -> list[dict]:
    score_lookup = {c.strategy_id: c.score for c in all_candidates}
    switches = []

    for s in selected:
        key = (s.instrument, s.session)
        cur_id = current.get(key)

        if cur_id is None:
            switches.append(
                {
                    "session": s.session,
                    "instrument": s.instrument,
                    "action": "NEW",
                    "old": None,
                    "new": s.strategy_id,
                    "improvement": None,
                }
            )
        elif cur_id != s.strategy_id:
            old_score = score_lookup.get(cur_id, 0)
            improvement = ((s.score - old_score) / old_score * 100) if old_score > 0 else 999
            action = "SWITCH" if improvement >= CARVER_SWITCH_THRESHOLD * 100 else "HOLD (< 20%)"
            switches.append(
                {
                    "session": s.session,
                    "instrument": s.instrument,
                    "action": action,
                    "old": cur_id,
                    "new": s.strategy_id,
                    "improvement": improvement,
                }
            )

    # Sessions in current but NOT in selected
    selected_keys = {(s.instrument, s.session) for s in selected}
    for key, sid in current.items():
        if key not in selected_keys:
            switches.append(
                {
                    "session": key[1],
                    "instrument": key[0],
                    "action": "DROP",
                    "old": sid,
                    "new": None,
                    "improvement": None,
                }
            )

    return switches


# ── ORB caps (P90 from adversarial audit) ──────────────────────────────────

_ORB_CAPS: dict[str, dict[str, float]] = {
    "MNQ": {
        "CME_PRECLOSE": 120.0,
        "NYSE_CLOSE": 100.0,
        "COMEX_SETTLE": 80.0,
        "US_DATA_1000": 120.0,
        "TOKYO_OPEN": 80.0,
        "NYSE_OPEN": 100.0,
        "SINGAPORE_OPEN": 80.0,
        "EUROPE_FLOW": 100.0,
        "LONDON_METALS": 100.0,
        "CME_REOPEN": 80.0,
        "US_DATA_830": 120.0,
        "BRISBANE_1025": 80.0,
    },
    "MES": {
        "CME_PRECLOSE": 30.0,
        "NYSE_CLOSE": 25.0,
        "COMEX_SETTLE": 20.0,
        "NYSE_OPEN": 25.0,
        "LONDON_METALS": 25.0,
        "SINGAPORE_OPEN": 20.0,
    },
    "MGC": {
        "TOKYO_OPEN": 26.0,
        "US_DATA_1000": 30.0,
        "EUROPE_FLOW": 20.0,
        "CME_REOPEN": 20.0,
        "LONDON_METALS": 20.0,
    },
}


# ── Output ──────────────────────────────────────────────────────────────────


def _print_results(
    selected: list[ScoredStrategy],
    firm: str,
    dd_limit: float,
    budget_pct: float,
    switches: list[dict] | None = None,
) -> None:
    budget = dd_limit * (budget_pct / 100)
    total_risk = sum(s.exec_risk * s.contracts for s in selected)
    total_contracts = sum(s.contracts for s in selected)

    print(f"\n{'=' * 155}")
    print(
        f"  OPTIMAL LANES -- Firm: {firm} | DD: ${dd_limit:,.0f} | "
        f"Budget: ${budget:,.0f} ({budget_pct:.0f}%) | Used: ${total_risk:,.0f} ({total_risk / dd_limit * 100:.1f}%) | "
        f"Contracts: {total_contracts}"
    )
    print(f"{'=' * 155}")

    print(
        f"{'#':>2} {'Score':>6} {'Inst':>4} {'Session':<16} {'Time':>5} {'Slot':<6} "
        f"{'Filter':<20} {'RR':>3} {'SM':>4} {'Cts':>3} {'N':>5} {'WR':>5} {'ExpR':>6} {'Sha':>5} "
        f"{'Fwd25':>6} {'TPY':>5} {'Family':<12} {'Risk$':>6} {'Tot$':>6} {'%DD':>5} {'Auto':>4}"
    )
    print("-" * 155)

    for i, s in enumerate(selected, 1):
        auto = "YES" if s.auto_eligible else "no"
        tot_risk = s.exec_risk * s.contracts
        print(
            f"{i:>2} {s.score:>6.3f} {s.instrument:>4} {s.session:<16} {s.bris_time:>5} {s.slot:<6} "
            f"{s.filter_type:<20} {s.rr_target:>3.1f} {s.stop_mult:>4.2f} {s.contracts:>3} {s.sample_size:>5} {s.win_rate:>5.1%} "
            f"{s.expectancy_r:>6.3f} {s.sharpe_ann:>5.2f} "
            f"{s.fwd_r:>+6.1f} {s.tpy:>5.1f} {(s.robustness or '?'):<12} "
            f"{s.exec_risk:>6.0f} {tot_risk:>6.0f} {tot_risk / dd_limit * 100:>5.1f} {auto:>4}"
        )

    # Summary
    instruments = sorted(set(s.instrument for s in selected))
    families = sorted(set(s.filter_family for s in selected))
    auto_count = sum(1 for s in selected if s.auto_eligible)
    total_fwd = sum(s.fwd_r * s.contracts for s in selected)
    total_tpy = sum(s.tpy for s in selected)

    print(f"\n  {len(selected)} lanes. Instruments: {instruments}. Filter families: {families}.")
    print(f"  Auto-eligible: {auto_count}/{len(selected)}. 2025 forward: {total_fwd:+.1f}R. TPY: {total_tpy:.0f}.")

    # SM analysis
    s10_count = sum(1 for s in selected if s.stop_mult >= 1.0)
    s075_count = sum(1 for s in selected if s.stop_mult < 1.0)
    print(f"  Stop multipliers: {s10_count}x S1.0, {s075_count}x S0.75 (optimized for ${dd_limit:,.0f} DD)")

    # Switches
    if switches:
        print(f"\n  {'-' * 70}")
        print("  SWITCHING ANALYSIS (Carver 20% threshold)")
        print(f"  {'-' * 70}")
        for sw in switches:
            if sw["action"] == "NEW":
                print(f"  + NEW:  {sw['instrument']} {sw['session']} -> {sw['new']}")
            elif sw["action"] == "DROP":
                print(f"  - DROP: {sw['instrument']} {sw['session']} <- {sw['old']}")
            else:
                imp = sw["improvement"]
                symbol = "->" if "SWITCH" in sw["action"] else "="
                print(
                    f"  {symbol} {sw['action']}: {sw['instrument']} {sw['session']} | "
                    f"+{imp:.1f}% | {sw['old']} -> {sw['new']}"
                )

    # DailyLaneSpec code
    print(f"\n  {'-' * 70}")
    print("  COPY-PASTE CODE for prop_profiles.py")
    print(f"  {'-' * 70}")
    print("        daily_lanes=(")
    for s in selected:
        cap = _ORB_CAPS.get(s.instrument, {}).get(s.session)
        print("            DailyLaneSpec(")
        print(f'                "{s.strategy_id}",')
        print(f'                "{s.instrument}",')
        print(f'                "{s.session}",')
        if cap:
            print(f"                max_orb_size_pts={cap},")
        print("            ),")
    print("        ),")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimal lane selector — multi-instrument portfolio optimizer")
    parser.add_argument("--firm", default="apex", help="Firm (default: apex)")
    parser.add_argument("--size", type=int, default=100000, help="Account size (default: 100000)")
    parser.add_argument("--dd", type=float, default=None, help="DD limit override")
    parser.add_argument(
        "--budget-pct",
        type=float,
        default=DEFAULT_BUDGET_PCT,
        help=f"Max %% of DD for all lanes (default: {DEFAULT_BUDGET_PCT})",
    )
    parser.add_argument(
        "--max-slots", type=int, default=DEFAULT_MAX_SLOTS, help=f"Max lanes (default: {DEFAULT_MAX_SLOTS})"
    )
    parser.add_argument(
        "--max-same-filter",
        type=int,
        default=DEFAULT_MAX_SAME_FILTER,
        help=f"Max lanes same filter family (default: {DEFAULT_MAX_SAME_FILTER})",
    )
    parser.add_argument("--show-switches", action="store_true", help="Compare vs current deployed lanes")
    parser.add_argument("--auto-only", action="store_true", help="Only auto-eligible strategies")
    args = parser.parse_args()

    # Resolve DD
    dd_limit = args.dd
    if dd_limit is None:
        tier = ACCOUNT_TIERS.get((args.firm, args.size))
        if tier:
            dd_limit = tier.max_dd
        else:
            firm_tiers = [(k, v) for k, v in ACCOUNT_TIERS.items() if k[0] == args.firm]
            dd_limit = max(v.max_dd for _, v in firm_tiers) if firm_tiers else 2000.0
            print(f"  Warning: no tier for ({args.firm}, {args.size}), using DD=${dd_limit:,.0f}")

    all_candidates = _load_all_candidates(args.firm, args.size, dd_limit)
    print(f"  Loaded {len(all_candidates)} candidates across {sorted(set(c.instrument for c in all_candidates))}")

    if args.auto_only:
        all_candidates = [c for c in all_candidates if c.auto_eligible]
        print(f"  Filtered to {len(all_candidates)} auto-eligible")

    selected = _select_lanes(
        all_candidates,
        dd_limit=dd_limit,
        budget_pct=args.budget_pct,
        max_slots=args.max_slots,
        max_same_filter=args.max_same_filter,
    )

    switches = None
    if args.show_switches:
        current = _get_current_deployed(args.firm)
        switches = _check_switches(selected, current, all_candidates)

    _print_results(selected, args.firm, dd_limit, args.budget_pct, switches)


if __name__ == "__main__":
    main()
