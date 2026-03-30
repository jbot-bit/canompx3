"""Lane scoring tool — composite score for strategy routing.

Reads validated_setups + edge_families, computes a 7-factor composite score,
and outputs a ranked table with manual/auto eligibility.

Usage:
    python scripts/tools/score_lanes.py                    # All eligible MNQ strategies
    python scripts/tools/score_lanes.py --instrument MGC   # MGC strategies
    python scripts/tools/score_lanes.py --top 10           # Top 10 only
    python scripts/tools/score_lanes.py --firm topstep     # Score with TopStep profit split
    python scripts/tools/score_lanes.py --session COMEX_SETTLE  # Filter to one session
    python scripts/tools/score_lanes.py --current          # Score currently deployed lanes only

Composite score formula (7 factors):
    score = ExpR * sharpe_adj * ayp * n_confidence * fitness * rr_adj * prop_sm

    ExpR           — raw edge (expectancy_r)
    sharpe_adj     — min(sharpe_ann / 1.5, 2.0)  [1.5 = good, cap at 2.0]
    ayp            — 1.2 if all_years_positive else 1.0
    n_confidence   — min(1.0, sample_size / 300)
    fitness        — ROBUST=1.0, WHITELISTED=0.9, other=0.7
    rr_adj         — 1.0 for RR1.0, 0.95 for RR1.5-2.0, 0.85 for RR2.5+
    prop_sm        — effective profit split rate (firm-specific)

Additional columns:
    forward_r      — 2025 total R (from yearly_results)
    forward_n      — 2025 trade count
    auto_eligible  — True if ROBUST/WHITELISTED + FDR<0.05 + fwd>0 + risk<5%DD
    slot           — "auto" (22:00-06:00 Brisbane) / "manual" (06:00-22:00) / "either"

@research-source score-driven lane selection 2026-03-31
@revalidated-for E2 event-based sessions, post-confluence filters (2026-03-31)
"""

import argparse
import json
from datetime import date

import duckdb

from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.prop_profiles import PROP_FIRM_SPECS

# ── Session timing (Brisbane hours, for routing) ─────────────────────────────

# Pre-compute Brisbane hours for "today" — used for manual/auto slot assignment.
# Sessions between 22:00-06:00 Brisbane = sleeping = auto-preferred.
_BRISBANE_HOURS: dict[str, tuple[int, int]] = {}


def _resolve_brisbane_hours(trading_day: date | None = None) -> dict[str, tuple[int, int]]:
    """Resolve all session start times in Brisbane for routing."""
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


def _slot_label(session: str, hours_map: dict[str, tuple[int, int]]) -> str:
    """Classify session as manual/auto/either based on Brisbane time."""
    hm = hours_map.get(session)
    if not hm:
        return "?"
    h = hm[0]
    # 22:00-06:00 = sleeping = auto-preferred
    if h >= 22 or h < 6:
        return "auto"
    # 06:00-10:00 = early morning = either (watchable but early)
    if 6 <= h < 10:
        return "either"
    # 10:00-22:00 = daytime = manual only (you're awake, not trading hours)
    return "manual"


def _brisbane_time_str(session: str, hours_map: dict[str, tuple[int, int]]) -> str:
    """Format Brisbane time as HH:MM string."""
    hm = hours_map.get(session)
    if not hm:
        return "??:??"
    return f"{hm[0]:02d}:{hm[1]:02d}"


# ── Composite score ──────────────────────────────────────────────────────────


def _sharpe_adj(sharpe_ann: float | None) -> float:
    """Sharpe adjustment: 1.5 = baseline (1.0x), capped at 2.0x."""
    if not sharpe_ann or sharpe_ann <= 0:
        return 0.1
    return min(sharpe_ann / 1.5, 2.0)


def _ayp_factor(all_years_positive: bool | None) -> float:
    return 1.2 if all_years_positive else 1.0


def _n_confidence(sample_size: int | None) -> float:
    if not sample_size or sample_size <= 0:
        return 0.1
    return min(1.0, sample_size / 300)


def _fitness_factor(robustness_status: str | None) -> float:
    if robustness_status == "ROBUST":
        return 1.0
    if robustness_status == "WHITELISTED":
        return 0.9
    return 0.7


def _rr_adj(rr_target: float | None) -> float:
    if not rr_target:
        return 1.0
    if rr_target <= 1.0:
        return 1.0
    if rr_target <= 2.0:
        return 0.95
    return 0.85


def _prop_split(firm: str) -> float:
    """Effective profit split rate for the firm. Fail-closed on unknown firm."""
    spec = PROP_FIRM_SPECS.get(firm)
    if not spec:
        raise ValueError(f"Unknown firm: {firm!r}. Valid: {sorted(PROP_FIRM_SPECS)}")
    # Use the LAST tier (highest rate) as the effective rate for scoring.
    # Rationale: we're scoring long-term potential, not first-payout drag.
    return spec.profit_split_tiers[-1][1]


def _forward_2025(yearly_results_json: str | None) -> tuple[float, int]:
    """Extract 2025 total_r and trade count from yearly_results JSON."""
    if not yearly_results_json:
        return 0.0, 0
    try:
        data = json.loads(yearly_results_json) if isinstance(yearly_results_json, str) else yearly_results_json
        entry = data.get("2025")
        if entry:
            return entry.get("total_r", 0.0), entry.get("trades", 0)
    except (json.JSONDecodeError, AttributeError):
        pass
    return 0.0, 0


def _auto_eligible(
    robustness: str | None,
    fdr_adj_p: float | None,
    forward_r: float,
    risk_pct_dd: float,
) -> bool:
    """Check if strategy is eligible for unattended automation."""
    if robustness not in ("ROBUST", "WHITELISTED"):
        return False
    if fdr_adj_p is None or fdr_adj_p > 0.05:
        return False  # fail-closed: NULL FDR = not proven significant
    if forward_r <= 0:
        return False
    if risk_pct_dd > 5.0:
        return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────


def score_lanes(
    instrument: str = "MNQ",
    firm: str = "topstep",
    dd_limit: float = 2000.0,
    session_filter: str | None = None,
    top_n: int = 30,
    current_only: bool = False,
) -> list[dict]:
    """Score all eligible strategies and return ranked list."""
    hours_map = _resolve_brisbane_hours()
    split = _prop_split(firm)

    # Base query: family heads from validated_setups + edge_families
    where_clauses = [
        "vs.instrument = ?",
        "vs.status = 'active'",
        "vs.entry_model = 'E2'",
        "vs.confirm_bars = 1",
        "vs.orb_minutes = 5",
        "vs.is_family_head = TRUE",
        "vs.sample_size >= 100",
    ]
    params: list = [instrument]

    if session_filter:
        where_clauses.append("vs.orb_label = ?")
        params.append(session_filter)

    if current_only:
        # Only score strategies currently deployed in any active profile.
        # Deployed lanes may NOT be family heads — drop that filter.
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        deployed_ids = set()
        for prof in ACCOUNT_PROFILES.values():
            if prof.active and prof.daily_lanes:
                for lane in prof.daily_lanes:
                    if lane.instrument == instrument:
                        deployed_ids.add(lane.strategy_id)
        if not deployed_ids:
            return []
        placeholders = ",".join(["?"] * len(deployed_ids))
        where_clauses.append(f"vs.strategy_id IN ({placeholders})")
        where_clauses = [c for c in where_clauses if "is_family_head" not in c]
        params.extend(sorted(deployed_ids))

    where_sql = " AND ".join(where_clauses)

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        rows = con.execute(
            f"""
            SELECT vs.strategy_id, vs.orb_label, vs.filter_type, vs.rr_target,
                   vs.sample_size, vs.win_rate, vs.expectancy_r, vs.sharpe_ann,
                   vs.all_years_positive, vs.wfe, vs.median_risk_dollars,
                   vs.stop_multiplier, vs.yearly_results,
                   vs.fdr_adjusted_p, vs.max_drawdown_r, vs.trades_per_year,
                   ef.robustness_status, ef.trade_tier, ef.member_count, ef.pbo
            FROM validated_setups vs
            LEFT JOIN edge_families ef ON vs.family_hash = ef.family_hash
            WHERE {where_sql}
            ORDER BY vs.expectancy_r DESC
            """,
            params,
        ).fetchall()

    results = []
    for r in rows:
        (
            sid,
            session,
            filter_type,
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
            tier,
            members,
            pbo,
        ) = r

        sm = sm or 1.0
        # Execution risk: median_risk_dollars uses raw stop distance (1.0x) regardless
        # of strategy SM. For prop accounts, apply the tighter of strategy SM or 0.75.
        effective_sm = min(sm, 0.75)
        exec_risk = (med_risk or 0) * effective_sm
        risk_pct_dd = (exec_risk / dd_limit * 100) if dd_limit > 0 else 0

        fwd_r, fwd_n = _forward_2025(yearly_json)

        # Composite score
        score = (
            (expr or 0)
            * _sharpe_adj(sharpe)
            * _ayp_factor(ayp)
            * _n_confidence(n)
            * _fitness_factor(robustness)
            * _rr_adj(rr)
            * split
        )

        auto_ok = _auto_eligible(robustness, fdr_p, fwd_r, risk_pct_dd)
        slot = _slot_label(session, hours_map)
        bris_time = _brisbane_time_str(session, hours_map)

        results.append(
            {
                "strategy_id": sid,
                "session": session,
                "bris_time": bris_time,
                "filter": filter_type,
                "rr": rr,
                "n": n,
                "wr": wr,
                "expr": expr,
                "sharpe": sharpe or 0,
                "wfe": wfe or 0,
                "ayp": ayp,
                "family": robustness or "?",
                "members": members or 0,
                "fdr_p": fdr_p,
                "fwd_r": fwd_r,
                "fwd_n": fwd_n,
                "exec_risk": exec_risk,
                "risk_pct_dd": risk_pct_dd,
                "score": score,
                "auto_ok": auto_ok,
                "slot": slot,
                "mdd": mdd or 0,
                "tpy": tpy or 0,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def print_table(results: list[dict], firm: str, dd_limit: float) -> None:
    """Print ranked results as a formatted table."""
    if not results:
        print("No eligible strategies found.")
        return

    print(f"\n{'=' * 130}")
    print(
        f"  LANE SCORING — {results[0]['strategy_id'].split('_')[0]} | Firm: {firm} | DD: ${dd_limit:,.0f} | Profile SM: 0.75"
    )
    print(f"{'=' * 130}")

    # Header
    print(
        f"{'#':>3} {'Score':>6} {'Session':<16} {'Time':>5} {'Slot':<6} "
        f"{'Filter':<20} {'RR':>3} {'N':>5} {'WR':>5} {'ExpR':>6} {'Sha':>5} "
        f"{'Fwd25':>6} {'FwdN':>4} {'Family':<12} {'Risk$':>6} {'%DD':>5} {'Auto':>4}"
    )
    print("-" * 130)

    for i, r in enumerate(results, 1):
        auto_mark = "YES" if r["auto_ok"] else "no"
        print(
            f"{i:>3} {r['score']:>6.3f} {r['session']:<16} {r['bris_time']:>5} {r['slot']:<6} "
            f"{r['filter']:<20} {r['rr']:>3.1f} {r['n']:>5} {r['wr']:>5.1%} {r['expr']:>6.3f} {r['sharpe']:>5.2f} "
            f"{r['fwd_r']:>+6.1f} {r['fwd_n']:>4} {r['family']:<12} {r['exec_risk']:>6.0f} {r['risk_pct_dd']:>5.1f} {auto_mark:>4}"
        )

    # Summary
    auto_count = sum(1 for r in results if r["auto_ok"])
    print(f"\n  {len(results)} strategies scored. {auto_count} auto-eligible.")

    # Top auto candidate
    auto_results = [r for r in results if r["auto_ok"]]
    if auto_results:
        best = auto_results[0]
        print(
            f"  Best auto lane: {best['session']} {best['filter']} (score {best['score']:.3f}, fwd +{best['fwd_r']:.1f}R)"
        )

    # Top manual candidate
    manual_results = [r for r in results if r["slot"] in ("manual", "either")]
    if manual_results:
        best = manual_results[0]
        print(
            f"  Best manual lane: {best['session']} {best['filter']} (score {best['score']:.3f}, fwd +{best['fwd_r']:.1f}R)"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lane scoring tool — composite score for strategy routing")
    parser.add_argument("--instrument", default="MNQ", help="Instrument (default: MNQ)")
    parser.add_argument("--firm", default="topstep", help="Firm for profit split (default: topstep)")
    parser.add_argument("--dd", type=float, default=2000.0, help="DD limit in dollars (default: 2000)")
    parser.add_argument("--session", default=None, help="Filter to one session")
    parser.add_argument("--top", type=int, default=30, help="Show top N (default: 30)")
    parser.add_argument("--current", action="store_true", help="Score only currently deployed lanes")
    args = parser.parse_args()

    results = score_lanes(
        instrument=args.instrument,
        firm=args.firm,
        dd_limit=args.dd,
        session_filter=args.session,
        top_n=args.top,
        current_only=args.current,
    )
    print_table(results, args.firm, args.dd)


if __name__ == "__main__":
    main()
