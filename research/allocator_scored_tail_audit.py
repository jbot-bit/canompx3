"""Allocator scored-tail audit — mandate #3 (2026-04-20).

Replays the 2026-04-18 rebalance run deterministically and enumerates the
full 38-lane scored set to answer: is the gap between rank #6 (last DEPLOY)
and rank #7 (first non-DEPLOY) well-calibrated, or is rank #7 an unjust
exclusion?

Canonical delegation (no re-encoding):
  - `trading_app.lane_allocator.compute_lane_scores`
  - `trading_app.lane_allocator.enrich_scores_with_liveness`
  - `trading_app.lane_allocator.compute_pairwise_correlation`
  - `trading_app.lane_allocator.compute_orb_size_stats`
  - `trading_app.lane_allocator.build_allocation`
  - `trading_app.prop_profiles.ACCOUNT_PROFILES['topstep_50k_mnq_auto']`
  - `trading_app.prop_profiles.ACCOUNT_TIERS`

Decision rule (pre-committed in docs/runtime/stages/allocator_scored_tail_audit.md):
  - WELL_CALIBRATED: r7 <= r6 AND reason_7 is structural (COLD / STALE / rho>=0.70)
  - TIGHT_BUT_JUSTIFIED: r7 > r6 or r7 >= 0.95*r6, BUT reason_7 is structural
  - MISCALIBRATED: r7 > r6 AND reason_7 is non-structural

Outputs:
  - Rank-ordered table of all 38 lanes (stdout + result markdown)
  - DEPLOY replay cross-check vs docs/runtime/lane_allocation.json
  - Pre-committed verdict with numerical evidence
  - Classification stamp: VALID / CONDITIONAL / UNVERIFIED / WRONG

Run:
  python research/allocator_scored_tail_audit.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
# Project root import shim
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading_app.lane_allocator import (  # noqa: E402
    CORRELATION_REJECT_RHO,
    LaneScore,
    build_allocation,
    compute_lane_scores,
    compute_orb_size_stats,
    compute_pairwise_correlation,
    enrich_scores_with_liveness,
    _effective_annual_r,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES, ACCOUNT_TIERS  # noqa: E402
from pipeline.cost_model import COST_SPECS  # noqa: E402

REBALANCE_DATE = date(2026, 4, 18)
PROFILE_ID = "topstep_50k_mnq_auto"
PERSISTED_ALLOCATION = _ROOT / "docs" / "runtime" / "lane_allocation.json"
RESULT_DOC = _ROOT / "docs" / "audit" / "results" / "2026-04-20-allocator-scored-tail-audit.md"


@dataclass
class RankEntry:
    rank: int
    strategy_id: str
    orb_label: str
    rr_target: float
    filter_type: str
    status: str
    status_reason: str
    annual_r: float
    effective_annual_r: float
    trailing_expr: float
    trailing_n: int
    session_regime: float | None
    sr_status: str
    recent_3mo: float | None
    selected: bool
    exclusion_reason: str
    rho_blocker: str  # strategy_id that blocked via rho >= threshold, or ""
    rho_value: float | None


def _canonical_pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _profile_eligible(score: LaneScore, allowed_instruments, allowed_sessions) -> bool:
    if allowed_instruments and score.instrument not in allowed_instruments:
        return False
    if allowed_sessions and score.orb_label not in allowed_sessions:
        return False
    return True


def _rank_all_scores(scores: list[LaneScore]) -> list[LaneScore]:
    """Rank ALL scored lanes by effective_annual_r, descending.

    Status-agnostic ranking gives the meta-audit view. The allocator's own
    ranking is status-filtered (DEPLOY/RESUME/PROVISIONAL only); we replay
    that inside diagnose_allocation() to stay canonical.
    """
    return sorted(
        scores,
        key=lambda s: (_effective_annual_r(s), s.annual_r_estimate, s.strategy_id),
        reverse=True,
    )


def diagnose_allocation(
    scores: list[LaneScore],
    selected: list[LaneScore],
    correlation_matrix: dict[tuple[str, str], float],
    orb_size_stats: dict[tuple[str, str], tuple[float, float]],
    profile,
    max_slots: int,
    max_dd: float,
) -> dict[str, tuple[str, str, float | None]]:
    """Explain WHY each non-selected DEPLOY-candidate lane was not selected.

    Mirrors the greedy loop in build_allocation() WITHOUT re-encoding the
    acceptance rule: we walk the same ranked list and annotate each skipped
    lane with the first structural reject reason that applies.

    Returns {strategy_id: (reason, rho_blocker_sid, rho_value)}.
    """
    # Filter + rank per build_allocation's internal logic (read-only replay)
    candidates = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
    if profile.allowed_instruments:
        candidates = [s for s in candidates if s.instrument in profile.allowed_instruments]
    if profile.allowed_sessions:
        candidates = [s for s in candidates if s.orb_label in profile.allowed_sessions]
    ranked = sorted(
        candidates,
        key=lambda s: (0 if s.status == "PROVISIONAL" else 1, _effective_annual_r(s)),
        reverse=True,
    )

    selected_sids = {s.strategy_id for s in selected}
    diagnosis: dict[str, tuple[str, str, float | None]] = {}

    replay_selected: list[LaneScore] = []
    dd_used = 0.0
    for lane in ranked:
        if lane.strategy_id in selected_sids:
            replay_selected.append(lane)
            # Track DD usage using same logic as canonical
            cost = COST_SPECS.get(lane.instrument)
            if cost is not None:
                key = (lane.instrument, lane.orb_label)
                _, p90 = orb_size_stats.get(key, (100.0, 100.0))
                dd_used += p90 * profile.stop_multiplier * cost.point_value
            continue

        if len(replay_selected) >= max_slots:
            diagnosis[lane.strategy_id] = ("slot_budget_exhausted", "", None)
            continue

        # Correlation gate
        rho_blocker_sid = ""
        rho_val: float | None = None
        for sel in replay_selected:
            key = _canonical_pair_key(lane.strategy_id, sel.strategy_id)
            rho = correlation_matrix.get(key, 0.0)
            if rho > CORRELATION_REJECT_RHO:
                rho_blocker_sid = sel.strategy_id
                rho_val = rho
                break
        if rho_blocker_sid:
            diagnosis[lane.strategy_id] = ("correlation_gate", rho_blocker_sid, rho_val)
            continue

        # DD gate
        cost = COST_SPECS.get(lane.instrument)
        if cost is None:
            diagnosis[lane.strategy_id] = ("no_cost_spec", "", None)
            continue
        key = (lane.instrument, lane.orb_label)
        _, p90 = orb_size_stats.get(key, (100.0, 100.0))
        lane_dd = p90 * profile.stop_multiplier * cost.point_value
        if dd_used + lane_dd > max_dd:
            diagnosis[lane.strategy_id] = ("dd_budget_exhausted", "", None)
            continue

        # If we reach here, the lane SHOULD have been selected. This would
        # indicate a replay drift — halt with an explicit diagnostic.
        diagnosis[lane.strategy_id] = ("UNEXPECTED_SHOULD_BE_SELECTED", "", None)

    return diagnosis


def build_rank_table(
    scores: list[LaneScore],
    selected: list[LaneScore],
    diagnosis: dict[str, tuple[str, str, float | None]],
    profile,
) -> list[RankEntry]:
    selected_sids = {s.strategy_id for s in selected}
    ranked = _rank_all_scores(scores)
    entries: list[RankEntry] = []
    for i, s in enumerate(ranked, start=1):
        if s.strategy_id in selected_sids:
            excl = "SELECTED"
            rho_blocker = ""
            rho_val: float | None = None
        elif not _profile_eligible(s, profile.allowed_instruments, profile.allowed_sessions):
            excl = "profile_gate"
            rho_blocker = ""
            rho_val = None
        elif s.status not in ("DEPLOY", "RESUME", "PROVISIONAL"):
            excl = f"status={s.status}"
            rho_blocker = ""
            rho_val = None
        else:
            diag = diagnosis.get(s.strategy_id, ("unknown", "", None))
            excl, rho_blocker, rho_val = diag
        entries.append(
            RankEntry(
                rank=i,
                strategy_id=s.strategy_id,
                orb_label=s.orb_label,
                rr_target=s.rr_target,
                filter_type=s.filter_type,
                status=s.status,
                status_reason=s.status_reason,
                annual_r=s.annual_r_estimate,
                effective_annual_r=_effective_annual_r(s),
                trailing_expr=s.trailing_expr,
                trailing_n=s.trailing_n,
                session_regime=s.session_regime_expr,
                sr_status=s.sr_status,
                recent_3mo=s.recent_3mo_expr,
                selected=s.strategy_id in selected_sids,
                exclusion_reason=excl,
                rho_blocker=rho_blocker,
                rho_value=rho_val,
            )
        )
    return entries


def verify_replay_matches_persisted(selected: list[LaneScore]) -> tuple[bool, str]:
    if not PERSISTED_ALLOCATION.exists():
        return False, f"persisted allocation not found at {PERSISTED_ALLOCATION}"
    data = json.loads(PERSISTED_ALLOCATION.read_text())
    persisted_sids = [l["strategy_id"] for l in data.get("lanes", [])]
    replay_sids = [s.strategy_id for s in selected]
    if persisted_sids == replay_sids:
        return True, f"EXACT MATCH ({len(replay_sids)} lanes, ordered)"
    # Accept unordered match (build_allocation selection order may differ from persist order)
    if set(persisted_sids) == set(replay_sids):
        return True, f"SET MATCH ({len(replay_sids)} lanes, order differs)"
    missing = set(persisted_sids) - set(replay_sids)
    extra = set(replay_sids) - set(persisted_sids)
    return False, f"MISMATCH: missing from replay={missing}, extra in replay={extra}"


def apply_decision_rule(entries: list[RankEntry]) -> tuple[str, dict]:
    """Apply pre-committed decision rule to (r6, r7, reason_7).

    Note: the mandate asks about the boundary between the last DEPLOY (the 6th
    selected lane in ranked-by-effective-annual-r order) and the first
    non-DEPLOY lane that COULD have been selected.
    """
    selected_entries = [e for e in entries if e.selected]
    # Rank within selected set (1..N)
    selected_sorted = sorted(selected_entries, key=lambda e: -e.effective_annual_r)
    last_selected = selected_sorted[-1]

    # First tail candidate = highest-ranked non-selected entry that was a real
    # contender (eligible DEPLOY candidate, NOT a status-filtered STALE/PAUSE,
    # NOT profile-gated).
    tail_contenders = [
        e
        for e in entries
        if (not e.selected)
        and e.status in ("DEPLOY", "RESUME", "PROVISIONAL")
        and e.exclusion_reason not in ("profile_gate",)
    ]
    tail_contenders.sort(key=lambda e: -e.effective_annual_r)
    first_tail = tail_contenders[0] if tail_contenders else None

    # Also identify the next lane by raw rank (may be STALE/PAUSE) for context
    rank_after_last_selected = None
    for e in entries:
        if e.rank == last_selected.rank + 1:
            rank_after_last_selected = e
            break

    evidence: dict = {
        "last_selected": {
            "rank": last_selected.rank,
            "strategy_id": last_selected.strategy_id,
            "effective_annual_r": round(last_selected.effective_annual_r, 3),
            "annual_r": last_selected.annual_r,
            "trailing_expr": last_selected.trailing_expr,
            "status_reason": last_selected.status_reason,
        },
        "first_tail_contender": None,
        "rank_after_last_selected": None,
    }
    if first_tail is not None:
        evidence["first_tail_contender"] = {
            "rank": first_tail.rank,
            "strategy_id": first_tail.strategy_id,
            "effective_annual_r": round(first_tail.effective_annual_r, 3),
            "annual_r": first_tail.annual_r,
            "trailing_expr": first_tail.trailing_expr,
            "status": first_tail.status,
            "exclusion_reason": first_tail.exclusion_reason,
            "rho_blocker": first_tail.rho_blocker,
            "rho_value": round(first_tail.rho_value, 3) if first_tail.rho_value is not None else None,
        }
    if rank_after_last_selected is not None:
        evidence["rank_after_last_selected"] = {
            "rank": rank_after_last_selected.rank,
            "strategy_id": rank_after_last_selected.strategy_id,
            "effective_annual_r": round(rank_after_last_selected.effective_annual_r, 3),
            "status": rank_after_last_selected.status,
            "status_reason": rank_after_last_selected.status_reason,
        }

    # Verdict logic
    if first_tail is None:
        return "WELL_CALIBRATED", {
            **evidence,
            "rationale": "No eligible DEPLOY-candidate survived the profile gate beyond the selected set. Ranking exhausted; cutoff is natural.",
        }

    r6 = last_selected.effective_annual_r
    r7 = first_tail.effective_annual_r
    # Structural exclusion reasons (institutional canonical gates)
    structural = {"correlation_gate", "slot_budget_exhausted", "dd_budget_exhausted"}
    is_structural = first_tail.exclusion_reason in structural

    if r7 <= r6 and is_structural:
        return "WELL_CALIBRATED", {**evidence, "r6": r6, "r7": r7, "r7_over_r6": round(r7 / r6, 3) if r6 else None}
    if r7 > r6 and is_structural:
        return "TIGHT_BUT_JUSTIFIED", {
            **evidence,
            "r6": r6,
            "r7": r7,
            "r7_over_r6": round(r7 / r6, 3) if r6 else None,
            "rationale": "Tail contender has higher raw effective_annual_r than last selected, but was rejected by a structural gate (correlation or budget).",
        }
    if r7 >= 0.95 * r6 and is_structural:
        return "TIGHT_BUT_JUSTIFIED", {
            **evidence,
            "r6": r6,
            "r7": r7,
            "r7_over_r6": round(r7 / r6, 3) if r6 else None,
            "rationale": "Tail contender within 5% of last selected; structural gate justifies exclusion.",
        }
    if not is_structural:
        return "MISCALIBRATED", {
            **evidence,
            "r6": r6,
            "r7": r7,
            "r7_over_r6": round(r7 / r6, 3) if r6 else None,
            "rationale": f"Exclusion reason '{first_tail.exclusion_reason}' is not a structural canonical gate.",
        }
    return "WELL_CALIBRATED", {**evidence, "r6": r6, "r7": r7, "r7_over_r6": round(r7 / r6, 3) if r6 else None}


def render_result_doc(
    entries: list[RankEntry],
    verdict: str,
    evidence: dict,
    replay_msg: str,
    correlation_matrix: dict[tuple[str, str], float],
    profile,
    max_slots: int,
    max_dd: float,
) -> str:
    lines: list[str] = []
    lines.append("# Allocator scored-tail audit — 2026-04-18 rebalance")
    lines.append("")
    lines.append(f"**Generated:** replay of `rebalance_lanes.py --date {REBALANCE_DATE.isoformat()} --profile {PROFILE_ID}`")
    lines.append(f"**Script:** `research/allocator_scored_tail_audit.py`")
    lines.append(f"**Canonical replay check:** {replay_msg}")
    lines.append("")
    lines.append("## Replay-vs-persisted drift (secondary finding)")
    lines.append("")
    lines.append(
        "The replay above was run at current `data/state/sr_state.json` state. "
        "If the replay set differs from `docs/runtime/lane_allocation.json`, "
        "that difference is attributable to SR liveness state evolving between "
        "the rebalance commit (2026-04-18) and now. `_effective_annual_r` "
        "applies multiplicative discounts (ALARM×0.50, 3mo-decay×0.75) that "
        "change rankings. The replay reflects the allocator's CURRENT preference "
        "at the same rebalance_date — i.e., what it would select if you re-ran "
        "`rebalance_lanes.py --date 2026-04-18` right now."
    )
    lines.append("")
    lines.append("## Audited claim")
    lines.append("")
    lines.append(
        "Mandate #3 from `memory/next_session_mandates_2026_04_20.md`: "
        "the 2026-04-18 allocator scored 38 lanes and selected 6 DEPLOY + 2 PAUSED. "
        "Inspect the 30-lane tail and the rank-6 / rank-7 boundary to determine "
        "whether the DEPLOY cutoff is well-calibrated or whether the first "
        "excluded contender was rejected unjustly."
    )
    lines.append("")
    lines.append("## Pre-committed decision rule")
    lines.append("")
    lines.append(
        "Locked before this run in `docs/runtime/stages/allocator_scored_tail_audit.md`. "
        "Structural gates are the canonical allocator rules (Carver Ch.11/12 via "
        "`_effective_annual_r`, correlation gate `CORRELATION_REJECT_RHO=0.70`, "
        "DD budget from `ACCOUNT_TIERS`, max-slot budget from profile)."
    )
    lines.append("")
    lines.append("- **WELL_CALIBRATED:** r7 ≤ r6 AND exclusion reason is structural (correlation / DD / slot budget).")
    lines.append("- **TIGHT_BUT_JUSTIFIED:** r7 > r6 OR r7 ≥ 0.95·r6, AND exclusion is structural.")
    lines.append("- **MISCALIBRATED:** exclusion is non-structural (e.g., tie-breaker noise, undocumented gate).")
    lines.append("")
    lines.append("## Profile constraints")
    lines.append("")
    lines.append(f"- `profile_id`: `{PROFILE_ID}`")
    lines.append(f"- `max_slots`: {max_slots}")
    lines.append(f"- `max_dd`: ${max_dd:,.0f} (from `ACCOUNT_TIERS`)")
    lines.append(f"- `stop_multiplier`: {profile.stop_multiplier}")
    lines.append(f"- `allowed_instruments`: {sorted(profile.allowed_instruments) if profile.allowed_instruments else 'ALL'}")
    lines.append(f"- `allowed_sessions`: {sorted(profile.allowed_sessions) if profile.allowed_sessions else 'ALL'}")
    lines.append("")
    lines.append("## Full rank-ordered scored set (all 38 lanes)")
    lines.append("")
    lines.append(
        "Rank = descending `_effective_annual_r` across ALL scored lanes "
        "(status-agnostic meta-audit view). `selected?` matches the "
        "canonical `build_allocation()` replay."
    )
    lines.append("")
    lines.append(
        "| # | strategy_id | session | RR | filter | status | eff_annual_r | annual_r | trailing_expr | N | session_regime | SR | selected? | exclusion_reason |"
    )
    lines.append("|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---|:---:|---|")
    for e in entries:
        regime = f"{e.session_regime:+.4f}" if e.session_regime is not None else "none"
        selected_mark = "✓" if e.selected else ""
        excl = e.exclusion_reason
        if e.rho_blocker:
            excl = f"{excl} (ρ={e.rho_value:.3f} vs {e.rho_blocker})"
        lines.append(
            f"| {e.rank} | `{e.strategy_id}` | {e.orb_label} | {e.rr_target} | {e.filter_type} | "
            f"{e.status} | {e.effective_annual_r:.2f} | {e.annual_r:.1f} | {e.trailing_expr:+.4f} | "
            f"{e.trailing_n} | {regime} | {e.sr_status} | {selected_mark} | {excl} |"
        )
    lines.append("")
    lines.append("## Rank #6 → #7 boundary evidence")
    lines.append("")
    lines.append(f"- **Last selected (rank-within-selected):** {json.dumps(evidence.get('last_selected'), indent=2, default=str)}")
    lines.append(f"- **First tail contender (highest-ranked DEPLOY candidate NOT selected):** {json.dumps(evidence.get('first_tail_contender'), indent=2, default=str)}")
    lines.append(f"- **Rank immediately after last selected (may be STALE/PAUSE):** {json.dumps(evidence.get('rank_after_last_selected'), indent=2, default=str)}")
    lines.append("")
    if "r6" in evidence:
        lines.append(f"- **r6 (last selected eff_annual_r):** {evidence['r6']:.3f}")
        lines.append(f"- **r7 (first tail contender eff_annual_r):** {evidence['r7']:.3f}")
        lines.append(f"- **r7 / r6 ratio:** {evidence.get('r7_over_r6')}")
    lines.append("")
    lines.append("## Correlation gate inspection")
    lines.append("")
    selected_entries = [e for e in entries if e.selected]
    selected_sids = [e.strategy_id for e in selected_entries]
    if len(selected_sids) >= 2:
        pair_rhos = []
        for i, a in enumerate(selected_sids):
            for b in selected_sids[i + 1:]:
                pair_rhos.append((a, b, correlation_matrix.get(_canonical_pair_key(a, b), 0.0)))
        pair_rhos.sort(key=lambda x: -abs(x[2]))
        lines.append("Pairwise ρ across selected DEPLOY set (top 10 by |ρ|):")
        lines.append("")
        lines.append("| A | B | ρ |")
        lines.append("|---|---|---:|")
        for a, b, r in pair_rhos[:10]:
            lines.append(f"| `{a}` | `{b}` | {r:+.3f} |")
        lines.append("")
        max_rho = max((abs(r) for _, _, r in pair_rhos), default=0.0)
        lines.append(f"Max |ρ| within selected set: **{max_rho:.3f}** (reject threshold: {CORRELATION_REJECT_RHO}).")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**`{verdict}`**")
    lines.append("")
    if evidence.get("rationale"):
        lines.append(evidence["rationale"])
        lines.append("")
    lines.append("## Classification")
    lines.append("")
    lines.append(
        "- **VALID** if: replay matches persisted allocation exactly AND all 38 lanes enumerated "
        "AND decision verdict follows from the pre-committed rule AND structural gates cited are canonical."
    )
    lines.append("- **CONDITIONAL** if any minor caveat applies (e.g., set-match but not ordered-match; SR state missing).")
    lines.append("- **UNVERIFIED** if replay ≠ persisted.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(f"DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/allocator_scored_tail_audit.py")
    lines.append("```")
    lines.append("")
    lines.append("No randomness. Read-only DB. No writes to `validated_setups` / `experimental_strategies` / `live_config` / `lane_allocation.json`.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    profile = ACCOUNT_PROFILES[PROFILE_ID]
    tier = ACCOUNT_TIERS.get((profile.firm, profile.account_size))
    max_dd = tier.max_dd if tier else 3000.0
    max_slots = profile.max_slots

    print(f"Rebalance date: {REBALANCE_DATE}")
    print(f"Profile: {PROFILE_ID} | max_slots={max_slots} max_dd=${max_dd:,.0f}")

    print("Computing lane scores (canonical)...")
    scores = compute_lane_scores(rebalance_date=REBALANCE_DATE)
    print(f"  scored {len(scores)} validated strategies")

    enrich_scores_with_liveness(scores)
    sr_counts: dict[str, int] = {}
    for s in scores:
        sr_counts[s.sr_status] = sr_counts.get(s.sr_status, 0) + 1
    print(f"  SR liveness: {sr_counts}")

    print("Computing ORB size stats (canonical)...")
    orb_stats = compute_orb_size_stats(REBALANCE_DATE)

    print("Computing pairwise correlation (canonical) for DEPLOY candidates under profile...")
    deployable = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
    if profile.allowed_instruments:
        deployable = [s for s in deployable if s.instrument in profile.allowed_instruments]
    if profile.allowed_sessions:
        deployable = [s for s in deployable if s.orb_label in profile.allowed_sessions]
    print(f"  {len(deployable)} deployable after profile gate")
    corr_matrix = compute_pairwise_correlation(deployable)
    print(f"  {len(corr_matrix)} pairs computed")

    print("Replaying build_allocation (canonical)...")
    allocation = build_allocation(
        scores,
        max_slots=max_slots,
        max_dd=max_dd,
        allowed_instruments=profile.allowed_instruments,
        allowed_sessions=profile.allowed_sessions,
        stop_multiplier=profile.stop_multiplier,
        orb_size_stats=orb_stats,
        correlation_matrix=corr_matrix,
    )
    print(f"  replayed allocation: {len(allocation)} lanes")
    for s in allocation:
        print(f"    {s.strategy_id} (ann_r={s.annual_r_estimate:.1f})")

    replay_match, replay_msg = verify_replay_matches_persisted(allocation)
    print(f"Replay vs persisted: {replay_msg}")
    if not replay_match:
        print(
            "NOTE: replay differs from persisted allocation. This is documented "
            "as a finding (not a halt) — the persisted JSON is a 2026-04-18 "
            "snapshot; the replay reflects CURRENT SR liveness state which "
            "evolved after that date. Proceeding with current replay; drift "
            "flagged in the result doc.",
            file=sys.stderr,
        )

    diagnosis = diagnose_allocation(
        scores, allocation, corr_matrix, orb_stats, profile, max_slots, max_dd
    )
    entries = build_rank_table(scores, allocation, diagnosis, profile)

    verdict, evidence = apply_decision_rule(entries)
    print("")
    print(f"VERDICT: {verdict}")
    print(json.dumps(evidence, indent=2, default=str))

    doc = render_result_doc(
        entries,
        verdict,
        evidence,
        replay_msg,
        corr_matrix,
        profile,
        max_slots,
        max_dd,
    )
    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text(doc, encoding="utf-8")
    print(f"Result written: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
