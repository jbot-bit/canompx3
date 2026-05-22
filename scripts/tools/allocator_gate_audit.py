"""Allocator-gate attrition audit — read-only diagnostic.

Maps strategy attrition from validated_setups (deployable scope) through every
allocator gate down to the final deployed N. Reports per-gate counts, verbatim
status_reason exemplars, and verifies a conservation invariant.

USAGE
    python scripts/tools/allocator_gate_audit.py --profile topstep_50k_mnq_auto
    python scripts/tools/allocator_gate_audit.py --all-profiles
    python scripts/tools/allocator_gate_audit.py --profile <id> --json out.json

NOT a production tool — this is a one-shot diagnostic. No writes, no save_allocation,
no INSERT/UPDATE/DELETE. DB is opened read-only. Gate attribution is by parsing
`status_reason` strings produced by the canonical classifier; this file does NOT
re-encode any predicate.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.db_contracts import deployable_validated_relation
from pipeline.paths import GOLD_DB_PATH
from trading_app.lane_allocator import (
    LaneScore,
    apply_chordia_gate,
    build_allocation,
    compute_lane_scores,
    compute_orb_size_stats,
    compute_pairwise_correlation,
    enrich_scores_with_liveness,
)
from trading_app.prop_profiles import (
    ACCOUNT_PROFILES,
    ACCOUNT_TIERS,
    legacy_lane_allocation_path,
    resolve_allocation_json,
)


def gate_0_entry_count() -> tuple[int, dict[str, int]]:
    """Count canonical deployable shelf — the entry universe for the allocator."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        rel = deployable_validated_relation(con, alias="vs")
        n = con.execute(f"SELECT COUNT(*) FROM {rel}").fetchone()[0]
        rows = con.execute(f"SELECT instrument, COUNT(*) FROM {rel} GROUP BY instrument ORDER BY 1").fetchall()
        return int(n), {r[0]: int(r[1]) for r in rows}
    finally:
        con.close()


def classify_pre_chordia(scores: list[LaneScore]) -> dict[str, list[LaneScore]]:
    """Bucket post-compute_lane_scores rows by status_reason fingerprint.

    Mutually exclusive — each LaneScore is in exactly one bucket. Mapping is
    verified against lane_allocator.py source:

      Gate 1 (E2_SAFETY)  : status_reason.startswith("live tradeability gate")
      Gate 2a (STALE)     : status_reason == "No trades in trailing window"
      Gate 2b (STALE)     : status_reason.startswith("No regime data, thin trades")
      Gate 3 (REGIME_COLD): status_reason.startswith("Session regime COLD")
      Gate 3b (NO_REGIME) : status_reason.startswith("No regime data, trailing negative")
      DEPLOYABLE          : status in (DEPLOY, RESUME, PROVISIONAL) — passes to Chordia
    """
    buckets: dict[str, list[LaneScore]] = {
        "gate_1_e2_safety": [],
        "gate_2a_no_trades": [],
        "gate_2b_thin_no_regime": [],
        "gate_3_regime_cold": [],
        "gate_3b_no_regime_negative": [],
        "deployable_pre_chordia": [],
        "other_pause": [],
    }
    for s in scores:
        r = s.status_reason or ""
        if r.startswith("live tradeability gate"):
            buckets["gate_1_e2_safety"].append(s)
        elif r == "No trades in trailing window":
            buckets["gate_2a_no_trades"].append(s)
        elif r.startswith("No regime data, thin trades"):
            buckets["gate_2b_thin_no_regime"].append(s)
        elif r.startswith("Session regime COLD"):
            buckets["gate_3_regime_cold"].append(s)
        elif r.startswith("No regime data, trailing negative"):
            buckets["gate_3b_no_regime_negative"].append(s)
        elif s.status in ("DEPLOY", "RESUME", "PROVISIONAL"):
            buckets["deployable_pre_chordia"].append(s)
        else:
            buckets["other_pause"].append(s)
    return buckets


def bucket_summary(name: str, rows: list[LaneScore]) -> dict:
    examples = []
    seen_reasons = []
    for s in rows[:3]:
        examples.append(s.strategy_id)
    for s in rows:
        if s.status_reason not in seen_reasons:
            seen_reasons.append(s.status_reason)
            if len(seen_reasons) >= 3:
                break
    return {
        "name": name,
        "count": len(rows),
        "examples": examples,
        "reason_exemplars": seen_reasons,
    }


def chordia_drops(pre: list[LaneScore], post: list[LaneScore]) -> list[LaneScore]:
    """Return rows whose status flipped to PAUSE because of apply_chordia_gate."""
    pre_status = {s.strategy_id: s.status for s in pre}
    drops: list[LaneScore] = []
    for s in post:
        prior = pre_status.get(s.strategy_id)
        if prior in ("DEPLOY", "RESUME", "PROVISIONAL") and s.status == "PAUSE":
            drops.append(s)
    return drops


def sr_distribution(scores: list[LaneScore]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in scores:
        out[s.sr_status or "UNKNOWN"] = out.get(s.sr_status or "UNKNOWN", 0) + 1
    return out


def audit_profile(
    profile_id: str,
    scores_after_chordia: list[LaneScore],
    orb_stats,
    rebalance_date: date,
) -> dict:
    """Run profile-scoped gates (5-10) against an already-Chordia-gated score list."""
    profile = ACCOUNT_PROFILES[profile_id]
    tier = ACCOUNT_TIERS.get((profile.firm, profile.account_size))
    max_dd = tier.max_dd if tier else 3000.0

    # Scope Gates 5/6 to rows that survived Gates 1-4 — i.e., still DEPLOY-class.
    # Counting whitelist failures against the full 844 would double-count rows
    # already attributed to E2_SAFETY / STALE / REGIME / CHORDIA.
    survivors = [s for s in scores_after_chordia if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]

    # Gate 5: profile whitelist (instrument + session) — applied to survivors.
    after_instr = [
        s for s in survivors if not profile.allowed_instruments or s.instrument in profile.allowed_instruments
    ]
    after_session = [s for s in after_instr if not profile.allowed_sessions or s.orb_label in profile.allowed_sessions]
    gate_5_dropped_instr = len(survivors) - len(after_instr)
    gate_5_dropped_session = len(after_instr) - len(after_session)
    gate_5_dropped = gate_5_dropped_instr + gate_5_dropped_session

    # Gate 6: NON_DEPLOYABLE residual — should be 0 since survivors are already
    # filtered to DEPLOY-class. Kept for invariant-clarity; non-zero would
    # indicate a status-class drift between this script and build_allocation.
    deployable = [s for s in after_session if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
    gate_6_dropped = len(after_session) - len(deployable)
    eligible = len(deployable)

    # Correlation matrix for the eligible candidates
    corr_matrix = compute_pairwise_correlation(deployable) if deployable else {}

    # Selection loop (Gates 7-10 lumped: correlation, DD budget, hysteresis, slot cap)
    # build_allocation re-applies chordia + tradeability gates; that's idempotent
    # because rows already PAUSE/STALE are skipped at apply_*_gate entry points.
    selected = build_allocation(
        scores_after_chordia,
        max_slots=profile.max_slots,
        max_dd=max_dd,
        allowed_instruments=profile.allowed_instruments,
        allowed_sessions=profile.allowed_sessions,
        stop_multiplier=profile.stop_multiplier,
        orb_size_stats=orb_stats,
        correlation_matrix=corr_matrix,
    )
    selection_loop_dropped = eligible - len(selected)

    # Informational: SR-alarm count among eligible-but-not-selected
    selected_ids = {s.strategy_id for s in selected}
    sr_alarm_among_dropped = sum(1 for s in deployable if s.strategy_id not in selected_ids and s.sr_status == "ALARM")

    return {
        "profile_id": profile_id,
        "max_slots": profile.max_slots,
        "max_dd": max_dd,
        "allowed_instruments": sorted(profile.allowed_instruments) if profile.allowed_instruments else None,
        "allowed_sessions": sorted(profile.allowed_sessions) if profile.allowed_sessions else None,
        "gate_5_profile_whitelist": {
            "dropped_total": gate_5_dropped,
            "dropped_by_instrument": gate_5_dropped_instr,
            "dropped_by_session": gate_5_dropped_session,
        },
        "gate_6_non_deployable": gate_6_dropped,
        "eligible_for_selection": eligible,
        "selection_loop_dropped": selection_loop_dropped,
        "sr_alarm_among_loop_dropped": sr_alarm_among_dropped,
        "selected": [s.strategy_id for s in selected],
        "selected_count": len(selected),
    }


def cross_check_lane_allocation(profile_id: str, selected_ids: list[str]) -> dict:
    """Compare against committed allocation file if it targets the same profile.

    Delegates to ``resolve_allocation_json`` (Stage 1b authority inversion):
    honors new-path-first + profile-mismatch guard. On miss, residual probe
    of the legacy file surfaces a profile_id-mismatch result distinct from
    file-absent, preserving the original three-way response shape.
    """
    result = resolve_allocation_json(profile_id)
    if result.data is not None:
        data = result.data
        committed = [lane.get("strategy_id") for lane in data.get("lanes", [])]
        return {
            "present": True,
            "readable": True,
            "profile_id_in_file": profile_id,
            "matches_audit_profile": True,
            "committed_strategy_ids": committed,
            "audit_strategy_ids": selected_ids,
            "set_match": set(committed) == set(selected_ids),
        }
    legacy_path = legacy_lane_allocation_path()
    if not legacy_path.exists():
        return {"present": False}
    try:
        legacy_data = json.loads(legacy_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"present": True, "readable": False, "error": str(exc)}
    return {
        "present": True,
        "readable": True,
        "profile_id_in_file": legacy_data.get("profile_id") if isinstance(legacy_data, dict) else None,
        "matches_audit_profile": False,
    }


def print_profile_table(profile_audit: dict, gate_0: int, gate_counts: dict) -> None:
    pid = profile_audit["profile_id"]
    print()
    print(f"=== Profile: {pid} ===")
    rows = [
        ("Gate 0", "ENTRY (deployable validated)", gate_0, ""),
        (
            "Gate 1",
            "E2_SAFETY",
            gate_counts["gate_1_e2_safety"]["count"],
            _first(gate_counts["gate_1_e2_safety"]["reason_exemplars"]),
        ),
        (
            "Gate 2a",
            "STALE (no monthly data)",
            gate_counts["gate_2a_no_trades"]["count"],
            _first(gate_counts["gate_2a_no_trades"]["reason_exemplars"]),
        ),
        (
            "Gate 2b",
            "STALE (thin + no regime)",
            gate_counts["gate_2b_thin_no_regime"]["count"],
            _first(gate_counts["gate_2b_thin_no_regime"]["reason_exemplars"]),
        ),
        (
            "Gate 3",
            "REGIME_COLD",
            gate_counts["gate_3_regime_cold"]["count"],
            _first(gate_counts["gate_3_regime_cold"]["reason_exemplars"]),
        ),
        (
            "Gate 3b",
            "NO_REGIME (trailing neg)",
            gate_counts["gate_3b_no_regime_negative"]["count"],
            _first(gate_counts["gate_3b_no_regime_negative"]["reason_exemplars"]),
        ),
        ("Other", "Pre-Chordia PAUSE (uncategorized)", gate_counts["other_pause"]["count"], ""),
        (
            "Gate 4",
            "CHORDIA",
            gate_counts["gate_4_chordia"]["count"],
            _first(gate_counts["gate_4_chordia"]["reason_exemplars"]),
        ),
        (
            "Gate 5",
            "PROFILE_WHITELIST (instrument+session)",
            profile_audit["gate_5_profile_whitelist"]["dropped_total"],
            f"by_instr={profile_audit['gate_5_profile_whitelist']['dropped_by_instrument']}, "
            f"by_session={profile_audit['gate_5_profile_whitelist']['dropped_by_session']}",
        ),
        ("Gate 6", "NON_DEPLOYABLE residual", profile_audit["gate_6_non_deployable"], ""),
        (
            "Gate 7-10",
            "SELECTION_LOOP (corr+DD+hysteresis+slot)",
            profile_audit["selection_loop_dropped"],
            f"sr_alarm_among_dropped={profile_audit['sr_alarm_among_loop_dropped']}",
        ),
        ("SELECTED", "deployed lanes", profile_audit["selected_count"], ""),
    ]
    width_name = max(len(r[1]) for r in rows)
    for code, name, count, note in rows:
        print(f"  {code:<10} {name:<{width_name}}  count={count:>4}   {note}")

    inv_sum = (
        gate_counts["gate_1_e2_safety"]["count"]
        + gate_counts["gate_2a_no_trades"]["count"]
        + gate_counts["gate_2b_thin_no_regime"]["count"]
        + gate_counts["gate_3_regime_cold"]["count"]
        + gate_counts["gate_3b_no_regime_negative"]["count"]
        + gate_counts["other_pause"]["count"]
        + gate_counts["gate_4_chordia"]["count"]
        + profile_audit["gate_5_profile_whitelist"]["dropped_total"]
        + profile_audit["gate_6_non_deployable"]
        + profile_audit["selection_loop_dropped"]
        + profile_audit["selected_count"]
    )
    ok = inv_sum == gate_0
    print(f"  Invariant: sum({inv_sum}) == gate_0({gate_0})  {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"Conservation invariant FAILED for profile {pid}: sum={inv_sum} gate_0={gate_0}")

    cross = profile_audit.get("cross_check", {})
    if cross.get("matches_audit_profile"):
        match = cross.get("set_match")
        tag = "MATCH" if match else "MISMATCH"
        print(
            f"  allocation file: {tag} (committed={len(cross['committed_strategy_ids'])}, "
            f"audit={len(cross['audit_strategy_ids'])})"
        )
        if not match:
            missing_in_audit = set(cross["committed_strategy_ids"]) - set(cross["audit_strategy_ids"])
            missing_in_committed = set(cross["audit_strategy_ids"]) - set(cross["committed_strategy_ids"])
            if missing_in_audit:
                print(f"    in_file_not_in_audit: {sorted(missing_in_audit)}")
            if missing_in_committed:
                print(f"    in_audit_not_in_file: {sorted(missing_in_committed)}")


def _first(seq: list[str]) -> str:
    return seq[0] if seq else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only allocator-gate attrition audit.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile ID to audit (e.g., topstep_50k_mnq_auto).",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Audit every active profile in ACCOUNT_PROFILES.",
    )
    parser.add_argument(
        "--date",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="Rebalance date (default: today). Matches rebalance_lanes semantics.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional JSON output path for the full audit record.",
    )
    args = parser.parse_args()

    if not args.profile and not args.all_profiles:
        parser.error("provide --profile <id> or --all-profiles")

    if args.all_profiles:
        profile_ids = [pid for pid, p in ACCOUNT_PROFILES.items() if p.active]
    else:
        if args.profile not in ACCOUNT_PROFILES:
            parser.error(f"unknown profile: {args.profile}")
        profile_ids = [args.profile]

    print(f"Rebalance date: {args.date}")
    gate_0, gate_0_by_instr = gate_0_entry_count()
    print(f"Gate 0 entry (deployable validated_setups): {gate_0}  by_instrument={gate_0_by_instr}")

    print("Computing lane scores (canonical compute_lane_scores)...")
    scores_raw = compute_lane_scores(rebalance_date=args.date)
    if len(scores_raw) != gate_0:
        raise SystemExit(
            f"FATAL: compute_lane_scores returned {len(scores_raw)} rows, expected {gate_0} (deployable shelf count)."
        )

    # Pre-chordia attribution by status_reason fingerprint.
    pre_buckets = classify_pre_chordia(scores_raw)

    # Snapshot pre-chordia, then run chordia gate to compute its drops.
    pre_chordia_snapshot = copy.deepcopy(scores_raw)
    scores_chordia = apply_chordia_gate(scores_raw)
    chordia_dropped = chordia_drops(pre_chordia_snapshot, scores_chordia)

    # Enrich with SR liveness state (does not drop anything — ranking discount only).
    enrich_scores_with_liveness(scores_chordia)

    # ORB size stats — canonical helper, used inside build_allocation.
    orb_stats = compute_orb_size_stats(args.date)

    # Build gate_counts dict (Gates 1-4).
    gate_counts = {
        "gate_1_e2_safety": bucket_summary("gate_1_e2_safety", pre_buckets["gate_1_e2_safety"]),
        "gate_2a_no_trades": bucket_summary("gate_2a_no_trades", pre_buckets["gate_2a_no_trades"]),
        "gate_2b_thin_no_regime": bucket_summary("gate_2b_thin_no_regime", pre_buckets["gate_2b_thin_no_regime"]),
        "gate_3_regime_cold": bucket_summary("gate_3_regime_cold", pre_buckets["gate_3_regime_cold"]),
        "gate_3b_no_regime_negative": bucket_summary(
            "gate_3b_no_regime_negative", pre_buckets["gate_3b_no_regime_negative"]
        ),
        "other_pause": bucket_summary("other_pause", pre_buckets["other_pause"]),
        "gate_4_chordia": bucket_summary("gate_4_chordia", chordia_dropped),
    }

    sr_dist = sr_distribution(scores_chordia)
    print(f"SR liveness distribution (informational, ranking-discount only): {sr_dist}")

    all_profile_audits: list[dict] = []
    for pid in profile_ids:
        prof_audit = audit_profile(pid, scores_chordia, orb_stats, args.date)
        prof_audit["cross_check"] = cross_check_lane_allocation(pid, prof_audit["selected"])
        print_profile_table(prof_audit, gate_0, gate_counts)
        all_profile_audits.append(prof_audit)

    if args.json:
        record = {
            "rebalance_date": args.date.isoformat(),
            "gate_0_entry": gate_0,
            "gate_0_by_instrument": gate_0_by_instr,
            "gate_counts": gate_counts,
            "sr_distribution": sr_dist,
            "profiles": all_profile_audits,
        }
        Path(args.json).write_text(json.dumps(record, indent=2, default=str))
        print(f"\nWrote JSON record: {args.json}")


if __name__ == "__main__":
    main()
