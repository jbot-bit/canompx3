"""One-shot displaced-rotation analyzer for PASS_CHORDIA candidates.

Cross-references `docs/runtime/chordia_audit_log.yaml` (verdict=PASS_CHORDIA)
with the canonical allocation file (displaced[]), then re-runs the canonical
trade-level correlation gate for each candidate. Reports which displaced
PASS_CHORDIA edges sit close to the 0.70 rho gate and are worth operator
review for rotation.

Read-only. No mutations to the allocation file, chordia_audit_log.yaml, or
gold.db. Stdout only.

Canonical dependencies (NEVER re-encode):
  - trading_app.prop_profiles.resolve_allocation_json (Stage 1b/1c resolver)
  - trading_app.lane_correlation.check_candidate_correlation
  - trading_app.lane_correlation.RHO_REJECT_THRESHOLD
  - trading_app.eligibility.builder.parse_strategy_id
  - pipeline.paths.GOLD_DB_PATH

Usage:
  python scripts/research/displaced_rotation_analyzer.py --profile topstep_50k_mnq_auto
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import yaml

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.eligibility.builder import parse_strategy_id
from trading_app.lane_correlation import (
    RHO_REJECT_THRESHOLD,
    check_candidate_correlation,
)
from trading_app.prop_profiles import resolve_allocation_json
from trading_app.strategy_fitness import _load_strategy_outcomes

REPO_ROOT = Path(__file__).resolve().parents[2]
CHORDIA_LOG = REPO_ROOT / "docs" / "runtime" / "chordia_audit_log.yaml"

NEAR_GATE_BAND = 0.05


def _load_pass_chordia_ids() -> set[str]:
    with CHORDIA_LOG.open() as f:
        log = yaml.safe_load(f) or {}
    audits = log.get("audits", [])
    return {e["strategy_id"] for e in audits if e.get("verdict") == "PASS_CHORDIA"}


def _load_allocation(profile_id: str) -> dict:
    read = resolve_allocation_json(profile_id)
    if read.data is None:
        raise FileNotFoundError(
            f"No allocation data found for profile {profile_id!r} (resolver source={read.source}, path={read.path})"
        )
    return read.data


def _strategy_id_to_lane(strategy_id: str) -> dict:
    """Build the lane dict shape expected by lane_correlation."""
    parsed = parse_strategy_id(strategy_id)
    return {
        "strategy_id": strategy_id,
        "instrument": parsed["instrument"],
        "orb_label": parsed["orb_label"],
        "orb_minutes": parsed["orb_minutes"],
        "entry_model": parsed["entry_model"],
        "rr_target": parsed["rr_target"],
        "confirm_bars": parsed["confirm_bars"],
        "filter_type": parsed["filter_type"],
    }


def _classify_rotation(
    worst_rho: float,
    gate_pass: bool,
    cached_rho: float | None,
) -> tuple[str, str]:
    distance_from_gate = RHO_REJECT_THRESHOLD - worst_rho
    if gate_pass:
        return ("ROTATE_CANDIDATE", f"gate-pass; rho={worst_rho:.3f}<{RHO_REJECT_THRESHOLD}")
    if abs(distance_from_gate) <= NEAR_GATE_BAND:
        return (
            "NEAR_GATE_REVIEW",
            f"rho={worst_rho:.3f} within {NEAR_GATE_BAND} of gate {RHO_REJECT_THRESHOLD}",
        )
    cached_note = f" (cached={cached_rho:.3f})" if cached_rho is not None else ""
    return ("STAY", f"rho={worst_rho:.3f} clearly above gate{cached_note}")


def analyze(profile_id: str) -> int:
    pass_chordia_ids = _load_pass_chordia_ids()
    if not pass_chordia_ids:
        print("No PASS_CHORDIA entries in chordia_audit_log.yaml; nothing to analyze.")
        return 0

    alloc = _load_allocation(profile_id)
    profile = alloc["profile_id"]
    rebalance_date = alloc.get("rebalance_date", "?")
    displaced = alloc.get("displaced", [])

    candidates = []
    for entry in displaced:
        sid = entry.get("strategy_id")
        if sid in pass_chordia_ids:
            candidates.append(entry)

    print(f"Profile: {profile}  rebalance_date: {rebalance_date}")
    print(f"PASS_CHORDIA universe: {len(pass_chordia_ids)} entries")
    print(f"Displaced ∩ PASS_CHORDIA: {len(candidates)} candidates")
    print(f"Canonical gate: rho > {RHO_REJECT_THRESHOLD} rejects")
    print(f"Near-gate band: ±{NEAR_GATE_BAND}")
    print()
    print("=" * 100)

    results = []
    for entry in candidates:
        sid = entry["strategy_id"]
        cached_rho = entry.get("rho_with_incumbent") or entry.get("rho")
        cached_incumbent = entry.get("displaced_by") or entry.get("incumbent")

        try:
            lane = _strategy_id_to_lane(sid)
        except ValueError as exc:
            print(f"\n{sid}\n  PARSE_ERROR: {exc}")
            continue

        try:
            report = check_candidate_correlation(lane, profile_id=profile)
        except Exception as exc:  # noqa: BLE001 — surfaced, not swallowed
            print(f"\n{sid}\n  CORRELATION_ERROR: {type(exc).__name__}: {exc}")
            continue

        verdict, reason = _classify_rotation(
            worst_rho=report.worst_rho,
            gate_pass=report.gate_pass,
            cached_rho=cached_rho,
        )

        worst_pair = max(report.pairs, key=lambda p: p.pearson_rho) if report.pairs else None

        print()
        print(f"[{verdict}]  {sid}")
        print(
            f"  cached_rho={cached_rho if cached_rho is not None else '-'}  cached_incumbent={cached_incumbent or '-'}"
        )
        if worst_pair is not None:
            print(
                f"  fresh_rho={worst_pair.pearson_rho:.3f}  vs={worst_pair.deployed_id}  "
                f"shared_days={worst_pair.shared_days}  "
                f"cand_days={worst_pair.candidate_days}  dep_days={worst_pair.deployed_days}  "
                f"subset_cov={worst_pair.subset_coverage:.1%}"
            )
        else:
            print(f"  fresh_rho={report.worst_rho:.3f}  (no pair data)")
        print(f"  gate_pass={report.gate_pass}  reason={reason}")
        if report.reject_reasons:
            for r in report.reject_reasons:
                print(f"    reject: {r}")

        results.append(
            {
                "strategy_id": sid,
                "verdict": verdict,
                "cached_rho": cached_rho,
                "fresh_worst_rho": report.worst_rho,
                "gate_pass": report.gate_pass,
                "worst_pair_id": worst_pair.deployed_id if worst_pair else None,
                "shared_days": worst_pair.shared_days if worst_pair else 0,
            }
        )

    print()
    print("=" * 100)
    print("SUMMARY")
    rotate = [r for r in results if r["verdict"] == "ROTATE_CANDIDATE"]
    near = [r for r in results if r["verdict"] == "NEAR_GATE_REVIEW"]
    stay = [r for r in results if r["verdict"] == "STAY"]
    print(f"  ROTATE_CANDIDATE (gate-pass on fresh rho): {len(rotate)}")
    for r in rotate:
        print(f"    {r['strategy_id']}  fresh_rho={r['fresh_worst_rho']:.3f}")
    print(f"  NEAR_GATE_REVIEW (within ±{NEAR_GATE_BAND}): {len(near)}")
    for r in near:
        print(f"    {r['strategy_id']}  fresh_rho={r['fresh_worst_rho']:.3f}")
    print(f"  STAY (clearly above gate): {len(stay)}")

    if rotate or near:
        print()
        print(
            "Next step: review each ROTATE_CANDIDATE / NEAR_GATE_REVIEW row, then run "
            "`python scripts/tools/rebalance_lanes.py --date <today> --profile " + profile + "` "
            "to materialize the rotation if confirmed. This tool does NOT mutate the allocation file."
        )

    return 0


def _day_pnl_map(outcomes: list[dict]) -> dict:
    by_day: dict = defaultdict(float)
    for o in outcomes:
        if o.get("pnl_r") is not None:
            by_day[o["trading_day"]] += float(o["pnl_r"])
    return dict(by_day)


def _decompose_overlap(
    incumbent_outcomes: list[dict],
    candidate_outcomes: list[dict],
) -> dict:
    inc_pnl = _day_pnl_map(incumbent_outcomes)
    cand_pnl = _day_pnl_map(candidate_outcomes)
    inc_days = set(inc_pnl)
    cand_days = set(cand_pnl)
    overlap = inc_days & cand_days

    both_w = both_l = inc_w_cand_l = inc_l_cand_w = zero_either = 0
    diff_pnl = 0
    for d in overlap:
        iv = inc_pnl[d]
        cv = cand_pnl[d]
        if iv > 0 and cv > 0:
            both_w += 1
        elif iv < 0 and cv < 0:
            both_l += 1
        elif iv > 0 and cv < 0:
            inc_w_cand_l += 1
        elif iv < 0 and cv > 0:
            inc_l_cand_w += 1
        else:
            zero_either += 1
        if abs(iv - cv) > 0.01:
            diff_pnl += 1

    return {
        "inc_days": len(inc_days),
        "cand_days": len(cand_days),
        "overlap_days": len(overlap),
        "inc_only_days": len(inc_days - cand_days),
        "cand_only_days": len(cand_days - inc_days),
        "both_win": both_w,
        "both_loss": both_l,
        "inc_win_cand_loss": inc_w_cand_l,
        "inc_loss_cand_win": inc_l_cand_w,
        "zero_either": zero_either,
        "different_pnl_count": diff_pnl,
        "different_pnl_pct": (diff_pnl / len(overlap)) if overlap else 0.0,
    }


def _find_displaced_entry(strategy_id: str, alloc: dict) -> dict | None:
    for entry in alloc.get("displaced", []):
        if entry.get("strategy_id") == strategy_id:
            return entry
    return None


def _resolve_incumbent(displaced_entry: dict) -> str:
    return displaced_entry.get("displaced_by") or displaced_entry.get("incumbent") or ""


def _write_decision_report(
    *,
    strategy_id: str,
    incumbent_id: str,
    rebalance_date: str,
    profile_id: str,
    fresh_rho: float,
    cached_rho: float | None,
    gate_pass: bool,
    decomp: dict,
    chordia_verdict: str,
    chordia_date: str | None,
    report_dir: Path,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"rotation_decision_{strategy_id}_{rebalance_date}.md"

    disagree_pct = decomp["different_pnl_pct"] * 100.0
    win_swap = decomp["inc_win_cand_loss"] + decomp["inc_loss_cand_win"]

    if gate_pass:
        recommendation = "ROTATE — fresh ρ passes the canonical gate. Run `rebalance_lanes.py` to materialize."
    elif disagree_pct >= 30.0:
        recommendation = (
            "PARALLEL_DEPLOY_CANDIDATE — gate-fail on ρ but outcomes diverge on "
            f"{disagree_pct:.0f}% of overlap days. Operator may add as a second lane in the same session "
            "via manual profile edit (bypasses auto-allocator). DD-budget impact must be hand-computed."
        )
    else:
        recommendation = "STAY — fresh ρ above gate and outcomes track incumbent on most overlap days. No rotation."

    body = f"""# Rotation Decision Report

**Candidate:** `{strategy_id}`
**Incumbent:** `{incumbent_id}`
**Profile:** `{profile_id}`
**Rebalance date:** {rebalance_date}
**Generated:** {datetime.now(UTC).isoformat(timespec="seconds")}
**Tool:** `scripts/research/displaced_rotation_analyzer.py --diagnostic`

---

## Canonical correlation gate

| Metric | Value |
|---|---|
| Fresh trade-level Pearson ρ | {fresh_rho:.3f} |
| Cached ρ in allocation file | {cached_rho if cached_rho is not None else "n/a"} |
| Canonical gate threshold | {RHO_REJECT_THRESHOLD} |
| Gate pass | {gate_pass} |

Source: `trading_app.lane_correlation.check_candidate_correlation` (delegated, not re-encoded).

## Heavyweight Chordia provenance

| Field | Value |
|---|---|
| Chordia verdict | {chordia_verdict} |
| Audit date | {chordia_date or "n/a"} |

Source: `docs/runtime/chordia_audit_log.yaml`.

## Day-by-day outcome decomposition

| Bucket | Count |
|---|---|
| Incumbent trading days (IS+OOS) | {decomp["inc_days"]} |
| Candidate trading days (IS+OOS) | {decomp["cand_days"]} |
| Overlap (both fire same day) | {decomp["overlap_days"]} |
| Incumbent-only days | {decomp["inc_only_days"]} |
| Candidate-only days | {decomp["cand_only_days"]} |
| Both win | {decomp["both_win"]} |
| Both loss | {decomp["both_loss"]} |
| Incumbent W / Candidate L | {decomp["inc_win_cand_loss"]} |
| Incumbent L / Candidate W | {decomp["inc_loss_cand_win"]} |
| Zero P&L either side | {decomp["zero_either"]} |
| Days with different pnl_r (>0.01) | {decomp["different_pnl_count"]} ({disagree_pct:.1f}% of overlap) |
| Outcome swaps (W↔L) | {win_swap} ({100 * win_swap / max(decomp["overlap_days"], 1):.1f}% of overlap) |

## Interpretation

The lane correlation engine measures trade-level Pearson ρ on daily P&L summed within each session. A high ρ between two same-session lanes can mean EITHER (a) they are mechanically the same fill series (subset duplicates), OR (b) they trade the same session under correlated regimes but with mechanically-separable fills (e.g., different ORB minutes → different entry/exit prices on the same day).

**The decomposition table above distinguishes these two cases.**

- If `different pnl_r %` is low (< 10%) → mechanically redundant → STAY.
- If `different pnl_r %` is high (>= 30%) → real edge co-movement on separable fills → PARALLEL_DEPLOY_CANDIDATE worth manual evaluation.
- The canonical auto-allocator gate at ρ > {RHO_REJECT_THRESHOLD} stays in place regardless — this report is for operator manual override only.

## Operator options

### Option A — Stay (no action)
- Pros: Auto-allocator decision unchanged. No DD-budget reshuffle. Auditable.
- Cons: Forfeits the {disagree_pct:.0f}% of overlap days where the candidate trades a separable signal.

### Option B — Rotate (replace incumbent with candidate)
- Pros: Auto-allocator picks up the rotation on next rebalance via `rebalance_lanes.py`.
- Cons: Lose incumbent's track record. Only viable if fresh ρ passes the gate; here it does NOT, so this requires manual override doctrine which we do not have.
- Status: **NOT AVAILABLE** under current canonical rules (gate did not pass).

### Option C — Parallel-deploy (add candidate as a SECOND lane in the same session)
- Pros: Captures the {win_swap}-day outcome swap divergence. Combined lane exposure may improve session-level Sharpe.
- Cons: Doubles per-session position count. DD-budget must be hand-checked. Auto-allocator will resist on next rebalance — requires either a profile edit that grandfathers both lanes, OR a gate exception entry.
- Capital-class change: requires `topstep_50k_mnq_auto` profile review + adversarial-audit gate dispatch.

## Recommendation

{recommendation}

## Doctrine references

- `trading_app/lane_correlation.py:24` — canonical `RHO_REJECT_THRESHOLD = 0.70`.
- `trading_app/lane_allocator.py:938` — greedy-correlation selection gate.
- `feedback_high_r_inventory_comes_from_chordia_not_raw_expr.md` — the "golden nug" doctrine (this candidate IS one).
- `feedback_max_profit_grow_chordia_inventory_not_force_slots.md` — parallel-deploy must satisfy Chordia inventory rules.
- `.claude/rules/institutional-rigor.md` § 3 — do not patch the gate; surface the diagnostic for operator decision.

## Files NOT modified by this report

- The canonical lane allocation file (resolved via `trading_app.prop_profiles.resolve_allocation_json`)
- `docs/runtime/chordia_audit_log.yaml`
- `trading_app/prop_profiles.py`
- `gold.db`

Operator must take explicit action via `rebalance_lanes.py` or a profile edit to materialize any decision.
"""
    out_path.write_text(body, encoding="utf-8")
    return out_path


def diagnose(strategy_id: str, profile_id: str) -> int:
    pass_chordia_ids = _load_pass_chordia_ids()
    alloc = _load_allocation(profile_id=profile_id)
    profile = alloc["profile_id"]
    rebalance_date = alloc.get("rebalance_date", "?")

    if strategy_id not in pass_chordia_ids:
        print(
            f"ERROR: {strategy_id} is not a PASS_CHORDIA entry in chordia_audit_log.yaml. "
            f"Diagnostic mode refuses to evaluate non-heavyweight-audited candidates per "
            f"feedback_high_r_inventory_comes_from_chordia_not_raw_expr.md."
        )
        return 2

    displaced_entry = _find_displaced_entry(strategy_id, alloc)
    if displaced_entry is None:
        print(f"ERROR: {strategy_id} is not in allocation displaced[] for profile {profile}.")
        return 2

    incumbent_id = _resolve_incumbent(displaced_entry)
    if not incumbent_id:
        print(f"ERROR: cannot resolve incumbent for {strategy_id}.")
        return 2

    cached_rho = displaced_entry.get("rho_with_incumbent") or displaced_entry.get("rho")

    with CHORDIA_LOG.open() as f:
        log = yaml.safe_load(f) or {}
    chordia_entry = next(
        (e for e in log.get("audits", []) if e.get("strategy_id") == strategy_id),
        None,
    )
    chordia_verdict = (chordia_entry or {}).get("verdict", "NOT_AUDITED")
    chordia_date = (chordia_entry or {}).get("audit_date")

    try:
        cand_lane = _strategy_id_to_lane(strategy_id)
        inc_lane = _strategy_id_to_lane(incumbent_id)
    except ValueError as exc:
        print(f"PARSE_ERROR: {exc}")
        return 2

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        report = check_candidate_correlation(cand_lane, profile_id=profile, con=con)
        cand_outcomes = _load_strategy_outcomes(
            con,
            instrument=cand_lane["instrument"],
            orb_label=cand_lane["orb_label"],
            orb_minutes=cand_lane["orb_minutes"],
            entry_model=cand_lane["entry_model"],
            rr_target=cand_lane["rr_target"],
            confirm_bars=cand_lane["confirm_bars"],
            filter_type=cand_lane["filter_type"],
        )
        inc_outcomes = _load_strategy_outcomes(
            con,
            instrument=inc_lane["instrument"],
            orb_label=inc_lane["orb_label"],
            orb_minutes=inc_lane["orb_minutes"],
            entry_model=inc_lane["entry_model"],
            rr_target=inc_lane["rr_target"],
            confirm_bars=inc_lane["confirm_bars"],
            filter_type=inc_lane["filter_type"],
        )
    finally:
        con.close()

    decomp = _decompose_overlap(inc_outcomes, cand_outcomes)
    fresh_rho = report.worst_rho

    report_path = _write_decision_report(
        strategy_id=strategy_id,
        incumbent_id=incumbent_id,
        rebalance_date=rebalance_date,
        profile_id=profile,
        fresh_rho=fresh_rho,
        cached_rho=cached_rho,
        gate_pass=report.gate_pass,
        decomp=decomp,
        chordia_verdict=chordia_verdict,
        chordia_date=chordia_date,
        report_dir=REPO_ROOT / "docs" / "runtime",
    )

    print(f"Candidate:  {strategy_id}")
    print(f"Incumbent:  {incumbent_id}")
    print(f"Profile:    {profile}  ({rebalance_date})")
    print(f"Chordia:    {chordia_verdict} on {chordia_date}")
    print(f"Fresh ρ:    {fresh_rho:.3f}  (gate {RHO_REJECT_THRESHOLD}; pass={report.gate_pass})")
    print(f"Cached ρ:   {cached_rho if cached_rho is not None else '-'}")
    print()
    print("Day-by-day overlap decomposition:")
    for k in (
        "inc_days",
        "cand_days",
        "overlap_days",
        "inc_only_days",
        "cand_only_days",
        "both_win",
        "both_loss",
        "inc_win_cand_loss",
        "inc_loss_cand_win",
        "zero_either",
        "different_pnl_count",
    ):
        print(f"  {k}: {decomp[k]}")
    print(f"  different_pnl_pct: {decomp['different_pnl_pct'] * 100:.1f}%")
    print()
    print(f"Decision report written: {report_path.relative_to(REPO_ROOT)}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--profile",
        required=True,
        help="Allocator profile to inspect (e.g. topstep_50k_mnq_auto). Required since Stage 1c routes path resolution through trading_app.prop_profiles.resolve_allocation_json.",
    )
    p.add_argument(
        "--diagnostic",
        default=None,
        metavar="STRATEGY_ID",
        help="Deep-dive a single displaced candidate: print overlap decomposition + write a decision report.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.diagnostic:
        return diagnose(strategy_id=args.diagnostic, profile_id=args.profile)
    return analyze(profile_id=args.profile)


if __name__ == "__main__":
    sys.exit(main())
