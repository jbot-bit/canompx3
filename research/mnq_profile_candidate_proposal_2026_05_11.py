#!/usr/bin/env python3
"""Classify MNQ controlled-pilot candidates into a profile-safe proposal.

Read-only stage for `topstep_50k_mnq_auto`.

Inputs:
- docs/audit/results/2026-05-11-mnq-all-active-deployability.json
- canonical validated_setups metadata in gold.db
- current profile allocation / prop profile constraints

Outputs:
- docs/audit/results/2026-05-11-mnq-profile-candidate-proposal.md
- docs/audit/results/2026-05-11-mnq-profile-candidate-proposal.csv

No live allocation, broker state, schema, or deployment DB state is mutated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

from pipeline.cost_model import COST_SPECS
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from research.comprehensive_deployed_lane_scan import compute_deployed_filter, load_lane
from research.portfolio_additivity_engine import (
    LaneSpec,
    PortfolioSnapshot,
    compute_snapshot,
    fmt_num,
    load_live_lane_specs,
)
from trading_app.allocation_promotion import PromotionCandidate, apply_promotions
from trading_app.chordia import chordia_verdict_allows_deploy, load_chordia_audit_log
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.lane_correlation import RHO_REJECT_THRESHOLD, SUBSET_REJECT_THRESHOLD, _pearson
from trading_app.prop_portfolio import profile_static_gate_reason
from trading_app.prop_profiles import ACCOUNT_PROFILES, ACCOUNT_TIERS, resolve_allocation_json
from trading_app.strategy_fitness import _load_strategy_outcomes

PROFILE_ID = "topstep_50k_mnq_auto"
REPORT_DATE = date(2026, 5, 11)
INPUT_JSON = Path("docs/audit/results/2026-05-11-mnq-all-active-deployability.json")
RESULT_MD = Path("docs/audit/results/2026-05-11-mnq-profile-candidate-proposal.md")
RESULT_CSV = Path("docs/audit/results/2026-05-11-mnq-profile-candidate-proposal.csv")
RESULT_PROMOTION_PATCH_JSON = Path("docs/audit/results/2026-05-11-mnq-profile-allocation-promotion-patch.json")

CONTROLLED_VERDICT = "CONTROLLED_LIVE_PILOT_CANDIDATE"
SUPPORTED_STOP_MULTIPLIER = 1.0
RUNTIME_ACTIVE_STATUSES = {"DEPLOY", "PROVISIONAL"}


@dataclass(frozen=True)
class CandidateRecord:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    entry_model: str
    rr_target: float
    confirm_bars: int
    filter_type: str
    stop_multiplier: float
    family_hash: str
    expectancy_r: float
    sample_size: int
    deployability_verdict: str
    c8_oos_status: str | None
    replay_ok: bool
    current_fdr_pass: bool | None
    family_status: str | None
    hard_issue_ids: tuple[str, ...]
    slippage_status: str | None
    chordia_verdict: str
    chordia_audit_age_days: int | None
    runtime_control_evaluated: bool
    runtime_sr_status: str | None
    runtime_sr_review_outcome: str | None
    profile_allowed: bool
    win_rate: float | None = None
    trades_per_year: float | None = None


@dataclass(frozen=True)
class PortfolioGate:
    add_delta_annual_r: float | None
    add_delta_sharpe: float | None
    replace_delta_annual_r: float | None
    replace_delta_sharpe: float | None
    corr_gate_pass: bool | None
    corr_reject_reasons: tuple[str, ...]
    replacement_target: str | None
    replacement_target_status: str | None
    account_risk_ok: bool | None
    account_risk_detail: str


@dataclass(frozen=True)
class CandidateDecision:
    strategy_id: str
    decision: str
    primary_reason: str
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class CandidateResult:
    candidate: CandidateRecord
    decision: CandidateDecision
    gate: PortfolioGate | None


def _dedupe_key(candidate: CandidateRecord) -> tuple[str, str, str, int, float]:
    return (
        candidate.family_hash,
        candidate.orb_label,
        candidate.filter_type,
        candidate.orb_minutes,
        candidate.stop_multiplier,
    )


def _chordia_rank(verdict: str) -> int:
    if verdict == "PASS_CHORDIA":
        return 3
    if verdict == "PASS_PROTOCOL_A":
        return 2
    if verdict == "PARK":
        return 1
    return 0


def choose_dedupe_heads(candidates: list[CandidateRecord]) -> dict[str, bool]:
    """Select one representative per family/session/filter group.

    Chordia-cleared rows outrank higher-ExpR unaudited siblings because this
    stage is a profile-construction gate, not fresh signal selection.
    """
    groups: dict[tuple[str, str, str, int, float], list[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        groups[_dedupe_key(candidate)].append(candidate)

    out: dict[str, bool] = {}
    for rows in groups.values():
        head = max(
            rows,
            key=lambda row: (
                _chordia_rank(row.chordia_verdict),
                row.expectancy_r,
                row.sample_size,
                -row.rr_target,
                row.strategy_id,
            ),
        )
        for row in rows:
            out[row.strategy_id] = row.strategy_id == head.strategy_id
    return out


def _positive(value: float | None) -> bool:
    return value is not None and not math.isnan(value) and value > 0.0


def classify_candidate(
    candidate: CandidateRecord,
    *,
    dedupe_head: bool,
    gate: PortfolioGate | None,
) -> CandidateDecision:
    blockers: list[str] = []

    if not dedupe_head:
        return CandidateDecision(
            candidate.strategy_id,
            "KILL",
            "Dominated duplicate within same family/session/filter group.",
            ("dedupe_dominated_variant",),
        )

    if candidate.deployability_verdict != CONTROLLED_VERDICT:
        blockers.append(f"deployability_verdict={candidate.deployability_verdict}")
    if not candidate.replay_ok:
        blockers.append("replay_not_ok")
    if candidate.current_fdr_pass is not True:
        blockers.append("current_fdr_not_passed")
    if candidate.c8_oos_status != "PASSED":
        blockers.append(f"c8_oos_status={candidate.c8_oos_status}")
    if candidate.hard_issue_ids:
        blockers.extend(candidate.hard_issue_ids)
    if blockers:
        return CandidateDecision(candidate.strategy_id, "KILL", "Hard deployability evidence failed.", tuple(blockers))

    if candidate.entry_model != "E2" or candidate.confirm_bars != 1:
        return CandidateDecision(
            candidate.strategy_id,
            "ARCHITECTURE_REQUIRED",
            "Current proposal runner only supports MNQ E2 CB1 stop-market lanes.",
            ("unsupported_entry_model",),
        )
    if candidate.stop_multiplier != SUPPORTED_STOP_MULTIPLIER:
        return CandidateDecision(
            candidate.strategy_id,
            "ARCHITECTURE_REQUIRED",
            "Non-default stop-multiplier rows need a separate physical trade-stream replay.",
            ("non_default_stop_multiplier",),
        )
    if not candidate.profile_allowed:
        return CandidateDecision(
            candidate.strategy_id,
            "ARCHITECTURE_REQUIRED",
            "Candidate is outside the current profile static session/instrument route.",
            ("profile_static_gate",),
        )

    if not chordia_verdict_allows_deploy(candidate.chordia_verdict):
        return CandidateDecision(
            candidate.strategy_id,
            "PARK",
            f"Allocator Chordia gate does not permit deploy ({candidate.chordia_verdict}).",
            (f"chordia_{candidate.chordia_verdict.lower()}",),
        )

    if not candidate.runtime_control_evaluated:
        return CandidateDecision(
            candidate.strategy_id,
            "PARK",
            "Exact lane SR/runtime control was not evaluated; paper/sandbox monitor state is required first.",
            ("runtime_sr_not_evaluated",),
        )

    if gate is None:
        return CandidateDecision(
            candidate.strategy_id,
            "PARK",
            "Portfolio add/replace gate was not evaluated.",
            ("portfolio_gate_missing",),
        )
    if gate.account_risk_ok is not True:
        return CandidateDecision(
            candidate.strategy_id,
            "PARK",
            "Account-risk budget does not clear.",
            ("account_risk", gate.account_risk_detail),
        )

    if gate.replacement_target == candidate.strategy_id and gate.replacement_target_status in RUNTIME_ACTIVE_STATUSES:
        return CandidateDecision(
            candidate.strategy_id,
            "PARK",
            "Already selected in the current active profile; no proposal change.",
            ("already_selected",),
        )

    active_replacement = (
        gate.replacement_target is not None and gate.replacement_target_status in RUNTIME_ACTIVE_STATUSES
    )
    paused_replacement = (
        gate.replacement_target is not None and gate.replacement_target_status not in RUNTIME_ACTIVE_STATUSES
    )

    if active_replacement:
        if _positive(gate.replace_delta_annual_r) and _positive(gate.replace_delta_sharpe):
            return CandidateDecision(
                candidate.strategy_id,
                "PASS_REPLACE",
                "Same-session active replacement improves common-window IS annualized R and Sharpe.",
                (),
            )
        return CandidateDecision(
            candidate.strategy_id,
            "PARK",
            "Same-session active replacement does not improve both common-window IS annualized R and Sharpe.",
            ("replacement_math_not_positive",),
        )

    if paused_replacement:
        if _positive(gate.add_delta_annual_r) and _positive(gate.add_delta_sharpe) and gate.corr_gate_pass:
            return CandidateDecision(
                candidate.strategy_id,
                "PASS_REPLACE",
                "Candidate can replace a paused same-session profile lane; additive math versus active book clears.",
                (),
            )
        return CandidateDecision(
            candidate.strategy_id,
            "PARK",
            "Paused-lane replacement does not clear additive math and correlation gates.",
            ("paused_replacement_gate",),
        )

    if gate.corr_gate_pass is not True:
        return CandidateDecision(
            candidate.strategy_id,
            "PARK",
            "Additive route is blocked by the profile correlation gate.",
            ("correlation_gate", *gate.corr_reject_reasons),
        )
    if _positive(gate.add_delta_annual_r) and _positive(gate.add_delta_sharpe):
        return CandidateDecision(
            candidate.strategy_id,
            "PASS_ADD",
            "Additive route improves common-window IS annualized R and Sharpe and passes correlation/account gates.",
            (),
        )
    return CandidateDecision(
        candidate.strategy_id,
        "PARK",
        "Additive route does not improve both common-window IS annualized R and Sharpe.",
        ("additive_math_not_positive",),
    )


def _load_allocation() -> dict[str, Any]:
    resolved = resolve_allocation_json(PROFILE_ID)
    if resolved.data is None:
        raise RuntimeError(f"profile allocation file missing for {PROFILE_ID!r}")
    return resolved.data


def _allocation_status_by_strategy(allocation: dict[str, Any]) -> dict[str, str]:
    rows = list(allocation.get("lanes") or []) + list(allocation.get("paused") or [])
    return {str(row["strategy_id"]): str(row.get("status", "PAUSE")) for row in rows if row.get("strategy_id")}


def _allocation_same_session(allocation: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in list(allocation.get("lanes") or []) + list(allocation.get("paused") or []):
        if row.get("instrument") == "MNQ" and row.get("orb_label"):
            key = (str(row["instrument"]), str(row["orb_label"]))
            if key not in out or row.get("status") == "DEPLOY":
                out[key] = row
    return out


def _candidate_metadata(ids: list[str]) -> dict[str, dict[str, Any]]:
    placeholders = ", ".join("?" for _ in ids)
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        rows = con.execute(
            f"""
            SELECT strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   rr_target, confirm_bars, filter_type, stop_multiplier,
                   family_hash, sample_size, expectancy_r, win_rate, trades_per_year
            FROM validated_setups
            WHERE strategy_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        cols = [desc[0] for desc in con.description]
    out = {str(row[0]): dict(zip(cols, row, strict=False)) for row in rows}
    missing = sorted(set(ids) - set(out))
    if missing:
        raise RuntimeError(f"Missing validated_setups rows for {missing[:10]}")
    return out


def load_candidates() -> list[CandidateRecord]:
    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    raw = [row for row in payload["strategies"] if row.get("verdict") == CONTROLLED_VERDICT]
    ids = [str(row["strategy_id"]) for row in raw]
    meta = _candidate_metadata(ids)
    audit_log = load_chordia_audit_log()
    profile = ACCOUNT_PROFILES[PROFILE_ID]

    out: list[CandidateRecord] = []
    for row in raw:
        sid = str(row["strategy_id"])
        m = meta[sid]
        issue_ids = tuple(str(issue["id"]) for issue in row.get("issues", []) if issue.get("severity") == "hard")
        runtime = row.get("runtime_control") or {}
        static_reason = profile_static_gate_reason(profile, str(m["instrument"]), str(m["orb_label"]))
        out.append(
            CandidateRecord(
                strategy_id=sid,
                instrument=str(m["instrument"]),
                orb_label=str(m["orb_label"]),
                orb_minutes=int(m["orb_minutes"]),
                entry_model=str(m["entry_model"]),
                rr_target=float(m["rr_target"]),
                confirm_bars=int(m["confirm_bars"]),
                filter_type=str(m["filter_type"]),
                stop_multiplier=float(m["stop_multiplier"] or 1.0),
                family_hash=str(m["family_hash"]),
                expectancy_r=float(m["expectancy_r"] or 0.0),
                sample_size=int(m["sample_size"] or 0),
                deployability_verdict=str(row["verdict"]),
                c8_oos_status=(row.get("c8_oos") or {}).get("c8_oos_status"),
                replay_ok=bool((row.get("replay") or {}).get("ok")),
                current_fdr_pass=(row.get("current_k_fdr") or {}).get("current_pass"),
                family_status=(row.get("metrics") or {}).get("family_status"),
                hard_issue_ids=issue_ids,
                slippage_status=(row.get("metrics") or {}).get("slippage_validation_status"),
                chordia_verdict=audit_log.verdict(sid) or "MISSING",
                chordia_audit_age_days=audit_log.audit_age_days(sid, REPORT_DATE),
                runtime_control_evaluated=bool(runtime.get("evaluated")),
                runtime_sr_status=runtime.get("sr_status"),
                runtime_sr_review_outcome=runtime.get("sr_review_outcome"),
                profile_allowed=static_reason is None,
                win_rate=float(m["win_rate"]) if m.get("win_rate") is not None else None,
                trades_per_year=float(m["trades_per_year"]) if m.get("trades_per_year") is not None else None,
            )
        )
    return out


def _spec(candidate: CandidateRecord) -> LaneSpec:
    return LaneSpec(
        strategy_id=candidate.strategy_id,
        instrument=candidate.instrument,
        orb_label=candidate.orb_label,
        entry_model=candidate.entry_model,
        rr_target=candidate.rr_target,
        confirm_bars=candidate.confirm_bars,
        filter_type=candidate.filter_type,
        orb_minutes=candidate.orb_minutes,
    )


def _load_trades(con: duckdb.DuckDBPyConnection, spec: LaneSpec) -> list[dict[str, Any]]:
    outcomes = _load_strategy_outcomes(
        con,
        instrument=spec.instrument,
        orb_label=spec.orb_label,
        orb_minutes=spec.orb_minutes,
        entry_model=spec.entry_model,
        rr_target=spec.rr_target,
        confirm_bars=spec.confirm_bars,
        filter_type=spec.filter_type,
    )
    trades: list[dict[str, Any]] = []
    for row in outcomes:
        pnl_r = row.get("pnl_r")
        if pnl_r is None:
            continue
        td = row["trading_day"]
        trades.append(
            {
                "trading_day": td.date() if hasattr(td, "date") else td,
                "pnl_r": float(pnl_r),
                "strategy_id": spec.strategy_id,
            }
        )
    if not trades:
        raise RuntimeError(f"zero trades loaded for {spec.strategy_id}")
    return trades


def _common_window(*trade_sets: list[dict[str, Any]]) -> tuple[date, date]:
    starts = [min(t["trading_day"] for t in trades) for trades in trade_sets if trades]
    ends = [max(t["trading_day"] for t in trades) for trades in trade_sets if trades]
    return max(starts), min(ends)


def _daily_pnl(trades: list[dict[str, Any]], start_date: date, end_date: date) -> dict[date, float]:
    out: dict[date, float] = {}
    for trade in trades:
        day = trade["trading_day"]
        if start_date <= day <= end_date:
            out[day] = out.get(day, 0.0) + float(trade["pnl_r"])
    return out


def _pair_correlation_gate(
    candidate: LaneSpec,
    live_specs: list[LaneSpec],
    trades_all: dict[str, list[dict[str, Any]]],
    *,
    start_date: date,
    end_date: date,
) -> tuple[bool, tuple[str, ...]]:
    candidate_pnl = _daily_pnl(trades_all[candidate.strategy_id], start_date, end_date)
    reject_reasons: list[str] = []
    for live in live_specs:
        live_pnl = _daily_pnl(trades_all[live.strategy_id], start_date, end_date)
        shared = sorted(set(candidate_pnl) & set(live_pnl))
        smaller = min(len(candidate_pnl), len(live_pnl))
        subset = len(shared) / smaller if smaller else 0.0
        rho = 0.0
        if len(shared) >= 5:
            rho = _pearson([candidate_pnl[d] for d in shared], [live_pnl[d] for d in shared])
        same_session = candidate.orb_label == live.orb_label and candidate.instrument == live.instrument
        reasons: list[str] = []
        if rho > RHO_REJECT_THRESHOLD:
            reasons.append(f"rho={rho:.3f}>{RHO_REJECT_THRESHOLD}")
        if same_session and subset > SUBSET_REJECT_THRESHOLD:
            reasons.append(f"subset={subset:.1%}>{SUBSET_REJECT_THRESHOLD:.0%}")
        if reasons:
            reject_reasons.append(f"{live.strategy_id}: {'; '.join(reasons)}")
    return not reject_reasons, tuple(reject_reasons)


def _delta(candidate: PortfolioSnapshot | None, base: PortfolioSnapshot, attr: str) -> float | None:
    if candidate is None:
        return None
    value = getattr(candidate, attr)
    base_value = getattr(base, attr)
    if value is None or base_value is None:
        return None
    return float(value - base_value)


def _candidate_p90_orb(candidate: CandidateRecord) -> float | None:
    frame = load_lane(candidate.orb_label, candidate.orb_minutes, candidate.rr_target, candidate.instrument)
    if frame.empty:
        return None
    active = compute_deployed_filter(frame, candidate.filter_type, candidate.orb_label).astype(bool)
    selected = frame.loc[active]
    if selected.empty or "orb_size" not in selected:
        return None
    return float(selected["orb_size"].quantile(0.90))


def _allocation_risk(row: dict[str, Any], profile_stop_multiplier: float) -> float:
    instrument = str(row.get("instrument", "MNQ"))
    point_value = COST_SPECS[instrument].point_value
    p90_orb = float(row.get("p90_orb_pts") or 120.0)
    return p90_orb * profile_stop_multiplier * point_value


def _candidate_risk(candidate: CandidateRecord) -> tuple[float | None, str]:
    p90_orb = _candidate_p90_orb(candidate)
    if p90_orb is None:
        return None, "candidate p90 ORB unavailable"
    point_value = COST_SPECS[candidate.instrument].point_value
    risk = p90_orb * ACCOUNT_PROFILES[PROFILE_ID].stop_multiplier * point_value
    return risk, f"candidate_p90_orb={p90_orb:.1f}pts, candidate_risk=${risk:.0f}"


def _account_risk_gate(
    candidate: CandidateRecord,
    allocation: dict[str, Any],
    replacement_target: str | None,
    replacement_status: str | None,
) -> tuple[bool, str]:
    profile = ACCOUNT_PROFILES[PROFILE_ID]
    tier = ACCOUNT_TIERS[(profile.firm, profile.account_size)]
    deploy_rows = [row for row in allocation.get("lanes", []) if row.get("status") in RUNTIME_ACTIVE_STATUSES]
    current_risk = sum(_allocation_risk(row, profile.stop_multiplier) for row in deploy_rows)
    if replacement_target and replacement_status in RUNTIME_ACTIVE_STATUSES:
        current_risk -= sum(
            _allocation_risk(row, profile.stop_multiplier)
            for row in deploy_rows
            if row.get("strategy_id") == replacement_target
        )
    candidate_risk, candidate_detail = _candidate_risk(candidate)
    if candidate_risk is None:
        return False, candidate_detail
    proposed_risk = current_risk + candidate_risk
    proposed_slots = len(deploy_rows) + (0 if replacement_status in RUNTIME_ACTIVE_STATUSES else 1)
    if proposed_slots > profile.max_slots:
        return False, f"slots {proposed_slots}>{profile.max_slots}; {candidate_detail}"
    if proposed_risk > tier.max_dd:
        return False, f"worst_case=${proposed_risk:.0f}>max_dd=${tier.max_dd:.0f}; {candidate_detail}"
    return (
        True,
        f"worst_case=${proposed_risk:.0f}<={tier.max_dd:.0f}; slots={proposed_slots}/{profile.max_slots}; {candidate_detail}",
    )


def compute_portfolio_gate(
    candidate: CandidateRecord,
    *,
    live_specs: list[LaneSpec],
    live_trades: dict[str, list[dict[str, Any]]],
    allocation: dict[str, Any],
) -> PortfolioGate:
    candidate_spec = _spec(candidate)
    same_session = _allocation_same_session(allocation).get((candidate.instrument, candidate.orb_label))
    replacement_target = str(same_session["strategy_id"]) if same_session else None
    replacement_status = str(same_session.get("status", "PAUSE")) if same_session else None
    account_ok, account_detail = _account_risk_gate(candidate, allocation, replacement_target, replacement_status)

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        configure_connection(con)
        candidate_trades = _load_trades(con, candidate_spec)

    relevant_trade_sets = list(live_trades.values()) + [candidate_trades]
    start_all, end_all = _common_window(*relevant_trade_sets)
    is_end = min(end_all, HOLDOUT_SACRED_FROM - timedelta(days=1))
    if start_all > is_end:
        raise RuntimeError(f"No IS window for {candidate.strategy_id}")

    base_trades = dict(live_trades)
    base_is = compute_snapshot("Current active book", base_trades, start_all, is_end)
    add_trades = dict(base_trades)
    add_trades[candidate.strategy_id] = candidate_trades
    add_is = compute_snapshot(f"Current active book + {candidate.strategy_id}", add_trades, start_all, is_end)

    replace_is = None
    if replacement_target and replacement_status in RUNTIME_ACTIVE_STATUSES:
        replace_trades = dict(base_trades)
        replace_trades.pop(replacement_target, None)
        replace_trades[candidate.strategy_id] = candidate_trades
        replace_is = compute_snapshot(
            f"Replace {replacement_target} with {candidate.strategy_id}",
            replace_trades,
            start_all,
            is_end,
        )

    trades_all = dict(live_trades)
    trades_all[candidate.strategy_id] = candidate_trades
    corr_gate_pass, corr_reasons = _pair_correlation_gate(
        candidate_spec,
        live_specs,
        trades_all,
        start_date=start_all,
        end_date=is_end,
    )
    return PortfolioGate(
        add_delta_annual_r=_delta(add_is, base_is, "annual_r"),
        add_delta_sharpe=_delta(add_is, base_is, "sharpe_ann"),
        replace_delta_annual_r=_delta(replace_is, base_is, "annual_r"),
        replace_delta_sharpe=_delta(replace_is, base_is, "sharpe_ann"),
        corr_gate_pass=corr_gate_pass,
        corr_reject_reasons=corr_reasons,
        replacement_target=replacement_target,
        replacement_target_status=replacement_status,
        account_risk_ok=account_ok,
        account_risk_detail=account_detail,
    )


def _gate_needed(candidate: CandidateRecord, dedupe_head: bool) -> bool:
    if not dedupe_head:
        return False
    prelim = classify_candidate(candidate, dedupe_head=dedupe_head, gate=None)
    return not (
        prelim.decision in {"KILL", "ARCHITECTURE_REQUIRED"}
        or prelim.blockers
        and prelim.blockers[0].startswith("chordia_")
    )


def run() -> list[CandidateResult]:
    candidates = load_candidates()
    dedupe_heads = choose_dedupe_heads(candidates)
    allocation = _load_allocation()
    live_specs = load_live_lane_specs(PROFILE_ID)
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        configure_connection(con)
        live_trades = {spec.strategy_id: _load_trades(con, spec) for spec in live_specs}

    results: list[CandidateResult] = []
    for candidate in candidates:
        is_head = dedupe_heads[candidate.strategy_id]
        gate = (
            compute_portfolio_gate(
                candidate,
                live_specs=live_specs,
                live_trades=live_trades,
                allocation=allocation,
            )
            if _gate_needed(candidate, is_head)
            else None
        )
        decision = classify_candidate(candidate, dedupe_head=is_head, gate=gate)
        results.append(CandidateResult(candidate, decision, gate))
    return results


def _fmt_delta(value: float | None, digits: int = 3) -> str:
    return fmt_num(value, digits, signed=True)


def _render(results: list[CandidateResult]) -> str:
    counts = Counter(result.decision.decision for result in results)
    blocker_counts = Counter(blocker for result in results for blocker in result.decision.blockers[:1])
    pass_rows = [result for result in results if result.decision.decision in {"PASS_ADD", "PASS_REPLACE"}]
    park_rows = [result for result in results if result.decision.decision == "PARK"]

    lines = [
        "# MNQ Profile Candidate Proposal",
        "",
        "**Date:** 2026-05-11",
        f"**Profile:** `{PROFILE_ID}`",
        "**Live impact:** None. No DB, schema, broker, validated-setups, or `allocation file` mutation.",
        "",
        "## Scope",
        "",
        f"This classifies the {len(results)} `CONTROLLED_LIVE_PILOT_CANDIDATE` MNQ rows from `{INPUT_JSON}` into the queue-required profile-construction taxonomy.",
        "",
        "This is a paper/sandbox-only proposal gate. A `PASS_*` row is not live authorization; it still carries controlled-pilot slippage/event-tail status and must pass operator preflight before any allocation mutation.",
        "",
        "## Grounding / Authority",
        "",
        "- Multiple-testing and strict replay gate: `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` extracted from `resources/Two_Million_Trading_Strategies_FDR.pdf`.",
        "- Selection-bias and correlated-trial caution: `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` extracted from `resources/deflated-sharpe.pdf`.",
        "- Portfolio add/replace and correlation discipline: `docs/institutional/literature/carver_2015_ch11_portfolios.md` extracted from `resources/Robert Carver - Systematic Trading.pdf`.",
        "- Live drift monitoring requirement: `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` extracted from `resources/real_time_strategy_monitoring_cusum.pdf`.",
        "- Prop-firm profile constraints: `resources/prop-firm-official-rules.md` plus `trading_app/prop_profiles.py`.",
        "- Research/deployment separation: `docs/institutional/research_pipeline_contract.md` and `docs/institutional/pre_registered_criteria.md`.",
        "",
        "## Classification Counts",
        "",
        "| Decision | Count |",
        "|---|---:|",
    ]
    for label in ("PASS_ADD", "PASS_REPLACE", "PARK", "KILL", "ARCHITECTURE_REQUIRED"):
        lines.append(f"| `{label}` | {counts.get(label, 0)} |")

    lines.extend(["", "## Proposed Paper/Sandbox Change Set", ""])
    if not pass_rows:
        lines.append("No `PASS_ADD` or `PASS_REPLACE` candidates survived the profile-construction gate.")
    else:
        lines.extend(
            [
                "| Candidate | Decision | Target | Add dAnnR | Add dSharpe | Replace dAnnR | Replace dSharpe | Account Risk |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for result in pass_rows:
            gate = result.gate
            assert gate is not None
            lines.append(
                f"| `{result.candidate.strategy_id}` | `{result.decision.decision}` | "
                f"`{gate.replacement_target or 'new slot'}` | {_fmt_delta(gate.add_delta_annual_r, 1)} | "
                f"{_fmt_delta(gate.add_delta_sharpe, 3)} | {_fmt_delta(gate.replace_delta_annual_r, 1)} | "
                f"{_fmt_delta(gate.replace_delta_sharpe, 3)} | {gate.account_risk_detail} |"
            )

    lines.extend(
        [
            "",
            "## Main Blockers",
            "",
            "| Blocker | Count |",
            "|---|---:|",
        ]
    )
    for blocker, count in blocker_counts.most_common():
        lines.append(f"| `{blocker}` | {count} |")

    lines.extend(
        [
            "",
            "## Highest-Signal Parked Rows",
            "",
            "| Candidate | Reason | Chordia | Add dAnnR | Add dSharpe | Corr |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for result in sorted(park_rows, key=lambda r: r.candidate.expectancy_r, reverse=True)[:20]:
        gate = result.gate
        lines.append(
            f"| `{result.candidate.strategy_id}` | {result.decision.primary_reason} | "
            f"`{result.candidate.chordia_verdict}` | {_fmt_delta(gate.add_delta_annual_r, 1) if gate else '-'} | "
            f"{_fmt_delta(gate.add_delta_sharpe, 3) if gate else '-'} | "
            f"`{gate.corr_gate_pass}` |"
            if gate
            else f"| `{result.candidate.strategy_id}` | {result.decision.primary_reason} | "
            f"`{result.candidate.chordia_verdict}` | - | - | - |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "No direct allocation mutation is authorized by this report. Rows parked by `runtime_sr_not_evaluated` may only enter a PROVISIONAL bootstrap patch if a pre-promotion Criterion 12 SR evaluation returns `CONTINUE`, `NO_DATA`, or a code-backed `watch` review. An unreviewed SR `ALARM` remains a hard deployment block. Rows parked by missing Chordia audit need exact-lane strict replay before any profile proposal; dominated duplicates should not be re-opened unless the selected head fails under a newer audit.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-wsl/bin/python research/mnq_profile_candidate_proposal_2026_05_11.py",
            "# Optional patch emission path; still fails closed on unreviewed SR ALARM:",
            "./.venv-wsl/bin/python research/mnq_profile_candidate_proposal_2026_05_11.py --bootstrap-runtime-control",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(results: list[CandidateResult]) -> None:
    RESULT_MD.write_text(_render(results), encoding="utf-8")
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "strategy_id",
                "decision",
                "primary_reason",
                "blockers",
                "chordia_verdict",
                "expectancy_r",
                "sample_size",
                "orb_label",
                "filter_type",
                "rr_target",
                "replacement_target",
                "replacement_target_status",
                "add_delta_annual_r",
                "add_delta_sharpe",
                "replace_delta_annual_r",
                "replace_delta_sharpe",
                "corr_gate_pass",
                "corr_reject_reasons",
                "account_risk_ok",
                "account_risk_detail",
            ]
        )
        for result in results:
            gate = result.gate
            writer.writerow(
                [
                    result.candidate.strategy_id,
                    result.decision.decision,
                    result.decision.primary_reason,
                    ";".join(result.decision.blockers),
                    result.candidate.chordia_verdict,
                    result.candidate.expectancy_r,
                    result.candidate.sample_size,
                    result.candidate.orb_label,
                    result.candidate.filter_type,
                    result.candidate.rr_target,
                    gate.replacement_target if gate else "",
                    gate.replacement_target_status if gate else "",
                    gate.add_delta_annual_r if gate else "",
                    gate.add_delta_sharpe if gate else "",
                    gate.replace_delta_annual_r if gate else "",
                    gate.replace_delta_sharpe if gate else "",
                    gate.corr_gate_pass if gate else "",
                    ";".join(gate.corr_reject_reasons) if gate else "",
                    gate.account_risk_ok if gate else "",
                    gate.account_risk_detail if gate else "",
                ]
            )


RuntimeBootstrapChecker = Callable[[CandidateRecord], tuple[bool, str]]


def _runtime_bootstrap_allows(candidate: CandidateRecord) -> tuple[bool, str]:
    """Evaluate Criterion 12 for a not-yet-selected lane before patch emission."""
    from trading_app.sr_monitor import prepare_monitor_inputs
    from trading_app.sr_review_registry import get_sr_alarm_review

    params = {
        "mu0": candidate.expectancy_r,
        "sigma": 1.0,
        "instrument": candidate.instrument,
        "orb_label": candidate.orb_label,
        "orb_minutes": candidate.orb_minutes,
        "entry_model": candidate.entry_model,
        "rr_target": candidate.rr_target,
        "confirm_bars": candidate.confirm_bars,
        "filter_type": candidate.filter_type,
    }
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        configure_connection(con)
        monitor, trades, baseline_source, stream_source = prepare_monitor_inputs(con, candidate.strategy_id, params)

    status = "NO_DATA"
    alarm_trade = None
    for i, trade_r in enumerate(trades, 1):
        if monitor.update(trade_r):
            status = "ALARM"
            alarm_trade = i
            break
    else:
        if trades:
            status = "CONTINUE"

    detail = (
        f"SR bootstrap status={status}; n={len(trades)}; stat={monitor.sr_stat:.2f}; "
        f"threshold={monitor.threshold:.2f}; baseline={baseline_source}; stream={stream_source}"
    )
    if status != "ALARM":
        return True, detail
    review = get_sr_alarm_review(PROFILE_ID, candidate.strategy_id)
    if review is not None and review.outcome == "watch":
        return True, f"{detail}; reviewed WATCH at {review.reviewed_at}"
    suffix = f"; alarm_trade={alarm_trade}" if alarm_trade is not None else ""
    return False, detail + suffix


def _effective_promotion_decision(
    result: CandidateResult,
    *,
    allow_runtime_bootstrap: bool,
    runtime_bootstrap_checker: RuntimeBootstrapChecker | None,
) -> str | None:
    if result.decision.decision in {"PASS_ADD", "PASS_REPLACE"}:
        return result.decision.decision
    if (
        allow_runtime_bootstrap
        and result.decision.decision == "PARK"
        and result.decision.blockers == ("runtime_sr_not_evaluated",)
        and result.gate is not None
    ):
        candidate = replace(result.candidate, runtime_control_evaluated=True)
        bootstrap = classify_candidate(candidate, dedupe_head=True, gate=result.gate)
        if bootstrap.decision in {"PASS_ADD", "PASS_REPLACE"}:
            if runtime_bootstrap_checker is not None:
                allowed, _detail = runtime_bootstrap_checker(result.candidate)
                if not allowed:
                    return None
            return bootstrap.decision
    return None


def build_promotion_candidates(
    results: list[CandidateResult],
    *,
    allow_runtime_bootstrap: bool = False,
    runtime_bootstrap_checker: RuntimeBootstrapChecker | None = None,
) -> list[PromotionCandidate]:
    promotions: list[PromotionCandidate] = []
    for result in results:
        decision = _effective_promotion_decision(
            result,
            allow_runtime_bootstrap=allow_runtime_bootstrap,
            runtime_bootstrap_checker=runtime_bootstrap_checker,
        )
        if decision is None:
            continue
        gate = result.gate
        if gate is None:
            raise RuntimeError(f"{result.candidate.strategy_id}: PASS row missing portfolio gate")
        p90_orb = _candidate_p90_orb(result.candidate)
        annual_r = None
        if result.candidate.trades_per_year is not None:
            annual_r = result.candidate.expectancy_r * result.candidate.trades_per_year
        promotions.append(
            PromotionCandidate(
                profile_id=PROFILE_ID,
                strategy_id=result.candidate.strategy_id,
                decision=decision,
                instrument=result.candidate.instrument,
                orb_label=result.candidate.orb_label,
                orb_minutes=result.candidate.orb_minutes,
                rr_target=result.candidate.rr_target,
                filter_type=result.candidate.filter_type,
                status="PROVISIONAL",
                status_reason=(
                    f"{decision} via profile-construction gate; "
                    "controlled-pilot/event-tail slippage status retained; "
                    "post-promotion SR refresh required before session start"
                ),
                chordia_verdict=result.candidate.chordia_verdict,
                chordia_audit_age_days=result.candidate.chordia_audit_age_days,
                annual_r=annual_r,
                trailing_expr=result.candidate.expectancy_r,
                trailing_n=result.candidate.sample_size,
                trailing_wr=result.candidate.win_rate,
                months_negative=None,
                session_regime=None,
                avg_orb_pts=None,
                p90_orb_pts=p90_orb,
                replacement_target=gate.replacement_target,
                replacement_target_status=gate.replacement_target_status,
                source_path=str(RESULT_MD),
                account_risk_detail=gate.account_risk_detail,
            )
        )
    return promotions


def write_promotion_patch(
    results: list[CandidateResult],
    *,
    apply_allocation: bool = False,
    allow_runtime_bootstrap: bool = False,
) -> None:
    if apply_allocation:
        raise RuntimeError(
            "research proposal is patch-artifact only; apply allocations with the canonical operator rebalance flow"
        )
    promotions = build_promotion_candidates(
        results,
        allow_runtime_bootstrap=allow_runtime_bootstrap,
        runtime_bootstrap_checker=_runtime_bootstrap_allows if allow_runtime_bootstrap else None,
    )
    if not promotions:
        if RESULT_PROMOTION_PATCH_JSON.exists():
            RESULT_PROMOTION_PATCH_JSON.unlink()
        return
    allocation = _load_allocation()
    patch = apply_promotions(allocation, promotions, rebalance_date=REPORT_DATE)
    RESULT_PROMOTION_PATCH_JSON.write_text(patch.to_json(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply-allocation",
        action="store_true",
        help="Refused: research proposals emit patch artifacts only; use the canonical operator rebalance flow.",
    )
    parser.add_argument(
        "--bootstrap-runtime-control",
        action="store_true",
        help=(
            "Allow PROVISIONAL promotion when the only remaining blocker is runtime_control.evaluated=false "
            "because the candidate is not yet selected into the profile. Caller must refresh SR state after apply."
        ),
    )
    args = parser.parse_args()

    results = run()
    write_outputs(results)
    write_promotion_patch(
        results,
        apply_allocation=args.apply_allocation,
        allow_runtime_bootstrap=args.bootstrap_runtime_control,
    )
    counts = Counter(result.decision.decision for result in results)
    print(f"Wrote {RESULT_MD}")
    print(f"Wrote {RESULT_CSV}")
    if RESULT_PROMOTION_PATCH_JSON.exists():
        print(f"Wrote {RESULT_PROMOTION_PATCH_JSON}")
    print(dict(sorted(counts.items())))
    for result in results:
        if result.decision.decision in {"PASS_ADD", "PASS_REPLACE"}:
            print(f"{result.decision.decision}: {result.candidate.strategy_id} | {result.decision.primary_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
