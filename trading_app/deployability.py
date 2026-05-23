"""Full-shelf deployability audit for validated strategy candidates.

This module is deliberately stricter than the research validator. A row can be
valid research inventory while still being blocked from deployment because
runtime, slippage, account-risk, OOS power, or replay evidence is incomplete.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.data_era import is_micro
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.chordia import chordia_verdict_allows_deploy, chordia_verdict_label
from trading_app.config import (
    ALL_FILTERS,
    CORE_MIN_SAMPLES,
    MIN_WFE,
    REGIME_MIN_SAMPLES,
    is_e2_deployment_unsafe_filter,
)
from trading_app.lifecycle_state import read_lifecycle_state
from trading_app.opportunity_awareness import describe_opportunity_awareness
from trading_app.prop_profiles import get_profile_lane_definitions, resolve_profile_id
from trading_app.strategy_fitness import _load_strategy_outcomes
from trading_app.strategy_validator import _evaluate_criterion_8_oos, benjamini_hochberg

DEPLOYABLE_CANDIDATE = "DEPLOYABLE_CANDIDATE"
CONTROLLED_LIVE_PILOT_CANDIDATE = "CONTROLLED_LIVE_PILOT_CANDIDATE"
RESEARCH_PROVISIONAL = "RESEARCH_PROVISIONAL"
BLOCKED_MISSING_EVIDENCE = "BLOCKED_MISSING_EVIDENCE"
BLOCKED_REPLAY_MISMATCH = "BLOCKED_REPLAY_MISMATCH"
BLOCKED_OOS_UNDERPOWERED = "BLOCKED_OOS_UNDERPOWERED"
BLOCKED_CURRENT_K_FDR = "BLOCKED_CURRENT_K_FDR"
BLOCKED_FAMILY_FRAGILE = "BLOCKED_FAMILY_FRAGILE"
BLOCKED_SLIPPAGE = "BLOCKED_SLIPPAGE"
BLOCKED_ACCOUNT_RISK = "BLOCKED_ACCOUNT_RISK"
BLOCKED_RUNTIME = "BLOCKED_RUNTIME"
NO_GO_BIAS_OR_DATA = "NO_GO_BIAS_OR_DATA"
NOT_PROFILE_EVALUATED = "NOT_PROFILE_EVALUATED"

Scope = Literal["all-active", "profile"]
EvidenceClass = Literal["MEASURED", "INFERRED", "UNSUPPORTED"]

SLIPPAGE_PASS_STATUSES = {
    "PASS",
    "PASSED",
    "OK",
    "VALIDATED",
    "ROBUST",
    "SLIPPAGE_PASS",
    "SLIPPAGE_PASSED",
}

MNQ_ROUTINE_TBBO_SLIPPAGE_SESSIONS = frozenset(
    {
        "CME_PRECLOSE",
        "COMEX_SETTLE",
        "EUROPE_FLOW",
        "LONDON_METALS",
        "NYSE_OPEN",
        "SINGAPORE_OPEN",
        "TOKYO_OPEN",
        "US_DATA_1000",
        "US_DATA_830",
    }
)
MNQ_ROUTINE_TBBO_SLIPPAGE_BASIS = (
    "MNQ routine TBBO slippage measured conservative across all 9 deployed sessions; "
    "event-day tail remains an open known-unknown per docs/runtime/debt-ledger.md and "
    "docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v2-gap-fill.md"
)

_MES_ROUTINE_TBBO_SLIPPAGE_BASIS = (
    "MES routine TBBO slippage measured at or below modeled 1-tick on 4 deployable MES "
    "sessions per docs/audit/results/2026-04-24-mes-e2-slippage-pilot-v1.md "
    "(median 0.00 ticks, p95 0.00 ticks, 100% within 1 tick on N=40); "
    "event-day tail remains an open known-unknown per docs/runtime/debt-ledger.md"
)


@dataclass(frozen=True)
class RoutineTbboPilot:
    """Committed routine-TBBO slippage pilot evidence keyed by instrument.

    A registry entry means: pilot v1 has shipped a PASS verdict and routine-liquidity
    slippage is at or under the modeled assumption for the listed sessions on the
    listed entry model. The deployability audit treats matching rows as
    slippage_event_tail_pending (warning, controlled-pilot eligible) rather than
    slippage_missing (hard, blocked). Only the event-day tail remains as open debt
    per docs/runtime/debt-ledger.md.
    """

    instrument: str
    entry_model: str
    sessions: frozenset[str]
    basis: str


# Routine-TBBO slippage pilots that have shipped a PASS verdict. Coverage is enforced
# by pipeline.check_drift.check_routine_tbbo_slippage_registry_coverage: every committed
# pilot v1 result MD with verdict PASS must be registered here, and WARN/FAIL pilots
# must NOT be registered. Add a new instrument here when its pilot v1 lands PASS.
ROUTINE_TBBO_SLIPPAGE_REGISTRY: dict[str, RoutineTbboPilot] = {
    "MNQ": RoutineTbboPilot(
        instrument="MNQ",
        entry_model="E2",
        sessions=MNQ_ROUTINE_TBBO_SLIPPAGE_SESSIONS,
        basis=MNQ_ROUTINE_TBBO_SLIPPAGE_BASIS,
    ),
    "MES": RoutineTbboPilot(
        instrument="MES",
        entry_model="E2",
        sessions=frozenset(
            {
                "CME_PRECLOSE",
                "COMEX_SETTLE",
                "SINGAPORE_OPEN",
                "US_DATA_830",
            }
        ),
        basis=_MES_ROUTINE_TBBO_SLIPPAGE_BASIS,
    ),
}

REPLAY_CONNECTION_REFRESH_ROWS = 50

HARD_BLOCKER_TO_VERDICT = {
    "replay_mismatch": BLOCKED_REPLAY_MISMATCH,
    "unknown_filter": NO_GO_BIAS_OR_DATA,
    "e2_lookahead_filter": NO_GO_BIAS_OR_DATA,
    "e2_deployment_unsafe_filter": NO_GO_BIAS_OR_DATA,
    "current_k_fdr_fail": BLOCKED_CURRENT_K_FDR,
    "current_k_fdr_missing": BLOCKED_CURRENT_K_FDR,
    "family_purged": BLOCKED_FAMILY_FRAGILE,
    "family_singleton": BLOCKED_FAMILY_FRAGILE,
    "slippage_missing": BLOCKED_SLIPPAGE,
    "slippage_not_passed": BLOCKED_SLIPPAGE,
    "c8_missing": BLOCKED_MISSING_EVIDENCE,
    "c8_not_passed": BLOCKED_OOS_UNDERPOWERED,
    "account_risk_missing": BLOCKED_ACCOUNT_RISK,
    "account_risk_fail": BLOCKED_ACCOUNT_RISK,
    "criterion12_invalid": BLOCKED_RUNTIME,
    "criterion12_missing_lane": BLOCKED_RUNTIME,
    "sr_alarm_unreviewed": BLOCKED_RUNTIME,
    "lifecycle_blocked": BLOCKED_RUNTIME,
    "runtime_blocked": BLOCKED_RUNTIME,
}

RETIRE_OR_PURGE_ISSUES = {
    "unknown_filter",
    "e2_lookahead_filter",
    "e2_deployment_unsafe_filter",
    "current_k_fdr_fail",
    "family_purged",
    # `family_singleton` is intentionally NOT in this set per Stage 4 doctrine
    # (Disposition C, Stage 3 § 4). A SINGLETON-status row is a no-peer-evidence
    # state, not a rejected-family state — distinct from `family_purged`. When
    # the row also fails any binding criterion (C3/C4/C6/C7/C9/C10) the issue
    # is emitted as `hard` and BLOCKED_FAMILY_FRAGILE verdict is still returned
    # via HARD_BLOCKER_TO_VERDICT, but the row no longer routes to retire-or-purge
    # bucket on the singleton signal alone. When the row clears all binding
    # criteria the issue is emitted as `warning` and routed to
    # CONTROLLED_LIVE_PILOT_CANDIDATE via CONTROLLED_PILOT_WARNINGS.
    "replay_mismatch",
    "wfe_below_threshold",
}
EVIDENCE_GAP_ISSUES = {
    "slippage_missing",
    "slippage_not_passed",
    "c8_missing",
    "account_risk_missing",
    "account_risk_fail",
}

# Warning issues that, when present and not accompanied by any hard blocker,
# route the verdict to CONTROLLED_LIVE_PILOT_CANDIDATE rather than full
# DEPLOYABLE_CANDIDATE. These represent evidence states that are not
# blocking but warrant a supervised pilot before full auto-deploy.
# - slippage_event_tail_pending: routine TBBO slippage measured but
#   event-day tail debt unresolved.
# - sr_alarm_watch_reviewed: Shiryaev-Roberts monitor on WATCH; reviewed
#   but not cleared.
# - family_singleton: SINGLETON-status row that clears all binding
#   pre_registered_criteria.md C-criteria (Disposition C, Stage 4,
#   2026-05-11). No peer-evidence by construction, so route to pilot
#   for manual sign-off before lane allocation mutation.
CONTROLLED_PILOT_WARNINGS = {
    "slippage_event_tail_pending",
    "sr_alarm_watch_reviewed",
    "family_singleton",
}


@dataclass(frozen=True)
class DeployabilityIssue:
    id: str
    severity: Literal["hard", "warning", "info"]
    evidence_class: EvidenceClass
    detail: Any


@dataclass(frozen=True)
class StrategyDeployability:
    strategy_id: str
    instrument: str | None
    verdict: str
    deployable: bool
    institutional_language_allowed: bool
    replay: dict[str, Any]
    current_k_fdr: dict[str, Any]
    c8_oos: dict[str, Any]
    trade_context: dict[str, Any]
    runtime_control: dict[str, Any]
    issues: list[DeployabilityIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [asdict(issue) for issue in self.issues]
        return data


def _active_in_sql() -> str:
    return ", ".join(f"'{inst}'" for inst in ACTIVE_ORB_INSTRUMENTS)


def _singleton_clears_binding_criteria(
    row: dict[str, Any],
) -> tuple[bool, list[str], bool]:
    """Evaluate the SINGLETON-pass floor per Stage 4 Disposition C.

    Returns (passes, failed_criterion_ids, dsr_reported). The floor
    enforces the binding criteria from
    `docs/institutional/pre_registered_criteria.md` Enforcement summary
    (lines 480-494, post-amendments authoritative):
        C3 BH FDR        — fdr_significant AND fdr_adjusted_p < 0.05
        C4 Chordia       — chordia_verdict_label in {PASS_CHORDIA, PASS_PROTOCOL_A}
        C6 WFE           — wfe >= 0.50
        C7 sample size   — sample_size >= 100
        C9 era stability — era_dependent == False
        C10 micro-era    — if filter.requires_micro_data, instrument must be micro

    C5 (DSR) is CROSS-CHECK ONLY per Amendment 2.1 (line 367-370) and
    therefore IS NOT a gating criterion in this function. The DSR
    reporting state is returned as the third tuple element
    `dsr_reported` (True when dsr_score IS NOT NULL) so callers can
    surface it in audit detail without conflating it with the binding
    gate. A SINGLETON with dsr_score IS NULL but all six binding
    criteria cleared MUST pass the floor — the reporting gap is
    informational, not blocking.

    For C4, `has_theory` defaults to False because SINGLETON candidates are
    by construction Pathway A family-pathway promotions (no per-strategy
    theory citation). BAND A (t >= 3.79) still passes; BAND B (t >= 3.00
    with theory) does NOT, by construction of has_theory=False.

    A row that is not SINGLETON returns (False, [], False) — callers
    should gate on `robustness_status == "SINGLETON"` before calling.

    The adversarial-audit gate on 2026-05-11 caught a prior version of
    this function that accidentally appended `C5_DSR_uncomputed` to the
    `failed` list, gating contrary to Amendment 2.1. Do not regress
    this behaviour.
    """
    failed: list[str] = []

    # C3 BH FDR
    fdr_sig = row.get("fdr_significant")
    fdr_q = row.get("fdr_adjusted_p")
    if not (fdr_sig and fdr_q is not None and float(fdr_q) < 0.05):
        failed.append("C3_BH_FDR")

    # C4 Chordia banded (delegate to canonical helper)
    sr = row.get("sharpe_ratio")
    n = row.get("sample_size")
    chordia_label = chordia_verdict_label(
        sharpe_ratio=float(sr) if sr is not None else None,
        sample_size=int(n) if n is not None else None,
        has_theory=False,
    )
    if not chordia_verdict_allows_deploy(chordia_label):
        failed.append(f"C4_Chordia[{chordia_label}]")

    # C5 DSR is CROSS-CHECK ONLY per pre_registered_criteria.md Amendment
    # 2.1 (line 367-370). Its absence MUST NOT fail the floor — the
    # criterion is informational pending the N_eff / ONC resolution
    # workstream. We record NULL as a reporting note for the audit trail
    # but do NOT append it to `failed`. This is what the docstring above
    # describes, and what the adversarial-audit gate (2026-05-11) caught
    # as a contradiction in the original Stage 4 implementation.
    dsr_reported = row.get("dsr_score") is not None

    # C6 WFE >= 0.50
    wfe = row.get("wfe")
    if wfe is None or float(wfe) < MIN_WFE:
        failed.append("C6_WFE")

    # C7 N >= 100
    if n is None or int(n) < CORE_MIN_SAMPLES:
        failed.append("C7_N")

    # C9 era stability — era_dependent should be False
    era_dep = row.get("era_dependent")
    if era_dep is None or bool(era_dep):
        failed.append("C9_era_dependent")

    # C10 micro-era compatibility — if filter requires micro data,
    # instrument must be a real micro contract. Delegate to canonical
    # predicates.
    filter_type = row.get("filter_type")
    instrument = row.get("instrument")
    if filter_type and instrument:
        filter_obj = ALL_FILTERS.get(filter_type)
        if filter_obj is not None and getattr(filter_obj, "requires_micro_data", False):
            try:
                if not is_micro(instrument):
                    failed.append("C10_micro_only_filter_on_non_micro")
            except Exception:  # noqa: BLE001 - fail closed
                failed.append("C10_micro_check_error")

    # Surface C5 reporting state alongside the failed list. C5 NULL is a
    # cross-check reporting gap, not a gate failure (Amendment 2.1).
    # Callers use the returned `failed` list to gate; the C5 reporting
    # state is observable via the `dsr_reported` flag we hand back as
    # part of the row's audit detail. We append a non-blocking advisory
    # tag to `failed` ONLY in the human-readable detail dict (see the
    # family-branch caller), never as a blocking criterion here.
    return (len(failed) == 0, failed, dsr_reported)


def _has_table(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        LIMIT 1
        """,
        [table_name],
    ).fetchone()
    return row is not None


def _load_candidate_rows(
    con: duckdb.DuckDBPyConnection,
    *,
    scope: Scope,
    profile_id: str | None,
    instruments: set[str] | None = None,
) -> list[dict[str, Any]]:
    fields = """
        vs.strategy_id, vs.instrument, vs.orb_label, vs.orb_minutes,
        vs.entry_model, vs.rr_target, vs.confirm_bars, vs.filter_type,
        LOWER(vs.status) AS status,
        LOWER(COALESCE(vs.deployment_scope, 'deployable')) AS deployment_scope,
        vs.sample_size, vs.win_rate, vs.expectancy_r, vs.oos_exp_r,
        vs.wfe, vs.dsr_score, vs.sharpe_ratio, vs.era_dependent,
        vs.years_tested, vs.first_trade_day,
        vs.last_trade_day, vs.trade_day_count, vs.fdr_adjusted_p,
        vs.fdr_significant, vs.discovery_k, vs.slippage_validation_status,
        vs.c8_oos_status, vs.validation_pathway,
        ef.robustness_status, ef.trade_tier, ef.pbo
    """
    if scope == "profile":
        resolved = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
        lane_ids = [str(lane["strategy_id"]) for lane in get_profile_lane_definitions(resolved)]
        if not lane_ids:
            return []
        placeholders = ", ".join("?" for _ in lane_ids)
        rows = con.execute(
            f"""
            SELECT {fields}
            FROM validated_setups vs
            LEFT JOIN edge_families ef ON ef.family_hash = vs.family_hash
            WHERE vs.strategy_id IN ({placeholders})
            ORDER BY vs.instrument, vs.orb_label, vs.strategy_id
            """,
            lane_ids,
        ).fetchall()
    else:
        rows = con.execute(
            f"""
            SELECT {fields}
            FROM validated_setups vs
            LEFT JOIN edge_families ef ON ef.family_hash = vs.family_hash
            WHERE LOWER(vs.status) = 'active'
              AND vs.instrument IN ({_active_in_sql()})
            ORDER BY vs.instrument, vs.orb_label, vs.strategy_id
            """
        ).fetchall()

    cols = [desc[0] for desc in con.description]
    out = [dict(zip(cols, row, strict=False)) for row in rows]
    if instruments:
        out = [row for row in out if str(row.get("instrument")) in instruments]
    return out


def _bh_adjusted_for_session(con: duckdb.DuckDBPyConnection, orb_label: str) -> dict[str, float]:
    rows = con.execute(
        f"""
        SELECT strategy_id, p_value
        FROM experimental_strategies
        WHERE is_canonical = TRUE
          AND orb_label = ?
          AND p_value IS NOT NULL
          AND instrument IN ({_active_in_sql()})
        ORDER BY p_value
        """,
        [orb_label],
    ).fetchall()
    if not rows:
        return {}
    p_values = [(str(strategy_id), float(p_value)) for strategy_id, p_value in rows]
    adjusted = benjamini_hochberg(p_values, alpha=0.05, total_tests=len(p_values))
    return {strategy_id: float(result["adjusted_p"]) for strategy_id, result in adjusted.items()}


def _current_k_fdr(
    con: duckdb.DuckDBPyConnection,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_session = sorted({str(row["orb_label"]) for row in rows if row.get("orb_label")})
    adjusted_by_session = {session: _bh_adjusted_for_session(con, session) for session in by_session}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row["strategy_id"])
        current_adj = adjusted_by_session.get(str(row.get("orb_label")), {}).get(sid)
        out[sid] = {
            "stored_adj_p": row.get("fdr_adjusted_p"),
            "stored_k": row.get("discovery_k"),
            "current_adj_p": current_adj,
            "current_pass": None if current_adj is None else current_adj < 0.05,
        }
    return out


def _replay_strategy(
    con: duckdb.DuckDBPyConnection,
    row: dict[str, Any],
) -> dict[str, Any]:
    start_date = row.get("first_trade_day")
    end_date = row.get("last_trade_day")
    try:
        outcomes = _load_strategy_outcomes(
            con,
            instrument=str(row["instrument"]),
            orb_label=str(row["orb_label"]),
            orb_minutes=int(row["orb_minutes"]),
            entry_model=str(row["entry_model"]),
            rr_target=float(row["rr_target"]),
            confirm_bars=int(row["confirm_bars"]),
            filter_type=str(row["filter_type"]),
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "recomputed_sample_size": None,
            "stored_sample_size": row.get("sample_size"),
            "null_pnl_count": None,
            "replay_window": {
                "start": str(start_date) if start_date else None,
                "end": str(end_date) if end_date else None,
            },
        }

    pnl = [float(o["pnl_r"]) for o in outcomes if o.get("pnl_r") is not None]
    n = len(pnl)
    avg = sum(pnl) / n if n else None
    stored_n = int(row["sample_size"]) if row.get("sample_size") is not None else None
    stored_trade_day_count = int(row["trade_day_count"]) if row.get("trade_day_count") is not None else None
    expected_n = stored_trade_day_count if stored_trade_day_count is not None else stored_n
    stored_exp = float(row["expectancy_r"]) if row.get("expectancy_r") is not None else None
    n_match = expected_n == n if expected_n is not None else False
    exp_match = avg is not None and stored_exp is not None and math.isclose(avg, stored_exp, abs_tol=0.005)
    # If trade_day_count exists, it is the promotion provenance count for the
    # replayed trade set. sample_size/expectancy_r can be split-specific legacy
    # metrics, so ExpR drift is reported but does not by itself make replay fail.
    ok = n_match and (stored_trade_day_count is not None or exp_match)
    return {
        "ok": bool(ok),
        "recomputed_sample_size": n,
        "stored_sample_size": stored_n,
        "stored_trade_day_count": stored_trade_day_count,
        "sample_count_source": "trade_day_count" if stored_trade_day_count is not None else "sample_size",
        "recomputed_expectancy_r": avg,
        "stored_expectancy_r": stored_exp,
        "sample_size_match": n_match,
        "expectancy_match": exp_match,
        "null_pnl_count": len(outcomes) - n,
        "replay_window": {"start": str(start_date) if start_date else None, "end": str(end_date) if end_date else None},
        "first_trade_day": min((str(o["trading_day"]) for o in outcomes), default=None),
        "last_trade_day": max((str(o["trading_day"]) for o in outcomes), default=None),
    }


def _iter_replay_results(
    db_path: Path,
    rows: list[dict[str, Any]],
    *,
    refresh_every: int = REPLAY_CONNECTION_REFRESH_ROWS,
):
    """Replay audit rows with periodic connection refresh to avoid native-driver buildup."""

    replay_con: duckdb.DuckDBPyConnection | None = None
    try:
        for idx, row in enumerate(rows):
            if replay_con is None or idx % refresh_every == 0:
                if replay_con is not None:
                    replay_con.close()
                replay_con = duckdb.connect(str(db_path), read_only=True)
                configure_connection(replay_con)
            yield row, _replay_strategy(replay_con, row)
    finally:
        if replay_con is not None:
            replay_con.close()


def _slippage_passes(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().upper() in SLIPPAGE_PASS_STATUSES


def _slippage_is_controlled_event_tail_pending(
    row: dict[str, Any],
    value: Any,
    *,
    registry: dict[str, RoutineTbboPilot] = ROUTINE_TBBO_SLIPPAGE_REGISTRY,
) -> bool:
    if str(value).strip().upper() != "PENDING_EVENT_TAIL":
        return False
    # Any instrument with a registered routine-TBBO slippage pilot is eligible to
    # carry an explicit PENDING_EVENT_TAIL status. Hardcoding MNQ here was the
    # second half of the class bug fixed by the routine-TBBO registry refactor;
    # MES (and any future registered instrument) must follow the same path.
    return str(row.get("instrument") or "").upper() in registry


def _routine_tbbo_pilot_for_row(
    row: dict[str, Any],
    *,
    registry: dict[str, RoutineTbboPilot] = ROUTINE_TBBO_SLIPPAGE_REGISTRY,
) -> RoutineTbboPilot | None:
    instrument = str(row.get("instrument") or "").upper()
    pilot = registry.get(instrument)
    if pilot is None:
        return None
    if str(row.get("entry_model") or "").upper() != pilot.entry_model:
        return None
    if str(row.get("orb_label") or "").upper() not in pilot.sessions:
        return None
    return pilot


def _routine_tbbo_slippage_applies(
    row: dict[str, Any],
    *,
    registry: dict[str, RoutineTbboPilot] = ROUTINE_TBBO_SLIPPAGE_REGISTRY,
) -> bool:
    return _routine_tbbo_pilot_for_row(row, registry=registry) is not None


def _controlled_slippage_event_tail_detail(
    row: dict[str, Any],
    value: Any,
    *,
    inferred: bool,
    registry: dict[str, RoutineTbboPilot] = ROUTINE_TBBO_SLIPPAGE_REGISTRY,
) -> dict[str, Any]:
    pilot = _routine_tbbo_pilot_for_row(row, registry=registry)
    instrument = str(row.get("instrument") or "").upper()
    fallback_pilot = pilot or registry.get(instrument)
    sessions = fallback_pilot.sessions if fallback_pilot is not None else frozenset()
    basis = (
        fallback_pilot.basis
        if fallback_pilot is not None
        else "routine TBBO slippage pilot evidence not registered for this instrument"
    )
    return {
        "status": value,
        "effective_status": "PENDING_EVENT_TAIL",
        "inferred_from_routine_tbbo": inferred,
        "covered_session": str(row.get("orb_label") or "").upper() in sessions,
        "basis": basis,
    }


def _issue(issue_id: str, severity: Literal["hard", "warning", "info"], detail: Any) -> DeployabilityIssue:
    evidence_class: EvidenceClass = "MEASURED" if severity == "hard" else "INFERRED"
    if severity == "info" or issue_id.endswith("_unsupported"):
        evidence_class = "UNSUPPORTED"
    return DeployabilityIssue(issue_id, severity, evidence_class, detail)


def _trade_context(row: dict[str, Any]) -> dict[str, Any]:
    """Label the trade type so deployment review uses role-appropriate tests."""

    filter_type = str(row.get("filter_type") or "")
    filter_upper = filter_type.upper()
    components: list[str] = []
    if "ORB_G" in filter_upper or "ORB_L" in filter_upper or "COST_LT" in filter_upper:
        components.append("friction_or_orb_size_regime_filter")
    if any(token in filter_upper for token in ("ATR", "VOL", "GARCH", "WIDE", "COMP")):
        components.append("volatility_or_participation_regime_filter")
    if any(token in filter_upper for token in ("DOW", "FRIDAY", "CALENDAR", "OPEX", "NFP")):
        components.append("calendar_session_regime_filter")
    if any(token in filter_upper for token in ("DBL", "DOUBLE", "RETRACE", "FADE")):
        components.append("momentum_vs_mean_reversion_regime_filter")
    if any(token in filter_upper for token in ("VWAP", "PDR", "OVN", "GAP", "CONGEST", "MID")):
        components.append("location_or_confluence_conditioner")
    if not components and filter_upper not in ("", "NO_FILTER"):
        components.append("conditional_filter_unspecified_role")

    role = "standalone_lane"
    if components:
        role = "standalone_lane_with_conditional_filters"

    sample_size = row.get("sample_size")
    if sample_size is None:
        sample_class = "UNKNOWN"
    elif int(sample_size) < REGIME_MIN_SAMPLES:
        sample_class = "INVALID"
    elif int(sample_size) < CORE_MIN_SAMPLES:
        sample_class = "REGIME_CONDITIONAL_ONLY"
    elif int(sample_size) < 200:
        sample_class = "PRELIMINARY"
    elif int(sample_size) < 500:
        sample_class = "CORE"
    else:
        sample_class = "HIGH_CONFIDENCE_IF_MULTI_REGIME"

    return {
        "instrument": row.get("instrument"),
        "session": row.get("orb_label"),
        "archetype": "event_session_orb_breakout",
        "entry_model": row.get("entry_model"),
        "research_role": role,
        "conditional_components": components,
        "sample_class": sample_class,
        "role_authority": "docs/institutional/conditional-edge-framework.md",
        "regime_rule": (
            "Treat session/regime filters as conditional mechanisms; do not promote "
            "selected-trade mean into standalone or portfolio capacity without role-appropriate evidence."
        ),
    }


def _runtime_control(
    row: dict[str, Any],
    *,
    lifecycle_state: dict[str, Any] | None,
    profile_lane_ids: set[str],
    scope: Scope,
) -> dict[str, Any]:
    strategy_id = str(row["strategy_id"])
    in_profile = strategy_id in profile_lane_ids
    if lifecycle_state is None or (scope == "all-active" and not in_profile):
        return {"evaluated": False, "reason": "not selected profile scope"}

    criterion12 = lifecycle_state.get("criterion12") or {}
    strategy_state = (lifecycle_state.get("strategy_states") or {}).get(strategy_id, {})
    return {
        "evaluated": True,
        "criterion12_valid": criterion12.get("valid"),
        "criterion12_reason": criterion12.get("reason"),
        "criterion12_state_age_days": criterion12.get("state_age_days"),
        "sr_status": strategy_state.get("sr_status"),
        "sr_review_outcome": strategy_state.get("sr_review_outcome"),
        "sr_reviewed_at": strategy_state.get("sr_reviewed_at"),
        "sr_review_summary": strategy_state.get("sr_review_summary"),
        "sr_recheck_trigger": strategy_state.get("sr_recheck_trigger"),
        "blocked": bool(strategy_state.get("blocked")),
        "block_source": strategy_state.get("block_source"),
        "block_reason": strategy_state.get("block_reason"),
        "paused": bool(strategy_state.get("paused")),
        "pause_reason": strategy_state.get("pause_reason"),
    }


def _classify_strategy(
    row: dict[str, Any],
    *,
    replay: dict[str, Any],
    current_fdr: dict[str, Any],
    c8: dict[str, Any],
    account_state: dict[str, Any] | None,
    lifecycle_state: dict[str, Any] | None = None,
    profile_lane_ids: set[str],
    scope: Scope = "profile",
) -> StrategyDeployability:
    issues: list[DeployabilityIssue] = []
    runtime_control = _runtime_control(
        row,
        lifecycle_state=lifecycle_state,
        profile_lane_ids=profile_lane_ids,
        scope=scope,
    )

    if row.get("status") != "active" or row.get("deployment_scope") != "deployable":
        issues.append(
            _issue("runtime_blocked", "hard", {"status": row.get("status"), "scope": row.get("deployment_scope")})
        )

    if runtime_control.get("evaluated"):
        if runtime_control.get("criterion12_valid") is False:
            issues.append(
                _issue(
                    "criterion12_invalid",
                    "hard",
                    {
                        "reason": runtime_control.get("criterion12_reason"),
                        "state_age_days": runtime_control.get("criterion12_state_age_days"),
                    },
                )
            )
        elif runtime_control.get("blocked"):
            issues.append(
                _issue(
                    "lifecycle_blocked",
                    "hard",
                    {
                        "block_source": runtime_control.get("block_source"),
                        "block_reason": runtime_control.get("block_reason"),
                    },
                )
            )
        elif runtime_control.get("sr_status") in (None, ""):
            issues.append(_issue("criterion12_missing_lane", "hard", runtime_control))
        elif runtime_control.get("sr_status") == "ALARM":
            if runtime_control.get("sr_review_outcome") == "watch":
                issues.append(_issue("sr_alarm_watch_reviewed", "warning", runtime_control))
            else:
                issues.append(_issue("sr_alarm_unreviewed", "hard", runtime_control))

    if row.get("filter_type") is None:
        issues.append(_issue("unknown_filter", "hard", "filter_type NULL"))
    if (
        row.get("entry_model") == "E2"
        and row.get("filter_type")
        and is_e2_deployment_unsafe_filter(str(row["filter_type"]))
    ):
        issues.append(_issue("e2_deployment_unsafe_filter", "hard", row["filter_type"]))

    if not replay.get("ok"):
        issues.append(_issue("replay_mismatch", "hard", replay))
    elif replay.get("expectancy_match") is False:
        issues.append(_issue("replay_expectancy_drift", "warning", replay))
    if replay.get("null_pnl_count"):
        issues.append(_issue("scratch_null_pnl_present", "warning", replay["null_pnl_count"]))

    if current_fdr.get("current_pass") is False:
        issues.append(_issue("current_k_fdr_fail", "hard", current_fdr))
    elif current_fdr.get("current_pass") is None:
        issues.append(_issue("current_k_fdr_missing", "hard", current_fdr))

    family = row.get("robustness_status")
    if family is None or family == "PURGED":
        issues.append(_issue("family_purged", "hard", family))
    elif family == "SINGLETON":
        # Stage 4 / Disposition C conditional downgrade. SINGLETON is
        # asymmetric vs PURGED: no peer-evidence, NOT a rejected family.
        # If the row clears all binding criteria from
        # `pre_registered_criteria.md` § Enforcement summary, downgrade
        # the issue to a warning and let it route via
        # CONTROLLED_PILOT_WARNINGS to CONTROLLED_LIVE_PILOT_CANDIDATE
        # (manual sign-off before lane allocation mutation).
        # Otherwise emit hard and let HARD_BLOCKER_TO_VERDICT route to
        # BLOCKED_FAMILY_FRAGILE.
        passes, failed_criteria, dsr_reported = _singleton_clears_binding_criteria(row)
        if passes:
            issues.append(
                _issue(
                    "family_singleton",
                    "warning",
                    {
                        "robustness_status": family,
                        "binding_criteria_cleared": "C3+C4+C6+C7+C9+C10",
                        "c5_dsr_reported": dsr_reported,
                        "c5_status": (
                            "computed_and_reported" if dsr_reported else "uncomputed_reporting_gap_cross_check_only"
                        ),
                        "verdict_route": "CONTROLLED_LIVE_PILOT_CANDIDATE",
                        "doctrine": (
                            "Stage 4 Disposition C — no peer-evidence is "
                            "tolerable when individual evidence clears the "
                            "locked binding criteria; supervised pilot before "
                            "full deploy. C5 is cross-check only per "
                            "pre_registered_criteria.md Amendment 2.1."
                        ),
                    },
                )
            )
        else:
            issues.append(
                _issue(
                    "family_singleton",
                    "hard",
                    {
                        "robustness_status": family,
                        "failed_binding_criteria": failed_criteria,
                        "c5_dsr_reported": dsr_reported,
                    },
                )
            )

    slip = row.get("slippage_validation_status")
    if slip in (None, ""):
        if _routine_tbbo_slippage_applies(row):
            issues.append(
                _issue(
                    "slippage_event_tail_pending",
                    "warning",
                    _controlled_slippage_event_tail_detail(row, slip, inferred=True),
                )
            )
        else:
            issues.append(_issue("slippage_missing", "hard", slip))
    elif _slippage_is_controlled_event_tail_pending(row, slip):
        issues.append(
            _issue(
                "slippage_event_tail_pending",
                "warning",
                _controlled_slippage_event_tail_detail(row, slip, inferred=False),
            )
        )
    elif not _slippage_passes(slip):
        issues.append(_issue("slippage_not_passed", "hard", slip))

    c8_status = c8.get("c8_oos_status") or row.get("c8_oos_status")
    if c8_status in (None, ""):
        issues.append(_issue("c8_missing", "hard", c8))
    elif c8_status != "PASSED":
        issues.append(_issue("c8_not_passed", "hard", c8))

    if row.get("wfe") is None:
        issues.append(_issue("wfe_missing", "warning", None))
    elif float(row["wfe"]) < MIN_WFE:
        issues.append(_issue("wfe_below_threshold", "hard", row["wfe"]))
    elif float(row["wfe"]) > 2.0:
        issues.append(_issue("wfe_over_amplified", "warning", row["wfe"]))

    if row.get("sample_size") is None or int(row["sample_size"]) < CORE_MIN_SAMPLES:
        issues.append(_issue("sample_size_below_deploy_threshold", "hard", row.get("sample_size")))

    if row.get("years_tested") is not None and float(row["years_tested"]) < 7:
        issues.append(_issue("short_history", "warning", row["years_tested"]))

    dsr = row.get("dsr_score")
    institutional_language_allowed = True
    if dsr is None:
        institutional_language_allowed = False
        issues.append(_issue("dsr_uncomputed_cross_check", "warning", None))
    elif float(dsr) < 0.95:
        institutional_language_allowed = False
        issues.append(_issue("dsr_below_cross_check", "warning", dsr))

    if scope == "all-active":
        if str(row["strategy_id"]) not in profile_lane_ids:
            issues.append(
                _issue(
                    "profile_not_evaluated",
                    "info",
                    "shelf audit only; run profile scope before deployment",
                )
            )
        elif account_state is not None and (not account_state.get("available") or not account_state.get("gate_ok")):
            issues.append(_issue("account_risk_fail", "hard", account_state))
    elif account_state is None:
        issues.append(_issue("account_risk_missing", "hard", "no profile supplied"))
    elif str(row["strategy_id"]) in profile_lane_ids:
        if not account_state.get("available") or not account_state.get("gate_ok"):
            issues.append(_issue("account_risk_fail", "hard", account_state))
    else:
        issues.append(
            _issue(
                "account_risk_missing",
                "hard",
                "not in selected profile; run portfolio MC for the proposed lane set",
            )
        )

    hard_ids = [issue.id for issue in issues if issue.severity == "hard"]
    controlled_warning_ids = {
        issue.id for issue in issues if issue.severity == "warning" and issue.id in CONTROLLED_PILOT_WARNINGS
    }
    if hard_ids:
        verdict = HARD_BLOCKER_TO_VERDICT.get(hard_ids[0], RESEARCH_PROVISIONAL)
    elif controlled_warning_ids:
        verdict = CONTROLLED_LIVE_PILOT_CANDIDATE
    else:
        verdict = DEPLOYABLE_CANDIDATE
    deployable = verdict in {DEPLOYABLE_CANDIDATE, CONTROLLED_LIVE_PILOT_CANDIDATE}
    return StrategyDeployability(
        strategy_id=str(row["strategy_id"]),
        instrument=row.get("instrument"),
        verdict=verdict,
        deployable=deployable,
        institutional_language_allowed=institutional_language_allowed and verdict == DEPLOYABLE_CANDIDATE,
        replay=replay,
        current_k_fdr=current_fdr,
        c8_oos=c8,
        trade_context=_trade_context(row),
        runtime_control=runtime_control,
        issues=issues,
        metrics={
            "sample_size": row.get("sample_size"),
            "expectancy_r": row.get("expectancy_r"),
            "oos_exp_r": row.get("oos_exp_r"),
            "wfe": row.get("wfe"),
            "dsr_score": row.get("dsr_score"),
            "years_tested": row.get("years_tested"),
            "family_status": row.get("robustness_status"),
            "slippage_validation_status": row.get("slippage_validation_status"),
            "c8_oos_status": row.get("c8_oos_status"),
        },
    )


def _counter_dict(values: list[Any]) -> dict[str, int]:
    return dict(Counter("<NULL>" if value in (None, "") else str(value) for value in values))


def _build_instrument_summary(strategy_reports: list[StrategyDeployability]) -> dict[str, Any]:
    by_instrument: dict[str, list[StrategyDeployability]] = {}
    for report in strategy_reports:
        by_instrument.setdefault(str(report.instrument or "<NULL>"), []).append(report)

    summary: dict[str, Any] = {}
    for instrument, reports in sorted(by_instrument.items()):
        hard_issues = [issue.id for report in reports for issue in report.issues if issue.severity == "hard"]
        current_fdr = [report.current_k_fdr for report in reports]
        replays = [report.replay for report in reports]
        summary[instrument] = {
            "total": len(reports),
            "deployable": sum(1 for report in reports if report.deployable),
            "institutional_language_allowed": sum(1 for report in reports if report.institutional_language_allowed),
            "verdict_counts": dict(Counter(report.verdict for report in reports)),
            "hard_issue_counts": dict(Counter(hard_issues)),
            "family_status_counts": _counter_dict([report.metrics.get("family_status") for report in reports]),
            "slippage_status_counts": _counter_dict(
                [report.metrics.get("slippage_validation_status") for report in reports]
            ),
            "c8_status_counts": _counter_dict([report.c8_oos.get("c8_oos_status") for report in reports]),
            "stored_c8_status_counts": _counter_dict([report.metrics.get("c8_oos_status") for report in reports]),
            "trade_role_counts": _counter_dict([report.trade_context.get("research_role") for report in reports]),
            "sample_class_counts": _counter_dict([report.trade_context.get("sample_class") for report in reports]),
            "sample_size_below_100": sum(
                1
                for report in reports
                if report.metrics.get("sample_size") is None or int(report.metrics["sample_size"]) < CORE_MIN_SAMPLES
            ),
            "current_k_fdr_fail": sum(1 for row in current_fdr if row.get("current_pass") is False),
            "current_k_fdr_missing": sum(1 for row in current_fdr if row.get("current_pass") is None),
            "replay_mismatch": sum(1 for row in replays if not row.get("ok")),
        }
    return summary


def _promotion_bucket(report: StrategyDeployability) -> str:
    hard_ids = {issue.id for issue in report.issues if issue.severity == "hard"}
    if report.deployable:
        return "deployable_now"
    if hard_ids & RETIRE_OR_PURGE_ISSUES:
        return "retire_or_purge"
    if hard_ids and hard_ids <= EVIDENCE_GAP_ISSUES and int(report.metrics.get("sample_size") or 0) >= 100:
        return "nearest_to_deployable"
    return "research_only"


def _build_promotion_queue(
    strategy_reports: list[StrategyDeployability], *, limit_per_bucket: int = 25
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "deployable_now": [],
        "nearest_to_deployable": [],
        "research_only": [],
        "retire_or_purge": [],
    }
    for report in strategy_reports:
        hard_ids = [issue.id for issue in report.issues if issue.severity == "hard"]
        buckets[_promotion_bucket(report)].append(
            {
                "strategy_id": report.strategy_id,
                "instrument": report.instrument,
                "verdict": report.verdict,
                "hard_issues": hard_ids,
                "sample_size": report.metrics.get("sample_size"),
                "expectancy_r": report.metrics.get("expectancy_r"),
                "family_status": report.metrics.get("family_status"),
                "c8_oos_status": report.c8_oos.get("c8_oos_status"),
                "slippage_validation_status": report.metrics.get("slippage_validation_status"),
            }
        )

    ordered: dict[str, Any] = {}
    for bucket, rows in buckets.items():
        rows.sort(
            key=lambda row: (
                str(row.get("instrument") or ""),
                -float(row.get("expectancy_r") or 0.0),
                str(row.get("strategy_id") or ""),
            )
        )
        ordered[bucket] = {
            "count": len(rows),
            "rows": rows[:limit_per_bucket],
            "truncated": max(0, len(rows) - limit_per_bucket),
        }
    return ordered


def build_deployability_audit(
    *,
    db_path: Path = GOLD_DB_PATH,
    scope: Scope = "all-active",
    profile_id: str | None = None,
    strict: bool = True,
    instruments: set[str] | None = None,
) -> dict[str, Any]:
    """Run the full-shelf deployability audit in read-only mode."""

    resolved_profile = (
        resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False) if profile_id else None
    )
    profile_lane_ids = (
        {str(lane["strategy_id"]) for lane in get_profile_lane_definitions(resolved_profile)}
        if resolved_profile
        else set()
    )
    lifecycle_state = read_lifecycle_state(resolved_profile, db_path=db_path) if resolved_profile else None
    account_state = lifecycle_state.get("criterion11") if lifecycle_state else None

    with duckdb.connect(str(db_path), read_only=True) as con:
        configure_connection(con)
        if not _has_table(con, "validated_setups"):
            raise RuntimeError("validated_setups table missing")
        rows = _load_candidate_rows(con, scope=scope, profile_id=resolved_profile, instruments=instruments)
        current_fdr = _current_k_fdr(con, rows)

    strategy_reports: list[StrategyDeployability] = []
    for row, replay in _iter_replay_results(db_path, rows):
        c8 = _evaluate_criterion_8_oos(row, db_path, strict_oos_n=str(row.get("validation_pathway")) == "individual")
        strategy_reports.append(
            _classify_strategy(
                row,
                replay=replay,
                current_fdr=current_fdr.get(str(row["strategy_id"]), {}),
                c8=c8,
                account_state=account_state if strict else None,
                lifecycle_state=lifecycle_state if strict else None,
                profile_lane_ids=profile_lane_ids,
                scope=scope,
            )
        )

    verdict_counts = Counter(report.verdict for report in strategy_reports)
    hard_issue_counts = Counter(
        issue.id for report in strategy_reports for issue in report.issues if issue.severity == "hard"
    )
    deployable_count = sum(1 for report in strategy_reports if report.deployable)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "db_path": str(db_path),
        "scope": scope,
        "profile_id": resolved_profile,
        "strict": strict,
        "instruments": sorted(instruments) if instruments else "ALL",
        "source_truth": {
            "candidate_source": "validated_setups as candidate list only",
            "replay_source": "orb_outcomes JOIN daily_features via trading_app.strategy_fitness._load_strategy_outcomes",
            "fdr_source": "experimental_strategies current full canonical session pool via trading_app.strategy_validator.benjamini_hochberg",
            "oos_source": "trading_app.strategy_validator._evaluate_criterion_8_oos",
            "account_source": "trading_app.lifecycle_state.read_lifecycle_state['criterion11']",
            "runtime_control_source": (
                "trading_app.lifecycle_state.read_lifecycle_state including Criterion 12 SR state "
                "and shadow opportunity awareness"
            ),
        },
        "resource_lit": {
            "multiple_testing": "docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md; docs/institutional/literature/harvey_liu_2015_backtesting.md",
            "dsr": "docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md",
            "lookahead": "docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md",
            "cost_realism": "docs/institutional/literature/carver_2015_ch12_speed_and_size.md",
            "prop_risk": "resources/prop-firm-official-rules.md",
        },
        "summary": {
            "total_candidates": len(strategy_reports),
            "deployable_candidates": deployable_count,
            "verdict_counts": dict(verdict_counts),
            "hard_issue_counts": dict(hard_issue_counts),
            "institutional_language_allowed": sum(1 for r in strategy_reports if r.institutional_language_allowed),
        },
        "profile_evaluation": {
            "mode": "strict_profile" if scope == "profile" else "shelf_only",
            "account_risk_is_deployment_gate": scope == "profile",
            "profile_lane_count": len(profile_lane_ids),
            "note": (
                "all-active scope labels non-profile rows as profile_not_evaluated instead of account failures"
                if scope == "all-active"
                else "profile scope fails closed on Criterion 11 for selected lanes"
            ),
        },
        "instrument_summary": _build_instrument_summary(strategy_reports),
        "promotion_queue": _build_promotion_queue(strategy_reports),
        "account_state": account_state,
        "opportunity_awareness": lifecycle_state.get("opportunity_awareness") if lifecycle_state else None,
        "strategies": [report.to_dict() for report in strategy_reports],
    }


def render_deployability_text(report: dict[str, Any], *, max_rows: int = 30) -> str:
    summary = report["summary"]
    lines = [
        f"Full-Shelf Deployability Audit | scope={report['scope']} | profile={report.get('profile_id')}",
        f"DB: {report['db_path']}",
        f"Instruments: {report.get('instruments')}",
        (
            "Summary: "
            f"total={summary['total_candidates']} "
            f"deployable={summary['deployable_candidates']} "
            f"institutional_language_allowed={summary['institutional_language_allowed']}"
        ),
        f"Verdicts: {summary['verdict_counts']}",
        f"Hard issues: {summary['hard_issue_counts']}",
        "Instrument summary:",
    ]
    opportunity_awareness = report.get("opportunity_awareness")
    if isinstance(opportunity_awareness, dict):
        status, detail = describe_opportunity_awareness(opportunity_awareness)
        lines.append(f"Opportunity awareness ({status}): {detail}")
    for instrument, row in report.get("instrument_summary", {}).items():
        lines.append(
            f"  - {instrument}: total={row['total']} deployable={row['deployable']} "
            f"verdicts={row['verdict_counts']} hard={row['hard_issue_counts']}"
        )
    lines.extend(
        [
            "Promotion queue:",
        ]
    )
    for bucket, bucket_data in report.get("promotion_queue", {}).items():
        lines.append(f"  - {bucket}: count={bucket_data['count']} truncated={bucket_data['truncated']}")
    lines.extend(
        [
            "Strategies:",
        ]
    )
    for row in report.get("strategies", [])[:max_rows]:
        hard = [i["id"] for i in row.get("issues", []) if i.get("severity") == "hard"]
        lines.append(f"  - {row['strategy_id']} | {row['verdict']} | hard={hard[:5]}")
    if len(report.get("strategies", [])) > max_rows:
        lines.append(f"  ... {len(report['strategies']) - max_rows} more")
    return "\n".join(lines)
