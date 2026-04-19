"""Pre-deploy trade-level correlation gate for profit expansion.

Prevents deploying a candidate lane that is trade-level-redundant with an
existing deployed lane. Catches subset relationships (e.g., COST_LT12 ⊂ ORB_G5)
that family_hash metadata does not detect.

@canonical-source trading_app/lane_correlation.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.prop_profiles import get_profile_lane_definitions
from trading_app.strategy_fitness import _load_strategy_outcomes

RHO_REJECT_THRESHOLD = 0.70
SUBSET_REJECT_THRESHOLD = 0.80


@dataclass(frozen=True)
class PairResult:
    candidate_id: str
    deployed_id: str
    shared_days: int
    candidate_days: int
    deployed_days: int
    subset_coverage: float
    pearson_rho: float
    reject: bool
    reason: str


@dataclass(frozen=True)
class CorrelationReport:
    candidate_id: str
    profile_id: str
    pairs: tuple[PairResult, ...]
    gate_pass: bool
    worst_rho: float
    worst_subset: float
    reject_reasons: tuple[str, ...]


def _daily_pnl(outcomes: list[dict]) -> dict[date, float]:
    by_day: dict[date, float] = defaultdict(float)
    for o in outcomes:
        if o.get("pnl_r") is not None:
            by_day[o["trading_day"]] += float(o["pnl_r"])
    return dict(by_day)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 5:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _load_lane_daily_pnl(
    con: duckdb.DuckDBPyConnection,
    lane: dict,
) -> dict[date, float]:
    outcomes = _load_strategy_outcomes(
        con,
        instrument=lane["instrument"],
        orb_label=lane["orb_label"],
        orb_minutes=lane["orb_minutes"],
        entry_model=lane["entry_model"],
        rr_target=lane["rr_target"],
        confirm_bars=lane["confirm_bars"],
        filter_type=lane["filter_type"],
    )
    return _daily_pnl(outcomes)


def check_candidate_correlation(
    candidate_lane: dict,
    profile_id: str,
    *,
    rho_threshold: float = RHO_REJECT_THRESHOLD,
    subset_threshold: float = SUBSET_REJECT_THRESHOLD,
    con: duckdb.DuckDBPyConnection | None = None,
) -> CorrelationReport:
    own_con = con is None
    if own_con:
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        configure_connection(con)

    try:
        candidate_pnl = _load_lane_daily_pnl(con, candidate_lane)
        deployed_lanes = get_profile_lane_definitions(profile_id)
        pairs: list[PairResult] = []

        for dlane in deployed_lanes:
            deployed_pnl = _load_lane_daily_pnl(con, dlane)
            shared = sorted(set(candidate_pnl) & set(deployed_pnl))
            n_shared = len(shared)
            n_cand = len(candidate_pnl)
            n_dep = len(deployed_pnl)
            smaller = min(n_cand, n_dep)
            subset_cov = n_shared / smaller if smaller > 0 else 0.0

            if n_shared >= 5:
                xs = [candidate_pnl[d] for d in shared]
                ys = [deployed_pnl[d] for d in shared]
                rho = _pearson(xs, ys)
            else:
                rho = 0.0

            same_session = candidate_lane.get("orb_label") == dlane.get("orb_label") and candidate_lane.get(
                "instrument"
            ) == dlane.get("instrument")

            reasons = []
            if rho > rho_threshold:
                reasons.append(f"rho={rho:.3f}>{rho_threshold}")
            if same_session and subset_cov > subset_threshold:
                reasons.append(f"subset={subset_cov:.1%}>{subset_threshold:.0%}")

            pairs.append(
                PairResult(
                    candidate_id=candidate_lane.get("strategy_id", "?"),
                    deployed_id=dlane["strategy_id"],
                    shared_days=n_shared,
                    candidate_days=n_cand,
                    deployed_days=n_dep,
                    subset_coverage=subset_cov,
                    pearson_rho=rho,
                    reject=bool(reasons),
                    reason="; ".join(reasons) if reasons else "OK",
                )
            )

        reject_reasons = tuple(f"{p.deployed_id}: {p.reason}" for p in pairs if p.reject)
        worst_rho = max((p.pearson_rho for p in pairs), default=0.0)
        worst_subset = max((p.subset_coverage for p in pairs), default=0.0)

        return CorrelationReport(
            candidate_id=candidate_lane.get("strategy_id", "?"),
            profile_id=profile_id,
            pairs=tuple(pairs),
            gate_pass=len(reject_reasons) == 0,
            worst_rho=worst_rho,
            worst_subset=worst_subset,
            reject_reasons=reject_reasons,
        )
    finally:
        if own_con:
            con.close()
