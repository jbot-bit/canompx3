#!/usr/bin/env python3
"""MNQ single-leg account-fit replacement audit.

Research-only execution of:
docs/audit/hypotheses/2026-06-02-mnq-single-leg-account-fit-replacement-v1.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.paths import GOLD_DB_PATH
from research.best_own_strategy_scan_v1 import _fmt
from research.filter_utils import filter_signal
from research.mnq_usdata_capital_fit_v1 import _max_drawdown, _simulate_survival
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.prop_profiles import get_account_tier, get_profile
from trading_app.topstep_scaling_plan import lots_for_position, max_lots_for_xfa

PROFILE_ID = "topstep_50k_mnq_auto"
SYMBOL = "MNQ"
EXPECTED_PRIMARY_TRIALS = 15
SURVIVAL_FLOOR = 0.70
DD_BUDGET_FRACTION = 0.80
HORIZON_DAYS = 90
N_PATHS = 10_000
SEED = 20_260_602
NO_THEORY_CHORDIA_T = 3.79
MIN_POWER_TRADES = 100

ALLOCATION_PATH = Path("docs/runtime/lane_allocation/topstep_50k_mnq_auto.json")
PREREG_PATH = Path("docs/audit/hypotheses/2026-06-02-mnq-single-leg-account-fit-replacement-v1.yaml")
RESULT_DIR = Path("docs/audit/results")
RESULT_DOC = RESULT_DIR / "2026-06-02-mnq-single-leg-account-fit-replacement-v1.md"
RESULT_CSV = RESULT_DIR / "2026-06-02-mnq-single-leg-account-fit-replacement-v1.csv"


@dataclass(frozen=True)
class CandidateLane:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    filter_name: str


@dataclass(frozen=True)
class IncumbentLane:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    filter_name: str
    chordia_verdict: str = "UNKNOWN"
    sr_status: str = "UNKNOWN"
    oos_status: str | None = None


@dataclass(frozen=True)
class ReplacementScenario:
    scenario_id: str
    replaced_incumbent: IncumbentLane
    candidate: CandidateLane
    lanes: tuple[IncumbentLane | CandidateLane, ...]


@dataclass(frozen=True)
class LaneResearchState:
    chordia_verdict: str = "MISSING"
    sr_status: str = "UNKNOWN"
    oos_status: str | None = None


LOCKED_CANDIDATES = (
    CandidateLane("MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10", "MNQ", "NYSE_OPEN", 15, 2.0, "COST_LT10"),
    CandidateLane("MNQ_US_DATA_1000_O15_E2_RR1_NO_FILTER", "MNQ", "US_DATA_1000", 15, 1.0, "NO_FILTER"),
    CandidateLane("MNQ_US_DATA_1000_O15_E2_RR1_5_NO_FILTER", "MNQ", "US_DATA_1000", 15, 1.5, "NO_FILTER"),
    CandidateLane("MNQ_US_DATA_1000_O15_E2_RR2_NO_FILTER", "MNQ", "US_DATA_1000", 15, 2.0, "NO_FILTER"),
    CandidateLane("MNQ_CME_PRECLOSE_O15_E2_RR2_COST_LT10", "MNQ", "CME_PRECLOSE", 15, 2.0, "COST_LT10"),
)


def _annual_per_drawdown(annual_dollars: float, max_dd_dollars: float) -> float:
    if not math.isfinite(annual_dollars) or not math.isfinite(max_dd_dollars) or max_dd_dollars <= 0:
        return float("nan")
    return annual_dollars / max_dd_dollars


def _sharpe(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return float("nan")
    std = float(vals.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(vals.mean() / std)


def _ttest(values: np.ndarray) -> tuple[float, float]:
    vals = values[np.isfinite(values)]
    if vals.size < 3 or float(vals.std(ddof=1)) <= 0:
        return float("nan"), float("nan")
    result = stats.ttest_1samp(vals, 0.0)
    return float(result.statistic), float(result.pvalue)


def parse_incumbent_lanes(path: Path = ALLOCATION_PATH) -> list[IncumbentLane]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("profile_id") != PROFILE_ID:
        raise ValueError(f"allocator profile_id mismatch: {payload.get('profile_id')!r}")
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 3:
        got = len(lanes) if isinstance(lanes, list) else "invalid"
        raise ValueError(f"expected exactly 3 active allocator lanes, got {got}")
    required = {"strategy_id", "instrument", "orb_label", "orb_minutes", "rr_target", "filter_type"}
    parsed: list[IncumbentLane] = []
    for row in lanes:
        if not isinstance(row, dict):
            raise ValueError("allocator lane row is not an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"allocator lane missing structured fields: {missing}")
        parsed.append(
            IncumbentLane(
                strategy_id=str(row["strategy_id"]),
                instrument=str(row["instrument"]),
                orb_label=str(row["orb_label"]),
                orb_minutes=int(row["orb_minutes"]),
                rr_target=float(row["rr_target"]),
                filter_name=str(row["filter_type"]),
                chordia_verdict=str(row.get("chordia_verdict") or "UNKNOWN"),
                sr_status=str(row.get("status") or row.get("session_regime") or "UNKNOWN"),
                oos_status=None if row.get("c8_oos_status") is None else str(row.get("c8_oos_status")),
            )
        )
    return parsed


def build_replacement_scenarios(
    candidates: list[CandidateLane] | tuple[CandidateLane, ...],
    incumbents: list[IncumbentLane] | tuple[IncumbentLane, ...],
) -> list[ReplacementScenario]:
    scenarios: list[ReplacementScenario] = []
    for incumbent_idx, incumbent in enumerate(incumbents, start=1):
        for candidate_idx, candidate in enumerate(candidates, start=1):
            lanes: list[IncumbentLane | CandidateLane] = [
                candidate if lane.strategy_id == incumbent.strategy_id else lane for lane in incumbents
            ]
            scenarios.append(
                ReplacementScenario(
                    scenario_id=f"R{incumbent_idx:02d}_C{candidate_idx:02d}",
                    replaced_incumbent=incumbent,
                    candidate=candidate,
                    lanes=tuple(lanes),
                )
            )
    return scenarios


def _status_lookup(path: Path = ALLOCATION_PATH) -> dict[str, LaneResearchState]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, LaneResearchState] = {}
    for section in ("lanes", "paused", "stale", "displaced"):
        for row in payload.get(section, []) or []:
            if not isinstance(row, dict) or "strategy_id" not in row:
                continue
            lookup[str(row["strategy_id"])] = LaneResearchState(
                chordia_verdict=str(row.get("chordia_verdict") or "MISSING"),
                sr_status=str(row.get("status") or row.get("session_regime") or "UNKNOWN"),
                oos_status=None if row.get("c8_oos_status") is None else str(row.get("c8_oos_status")),
            )
    return lookup


def _validate_prereg(path: Path = PREREG_PATH) -> None:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("prereg YAML did not load to an object")
    metadata = body.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("prereg metadata block missing")
    if metadata.get("theory_grant") is not False:
        raise ValueError("this no-theory audit requires metadata.theory_grant: false")
    if int(metadata.get("total_expected_trials", -1)) != EXPECTED_PRIMARY_TRIALS:
        raise ValueError("prereg K does not match EXPECTED_PRIMARY_TRIALS")
    prereg_ids = [str(row["id"]) for row in body.get("candidate_universe", {}).get("candidates", [])]
    locked_ids = [row.strategy_id for row in LOCKED_CANDIDATES]
    if prereg_ids != locked_ids:
        raise ValueError(f"locked candidate universe mismatch: {prereg_ids!r} != {locked_ids!r}")


def _load_base(con: duckdb.DuckDBPyConnection, lane: CandidateLane | IncumbentLane) -> pd.DataFrame:
    sql = """
        SELECT
            o.trading_day,
            o.entry_ts,
            o.exit_ts,
            o.outcome,
            COALESCE(o.pnl_r, o.ts_pnl_r, 0.0) AS pnl_r,
            (o.pnl_r IS NULL AND o.ts_pnl_r IS NULL) AS pnl_r_was_null,
            o.entry_price,
            o.stop_price,
            o.risk_dollars,
            d.*
        FROM orb_outcomes o
        INNER JOIN daily_features d
            ON o.symbol = d.symbol
           AND o.trading_day = d.trading_day
           AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.rr_target = ?
          AND o.confirm_bars = 1
          AND o.entry_model = 'E2'
        ORDER BY o.trading_day, o.entry_ts
    """
    df = con.execute(sql, [lane.instrument, lane.orb_label, lane.orb_minutes, lane.rr_target]).fetchdf()
    if df.empty:
        return df
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df["symbol"] = lane.instrument
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["risk_dollars"] = pd.to_numeric(df["risk_dollars"], errors="coerce").fillna(0.0)
    df["pnl_r"] = pd.to_numeric(df["pnl_r"], errors="coerce").fillna(0.0)
    return df


def canonical_filter_mask(df: pd.DataFrame, lane: CandidateLane | IncumbentLane) -> pd.Series:
    signal = filter_signal(df, lane.filter_name, lane.orb_label)
    return pd.Series(signal.astype(bool), index=df.index)


def load_lane_trades(
    con: duckdb.DuckDBPyConnection,
    lane: CandidateLane | IncumbentLane,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _load_base(con, lane)
    if base.empty:
        empty = pd.DataFrame(
            columns=["trading_day", "entry_ts", "pnl_r", "pnl_dollars", "risk_dollars", "pnl_r_was_null", "lane"]
        )
        return base, empty
    filtered = base[canonical_filter_mask(base, lane).fillna(False)].copy()
    filtered["pnl_dollars"] = filtered["pnl_r"].astype(float) * filtered["risk_dollars"].astype(float)
    filtered["lane"] = lane.strategy_id
    return base, filtered[
        ["trading_day", "entry_ts", "pnl_r", "pnl_dollars", "risk_dollars", "pnl_r_was_null", "lane"]
    ].copy()


def build_daily_book(calendar: pd.Series | pd.DatetimeIndex | list[Any], legs: list[pd.DataFrame]) -> pd.DataFrame:
    days = sorted(pd.to_datetime(pd.Series(calendar)).dt.date.dropna().unique())
    book = pd.DataFrame({"trading_day": days})
    book["pnl_r"] = 0.0
    book["pnl_dollars"] = 0.0
    book["active_trades"] = 0
    for leg in legs:
        if leg.empty:
            continue
        grouped = (
            leg.groupby("trading_day", as_index=False)
            .agg(pnl_r=("pnl_r", "sum"), pnl_dollars=("pnl_dollars", "sum"), active_trades=("pnl_r", "count"))
            .copy()
        )
        book = book.merge(grouped, on="trading_day", how="left", suffixes=("", "_leg"))
        book["pnl_r"] += book.pop("pnl_r_leg").fillna(0.0)
        book["pnl_dollars"] += book.pop("pnl_dollars_leg").fillna(0.0)
        book["active_trades"] += book.pop("active_trades_leg").fillna(0).astype(int)
    return book


def _calendar_for_lanes(
    lanes: list[IncumbentLane | CandidateLane],
    base_by_lane: dict[str, pd.DataFrame],
) -> pd.Series:
    calendars = [
        base_by_lane[lane.strategy_id]["trading_day"] for lane in lanes if not base_by_lane[lane.strategy_id].empty
    ]
    if not calendars:
        return pd.Series([], dtype="datetime64[ns]")
    return pd.concat(calendars, ignore_index=True)


def _score_daily_book(book: pd.DataFrame) -> dict[str, float | int]:
    is_book = book[book["trading_day"] < HOLDOUT_SACRED_FROM].copy()
    monitor = book[book["trading_day"] >= HOLDOUT_SACRED_FROM].copy()
    vals_dollars = is_book["pnl_dollars"].to_numpy(dtype=float)
    vals_r = is_book["pnl_r"].to_numpy(dtype=float)
    active = is_book[is_book["active_trades"] > 0]
    annual = float(np.nanmean(vals_dollars) * 252.0) if vals_dollars.size else float("nan")
    max_dd = _max_drawdown(vals_dollars)
    t_stat, p_value = _ttest(vals_r)
    return {
        "n_is_days": int(len(is_book)),
        "active_trade_days_is": int(len(active)),
        "n_is_trades": int(is_book["active_trades"].sum()) if len(is_book) else 0,
        "expected_r_after_costs": float(np.nanmean(vals_r)) if vals_r.size else float("nan"),
        "annual_dollars": annual,
        "max_drawdown_dollars": max_dd,
        "annual_dollars_per_max_drawdown": _annual_per_drawdown(annual, max_dd),
        "win_rate": float((active["pnl_dollars"] > 0).mean()) if len(active) else float("nan"),
        "sharpe": _sharpe(vals_r),
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_2026_dollars": float(monitor["pnl_dollars"].mean()) if len(monitor) else float("nan"),
        "mean_2026_r": float(monitor["pnl_r"].mean()) if len(monitor) else float("nan"),
    }


def _simulate_survival_local(
    values: np.ndarray,
    *,
    n_paths: int,
    seed: int,
    dd_limit: float,
    daily_loss_limit: float,
    freeze_at_balance: float,
) -> dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {"operational_survival": 0.0, "trailing_dd_breach": 1.0, "daily_loss_breach": 1.0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(n_paths, HORIZON_DAYS), replace=True)
    operational = 0
    dd_breaches = 0
    daily_breaches = 0
    for path in samples:
        balance = 0.0
        hwm = 0.0
        breached = False
        for day_pnl in path:
            if day_pnl <= -daily_loss_limit:
                daily_breaches += 1
                breached = True
                break
            low_balance = balance + min(float(day_pnl), 0.0)
            if hwm - low_balance >= dd_limit:
                dd_breaches += 1
                breached = True
                break
            balance += float(day_pnl)
            hwm = min(max(hwm, balance), freeze_at_balance)
        if not breached:
            operational += 1
    return {
        "operational_survival": operational / n_paths,
        "trailing_dd_breach": dd_breaches / n_paths,
        "daily_loss_breach": daily_breaches / n_paths,
    }


def _research_status_ok(state: LaneResearchState) -> bool:
    chordia = state.chordia_verdict.upper()
    sr = state.sr_status.upper()
    oos = "" if state.oos_status is None else state.oos_status.upper()
    return (
        chordia in {"PASS_CHORDIA", "PASS_PROTOCOL_A"}
        and "ALARM" not in sr
        and "FAIL" not in oos
        and "NEGATIVE" not in oos
    )


def _research_status_missing_or_ambiguous(state: LaneResearchState) -> bool:
    chordia = state.chordia_verdict.upper()
    sr = state.sr_status.upper()
    return chordia in {"", "MISSING", "UNKNOWN", "NONE"} or sr in {"", "UNKNOWN", "MISSING", "NONE"}


def _verdict_rank(verdict: str) -> int:
    return {"PASS": 0, "CONTINUE": 1, "PARK": 2, "KILL": 3}.get(verdict, 4)


def score_replacement_scenario(
    *,
    scenario_id: str,
    replaced_incumbent_lane: str,
    candidate_lane: str,
    book: pd.DataFrame,
    trades: pd.DataFrame,
    incumbent_annual_per_max_dd: float,
    research_state: LaneResearchState,
    monte_carlo_paths: int = N_PATHS,
    monte_carlo_seed: int = SEED,
) -> dict[str, Any]:
    _ = trades
    profile = get_profile(PROFILE_ID)
    tier = get_account_tier(profile.firm, profile.account_size)
    dd_limit = float(tier.max_dd)
    daily_loss_limit = float(profile.daily_loss_dollars or tier.daily_loss_limit or dd_limit)
    freeze_at_balance = dd_limit + 100.0 if profile.is_express_funded else profile.account_size + dd_limit + 100.0
    day1_lots = max_lots_for_xfa(profile.account_size, 0.0) if profile.is_express_funded else math.inf
    # Score the full locked calendar so the report can expose 2026 monitoring
    # metrics, but keep account-safety / promotion gates strictly in-sample.
    # The 2026+ sacred holdout is for monitoring only and must not influence
    # replacement verdicts.
    metrics = _score_daily_book(book)
    in_sample_book = book[book["trading_day"] < HOLDOUT_SACRED_FROM].copy()
    vals = in_sample_book["pnl_dollars"].to_numpy(dtype=float)
    hist_daily_breaches = int(np.sum(vals <= -daily_loss_limit)) if vals.size else 0
    if monte_carlo_paths == N_PATHS:
        survival = _simulate_survival(
            vals,
            contracts_per_leg=1,
            dd_limit=dd_limit,
            daily_loss_limit=daily_loss_limit,
            freeze_at_balance=freeze_at_balance,
        )
    else:
        survival = _simulate_survival_local(
            vals,
            n_paths=monte_carlo_paths,
            seed=monte_carlo_seed,
            dd_limit=dd_limit,
            daily_loss_limit=daily_loss_limit,
            freeze_at_balance=freeze_at_balance,
        )
    annual_per_dd = float(metrics["annual_dollars_per_max_drawdown"])
    improves = math.isfinite(annual_per_dd) and annual_per_dd > incumbent_annual_per_max_dd
    scaling_ok = lots_for_position(SYMBOL, 1) <= day1_lots
    account_safe = (
        scaling_ok
        and float(metrics["annual_dollars"]) > 0.0
        and float(metrics["expected_r_after_costs"]) > 0.0
        and math.isfinite(float(metrics["max_drawdown_dollars"]))
        and float(metrics["max_drawdown_dollars"]) <= dd_limit * DD_BUDGET_FRACTION
        and hist_daily_breaches == 0
        and survival["operational_survival"] >= SURVIVAL_FLOOR
        and survival["trailing_dd_breach"] == 0.0
        and survival["daily_loss_breach"] == 0.0
    )
    underpowered = int(metrics["n_is_trades"]) < MIN_POWER_TRADES
    chordia_severity = (
        "PASS"
        if math.isfinite(float(metrics["t_stat"])) and float(metrics["t_stat"]) >= NO_THEORY_CHORDIA_T
        else "MISS"
    )
    if not account_safe or not improves:
        verdict = "KILL"
    elif underpowered:
        verdict = "PARK"
    elif _research_status_ok(research_state):
        verdict = "PASS"
    elif _research_status_missing_or_ambiguous(research_state):
        verdict = "PARK"
    else:
        verdict = "CONTINUE"
    return {
        "scenario_id": scenario_id,
        "replaced_incumbent_lane": replaced_incumbent_lane,
        "candidate_lane": candidate_lane,
        **metrics,
        "historical_daily_loss_breaches": hist_daily_breaches,
        "ninety_day_account_survival": survival["operational_survival"],
        "daily_loss_breach_rate": survival["daily_loss_breach"],
        "trailing_drawdown_breach_rate": survival["trailing_dd_breach"],
        "scaling_ok": scaling_ok,
        "account_safe": account_safe,
        "improves_incumbent": improves,
        "incumbent_annual_dollars_per_max_drawdown": incumbent_annual_per_max_dd,
        "chordia_severity": chordia_severity,
        "chordia_verdict": research_state.chordia_verdict,
        "sr_regime_status": research_state.sr_status,
        "oos_status": research_state.oos_status or "UNKNOWN",
        "verdict": verdict,
    }


def run(db_path: Path = GOLD_DB_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_prereg()
    incumbents = parse_incumbent_lanes()
    scenarios = build_replacement_scenarios(LOCKED_CANDIDATES, incumbents)
    if len(scenarios) != EXPECTED_PRIMARY_TRIALS:
        raise ValueError(f"scenario K mismatch: {len(scenarios)}")
    status_lookup = _status_lookup()
    unique_lanes: dict[str, IncumbentLane | CandidateLane] = {lane.strategy_id: lane for lane in incumbents}
    unique_lanes.update({lane.strategy_id: lane for lane in LOCKED_CANDIDATES})
    base_by_lane: dict[str, pd.DataFrame] = {}
    trades_by_lane: dict[str, pd.DataFrame] = {}
    with duckdb.connect(str(db_path), read_only=True) as con:
        for lane in unique_lanes.values():
            base, trades = load_lane_trades(con, lane)
            base_by_lane[lane.strategy_id] = base
            trades_by_lane[lane.strategy_id] = trades
    incumbent_calendar = _calendar_for_lanes(incumbents, base_by_lane)
    incumbent_book = build_daily_book(incumbent_calendar, [trades_by_lane[lane.strategy_id] for lane in incumbents])
    incumbent_metrics = _score_daily_book(incumbent_book)
    incumbent_objective = float(incumbent_metrics["annual_dollars_per_max_drawdown"])
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_calendar = _calendar_for_lanes(scenario.lanes, base_by_lane)
        book = build_daily_book(scenario_calendar, [trades_by_lane[lane.strategy_id] for lane in scenario.lanes])
        trades = pd.concat([trades_by_lane[lane.strategy_id] for lane in scenario.lanes], ignore_index=True)
        rows.append(
            score_replacement_scenario(
                scenario_id=scenario.scenario_id,
                replaced_incumbent_lane=scenario.replaced_incumbent.strategy_id,
                candidate_lane=scenario.candidate.strategy_id,
                book=book,
                trades=trades,
                incumbent_annual_per_max_dd=incumbent_objective,
                research_state=status_lookup.get(scenario.candidate.strategy_id, LaneResearchState()),
            )
        )
    results = pd.DataFrame(rows)
    results["_verdict_rank"] = results["verdict"].map(_verdict_rank)
    results = results.sort_values(
        [
            "_verdict_rank",
            "account_safe",
            "improves_incumbent",
            "ninety_day_account_survival",
            "annual_dollars_per_max_drawdown",
        ],
        ascending=[True, False, False, False, False],
    ).drop(columns=["_verdict_rank"])
    meta = {
        "incumbents": incumbents,
        "incumbent_metrics": incumbent_metrics,
        "incumbent_account_safe": _baseline_account_safe(incumbent_book),
    }
    return results.reset_index(drop=True), meta


def _baseline_account_safe(book: pd.DataFrame) -> dict[str, Any]:
    profile = get_profile(PROFILE_ID)
    tier = get_account_tier(profile.firm, profile.account_size)
    vals = book[book["trading_day"] < HOLDOUT_SACRED_FROM]["pnl_dollars"].to_numpy(dtype=float)
    daily_loss_limit = float(profile.daily_loss_dollars or tier.daily_loss_limit or tier.max_dd)
    survival = _simulate_survival(
        vals,
        contracts_per_leg=1,
        dd_limit=float(tier.max_dd),
        daily_loss_limit=daily_loss_limit,
        freeze_at_balance=float(tier.max_dd) + 100.0,
    )
    return {
        "historical_daily_loss_breaches": int(np.sum(vals <= -daily_loss_limit)) if vals.size else 0,
        "max_drawdown_dollars": _max_drawdown(vals),
        "ninety_day_account_survival": survival["operational_survival"],
        "daily_loss_breach_rate": survival["daily_loss_breach"],
        "trailing_drawdown_breach_rate": survival["trailing_dd_breach"],
    }


def _write_table(rows: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    align = (
        "| "
        + " | ".join("---" if col.endswith("lane") or col in {"scenario_id", "verdict"} else "---:" for col in columns)
        + " |"
    )
    lines = [header, align]
    for _, row in rows.iterrows():
        parts: list[str] = []
        for col in columns:
            value = row[col]
            if col.endswith("lane") or col in {"scenario_id", "verdict"}:
                parts.append(f"`{value}`")
            elif isinstance(value, (bool, np.bool_)):
                parts.append(str(bool(value)))
            elif isinstance(value, (int, np.integer)):
                parts.append(str(int(value)))
            else:
                parts.append(_fmt(value, 4))
        lines.append("| " + " | ".join(parts) + " |")
    return lines


def _final_verdict_text(results: pd.DataFrame) -> str:
    if (results["verdict"] == "PASS").any():
        return "`PASS`: at least one replacement scenario is account-safe, improves the incumbent, and has no missing research-status gate. This remains research-only until a separate deployment translation."
    if (results["verdict"] == "CONTINUE").any():
        return "`CONTINUE`: at least one replacement scenario is account-safe and improves the incumbent, but it is not deployable from this artifact."
    if (results["verdict"] == "PARK").any():
        return "`PARK`: at least one replacement scenario is promising but underpowered or ambiguous."
    return "`KILL`: every replacement scenario is account-unsafe, non-economic, or worse than the incumbent comparator."


def write_report(results: pd.DataFrame, meta: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_CSV, index=False)
    profile = get_profile(PROFILE_ID)
    tier = get_account_tier(profile.firm, profile.account_size)
    incumbent_metrics = meta["incumbent_metrics"]
    baseline_account = meta["incumbent_account_safe"]
    best = results.iloc[0]
    verdict_counts = results["verdict"].value_counts().to_dict()
    cols = [
        "scenario_id",
        "replaced_incumbent_lane",
        "candidate_lane",
        "n_is_trades",
        "annual_dollars",
        "max_drawdown_dollars",
        "annual_dollars_per_max_drawdown",
        "expected_r_after_costs",
        "win_rate",
        "t_stat",
        "mean_2026_dollars",
        "mean_2026_r",
        "ninety_day_account_survival",
        "daily_loss_breach_rate",
        "trailing_drawdown_breach_rate",
        "historical_daily_loss_breaches",
        "chordia_verdict",
        "sr_regime_status",
        "oos_status",
        "verdict",
    ]
    lines = [
        "# MNQ Single-Leg Account-Fit Replacement v1",
        "",
        f"**Prereg:** `{PREREG_PATH}`",
        "**Status:** bounded allocator replacement audit; research-only; no deployment claim.",
        f"**Family K:** `{EXPECTED_PRIMARY_TRIALS}` selectable replacement scenarios.",
        "**Selection window:** `< 2026-01-01`; 2026 rows are monitoring only.",
        f"**Profile:** `{PROFILE_ID}`; daily belt `${float(profile.daily_loss_dollars or 0):.0f}`, max DD `${float(tier.max_dd):.0f}`, DD budget `${float(tier.max_dd) * DD_BUDGET_FRACTION:.0f}`.",
        "",
        "## Disconfirming Checks",
        "",
        "[MEASURED] Current incumbent lanes were loaded from structured allocator JSON fields, not parsed from strategy IDs.",
        "[MEASURED] The locked universe stayed at five candidates crossed with three incumbent replacement slots.",
        "[MEASURED] The runner used read-only canonical `orb_outcomes` plus `daily_features` and canonical filter delegation through `research.filter_utils.filter_signal`.",
        "[UNSUPPORTED] No scenario may be described as deployed, live-valid, validated, or OOS-clean from this artifact.",
        "",
        "## Incumbent Comparator",
        "",
        f"- Annual dollars: `${_fmt(incumbent_metrics['annual_dollars'], 2)}`",
        f"- Max DD dollars: `${_fmt(incumbent_metrics['max_drawdown_dollars'], 2)}`",
        f"- Annual dollars / max DD: `{_fmt(incumbent_metrics['annual_dollars_per_max_drawdown'])}`",
        f"- 90-day survival: `{_fmt(baseline_account['ninety_day_account_survival'])}`",
        f"- Daily loss breach rate: `{_fmt(baseline_account['daily_loss_breach_rate'])}`",
        f"- Trailing DD breach rate: `{_fmt(baseline_account['trailing_drawdown_breach_rate'])}`",
        "",
        "## Result",
        "",
        f"[MEASURED] Best ranked scenario is `{best['scenario_id']}` replacing `{best['replaced_incumbent_lane']}` with `{best['candidate_lane']}`, verdict `{best['verdict']}`.",
        f"[MEASURED] Verdict counts: `{verdict_counts}`.",
        "",
        "## Scenario Ranking",
        "",
        *_write_table(results, cols),
        "",
        "## Grounding",
        "",
        "- `docs/institutional/pre_registered_criteria.md` Criterion 11 requires account-death Monte Carlo with prop-firm daily loss/trailing DD rules and 90-day survival >= 70% before funded deployment.",
        "- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` grounds the no-theory t-stat severity benchmark at 3.79.",
        "- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` grounds inherited multiple-testing caution; this account audit does not reset alpha-discovery K.",
        "- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` grounds the no-OOS-tuning rule; 2026 is monitoring only.",
        "",
        "## Caveats",
        "",
        "- Account simulation uses daily-close PnL aggregation, matching the prior account-fit runners; it does not replay intraday path ordering.",
        "- Missing candidate Chordia/SR/OOS status is reported and prevents deployment language.",
        "- The result does not edit live allocation, prop profiles, or runtime config.",
        "",
        "## Verdict",
        "",
        _final_verdict_text(results),
        "",
        "SURVIVED SCRUTINY: locked K=15, structured incumbent parsing, canonical filters, read-only canonical DB, Topstep account gates, no 2026 tuning.",
        "DID NOT SURVIVE: no live deployment claim from this research artifact.",
    ]
    RESULT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()
    results, meta = run(args.db)
    write_report(results, meta)
    print(f"wrote {len(results)} scenarios to {RESULT_CSV}")
    print(f"wrote report to {RESULT_DOC}")


if __name__ == "__main__":
    main()
