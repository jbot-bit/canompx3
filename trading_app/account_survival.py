"""Criterion 11 account-survival Monte Carlo for deployment profiles.

This module answers a deployment question, not a discovery question:
"Given the currently configured daily lanes for a profile, what is the
probability that one account survives the next 90 trading days under the
firm's risk rules?"

Design choices:
- Uses canonical strategy outcomes from `orb_outcomes` + `daily_features`
- Preserves cross-lane dependence by bootstrapping DAILY trade-path scenarios
- Replays conservative intraday low/high envelopes from timestamps + MAE/MFE
- Applies profile/firm rules (trailing DD, DLL, consistency, dynamic scaling)
- Does not mutate validated strategy truth or live deployment state
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from pipeline.cost_model import get_cost_spec, risk_in_dollars
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import apply_tight_stop
from trading_app.derived_state import build_profile_fingerprint
from trading_app.prop_profiles import (
    get_account_tier,
    get_firm_spec,
    get_profile,
    get_profile_lane_definitions,
    resolve_profile_id,
)
from trading_app.prop_firm_policies import get_payout_policy
from trading_app.strategy_fitness import _load_strategy_outcomes
from trading_app.topstep_scaling_plan import lots_for_position, max_lots_for_xfa

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
MIN_SURVIVAL_PROBABILITY = 0.70
DEFAULT_REPORT_MAX_AGE_DAYS = 30


@dataclass(frozen=True)
class SurvivalRules:
    """Account-level rules for one profile."""

    profile_id: str
    firm: str
    account_size: int
    dd_type: str
    starting_balance: float
    dd_limit_dollars: float
    daily_loss_limit: float | None
    consistency_rule: float | None
    freeze_at_balance: float | None
    contracts_per_trade_micro: int
    topstep_day1_max_lots: int | None


@dataclass(frozen=True)
class SurvivalSummary:
    """High-level Monte Carlo results for one profile."""

    profile_id: str
    generated_at_utc: str
    as_of_date: str
    horizon_days: int
    n_paths: int
    seed: int
    source_days: int
    source_start: str
    source_end: str
    dd_survival_probability: float
    operational_pass_probability: float
    consistency_pass_probability: float | None
    trailing_dd_breach_probability: float
    daily_loss_breach_probability: float
    scaling_breach_probability: float
    consistency_breach_probability: float
    scaling_feasible: bool
    intraday_approximated: bool
    path_model: str
    min_operational_pass_probability: float
    gate_pass: bool
    p50_final_balance: float
    p05_final_balance: float
    p95_final_balance: float
    p50_total_pnl: float
    p05_total_pnl: float
    p95_total_pnl: float
    p50_max_dd: float
    p95_max_dd: float
    median_best_day: float


@dataclass(frozen=True)
class DailyScenario:
    """One historical day of portfolio PnL for bootstrap sampling."""

    trading_day: str
    total_pnl_dollars: float
    positive_pnl_dollars: float
    active_lane_count: int
    min_balance_delta_dollars: float = 0.0
    max_balance_delta_dollars: float = 0.0
    max_open_lots: int = 0


@dataclass(frozen=True)
class TradePath:
    """Canonical per-trade path summary used for conservative intraday replay."""

    trading_day: date
    strategy_id: str
    entry_ts: datetime | None
    exit_ts: datetime | None
    pnl_dollars: float
    mae_dollars: float
    mfe_dollars: float
    lots: int


def get_survival_report_path(profile_id: str | None = None) -> Path:
    """Return the canonical Criterion 11 state path for one profile."""
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    return STATE_DIR / f"account_survival_{resolved_profile_id}.json"


def _build_profile_fingerprint(profile) -> str:
    """Backwards-compatible wrapper around the canonical shared helper."""
    return build_profile_fingerprint(profile)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _load_strategy_snapshot(con: duckdb.DuckDBPyConnection, strategy_id: str) -> dict:
    row = con.execute(
        """
        SELECT strategy_id, instrument, orb_label, orb_minutes,
               entry_model, rr_target, confirm_bars, filter_type,
               COALESCE(stop_multiplier, 1.0) AS stop_multiplier
        FROM validated_setups
        WHERE strategy_id = ?
        LIMIT 1
        """,
        [strategy_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"Strategy {strategy_id!r} missing from validated_setups")
    cols = [d[0] for d in con.description]
    return dict(zip(cols, row, strict=False))


def _load_lane_daily_pnl(
    con: duckdb.DuckDBPyConnection,
    strategy_id: str,
    *,
    as_of_date: date,
    effective_stop_multiplier: float | None = None,
) -> dict[date, float]:
    """Load one lane's historical daily PnL in dollars from canonical outcomes."""
    daily: dict[date, float] = {}
    for trade in _load_lane_trade_paths(
        con,
        strategy_id,
        as_of_date=as_of_date,
        effective_stop_multiplier=effective_stop_multiplier,
    ):
        daily[trade.trading_day] = daily.get(trade.trading_day, 0.0) + trade.pnl_dollars
    return daily


def _load_lane_trade_paths(
    con: duckdb.DuckDBPyConnection,
    strategy_id: str,
    *,
    as_of_date: date,
    effective_stop_multiplier: float | None = None,
) -> list[TradePath]:
    """Load one lane's trade history in dollars from canonical outcomes."""
    params = _load_strategy_snapshot(con, strategy_id)
    outcomes = _load_strategy_outcomes(
        con,
        instrument=params["instrument"],
        orb_label=params["orb_label"],
        orb_minutes=params["orb_minutes"],
        entry_model=params["entry_model"],
        rr_target=params["rr_target"],
        confirm_bars=params["confirm_bars"],
        filter_type=params["filter_type"],
        end_date=as_of_date,
    )

    stop_multiplier = (
        float(effective_stop_multiplier)
        if effective_stop_multiplier is not None
        else float(params.get("stop_multiplier") or 1.0)
    )
    if stop_multiplier != 1.0:
        cost_spec = get_cost_spec(params["instrument"])
        outcomes = apply_tight_stop(outcomes, stop_multiplier, cost_spec)

    cost_spec = get_cost_spec(params["instrument"])
    lots = lots_for_position(params["instrument"], 1)
    trades: list[TradePath] = []
    for outcome in outcomes:
        trading_day = outcome["trading_day"]
        if outcome.get("outcome") not in ("win", "loss"):
            continue
        entry_price = outcome.get("entry_price")
        stop_price = outcome.get("stop_price")
        pnl_r = outcome.get("pnl_r")
        if entry_price is None or stop_price is None or pnl_r is None:
            continue
        risk_dollars = risk_in_dollars(cost_spec, float(entry_price), float(stop_price))
        pnl_dollars = float(pnl_r) * risk_dollars
        mae_r = max(0.0, float(outcome.get("mae_r") or 0.0))
        mfe_r = max(0.0, float(outcome.get("mfe_r") or 0.0))
        trades.append(
            TradePath(
                trading_day=trading_day,
                strategy_id=strategy_id,
                entry_ts=outcome.get("entry_ts"),
                exit_ts=outcome.get("exit_ts"),
                pnl_dollars=float(pnl_dollars),
                mae_dollars=float(mae_r * risk_dollars),
                mfe_dollars=float(mfe_r * risk_dollars),
                lots=lots,
            )
        )
    return trades


def _scenario_from_trade_paths(trading_day: date, trades: list[TradePath]) -> DailyScenario:
    """Build one conservative daily replay scenario from per-trade paths."""
    if not trades:
        return DailyScenario(
            trading_day=str(trading_day),
            total_pnl_dollars=0.0,
            positive_pnl_dollars=0.0,
            active_lane_count=0,
            min_balance_delta_dollars=0.0,
            max_balance_delta_dollars=0.0,
            max_open_lots=0,
        )

    ordered_trades = sorted(
        trades,
        key=lambda trade: (
            trade.entry_ts or datetime.min.replace(tzinfo=UTC),
            trade.exit_ts or datetime.min.replace(tzinfo=UTC),
        ),
    )
    events: list[tuple[datetime, int, TradePath]] = []
    for trade in ordered_trades:
        entry_ts = trade.entry_ts or datetime.min.replace(tzinfo=UTC)
        exit_ts = trade.exit_ts or entry_ts
        events.append((entry_ts, 0, trade))
        events.append((exit_ts, 1, trade))
    events.sort(key=lambda item: (item[0], item[1]))

    realized_pnl = 0.0
    open_trades: list[TradePath] = []
    open_lots = 0
    max_open_lots = 0
    min_delta = 0.0
    max_delta = 0.0

    for _ts, event_type, trade in events:
        if event_type == 0:
            open_trades.append(trade)
            open_lots += trade.lots
            max_open_lots = max(max_open_lots, open_lots)
        else:
            realized_pnl += trade.pnl_dollars
            for idx, open_trade in enumerate(open_trades):
                if open_trade is trade:
                    del open_trades[idx]
                    break
            open_lots = max(0, open_lots - trade.lots)

        adverse_open = sum(open_trade.mae_dollars for open_trade in open_trades)
        favorable_open = sum(open_trade.mfe_dollars for open_trade in open_trades)
        min_delta = min(min_delta, realized_pnl - adverse_open)
        max_delta = max(max_delta, realized_pnl + favorable_open, realized_pnl)

    total_pnl = round(sum(trade.pnl_dollars for trade in trades), 2)
    return DailyScenario(
        trading_day=str(trading_day),
        total_pnl_dollars=total_pnl,
        positive_pnl_dollars=round(max(total_pnl, 0.0), 2),
        active_lane_count=len(trades),
        min_balance_delta_dollars=round(min_delta, 2),
        max_balance_delta_dollars=round(max_delta, 2),
        max_open_lots=max_open_lots,
    )


def _load_profile_daily_scenarios(
    profile_id: str,
    *,
    as_of_date: date,
    db_path: Path | None = None,
) -> tuple[list[DailyScenario], dict]:
    """Build historical daily portfolio scenarios for one profile."""
    db = db_path or GOLD_DB_PATH
    profile = get_profile(profile_id)
    lane_defs = get_profile_lane_definitions(profile_id)

    con = duckdb.connect(str(db), read_only=True)
    configure_connection(con)
    try:
        lane_first_days: dict[str, date] = {}
        trades_by_day: dict[date, list[TradePath]] = {}
        instruments = sorted({lane["instrument"] for lane in lane_defs})
        effective_stop_by_strategy = {
            lane.strategy_id: float(lane.planned_stop_multiplier or profile.stop_multiplier)
            for lane in profile.daily_lanes
        }

        for lane in lane_defs:
            trade_paths = _load_lane_trade_paths(
                con,
                lane["strategy_id"],
                as_of_date=as_of_date,
                effective_stop_multiplier=effective_stop_by_strategy.get(lane["strategy_id"]),
            )
            daily: dict[date, float] = {}
            for trade in trade_paths:
                daily[trade.trading_day] = daily.get(trade.trading_day, 0.0) + trade.pnl_dollars
                trades_by_day.setdefault(trade.trading_day, []).append(trade)
            if not daily:
                raise ValueError(f"Lane {lane['strategy_id']} has no canonical outcome history")
            lane_first_days[lane["strategy_id"]] = min(daily)

        common_start = max(lane_first_days.values())

        placeholders = ", ".join("?" for _ in instruments)
        calendar_days = [
            r[0]
            for r in con.execute(
                f"""
                SELECT DISTINCT trading_day
                FROM daily_features
                WHERE symbol IN ({placeholders})
                  AND trading_day >= ?
                  AND trading_day <= ?
                ORDER BY trading_day
                """,
                [*instruments, common_start, as_of_date],
            ).fetchall()
        ]

        scenarios: list[DailyScenario] = []
        for trading_day in calendar_days:
            scenarios.append(_scenario_from_trade_paths(trading_day, trades_by_day.get(trading_day, [])))

        if not scenarios:
            raise ValueError(f"Profile {profile_id!r} has no common-support daily scenarios")

        metadata = {
            "profile_id": profile.profile_id,
            "source_start": str(common_start),
            "source_end": str(as_of_date),
            "source_days": len(scenarios),
            "lane_ids": [lane["strategy_id"] for lane in lane_defs],
            "instruments": instruments,
            "profile_fingerprint": _build_profile_fingerprint(profile),
        }
        return scenarios, metadata
    finally:
        con.close()


def _build_rules(profile: AccountProfile) -> SurvivalRules:
    tier = get_account_tier(profile.firm, profile.account_size)
    firm_spec = get_firm_spec(profile.firm)
    starting_balance = 0.0 if profile.is_express_funded else float(profile.account_size)

    freeze_at = None
    if firm_spec.dd_type == "eod_trailing":
        if profile.is_express_funded:
            freeze_at = float(tier.max_dd + 100)
        else:
            freeze_at = float(profile.account_size + tier.max_dd + 100)

    topstep_day1_max_lots = None
    if profile.firm == "topstep" and profile.is_express_funded:
        topstep_day1_max_lots = max_lots_for_xfa(profile.account_size, starting_balance)

    return SurvivalRules(
        profile_id=profile.profile_id,
        firm=profile.firm,
        account_size=profile.account_size,
        dd_type=firm_spec.dd_type,
        starting_balance=starting_balance,
        dd_limit_dollars=float(tier.max_dd),
        daily_loss_limit=float(tier.daily_loss_limit) if tier.daily_loss_limit is not None else None,
        consistency_rule=None,
        freeze_at_balance=freeze_at,
        # Current project daily lanes are 1-contract micro lanes per account.
        contracts_per_trade_micro=1,
        topstep_day1_max_lots=topstep_day1_max_lots,
    )


def _with_consistency_rule(rules: SurvivalRules, profile: AccountProfile) -> SurvivalRules:
    consistency_rule = get_firm_spec(profile.firm).consistency_rule
    if profile.payout_policy_id is not None:
        consistency_rule = get_payout_policy(profile.payout_policy_id).consistency_rule
    return SurvivalRules(
        profile_id=rules.profile_id,
        firm=rules.firm,
        account_size=rules.account_size,
        dd_type=rules.dd_type,
        starting_balance=rules.starting_balance,
        dd_limit_dollars=rules.dd_limit_dollars,
        daily_loss_limit=rules.daily_loss_limit,
        consistency_rule=consistency_rule,
        freeze_at_balance=rules.freeze_at_balance,
        contracts_per_trade_micro=rules.contracts_per_trade_micro,
        topstep_day1_max_lots=rules.topstep_day1_max_lots,
    )


def simulate_survival(
    scenarios: list[DailyScenario],
    rules: SurvivalRules,
    *,
    horizon_days: int = 90,
    n_paths: int = 10_000,
    seed: int = 0,
) -> dict:
    """Run the profile survival Monte Carlo on conservative daily path scenarios."""
    if not scenarios:
        raise ValueError("At least one daily scenario is required")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    rng = random.Random(seed)
    best_day_list: list[float] = []
    final_balances: list[float] = []
    total_pnls: list[float] = []
    max_dds: list[float] = []

    dd_survivors = 0
    operational_survivors = 0
    consistency_passes = 0
    trailing_dd_breaches = 0
    daily_loss_breaches = 0
    scaling_breaches = 0
    consistency_breaches = 0

    for _ in range(n_paths):
        balance = rules.starting_balance
        hwm = rules.starting_balance
        frozen = bool(rules.freeze_at_balance is not None and hwm >= rules.freeze_at_balance)
        max_dd_used = 0.0
        total_pnl = 0.0
        positive_profit = 0.0
        best_day = 0.0
        breach_reason: str | None = None
        scaling_feasible = True

        for _day in range(horizon_days):
            scenario = scenarios[rng.randrange(len(scenarios))]
            if rules.topstep_day1_max_lots is not None:
                allowed_lots = max_lots_for_xfa(rules.account_size, max(balance, 0.0))
                if scenario.max_open_lots > allowed_lots:
                    breach_reason = "SCALING"
                    scaling_breaches += 1
                    scaling_feasible = False
                    break

            day_pnl = scenario.total_pnl_dollars
            total_pnl += day_pnl

            if day_pnl > 0:
                positive_profit += day_pnl
                if day_pnl > best_day:
                    best_day = day_pnl

            day_min_delta = min(scenario.min_balance_delta_dollars, day_pnl)
            day_max_delta = max(scenario.max_balance_delta_dollars, day_pnl)

            if rules.daily_loss_limit is not None and day_min_delta <= -rules.daily_loss_limit:
                breach_reason = "DAILY_LOSS"
                daily_loss_breaches += 1
                break

            day_low_balance = balance + day_min_delta
            dd_reference = hwm
            if rules.dd_type == "intraday_trailing":
                day_high_balance = balance + day_max_delta
                if not frozen and day_high_balance > dd_reference:
                    dd_reference = day_high_balance
            dd_used = max(0.0, dd_reference - day_low_balance)
            if dd_used > max_dd_used:
                max_dd_used = dd_used

            if dd_used >= rules.dd_limit_dollars:
                breach_reason = "TRAILING_DD"
                trailing_dd_breaches += 1
                break

            balance += day_pnl
            if not frozen and balance > hwm:
                hwm = balance
                if rules.freeze_at_balance is not None and hwm >= rules.freeze_at_balance:
                    frozen = True

        consistency_breach = False
        if rules.consistency_rule is not None and positive_profit > 0:
            consistency_breach = (best_day / positive_profit) > rules.consistency_rule
            if consistency_breach:
                consistency_breaches += 1

        dd_survived = breach_reason is None
        if dd_survived:
            dd_survivors += 1
        if not consistency_breach:
            consistency_passes += 1
        if dd_survived and not consistency_breach and scaling_feasible:
            operational_survivors += 1

        best_day_list.append(best_day)
        final_balances.append(balance)
        total_pnls.append(total_pnl)
        max_dds.append(max_dd_used)

    return {
        "dd_survival_probability": dd_survivors / n_paths,
        "operational_pass_probability": operational_survivors / n_paths,
        "consistency_pass_probability": (consistency_passes / n_paths) if rules.consistency_rule is not None else None,
        "trailing_dd_breach_probability": trailing_dd_breaches / n_paths,
        "daily_loss_breach_probability": daily_loss_breaches / n_paths,
        "scaling_breach_probability": scaling_breaches / n_paths,
        "consistency_breach_probability": consistency_breaches / n_paths,
        "scaling_feasible": scaling_breaches == 0,
        "intraday_approximated": False,
        "path_model": "trade_path_conservative",
        "p50_final_balance": round(_quantile(final_balances, 0.50), 2),
        "p05_final_balance": round(_quantile(final_balances, 0.05), 2),
        "p95_final_balance": round(_quantile(final_balances, 0.95), 2),
        "p50_total_pnl": round(_quantile(total_pnls, 0.50), 2),
        "p05_total_pnl": round(_quantile(total_pnls, 0.05), 2),
        "p95_total_pnl": round(_quantile(total_pnls, 0.95), 2),
        "p50_max_dd": round(_quantile(max_dds, 0.50), 2),
        "p95_max_dd": round(_quantile(max_dds, 0.95), 2),
        "median_best_day": round(_quantile(best_day_list, 0.50), 2),
    }


def evaluate_profile_survival(
    profile_id: str | None = None,
    *,
    as_of_date: date | None = None,
    horizon_days: int = 90,
    n_paths: int = 10_000,
    seed: int = 0,
    db_path: Path | None = None,
    write_state: bool = True,
    min_survival_probability: float = MIN_SURVIVAL_PROBABILITY,
) -> SurvivalSummary:
    """Evaluate one profile and optionally persist the latest report."""
    if as_of_date is None:
        as_of_date = date.today()
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    profile = get_profile(resolved_profile_id)
    scenarios, metadata = _load_profile_daily_scenarios(resolved_profile_id, as_of_date=as_of_date, db_path=db_path)
    metadata = {**metadata, "profile_fingerprint": _build_profile_fingerprint(profile)}
    rules = _with_consistency_rule(_build_rules(profile), profile)
    result = simulate_survival(scenarios, rules, horizon_days=horizon_days, n_paths=n_paths, seed=seed)
    operational_pass_probability = round(result["operational_pass_probability"], 4)
    gate_pass = operational_pass_probability >= float(min_survival_probability)

    summary = SurvivalSummary(
        profile_id=resolved_profile_id,
        generated_at_utc=datetime.now(UTC).isoformat(),
        as_of_date=str(as_of_date),
        horizon_days=horizon_days,
        n_paths=n_paths,
        seed=seed,
        source_days=metadata["source_days"],
        source_start=metadata["source_start"],
        source_end=metadata["source_end"],
        dd_survival_probability=round(result["dd_survival_probability"], 4),
        operational_pass_probability=operational_pass_probability,
        consistency_pass_probability=(
            round(result["consistency_pass_probability"], 4)
            if result["consistency_pass_probability"] is not None
            else None
        ),
        trailing_dd_breach_probability=round(result["trailing_dd_breach_probability"], 4),
        daily_loss_breach_probability=round(result["daily_loss_breach_probability"], 4),
        scaling_breach_probability=round(result["scaling_breach_probability"], 4),
        consistency_breach_probability=round(result["consistency_breach_probability"], 4),
        scaling_feasible=result["scaling_feasible"],
        intraday_approximated=result["intraday_approximated"],
        path_model=result["path_model"],
        min_operational_pass_probability=float(min_survival_probability),
        gate_pass=gate_pass,
        p50_final_balance=result["p50_final_balance"],
        p05_final_balance=result["p05_final_balance"],
        p95_final_balance=result["p95_final_balance"],
        p50_total_pnl=result["p50_total_pnl"],
        p05_total_pnl=result["p05_total_pnl"],
        p95_total_pnl=result["p95_total_pnl"],
        p50_max_dd=result["p50_max_dd"],
        p95_max_dd=result["p95_max_dd"],
        median_best_day=result["median_best_day"],
    )

    if write_state:
        out_path = get_survival_report_path(resolved_profile_id)
        payload = {
            "summary": asdict(summary),
            "rules": asdict(rules),
            "metadata": metadata,
        }
        out_path.write_text(json.dumps(payload, indent=2))

    return summary


def check_survival_report_gate(
    profile_id: str | None = None,
    *,
    today: date | None = None,
    min_survival_probability: float = MIN_SURVIVAL_PROBABILITY,
    max_age_days: int = DEFAULT_REPORT_MAX_AGE_DAYS,
) -> tuple[bool, str]:
    """Fail-closed gate for Criterion 11 deployment evidence."""
    if today is None:
        today = date.today()
    report_path = get_survival_report_path(profile_id)
    if not report_path.exists():
        return (
            False,
            "BLOCKED: no Criterion 11 survival report. "
            f"Run: python -m trading_app.account_survival --profile {report_path.stem.removeprefix('account_survival_')}",
        )

    try:
        payload = json.loads(report_path.read_text())
        summary = payload["summary"]
        metadata = payload["metadata"]
        as_of_date = date.fromisoformat(summary["as_of_date"])
        report_age_days = (today - as_of_date).days
        operational_pass = float(summary["operational_pass_probability"])
        horizon_days = int(summary["horizon_days"])
        n_paths = int(summary["n_paths"])
        scaling_feasible = bool(summary.get("scaling_feasible", False))
        intraday_approximated = bool(summary.get("intraday_approximated", False))
        path_model = str(summary.get("path_model", "daily_close"))
        profile_fingerprint = str(metadata["profile_fingerprint"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return False, f"BLOCKED: unreadable Criterion 11 survival report ({exc})"

    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    current_profile_fingerprint = _build_profile_fingerprint(get_profile(resolved_profile_id))

    if horizon_days != 90:
        return False, f"BLOCKED: Criterion 11 report horizon is {horizon_days}d, expected 90d"
    if n_paths < 10_000:
        return False, f"BLOCKED: Criterion 11 report uses {n_paths} paths, expected >= 10000"
    if profile_fingerprint != current_profile_fingerprint:
        return False, "BLOCKED: Criterion 11 report does not match the current profile lane/risk definition"
    if report_age_days > max_age_days:
        return (
            False,
            f"BLOCKED: Criterion 11 report is {report_age_days}d old (> {max_age_days}d). "
            "Re-run account survival.",
        )
    if path_model != "trade_path_conservative":
        return False, f"BLOCKED: Criterion 11 report uses unsupported path model {path_model!r}"
    if intraday_approximated:
        return False, "BLOCKED: Criterion 11 report is marked as unsupported intraday approximation"
    if not scaling_feasible:
        return False, "BLOCKED: Criterion 11 report fails scaling-feasibility check"
    if operational_pass < min_survival_probability:
        return (
            False,
            f"BLOCKED: Criterion 11 operational pass {operational_pass:.1%} < {min_survival_probability:.0%}",
        )

    return (
        True,
        f"Criterion 11 pass: operational {operational_pass:.1%}, as_of={as_of_date}, "
        f"age={report_age_days}d, paths={n_paths}",
    )


def _print_summary(summary: SurvivalSummary) -> None:
    print("=" * 100)
    print(f"ACCOUNT SURVIVAL | {summary.profile_id} | as_of={summary.as_of_date}")
    print("=" * 100)
    print(
        f"Horizon={summary.horizon_days}d | paths={summary.n_paths} | "
        f"source_days={summary.source_days} ({summary.source_start} -> {summary.source_end})"
    )
    print(
        f"Generated={summary.generated_at_utc} | "
        f"gate={'PASS' if summary.gate_pass else 'FAIL'} @ {summary.min_operational_pass_probability:.0%}"
    )
    print(
        f"DD survival={summary.dd_survival_probability:.1%} | "
        f"operational pass={summary.operational_pass_probability:.1%} | "
        f"consistency pass="
        + (
            f"{summary.consistency_pass_probability:.1%}"
            if summary.consistency_pass_probability is not None
            else "n/a"
        )
    )
    print(
        f"Breach rates: trailing_dd={summary.trailing_dd_breach_probability:.1%} | "
        f"daily_loss={summary.daily_loss_breach_probability:.1%} | "
        f"scaling={summary.scaling_breach_probability:.1%} | "
        f"consistency={summary.consistency_breach_probability:.1%}"
    )
    print(
        f"Final balance p05/p50/p95 = ${summary.p05_final_balance:,.0f} / "
        f"${summary.p50_final_balance:,.0f} / ${summary.p95_final_balance:,.0f}"
    )
    print(
        f"Total PnL p05/p50/p95 = ${summary.p05_total_pnl:,.0f} / "
        f"${summary.p50_total_pnl:,.0f} / ${summary.p95_total_pnl:,.0f}"
    )
    print(f"Max DD p50/p95 = ${summary.p50_max_dd:,.0f} / ${summary.p95_max_dd:,.0f}")
    print(f"Path model = {summary.path_model} | scaling feasible = {summary.scaling_feasible}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Criterion 11 account-survival Monte Carlo")
    parser.add_argument("--profile", default=None, help="Account profile id")
    parser.add_argument("--as-of", default=None, help="As-of date YYYY-MM-DD")
    parser.add_argument("--horizon-days", type=int, default=90, help="Simulation horizon in trading days")
    parser.add_argument("--paths", type=int, default=10_000, help="Number of Monte Carlo paths")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=MIN_SURVIVAL_PROBABILITY,
        help="Exit non-zero if operational pass is below this threshold",
    )
    parser.add_argument("--no-write-state", action="store_true", help="Do not persist latest report to data/state")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    summary = evaluate_profile_survival(
        profile_id=args.profile,
        as_of_date=as_of,
        horizon_days=args.horizon_days,
        n_paths=args.paths,
        seed=args.seed,
        write_state=not args.no_write_state,
        min_survival_probability=args.fail_under,
    )
    _print_summary(summary)
    if summary.operational_pass_probability < args.fail_under:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
