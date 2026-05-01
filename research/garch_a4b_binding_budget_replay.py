"""Stage 1 replay for the locked A4b binding-budget allocator hypothesis.

Purpose:
  Test whether a locked garch-aware ranking term improves the canonical
  shelf-level allocator on a genuinely binding `max_slots=5` surface.

Design notes:
  - Baseline and candidate share the same universe, DD gate, correlation gate,
    hysteresis, and monthly rebalance cadence.
  - The only intended difference is the candidate ranking term.
  - The baseline implementation is audited every rebalance against
    `trading_app.lane_allocator.build_allocation()` so the research harness
    cannot silently drift away from the canonical comparator.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pipeline.cost_model import COST_SPECS, get_cost_spec
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.account_survival import _load_lane_trade_paths as _unused  # noqa: F401
from trading_app.config import apply_tight_stop
from trading_app.lane_allocator import (
    CORRELATION_REJECT_RHO,
    HYSTERESIS_PCT,
    LaneScore,
    _classify_status,
    _effective_annual_r,
    _month_range,
    build_allocation,
    compute_orb_size_stats,
)
from trading_app.lane_correlation import _pearson
from trading_app.prop_profiles import ACCOUNT_PROFILES, ACCOUNT_TIERS
from trading_app.strategy_fitness import _load_strategy_outcomes
from trading_app.validated_shelf import deployable_validated_relation

HYPOTHESIS_FILE = "docs/audit/hypotheses/2026-04-17-garch-a4b-binding-budget.yaml"
OUTPUT_MD = Path("docs/audit/results/2026-04-17-garch-a4b-binding-budget-replay.md")
OUTPUT_JSON = Path("research/output/garch_a4b_binding_budget_replay.json")

IS_START_MONTH = date(2020, 1, 1)
IS_END_MONTH = date(2025, 12, 1)
HOLDOUT_BOUNDARY = date(2026, 1, 1)
MAX_SLOTS = 5
FIXED_STOP_MULTIPLIER = 0.75
GARCH_HIGH_THRESHOLD = 70.0
MIN_GARCH_HIGH_N = 20
WEIGHT_GARCH = 0.5
WEIGHT_BASELINE = 0.5
MIN_LIFT_R_PER_YEAR = 20.0
MIN_SHARPE_LIFT = 0.05
MAX_DD_INFLATION = 1.20
BINDING_PASS_RATIO = 0.80
SELECTION_CHURN_CAP = 0.50
SHUFFLE_SEED = 20260417
DD_BUDGET_PROFILE_ID = "bulenox_50k"


@dataclass(frozen=True)
class StrategyMeta:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    entry_model: str
    rr_target: float
    confirm_bars: int
    filter_type: str


@dataclass(frozen=True)
class TradeRecord:
    trading_day: date
    pnl_r: float
    is_win: bool


@dataclass
class StrategyHistory:
    meta: StrategyMeta
    trades: list[TradeRecord]
    daily_pnl_r: dict[date, float]


@dataclass
class PolicyAggregate:
    total_r: float
    annualized_r: float
    sharpe_is: float
    dd_is: float
    slot_hit_rate_per_day: float
    trading_days_covered: int
    monthly_outputs: list[dict[str, object]]
    daily_pnl: list[float]
    selected_by_rebalance: list[list[str]]


def _next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _month_starts(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = date(start.year, start.month, 1)
    end_m = date(end.year, end.month, 1)
    while cur <= end_m:
        out.append(cur)
        cur = _next_month(cur)
    return out


def _annualized_sharpe(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    std = math.sqrt(var)
    if std <= 0:
        return 0.0
    return float((mean / std) * math.sqrt(252.0))


def _max_drawdown(xs: list[float]) -> float:
    eq = 0.0
    peak = 0.0
    worst = 0.0
    for x in xs:
        eq += x
        peak = max(peak, eq)
        worst = min(worst, eq - peak)
    return abs(float(worst))


def _jaccard_distance(a: list[str], b: list[str]) -> float:
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - (len(sa & sb) / float(len(union)))


def _lookup_dd_budget() -> tuple[str, float]:
    profile = ACCOUNT_PROFILES[DD_BUDGET_PROFILE_ID]
    tier = ACCOUNT_TIERS[(profile.firm, profile.account_size)]
    return DD_BUDGET_PROFILE_ID, float(tier.max_dd)


def _first_trading_days_by_month(con: duckdb.DuckDBPyConnection, start: date, end: date) -> dict[date, date]:
    out: dict[date, date] = {}
    for month_start in _month_starts(start, end):
        next_start = _next_month(month_start)
        row = con.execute(
            """
            SELECT MIN(trading_day)
            FROM orb_outcomes
            WHERE trading_day >= ? AND trading_day < ?
            """,
            [month_start, next_start],
        ).fetchone()
        if row and row[0] is not None:
            out[month_start] = row[0]
    return out


def _calendar_days(con: duckdb.DuckDBPyConnection, start: date, end_exclusive: date) -> list[date]:
    rows = con.execute(
        """
        SELECT DISTINCT trading_day
        FROM daily_features
        WHERE orb_minutes = 5
          AND trading_day >= ?
          AND trading_day < ?
        ORDER BY trading_day
        """,
        [start, end_exclusive],
    ).fetchall()
    return [r[0] for r in rows]


def _load_strategy_meta(con: duckdb.DuckDBPyConnection) -> list[StrategyMeta]:
    rel = deployable_validated_relation(con, alias="vs")
    rows = con.execute(
        f"""
        SELECT strategy_id,
               instrument,
               orb_label,
               orb_minutes,
               entry_model,
               rr_target,
               confirm_bars,
               filter_type
        FROM {rel}
        ORDER BY instrument, orb_label, strategy_id
        """
    ).fetchall()
    return [
        StrategyMeta(
            strategy_id=sid,
            instrument=inst,
            orb_label=orb,
            orb_minutes=int(orb_minutes),
            entry_model=entry_model,
            rr_target=float(rr_target),
            confirm_bars=int(confirm_bars),
            filter_type=filter_type,
        )
        for sid, inst, orb, orb_minutes, entry_model, rr_target, confirm_bars, filter_type in rows
    ]


def _adjusted_trade_records(
    con: duckdb.DuckDBPyConnection,
    meta: StrategyMeta,
    *,
    stop_multiplier: float,
    end_date: date,
) -> list[TradeRecord]:
    outcomes = _load_strategy_outcomes(
        con,
        instrument=meta.instrument,
        orb_label=meta.orb_label,
        orb_minutes=meta.orb_minutes,
        entry_model=meta.entry_model,
        rr_target=meta.rr_target,
        confirm_bars=meta.confirm_bars,
        filter_type=meta.filter_type,
        end_date=end_date,
    )
    if stop_multiplier < 1.0:
        outcomes = apply_tight_stop(outcomes, stop_multiplier, get_cost_spec(meta.instrument))

    out: list[TradeRecord] = []
    for row in outcomes:
        if row.get("outcome") not in ("win", "loss"):
            continue
        pnl_r = row.get("pnl_r")
        td = row.get("trading_day")
        if pnl_r is None or td is None:
            continue
        out.append(
            TradeRecord(
                trading_day=td,
                pnl_r=float(pnl_r),
                is_win=float(pnl_r) > 0.0,
            )
        )
    return out


def _build_histories(
    con: duckdb.DuckDBPyConnection, metas: list[StrategyMeta], as_of: date
) -> dict[str, StrategyHistory]:
    out: dict[str, StrategyHistory] = {}
    for meta in metas:
        trades = _adjusted_trade_records(
            con,
            meta,
            stop_multiplier=FIXED_STOP_MULTIPLIER,
            end_date=as_of,
        )
        daily: dict[date, float] = {}
        for trade in trades:
            daily[trade.trading_day] = daily.get(trade.trading_day, 0.0) + trade.pnl_r
        out[meta.strategy_id] = StrategyHistory(meta=meta, trades=trades, daily_pnl_r=daily)
    return out


def _build_session_regime_cache(
    con: duckdb.DuckDBPyConnection,
    metas: list[StrategyMeta],
    as_of: date,
) -> dict[tuple[str, str], list[TradeRecord]]:
    out: dict[tuple[str, str], list[TradeRecord]] = {}
    seen: set[tuple[str, str]] = set()
    for meta in metas:
        key = (meta.instrument, meta.orb_label)
        if key in seen:
            continue
        seen.add(key)
        regime_meta = StrategyMeta(
            strategy_id=f"{meta.instrument}_{meta.orb_label}_REGIME",
            instrument=meta.instrument,
            orb_label=meta.orb_label,
            orb_minutes=5,
            entry_model="E2",
            rr_target=1.0,
            confirm_bars=1,
            filter_type="NO_FILTER",
        )
        out[key] = _adjusted_trade_records(
            con,
            regime_meta,
            stop_multiplier=FIXED_STOP_MULTIPLIER,
            end_date=as_of,
        )
    return out


def _build_garch_pct_by_day(
    con: duckdb.DuckDBPyConnection, metas: list[StrategyMeta]
) -> dict[tuple[str, date], float | None]:
    instruments = sorted({m.instrument for m in metas})
    out: dict[tuple[str, date], float | None] = {}
    for inst in instruments:
        rows = con.execute(
            """
            SELECT trading_day, garch_forecast_vol_pct
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5
            ORDER BY trading_day
            """,
            [inst],
        ).fetchall()
        for td, gp in rows:
            out[(inst, td)] = None if gp is None else float(gp)
    return out


def _shuffle_garch_pct(
    gp_by_day: dict[tuple[str, date], float | None],
    *,
    seed: int,
) -> dict[tuple[str, date], float | None]:
    rng = random.Random(seed)
    out = dict(gp_by_day)
    instruments = sorted({inst for inst, _ in gp_by_day})
    for inst in instruments:
        keys = sorted(k for k in gp_by_day if k[0] == inst)
        vals = [gp_by_day[k] for k in keys]
        rng.shuffle(vals)
        for key, val in zip(keys, vals, strict=True):
            out[key] = val
    return out


def _monthly_from_trades(trades: list[TradeRecord]) -> tuple[list[tuple[str, float, int]], int, int]:
    monthly: dict[str, list[float]] = {}
    total_wins = 0
    for trade in trades:
        ym = f"{trade.trading_day.year}-{trade.trading_day.month:02d}"
        monthly.setdefault(ym, []).append(trade.pnl_r)
        if trade.is_win:
            total_wins += 1
    out: list[tuple[str, float, int]] = []
    for ym in sorted(monthly.keys(), reverse=True):
        values = monthly[ym]
        out.append((ym, round(sum(values) / len(values), 4), len(values)))
    return out, total_wins, len(trades)


def _window_trades(trades: list[TradeRecord], start: date, end_exclusive: date) -> list[TradeRecord]:
    return [t for t in trades if start <= t.trading_day < end_exclusive]


def _compute_scores(
    rebalance_date: date,
    histories: dict[str, StrategyHistory],
    session_regime_cache: dict[tuple[str, str], list[TradeRecord]],
) -> list[LaneScore]:
    out: list[LaneScore] = []
    trailing_start, trailing_end = _month_range(rebalance_date, 12)
    regime_start, regime_end = _month_range(rebalance_date, 6)

    for history in histories.values():
        meta = history.meta
        trailing_trades = _window_trades(history.trades, trailing_start, trailing_end)
        if not trailing_trades:
            out.append(
                LaneScore(
                    strategy_id=meta.strategy_id,
                    instrument=meta.instrument,
                    orb_label=meta.orb_label,
                    orb_minutes=meta.orb_minutes,
                    rr_target=meta.rr_target,
                    filter_type=meta.filter_type,
                    confirm_bars=meta.confirm_bars,
                    stop_multiplier=FIXED_STOP_MULTIPLIER,
                    trailing_expr=0.0,
                    trailing_n=0,
                    trailing_months=12,
                    annual_r_estimate=0.0,
                    trailing_wr=0.0,
                    session_regime_expr=None,
                    months_negative=12,
                    months_positive_since_last_neg_streak=0,
                    status="STALE",
                    status_reason="No trades in trailing window",
                )
            )
            continue

        monthly, total_wins, total_trades = _monthly_from_trades(trailing_trades)
        all_trades_n = sum(n for _, _, n in monthly)
        total_pnl = sum(expr * n for _, expr, n in monthly)
        trailing_expr = round(total_pnl / all_trades_n, 4) if all_trades_n else 0.0
        actual_months = len(monthly)
        annual_r = round(trailing_expr * all_trades_n / (actual_months / 12.0), 1) if actual_months else 0.0
        trailing_wr = round(total_wins / total_trades, 3) if total_trades else 0.0

        months_neg = 0
        for _, expr_m, _ in monthly:
            if expr_m < 0:
                months_neg += 1
            else:
                break

        months_pos_since = 0
        found_neg_streak = False
        for _, expr_m, _ in monthly:
            if not found_neg_streak:
                if expr_m >= 0:
                    months_pos_since += 1
                else:
                    found_neg_streak = True
            else:
                break

        regime_trades = _window_trades(
            session_regime_cache[(meta.instrument, meta.orb_label)], regime_start, regime_end
        )
        session_regime_expr = None
        if regime_trades:
            session_regime_expr = round(sum(t.pnl_r for t in regime_trades) / len(regime_trades), 4)

        status, reason = _classify_status(
            trailing_expr=trailing_expr,
            trailing_n=all_trades_n,
            actual_months=actual_months,
            months_neg=months_neg,
            months_pos_since=months_pos_since,
            annual_r=annual_r,
            session_regime_expr=session_regime_expr,
            monthly=monthly,
        )

        recent_3mo_expr = None
        if len(monthly) >= 3:
            recent = monthly[:3]
            recent_n = sum(n for _, _, n in recent)
            recent_pnl = sum(expr * n for _, expr, n in recent)
            recent_3mo_expr = round(recent_pnl / recent_n, 4) if recent_n else None

        out.append(
            LaneScore(
                strategy_id=meta.strategy_id,
                instrument=meta.instrument,
                orb_label=meta.orb_label,
                orb_minutes=meta.orb_minutes,
                rr_target=meta.rr_target,
                filter_type=meta.filter_type,
                confirm_bars=meta.confirm_bars,
                stop_multiplier=FIXED_STOP_MULTIPLIER,
                trailing_expr=trailing_expr,
                trailing_n=all_trades_n,
                trailing_months=actual_months,
                annual_r_estimate=annual_r,
                trailing_wr=trailing_wr,
                session_regime_expr=session_regime_expr,
                months_negative=months_neg,
                months_positive_since_last_neg_streak=months_pos_since,
                status=status,
                status_reason=reason,
                recent_3mo_expr=recent_3mo_expr,
            )
        )
    return out


def _candidate_garch_expr(
    history: StrategyHistory,
    rebalance_date: date,
    gp_by_day: dict[tuple[str, date], float | None],
) -> tuple[float | None, int]:
    start, end_exclusive = _month_range(rebalance_date, 12)
    trailing_trades = _window_trades(history.trades, start, end_exclusive)
    selected = [
        t.pnl_r
        for t in trailing_trades
        if (gp_by_day.get((history.meta.instrument, t.trading_day)) is not None)
        and float(gp_by_day[(history.meta.instrument, t.trading_day)]) >= GARCH_HIGH_THRESHOLD
    ]
    if len(selected) < MIN_GARCH_HIGH_N:
        return None, len(selected)
    return round(sum(selected) / len(selected), 4), len(selected)


def _pairwise_correlation_as_of(
    strategy_ids: list[str],
    histories: dict[str, StrategyHistory],
    rebalance_date: date,
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    daily = {
        sid: {d: pnl for d, pnl in histories[sid].daily_pnl_r.items() if d < rebalance_date} for sid in strategy_ids
    }
    ordered = sorted(strategy_ids)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            shared = sorted(set(daily[a]) & set(daily[b]))
            if len(shared) >= 5:
                xs = [daily[a][d] for d in shared]
                ys = [daily[b][d] for d in shared]
                out[(a, b)] = _pearson(xs, ys)
            else:
                out[(a, b)] = 0.0
    return out


def _build_ranked_allocation(
    scores: list[LaneScore],
    *,
    ranking_values: dict[str, float | None],
    max_slots: int,
    max_dd: float,
    orb_size_stats: dict[tuple[str, str], tuple[float, float]],
    correlation_matrix: dict[tuple[str, str], float],
    prior_allocation: list[str] | None,
    exclude_none_ranks: bool,
) -> list[LaneScore]:
    candidates = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
    if exclude_none_ranks:
        candidates = [s for s in candidates if ranking_values.get(s.strategy_id) is not None]

    ranked = sorted(
        candidates,
        key=lambda s: (
            0 if s.status == "PROVISIONAL" else 1,
            float("-inf") if ranking_values.get(s.strategy_id) is None else float(ranking_values[s.strategy_id]),
        ),
        reverse=True,
    )

    selected: list[LaneScore] = []
    dd_used = 0.0
    for lane in ranked:
        if len(selected) >= max_slots:
            break

        corr_reject = False
        for sel in selected:
            key = (
                (lane.strategy_id, sel.strategy_id)
                if lane.strategy_id < sel.strategy_id
                else (sel.strategy_id, lane.strategy_id)
            )
            if correlation_matrix.get(key, 0.0) > CORRELATION_REJECT_RHO:
                corr_reject = True
                break
        if corr_reject:
            continue

        _, p90_orb = orb_size_stats.get((lane.instrument, lane.orb_label), (100.0, 100.0))
        cost = COST_SPECS.get(lane.instrument)
        if cost is None:
            continue
        lane_dd = p90_orb * FIXED_STOP_MULTIPLIER * cost.point_value
        if dd_used + lane_dd > max_dd:
            continue

        if prior_allocation and lane.strategy_id not in prior_allocation:
            session_key = (lane.instrument, lane.orb_label)
            prior_in_session = [
                s for s in scores if s.strategy_id in prior_allocation and (s.instrument, s.orb_label) == session_key
            ]
            if prior_in_session:
                best_prior = max(prior_in_session, key=lambda s: s.annual_r_estimate)
                if best_prior.annual_r_estimate > 0:
                    improvement = (lane.annual_r_estimate - best_prior.annual_r_estimate) / best_prior.annual_r_estimate
                    if improvement < HYSTERESIS_PCT:
                        if best_prior.status in ("DEPLOY", "RESUME", "PROVISIONAL"):
                            selected.append(best_prior)
                            dd_used += lane_dd
                        continue

        selected.append(lane)
        dd_used += lane_dd

    return selected


def _forward_daily_series(
    selected_ids: list[str],
    histories: dict[str, StrategyHistory],
    calendar_days: list[date],
) -> tuple[list[float], float]:
    daily: list[float] = []
    hit_total = 0.0
    for day in calendar_days:
        pnl = 0.0
        hits = 0
        for sid in selected_ids:
            day_pnl = histories[sid].daily_pnl_r.get(day, 0.0)
            pnl += day_pnl
            if day_pnl != 0.0:
                hits += 1
        daily.append(round(pnl, 6))
        hit_total += hits
    hit_rate = hit_total / float(len(calendar_days)) if calendar_days else 0.0
    return daily, hit_rate


def _run_policy(
    *,
    policy_name: str,
    rebalance_months: list[date],
    month_to_rebalance: dict[date, date],
    histories: dict[str, StrategyHistory],
    session_regime_cache: dict[tuple[str, str], list[TradeRecord]],
    gp_by_day: dict[tuple[str, date], float | None] | None,
    max_dd: float,
    month_to_calendar_days: dict[date, list[date]],
    baseline_audit: bool,
) -> PolicyAggregate:
    monthly_outputs: list[dict[str, object]] = []
    all_daily: list[float] = []
    all_selected: list[list[str]] = []
    prior_allocation: list[str] | None = None

    for month_start in rebalance_months:
        rebalance_date = month_to_rebalance[month_start]
        scores = _compute_scores(rebalance_date, histories, session_regime_cache)
        deployable_ids = [s.strategy_id for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
        corr = _pairwise_correlation_as_of(deployable_ids, histories, rebalance_date)
        orb_stats = compute_orb_size_stats(rebalance_date)

        baseline_values = {s.strategy_id: _effective_annual_r(s) for s in scores}
        if policy_name == "baseline":
            ranking_values = baseline_values
            exclude_none = False
        elif policy_name == "positive_control":
            ranking_values = {s.strategy_id: s.trailing_expr for s in scores}
            exclude_none = False
        else:
            assert gp_by_day is not None
            ranking_values = {}
            garch_ns: dict[str, int] = {}
            for score in scores:
                expr, n_garch = _candidate_garch_expr(histories[score.strategy_id], rebalance_date, gp_by_day)
                garch_ns[score.strategy_id] = n_garch
                ranking_values[score.strategy_id] = (
                    None
                    if expr is None
                    else round((WEIGHT_GARCH * expr) + (WEIGHT_BASELINE * _effective_annual_r(score)), 6)
                )
            exclude_none = True

        selected = _build_ranked_allocation(
            scores,
            ranking_values=ranking_values,
            max_slots=MAX_SLOTS,
            max_dd=max_dd,
            orb_size_stats=orb_stats,
            correlation_matrix=corr,
            prior_allocation=prior_allocation,
            exclude_none_ranks=exclude_none,
        )

        if baseline_audit:
            canonical = build_allocation(
                scores,
                max_slots=MAX_SLOTS,
                max_dd=max_dd,
                stop_multiplier=FIXED_STOP_MULTIPLIER,
                orb_size_stats=orb_stats,
                correlation_matrix=corr,
                prior_allocation=prior_allocation,
            )
            local_ids = [s.strategy_id for s in selected]
            canonical_ids = [s.strategy_id for s in canonical]
            if local_ids != canonical_ids:
                raise AssertionError(
                    f"Baseline local allocator diverged from canonical build_allocation on {rebalance_date}: "
                    f"{local_ids} != {canonical_ids}"
                )

        next_month = _next_month(month_start)
        end_exclusive = month_to_rebalance.get(next_month)
        if end_exclusive is None:
            continue
        calendar_days = month_to_calendar_days[month_start]
        selected_ids = [s.strategy_id for s in selected]
        daily_series, hit_rate = _forward_daily_series(selected_ids, histories, calendar_days)
        total_r = round(sum(daily_series), 6)
        monthly_outputs.append(
            {
                "month": month_start.isoformat(),
                "rebalance_date": rebalance_date.isoformat(),
                "selected_lanes": selected_ids,
                "forward_window_start": rebalance_date.isoformat(),
                "forward_window_end_exclusive": end_exclusive.isoformat(),
                "forward_month_total_r": total_r,
                "forward_month_daily_sharpe": _annualized_sharpe(daily_series),
                "slot_hit_rate_per_day": hit_rate,
            }
        )
        all_daily.extend(daily_series)
        all_selected.append(selected_ids)
        prior_allocation = selected_ids

    total_r = round(sum(all_daily), 6)
    annualized_r = total_r * (252.0 / len(all_daily)) if all_daily else 0.0
    return PolicyAggregate(
        total_r=total_r,
        annualized_r=float(annualized_r),
        sharpe_is=_annualized_sharpe(all_daily),
        dd_is=_max_drawdown(all_daily),
        slot_hit_rate_per_day=(sum(m["slot_hit_rate_per_day"] for m in monthly_outputs) / len(monthly_outputs))
        if monthly_outputs
        else 0.0,
        trading_days_covered=len(all_daily),
        monthly_outputs=monthly_outputs,
        daily_pnl=all_daily,
        selected_by_rebalance=all_selected,
    )


def _candidate_eligibility_by_rebalance(
    rebalance_months: list[date],
    month_to_rebalance: dict[date, date],
    histories: dict[str, StrategyHistory],
    session_regime_cache: dict[tuple[str, str], list[TradeRecord]],
    gp_by_day: dict[tuple[str, date], float | None],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for month_start in rebalance_months:
        rebalance_date = month_to_rebalance[month_start]
        scores = _compute_scores(rebalance_date, histories, session_regime_cache)
        eligible = 0
        deployable = 0
        for score in scores:
            if score.status not in ("DEPLOY", "RESUME", "PROVISIONAL"):
                continue
            deployable += 1
            expr, n_garch = _candidate_garch_expr(histories[score.strategy_id], rebalance_date, gp_by_day)
            if expr is not None:
                eligible += 1
        out.append(
            {
                "month": month_start.isoformat(),
                "rebalance_date": rebalance_date.isoformat(),
                "deployable_lane_count": deployable,
                "candidate_eligible_lane_count": eligible,
                "binding_pass": eligible > (MAX_SLOTS * 2),
            }
        )
    return out


def _mean_jaccard(a_sets: list[list[str]], b_sets: list[list[str]]) -> float:
    if not a_sets or not b_sets:
        return 0.0
    vals = [_jaccard_distance(a, b) for a, b in zip(a_sets, b_sets, strict=True)]
    return float(sum(vals) / len(vals))


def _effect_ratio(oos: float, ins: float) -> float | None:
    if abs(ins) < 1e-9:
        return None
    return float(oos / ins)


def _pass_primary(
    candidate: PolicyAggregate, baseline: PolicyAggregate, binding_pass: bool, mean_jaccard: float
) -> dict[str, object]:
    annualized_delta = candidate.annualized_r - baseline.annualized_r
    sharpe_delta = candidate.sharpe_is - baseline.sharpe_is
    dd_ratio = (candidate.dd_is / baseline.dd_is) if baseline.dd_is > 0 else math.inf
    return {
        "annualized_r_delta": annualized_delta,
        "sharpe_delta": sharpe_delta,
        "dd_ratio": dd_ratio,
        "binding_pass": binding_pass,
        "churn_pass": mean_jaccard <= SELECTION_CHURN_CAP,
        "primary_pass": (
            annualized_delta >= MIN_LIFT_R_PER_YEAR
            and sharpe_delta >= MIN_SHARPE_LIFT
            and dd_ratio <= MAX_DD_INFLATION
            and binding_pass
            and mean_jaccard <= SELECTION_CHURN_CAP
        ),
    }


def _verdict_text(
    *,
    primary: dict[str, object],
    oos_direction_match: bool | None,
    oos_effect_ratio: float | None,
    shuffle_pass: bool,
    positive_control_pass: bool,
    binding_pass: bool,
) -> str:
    if not binding_pass:
        return "NULL_BY_CONSTRUCTION"
    if not shuffle_pass:
        return "KILLED_DESTRUCTION_SHUFFLE"
    if not positive_control_pass:
        return "KILLED_POSITIVE_CONTROL"
    if not bool(primary["primary_pass"]):
        return "IS_FAIL"
    if oos_direction_match is False:
        return "KILLED_OOS_DIRECTION_FLIP"
    if oos_effect_ratio is not None and oos_effect_ratio < 0.40:
        return "KILLED_OOS_EFFECT_RATIO"
    return "PASS"


def _emit_markdown(
    path: Path,
    *,
    as_of: date,
    dd_budget_profile_id: str,
    dd_budget: float,
    shelf_count: int,
    binding_rows: list[dict[str, object]],
    baseline_is: PolicyAggregate,
    candidate_is: PolicyAggregate,
    shuffle_is: PolicyAggregate,
    positive_control_is: PolicyAggregate,
    baseline_oos: PolicyAggregate,
    candidate_oos: PolicyAggregate,
    primary: dict[str, object],
    shuffle_primary: dict[str, object],
    positive_primary: dict[str, object],
    mean_jaccard: float,
    oos_direction_match: bool | None,
    oos_effect_ratio: float | None,
    verdict: str,
) -> None:
    binding_passes = sum(1 for row in binding_rows if row["binding_pass"])
    binding_ratio = binding_passes / float(len(binding_rows)) if binding_rows else 0.0
    top_binding = sorted(binding_rows, key=lambda row: int(row["candidate_eligible_lane_count"]), reverse=True)[:12]
    lines = [
        "# Garch A4b Binding-Budget Replay",
        "",
        f"**Date:** {as_of}",
        f"**Pre-registration:** `{HYPOTHESIS_FILE}`",
        f"**Universe:** deployable validated shelf (`{shelf_count}` lanes; no profile/session/instrument filter inside A4b).",
        f"**Binding budget:** `max_slots={MAX_SLOTS}`, `max_dd=${dd_budget:,.0f}` derived from `{dd_budget_profile_id}` as the real profile scalar matching the locked 5-slot budget, without reusing that profile's lane universe.",
        f"**Stop policy:** fixed `SM={FIXED_STOP_MULTIPLIER}` applied via canonical `apply_tight_stop()` path.",
        f"**Verdict:** `{verdict}`",
        "",
        "## Binding Preflight",
        "",
        f"- Rebalance months in IS: `{len(binding_rows)}`",
        f"- Binding pass count: `{binding_passes}` / `{len(binding_rows)}`",
        f"- Binding pass ratio: `{binding_ratio:.3f}` (need `>= {BINDING_PASS_RATIO:.2f}`)",
        f"- Pass rule: candidate-eligible lanes `> {MAX_SLOTS * 2}` on at least `80%` of IS rebalance dates.",
        "",
        "| Month | Rebalance | Deployable | Candidate Eligible | Binding Pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top_binding:
        lines.append(
            f"| {row['month']} | {row['rebalance_date']} | {row['deployable_lane_count']} | {row['candidate_eligible_lane_count']} | {row['binding_pass']} |"
        )

    lines += [
        "",
        "## IS Results",
        "",
        "| Route | Annualized R | Sharpe | MaxDD R | Slot hit/day | Trading days |",
        "|---|---:|---:|---:|---:|---:|",
        f"| BASELINE_LIT_GROUNDED | {baseline_is.annualized_r:+.2f} | {baseline_is.sharpe_is:+.3f} | {baseline_is.dd_is:.2f} | {baseline_is.slot_hit_rate_per_day:.3f} | {baseline_is.trading_days_covered} |",
        f"| CANDIDATE_GARCH_RANK | {candidate_is.annualized_r:+.2f} | {candidate_is.sharpe_is:+.3f} | {candidate_is.dd_is:.2f} | {candidate_is.slot_hit_rate_per_day:.3f} | {candidate_is.trading_days_covered} |",
        f"| DESTRUCTION_SHUFFLE | {shuffle_is.annualized_r:+.2f} | {shuffle_is.sharpe_is:+.3f} | {shuffle_is.dd_is:.2f} | {shuffle_is.slot_hit_rate_per_day:.3f} | {shuffle_is.trading_days_covered} |",
        f"| POSITIVE_CONTROL_TRAILING_EXPR | {positive_control_is.annualized_r:+.2f} | {positive_control_is.sharpe_is:+.3f} | {positive_control_is.dd_is:.2f} | {positive_control_is.slot_hit_rate_per_day:.3f} | {positive_control_is.trading_days_covered} |",
        "",
        "## Decision Rules",
        "",
        f"- Candidate annualized R delta vs baseline: `{primary['annualized_r_delta']:+.2f}` (need `>= +{MIN_LIFT_R_PER_YEAR:.1f}`)",
        f"- Candidate Sharpe delta vs baseline: `{primary['sharpe_delta']:+.3f}` (need `>= +{MIN_SHARPE_LIFT:.2f}`)",
        f"- Candidate DD ratio vs baseline: `{primary['dd_ratio']:.3f}` (need `<= {MAX_DD_INFLATION:.2f}`)",
        f"- Mean selection churn: `{mean_jaccard:.3f}` (need `<= {SELECTION_CHURN_CAP:.2f}`)",
        f"- Primary IS pass: `{primary['primary_pass']}`",
        f"- Destruction shuffle passes primary: `{shuffle_primary['primary_pass']}` (must be `False`)",
        f"- Positive control passes primary: `{positive_primary['primary_pass']}` (must be `True`)",
        "",
        "## OOS Descriptive",
        "",
        "| Route | Annualized R | Sharpe | MaxDD R | Trading days |",
        "|---|---:|---:|---:|---:|",
        f"| BASELINE_LIT_GROUNDED | {baseline_oos.annualized_r:+.2f} | {baseline_oos.sharpe_is:+.3f} | {baseline_oos.dd_is:.2f} | {baseline_oos.trading_days_covered} |",
        f"| CANDIDATE_GARCH_RANK | {candidate_oos.annualized_r:+.2f} | {candidate_oos.sharpe_is:+.3f} | {candidate_oos.dd_is:.2f} | {candidate_oos.trading_days_covered} |",
        "",
        f"- OOS direction match vs IS effect: `{oos_direction_match}`",
        f"- OOS effect ratio vs IS effect: `{'n/a' if oos_effect_ratio is None else f'{oos_effect_ratio:.3f}'}`",
        "",
        "SURVIVED SCRUTINY:",
    ]

    survived: list[str] = []
    if binding_ratio >= BINDING_PASS_RATIO:
        survived.append(f"Binding preflight cleared at {binding_ratio:.3f}.")
    if not bool(shuffle_primary["primary_pass"]):
        survived.append("Destruction shuffle failed to pass, so the garch term did not survive randomization.")
    if bool(positive_primary["primary_pass"]):
        survived.append("Positive control cleared the same harness.")
    if bool(primary["primary_pass"]):
        survived.append("Candidate cleared the locked IS utility bar.")
    if not survived:
        survived.append("No load-bearing criterion survived strongly enough to promote the candidate.")
    for row in survived:
        lines.append(f"- {row}")

    lines += [
        "",
        "DID NOT SURVIVE:",
    ]
    failed: list[str] = []
    if binding_ratio < BINDING_PASS_RATIO:
        failed.append("Binding preflight failed; the stage is null by construction.")
    if not bool(positive_primary["primary_pass"]):
        failed.append("Positive control failed, so the harness cannot claim a clean utility test.")
    if bool(shuffle_primary["primary_pass"]):
        failed.append("Destruction shuffle also passed, which is a data-mining kill.")
    if not bool(primary["primary_pass"]):
        failed.append("Candidate failed the locked IS pass rule.")
    if oos_direction_match is False:
        failed.append("OOS direction flipped versus the IS effect.")
    if oos_effect_ratio is not None and oos_effect_ratio < 0.40:
        failed.append(f"OOS effect ratio {oos_effect_ratio:.3f} is below the 0.40 kill line.")
    if not failed:
        failed.append("No extra kill criterion fired after the core IS decision.")
    for row in failed:
        lines.append(f"- {row}")

    lines += [
        "",
        "CAVEATS:",
        f"- A4b uses the canonical allocator surface but a shelf-level fixed stop policy (`SM={FIXED_STOP_MULTIPLIER}`) and a profile-derived DD scalar only; it is not a deployment claim.",
        "- Early IS rebalances face thinner garch history, so candidate eligibility can be sparse before the feature fully matures.",
        "- The comparator remains a validated-shelf utility test, not profile translation, not forward shadow proof, and not standalone garch edge evidence.",
        "",
        "NEXT STEPS:",
    ]
    if verdict == "PASS":
        lines.append("- Promote to the reserved sensitivity stage on `w1/w2` only, then queue forward shadow.")
    elif verdict == "NULL_BY_CONSTRUCTION":
        lines.append(
            "- Redesign the scarce-resource surface before interpreting utility; do not rescue this stage with narrative."
        )
    else:
        lines.append("- Treat A4b as failed or parked on the current locked design and do not tune around the miss.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked A4b binding-budget replay.")
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    args = parser.parse_args()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        as_of = con.execute("SELECT MAX(trading_day) FROM daily_features").fetchone()[0]
        metas = _load_strategy_meta(con)
        histories = _build_histories(con, metas, as_of)
        session_regime_cache = _build_session_regime_cache(con, metas, as_of)
        gp_by_day = _build_garch_pct_by_day(con, metas)
    finally:
        con.close()

    dd_budget_profile_id, dd_budget = _lookup_dd_budget()
    shuffled_gp = _shuffle_garch_pct(gp_by_day, seed=SHUFFLE_SEED)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        month_to_rebalance = _first_trading_days_by_month(con, IS_START_MONTH, date(as_of.year, as_of.month, 1))
        # Add the 2026-01 boundary explicitly for the final IS forward window.
        if date(2026, 1, 1) not in month_to_rebalance:
            extra = _first_trading_days_by_month(con, date(2026, 1, 1), date(2026, 1, 1))
            month_to_rebalance.update(extra)

        month_to_calendar_days: dict[date, list[date]] = {}
        for month_start, rebalance_date in month_to_rebalance.items():
            next_month = _next_month(month_start)
            end_exclusive = month_to_rebalance.get(next_month)
            if end_exclusive is None:
                if month_start.year == as_of.year and month_start.month == as_of.month:
                    end_exclusive = as_of + timedelta(days=1)
                else:
                    continue
            month_to_calendar_days[month_start] = _calendar_days(con, rebalance_date, end_exclusive)
    finally:
        con.close()

    is_months = [
        m
        for m in _month_starts(IS_START_MONTH, IS_END_MONTH)
        if m in month_to_rebalance and m in month_to_calendar_days
    ]
    oos_months = [
        m
        for m in _month_starts(HOLDOUT_BOUNDARY, date(as_of.year, as_of.month, 1))
        if m in month_to_rebalance and m in month_to_calendar_days
    ]

    binding_rows = _candidate_eligibility_by_rebalance(
        is_months,
        month_to_rebalance,
        histories,
        session_regime_cache,
        gp_by_day,
    )
    binding_ratio = (
        sum(1 for row in binding_rows if row["binding_pass"]) / float(len(binding_rows)) if binding_rows else 0.0
    )
    binding_pass = binding_ratio >= BINDING_PASS_RATIO

    baseline_is = _run_policy(
        policy_name="baseline",
        rebalance_months=is_months,
        month_to_rebalance=month_to_rebalance,
        histories=histories,
        session_regime_cache=session_regime_cache,
        gp_by_day=None,
        max_dd=dd_budget,
        month_to_calendar_days=month_to_calendar_days,
        baseline_audit=True,
    )
    candidate_is = _run_policy(
        policy_name="candidate",
        rebalance_months=is_months,
        month_to_rebalance=month_to_rebalance,
        histories=histories,
        session_regime_cache=session_regime_cache,
        gp_by_day=gp_by_day,
        max_dd=dd_budget,
        month_to_calendar_days=month_to_calendar_days,
        baseline_audit=False,
    )
    shuffle_is = _run_policy(
        policy_name="candidate",
        rebalance_months=is_months,
        month_to_rebalance=month_to_rebalance,
        histories=histories,
        session_regime_cache=session_regime_cache,
        gp_by_day=shuffled_gp,
        max_dd=dd_budget,
        month_to_calendar_days=month_to_calendar_days,
        baseline_audit=False,
    )
    positive_control_is = _run_policy(
        policy_name="positive_control",
        rebalance_months=is_months,
        month_to_rebalance=month_to_rebalance,
        histories=histories,
        session_regime_cache=session_regime_cache,
        gp_by_day=None,
        max_dd=dd_budget,
        month_to_calendar_days=month_to_calendar_days,
        baseline_audit=False,
    )
    baseline_oos = _run_policy(
        policy_name="baseline",
        rebalance_months=oos_months,
        month_to_rebalance=month_to_rebalance,
        histories=histories,
        session_regime_cache=session_regime_cache,
        gp_by_day=None,
        max_dd=dd_budget,
        month_to_calendar_days=month_to_calendar_days,
        baseline_audit=True,
    )
    candidate_oos = _run_policy(
        policy_name="candidate",
        rebalance_months=oos_months,
        month_to_rebalance=month_to_rebalance,
        histories=histories,
        session_regime_cache=session_regime_cache,
        gp_by_day=gp_by_day,
        max_dd=dd_budget,
        month_to_calendar_days=month_to_calendar_days,
        baseline_audit=False,
    )

    mean_jaccard = _mean_jaccard(baseline_is.selected_by_rebalance, candidate_is.selected_by_rebalance)
    primary = _pass_primary(candidate_is, baseline_is, binding_pass, mean_jaccard)
    shuffle_primary = _pass_primary(
        shuffle_is,
        baseline_is,
        binding_pass,
        _mean_jaccard(baseline_is.selected_by_rebalance, shuffle_is.selected_by_rebalance),
    )
    positive_primary = _pass_primary(
        positive_control_is,
        baseline_is,
        binding_pass,
        _mean_jaccard(baseline_is.selected_by_rebalance, positive_control_is.selected_by_rebalance),
    )
    is_effect = candidate_is.annualized_r - baseline_is.annualized_r
    oos_effect = candidate_oos.annualized_r - baseline_oos.annualized_r
    oos_direction_match = (
        None if abs(is_effect) < 1e-9 else (math.copysign(1.0, oos_effect) == math.copysign(1.0, is_effect))
    )
    oos_effect_ratio = _effect_ratio(oos_effect, is_effect)
    verdict = _verdict_text(
        primary=primary,
        oos_direction_match=oos_direction_match,
        oos_effect_ratio=oos_effect_ratio,
        shuffle_pass=not bool(shuffle_primary["primary_pass"]),
        positive_control_pass=bool(positive_primary["primary_pass"]),
        binding_pass=binding_pass,
    )

    payload = {
        "as_of": as_of.isoformat(),
        "hypothesis_file": HYPOTHESIS_FILE,
        "config": {
            "max_slots": MAX_SLOTS,
            "fixed_stop_multiplier": FIXED_STOP_MULTIPLIER,
            "garch_high_threshold": GARCH_HIGH_THRESHOLD,
            "min_garch_high_n": MIN_GARCH_HIGH_N,
            "weights": {"w1": WEIGHT_GARCH, "w2": WEIGHT_BASELINE},
            "dd_budget_profile_id": dd_budget_profile_id,
            "dd_budget": dd_budget,
            "shuffle_seed": SHUFFLE_SEED,
        },
        "binding_check": {
            "rows": binding_rows,
            "pass_ratio": binding_ratio,
            "pass": binding_pass,
        },
        "is_results": {
            "baseline": asdict(baseline_is),
            "candidate": asdict(candidate_is),
            "destruction_shuffle": asdict(shuffle_is),
            "positive_control": asdict(positive_control_is),
            "primary": primary,
            "shuffle_primary": shuffle_primary,
            "positive_primary": positive_primary,
            "mean_jaccard_distance": mean_jaccard,
        },
        "oos_results": {
            "baseline": asdict(baseline_oos),
            "candidate": asdict(candidate_oos),
            "direction_match": oos_direction_match,
            "effect_ratio": oos_effect_ratio,
        },
        "verdict": verdict,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[raw] {output_json}")

    _emit_markdown(
        Path(args.output_md),
        as_of=as_of,
        dd_budget_profile_id=dd_budget_profile_id,
        dd_budget=dd_budget,
        shelf_count=len(metas),
        binding_rows=binding_rows,
        baseline_is=baseline_is,
        candidate_is=candidate_is,
        shuffle_is=shuffle_is,
        positive_control_is=positive_control_is,
        baseline_oos=baseline_oos,
        candidate_oos=candidate_oos,
        primary=primary,
        shuffle_primary=shuffle_primary,
        positive_primary=positive_primary,
        mean_jaccard=mean_jaccard,
        oos_direction_match=oos_direction_match,
        oos_effect_ratio=oos_effect_ratio,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
