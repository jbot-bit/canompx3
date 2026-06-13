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
import logging
import random
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from pipeline.cost_model import get_cost_spec, risk_in_dollars
from pipeline.db_config import configure_connection
from pipeline.db_connect import open_read_only_with_retry
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import apply_tight_stop
from trading_app.derived_state import (
    build_code_fingerprint,
    build_db_identity,
    build_profile_fingerprint,
    build_state_envelope,
    classify_state_reason,
    get_git_head,
    validate_state_envelope,
)
from trading_app.portfolio import (
    build_profile_portfolio,
    compute_position_size_vol_scaled,
    compute_vol_scalar,
)
from trading_app.prop_firm_policies import get_payout_policy
from trading_app.prop_profiles import (
    AccountProfile,
    get_account_tier,
    get_firm_spec,
    get_profile,
    get_profile_lane_definitions,
    resolve_profile_id,
)
from trading_app.strategy_fitness import _load_strategy_outcomes
from trading_app.topstep_scaling_plan import lots_for_position, max_lots_for_xfa

log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
MIN_SURVIVAL_PROBABILITY = 0.70
DEFAULT_REPORT_MAX_AGE_DAYS = 30
CRITERION11_STATE_TYPE = "account_survival"
CRITERION11_STATE_SCHEMA_VERSION = 1
# Strict-DD budget fraction is PROFILE-AWARE — resolved by
# `effective_strict_dd_budget()`, NOT applied as a flat constant. Express-funded
# (prop wrappers) carry an operator-chosen safety belt below the firm's MLL;
# self-funded (real capital) is risk-first and NEVER prop-fraction-capped
# (see .claude/rules/self-funded-sizing-doctrine.md).
#   @margin-guard-not-earnings-cap — the express fraction is a survival safety
#   belt on the prop MLL, not an earnings ceiling; it must never reach a
#   self-funded sizing path.
# Express belt: 0.90 of the $2,000 Topstep MLL = $1,800 (operator risk-knob,
# 2026-06-04; raised from the prior arbitrary 0.80/$1,600 — see
# docs/audit/results/2026-06-03-c11-clearance-audit.md). Self-funded: 1.00 (full
# self-imposed firm DD; its risk-first sizing is enforced elsewhere via
# max_risk_per_trade, not by haircutting this budget like a prop account).
STRICT_DD_BUDGET_FRACTION_EXPRESS = 0.90
STRICT_DD_BUDGET_FRACTION_SELF_FUNDED = 1.00
STRICT_DD_HORIZON_DAYS = 90


def effective_strict_dd_budget(profile: AccountProfile, rules: SurvivalRules) -> float:
    """Resolve the strict 90-day DD budget in dollars for one profile.

    Single source of truth for BOTH the survival gate verdict and any
    budget-aware lane selection (Stage 2+). Fail-CLOSED: an unresolved or
    falsy `is_express_funded` resolves to the STRICTER express belt, never the
    relaxed self-funded one — a budget can only ever be too conservative by
    default, never too loose.

    EXPRESS-FUNDED (prop wrappers): a fraction of the firm MLL
    (``rules.dd_limit_dollars`` = ``tier.max_dd``). The fraction is an operator
    safety belt on the firm number, which is the correct binding guard for a
    funded wrapper.

    SELF-FUNDED (real capital): the budget is the profile's OWN self-imposed DD
    HALT (``self_imposed_dd_dollars``), NOT a prop-firm tier figure. Sourcing it
    from ``rules.dd_limit_dollars`` (= ``tier.max_dd``) would bound
    personal-capital risk by a prop-shaped number — the exact leak forbidden by
    .claude/rules/self-funded-sizing-doctrine.md. A self_funded profile that
    omits ``self_imposed_dd_dollars`` fail-CLOSES to the stricter express belt
    on the firm number, never to the looser prop figure at full fraction.
    """
    # Self-funded relaxation requires an explicit, truthy non-express flag.
    # Anything else (None, missing, True) falls through to the express belt.
    if getattr(profile, "is_express_funded", True) is False:
        # Risk-first SOURCE: the profile's own self-imposed DD halt, never the
        # prop tier number. Fail-CLOSED if the source is missing/non-positive/
        # non-finite. `bool` is a subclass of `int` in Python (True > 0 is True),
        # so it is excluded explicitly — a stray True must never resolve to a $1
        # budget. NaN is rejected by `> 0` being False.
        self_imposed = getattr(profile, "self_imposed_dd_dollars", None)
        if isinstance(self_imposed, (int, float)) and not isinstance(self_imposed, bool) and self_imposed > 0:
            return round(float(self_imposed) * STRICT_DD_BUDGET_FRACTION_SELF_FUNDED, 2)
        # Missing risk-first source → stricter express belt on the firm number.
        return round(rules.dd_limit_dollars * STRICT_DD_BUDGET_FRACTION_EXPRESS, 2)
    return round(rules.dd_limit_dollars * STRICT_DD_BUDGET_FRACTION_EXPRESS, 2)


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
    # VESTIGIAL (D-3 seam Stage 1): set but never read by simulate_survival. Per-trade
    # DD is now sized + pre-scaled at TradePath construction (_load_lane_trade_paths via
    # SizingContext). Do NOT treat this as a sizing knob — it does not bind the sim's
    # contract count. Kept (not removed) because the spec audit confirmed PnL is
    # pre-scaled at TradePath, so removal is a wider refactor than Stage 1's scope.
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
    strict_account_gate_pass: bool
    effective_dd_budget_dollars: float
    historical_daily_loss_breach_days: list[str]
    historical_daily_loss_breach_count: int
    historical_max_observed_90d_dd_dollars: float
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
class SizingContext:
    """Transports the LIVE engine's sizing inputs into the survival sim so the
    gate projects DD at the contract count the engine WOULD trade.

    Transport only — it does NOT re-encode any lot ladder. ``account_equity`` is
    the Portfolio NOTIONAL (the value execution_engine.py:267 sizes from), NEVER
    SurvivalRules.starting_balance (which is 0.0 for express-funded XFA accounts
    and would zero the sizer -> a false PASS).
    """

    account_equity: float
    risk_per_trade_pct: float
    account_size: int
    max_contracts_by_strategy: dict[str, int]

    def max_contracts_for(self, strategy_id: str) -> int:
        # Fail closed to 1 for an unknown lane - never size an unrecognised lane up.
        return int(self.max_contracts_by_strategy.get(strategy_id, 1))


@dataclass(frozen=True)
class TradePath:
    """Canonical per-trade path summary used for conservative intraday replay.

    `contracts` and `instrument` are the canonical fields for Scaling Plan
    aggregation. `lots` is kept for legacy readers but MUST NOT be summed
    across trades to compute concurrent exposure — that is the exact bug
    that created the F-1 false alarm (see
    docs/audit/2026-04-11-criterion-11-f1-false-alarm.md). Use
    `lots_for_position(inst, sum_of_contracts)` on aggregated contracts
    instead.
    """

    trading_day: date
    strategy_id: str
    entry_ts: datetime | None
    exit_ts: datetime | None
    pnl_dollars: float
    mae_dollars: float
    mfe_dollars: float
    lots: int
    # Planned entry-to-stop distance in points (abs(entry_price - stop_price)) —
    # the SAME basis the live engine sizes on (execution_engine.py, 8 sites). The
    # survival sizer MUST size on this, not on realized MAE: mae_dollars = mae_r *
    # risk_dollars with mae_r <= 1.0 (small for winners), so deriving risk_points
    # from MAE under-states the stop distance and over-sizes contracts (D-3 fix).
    # Per-contract price distance — invariant to contract count; NEVER scaled by n.
    risk_points: float
    contracts: int = 1
    instrument: str = ""


def get_survival_report_path(profile_id: str | None = None) -> Path:
    """Return the canonical Criterion 11 state path for one profile."""
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    return STATE_DIR / f"account_survival_{resolved_profile_id}.json"


def _build_profile_fingerprint(profile) -> str:
    """Backwards-compatible wrapper around the canonical shared helper."""
    return build_profile_fingerprint(profile)


def _criterion11_code_paths() -> list[Path]:
    """Files whose content the cached Criterion 11 PASS depends on.

    Capital review D (2026-06-06): the fingerprint previously covered only
    account_survival.py + derived_state.py, so a change to live ORB-cap
    enforcement, position sizing, lane registry, or routing could leave an
    existing C11 PASS valid even though the real live risk math had drifted.
    Include every live-risk execution dependency so a change to any of them
    invalidates the cached PASS (fail-closed staleness detection).
    """
    root = Path(__file__).resolve().parents[1]
    return [
        Path(__file__).resolve(),
        root / "trading_app" / "derived_state.py",
        root / "trading_app" / "prop_profiles.py",
        root / "trading_app" / "portfolio.py",
        root / "trading_app" / "execution_engine.py",
        root / "trading_app" / "live" / "session_orchestrator.py",
    ]


def _current_survival_canonical_inputs(
    profile_id: str | None = None,
    *,
    db_path: Path | None = None,
) -> dict[str, object]:
    """Return the canonical-input contract for Criterion 11 state."""
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    profile = get_profile(resolved_profile_id)
    effective_db_path = (db_path or GOLD_DB_PATH).resolve()
    lane_ids = [str(lane["strategy_id"]) for lane in get_profile_lane_definitions(resolved_profile_id)]
    return {
        "profile_id": resolved_profile_id,
        "profile_fingerprint": _build_profile_fingerprint(profile),
        "lane_ids": lane_ids,
        "db_path": str(effective_db_path),
        "db_identity": build_db_identity(effective_db_path),
        "code_fingerprint": build_code_fingerprint(_criterion11_code_paths()),
    }


def read_survival_report_state(
    profile_id: str | None = None,
    *,
    db_path: Path | None = None,
    today: date | None = None,
) -> dict[str, object]:
    """Read and validate the current Criterion 11 derived-state file."""
    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    report_path = get_survival_report_path(resolved_profile_id)
    base: dict[str, object] = {
        "profile_id": resolved_profile_id,
        "available": report_path.exists(),
        "valid": False,
        "reason": None,
        "summary": {},
        "rules": {},
        "metadata": {},
        "canonical_inputs": {},
        "freshness": {},
    }
    if not report_path.exists():
        base["reason"] = "missing"
        return base

    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        base["reason"] = f"unreadable: {exc}"
        return base

    if not isinstance(raw, dict):
        base["reason"] = "invalid state payload"
        return base

    if "payload" not in raw or "canonical_inputs" not in raw or "freshness" not in raw:
        base["summary"] = raw.get("summary", {}) if isinstance(raw.get("summary"), dict) else {}
        base["rules"] = raw.get("rules", {}) if isinstance(raw.get("rules"), dict) else {}
        base["metadata"] = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
        base["reason"] = "legacy state: missing versioned envelope"
        return base

    current_inputs = _current_survival_canonical_inputs(resolved_profile_id, db_path=db_path)
    valid, reason, envelope = validate_state_envelope(
        raw,
        expected_state_type=CRITERION11_STATE_TYPE,
        expected_schema_version=CRITERION11_STATE_SCHEMA_VERSION,
        current_profile_id=str(current_inputs["profile_id"]),
        current_profile_fingerprint=str(current_inputs["profile_fingerprint"]),
        current_lane_ids=list(current_inputs["lane_ids"]),
        current_db_identity=str(current_inputs["db_identity"]),
        current_code_fingerprint=str(current_inputs["code_fingerprint"]),
        today=today or date.today(),
    )

    payload = raw.get("payload", {})
    base["summary"] = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    base["rules"] = payload.get("rules", {}) if isinstance(payload.get("rules"), dict) else {}
    base["metadata"] = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    base["canonical_inputs"] = raw.get("canonical_inputs", {}) if isinstance(raw.get("canonical_inputs"), dict) else {}
    base["freshness"] = raw.get("freshness", {}) if isinstance(raw.get("freshness"), dict) else {}

    if not valid or envelope is None:
        base["reason"] = reason or "invalid state envelope"
        return base

    base["valid"] = True
    base["reason"] = None
    base["canonical_inputs"] = envelope["canonical_inputs"]
    base["freshness"] = envelope["freshness"]
    return base


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
    max_orb_size_pts: float | None = None,
) -> dict[date, float]:
    """Load one lane's historical daily PnL in dollars from canonical outcomes."""
    daily: dict[date, float] = {}
    for trade in _load_lane_trade_paths(
        con,
        strategy_id,
        as_of_date=as_of_date,
        effective_stop_multiplier=effective_stop_multiplier,
        max_orb_size_pts=max_orb_size_pts,
    ):
        daily[trade.trading_day] = daily.get(trade.trading_day, 0.0) + trade.pnl_dollars
    return daily


def _lane_atr_by_day(con, instrument: str, orb_minutes: int, days: set) -> dict:
    """Per-day atr_20 for an instrument over the given trade days (one batched query).

    Uses orb_minutes=5 because atr_20 is a per-day, non-aperture value (verified:
    0 symbol-days with cross-aperture variance). Days with NULL atr_20 are omitted
    — the caller falls back to vol_scalar=1.0 (engine parity, execution_engine.py:280).

    Errors (missing table / query failure) propagate intentionally: a structural
    inability to read ATR must fail the gate closed, not silently fall back.
    """
    if not days:
        return {}
    rows = con.execute(
        """SELECT trading_day, atr_20 FROM daily_features
           WHERE symbol = ? AND orb_minutes = ? AND atr_20 IS NOT NULL""",
        [instrument, orb_minutes],
    ).fetchall()
    wanted = set(days)
    return {r[0]: float(r[1]) for r in rows if r[0] in wanted}


def _lane_median_atr(con, instrument: str, days: set) -> dict:
    """Trailing 252d median atr_20 as-of each trade day, via the CANONICAL helper
    ``paper_trader._get_median_atr_20`` (already point-in-time: its SQL filters
    ``trading_day < ?`` — no look-ahead). No re-encoded SQL here (institutional
    rigor § 4: delegate to canonical, never re-implement).

    Only truthy medians are mapped; ``_get_median_atr_20`` returns 0.0 when no
    trailing data exists, which we omit so the caller falls back to
    ``vol_scalar=1.0`` (engine parity) rather than dividing by zero.
    """
    from trading_app.paper_trader import _get_median_atr_20

    out: dict = {}
    for d in days:
        med = _get_median_atr_20(con, instrument, d)
        if med:
            out[d] = float(med)
    return out


def _build_trade_paths_from_outcomes(
    con: duckdb.DuckDBPyConnection,
    params: dict,
    *,
    as_of_date: date,
    effective_stop_multiplier: float | None,
    max_orb_size_pts: float | None,
    strategy_id: str,
) -> list[TradePath]:
    """Load raw (unscaled, 1-contract) TradePaths from canonical outcomes.

    Pure extraction of the original _load_lane_trade_paths body — behaviour
    is byte-identical to the pre-refactor code.
    """
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
    # `lots` retained for legacy callers but NEVER summed across trades (see
    # TradePath docstring + F-1 audit). The simulation below aggregates
    # `contracts` per instrument and applies the ceiling once on the total.
    contracts_per_trade = 1
    lots = lots_for_position(params["instrument"], contracts_per_trade)
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
        entry_price_f = float(entry_price)
        stop_price_f = float(stop_price)
        risk_points = abs(entry_price_f - stop_price_f)
        if max_orb_size_pts is not None and risk_points >= float(max_orb_size_pts):
            continue
        risk_dollars = risk_in_dollars(cost_spec, entry_price_f, stop_price_f)
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
                risk_points=float(risk_points),
                contracts=contracts_per_trade,
                instrument=params["instrument"],
            )
        )
    return trades


def _load_lane_trade_paths(
    con: duckdb.DuckDBPyConnection,
    strategy_id: str,
    *,
    as_of_date: date,
    effective_stop_multiplier: float | None = None,
    max_orb_size_pts: float | None = None,
    size_model: SizingContext | None = None,
) -> list[TradePath]:
    """Load one lane's trade history in dollars from canonical outcomes.

    When ``size_model`` is None (all current callers) behaviour is byte-identical
    to the pre-refactor code: 1-contract, unscaled TradePaths returned as-is.

    When a ``SizingContext`` is supplied the trades are vol-scaled via the
    canonical sizer (``compute_vol_scalar`` / ``compute_position_size_vol_scaled``
    from ``trading_app.portfolio``) and ALL contract-derived TradePath fields
    (contracts, pnl_dollars, mae_dollars, mfe_dollars, lots) scale together.
    """
    params = _load_strategy_snapshot(con, strategy_id)
    trades = _build_trade_paths_from_outcomes(
        con,
        params,
        as_of_date=as_of_date,
        effective_stop_multiplier=effective_stop_multiplier,
        max_orb_size_pts=max_orb_size_pts,
        strategy_id=strategy_id,
    )
    if size_model is None:
        return trades
    if size_model.account_equity <= 0:
        raise ValueError(
            f"SizingContext.account_equity must be > 0 (got {size_model.account_equity}); "
            "a zero-equity drawdown projection is never a valid survival PASS"
        )
    instrument = params["instrument"]
    cost_spec = get_cost_spec(instrument)
    cap = size_model.max_contracts_for(strategy_id)
    days = {t.trading_day for t in trades}
    atr_by_day = _lane_atr_by_day(con, instrument, params["orb_minutes"], days)
    median_by_day = _lane_median_atr(con, instrument, days)
    xfa_cap = max_lots_for_xfa(size_model.account_size, size_model.account_equity)
    scaled: list[TradePath] = []
    for t in trades:
        # Size on the PLANNED entry-to-stop distance the live engine sizes on
        # (carried on the TradePath from _build_trade_paths_from_outcomes), NOT on
        # realized MAE. Deriving risk_points from mae_dollars under-states the stop
        # for winners (mae_r <= 1.0) and over-sizes contracts → too-optimistic
        # survival PASS (D-3 fix). cost_spec is still bound above and passed to the
        # sizer below; only the mae-based risk_points derivation is removed.
        risk_points = t.risk_points
        atr = atr_by_day.get(t.trading_day)
        med = median_by_day.get(t.trading_day)
        if atr and med and atr > 0 and med > 0:
            vol_scalar = compute_vol_scalar(atr, med)
        else:
            vol_scalar = 1.0
            log.warning(
                "survival sizing: vol_scalar=1.0 fallback for %s %s (atr=%s med=%s)",
                instrument,
                t.trading_day,
                atr,
                med,
            )
        n = compute_position_size_vol_scaled(
            size_model.account_equity,
            size_model.risk_per_trade_pct,
            risk_points,
            cost_spec,
            vol_scalar,
        )
        # Floor a REAL historical trade at 1 contract: the engine sizer returns 0
        # whenever the stop is wide enough that the per-trade risk budget buys zero
        # contracts (NOT only for zero-risk trades — measured). A historical trade
        # DID occur, so the survival sim must carry its DD contribution; zeroing it
        # would silently understate drawdown. At cap=1 this makes every field x1 =
        # byte-identical to today (spec § "contracts_per_trade = 1"), and at a lifted
        # cap it scales up only as far as both the lane cap and the XFA cap allow.
        hard_cap = min(cap, xfa_cap)
        n = max(1, min(n, hard_cap)) if hard_cap >= 1 else 0
        base_contracts = t.contracts or 1
        scaled.append(
            replace(
                t,
                contracts=base_contracts * n,
                pnl_dollars=t.pnl_dollars * n,
                mae_dollars=t.mae_dollars * n,
                mfe_dollars=t.mfe_dollars * n,
                lots=lots_for_position(instrument, base_contracts * n),
            )
        )
    return scaled


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
    # Track raw contract count per instrument so max_open_lots can be computed
    # via canonical aggregate-then-ceiling (not buggy ceiling-then-sum). See
    # docs/audit/2026-04-11-criterion-11-f1-false-alarm.md for the root cause
    # of the prior over-counting and why per-trade lot aggregation is wrong.
    open_contracts_by_inst: dict[str, int] = {}
    max_open_lots = 0
    min_delta = 0.0
    max_delta = 0.0

    def _current_total_lots() -> int:
        return sum(lots_for_position(inst, n) for inst, n in open_contracts_by_inst.items() if n > 0)

    for _ts, event_type, trade in events:
        if event_type == 0:
            open_trades.append(trade)
            if trade.instrument and trade.contracts > 0:
                open_contracts_by_inst[trade.instrument] = (
                    open_contracts_by_inst.get(trade.instrument, 0) + trade.contracts
                )
                max_open_lots = max(max_open_lots, _current_total_lots())
        else:
            realized_pnl += trade.pnl_dollars
            for idx, open_trade in enumerate(open_trades):
                if open_trade is trade:
                    del open_trades[idx]
                    break
            if trade.instrument and trade.contracts > 0:
                open_contracts_by_inst[trade.instrument] = max(
                    0,
                    open_contracts_by_inst.get(trade.instrument, 0) - trade.contracts,
                )

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

    con = open_read_only_with_retry(str(db))
    configure_connection(con)
    try:
        instruments = sorted({lane["instrument"] for lane in lane_defs})
        from trading_app.prop_profiles import load_allocation_lanes

        lane_specs = profile.daily_lanes
        if not lane_specs:
            lane_specs = load_allocation_lanes(profile.profile_id)
        effective_stop_by_strategy = {
            lane.strategy_id: float(lane.planned_stop_multiplier or profile.stop_multiplier) for lane in lane_specs
        }

        _pf = build_profile_portfolio(profile_id=profile_id)
        size_model = SizingContext(
            account_equity=_pf.account_equity,
            risk_per_trade_pct=_pf.risk_per_trade_pct,
            account_size=profile.account_size,
            max_contracts_by_strategy={s.strategy_id: s.max_contracts for s in _pf.strategies},
        )

        return _scenarios_for_context(
            con,
            profile=profile,
            lane_defs=lane_defs,
            instruments=instruments,
            effective_stop_by_strategy=effective_stop_by_strategy,
            as_of_date=as_of_date,
            size_model=size_model,
        )
    finally:
        con.close()


def _scenarios_for_context(
    con,
    *,
    profile: AccountProfile,
    lane_defs: list[dict],
    instruments: list[str],
    effective_stop_by_strategy: dict[str, float],
    as_of_date: date,
    size_model: SizingContext,
) -> tuple[list[DailyScenario], dict]:
    """Build common-support daily scenarios for one profile under a SizingContext.

    Pure extract of the lane->scenario body of ``_load_profile_daily_scenarios``
    (D-3 seam Stage 1). Factored so the contract cap can be varied via
    ``size_model`` (cap=1 reconciles to the known live DD; cap>1 exercises the
    seam's scaling) without duplicating the calendar/scenario logic.
    """
    lane_first_days: dict[str, date] = {}
    trades_by_day: dict[date, list[TradePath]] = {}

    for lane in lane_defs:
        trade_paths = _load_lane_trade_paths(
            con,
            lane["strategy_id"],
            as_of_date=as_of_date,
            effective_stop_multiplier=effective_stop_by_strategy.get(lane["strategy_id"]),
            max_orb_size_pts=lane.get("max_orb_size_pts"),
            size_model=size_model,
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
        raise ValueError(f"Profile {profile.profile_id!r} has no common-support daily scenarios")

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

    daily_loss_limit = profile.daily_loss_dollars
    if daily_loss_limit is None:
        daily_loss_limit = float(tier.daily_loss_limit) if tier.daily_loss_limit is not None else None

    return SurvivalRules(
        profile_id=profile.profile_id,
        firm=profile.firm,
        account_size=profile.account_size,
        dd_type=firm_spec.dd_type,
        starting_balance=starting_balance,
        dd_limit_dollars=float(tier.max_dd),
        daily_loss_limit=float(daily_loss_limit) if daily_loss_limit is not None else None,
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


def _historical_daily_loss_breach_days(scenarios: list[DailyScenario], rules: SurvivalRules) -> list[str]:
    """Return historical days that would trip the effective daily-loss belt."""
    if rules.daily_loss_limit is None:
        return []
    breach_days: list[str] = []
    for scenario in scenarios:
        day_min_delta = min(scenario.min_balance_delta_dollars, scenario.total_pnl_dollars)
        if day_min_delta <= -rules.daily_loss_limit:
            breach_days.append(scenario.trading_day)
    return breach_days


def _max_observed_rolling_drawdown(scenarios: list[DailyScenario], *, horizon_days: int) -> float:
    """Return max observed close-to-close drawdown over rolling horizon windows."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    max_dd = 0.0
    for start_idx in range(len(scenarios)):
        balance = 0.0
        peak = 0.0
        for scenario in scenarios[start_idx : start_idx + horizon_days]:
            balance += scenario.total_pnl_dollars
            if balance > peak:
                peak = balance
            dd_used = peak - balance
            if dd_used > max_dd:
                max_dd = dd_used
    return round(max_dd, 2)


@dataclass(frozen=True)
class _GateEvaluation:
    """Canonical Criterion 11 gate verdict for one (scenarios, rules, sim-result).

    Single source of truth for "does this configuration survive": both
    ``evaluate_profile_survival`` and ``sweep_survival_cap`` resolve the gate
    through ``_evaluate_gate`` so the sweep can NEVER drift from the real
    evaluation's pass criteria (institutional-rigor § 4 — one definition, never
    re-encode).
    """

    gate_pass: bool
    operational_gate_pass: bool
    strict_account_gate_pass: bool
    operational_pass_probability: float
    effective_dd_budget_dollars: float
    historical_daily_loss_breach_days: list[str]
    historical_max_observed_90d_dd_dollars: float


def _evaluate_gate(
    scenarios: list[DailyScenario],
    rules: SurvivalRules,
    result: dict,
    profile: AccountProfile,
    *,
    min_survival_probability: float,
    sizing_parity_ok: bool,
) -> _GateEvaluation:
    """Resolve the canonical C11 gate verdict for one simulated configuration.

    Pure extract of the verdict math previously inline in
    ``evaluate_profile_survival`` (D-3 survival-cap sweep Stage 1). The sizing-
    parity gate is passed in (not recomputed here) because parity is a per-
    PROFILE property — it does not vary with the swept contract count, so the
    sweep evaluates it once and reuses the verdict across all ``n``.
    """
    operational_pass_probability = round(result["operational_pass_probability"], 4)
    effective_dd_budget_dollars = effective_strict_dd_budget(profile, rules)
    historical_daily_loss_breach_days = _historical_daily_loss_breach_days(scenarios, rules)
    historical_max_observed_90d_dd_dollars = _max_observed_rolling_drawdown(
        scenarios,
        horizon_days=STRICT_DD_HORIZON_DAYS,
    )
    operational_gate_pass = operational_pass_probability >= float(min_survival_probability)
    strict_account_gate_pass = (
        len(historical_daily_loss_breach_days) == 0
        and historical_max_observed_90d_dd_dollars <= effective_dd_budget_dollars
    )
    gate_pass = operational_gate_pass and strict_account_gate_pass and sizing_parity_ok
    return _GateEvaluation(
        gate_pass=gate_pass,
        operational_gate_pass=operational_gate_pass,
        strict_account_gate_pass=strict_account_gate_pass,
        operational_pass_probability=operational_pass_probability,
        effective_dd_budget_dollars=effective_dd_budget_dollars,
        historical_daily_loss_breach_days=historical_daily_loss_breach_days,
        historical_max_observed_90d_dd_dollars=historical_max_observed_90d_dd_dollars,
    )


def _assert_sizing_parity(profile_id: str) -> tuple[bool, str]:
    """D-3 sizing-parity guard for Criterion 11 (D-3 seam Stage 1, 2026-06-07).

    The survival sim now sizes each trade like the LIVE execution engine
    (vol-scaled off equity/risk/volatility, clamped to ``max_contracts`` — see
    ``_load_lane_trade_paths`` + ``SizingContext``). It therefore no longer needs
    to FORBID ``max_contracts > 1``: the honest sim fails the operational gate at
    unsafe size on its own (measured 2026-06-07: ``operational_pass_prob`` drops
    0.99 -> 0.10 at cap=2). The old forbid-cap>1 handcuff is gone.

    The guard's remaining, narrower job is to prove the sim CAN size like the
    engine: the profile portfolio builds, and the equity that feeds the sizer is
    positive. It fails CLOSED only when parity is UNPROVABLE:
      * ``build_profile_portfolio`` raises (cannot construct the live sizing inputs), or
      * ``account_equity <= 0`` — express-funded XFA accounts present
        ``starting_balance == 0.0``; a $0-equity DD projection zeros the sizer and
        would yield a FALSE PASS. (The sizer reads the Portfolio NOTIONAL equity,
        never ``starting_balance`` — see ``SizingContext``.)

    Returns ``(ok, message)``; ``ok=False`` must fail the C11 gate closed.
    """
    try:
        portfolio = build_profile_portfolio(profile_id=profile_id)
    except Exception as e:  # fail closed — cannot prove sizing parity
        return (False, f"sizing-parity unprovable: build_profile_portfolio failed: {e}")

    if portfolio.account_equity <= 0:
        return (
            False,
            "D-3 sizing parity FAILED: portfolio account_equity="
            f"{portfolio.account_equity} <= 0 would zero the sizer and project "
            "DD=$0 (a false PASS).",
        )
    return (
        True,
        f"sizing parity OK ({len(portfolio.strategies)} lanes; equity={portfolio.account_equity})",
    )


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
    rules = _with_consistency_rule(_build_rules(profile), profile)
    result = simulate_survival(scenarios, rules, horizon_days=horizon_days, n_paths=n_paths, seed=seed)
    # D-3 sizing-parity guard (seam Stage 1): the survival sim now sizes like the
    # live engine (vol-scaled, capped). Fail the gate closed only when parity is
    # unprovable — the portfolio can't be built or its equity is non-positive (a
    # $0-equity express account would zero the sizer and project DD=$0 = false PASS).
    sizing_parity_ok, sizing_parity_msg = _assert_sizing_parity(resolved_profile_id)
    if not sizing_parity_ok:
        log.warning("Criterion 11 sizing-parity guard: %s", sizing_parity_msg)
    gate = _evaluate_gate(
        scenarios,
        rules,
        result,
        profile,
        min_survival_probability=min_survival_probability,
        sizing_parity_ok=sizing_parity_ok,
    )
    operational_pass_probability = gate.operational_pass_probability
    effective_dd_budget_dollars = gate.effective_dd_budget_dollars
    historical_daily_loss_breach_days = gate.historical_daily_loss_breach_days
    historical_max_observed_90d_dd_dollars = gate.historical_max_observed_90d_dd_dollars
    strict_account_gate_pass = gate.strict_account_gate_pass
    gate_pass = gate.gate_pass

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
        strict_account_gate_pass=strict_account_gate_pass,
        effective_dd_budget_dollars=effective_dd_budget_dollars,
        historical_daily_loss_breach_days=historical_daily_loss_breach_days,
        historical_daily_loss_breach_count=len(historical_daily_loss_breach_days),
        historical_max_observed_90d_dd_dollars=historical_max_observed_90d_dd_dollars,
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
        envelope = build_state_envelope(
            schema_version=CRITERION11_STATE_SCHEMA_VERSION,
            state_type=CRITERION11_STATE_TYPE,
            tool="account_survival",
            canonical_inputs=_current_survival_canonical_inputs(resolved_profile_id, db_path=db_path),
            freshness={
                "as_of_date": str(as_of_date),
                "max_age_days": DEFAULT_REPORT_MAX_AGE_DAYS,
            },
            payload={
                "summary": asdict(summary),
                "rules": asdict(rules),
                "metadata": {
                    "source_start": metadata["source_start"],
                    "source_end": metadata["source_end"],
                    "source_days": metadata["source_days"],
                    "instruments": metadata["instruments"],
                },
            },
            git_head=get_git_head(Path(__file__).resolve().parents[1]),
        )
        out_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

    return summary


# Maximum contract ceiling the sweep will probe. The sweep is bounded (Bailey-
# style — never an unbounded search); 1..DEFAULT_SWEEP_CEILING covers the
# realistic micro-futures lane sizes. Non-binding while DEPLOYED_MAX_CONTRACTS_CLAMP=1.
DEFAULT_SWEEP_CEILING = 5


def _contiguous_safe_ceiling(per_cap: list[dict]) -> int:
    """Largest contract cap CONTIGUOUS from 1 whose gate passes (fail-closed).

    ``per_cap`` is the ordered (1..ceiling) list of per-cap gate results. Survival
    is NOT guaranteed monotonic in cap once a strict-DD belt trips, so a pass ABOVE
    a failure cannot be honored — only the unbroken passing run starting at cap=1
    counts. If even cap=1 fails, returns 0: a fail-closed signal the operator must
    see (the profile cannot survive at ANY size).
    """
    ceiling = 0
    for entry in per_cap:
        if entry.get("gate_pass"):
            ceiling = int(entry["contracts"])
        else:
            break
    return ceiling


@dataclass(frozen=True)
class SurvivalCapSweepResult:
    """Result of probing contract caps {1..ceiling} against the C11 survival gate.

    ``survival_safe_ceiling`` is the LARGEST contract count, contiguous from 1,
    for which the gate still PASSes. Fail-closed to 1: if even cap=1 fails, or any
    cap below the first failure passes non-contiguously, only the contiguous-from-1
    passing run is honored (a gap means a higher cap that "passes" cannot be
    trusted — survival is not monotonic in cap once a strict-DD belt trips).
    """

    profile_id: str
    ceiling_probed: int
    survival_safe_ceiling: int
    sizing_parity_ok: bool
    sizing_parity_msg: str
    per_cap: list[dict]  # [{contracts, gate_pass, operational_pass_probability, ...}]
    horizon_days: int = 90
    n_paths: int = 10_000
    seed: int = 0
    min_survival_probability: float = MIN_SURVIVAL_PROBABILITY
    as_of_date: str = ""


def sweep_survival_cap(
    profile_id: str | None = None,
    *,
    ceiling: int = DEFAULT_SWEEP_CEILING,
    as_of_date: date | None = None,
    horizon_days: int = 90,
    n_paths: int = 10_000,
    seed: int = 0,
    db_path: Path | None = None,
    min_survival_probability: float = MIN_SURVIVAL_PROBABILITY,
    write_state: bool = True,
) -> SurvivalCapSweepResult:
    """Probe contract caps {1..ceiling} and persist the survival-safe ceiling.

    On-demand only (NOT folded into ``evaluate_profile_survival`` — it reruns the
    sim per ``n``, ``ceiling``x the cost). For each ``n`` it rebuilds the SAME
    historical scenarios under ``SizingContext(max_contracts_by_strategy={sid: n})``
    and runs the canonical ``simulate_survival`` + ``_evaluate_gate`` — it
    re-encodes ZERO sim or gate math (institutional-rigor § 4). The result is the
    largest cap, contiguous from 1, whose gate still PASSes.

    At ``DEPLOYED_MAX_CONTRACTS_CLAMP=1`` this is pure deploy-readiness EVIDENCE —
    the live order path is clamped to 1 regardless of the swept ceiling. The swept
    ceiling is persisted into the existing Criterion 11 state envelope so the drift
    guard can prove any future non-1 live cap traces to a swept-and-passed value.
    """
    if ceiling < 1:
        raise ValueError(f"ceiling must be >= 1, got {ceiling}")
    if as_of_date is None:
        as_of_date = date.today()

    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    profile = get_profile(resolved_profile_id)
    rules = _with_consistency_rule(_build_rules(profile), profile)

    # Sizing-parity is a per-PROFILE property (does the portfolio build with
    # positive equity) — invariant to the swept cap. Evaluate once, reuse across n.
    sizing_parity_ok, sizing_parity_msg = _assert_sizing_parity(resolved_profile_id)
    if not sizing_parity_ok:
        log.warning("Criterion 11 sizing-parity guard (sweep): %s", sizing_parity_msg)

    lane_defs = get_profile_lane_definitions(resolved_profile_id)
    instruments = sorted({lane["instrument"] for lane in lane_defs})

    from trading_app.prop_profiles import load_allocation_lanes

    lane_specs = profile.daily_lanes or load_allocation_lanes(resolved_profile_id)
    effective_stop_by_strategy = {
        lane.strategy_id: float(lane.planned_stop_multiplier or profile.stop_multiplier) for lane in lane_specs
    }
    _pf = build_profile_portfolio(profile_id=resolved_profile_id)
    strategy_ids = [s.strategy_id for s in _pf.strategies]

    db = db_path or GOLD_DB_PATH
    con = open_read_only_with_retry(str(db))
    configure_connection(con)
    per_cap: list[dict] = []
    try:
        for n in range(1, ceiling + 1):
            size_model = SizingContext(
                account_equity=_pf.account_equity,
                risk_per_trade_pct=_pf.risk_per_trade_pct,
                account_size=profile.account_size,
                max_contracts_by_strategy={sid: n for sid in strategy_ids},
            )
            scenarios, _meta = _scenarios_for_context(
                con,
                profile=profile,
                lane_defs=lane_defs,
                instruments=instruments,
                effective_stop_by_strategy=effective_stop_by_strategy,
                as_of_date=as_of_date,
                size_model=size_model,
            )
            result = simulate_survival(scenarios, rules, horizon_days=horizon_days, n_paths=n_paths, seed=seed)
            gate = _evaluate_gate(
                scenarios,
                rules,
                result,
                profile,
                min_survival_probability=min_survival_probability,
                sizing_parity_ok=sizing_parity_ok,
            )
            per_cap.append(
                {
                    "contracts": n,
                    "gate_pass": gate.gate_pass,
                    "operational_gate_pass": gate.operational_gate_pass,
                    "strict_account_gate_pass": gate.strict_account_gate_pass,
                    "operational_pass_probability": gate.operational_pass_probability,
                    "historical_max_observed_90d_dd_dollars": gate.historical_max_observed_90d_dd_dollars,
                    "effective_dd_budget_dollars": gate.effective_dd_budget_dollars,
                }
            )
    finally:
        con.close()

    survival_safe_ceiling = _contiguous_safe_ceiling(per_cap)

    sweep = SurvivalCapSweepResult(
        profile_id=resolved_profile_id,
        ceiling_probed=ceiling,
        survival_safe_ceiling=survival_safe_ceiling,
        sizing_parity_ok=sizing_parity_ok,
        sizing_parity_msg=sizing_parity_msg,
        per_cap=per_cap,
        horizon_days=horizon_days,
        n_paths=n_paths,
        seed=seed,
        min_survival_probability=float(min_survival_probability),
        as_of_date=str(as_of_date),
    )

    if write_state:
        _persist_sweep_into_c11_envelope(resolved_profile_id, sweep, db_path=db_path)

    return sweep


@dataclass(frozen=True)
class PerAccountSurvivalResult:
    """Per-account C11 survival verdicts for a non-uniform account_contracts map.

    Stage 3a — before a non-uniform per-account contract map may arm a live cap
    > 1, each account's divergent belt must be proven safe at its OWN contract
    scale. ``per_account`` carries one verdict dict per live account_id; verdicts
    are computed once per DISTINCT contract count (the gate is a function of the
    contract scale, not the account_id) and fanned back out, so accounts sharing a
    count share a verdict. Read-only EVIDENCE: this does NOT persist and does NOT
    lift ``DEPLOYED_MAX_CONTRACTS_CLAMP`` — the live order path stays clamped at 1
    until the operator GOes on the evidence.
    """

    profile_id: str
    account_contracts: dict[int, int]
    per_account: list[dict]  # [{account_id, contracts, gate_pass, operational_pass_probability, ...}]
    all_pass: bool
    sizing_parity_ok: bool
    sizing_parity_msg: str
    horizon_days: int = 90
    n_paths: int = 10_000
    seed: int = 0
    min_survival_probability: float = MIN_SURVIVAL_PROBABILITY
    as_of_date: str = ""


def evaluate_per_account_survival(
    profile_id: str | None,
    account_contracts: dict[int, int],
    *,
    as_of_date: date | None = None,
    horizon_days: int = 90,
    n_paths: int = 10_000,
    seed: int = 0,
    db_path: Path | None = None,
    min_survival_probability: float = MIN_SURVIVAL_PROBABILITY,
) -> PerAccountSurvivalResult:
    """Run C11 survival for each account in a per-account contract map (Stage 3a).

    For a non-uniform ``account_contracts`` (``{account_id: contracts}``), each
    account trades the SAME lanes at a DIFFERENT contract count, so its modeled
    drawdown — and therefore its Stage-2 daily-loss belt — diverges. This proves
    each divergent belt is survivable BEFORE any clamp lift could arm it live.

    Delegates entirely to the canonical chain ``_scenarios_for_context`` ->
    ``simulate_survival`` -> ``_evaluate_gate`` (the same machinery
    ``sweep_survival_cap`` uses), keyed on the DISTINCT contract counts present in
    the map — re-encoding ZERO sim or gate math (institutional-rigor § 4). The
    contract scale enters via ``SizingContext`` (NOT the vestigial
    ``SurvivalRules.contracts_per_trade_micro``).

    Read-only: no state is persisted; ``DEPLOYED_MAX_CONTRACTS_CLAMP`` is untouched.
    """
    if not account_contracts:
        raise ValueError("evaluate_per_account_survival: account_contracts must be non-empty")
    for aid, n in account_contracts.items():
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"evaluate_per_account_survival: account_contracts[{aid}]={n!r} must be int >= 1")
    if as_of_date is None:
        as_of_date = date.today()

    resolved_profile_id = resolve_profile_id(profile_id, active_only=False, exclude_self_funded=False)
    profile = get_profile(resolved_profile_id)
    rules = _with_consistency_rule(_build_rules(profile), profile)

    # Sizing-parity is a per-PROFILE property (does the portfolio build with
    # positive equity) — invariant to the contract count. Evaluate once, reuse.
    sizing_parity_ok, sizing_parity_msg = _assert_sizing_parity(resolved_profile_id)
    if not sizing_parity_ok:
        log.warning("Criterion 11 sizing-parity guard (per-account): %s", sizing_parity_msg)

    lane_defs = get_profile_lane_definitions(resolved_profile_id)
    instruments = sorted({lane["instrument"] for lane in lane_defs})

    from trading_app.prop_profiles import load_allocation_lanes

    lane_specs = profile.daily_lanes or load_allocation_lanes(resolved_profile_id)
    effective_stop_by_strategy = {
        lane.strategy_id: float(lane.planned_stop_multiplier or profile.stop_multiplier) for lane in lane_specs
    }
    _pf = build_profile_portfolio(profile_id=resolved_profile_id)
    strategy_ids = [s.strategy_id for s in _pf.strategies]

    db = db_path or GOLD_DB_PATH
    con = open_read_only_with_retry(str(db))
    configure_connection(con)
    # Compute one verdict per DISTINCT contract count, then fan out to accounts.
    verdict_by_count: dict[int, dict] = {}
    try:
        for n in sorted(set(account_contracts.values())):
            size_model = SizingContext(
                account_equity=_pf.account_equity,
                risk_per_trade_pct=_pf.risk_per_trade_pct,
                account_size=profile.account_size,
                max_contracts_by_strategy={sid: n for sid in strategy_ids},
            )
            scenarios, _meta = _scenarios_for_context(
                con,
                profile=profile,
                lane_defs=lane_defs,
                instruments=instruments,
                effective_stop_by_strategy=effective_stop_by_strategy,
                as_of_date=as_of_date,
                size_model=size_model,
            )
            result = simulate_survival(scenarios, rules, horizon_days=horizon_days, n_paths=n_paths, seed=seed)
            gate = _evaluate_gate(
                scenarios,
                rules,
                result,
                profile,
                min_survival_probability=min_survival_probability,
                sizing_parity_ok=sizing_parity_ok,
            )
            verdict_by_count[n] = {
                "contracts": n,
                "gate_pass": gate.gate_pass,
                "operational_gate_pass": gate.operational_gate_pass,
                "strict_account_gate_pass": gate.strict_account_gate_pass,
                "operational_pass_probability": gate.operational_pass_probability,
                "historical_max_observed_90d_dd_dollars": gate.historical_max_observed_90d_dd_dollars,
                "effective_dd_budget_dollars": gate.effective_dd_budget_dollars,
            }
    finally:
        con.close()

    per_account: list[dict] = []
    for aid in sorted(account_contracts):
        n = account_contracts[aid]
        per_account.append({"account_id": aid, **verdict_by_count[n]})
    all_pass = all(v["gate_pass"] for v in per_account)

    return PerAccountSurvivalResult(
        profile_id=resolved_profile_id,
        account_contracts=dict(account_contracts),
        per_account=per_account,
        all_pass=all_pass,
        sizing_parity_ok=sizing_parity_ok,
        sizing_parity_msg=sizing_parity_msg,
        horizon_days=horizon_days,
        n_paths=n_paths,
        seed=seed,
        min_survival_probability=float(min_survival_probability),
        as_of_date=str(as_of_date),
    )


def _persist_sweep_into_c11_envelope(
    profile_id: str,
    sweep: SurvivalCapSweepResult,
    *,
    db_path: Path | None = None,
) -> None:
    """Merge the swept-ceiling result into the existing C11 state envelope.

    Additive: writes a ``survival_cap_sweep`` block into the existing
    ``account_survival_<profile>.json`` payload, leaving the ``summary`` /
    ``rules`` / ``metadata`` produced by ``evaluate_profile_survival`` intact. The
    envelope is RE-STAMPED through ``build_state_envelope`` so its canonical_inputs
    fingerprints (db_identity, code_fingerprint, profile_fingerprint) cover the
    moment the sweep was computed — a stale sweep then invalidates exactly like a
    stale summary. Requires the base report to exist (the sweep is supplementary
    evidence ON a profile that already has a survival report); raises otherwise so
    a missing base report is never silently masked.
    """
    report_path = get_survival_report_path(profile_id)
    if not report_path.exists():
        raise FileNotFoundError(
            f"cannot persist sweep: base C11 report missing for {profile_id!r} "
            f"({report_path}). Run evaluate_profile_survival first."
        )
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"cannot persist sweep: C11 report for {profile_id!r} has no versioned payload")

    payload = dict(payload)
    payload["survival_cap_sweep"] = {
        "ceiling_probed": sweep.ceiling_probed,
        "survival_safe_ceiling": sweep.survival_safe_ceiling,
        "sizing_parity_ok": sweep.sizing_parity_ok,
        "horizon_days": sweep.horizon_days,
        "n_paths": sweep.n_paths,
        "seed": sweep.seed,
        "min_survival_probability": sweep.min_survival_probability,
        "as_of_date": sweep.as_of_date,
        "per_cap": sweep.per_cap,
        "computed_at_utc": datetime.now(UTC).isoformat(),
    }

    freshness = raw.get("freshness", {})
    as_of = freshness.get("as_of_date") if isinstance(freshness, dict) else None
    envelope = build_state_envelope(
        schema_version=CRITERION11_STATE_SCHEMA_VERSION,
        state_type=CRITERION11_STATE_TYPE,
        tool="account_survival",
        canonical_inputs=_current_survival_canonical_inputs(profile_id, db_path=db_path),
        freshness={
            "as_of_date": as_of or str(date.today()),
            "max_age_days": DEFAULT_REPORT_MAX_AGE_DAYS,
        },
        payload=payload,
        git_head=get_git_head(Path(__file__).resolve().parents[1]),
    )
    report_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")


def check_survival_report_gate(
    profile_id: str | None = None,
    *,
    db_path: Path | None = None,
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

    state = read_survival_report_state(profile_id, db_path=db_path, today=today)
    if not bool(state["valid"]):
        raw_reason = state.get("reason")
        reason = str(raw_reason or "invalid state")
        _, guidance = classify_state_reason(str(raw_reason) if raw_reason is not None else None)
        return False, f"BLOCKED: Criterion 11 state {reason}. {guidance}"

    summary = state["summary"]
    if not isinstance(summary, dict):
        # valid envelope but malformed payload — a structural DEFECT, not routine
        # staleness. Route through the classifier (which returns DEFECT for this
        # unrecognized reason) so the wording matches its sibling above.
        _, guidance = classify_state_reason("missing summary payload")
        return False, f"BLOCKED: Criterion 11 state missing summary payload. {guidance}"

    try:
        as_of_date = date.fromisoformat(str(summary["as_of_date"]))
        report_age_days = (today - as_of_date).days
        operational_pass = float(summary["operational_pass_probability"])
        horizon_days = int(summary["horizon_days"])
        n_paths = int(summary["n_paths"])
        scaling_feasible = bool(summary.get("scaling_feasible", False))
        intraday_approximated = bool(summary.get("intraday_approximated", False))
        path_model = str(summary.get("path_model", "daily_close"))
        gate_pass = bool(summary["gate_pass"])
        strict_account_gate_pass = bool(summary["strict_account_gate_pass"])
        historical_daily_loss_breach_count = int(summary["historical_daily_loss_breach_count"])
        effective_dd_budget_dollars = float(summary["effective_dd_budget_dollars"])
        historical_max_observed_90d_dd_dollars = float(summary["historical_max_observed_90d_dd_dollars"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"BLOCKED: unreadable Criterion 11 survival summary ({exc})"

    if horizon_days != 90:
        return False, f"BLOCKED: Criterion 11 report horizon is {horizon_days}d, expected 90d"
    if n_paths < 10_000:
        return False, f"BLOCKED: Criterion 11 report uses {n_paths} paths, expected >= 10000"
    if report_age_days > max_age_days:
        return (
            False,
            f"BLOCKED: Criterion 11 report is {report_age_days}d old (> {max_age_days}d). Re-run account survival.",
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
    strict_failures: list[str] = []
    if historical_daily_loss_breach_count > 0:
        strict_failures.append(f"historical_daily_loss_days={historical_daily_loss_breach_count}")
    if historical_max_observed_90d_dd_dollars > effective_dd_budget_dollars:
        strict_failures.append(
            f"max_90d_dd=${historical_max_observed_90d_dd_dollars:,.0f}/${effective_dd_budget_dollars:,.0f}"
        )
    if not strict_account_gate_pass or strict_failures:
        detail = ", ".join(strict_failures) if strict_failures else "strict_account_gate_pass=false"
        return False, f"BLOCKED: Criterion 11 strict account diagnostics failed ({detail}). Re-run account survival."
    if not gate_pass:
        return False, "BLOCKED: Criterion 11 account-survival gate failed. Re-run account survival."
    return (
        True,
        f"Criterion 11 pass: operational {operational_pass:.1%}, as_of={as_of_date}, "
        f"age={report_age_days}d, paths={n_paths}, strict_account=PASS",
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
        + (f"{summary.consistency_pass_probability:.1%}" if summary.consistency_pass_probability is not None else "n/a")
    )
    print(
        f"Breach rates: trailing_dd={summary.trailing_dd_breach_probability:.1%} | "
        f"daily_loss={summary.daily_loss_breach_probability:.1%} | "
        f"scaling={summary.scaling_breach_probability:.1%} | "
        f"consistency={summary.consistency_breach_probability:.1%}"
    )
    print("Expectancy edge: not evaluated by Criterion 11 account survival")
    print(
        f"Strict diagnostics: effective_dd_budget=${summary.effective_dd_budget_dollars:,.0f} | "
        f"historical_daily_loss_breaches={summary.historical_daily_loss_breach_count} | "
        f"historical_max_observed_90d_dd=${summary.historical_max_observed_90d_dd_dollars:,.0f}"
    )
    print(f"Prop-account path safety={'PASS' if summary.strict_account_gate_pass else 'FAIL'}")
    print(f"Final deployability gate={'PASS' if summary.gate_pass else 'FAIL'}")
    if summary.historical_daily_loss_breach_days:
        print(
            f"Historical daily-loss breach days ({summary.historical_daily_loss_breach_count}): "
            + ", ".join(summary.historical_daily_loss_breach_days)
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
    if not summary.gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
