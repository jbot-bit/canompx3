"""
Prop firm compliance monitors — consistency rule, payout eligibility,
idle warnings, microscalp compliance.

All read-only trackers. No order logic. Reads from paper_trades table.
Integrates into weekly_review and pre_session_check.
"""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.prop_firm_policies import get_default_payout_policy_for_firm, get_payout_policy

log = logging.getLogger(__name__)

_PAPER_TRADES_TABLE = "paper_trades"


@dataclass
class ConsistencyResult:
    best_day_date: str
    best_day_pnl: float
    total_profit: float
    windfall_pct: float
    limit_pct: float
    status: str  # "OK" | "WARN" | "BREACH"


@dataclass
class PayoutEligibility:
    policy_id: str | None
    policy_status: str
    trading_days: int
    min_trading_days: int
    profitable_days_50: int  # days with >= profit_threshold
    profit_threshold: float
    min_profitable_days: int
    eligible: bool
    notes: list[str]


@dataclass
class MicroscalpResult:
    total_trades: int
    trades_over_10s: int
    pct_trades_over_10s: float
    profit_from_over_10s: float
    total_profit: float
    pct_profit_from_over_10s: float
    compliant: bool


def _get_paper_trades_db() -> Path:
    """Return path to paper_trades DB (gold.db — shared with pipeline)."""
    return GOLD_DB_PATH


def _has_paper_trades(con) -> bool:
    """Check if paper_trades table exists."""
    try:
        con.execute(f"SELECT 1 FROM {_PAPER_TRADES_TABLE} LIMIT 0")
        return True
    except duckdb.CatalogException:
        return False


def check_consistency(
    firm: str,
    instrument: str | None = None,
    db_path: Path | None = None,
    policy_id: str | None = None,
) -> ConsistencyResult | None:
    """Check prop firm consistency rule (windfall day limit).

    Uses the payout-path rule when policy_id is provided.
    Returns None when the firm/path has no active consistency rule.
    """
    from trading_app.prop_profiles import PROP_FIRM_SPECS

    spec = PROP_FIRM_SPECS.get(firm)
    limit_pct = None
    if policy_id is not None:
        policy = get_payout_policy(policy_id)
        limit_pct = policy.consistency_rule
    elif spec is not None:
        limit_pct = spec.consistency_rule

    if limit_pct is None:
        return None  # no consistency rule for this firm
    db = db_path or _get_paper_trades_db()

    with duckdb.connect(str(db), read_only=True) as con:
        configure_connection(con)
        if not _has_paper_trades(con):
            return None

        where = "WHERE pnl_dollar IS NOT NULL"
        params = []
        if instrument:
            where += " AND instrument = ?"
            params.append(instrument)

        rows = con.execute(
            f"""
            SELECT CAST(entry_time AS DATE) AS trade_date,
                   SUM(pnl_dollar) AS daily_pnl
            FROM {_PAPER_TRADES_TABLE}
            {where}
            GROUP BY 1
            ORDER BY daily_pnl DESC
            """,
            params,
        ).fetchall()

        if not rows:
            return None

        best_date, best_pnl = rows[0]
        total_profit = sum(r[1] for r in rows if r[1] > 0)

        if total_profit <= 0:
            return ConsistencyResult(
                best_day_date=str(best_date),
                best_day_pnl=round(best_pnl, 2),
                total_profit=0.0,
                windfall_pct=0.0,
                limit_pct=limit_pct,
                status="OK",
            )

        windfall = best_pnl / total_profit if total_profit > 0 else 0.0

        if windfall > limit_pct:
            status = "BREACH"
        elif windfall > limit_pct * 0.85:
            status = "WARN"
        else:
            status = "OK"

        return ConsistencyResult(
            best_day_date=str(best_date),
            best_day_pnl=round(best_pnl, 2),
            total_profit=round(total_profit, 2),
            windfall_pct=round(windfall * 100, 1),
            limit_pct=limit_pct,
            status=status,
        )


def check_payout_eligibility(
    firm: str,
    instrument: str | None = None,
    db_path: Path | None = None,
    policy_id: str | None = None,
) -> PayoutEligibility:
    """Check if payout eligibility requirements are met.

    Uses canonical payout-policy definitions where available.
    This models what can be inferred from paper_trades only. It does not
    know about prior payouts, balance-based caps, or manual approval state.
    """
    policy = get_payout_policy(policy_id) if policy_id else get_default_payout_policy_for_firm(firm)

    if policy is None:
        return PayoutEligibility(
            policy_id=None,
            policy_status="unmodeled",
            trading_days=0,
            min_trading_days=0,
            profitable_days_50=0,
            profit_threshold=0.0,
            min_profitable_days=0,
            eligible=False,
            notes=[f"No modeled payout policy for firm={firm!r}"],
        )

    min_trading_days = policy.min_trading_days or policy.winning_days_required or 0
    min_profitable_days = policy.winning_days_required or 0
    profit_threshold = policy.winning_day_profit_threshold or 0.0
    policy_status = policy.model_status

    db = db_path or _get_paper_trades_db()

    with duckdb.connect(str(db), read_only=True) as con:
        configure_connection(con)
        if not _has_paper_trades(con):
            return PayoutEligibility(
                policy.policy_id,
                policy_status,
                0,
                min_trading_days,
                0,
                profit_threshold,
                min_profitable_days,
                False,
                ["No paper_trades table"],
            )

        where = "WHERE pnl_dollar IS NOT NULL"
        params = []
        if instrument:
            where += " AND instrument = ?"
            params.append(instrument)

        rows = con.execute(
            f"""
            SELECT CAST(entry_time AS DATE) AS trade_date,
                   SUM(pnl_dollar) AS daily_pnl
            FROM {_PAPER_TRADES_TABLE}
            {where}
            GROUP BY 1
            ORDER BY 1
            """,
            params,
        ).fetchall()

        trading_days = len(rows)
        profitable_days = sum(1 for _, pnl in rows if pnl >= profit_threshold)
        total_profit = sum(pnl for _, pnl in rows)

        notes = []
        if policy_status != "complete":
            notes.append(
                f"Payout policy {policy.policy_id} is only partially modeled; eligibility is not determinable yet"
            )
        if trading_days < min_trading_days:
            notes.append(f"Need {min_trading_days - trading_days} more trading days")
        if min_profitable_days > 0 and profitable_days < min_profitable_days:
            notes.append(f"Need {min_profitable_days - profitable_days} more profitable days (${profit_threshold}+)")

        consistency_ok = True
        if policy.consistency_rule is not None:
            positive_days = [pnl for _, pnl in rows if pnl > 0]
            best_day = max(positive_days, default=0.0)
            positive_total = sum(positive_days)
            if positive_total <= 0:
                consistency_ok = False
                notes.append("Need positive net profit for consistency-based payout path")
            else:
                windfall = best_day / positive_total
                max_allowed = policy.consistency_rule
                if windfall > max_allowed:
                    consistency_ok = False
                    notes.append(
                        f"Largest day is {windfall:.1%} of positive profit, above {max_allowed:.0%} consistency limit"
                    )

        positive_balance_ok = total_profit > 0
        if not positive_balance_ok:
            notes.append("Need net profit above $0 in the current payout window")

        eligible = (
            policy_status == "complete"
            and trading_days >= min_trading_days
            and profitable_days >= min_profitable_days
            and consistency_ok
            and positive_balance_ok
        )

        if policy.payout_cap_balance_pct is not None:
            notes.append(f"Payout cap: up to {policy.payout_cap_balance_pct:.0%} of account balance")
        if policy.payout_cap_dollars is not None:
            notes.append(f"Per-request cap: ${policy.payout_cap_dollars:,.0f}")
        if policy.additional_days_after_payout is not None:
            notes.append(
                f"After payout: restart cycle and add {policy.additional_days_after_payout} new qualifying day(s)"
            )
        if policy.daily_payouts_unlock_winning_days is not None:
            notes.append(
                f"Daily payouts unlock after {policy.daily_payouts_unlock_winning_days} winning day(s) in this stage"
            )

        return PayoutEligibility(
            policy_id=policy.policy_id,
            policy_status=policy_status,
            trading_days=trading_days,
            min_trading_days=min_trading_days,
            profitable_days_50=profitable_days,
            profit_threshold=profit_threshold,
            min_profitable_days=min_profitable_days,
            eligible=eligible,
            notes=notes,
        )


def check_profile_consistency(
    profile_id: str,
    instrument: str | None = None,
    db_path: Path | None = None,
) -> ConsistencyResult | None:
    """Profile-aware consistency check using the account's payout path."""
    from trading_app.prop_profiles import get_profile

    profile = get_profile(profile_id)
    return check_consistency(
        firm=profile.firm,
        instrument=instrument,
        db_path=db_path,
        policy_id=profile.payout_policy_id,
    )


def check_profile_payout_eligibility(
    profile_id: str,
    instrument: str | None = None,
    db_path: Path | None = None,
) -> PayoutEligibility:
    """Profile-aware payout eligibility using the account's payout path."""
    from trading_app.prop_profiles import get_profile

    profile = get_profile(profile_id)
    return check_payout_eligibility(
        firm=profile.firm,
        instrument=instrument,
        db_path=db_path,
        policy_id=profile.payout_policy_id,
    )


def check_account_idle(
    instrument: str | None = None,
    db_path: Path | None = None,
) -> tuple[str, int]:
    """Check days since last trade (Tradeify idle rule: 1 trade/week).

    Returns (status, days_since_last_trade).
    Status: "OK" | "IDLE_WARNING" (6 days) | "IDLE_BREACH" (7+ days).
    """
    db = db_path or _get_paper_trades_db()

    with duckdb.connect(str(db), read_only=True) as con:
        configure_connection(con)
        if not _has_paper_trades(con):
            return "IDLE_BREACH", 999

        where = ""
        params = []
        if instrument:
            where = "WHERE instrument = ?"
            params.append(instrument)

        row = con.execute(
            f"SELECT MAX(CAST(entry_time AS DATE)) FROM {_PAPER_TRADES_TABLE} {where}",
            params,
        ).fetchone()

        if not row or row[0] is None:
            return "IDLE_BREACH", 999

        last_date = row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0]))
        days_since = (date.today() - last_date).days

        if days_since >= 7:
            return "IDLE_BREACH", days_since
        elif days_since >= 6:
            return "IDLE_WARNING", days_since
        return "OK", days_since


def check_microscalp_compliance(
    instrument: str | None = None,
    db_path: Path | None = None,
) -> MicroscalpResult | None:
    """Check Tradeify microscalp rule: >50% trades >10s, >50% profit from >10s trades.

    Returns None if no trades with timing data.
    """
    db = db_path or _get_paper_trades_db()

    with duckdb.connect(str(db), read_only=True) as con:
        configure_connection(con)
        if not _has_paper_trades(con):
            return None

        where = "WHERE entry_time IS NOT NULL AND exit_time IS NOT NULL AND pnl_dollar IS NOT NULL"
        params = []
        if instrument:
            where += " AND instrument = ?"
            params.append(instrument)

        rows = con.execute(
            f"""
            SELECT pnl_dollar,
                   EPOCH(exit_time - entry_time) AS hold_seconds
            FROM {_PAPER_TRADES_TABLE}
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return None

        total = len(rows)
        over_10s = [(pnl, secs) for pnl, secs in rows if secs is not None and secs >= 10]
        profit_over_10s = sum(pnl for pnl, _ in over_10s if pnl > 0)
        total_profit = sum(pnl for pnl, _ in rows if pnl > 0)

        pct_trades = len(over_10s) / total if total > 0 else 0.0
        pct_profit = profit_over_10s / total_profit if total_profit > 0 else 1.0

        return MicroscalpResult(
            total_trades=total,
            trades_over_10s=len(over_10s),
            pct_trades_over_10s=round(pct_trades * 100, 1),
            profit_from_over_10s=round(profit_over_10s, 2),
            total_profit=round(total_profit, 2),
            pct_profit_from_over_10s=round(pct_profit * 100, 1),
            compliant=pct_trades >= 0.50 and pct_profit >= 0.50,
        )
