"""Canonical payout-policy layer for prop firms.

Execution constraints live in trading_app.prop_profiles.
Payout mechanics live here so firm programs like Topstep Express Standard,
Topstep Express Consistency, and Topstep Live Funded can be compared cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PayoutPolicy:
    """Structured payout policy for one firm program/path."""

    policy_id: str
    firm: str
    display_name: str
    stage: str
    profit_split_pct: float
    model_status: str = "complete"
    winning_days_required: int | None = None
    winning_day_profit_threshold: float | None = None
    min_trading_days: int | None = None
    consistency_rule: float | None = None
    payout_cap_balance_pct: float | None = None
    payout_cap_dollars: float | None = None
    min_payout_amount: float | None = None
    additional_days_after_payout: int | None = None
    daily_payouts_unlock_winning_days: int | None = None
    daily_payout_cap_balance_pct: float | None = None
    notes: str = ""


PAYOUT_POLICIES: dict[str, PayoutPolicy] = {
    "topstep_express_standard": PayoutPolicy(
        policy_id="topstep_express_standard",
        firm="topstep",
        display_name="Topstep Express Standard",
        stage="express_funded",
        profit_split_pct=0.90,
        winning_days_required=5,
        winning_day_profit_threshold=150.0,
        min_trading_days=5,
        payout_cap_balance_pct=0.50,
        payout_cap_dollars=5_000.0,
        min_payout_amount=125.0,
        additional_days_after_payout=5,
        notes=(
            "Current standard path from Topstep help center (updated Apr 2026 in-repo). "
            "Requires profit above zero since last payout, which is not inferable from "
            "paper_trades alone without payout ledger state."
        ),
    ),
    "topstep_express_consistency": PayoutPolicy(
        policy_id="topstep_express_consistency",
        firm="topstep",
        display_name="Topstep Express Consistency",
        stage="express_funded",
        profit_split_pct=0.90,
        min_trading_days=3,
        consistency_rule=0.40,
        payout_cap_balance_pct=0.50,
        payout_cap_dollars=6_000.0,
        min_payout_amount=125.0,
        additional_days_after_payout=3,
        notes=(
            "Largest winning day must be <= 40% of total net profit during the payout "
            "window. This path is a payout-smoothing program, not a generic Topstep rule."
        ),
    ),
    "topstep_live_funded": PayoutPolicy(
        policy_id="topstep_live_funded",
        firm="topstep",
        display_name="Topstep Live Funded",
        stage="live_funded",
        profit_split_pct=0.90,
        winning_days_required=5,
        winning_day_profit_threshold=150.0,
        min_trading_days=5,
        payout_cap_balance_pct=0.50,
        min_payout_amount=125.0,
        additional_days_after_payout=5,
        daily_payouts_unlock_winning_days=30,
        daily_payout_cap_balance_pct=1.00,
        notes=(
            "After 30 non-consecutive live winning days, daily payouts unlock and up to "
            "100% of balance becomes requestable. A full 100% payout closes the account."
        ),
    ),
    "tradeify_select_funded": PayoutPolicy(
        policy_id="tradeify_select_funded",
        firm="tradeify",
        display_name="Tradeify Select Funded",
        stage="sim_funded",
        profit_split_pct=0.90,
        model_status="partial",
        notes=(
            "Execution/compliance modeled in prop_profiles. Payout gating not re-sourced "
            "in this turn, so only the split is canonical here."
        ),
    ),
    "self_funded": PayoutPolicy(
        policy_id="self_funded",
        firm="self_funded",
        display_name="Self-Funded",
        stage="brokerage",
        profit_split_pct=1.00,
        notes="No payout gate. Account equity is fully owned capital.",
    ),
}


DEFAULT_PAYOUT_POLICY_BY_FIRM: dict[str, str] = {
    "topstep": "topstep_express_standard",
    "tradeify": "tradeify_select_funded",
    "self_funded": "self_funded",
}


def get_payout_policy(policy_id: str) -> PayoutPolicy:
    """Return one payout policy by id."""
    return PAYOUT_POLICIES[policy_id]


def get_default_payout_policy_for_firm(firm: str) -> PayoutPolicy | None:
    """Return the default payout policy for a firm, if modeled."""
    policy_id = DEFAULT_PAYOUT_POLICY_BY_FIRM.get(firm)
    if policy_id is None:
        return None
    return PAYOUT_POLICIES[policy_id]


def list_payout_policies_for_firm(firm: str) -> list[PayoutPolicy]:
    """Return all modeled payout paths for a firm."""
    return [policy for policy in PAYOUT_POLICIES.values() if policy.firm == firm]
