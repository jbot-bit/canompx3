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
    # ─── topstep_express_standard ─────────────────────────────────────
    # @canonical-source docs/research-input/topstep/topstep_payout_policy.txt  (article 8284233, scraped 2026-04-08)
    # @verbatim-section "Express Funded Account Consistency vs Express Funded Account Standard"
    # @verbatim "5 winning days of $150+ / Request 50% of account balance up to $5000 / 90/10 split"
    # @canonical-source docs/research-input/topstep/topstep_payout_policy.txt
    # @verbatim "The Minimum Payout Request is $125.00."
    # @canonical-source docs/research-input/topstep/topstep_payout_policy.txt
    # @verbatim "For new Traders who join on or after January 12, 2026: All payouts will
    #            be subject to a 90/10 profit split: Traders receive 90% of approved
    #            payouts. Topstep retains 10% of the requested payout."
    # @user-status User joined post-2026-01-12 (all new accounts) → flat 90/10 applies. Verified 2026-04-08.
    # @audit-finding F-10 (CONFIRMED — all fields match canonical)
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
        # @inferred The post-payout cycle reset for Standard is NOT explicitly stated
        # in topstep_payout_policy.txt — the canonical doc only describes after-payout
        # mechanics for the Consistency path (lines 178-211). Setting to 5 (same as the
        # winning-days requirement) is a reasonable interpretation but not canon.
        # @audit-finding F-11 (INFERRED)
        additional_days_after_payout=5,
        notes=(
            "Current standard path from Topstep help center. "
            "@canonical-source docs/research-input/topstep/topstep_payout_policy.txt (2026-04-08). "
            "Requires profit above zero since last payout, which is not inferable from "
            "paper_trades alone without payout ledger state."
        ),
    ),
    # ─── topstep_express_consistency ──────────────────────────────────
    # @canonical-source docs/research-input/topstep/topstep_xfa_parameters.txt  (article 8284215, scraped 2026-04-08)
    # @verbatim "Payout Eligibility: 3 days with 40% consistency target / A minimum of
    #            3 trading days / At least one trade per day"
    # @canonical-source docs/research-input/topstep/topstep_xfa_parameters.txt
    # @verbatim "Consistency = Largest Winning Day ÷ Current Total Net Profit"
    # @canonical-source docs/research-input/topstep/topstep_xfa_parameters.txt
    # @verbatim "What is the Payout Cap in the Express Funded Account Consistency? The
    #            payout cap for XFA Consistency is $6,000."
    # @audit-finding F-10 (CONFIRMED)
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
            "Largest winning day must be <= 40% of total net profit during the payout window. "
            "@canonical-source docs/research-input/topstep/topstep_xfa_parameters.txt (2026-04-08). "
            "This path is a payout-smoothing program, not a generic Topstep rule."
        ),
    ),
    # ─── topstep_live_funded ──────────────────────────────────────────
    # @canonical-source docs/research-input/topstep/topstep_payout_policy.txt  (article 8284233, scraped 2026-04-08)
    # @verbatim "You can take payouts daily and access up to 100% of your profits after
    #            accumulating 30 winning trading days in a Live Funded Account."
    # @canonical-source docs/research-input/topstep/topstep_live_funded_parameters.md  (article 10657969, scraped 2026-04-08)
    # @verbatim "Live Funded Accounts begin with a Daily Loss Limit based on account
    #            size: $2,000 for $50K accounts, $3,000 for $100K accounts, $4,500 for
    #            $150K accounts."
    # @audit-finding F-3 (DEFERRED — LFA DLL not yet wired into AccountHWMTracker)
    # @audit-finding F-10 (CONFIRMED for the fields modeled here)
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
        # @inferred Same caveat as topstep_express_standard. F-11.
        additional_days_after_payout=5,
        daily_payouts_unlock_winning_days=30,
        daily_payout_cap_balance_pct=1.00,
        notes=(
            "After 30 non-consecutive live winning days, daily payouts unlock and up to "
            "100% of balance becomes requestable. A full 100% payout closes the account. "
            "@canonical-source docs/research-input/topstep/topstep_payout_policy.txt (2026-04-08)."
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
