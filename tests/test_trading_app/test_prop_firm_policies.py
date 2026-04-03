"""Tests for canonical payout-policy modeling."""

from trading_app.prop_firm_policies import (
    DEFAULT_PAYOUT_POLICY_BY_FIRM,
    get_default_payout_policy_for_firm,
    get_payout_policy,
    list_payout_policies_for_firm,
)


class TestTopstepPolicies:
    def test_default_topstep_policy_is_express_standard(self):
        assert DEFAULT_PAYOUT_POLICY_BY_FIRM["topstep"] == "topstep_express_standard"
        policy = get_default_payout_policy_for_firm("topstep")
        assert policy is not None
        assert policy.policy_id == "topstep_express_standard"

    def test_topstep_express_standard_fields(self):
        policy = get_payout_policy("topstep_express_standard")
        assert policy.profit_split_pct == 0.90
        assert policy.winning_days_required == 5
        assert policy.winning_day_profit_threshold == 150.0
        assert policy.payout_cap_balance_pct == 0.50
        assert policy.payout_cap_dollars == 5_000.0

    def test_topstep_consistency_fields(self):
        policy = get_payout_policy("topstep_express_consistency")
        assert policy.consistency_rule == 0.40
        assert policy.min_trading_days == 3
        assert policy.payout_cap_dollars == 6_000.0

    def test_topstep_live_fields(self):
        policy = get_payout_policy("topstep_live_funded")
        assert policy.winning_days_required == 5
        assert policy.winning_day_profit_threshold == 150.0
        assert policy.daily_payouts_unlock_winning_days == 30
        assert policy.daily_payout_cap_balance_pct == 1.00

    def test_topstep_lists_all_paths(self):
        policies = list_payout_policies_for_firm("topstep")
        ids = {p.policy_id for p in policies}
        assert ids == {
            "topstep_express_standard",
            "topstep_express_consistency",
            "topstep_live_funded",
        }
