"""MFFU plan×size encoding tests — assert prop_profiles + prop_firm_policies
match the verbatim help-center snapshots in docs/research-input/mffu/.

These are CONFORMANCE tests: every asserted number is quoted (with the source
article) in the spec/tier comments. If a future re-scrape changes a firm rule,
update the snapshot + spec + this test together (the Stage-3 freshness check
will flag staleness).
"""

from __future__ import annotations

from trading_app.prop_firm_policies import (
    get_default_payout_policy_for_firm,
    get_payout_policy,
)
from trading_app.prop_profiles import ACCOUNT_TIERS, get_firm_spec


def test_existing_specs_unchanged_firm_specific_rules_defaults_none():
    """The new optional field must not perturb any pre-existing spec."""
    for firm in ("topstep", "mffu", "tradeify", "self_funded", "bulenox"):
        assert get_firm_spec(firm).firm_specific_rules is None, firm


def test_rapid_sim_funded_caps_match_verbatim():
    """Rapid caps are the SIM-FUNDED ladder, not the live-reduced ladder.

    Verbatim (per-size Rapid Sim Funded articles, scraped 2026-05-31):
    25k=3/30, 50k=5/50, 100k=10/100, 150k=15/150.
    Regression guard for the 2026-05-31 fix (100k was 6/60, 150k was 8/80 —
    the live-funded reduced ladder, wrong for the sim account).
    """
    expected = {25_000: (3, 30), 50_000: (5, 50), 100_000: (10, 100), 150_000: (15, 150)}
    for size, (mini, micro) in expected.items():
        tier = ACCOUNT_TIERS[("mffu", size)]
        assert (tier.max_contracts_mini, tier.max_contracts_micro) == (mini, micro), size


def test_builder_spec_matches_verbatim():
    spec = get_firm_spec("mffu_builder")
    assert spec.dd_type == "eod_trailing"
    assert spec.profit_split_tiers[-1][1] == 0.80  # 80/20
    assert spec.consistency_rule is None  # eval/sim none; 50% is payout-stage
    assert spec.news_restriction is False  # unrestricted
    fsr = spec.firm_specific_rules
    assert fsr is not None
    assert fsr["account_size_only"] == 50_000
    assert fsr["profit_target"] == 3_000.0
    assert fsr["mll_options"] == {"default": 2_000.0, "add_on": 1_500.0}
    assert fsr["max_contracts_mini"] == 4 and fsr["max_contracts_micro"] == 40
    assert fsr["payout_cap_per_cycle"] == 2_000.0
    assert fsr["max_sim_payouts"] == 5
    assert fsr["payout_consistency_rule"] == 0.50
    assert fsr["forced_live_after_sim_payouts"] == 5
    assert fsr["max_live_accounts"] == 1
    assert fsr["post_breach_cooldown_days"] == 21


def test_builder_tier_matches_verbatim():
    tier = ACCOUNT_TIERS[("mffu_builder", 50_000)]
    assert tier.max_dd == 2_000  # Default MLL
    assert (tier.max_contracts_mini, tier.max_contracts_micro) == (4, 40)


def test_flex_spec_and_tiers_match_verbatim():
    spec = get_firm_spec("mffu_flex")
    assert spec.dd_type == "eod_trailing"
    assert spec.profit_split_tiers[-1][1] == 0.80
    fsr = spec.firm_specific_rules
    assert fsr is not None
    assert fsr["eval_consistency_rule"] == 0.50
    assert fsr["payout_consistency_rule"] is None  # payout stage none
    assert fsr["total_sim_payout_cap"] == 100_000.0
    by_size = fsr["by_size"]
    assert by_size[25_000]["mll"] == 1_000.0 and by_size[25_000]["max_contracts_micro"] == 20
    assert by_size[50_000]["mll"] == 2_000.0 and by_size[50_000]["max_contracts_micro"] == 30

    assert ACCOUNT_TIERS[("mffu_flex", 25_000)].max_dd == 1_000
    assert ACCOUNT_TIERS[("mffu_flex", 50_000)].max_dd == 2_000


def test_payout_policies_match_verbatim():
    builder = get_payout_policy("mffu_builder_sim")
    assert builder.profit_split_pct == 0.80
    assert builder.payout_cap_dollars == 2_000.0
    assert builder.consistency_rule == 0.50  # payout stage
    assert builder.min_trading_days == 2
    assert builder.min_payout_amount == 500.0

    flex = get_payout_policy("mffu_flex_sim")
    assert flex.profit_split_pct == 0.80
    assert flex.payout_cap_balance_pct == 0.50  # 50% of profits
    assert flex.payout_cap_dollars == 2_000.0
    assert flex.consistency_rule is None  # payout stage none
    assert flex.winning_days_required == 5


def test_firm_default_payout_policies_wired():
    assert get_default_payout_policy_for_firm("mffu_builder").policy_id == "mffu_builder_sim"
    assert get_default_payout_policy_for_firm("mffu_flex").policy_id == "mffu_flex_sim"
