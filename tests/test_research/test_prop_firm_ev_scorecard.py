"""Stage 3 — Prop-Firm EV combiner / NO_NUMERIC_EV seam tests.

Covers the EV formula combiner, the fail-closed payout-cap resolver, the Frechet
worst-dependence tail bound, N_eff capacity, and the zero-mutation guard. The headline
assertion (test 1) is that EVERY firm currently returns NoNumericEv — the correct
honest output, not a defect.

Read-only: the survival adapter is invoked with ``write_state=False`` (asserted in
test 11); no DB write, no report write unless ``--write-report`` is passed.
"""

from __future__ import annotations

import math

import pytest

from research.prop_firm_ev_formula import (
    DEFAULT_REPORT_PATH,
    NoNumericEv,
    NoNumericEvReason,
    NumericEv,
    combine_ev,
    compute_n_eff,
    evaluate_all_firm_ev,
    frechet_tail_haircut,
    portfolio_tail_bound,
    resolve_fee_leg,
    resolve_payout_cap,
)
from research.prop_firm_ev_scorecard import SurvivalAdapterRecord
from research.prop_firm_state_machine import FirmStateMachine, load_state_machines
from trading_app.account_survival import SurvivalSummary

# NumericEv / NoNumericEv / NoNumericEvReason / compute_n_eff / combine_ev /
# DEFAULT_REPORT_PATH are exercised below; the imports above are the public surface.

# A small path count keeps the survival engine fast; the verdict (NoNumericEv) is
# independent of n_paths under current ground truth (no leg is path-count-sensitive).
_FAST_PATHS = 500


def _deployed_survival_record(profile_id: str) -> SurvivalAdapterRecord:
    fixtures = {
        "topstep_50k_mnq_auto": ("topstep", "trailing_intraday"),
        "tradeify_50k": ("tradeify", "trailing_eod"),
        "bulenox_50k": ("bulenox", "trailing_eod"),
    }
    firm, dd_type = fixtures[profile_id]
    return SurvivalAdapterRecord(
        profile_id=profile_id,
        firm=firm,
        account_size=50_000,
        dd_type=dd_type,
        survival_prob=0.80,
        dd_survival_prob=0.90,
        reach_payout_prob=None,
        reach_is_coarse=False,
        profit_target=None,
        target_reason="no_profit_target_encoded",
        target_source="hermetic-test-fixture",
        resolved_profile_id=profile_id,
        profile_id_aliased=False,
        sizing_parity_warning=False,
        notes=(),
    )


@pytest.fixture(autouse=True)
def hermetic_survival_adapter(monkeypatch):
    """Keep EV combiner tests independent of canonical gold.db availability."""
    import research.prop_firm_ev_formula as formula

    def fake_adapter(profile_id: str, **_kwargs) -> SurvivalAdapterRecord:
        return _deployed_survival_record(profile_id)

    monkeypatch.setattr(formula, "evaluate_survival_adapter", fake_adapter)


def _survival_summary(profile_id: str) -> SurvivalSummary:
    return SurvivalSummary(
        profile_id=profile_id,
        generated_at_utc="2026-06-11T00:00:00Z",
        as_of_date="2026-06-10",
        horizon_days=90,
        n_paths=_FAST_PATHS,
        seed=0,
        source_days=10,
        source_start="2026-05-01",
        source_end="2026-06-10",
        dd_survival_probability=0.90,
        operational_pass_probability=0.80,
        consistency_pass_probability=None,
        trailing_dd_breach_probability=0.10,
        daily_loss_breach_probability=0.0,
        scaling_breach_probability=0.0,
        consistency_breach_probability=0.0,
        scaling_feasible=True,
        intraday_approximated=False,
        path_model="hermetic-test-fixture",
        min_operational_pass_probability=0.70,
        gate_pass=True,
        strict_account_gate_pass=True,
        effective_dd_budget_dollars=1_800.0,
        historical_daily_loss_breach_days=[],
        historical_daily_loss_breach_count=0,
        historical_max_observed_90d_dd_dollars=1_000.0,
        p50_final_balance=500.0,
        p05_final_balance=-500.0,
        p95_final_balance=1_500.0,
        p50_total_pnl=500.0,
        p05_total_pnl=-500.0,
        p95_total_pnl=1_500.0,
        p50_max_dd=500.0,
        p95_max_dd=1_000.0,
        median_best_day=100.0,
    )


@pytest.fixture(scope="module")
def all_results() -> list:
    """Evaluate every firm once (module-scoped — the MC engine is the slow part)."""
    return evaluate_all_firm_ev(n_paths=_FAST_PATHS, seed=0)


def _by_firm(results: list, firm_substr: str):
    """The single result whose firm name contains ``firm_substr`` (case-insensitive)."""
    hits = [r for r in results if firm_substr.lower() in r.firm.lower()]
    assert len(hits) == 1, f"expected exactly one {firm_substr!r}, got {[r.firm for r in hits]}"
    return hits[0]


# ── 1. Headline: every firm is NoNumericEv under current data ────────────────


def test_all_firms_no_numeric_ev(all_results):
    assert len(all_results) == 4, f"expected 4 firms, got {[r.firm for r in all_results]}"
    for r in all_results:
        assert isinstance(r, NoNumericEv), f"{r.firm} unexpectedly NUMERIC"
        assert len(r.reasons) >= 1


# ── 2. MFFU ⇒ MISSING_SURVIVE_LEG (no AccountProfile) ────────────────────────


def test_mffu_missing_survive_leg(all_results):
    mffu = _by_firm(all_results, "MyFundedFutures")
    assert NoNumericEvReason.MISSING_SURVIVE_LEG in mffu.reasons
    assert mffu.survive_prob is None


# ── 3. Each DEPLOYED firm ⇒ MISSING_REACH_LEG (no profit_target encoded) ─────


@pytest.mark.parametrize("firm", ["Topstep", "Tradeify", "Bulenox"])
def test_deployed_missing_reach_leg(all_results, firm):
    r = _by_firm(all_results, firm)
    assert NoNumericEvReason.MISSING_REACH_LEG in r.reasons
    assert r.reach_prob is None


# ── 4. Tradeify ⇒ FEE_CONFLICT ($369 vs $251 NOT averaged) ──────────────────


def test_tradeify_fee_conflict(all_results):
    tradeify = _by_firm(all_results, "Tradeify")
    assert NoNumericEvReason.FEE_CONFLICT in tradeify.reasons
    # Sourced from fees.evidence_label, never an averaged number ($369 vs $251).
    fee = resolve_fee_leg("Tradeify")
    assert fee.conflict is True
    assert fee.fee is None  # conflicting prices are never coerced to a number.


# ── 5. Each firm ⇒ COMPLIANCE_BLOCKER (every gate emits >=1) ─────────────────


@pytest.mark.parametrize("firm", ["Topstep", "Tradeify", "MyFundedFutures", "Bulenox"])
def test_every_firm_has_compliance_blocker(all_results, firm):
    r = _by_firm(all_results, firm)
    assert NoNumericEvReason.COMPLIANCE_BLOCKER in r.reasons


# ── 6. Reasons collected, not short-circuited (Tradeify carries 3+ at once) ──


def test_reasons_not_short_circuited(all_results):
    tradeify = _by_firm(all_results, "Tradeify")
    expected = {
        NoNumericEvReason.MISSING_REACH_LEG,
        NoNumericEvReason.FEE_CONFLICT,
        NoNumericEvReason.COMPLIANCE_BLOCKER,
    }
    assert expected.issubset(set(tradeify.reasons)), tradeify.reasons
    # De-duped: no reason appears twice.
    assert len(tradeify.reasons) == len(set(tradeify.reasons))
    # One detail string per reason.
    assert len(tradeify.detail) == len(tradeify.reasons)


# ── 7. Synthetic all-legs-present fixture ⇒ NumericEv, arithmetic exact ──────


def _clean_state_machine() -> FirmStateMachine:
    """A synthetic firm with NO compliance blockers and all four legs resolvable.

    Borrows 'MyFundedFutures': the real resolvers return a clean payout cap ($2,000 at
    50K) AND a clean fee leg ($153 eval fee, no purchase-price-unsupported marker) — it
    is the only firm whose fee leg resolves to a dollar. We strip the manifest's
    compliance blockers on the synthetic sm to exercise the all-clear arithmetic
    branch (the live MFFU firm is NoNumericEv on MISSING_SURVIVE_LEG + compliance).
    """
    return FirmStateMachine(
        firm="MyFundedFutures",
        plan="synthetic-clean",
        verdict="WATCH",
        evidence_label="MEASURED",  # non-conflicting (state-machine label, unused by fee resolver).
        states=(),
        transition_notes=(),
        compliance_blockers=(),  # no blocker -> arithmetic path reachable.
        transition_haircut=1.0,
        correlated_tail_note="synthetic",
    )


def _survival_record(survive: float, reach: float) -> SurvivalAdapterRecord:
    return SurvivalAdapterRecord(
        profile_id="synthetic",
        firm="MyFundedFutures",
        account_size=50_000,
        dd_type="trailing_intraday",
        survival_prob=survive,
        dd_survival_prob=survive,
        reach_payout_prob=reach,
        reach_is_coarse=False,
        profit_target=2_000.0,
        target_reason=None,
        target_source="synthetic",
        resolved_profile_id="synthetic",
        profile_id_aliased=False,
        sizing_parity_warning=False,
        notes=(),
    )


def test_synthetic_all_legs_present_numeric_ev():
    sm = _clean_state_machine()
    survival = _survival_record(survive=0.80, reach=0.50)
    result = combine_ev(sm, survival, profile_id="synthetic", account_size=50_000)

    assert isinstance(result, NumericEv), getattr(result, "reasons", None)
    assert math.isfinite(result.ev)

    # ev == survive*reach*capped_capacity - fees - compliance_haircut - tail_haircut.
    expected = (
        result.survive_prob * result.reach_prob * result.capped_payout_capacity
        - result.fees
        - result.compliance_haircut
        - result.correlated_tail_haircut
    )
    assert result.ev == pytest.approx(expected)
    # The fee leg resolved to the MFFU evaluation fee ($153), not a fabricated 0.0.
    assert result.fees == pytest.approx(153.0)
    assert result.capped_payout_capacity == pytest.approx(2_000.0)  # cap $2,000 * n_eff 1.0.
    # Per-firm correlated-tail haircut is 0.0 BY CONTRACT: single-firm ruin
    # (1 - survive) is already priced into the survive multiplicand, so charging the
    # Frechet bound here would double-count. The cross-firm bound lives in
    # portfolio_tail_bound (k>=2). is_bound flags the dormant machinery is Frechet.
    assert result.correlated_tail_haircut == pytest.approx(0.0)
    assert result.tail_is_bound is True


# ── 8. N_eff: compliance-blocked ⇒ 0.0; clean ⇒ min(raw, 1) ─────────────────


def test_n_eff_compliance_blocked_is_zero():
    assert compute_n_eff(11.0, 1.0, compliance_blocked=True) == 0.0


def test_n_eff_clean_is_min_raw_one():
    assert compute_n_eff(11.0, 1.0, compliance_blocked=False) == 1.0
    assert compute_n_eff(3.0, 0.5, compliance_blocked=False) == 1.0  # min(1.5, 1).
    assert compute_n_eff(0.5, 1.0, compliance_blocked=False) == 0.5  # below 1 stays.


# ── 9. Frechet bound: worst-dependence endpoint + interior independence ──────


def test_frechet_worst_dependence_bound():
    bound = frechet_tail_haircut((0.85, 0.70))
    # Plan-pinned figures (verified by execution before locking the plan).
    assert bound.tail_least == pytest.approx(0.300)
    assert bound.tail_most == pytest.approx(0.450)
    assert bound.haircut == pytest.approx(0.450)  # haircut IS the most-tail endpoint.
    # Independence estimate sits strictly INSIDE the interval, not an endpoint.
    assert bound.tail_independence == pytest.approx(0.405)
    assert bound.tail_least < bound.tail_independence < bound.tail_most
    assert bound.is_bound is True


def test_frechet_haircut_is_fail_closed_not_optimistic():
    # The haircut must be the MOST-tail (worst-dependence) endpoint, never 1-min(s).
    bound = frechet_tail_haircut((0.85, 0.70))
    optimistic = 1.0 - min(0.85, 0.70)  # 0.300 — the WRONG (anti-conservative) value.
    assert bound.haircut != pytest.approx(optimistic)
    assert bound.haircut > optimistic


def test_frechet_rejects_out_of_range():
    with pytest.raises(ValueError):
        frechet_tail_haircut((1.2,))
    with pytest.raises(ValueError):
        frechet_tail_haircut(())


def test_portfolio_tail_bound_dormant_under_current_data(all_results):
    # Every firm is NoNumericEv -> 0 numeric-eligible -> bound is None (dormant).
    assert portfolio_tail_bound(all_results) is None


def test_portfolio_tail_bound_activates_at_two_numeric_firms():
    # Two synthetic NumericEv firms with survive [0.85, 0.70] -> the cross-firm bound.
    def _numeric(firm, survive):
        return NumericEv(
            profile_id=firm,
            firm=firm,
            ev=1.0,
            survive_prob=survive,
            reach_prob=0.5,
            reach_is_coarse=False,
            capped_payout_capacity=2_000.0,
            n_eff=1.0,
            fees=0.0,
            transition_haircut=1.0,
            compliance_haircut=0.0,
            correlated_tail_haircut=0.0,
            tail_is_bound=True,
            notes=(),
        )

    bound = portfolio_tail_bound([_numeric("A", 0.85), _numeric("B", 0.70)])
    assert bound is not None
    assert bound.haircut == pytest.approx(0.450)  # worst-dependence joint ruin.
    assert bound.tail_least < bound.tail_independence < bound.tail_most


# ── 9a. resolve_payout_cap: size-correct, malformed-raises, absent-None ──────


def test_payout_cap_topstep_size_correct():
    # 50K must resolve to $2,000, NOT 150K's $5,000 (size-nested prose).
    res = resolve_payout_cap("Topstep", 50_000)
    assert res.cap == 2_000.0
    res150 = resolve_payout_cap("Topstep", 150_000)
    assert res150.cap == 5_000.0


def test_payout_cap_flat_firms():
    assert resolve_payout_cap("MyFundedFutures", 50_000).cap == 2_000.0
    assert resolve_payout_cap("Bulenox", 50_000).cap == 1_500.0


def test_payout_cap_tradeify_size_mismatch_is_absent():
    # Tradeify's cap prose is for 150K; a 50K profile must NOT inherit it.
    res = resolve_payout_cap("Tradeify", 50_000)
    assert res.cap is None
    assert "150K" in (res.reason or "")


def test_payout_cap_malformed_raises(monkeypatch):
    # A size-nested prose that names the size but has no dollar after it is malformed-
    # but-present -> must RAISE, never return a fabricated/None cap silently.
    import research.prop_firm_ev_formula as mod

    def fake_field(firm, field):
        # 3 sizes + 2 dollars => classified size-nested; but the target 50K token is
        # LAST with no dollar following it (the malformed-but-present case).
        return "100K $3,000; 150K $5,000; 50K."

    monkeypatch.setattr(mod, "_firm_payout_field", fake_field)
    with pytest.raises(ValueError, match="no dollar amount follows"):
        resolve_payout_cap("Topstep", 50_000)


def test_payout_cap_genuinely_absent_is_none(monkeypatch):
    import research.prop_firm_ev_formula as mod

    monkeypatch.setattr(mod, "_firm_payout_field", lambda firm, field: None)
    res = resolve_payout_cap("Topstep", 50_000)
    assert res.cap is None
    assert res.reason is not None


def test_missing_payout_cap_reason_surfaces():
    # Tradeify (cap absent for 50K) must carry MISSING_PAYOUT_CAP.
    sm = load_state_machines()["tradeify"]
    survival = _survival_record(survive=0.8, reach=0.5)
    # Force the survive/reach legs present so MISSING_PAYOUT_CAP is the salient one.
    result = combine_ev(sm, survival, profile_id="t", account_size=50_000)
    assert isinstance(result, NoNumericEv)
    assert NoNumericEvReason.MISSING_PAYOUT_CAP in result.reasons


# ── 9b. resolve_fee_leg: all four fee outcomes reachable (no dead enum) ──────


def test_fee_leg_clean_dollar():
    # MFFU is the only firm with a clean fee leg ($153 eval fee, price not unsupported).
    fee = resolve_fee_leg("MyFundedFutures")
    assert fee.fee == pytest.approx(153.0)
    assert not fee.conflict and not fee.unsupported


def test_fee_leg_conflict():
    fee = resolve_fee_leg("Tradeify")
    assert fee.conflict is True and fee.fee is None


@pytest.mark.parametrize("firm", ["Topstep", "Bulenox"])
def test_fee_leg_unsupported(firm):
    # Both flag the official purchase price as an unsupported_field -> leg is gated.
    fee = resolve_fee_leg(firm)
    assert fee.unsupported is True and fee.fee is None


def test_fee_leg_missing_when_no_price_field(monkeypatch):
    # A firm with no conflict, no unsupported price marker, and no fee-price field
    # encoded -> MISSING_FEE (the genuinely-absent case).
    import research.prop_firm_ev_formula as mod

    monkeypatch.setattr(mod, "_firm_card", lambda firm: {"fees": {"fields": {}}})
    fee = resolve_fee_leg("SomeFirm")
    assert fee.fee is None
    assert not fee.conflict and not fee.unsupported


def test_unsupported_field_blocks_reason_reachable(all_results):
    # Topstep/Bulenox must carry UNSUPPORTED_FIELD_BLOCKS (purchase price unsupported).
    for firm in ("Topstep", "Bulenox"):
        r = _by_firm(all_results, firm)
        assert NoNumericEvReason.UNSUPPORTED_FIELD_BLOCKS in r.reasons


def test_missing_fee_reason_reachable(monkeypatch):
    # Force a clean-price-field-absent firm through combine_ev -> MISSING_FEE surfaces.
    import research.prop_firm_ev_formula as mod

    monkeypatch.setattr(mod, "_firm_card", lambda firm: {"fees": {"fields": {}}})
    sm = _clean_state_machine()
    survival = _survival_record(survive=0.8, reach=0.5)
    result = combine_ev(sm, survival, profile_id="x", account_size=50_000)
    assert isinstance(result, NoNumericEv)
    assert NoNumericEvReason.MISSING_FEE in result.reasons


# ── 10. No fabricated number on any missing leg (never 0.0) ──────────────────


def test_no_fabricated_number_on_missing_leg(all_results):
    for r in all_results:
        assert isinstance(r, NoNumericEv)
        # NoNumericEv has NO ev attribute at all — a missing leg can never read as EV.
        assert not hasattr(r, "ev")


# ── 11. Zero-mutation guard ──────────────────────────────────────────────────


def test_survival_invoked_write_state_false(monkeypatch):
    """combine path must invoke the survival engine with write_state=False."""
    import research.prop_firm_ev_scorecard as scorecard

    captured = {}

    def spy(profile_id, **kwargs):
        captured["write_state"] = kwargs.get("write_state")
        return _survival_summary(profile_id)

    monkeypatch.setattr(scorecard, "evaluate_profile_survival", spy)

    scorecard.evaluate_survival_adapter("topstep_50k_mnq_auto", n_paths=_FAST_PATHS, seed=0)
    assert captured["write_state"] is False


def test_report_not_written_on_dry_run():
    """The dry-run default writes nothing; main() only writes on --write-report."""
    from research.prop_firm_ev_formula import main

    existed_before = DEFAULT_REPORT_PATH.exists()
    mtime_before = DEFAULT_REPORT_PATH.stat().st_mtime if existed_before else None

    rc = main(["--n-paths", str(_FAST_PATHS), "--no-write-report"])
    assert rc == 0

    if existed_before:
        assert DEFAULT_REPORT_PATH.stat().st_mtime == mtime_before
    else:
        assert not DEFAULT_REPORT_PATH.exists()
