# tests/test_trading_app/test_regime_prop_test_profile.py
"""Tests for the topstep_50k_regime_test profile — the zero-evidence REGIME
forward live-test wiring (stage: regime-prop-test-wiring).

Operator-approved, express-funded only, ships default-OFF (profile active=False
is the binding code-level arm gate). 6 lanes (3 MNQ COMEX_SETTLE + 3 MGC
LONDON_METALS); the thin MES N=16 lane was deliberately dropped.

Caps (max_orb_size_pts) are the canonical P90 from compute_orb_size_stats:
  (MNQ, COMEX_SETTLE, 5)   -> P90 = 50.1
  (MGC, LONDON_METALS, 30) -> P90 = 25.8
Both within [avg, p90] (cap == p90, the band ceiling).
"""

from trading_app.prop_profiles import (
    ACCOUNT_PROFILES,
    effective_daily_lanes,
    get_profile,
)

PROFILE_ID = "topstep_50k_regime_test"

# Canonical per-(instrument, session, aperture) caps, measured live from
# compute_orb_size_stats against gold.db (cap = P90). See module docstring.
_EXPECTED_CAPS = {
    "MNQ": 50.1,  # (MNQ, COMEX_SETTLE, 5)   avg 29.0 / p90 50.1
    "MGC": 25.8,  # (MGC, LONDON_METALS, 30) avg 13.8 / p90 25.8
}

_EXPECTED_LANES = {
    # strategy_id: (instrument, orb_label)
    "MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K": ("MNQ", "COMEX_SETTLE"),
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K": ("MNQ", "COMEX_SETTLE"),
    "MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_VOL_16K": ("MNQ", "COMEX_SETTLE"),
    "MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30": ("MGC", "LONDON_METALS"),
    "MGC_LONDON_METALS_E2_RR1.5_CB1_ORB_VOL_8K_O30": ("MGC", "LONDON_METALS"),
    "MGC_LONDON_METALS_E1_RR1.5_CB1_ORB_VOL_8K_O30": ("MGC", "LONDON_METALS"),
}


class TestRegimePropTestProfile:
    def test_profile_exists_and_express_funded(self):
        assert PROFILE_ID in ACCOUNT_PROFILES
        p = get_profile(PROFILE_ID)
        assert p.is_express_funded is True
        assert p.firm == "topstep"
        assert p.account_size == 50_000

    def test_default_off_active_false(self):
        # Binding code-level arm gate: resolve_profile_id(active_only=True)
        # raises while active=False, so the operator cannot arm it until they
        # deliberately flip it in code. This is the committable default-OFF.
        assert get_profile(PROFILE_ID).active is False

    def test_holds_exactly_the_six_lanes(self):
        lanes = effective_daily_lanes(get_profile(PROFILE_ID))
        got = {lane.strategy_id for lane in lanes}
        assert got == set(_EXPECTED_LANES), (
            f"lane set mismatch: missing={set(_EXPECTED_LANES) - got}, extra={got - set(_EXPECTED_LANES)}"
        )
        assert len(lanes) == 6  # no MES lane

    def test_each_lane_instrument_and_session_match(self):
        lanes = {lane.strategy_id: lane for lane in effective_daily_lanes(get_profile(PROFILE_ID))}
        for sid, (inst, sess) in _EXPECTED_LANES.items():
            assert lanes[sid].instrument == inst, f"{sid} instrument"
            assert lanes[sid].orb_label == sess, f"{sid} session"

    def test_all_caps_non_none_and_match_canonical_p90(self):
        # max_orb_size_pts is read by account_survival.py:439 +
        # session_orchestrator.py:395 as the per-lane size guard. A None cap =
        # NO size guard on real capital. Must be non-None for all 6.
        for lane in effective_daily_lanes(get_profile(PROFILE_ID)):
            assert lane.max_orb_size_pts is not None, f"{lane.strategy_id} has None cap"
            assert lane.max_orb_size_pts == _EXPECTED_CAPS[lane.instrument], (
                f"{lane.strategy_id} cap {lane.max_orb_size_pts} != canonical P90 {_EXPECTED_CAPS[lane.instrument]}"
            )

    def test_required_fitness_is_fit(self):
        # required_fitness=('FIT',) lets the allocator auto-pause a decaying
        # lane with no manual gate.
        for lane in effective_daily_lanes(get_profile(PROFILE_ID)):
            assert lane.required_fitness == ("FIT",), lane.strategy_id

    def test_multi_instrument_resolves(self):
        insts = {lane.instrument for lane in effective_daily_lanes(get_profile(PROFILE_ID))}
        assert insts == {"MNQ", "MGC"}

    def test_self_funded_profile_untouched(self):
        # Hard guard: the only real-capital profile must stay is_express_funded=False.
        sf = get_profile("self_funded_tradovate")
        assert sf.is_express_funded is False
