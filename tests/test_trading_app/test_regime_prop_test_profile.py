# tests/test_trading_app/test_regime_prop_test_profile.py
"""Tests for the topstep_50k_regime_test profile — the zero-evidence REGIME
forward live-test wiring (stage: regime-prop-test-wiring).

Operator-approved, express-funded only, ships default-OFF (profile active=False
is the binding code-level arm gate). 3 lanes — one best lane per INDEPENDENT
session (correlation-pruned): MNQ COMEX_SETTLE E2 RR2.0, MNQ CME_PRECLOSE
X_MES_ATR70 E2 RR1.0 O15, MGC LONDON_METALS E2 RR2.0. DROPPED: MES N=16
(small-sample), MNQ COMEX RR1.5/RR2.5 (rho~0.83 w/ RR2.0), MGC E2RR1.5/E1RR1.5
(rho~0.90 w/ E2RR2.0) — all same-session redundancy, ~1 bet not N.

Caps (max_orb_size_pts) are canonical P90 from compute_orb_size_stats, keyed
per-(instrument, session, APERTURE):
  (MNQ, COMEX_SETTLE, 5)   -> P90 = 50.1
  (MNQ, CME_PRECLOSE, 15)  -> P90 = 99.2  (O15 = wider ORB than O5)
  (MGC, LONDON_METALS, 30) -> P90 = 25.8
"""

from trading_app.prop_profiles import (
    ACCOUNT_PROFILES,
    effective_daily_lanes,
    get_profile,
)

PROFILE_ID = "topstep_50k_regime_test"

# Canonical per-(instrument, session, APERTURE) caps, measured live from
# compute_orb_size_stats against gold.db (cap = P90). Keyed by strategy_id
# because MNQ has TWO caps (O5 COMEX 50.1 vs O15 CME_PRECLOSE 99.2 — the
# per-aperture method, not per-instrument). See module docstring.
_EXPECTED_LANES = {
    # strategy_id: (instrument, orb_label, expected_cap_p90)
    # ONE best lane per INDEPENDENT session. Same-session RR/entry-model variants
    # were rho~0.83 (MNQ COMEX) / rho~0.90 (MGC LONDON) = ~1 bet each, dropped.
    "MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K": ("MNQ", "COMEX_SETTLE", 50.1),
    "MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15": ("MNQ", "CME_PRECLOSE", 99.2),
    "MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30": ("MGC", "LONDON_METALS", 25.8),
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

    def test_holds_exactly_the_expected_lanes(self):
        lanes = effective_daily_lanes(get_profile(PROFILE_ID))
        got = {lane.strategy_id for lane in lanes}
        assert got == set(_EXPECTED_LANES), (
            f"lane set mismatch: missing={set(_EXPECTED_LANES) - got}, extra={got - set(_EXPECTED_LANES)}"
        )
        assert len(lanes) == 3  # 3 independent sessions; no same-session redundant variants

    def test_each_lane_instrument_and_session_match(self):
        lanes = {lane.strategy_id: lane for lane in effective_daily_lanes(get_profile(PROFILE_ID))}
        for sid, (inst, sess, _cap) in _EXPECTED_LANES.items():
            assert lanes[sid].instrument == inst, f"{sid} instrument"
            assert lanes[sid].orb_label == sess, f"{sid} session"

    def test_all_caps_non_none_and_match_canonical_p90(self):
        # max_orb_size_pts is read by account_survival.py:439 +
        # session_orchestrator.py:395 as the per-lane size guard. A None cap =
        # NO size guard on real capital. Caps are per-APERTURE (MNQ O5=50.1 vs
        # O15=99.2), so the expected value is keyed by strategy_id, not instrument.
        lanes = {lane.strategy_id: lane for lane in effective_daily_lanes(get_profile(PROFILE_ID))}
        for sid, (_inst, _sess, cap) in _EXPECTED_LANES.items():
            assert lanes[sid].max_orb_size_pts is not None, f"{sid} has None cap"
            assert lanes[sid].max_orb_size_pts == cap, f"{sid} cap {lanes[sid].max_orb_size_pts} != canonical P90 {cap}"

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
