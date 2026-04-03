"""Focused tests for dashboard metadata and legacy state compatibility."""

from datetime import date
from types import SimpleNamespace

from trading_app.live.bot_dashboard import _legacy_lanes_to_lane_cards, _strategy_meta
from trading_app.live.bot_state import build_state_snapshot
from trading_app.prop_profiles import ACCOUNT_PROFILES


def test_build_state_snapshot_uses_explicit_trading_day_for_session_times():
    strategy = SimpleNamespace(
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR70_VOL",
        instrument="MNQ",
        orb_label="NYSE_OPEN",
        filter_type="ATR70_VOL",
        rr_target=1.0,
        orb_minutes=5,
        entry_model="E2",
        confirm_bars=1,
    )

    snapshot = build_state_snapshot(
        mode="SIGNAL",
        instrument="MNQ",
        contract="MNQM6",
        trading_day=date(2026, 4, 3),
        account_id=0,
        account_name="profile_topstep_50k_mnq_auto",
        daily_pnl_r=0.0,
        daily_loss_limit_r=-5.0,
        max_equity_dd_r=None,
        bars_received=12,
        strategies=[strategy],
        active_trades=[],
        completed_trades=[],
    )

    assert snapshot["trading_day"] == "2026-04-03"
    assert snapshot["lane_cards"][0]["session_time_brisbane"] == "23:30"


def test_strategy_meta_extracts_human_readable_lane_fields():
    meta = _strategy_meta("MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6", date(2026, 4, 3))

    assert meta["instrument_label"] == "MGC"
    assert meta["session_name"] == "CME_REOPEN"
    assert meta["session_time_brisbane"] == "08:00"
    assert meta["entry_model"] == "E2"
    assert meta["rr_target"] == 2.5
    assert meta["confirm_bars"] == 1
    assert meta["filter_type"] == "ORB_G6"
    assert meta["lane_label"] == "MGC CME_REOPEN"


def test_legacy_lanes_reconstruct_all_profile_lanes_and_mark_ambiguous_shared_sessions():
    profile = ACCOUNT_PROFILES["topstep_50k_type_a"]
    mnq_comex = next(l for l in profile.daily_lanes if l.strategy_id == "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K")
    mes_comex = next(l for l in profile.daily_lanes if l.strategy_id == "MES_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_S075")

    cards = _legacy_lanes_to_lane_cards(
        lanes={
            "COMEX_SETTLE": {
                "strategy_id": mnq_comex.strategy_id,
                "status": "IN_TRADE",
                "direction": "long",
                "entry_price": 21543.25,
                "current_pnl_r": 0.8,
                "rr_target": 1.5,
                "orb_minutes": 5,
                "filter_type": "ORB_VOL_8K",
            }
        },
        trading_day=date(2026, 4, 3),
        account_name="profile_topstep_50k_type_a",
    )

    assert len(cards) == len(profile.daily_lanes)

    mnq_card = next(card for card in cards if card["strategy_id"] == mnq_comex.strategy_id)
    mes_card = next(card for card in cards if card["strategy_id"] == mes_comex.strategy_id)

    assert mnq_card["status"] == "IN_TRADE"
    assert mnq_card["direction"] == "long"
    assert mnq_card["session_time_brisbane"] == "03:30"

    assert mes_card["status"] == "UNKNOWN"
    assert mes_card["status_detail"] is not None
    assert "cannot disambiguate" in mes_card["status_detail"]
    assert mes_card["session_time_brisbane"] == "03:30"
