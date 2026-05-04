import duckdb

from trading_app import sprt_monitor


def test_build_lanes_carries_canonical_strategy_params(monkeypatch):
    monkeypatch.setattr(
        sprt_monitor,
        "_load_reference_stats",
        lambda: {"MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8": {"mu0": 0.5, "sigma": 1.2}},
    )
    monkeypatch.setattr(
        sprt_monitor,
        "resolve_profile_id",
        lambda profile_id=None, active_only=True: "topstep_50k_mnq_auto",
        raising=False,
    )

    def _fake_lanes(_profile_id):
        return [
            {
                "strategy_id": "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8",
                "instrument": "MNQ",
                "orb_label": "NYSE_CLOSE",
                "orb_minutes": 5,
                "entry_model": "E2",
                "rr_target": 1.0,
                "confirm_bars": 1,
                "filter_type": "ORB_G8",
                "shadow_only": False,
            }
        ]

    monkeypatch.setattr(
        "trading_app.prop_profiles.get_profile_lane_definitions",
        _fake_lanes,
    )

    lanes = sprt_monitor._build_lanes()
    lane = lanes["MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8"]

    assert lane["instrument"] == "MNQ"
    assert lane["orb_minutes"] == 5
    assert lane["entry_model"] == "E2"
    assert lane["rr_target"] == 1.0
    assert lane["confirm_bars"] == 1
    assert lane["filter_type"] == "ORB_G8"


def test_load_trade_stream_prefers_paper_trades(monkeypatch):
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE paper_trades (
            strategy_id VARCHAR,
            trading_day DATE,
            pnl_r DOUBLE
        )
        """
    )
    con.execute(
        "INSERT INTO paper_trades VALUES (?, DATE '2026-01-02', ?), (?, DATE '2026-01-03', ?)",
        ["SID1", 1.25, "SID1", -1.0],
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("canonical fallback should not be used when paper_trades exist")

    monkeypatch.setattr(sprt_monitor, "_load_strategy_outcomes", _unexpected)

    trades, source = sprt_monitor._load_trade_stream(
        con,
        "SID1",
        {
            "instrument": "MNQ",
            "orb_label": "NYSE_CLOSE",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 1.0,
            "confirm_bars": 1,
            "filter_type": "ORB_G8",
        },
    )

    assert source == "paper_trades"
    assert trades == [1.25, -1.0]


def test_load_trade_stream_falls_back_to_canonical_forward(monkeypatch):
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE paper_trades (
            strategy_id VARCHAR,
            trading_day DATE,
            pnl_r DOUBLE
        )
        """
    )

    monkeypatch.setattr(
        sprt_monitor,
        "_load_strategy_outcomes",
        lambda *args, **kwargs: [
            {"pnl_r": 2.0, "outcome": "target"},
            {"pnl_r": None, "outcome": None},
            {"pnl_r": 0.0, "outcome": "scratch"},
            {"pnl_r": -1.0, "outcome": "stop"},
        ],
    )

    trades, source = sprt_monitor._load_trade_stream(
        con,
        "SID2",
        {
            "instrument": "MGC",
            "orb_label": "CME_REOPEN",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_type": "ORB_G6",
        },
    )

    assert source == "canonical_forward"
    assert trades == [2.0, -1.0]
