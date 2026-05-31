from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import pandas as pd
import pytest

from research.track_d_gate0_microstructure import (
    DEFAULT_DRY_RUN_COST_SAMPLE_SIZE,
    SCHEMA,
    Gate0Window,
    _stratified_sample,
    assert_pre_entry_events,
    build_window_from_record,
    compute_feature_for_window,
    evaluate_sidecar,
    main,
    pull_pending,
    summarize_manifest,
    write_manifest,
)


def _window(*, break_dir: str = "long", pnl_r: float = 1.0, trading_day: str = "2025-01-02") -> Gate0Window:
    return build_window_from_record(
        {
            "trading_day": trading_day,
            "entry_ts": datetime.fromisoformat(f"{trading_day}T03:35:00+00:00"),
            "pnl_r": pnl_r,
            "break_dir": break_dir,
        },
        sha="test-sha",
        created_at=datetime(2026, 5, 29, tzinfo=UTC),
    )


def test_build_window_is_deterministic_and_exact_family() -> None:
    first = _window()
    second = _window()

    assert first.window_id == second.window_id
    assert first.symbol == "MNQ"
    assert first.orb_label == "COMEX_SETTLE"
    assert first.orb_minutes == 5
    assert first.entry_model == "E2"
    assert first.rr_target == 1.5
    assert first.confirm_bars == 1
    assert first.schema_used == SCHEMA
    assert first.window_start_utc == first.entry_ts - timedelta(seconds=60)
    assert first.window_end_utc == first.entry_ts + timedelta(seconds=16)


def test_build_window_rejects_null_entry_ts() -> None:
    with pytest.raises(ValueError, match="entry_ts is required"):
        build_window_from_record(
            {"trading_day": "2025-01-02", "entry_ts": None, "pnl_r": 1.0, "break_dir": "long"},
            sha="test-sha",
        )


def test_manifest_summary_splits_is_and_oos() -> None:
    rows = [
        _window(trading_day="2025-12-31"),
        _window(trading_day="2026-01-02"),
    ]

    assert summarize_manifest(rows) == {
        "total": 2,
        "is": 1,
        "oos": 1,
        "min_day": "2025-12-31",
        "max_day": "2026-01-02",
    }


def test_cost_dry_run_sample_is_small_and_deterministic() -> None:
    rows = [_window(trading_day=f"2025-01-{day:02d}") for day in range(1, 11)]

    sample = _stratified_sample(rows)

    assert len(sample) == DEFAULT_DRY_RUN_COST_SAMPLE_SIZE
    assert sample[0].window_id == rows[0].window_id
    assert sample[-1].window_id == rows[-1].window_id
    assert [row.window_id for row in sample] == [row.window_id for row in _stratified_sample(rows)]


def test_pre_entry_guard_rejects_entry_or_later_events() -> None:
    entry_ts = datetime(2025, 1, 2, 3, 35, tzinfo=UTC)
    events = pd.DataFrame({"ts_event": [entry_ts - timedelta(milliseconds=1), entry_ts]})

    with pytest.raises(ValueError, match="ts_event >= entry_ts"):
        assert_pre_entry_events(events, entry_ts)


def test_feature_math_signs_ofi_qi_and_tbi_for_long() -> None:
    window = _window(break_dir="long")
    events = pd.DataFrame(
        [
            {
                "ts_event": window.entry_ts - timedelta(seconds=3),
                "sequence": 1,
                "action": "A",
                "side": "N",
                "price": 100.0,
                "size": 0,
                "bid_px_00": 100.00,
                "ask_px_00": 100.25,
                "bid_sz_00": 10,
                "ask_sz_00": 10,
            },
            {
                "ts_event": window.entry_ts - timedelta(seconds=2),
                "sequence": 2,
                "action": "T",
                "side": "B",
                "price": 100.25,
                "size": 3,
                "bid_px_00": 100.25,
                "ask_px_00": 100.50,
                "bid_sz_00": 14,
                "ask_sz_00": 7,
            },
            {
                "ts_event": window.entry_ts - timedelta(seconds=1),
                "sequence": 3,
                "action": "T",
                "side": "A",
                "price": 100.25,
                "size": 1,
                "bid_px_00": 100.25,
                "ask_px_00": 100.50,
                "bid_sz_00": 12,
                "ask_sz_00": 6,
            },
        ]
    )

    feature = compute_feature_for_window(events, window)

    assert feature.signed_ofi_60s is not None
    assert feature.signed_ofi_60s > 0
    assert feature.signed_tbi_60s == 2.0
    assert feature.signed_qi_last_1s == pytest.approx((12 - 6) / (12 + 6))
    assert feature.event_count_mbp1 == 3
    assert feature.event_count_trade == 2


def test_feature_math_inverts_for_short() -> None:
    long_window = _window(break_dir="long")
    short_window = _window(break_dir="short")
    events = pd.DataFrame(
        [
            {
                "ts_event": long_window.entry_ts - timedelta(seconds=2),
                "sequence": 1,
                "action": "A",
                "side": "N",
                "price": 100.0,
                "size": 0,
                "bid_px_00": 100.00,
                "ask_px_00": 100.25,
                "bid_sz_00": 5,
                "ask_sz_00": 15,
            },
            {
                "ts_event": long_window.entry_ts - timedelta(seconds=1),
                "sequence": 2,
                "action": "T",
                "side": "B",
                "price": 100.25,
                "size": 4,
                "bid_px_00": 100.00,
                "ask_px_00": 100.25,
                "bid_sz_00": 6,
                "ask_sz_00": 18,
            },
        ]
    )

    long_feature = compute_feature_for_window(events, long_window)
    short_feature = compute_feature_for_window(events, short_window)

    assert short_feature.signed_tbi_60s == -long_feature.signed_tbi_60s
    assert short_feature.signed_qi_last_1s == -long_feature.signed_qi_last_1s


def test_zero_qi_denominator_stays_null() -> None:
    window = _window()
    events = pd.DataFrame(
        [
            {
                "ts_event": window.entry_ts - timedelta(seconds=1),
                "sequence": 1,
                "action": "A",
                "side": "N",
                "price": 100.0,
                "size": 0,
                "bid_px_00": 100.00,
                "ask_px_00": 100.25,
                "bid_sz_00": 0,
                "ask_sz_00": 0,
            }
        ]
    )

    feature = compute_feature_for_window(events, window)

    assert feature.signed_qi_last_1s is None
    assert feature.signed_qi_mean_10s is None


def test_pull_requires_cap_and_yes(tmp_path) -> None:
    sidecar = tmp_path / "track_d.duckdb"
    write_manifest([_window()], sidecar)

    with pytest.raises(ValueError, match="--max-cost-usd"):
        pull_pending(sidecar, max_cost_usd=None, yes=True)
    with pytest.raises(ValueError, match="--yes"):
        pull_pending(sidecar, max_cost_usd=75.0, yes=False)


def test_cli_pull_refuses_without_cap_and_yes(tmp_path) -> None:
    sidecar = tmp_path / "track_d.duckdb"
    write_manifest([_window()], sidecar)

    with pytest.raises(ValueError, match="--max-cost-usd"):
        main(["--pull", "--sidecar-db", str(sidecar)])
    with pytest.raises(ValueError, match="--yes"):
        main(["--pull", "--sidecar-db", str(sidecar), "--max-cost-usd", "75"])


def test_pull_refuses_when_recorded_cost_exceeds_cap(tmp_path) -> None:
    sidecar = tmp_path / "track_d.duckdb"
    row = _window()
    write_manifest([row], sidecar)
    con = duckdb.connect(str(sidecar))
    try:
        con.execute("UPDATE micro_gate0_windows SET metadata_cost_usd = 80 WHERE window_id = ?", [row.window_id])
    finally:
        con.close()

    with pytest.raises(ValueError, match="exceeds cap"):
        pull_pending(sidecar, max_cost_usd=75.0, yes=True)


def test_evaluate_uses_is_thresholds_only(tmp_path) -> None:
    sidecar = tmp_path / "track_d.duckdb"
    rows = [
        _window(trading_day="2025-01-02", pnl_r=-1.0),
        _window(trading_day="2025-01-03", pnl_r=1.0),
        _window(trading_day="2026-01-02", pnl_r=5.0),
    ]
    write_manifest(rows, sidecar)
    con = duckdb.connect(str(sidecar))
    try:
        con.execute(
            """
            INSERT INTO micro_gate0_features VALUES
            ('f1', ?, DATE '2025-01-02', 'MNQ', 'COMEX_SETTLE', 5, 'E2', 1.5, 1, 'long',
             TIMESTAMPTZ '2025-01-02 03:35:00+00', 60, 1, 1, 1, 1, 1, 1, 2, 1, 'test'),
            ('f2', ?, DATE '2025-01-03', 'MNQ', 'COMEX_SETTLE', 5, 'E2', 1.5, 1, 'long',
             TIMESTAMPTZ '2025-01-03 03:35:00+00', 60, 3, 3, 3, 3, 1, 1, 2, 1, 'test'),
            ('f3', ?, DATE '2026-01-02', 'MNQ', 'COMEX_SETTLE', 5, 'E2', 1.5, 1, 'long',
             TIMESTAMPTZ '2026-01-02 03:35:00+00', 60, 100, 100, 100, 100, 1, 1, 2, 1, 'test')
            """,
            [rows[0].window_id, rows[1].window_id, rows[2].window_id],
        )
    finally:
        con.close()

    results = evaluate_sidecar(sidecar)
    ofi_is = results[(results["feature_id"] == "signed_ofi_60s_high") & (results["sample_split"] == "IS")].iloc[0]

    assert ofi_is["threshold_value"] == pytest.approx(2.5)


def test_cli_evaluate_refuses_oos_only_threshold_fit(tmp_path) -> None:
    sidecar = tmp_path / "track_d.duckdb"
    row = _window(trading_day="2026-01-02", pnl_r=5.0)
    write_manifest([row], sidecar)
    con = duckdb.connect(str(sidecar))
    try:
        con.execute(
            """
            INSERT INTO micro_gate0_features VALUES
            ('f1', ?, DATE '2026-01-02', 'MNQ', 'COMEX_SETTLE', 5, 'E2', 1.5, 1, 'long',
             TIMESTAMPTZ '2026-01-02 03:35:00+00', 60, 100, 100, 100, 100, 1, 1, 2, 1, 'test')
            """,
            [row.window_id],
        )
    finally:
        con.close()

    with pytest.raises(ValueError, match="without IS rows"):
        main(["--evaluate", "--sidecar-db", str(sidecar)])
