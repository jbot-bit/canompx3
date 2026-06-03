from __future__ import annotations

from datetime import date

from trading_app import validation_provenance
from trading_app.validation_provenance import StrategyTradeWindowResolver


def test_volume_enrichment_reused_for_same_session_and_lookback(monkeypatch):
    trade_day = date(2020, 1, 2)
    features = [{"trading_day": trade_day, "orb_CME_REOPEN_break_dir": "LONG"}]
    enrichment_calls = []

    def fake_load_daily_features(con, instrument, orb_minutes, start_date, end_date):
        assert start_date is None
        assert end_date is None
        return features

    def fake_compute_relative_volumes(con, rows, instrument, orb_labels, filters):
        enrichment_calls.append((instrument, tuple(orb_labels), tuple(filters)))
        for row in rows:
            row["rel_vol_CME_REOPEN"] = 2.0

    def fake_build_filter_day_sets(rows, orb_labels, filters):
        assert rows is features
        return {(filter_type, orb_labels[0]): {trade_day} for filter_type in filters}

    def fake_load_outcomes_bulk(con, instrument, orb_minutes, orb_labels, entry_models, holdout_date, start_date):
        assert holdout_date is None
        assert start_date is None
        return {
            ("CME_REOPEN", "E1", 1.0, 1): [
                {"trading_day": trade_day},
            ]
        }

    monkeypatch.setattr(validation_provenance, "_load_daily_features", fake_load_daily_features)
    monkeypatch.setattr(validation_provenance, "_compute_relative_volumes", fake_compute_relative_volumes)
    monkeypatch.setattr(validation_provenance, "_build_filter_day_sets", fake_build_filter_day_sets)
    monkeypatch.setattr(validation_provenance, "_load_outcomes_bulk", fake_load_outcomes_bulk)

    resolver = StrategyTradeWindowResolver(con=object())

    first = resolver.resolve(
        instrument="MNQ",
        orb_label="CME_REOPEN",
        orb_minutes=5,
        entry_model="E1",
        rr_target=1.0,
        confirm_bars=1,
        filter_type="VOL_RV12_N20",
    )
    second = resolver.resolve(
        instrument="MNQ",
        orb_label="CME_REOPEN",
        orb_minutes=5,
        entry_model="E1",
        rr_target=1.0,
        confirm_bars=1,
        filter_type="VOL_RV15_N20",
    )

    assert first.first_trade_day == trade_day
    assert second.first_trade_day == trade_day
    assert enrichment_calls == [("MNQ", ("CME_REOPEN",), ("VOL_RV12_N20",))]
