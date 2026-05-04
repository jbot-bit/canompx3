from __future__ import annotations

from datetime import date

import duckdb

from research.garch_r3_shadow_ledger import _audit_feature_gaps


def test_audit_feature_gaps_returns_empty_typed_frame_when_no_gaps(tmp_path):
    db_path = tmp_path / "garch_shadow.db"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE daily_features (
                trading_day DATE,
                symbol VARCHAR,
                orb_minutes INTEGER,
                garch_forecast_vol_pct DOUBLE
            )
            """
        )
        con.execute(
            """
            INSERT INTO daily_features VALUES
                ('2026-04-08', 'MNQ', 5, 42.0),
                ('2026-04-09', 'MNQ', 5, 58.0)
            """
        )

        lane_defs = [{"instrument": "MNQ", "orb_minutes": 5}]
        result = _audit_feature_gaps(con, lane_defs, date(2026, 4, 8), date(2026, 4, 9))

        assert result.empty
        assert list(result.columns) == ["trading_day", "instrument", "orb_minutes"]
    finally:
        con.close()
