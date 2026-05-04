import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb

from research.oos_evidence import assess_targets


def _init_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA

    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    con.close()

    from trading_app.db_manager import init_trading_app_schema

    init_trading_app_schema(db_path=db_path)


def _base_row(**overrides) -> dict:
    row = {
        "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER",
        "instrument": "MNQ",
        "orb_label": "NYSE_OPEN",
        "orb_minutes": 5,
        "rr_target": 1.5,
        "confirm_bars": 1,
        "entry_model": "E2",
        "filter_type": "NO_FILTER",
        "filter_params": "{}",
        "stop_multiplier": 1.0,
        "sample_size": 200,
        "win_rate": 0.55,
        "avg_win_r": 1.8,
        "avg_loss_r": 1.0,
        "expectancy_r": 0.20,
        "sharpe_ratio": 0.40,
        "sharpe_ann": 1.20,
        "max_drawdown_r": 5.0,
        "median_risk_points": 10.0,
        "avg_risk_points": 10.5,
        "trades_per_year": 40.0,
        "p_value": 0.03,
        "is_canonical": True,
        "yearly_results": json.dumps(
            {
                "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.20},
                "2023": {"trades": 50, "wins": 27, "total_r": 8.0, "avg_r": 0.16},
                "2024": {"trades": 50, "wins": 29, "total_r": 11.0, "avg_r": 0.22},
                "2025": {"trades": 50, "wins": 28, "total_r": 9.0, "avg_r": 0.18},
            }
        ),
        "hypothesis_file_sha": "c" * 64,
        "created_at": datetime(2026, 4, 9, 12, 0, 0, tzinfo=UTC),
    }
    row.update(overrides)
    return row


def _insert_experimental(con, row: dict) -> None:
    cols = list(row.keys())
    placeholders = ", ".join(["?"] * len(cols))
    con.execute(
        f"INSERT INTO experimental_strategies ({', '.join(cols)}) VALUES ({placeholders})",
        list(row.values()),
    )


def _seed_oos(con, row: dict, oos_pnls: list[float]) -> None:
    base_day = date(2026, 1, 5)
    for i, pnl in enumerate(oos_pnls):
        day = base_day + timedelta(days=i)
        con.execute(
            "INSERT INTO daily_features (trading_day, symbol, orb_minutes, orb_nyse_open_break_dir) VALUES (?, ?, ?, ?)",
            [day, row["instrument"], row["orb_minutes"], "LONG"],
        )
        con.execute(
            """
            INSERT INTO orb_outcomes (
                trading_day, symbol, orb_minutes, orb_label, entry_model,
                confirm_bars, rr_target, outcome, pnl_r, mae_r, mfe_r,
                entry_price, stop_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                day,
                row["instrument"],
                row["orb_minutes"],
                row["orb_label"],
                row["entry_model"],
                row["confirm_bars"],
                row["rr_target"],
                "win" if pnl >= 0 else "loss",
                pnl,
                0.2,
                1.0,
                100.0,
                95.0,
            ],
        )


def test_assess_targets_sparse_oos_interprets_as_unverified(tmp_path):
    db_path = tmp_path / "oos_evidence_sparse.db"
    _init_db(db_path)
    con = duckdb.connect(str(db_path))
    row = _base_row(
        validation_status="REJECTED",
        rejection_reason="criterion_8: N_oos=8 < 30 (Amendment 3.0 condition 4: no insufficient-OOS-data exemptions for Pathway B individual testing mode)",
        validation_pathway="individual",
        c8_oos_status="INSUFFICIENT_N_PATHWAY_B_REJECT",
    )
    _insert_experimental(con, row)
    _seed_oos(con, row, [1.0, -0.5, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1])
    con.commit()
    con.close()

    report = assess_targets(db_path=db_path, strategy_id=row["strategy_id"])[0]
    assert report["c8_oos_status"] == "INSUFFICIENT_N_PATHWAY_B_REJECT"
    assert report["interpretation"] == "UNVERIFIED_SPARSE_OOS"
    assert report["oos_power_tier"] == "STATISTICALLY_USELESS"


def test_assess_targets_non_oos_reject_is_classified_separately(tmp_path):
    db_path = tmp_path / "oos_evidence_non_oos.db"
    _init_db(db_path)
    con = duckdb.connect(str(db_path))
    row = _base_row(
        validation_status="REJECTED",
        rejection_reason="criterion_3_pathway_b: raw p=0.0800>=0.05",
        validation_pathway="individual",
        c8_oos_status="PASSED",
        p_value=0.08,
    )
    _insert_experimental(con, row)
    _seed_oos(con, row, [1.0] * 12 + [-0.5] * 9 + [0.0] * 9)
    con.commit()
    con.close()

    report = assess_targets(db_path=db_path, strategy_id=row["strategy_id"])[0]
    assert report["c8_oos_status"] == "PASSED"
    assert report["interpretation"] == "REJECTED_NON_OOS"


def test_assess_targets_no_oos_is_inconclusive(tmp_path):
    db_path = tmp_path / "oos_evidence_no_oos.db"
    _init_db(db_path)
    con = duckdb.connect(str(db_path))
    row = _base_row(
        validation_status="REJECTED",
        rejection_reason="criterion_8: no OOS rows yet",
        validation_pathway="family",
        c8_oos_status="NO_OOS_DATA",
    )
    _insert_experimental(con, row)
    con.commit()
    con.close()

    report = assess_targets(db_path=db_path, strategy_id=row["strategy_id"])[0]
    assert report["interpretation"] == "INCONCLUSIVE_NO_OOS"
    assert report["n_oos"] == 0


def test_cli_json_supports_hypothesis_sha_lookup(tmp_path):
    db_path = tmp_path / "oos_evidence_cli.db"
    _init_db(db_path)
    con = duckdb.connect(str(db_path))
    row = _base_row(
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER_ALT",
        hypothesis_file_sha="d" * 64,
        validation_status="REJECTED",
        rejection_reason="criterion_8: N_oos=8 < 30 (Amendment 3.0 condition 4: no insufficient-OOS-data exemptions for Pathway B individual testing mode)",
        validation_pathway="individual",
        c8_oos_status="INSUFFICIENT_N_PATHWAY_B_REJECT",
    )
    _insert_experimental(con, row)
    _seed_oos(con, row, [1.0, -0.5, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1])
    con.commit()
    con.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.oos_evidence",
            "--db",
            str(db_path),
            "--hypothesis-sha",
            "d" * 64,
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["strategy_id"] == row["strategy_id"]
    assert payload[0]["interpretation"] == "UNVERIFIED_SPARSE_OOS"
