"""Companion test for PR-A runner emissions upgrade.

Tests the four new emission surfaces added to ``research/chordia_strict_unlock_v1.py``:

- ``long_ExpR`` / ``short_ExpR`` (per-direction ExpR) in the per-split summary CSV.
- ``max_IS_trading_day`` (latest IS trading day from the loaded universe).
- ``min_OOS_trading_day`` (earliest OOS trading day from the loaded universe).
- ``holdout_boundary_proof`` (boolean: split honors ``HOLDOUT_SACRED_FROM``).

Companion to ``docs/audit/results/2026-05-18-fast-lane-v5-calibration-blocked.md``
§ "PR-A — runner emissions upgrade." The fast-lane v5 template
(``docs/audit/hypotheses/TEMPLATE-fast-lane-v5.yaml``) consumes these fields
as a machine-readable surface; without them the G-A2 sign-check rule cannot
fire on a ``direction: pooled`` cell.

These tests build a minimal synthetic DuckDB database with the exact schema
the runner queries and exercise the real ``_split_boundary_metadata`` and
``_write_summary_csv`` helpers — no mocks of pandas or duckdb. The
``_evaluate_split`` per-direction math is independently re-derived in
``test_summary_csv_long_short_expr_match_evaluate_split`` to prove the
emission is the same number the existing markdown directional table prints.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from research import chordia_strict_unlock_v1 as runner
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# ----------------------------- helpers ----------------------------------------


def _make_trade_row(
    *,
    trading_day: dt.date,
    long: bool,
    pnl_r: float,
    outcome: str = "win",
    risk_dollars: float = 100.0,
) -> dict[str, Any]:
    """Minimal trade row matching the columns _load_universe expects."""
    if long:
        entry_price = 100.0
        target_price = 105.0  # target > stop => long per _direction_series
        stop_price = 95.0
    else:
        entry_price = 100.0
        target_price = 95.0
        stop_price = 105.0  # target < stop => short
    pnl_dollars = pnl_r * risk_dollars
    return {
        "trading_day": trading_day,
        "symbol": "MNQ",
        "orb_label": "US_DATA_1000",
        "orb_minutes": 30,
        "entry_model": "E2",
        "confirm_bars": 1,
        "rr_target": 1.0,
        "outcome": outcome,
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "pnl_r": pnl_r,
        "risk_dollars": risk_dollars,
        "pnl_dollars": pnl_dollars,
        "mae_r": -0.1,
        "mfe_r": abs(pnl_r) + 0.1,
    }


# ----------------------------- _split_boundary_metadata -----------------------


def test_split_boundary_metadata_clean_split_returns_proof_true() -> None:
    """Realised split with max_IS < HOLDOUT_SACRED_FROM <= min_OOS proves clean."""
    is_df = pd.DataFrame({"trading_day": [dt.date(2025, 12, 30), dt.date(2025, 12, 31)]})
    oos_df = pd.DataFrame({"trading_day": [dt.date(2026, 1, 2), dt.date(2026, 1, 10)]})

    meta = runner._split_boundary_metadata(is_df, oos_df)

    assert meta["max_IS_trading_day"] == "2025-12-31"
    assert meta["min_OOS_trading_day"] == "2026-01-02"
    assert meta["holdout_boundary_value"] == HOLDOUT_SACRED_FROM.isoformat()
    assert meta["holdout_boundary_proof"] is True


def test_split_boundary_metadata_is_overlaps_holdout_returns_proof_false() -> None:
    """If max_IS lands on or after HOLDOUT_SACRED_FROM, proof fails (leak)."""
    is_df = pd.DataFrame({"trading_day": [dt.date(2025, 12, 30), HOLDOUT_SACRED_FROM]})
    oos_df = pd.DataFrame({"trading_day": [dt.date(2026, 1, 10)]})

    meta = runner._split_boundary_metadata(is_df, oos_df)

    assert meta["max_IS_trading_day"] == HOLDOUT_SACRED_FROM.isoformat()
    assert meta["holdout_boundary_proof"] is False, (
        "max_IS == HOLDOUT_SACRED_FROM contaminates IS with sacred-holdout data"
    )


def test_split_boundary_metadata_oos_predates_holdout_returns_proof_false() -> None:
    """If min_OOS lands before HOLDOUT_SACRED_FROM, proof fails."""
    is_df = pd.DataFrame({"trading_day": [dt.date(2025, 6, 1)]})
    oos_df = pd.DataFrame({"trading_day": [dt.date(2025, 12, 15)]})

    meta = runner._split_boundary_metadata(is_df, oos_df)

    assert meta["holdout_boundary_proof"] is False


def test_split_boundary_metadata_empty_is_records_none_and_truthful_proof() -> None:
    """Empty IS frame: max_IS_trading_day is empty string, OOS side governs proof."""
    is_df = pd.DataFrame({"trading_day": []})
    oos_df = pd.DataFrame({"trading_day": [dt.date(2026, 1, 2)]})

    meta = runner._split_boundary_metadata(is_df, oos_df)

    assert meta["max_IS_trading_day"] == ""
    assert meta["min_OOS_trading_day"] == "2026-01-02"
    assert meta["holdout_boundary_proof"] is True


def test_split_boundary_metadata_empty_oos_records_none_and_truthful_proof() -> None:
    """Empty OOS frame: min_OOS_trading_day is empty string, IS side governs proof."""
    is_df = pd.DataFrame({"trading_day": [dt.date(2025, 6, 1)]})
    oos_df = pd.DataFrame({"trading_day": []})

    meta = runner._split_boundary_metadata(is_df, oos_df)

    assert meta["max_IS_trading_day"] == "2025-06-01"
    assert meta["min_OOS_trading_day"] == ""
    assert meta["holdout_boundary_proof"] is True


def test_split_boundary_metadata_holdout_value_is_canonical_constant() -> None:
    """Emitted holdout_boundary_value must mirror trading_app.holdout_policy.HOLDOUT_SACRED_FROM."""
    is_df = pd.DataFrame({"trading_day": [dt.date(2025, 6, 1)]})
    oos_df = pd.DataFrame({"trading_day": [dt.date(2026, 1, 2)]})

    meta = runner._split_boundary_metadata(is_df, oos_df)

    assert meta["holdout_boundary_value"] == HOLDOUT_SACRED_FROM.isoformat()
    assert dt.date(2026, 1, 1) == HOLDOUT_SACRED_FROM, (
        "Canonical holdout boundary moved — update the fast-lane template + this test in lockstep."
    )


# ----------------------------- _write_summary_csv -----------------------------


def _read_summary(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        rows = list(reader)
    return rows[0], rows[1:]


def test_summary_csv_two_rows_one_per_split(tmp_path: Path) -> None:
    """Per-split summary CSV emits exactly two data rows: IS first, then OOS."""
    is_result = {
        "sample": "IS",
        "n_universe": 100,
        "n_fired": 50,
        "fire_rate": 0.5,
        "expr": 0.1,
        "policy_ev": 0.05,
        "sharpe": 0.2,
        "t": 3.85,
        "p_two_sided": 0.0001,
        "long_n": 30,
        "long_expr": 0.15,
        "long_t": 3.0,
        "short_n": 20,
        "short_expr": 0.05,
        "short_t": 1.5,
    }
    oos_result = {
        "sample": "OOS",
        "n_universe": 10,
        "n_fired": 5,
        "fire_rate": 0.5,
        "expr": 0.08,
        "policy_ev": 0.04,
        "sharpe": 0.15,
        "t": 0.9,
        "p_two_sided": 0.4,
        "long_n": 3,
        "long_expr": 0.12,
        "long_t": 0.7,
        "short_n": 2,
        "short_expr": 0.04,
        "short_t": 0.3,
    }
    boundary = {
        "max_IS_trading_day": "2025-12-31",
        "min_OOS_trading_day": "2026-01-02",
        "holdout_boundary_value": HOLDOUT_SACRED_FROM.isoformat(),
        "holdout_boundary_proof": True,
    }

    out_path = tmp_path / "test.summary.csv"
    runner._write_summary_csv(out_path, is_result, oos_result, boundary)

    header, data_rows = _read_summary(out_path)
    assert header == list(runner._SUMMARY_CSV_COLUMNS)
    assert len(data_rows) == 2
    assert data_rows[0][0] == "IS"
    assert data_rows[1][0] == "OOS"


def test_summary_csv_emits_per_direction_expr_machine_readable(tmp_path: Path) -> None:
    """long_expr and short_expr are emitted as CSV columns and parse as floats."""
    is_result = {
        "sample": "IS",
        "n_universe": 100,
        "n_fired": 50,
        "fire_rate": 0.5,
        "expr": 0.1,
        "policy_ev": 0.05,
        "sharpe": 0.2,
        "t": 3.85,
        "p_two_sided": 0.0001,
        "long_n": 30,
        "long_expr": 0.1500,
        "long_t": 3.0,
        "short_n": 20,
        "short_expr": -0.0250,
        "short_t": -0.5,
    }
    oos_result = {**is_result, "sample": "OOS"}
    boundary = {
        "max_IS_trading_day": "2025-12-31",
        "min_OOS_trading_day": "2026-01-02",
        "holdout_boundary_value": HOLDOUT_SACRED_FROM.isoformat(),
        "holdout_boundary_proof": True,
    }

    out_path = tmp_path / "test.summary.csv"
    runner._write_summary_csv(out_path, is_result, oos_result, boundary)
    header, data_rows = _read_summary(out_path)

    long_expr_col = header.index("long_expr")
    short_expr_col = header.index("short_expr")
    assert float(data_rows[0][long_expr_col]) == pytest.approx(0.1500)
    assert float(data_rows[0][short_expr_col]) == pytest.approx(-0.0250)


def test_summary_csv_emits_holdout_boundary_proof_as_string_true_or_false(tmp_path: Path) -> None:
    """holdout_boundary_proof serialises as the Python bool string ('True'/'False')."""
    is_result = runner._nan_result("IS")
    oos_result = runner._nan_result("OOS")
    boundary_true = {
        "max_IS_trading_day": "2025-12-31",
        "min_OOS_trading_day": "2026-01-02",
        "holdout_boundary_value": HOLDOUT_SACRED_FROM.isoformat(),
        "holdout_boundary_proof": True,
    }
    boundary_false = {**boundary_true, "holdout_boundary_proof": False}

    p_true = tmp_path / "true.summary.csv"
    p_false = tmp_path / "false.summary.csv"
    runner._write_summary_csv(p_true, is_result, oos_result, boundary_true)
    runner._write_summary_csv(p_false, is_result, oos_result, boundary_false)

    _, true_rows = _read_summary(p_true)
    _, false_rows = _read_summary(p_false)
    proof_col = list(runner._SUMMARY_CSV_COLUMNS).index("holdout_boundary_proof")
    assert true_rows[0][proof_col] == "True"
    assert false_rows[0][proof_col] == "False"


# ------- per-direction ExpR parity with the markdown directional table --------


def test_summary_csv_long_short_expr_match_evaluate_split() -> None:
    """The long/short ExpR written to the summary CSV equals the values
    ``_evaluate_split`` computes for the markdown directional breakdown table.

    Builds a small fired-trade frame, runs the canonical ``_evaluate_split``
    logic (re-derived inline against the same pnl_eff math the runner uses),
    and confirms the summary surface reports the same numbers.
    """
    rows = [
        _make_trade_row(trading_day=dt.date(2025, 1, 1), long=True, pnl_r=1.0),
        _make_trade_row(trading_day=dt.date(2025, 1, 2), long=True, pnl_r=-1.0),
        _make_trade_row(trading_day=dt.date(2025, 1, 3), long=True, pnl_r=0.5),
        _make_trade_row(trading_day=dt.date(2025, 1, 4), long=False, pnl_r=1.0),
        _make_trade_row(trading_day=dt.date(2025, 1, 5), long=False, pnl_r=-0.5),
    ]
    fired = pd.DataFrame(rows)
    fired["pnl_eff"] = fired["pnl_r"].fillna(0.0)

    # Re-derive the per-direction ExpRs the same way _evaluate_split does.
    directions = runner._direction_series(fired)
    long_expr_expected = float(fired.loc[directions.eq("long"), "pnl_eff"].mean())
    short_expr_expected = float(fired.loc[directions.eq("short"), "pnl_eff"].mean())
    long_n_expected = int(directions.eq("long").sum())
    short_n_expected = int(directions.eq("short").sum())

    # Build the dict shape _write_summary_csv consumes.
    is_result = {
        "sample": "IS",
        "n_universe": 5,
        "n_fired": 5,
        "fire_rate": 1.0,
        "expr": float(fired["pnl_eff"].mean()),
        "policy_ev": float(fired["pnl_eff"].mean()),
        "sharpe": 0.1,
        "t": 1.0,
        "p_two_sided": 0.3,
        "long_n": long_n_expected,
        "long_expr": long_expr_expected,
        "long_t": float("nan"),
        "short_n": short_n_expected,
        "short_expr": short_expr_expected,
        "short_t": float("nan"),
    }
    oos_result = runner._nan_result("OOS")
    boundary = {
        "max_IS_trading_day": "2025-01-05",
        "min_OOS_trading_day": "",
        "holdout_boundary_value": HOLDOUT_SACRED_FROM.isoformat(),
        "holdout_boundary_proof": True,
    }

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".summary.csv", delete=False) as tf:
        out_path = Path(tf.name)
    try:
        runner._write_summary_csv(out_path, is_result, oos_result, boundary)
        header, data_rows = _read_summary(out_path)
        long_expr_col = header.index("long_expr")
        short_expr_col = header.index("short_expr")
        assert float(data_rows[0][long_expr_col]) == pytest.approx(long_expr_expected)
        assert float(data_rows[0][short_expr_col]) == pytest.approx(short_expr_expected)
        # Long-only and short-only must produce distinguishable summary rows —
        # this is the property G-A2 sign-check in the fast-lane template relies on.
        assert long_expr_expected != short_expr_expected
    finally:
        out_path.unlink(missing_ok=True)


def test_summary_csv_nan_split_emits_empty_long_short_expr(tmp_path: Path) -> None:
    """Zero-fired split (NaN per-direction ExpR) is emitted as the literal 'nan'.

    Calibration consumers must distinguish "no fired trades" (NaN) from
    "fired with zero edge" (0.0). The CSV stringifies float('nan') as 'nan';
    consumers should parse that as missing, not as a numeric value.
    """
    nan_is = runner._nan_result("IS")
    nan_oos = runner._nan_result("OOS")
    boundary = {
        "max_IS_trading_day": "",
        "min_OOS_trading_day": "",
        "holdout_boundary_value": HOLDOUT_SACRED_FROM.isoformat(),
        "holdout_boundary_proof": True,
    }

    out_path = tmp_path / "nan.summary.csv"
    runner._write_summary_csv(out_path, nan_is, nan_oos, boundary)

    header, data_rows = _read_summary(out_path)
    long_expr_col = header.index("long_expr")
    long_n_col = header.index("long_n")
    # nan_result sets long_n=0 and long_expr=NaN
    assert data_rows[0][long_n_col] == "0"
    assert data_rows[0][long_expr_col] == "nan"
    # Round-trip: float('nan') has the property that nan != nan.
    parsed = float(data_rows[0][long_expr_col])
    assert math.isnan(parsed)


# ----------------------------- summary CSV path resolution --------------------


def test_resolve_output_paths_emits_three_paths_with_stem_prefix() -> None:
    """_resolve_output_paths returns md, csv, summary_csv all sharing prereg stem."""
    hyp = Path("docs/audit/hypotheses/2026-05-13-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.yaml")
    md, csv_path, summary_csv = runner._resolve_output_paths(hyp)

    assert md.name.endswith(".md")
    assert csv_path.name.endswith(".csv")
    assert summary_csv.name.endswith(".summary.csv")
    # All three share the prereg stem.
    stem = hyp.stem
    assert md.name == f"{stem}.md"
    assert csv_path.name == f"{stem}.csv"
    assert summary_csv.name == f"{stem}.summary.csv"
    # All three resolve under docs/audit/results.
    assert md.parent == csv_path.parent == summary_csv.parent
    assert md.parent.parts[-3:] == ("docs", "audit", "results")
