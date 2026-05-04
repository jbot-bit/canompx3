from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research.research_mgc_e2_microstructure_pilot import choose_microstructure_schema, stratified_sample_days


def test_choose_microstructure_schema_prefers_tbbo_then_mbp1():
    assert choose_microstructure_schema(["ohlcv-1m", "mbp-1", "trades"]) == "mbp-1"
    assert choose_microstructure_schema(["tbbo", "trades"]) == "tbbo"
    assert choose_microstructure_schema(["trades", "ohlcv-1m"]) is None


def test_stratified_sample_days_is_balanced_and_deterministic():
    rows = []
    for idx in range(6):
        rows.append({"trading_day": date(2025, 1, idx + 1), "atr_bucket": "high_vol", "atr_20_pct": 75.0 + idx})
        rows.append({"trading_day": date(2025, 2, idx + 1), "atr_bucket": "low_vol", "atr_20_pct": 25.0 + idx})
    df = pd.DataFrame(rows)

    sample_a = stratified_sample_days(df, high_count=3, low_count=2, seed=11)
    sample_b = stratified_sample_days(df, high_count=3, low_count=2, seed=11)

    assert sample_a["trading_day"].tolist() == sample_b["trading_day"].tolist()
    assert (sample_a["atr_bucket"] == "high_vol").sum() == 3
    assert (sample_a["atr_bucket"] == "low_vol").sum() == 2


def test_stratified_sample_days_requires_enough_rows():
    df = pd.DataFrame(
        [
            {"trading_day": date(2025, 1, 1), "atr_bucket": "high_vol", "atr_20_pct": 80.0},
            {"trading_day": date(2025, 1, 2), "atr_bucket": "low_vol", "atr_20_pct": 20.0},
        ]
    )

    with pytest.raises(ValueError, match="Need 2 high-vol days"):
        stratified_sample_days(df, high_count=2, low_count=1, seed=3)
