from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.mnq_atr_p50_cross_session_generalization_v1 import (
    _aperture_verdict,
    _bh_fdr,
    _overall_verdict,
)


def test_bh_fdr_is_monotone_and_nan_safe():
    series = pd.Series([0.01, 0.04, np.nan, 0.03, 0.20])
    result = _bh_fdr(series)
    assert math.isnan(result.iloc[2])
    finite = result.dropna().to_numpy()
    assert np.all(finite >= 0.0)
    assert np.all(finite <= 1.0)
    assert list(finite) == sorted(finite)


def test_aperture_verdict_generalizes_when_non_sgp_survivor_exists():
    df = pd.DataFrame(
        [
            {
                "session": "SINGAPORE_OPEN",
                "delta_expr": 0.10,
                "welch_p_raw": 0.01,
                "welch_q_local": 0.02,
                "powered": True,
            },
            {
                "session": "COMEX_SETTLE",
                "delta_expr": 0.06,
                "welch_p_raw": 0.02,
                "welch_q_local": 0.03,
                "powered": True,
            },
            {"session": "EUROPE_FLOW", "delta_expr": 0.03, "welch_p_raw": 0.30, "welch_q_local": 0.40, "powered": True},
        ]
    )
    verdict, median = _aperture_verdict(df)
    assert verdict == "CLASS_GENERALIZES"
    assert median > 0


def test_overall_verdict_kills_when_both_apertures_fail():
    verdict = _overall_verdict({15: "SGP_FAILS_RAW", 30: "KILL_FAMILY"}, {15: float("nan"), 30: -0.01})
    assert verdict == "KILL_FAMILY"
