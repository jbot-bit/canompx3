from __future__ import annotations

import pandas as pd

from research.mnq_nyse_close_orbg8_rr10_prereg import Verdict, evaluate_verdict


def _summary_row(
    partition: str,
    n: int,
    fire_rate: float,
    expr: float,
    win_rate: float,
    t_stat: float = float("nan"),
    p_one: float = float("nan"),
) -> dict:
    return {
        "partition": partition,
        "n": n,
        "fire_rate": fire_rate,
        "expr": expr,
        "win_rate": win_rate,
        "t_stat": t_stat,
        "p_one": p_one,
    }


def test_evaluate_verdict_kills_on_era_instability() -> None:
    is_summary = pd.DataFrame(
        [
            _summary_row("baseline", 805, 1.0, 0.08, 0.33),
            _summary_row("on_signal", 720, 0.894, 0.11, 0.60, t_stat=3.28, p_one=0.0005),
            _summary_row("off_signal", 85, 0.106, -0.14, 0.10),
        ]
    )
    yearly_on = pd.DataFrame(
        {
            "year": [2024, 2025],
            "n_on": [109, 132],
            "expr_on": [0.15, -0.07],
        }
    )
    oos_summary = pd.DataFrame(
        [
            _summary_row("baseline", 42, 1.0, 0.48, 0.79),
            _summary_row("on_signal", 42, 1.0, 0.48, 0.79),
            _summary_row("off_signal", 0, float("nan"), float("nan"), float("nan")),
        ]
    )
    verdict = evaluate_verdict(is_summary, yearly_on, oos_summary)
    assert verdict == Verdict("KILL", "Era-stability kill triggered on year(s): 2025.")


def test_evaluate_verdict_continues_when_all_prereg_gates_pass() -> None:
    is_summary = pd.DataFrame(
        [
            _summary_row("baseline", 805, 1.0, 0.08, 0.33),
            _summary_row("on_signal", 400, 0.50, 0.12, 0.58, t_stat=3.4, p_one=0.0004),
            _summary_row("off_signal", 405, 0.50, 0.04, 0.32),
        ]
    )
    yearly_on = pd.DataFrame(
        {
            "year": [2022, 2023, 2024, 2025],
            "n_on": [60, 70, 80, 90],
            "expr_on": [0.08, 0.04, 0.07, 0.02],
        }
    )
    oos_summary = pd.DataFrame(
        [
            _summary_row("baseline", 20, 1.0, 0.03, 0.55),
            _summary_row("on_signal", 12, 0.60, 0.02, 0.58),
            _summary_row("off_signal", 8, 0.40, 0.01, 0.50),
        ]
    )
    verdict = evaluate_verdict(is_summary, yearly_on, oos_summary)
    assert verdict == Verdict(
        "CONTINUE", "Exact ORB_G8 path survives its prereg and can move to a separate promotion gate."
    )
