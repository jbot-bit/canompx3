"""Sequential Probability Ratio Test (SPRT) monitor for lane degradation.

Wald SPRT formulation:
  H0: lane ExpR = backtest ExpR (performing as expected)
  H1: lane ExpR = 0 (no edge)
  alpha=0.10 (type I error), beta=0.20 (type II error)
  A = log((1-beta)/alpha) = log(0.80/0.10) = log(8) = 2.079
  B = log((1-alpha)/beta) = log(0.90/0.20) = log(4.5) = 1.504

At each trade, update log-likelihood ratio.
If LR < -A: accept H1 (DEGRADED — lane has no edge)
If LR > B: accept H0 (SIGNAL — lane performing as expected)
Otherwise: CONTINUE (not enough evidence)

Usage:
    python -m trading_app.sprt_monitor
"""

import json
import math
import sys
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.live.performance_monitor import _compute_std_r
from trading_app.strategy_fitness import _load_strategy_outcomes

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# SPRT boundaries (Wald)
ALPHA = 0.10
BETA = 0.20
A = math.log((1 - BETA) / ALPHA)  # 2.079 — reject H0 (accept DEGRADED) when LR < -A
B = math.log((1 - ALPHA) / BETA)  # 1.504 — accept H0 (SIGNAL) when LR > B

def _load_reference_stats() -> dict[str, dict]:
    """Load SPRT reference parameters from current active validated rows."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        rows = con.execute(
            """
            SELECT strategy_id, win_rate, rr_target, expectancy_r
            FROM validated_setups
            WHERE LOWER(status) = 'active'
            """
        ).fetchall()
    finally:
        con.close()

    stats: dict[str, dict] = {}
    for strategy_id, win_rate, rr_target, expectancy_r in rows:
        if win_rate is None or rr_target is None or expectancy_r is None:
            continue
        mu0 = float(expectancy_r)
        stats[strategy_id] = {
            "mu0": mu0,
            "sigma": _compute_std_r(float(win_rate), float(rr_target), mu0),
        }
    return stats


def _build_lanes() -> dict[str, dict]:
    from trading_app.prop_profiles import get_profile_lane_definitions, resolve_profile_id

    lanes = {}
    profile_id = resolve_profile_id()
    reference_stats = _load_reference_stats()
    for i, lane in enumerate(get_profile_lane_definitions(profile_id), 1):
        label = lane["orb_label"]
        strategy_id = lane["strategy_id"]
        bt = reference_stats.get(strategy_id, {"mu0": 0.10, "sigma": 1.0})
        suffix = " (shadow)" if lane.get("shadow_only") else ""
        lanes[strategy_id] = {
            "mu0": bt["mu0"],
            "sigma": bt["sigma"],
            "instrument": lane["instrument"],
            "orb_label": label,
            "orb_minutes": lane["orb_minutes"],
            "entry_model": lane["entry_model"],
            "rr_target": lane["rr_target"],
            "confirm_bars": lane["confirm_bars"],
            "filter_type": lane["filter_type"],
            "label": f"L{i} {label} {lane['filter_type']}{suffix}",
        }
    return lanes


def _load_trade_stream(
    con: duckdb.DuckDBPyConnection,
    strategy_id: str,
    params: dict,
) -> tuple[list[float], str]:
    """Load the best available forward trade stream for SPRT monitoring.

    Source priority:
    1. `paper_trades` for actual logged live/shadow execution history
    2. canonical forward outcomes as a transparent shadow fallback

    Criterion 12 is about the live R stream, so we prefer paper/live logs.
    The canonical fallback keeps the monitor operational when paper-trade
    coverage is incomplete, but the source is labeled explicitly.
    """
    paper_rows = con.execute(
        """SELECT pnl_r FROM paper_trades
           WHERE strategy_id = ? AND pnl_r IS NOT NULL
           ORDER BY trading_day""",
        [strategy_id],
    ).fetchall()
    paper_trades = [r[0] for r in paper_rows]
    if paper_trades:
        return paper_trades, "paper_trades"

    outcomes = _load_strategy_outcomes(
        con,
        instrument=params["instrument"],
        orb_label=params["orb_label"],
        orb_minutes=params["orb_minutes"],
        entry_model=params["entry_model"],
        rr_target=params["rr_target"],
        confirm_bars=params["confirm_bars"],
        filter_type=params["filter_type"],
        start_date=date(2026, 1, 1),
    )
    forward_trades = [
        o["pnl_r"]
        for o in outcomes
        if o.get("pnl_r") is not None and o.get("outcome") not in (None, "scratch")
    ]
    if forward_trades:
        return forward_trades, "canonical_forward"

    return [], "none"


def compute_sprt(trades: list[float], mu0: float, sigma: float) -> tuple[float, str]:
    """Compute SPRT log-likelihood ratio.

    Under H0: X ~ N(mu0, sigma^2)
    Under H1: X ~ N(0, sigma^2)

    Log-LR per trade = (x * mu0) / sigma^2 - mu0^2 / (2 * sigma^2)
    Cumulative LR = sum of per-trade log-LRs.
    """
    if sigma <= 0 or mu0 <= 0:
        return 0.0, "CONTINUE"

    sigma2 = sigma**2
    lr = 0.0
    for x in trades:
        lr += (x * mu0) / sigma2 - (mu0**2) / (2 * sigma2)

    if lr < -A:
        return lr, "DEGRADED"
    elif lr > B:
        return lr, "SIGNAL"
    else:
        return lr, "CONTINUE"


def compute_streak(trades: list[float]) -> tuple[int, int]:
    """Compute current consecutive loss streak and max streak."""
    current = 0
    max_streak = 0
    for x in trades:
        if x < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return current, max_streak


def run_monitor():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)

    print(f"{'=' * 90}")
    print(f"SPRT DEGRADATION MONITOR | {date.today()}")
    print(f"Boundaries: A={A:.3f} (DEGRADED), B={B:.3f} (SIGNAL)")
    print(f"Alpha={ALPHA}, Beta={BETA}")
    print(f"{'=' * 90}")
    print(
        f"\n{'Lane':<40} {'N':>4} {'SPRT':>8} {'Lower':>8} {'Upper':>8} "
        f"{'Status':<12} {'Source':<18} {'Streak':>6} {'MaxStr':>6}"
    )
    print("-" * 129)

    lanes = _build_lanes()
    results = []
    for strategy_id, params in lanes.items():
        trades, trade_source = _load_trade_stream(con, strategy_id, params)
        n = len(trades)

        if n == 0:
            print(
                f"{params['label']:<40} {0:>4} {'N/A':>8} {-A:>8.3f} {B:>8.3f} "
                f"{'NO DATA':<12} {trade_source:<18} {0:>6} {0:>6}"
            )
            results.append(
                {
                    "strategy_id": strategy_id,
                    "orb_label": params["orb_label"],
                    "n": 0,
                    "sprt": 0,
                    "status": "NO_DATA",
                    "source": trade_source,
                }
            )
            continue

        lr, status = compute_sprt(trades, params["mu0"], params["sigma"])
        current_streak, max_streak = compute_streak(trades)

        print(
            f"{params['label']:<40} {n:>4} {lr:>+8.3f} {-A:>8.3f} {B:>8.3f} {status:<12} "
            f"{trade_source:<18} {current_streak:>6} {max_streak:>6}"
        )

        # Streak warnings
        if current_streak >= 8:
            print(f"  ** ALERT: {current_streak} consecutive losses. P(this|WR) computed below. **")
        elif current_streak >= 5:
            wr_est = sum(1 for t in trades if t > 0) / n if n > 0 else 0.5
            p_streak = (1 - wr_est) ** current_streak
            print(f"  WARNING: {current_streak} consecutive losses (p={p_streak:.4f} under WR={wr_est:.1%})")

        results.append(
            {
                "strategy_id": strategy_id,
                "orb_label": params["orb_label"],
                "n": n,
                "sprt": round(lr, 4),
                "status": status,
                "source": trade_source,
                "current_streak": current_streak,
                "max_streak": max_streak,
            }
        )

    print(f"\n{'=' * 90}")
    print("Status key:")
    print("  CONTINUE  = Not enough evidence to decide (keep trading)")
    print("  DEGRADED  = SPRT accepts H1 (lane has no edge) — flag for human review")
    print("  SIGNAL    = SPRT accepts H0 (lane performing as expected)")
    print("  NO DATA   = No trades recorded yet")
    print("Source key:")
    print("  paper_trades      = logged live/shadow execution stream")
    print("  canonical_forward = forward outcome shadow stream from orb_outcomes")
    print(f"{'=' * 90}")

    # Save state
    state_file = STATE_DIR / "sprt_state.json"
    state_file.write_text(json.dumps({"date": str(date.today()), "results": results}, indent=2))

    con.close()


if __name__ == "__main__":
    run_monitor()
