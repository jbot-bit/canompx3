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

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# SPRT boundaries (Wald)
ALPHA = 0.10
BETA = 0.20
A = math.log((1 - BETA) / ALPHA)  # 2.079 — reject H0 (accept DEGRADED) when LR < -A
B = math.log((1 - ALPHA) / BETA)  # 1.504 — accept H0 (SIGNAL) when LR > B

# Lane parameters (backtest ExpR and std from validated_setups)
LANES = {
    "NYSE_CLOSE": {"mu0": 0.2078, "sigma": 0.891, "label": "L1 NYSE_CLOSE VOL_RV12_N20"},
    "SINGAPORE_OPEN": {"mu0": 0.1587, "sigma": 1.844, "label": "L2 SINGAPORE_OPEN ORB_G8 RR4.0"},
    "COMEX_SETTLE": {"mu0": 0.1300, "sigma": 0.864, "label": "L3 COMEX_SETTLE ORB_G8"},
    "NYSE_OPEN": {"mu0": 0.0933, "sigma": 0.956, "label": "L4 NYSE_OPEN X_MES_ATR60"},
    "TOKYO_OPEN": {"mu0": 0.2832, "sigma": 1.42, "label": "L5 MGC TOKYO_OPEN (shadow)"},
}


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
    print(f"\n{'Lane':<40} {'N':>4} {'SPRT':>8} {'Lower':>8} {'Upper':>8} {'Status':<12} {'Streak':>6} {'MaxStr':>6}")
    print("-" * 110)

    results = []
    for session, params in LANES.items():
        rows = con.execute(
            """SELECT pnl_r FROM paper_trades
               WHERE orb_label = ? AND pnl_r IS NOT NULL
               ORDER BY trading_day""",
            [session],
        ).fetchall()

        trades = [r[0] for r in rows]
        n = len(trades)

        if n == 0:
            print(f"{params['label']:<40} {0:>4} {'N/A':>8} {-A:>8.3f} {B:>8.3f} {'NO DATA':<12} {0:>6} {0:>6}")
            results.append({"session": session, "n": 0, "sprt": 0, "status": "NO_DATA"})
            continue

        lr, status = compute_sprt(trades, params["mu0"], params["sigma"])
        current_streak, max_streak = compute_streak(trades)

        print(
            f"{params['label']:<40} {n:>4} {lr:>+8.3f} {-A:>8.3f} {B:>8.3f} {status:<12} "
            f"{current_streak:>6} {max_streak:>6}"
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
                "session": session,
                "n": n,
                "sprt": round(lr, 4),
                "status": status,
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
    print(f"{'=' * 90}")

    # Save state
    state_file = STATE_DIR / "sprt_state.json"
    state_file.write_text(json.dumps({"date": str(date.today()), "results": results}, indent=2))

    con.close()


if __name__ == "__main__":
    run_monitor()
