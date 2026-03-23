"""Exhaustive ML sweep: RR2.0, RR1.5, RR1.0 + per-aperture at RR2.0.

Run unattended. Logs to logs/ml_sweep_*.log. Prints summary at end.
"""

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "ml_sweep_master.log"),
    ],
)
log = logging.getLogger(__name__)


def run_ml(label: str, rr: float, per_aperture: bool = False) -> tuple[str, int, float]:
    """Run one ML training pass. Returns (label, returncode, elapsed_seconds)."""
    logfile = LOG_DIR / f"ml_sweep_{label}.log"
    cmd = [
        sys.executable,
        "-m",
        "trading_app.ml.meta_label",
        "--instrument",
        "MNQ",
        "--single-config",
        "--rr-target",
        str(rr),
        "--config-selection",
        "max_samples",
        "--skip-filter",
    ]
    if per_aperture:
        cmd.append("--per-aperture")

    log.info(f"{'=' * 60}")
    log.info(f"  STARTING: {label} (RR={rr}, per_aperture={per_aperture})")
    log.info(f"  Log: {logfile}")
    log.info(f"{'=' * 60}")

    start = time.time()
    with open(logfile, "w") as f:
        result = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parent.parent.parent)
        )
    elapsed = time.time() - start

    log.info(f"  DONE: {label} — exit={result.returncode}, {elapsed:.0f}s")
    return label, result.returncode, elapsed


def run_univariate_audit() -> None:
    """Quick univariate feature audit at each RR level."""
    logfile = LOG_DIR / "ml_sweep_univariate.log"
    log.info("Running univariate feature audit...")

    script = """
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

for rr in [1.0, 1.5, 2.0]:
    print(f"\\n{'='*60}")
    print(f"  UNIVARIATE AUDIT: MNQ E2 RR{rr}")
    print(f"{'='*60}")

    # ATR percentile
    print(f"\\n  ATR20_pct quartile:")
    print(con.sql(f'''
    SELECT
      CASE WHEN d.atr_20_pct < 25 THEN 'Q1_low'
           WHEN d.atr_20_pct < 50 THEN 'Q2'
           WHEN d.atr_20_pct < 75 THEN 'Q3'
           ELSE 'Q4_high' END as q,
      COUNT(*) as n, ROUND(AVG(o.pnl_r), 4) as expr,
      ROUND(COUNT(CASE WHEN o.pnl_r > 0 THEN 1 END)*100.0/COUNT(*), 1) as wr
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND d.orb_minutes=5
    WHERE o.symbol='MNQ' AND o.entry_model='E2' AND o.rr_target={rr}
      AND o.confirm_bars=1 AND o.orb_minutes=5 AND d.atr_20_pct IS NOT NULL
    GROUP BY q ORDER BY q
    ''').fetchdf().to_string(index=False))

    # ATR velocity regime
    print(f"\\n  ATR velocity regime:")
    print(con.sql(f'''
    SELECT d.atr_vel_regime as regime, COUNT(*) as n,
      ROUND(AVG(o.pnl_r), 4) as expr,
      ROUND(COUNT(CASE WHEN o.pnl_r > 0 THEN 1 END)*100.0/COUNT(*), 1) as wr
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND d.orb_minutes=5
    WHERE o.symbol='MNQ' AND o.entry_model='E2' AND o.rr_target={rr}
      AND o.confirm_bars=1 AND o.orb_minutes=5 AND d.atr_vel_regime IS NOT NULL
    GROUP BY d.atr_vel_regime ORDER BY expr DESC
    ''').fetchdf().to_string(index=False))

    # Gap type
    print(f"\\n  Gap type:")
    print(con.sql(f'''
    SELECT d.gap_type, COUNT(*) as n, ROUND(AVG(o.pnl_r), 4) as expr,
      ROUND(COUNT(CASE WHEN o.pnl_r > 0 THEN 1 END)*100.0/COUNT(*), 1) as wr
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND d.orb_minutes=5
    WHERE o.symbol='MNQ' AND o.entry_model='E2' AND o.rr_target={rr}
      AND o.confirm_bars=1 AND o.orb_minutes=5 AND d.gap_type IS NOT NULL
    GROUP BY d.gap_type ORDER BY expr DESC
    ''').fetchdf().to_string(index=False))

    # Per-session top 5 by ExpR spread (best quartile - worst quartile for ORB size)
    print(f"\\n  ORB size signal per session (Q2 vs Q4):")
    _sessions = con.sql("SELECT DISTINCT orb_label FROM orb_outcomes WHERE symbol='MNQ' ORDER BY orb_label").fetchdf()['orb_label'].tolist()
    for sess in _sessions:
        col = f'orb_{sess}_size'
        try:
            result = con.sql(f'''
            WITH sized AS (
              SELECT o.pnl_r, d.{col},
                NTILE(4) OVER (ORDER BY d.{col}) as q
              FROM orb_outcomes o
              JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND d.orb_minutes=5
              WHERE o.symbol='MNQ' AND o.entry_model='E2' AND o.rr_target={rr}
                AND o.confirm_bars=1 AND o.orb_minutes=5
                AND o.orb_label='{sess}' AND d.{col} IS NOT NULL
            )
            SELECT
              ROUND(AVG(CASE WHEN q=2 THEN pnl_r END), 4) as q2_expr,
              ROUND(AVG(CASE WHEN q=4 THEN pnl_r END), 4) as q4_expr,
              ROUND(AVG(CASE WHEN q=2 THEN pnl_r END) - AVG(CASE WHEN q=4 THEN pnl_r END), 4) as spread
            FROM sized
            ''').fetchone()
            if result:
                print(f"    {sess:<20} Q2={result[0]:+.4f}  Q4={result[1]:+.4f}  spread={result[2]:+.4f}")
        except Exception:
            print(f"    {sess:<20} (no data)")

con.close()
print("\\nUnivariate audit complete.")
"""

    with open(logfile, "w") as f:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
    if proc.returncode != 0:
        log.warning(f"  Univariate audit FAILED (exit code {proc.returncode}) — see {logfile}")
    else:
        log.info(f"  Univariate audit saved to {logfile}")


def print_summary(results: list[tuple[str, int, float]]) -> None:
    """Parse each log file for key results and print summary."""
    log.info(f"\n{'=' * 60}")
    log.info("  SWEEP COMPLETE — SUMMARY")
    log.info(f"{'=' * 60}")

    for label, rc, elapsed in results:
        logfile = LOG_DIR / f"ml_sweep_{label}.log"
        log.info(f"\n  --- {label} (exit={rc}, {elapsed:.0f}s) ---")

        if not logfile.exists():
            log.info("    (no log file)")
            continue

        text = logfile.read_text()
        # Extract SUMMARY section
        for line in text.split("\n"):
            if any(
                kw in line
                for kw in ["SUMMARY:", "Honest delta", "Full delta", "Selection uplift", ">> ML t=", "NO_MODEL"]
            ):
                log.info(f"    {line.strip()}")

    # Also print the univariate audit path
    uni_log = LOG_DIR / "ml_sweep_univariate.log"
    if uni_log.exists():
        log.info(f"\n  Univariate audit: {uni_log}")

    log.info(f"\n  All logs in: {LOG_DIR}")
    log.info(f"  Master log: {LOG_DIR / 'ml_sweep_master.log'}")
    log.info(f"\n  NEXT: Review results, then implement bootstrap for any survivors.")
    log.info(f"{'=' * 60}")


def main():
    started = datetime.now()
    log.info(f"ML Exhaustive Sweep — started {started.isoformat()}")
    log.info(f"Instrument: MNQ, Entry: E2, Aperture: O5")
    log.info(f"RR targets: 2.0, 1.5, 1.0 (+ per-aperture at RR2.0)")
    log.info(f"Mode: bypass_validated=False, skip_filter=True, config_selection=max_samples")

    # Phase 0.5: Univariate audit
    run_univariate_audit()

    # Phase 1: Multi-RR sweep (RR2.0 first — strongest univariate signal)
    results = []
    results.append(run_ml("rr20_flat", rr=2.0))
    results.append(run_ml("rr15_flat", rr=1.5))
    results.append(run_ml("rr10_flat", rr=1.0))

    # Phase 1b: Per-aperture at RR2.0 (strongest signal)
    results.append(run_ml("rr20_aperture", rr=2.0, per_aperture=True))

    # Summary
    elapsed_total = (datetime.now() - started).total_seconds()
    log.info(f"\nTotal elapsed: {elapsed_total:.0f}s ({elapsed_total / 60:.1f}m)")
    print_summary(results)


if __name__ == "__main__":
    main()
