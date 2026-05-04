"""MinBTL retro-report -- selection-bias multiplier for a discovery run.

Computes Bailey-Lopez de Prado 2013 Theorem 1 strict bound for a given
instrument's clean-data horizon and compares against the realized trial count
of a discovery run, reporting the selection-bias multiplier vs both the strict
Bailey bound (E[max_N]=1.0) and the locked operational cap (N <= 300, see
``docs/institutional/pre_registered_criteria.md`` Criterion 2).

Read-only. No DB writes. Pure formula + a single SELECT for trial count.

Authority chain:
- ``docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md``
  Theorem 1 (the formula).
- ``docs/institutional/pre_registered_criteria.md`` Criterion 2 (worked
  bounds: MNQ/MES N<=28 strict E=1.0, MGC N<=4 strict at 2.7yr).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Clean-data horizons from pre_registered_criteria.md Criterion 2 (Amendment 2.8).
CLEAN_YEARS_BY_INSTRUMENT: dict[str, float] = {
    "MNQ": 6.65,
    "MES": 6.65,
    "MGC": 2.70,
}

LOCKED_OPERATIONAL_CAP = 300


def strict_bailey_n(clean_years: float, e_max: float = 1.0) -> int:
    """Strict Bailey Theorem 1 bound: 2*Ln[N] / E[max_N]^2 <= clean_years."""
    if clean_years <= 0:
        return 0
    log_n_max = (clean_years * (e_max**2)) / 2.0
    return int(math.floor(math.exp(log_n_max)))


def query_n_trials(con, instrument: str) -> int:
    row = con.execute(
        "SELECT COUNT(*) FROM experimental_strategies WHERE instrument = ?",
        [instrument],
    ).fetchone()
    return int(row[0]) if row else 0


def build_report(instrument: str, n_trials: int, discovery_date: str | None) -> str:
    clean_years = CLEAN_YEARS_BY_INSTRUMENT.get(instrument)
    if clean_years is None:
        raise SystemExit(
            f"Unknown instrument {instrument!r}. Known: "
            f"{sorted(CLEAN_YEARS_BY_INSTRUMENT)}. Add to "
            "CLEAN_YEARS_BY_INSTRUMENT after amending Criterion 2."
        )

    n_strict_e10 = strict_bailey_n(clean_years, e_max=1.0)
    n_strict_e12 = strict_bailey_n(clean_years, e_max=1.2)
    n_strict_e15 = strict_bailey_n(clean_years, e_max=1.5)

    mult_strict = n_trials / n_strict_e10 if n_strict_e10 > 0 else float("inf")
    mult_locked = n_trials / LOCKED_OPERATIONAL_CAP if LOCKED_OPERATIONAL_CAP > 0 else float("inf")

    lines = [
        "MinBTL retro-report",
        "===================",
        f"Instrument:           {instrument}",
        f"Discovery date:       {discovery_date or '(unspecified -- current pool)'}",
        f"Clean horizon (yr):   {clean_years:.2f}",
        f"Realized N (trials):  {n_trials}",
        "",
        "Bailey 2013 Theorem 1 strict bounds (2*Ln[N]/E^2 <= horizon):",
        f"  E[max_N]=1.0  ->  N <= {n_strict_e10}",
        f"  E[max_N]=1.2  ->  N <= {n_strict_e12}",
        f"  E[max_N]=1.5  ->  N <= {n_strict_e15}",
        "",
        f"Locked operational cap (Criterion 2, v1 lock):  N <= {LOCKED_OPERATIONAL_CAP}",
        "",
        "Selection-bias multipliers vs realized N:",
        f"  vs strict E=1.0      -> {mult_strict:>10.1f}x  (institutional max-rigor)",
        f"  vs locked op cap     -> {mult_locked:>10.1f}x  (operational ceiling)",
        "",
    ]

    verdict = []
    if n_trials > LOCKED_OPERATIONAL_CAP:
        verdict.append(
            f"VIOLATION: realized N ({n_trials}) exceeds locked operational "
            f"cap ({LOCKED_OPERATIONAL_CAP}). Discovery is doctrinally banned."
        )
    if n_trials > n_strict_e10:
        verdict.append(
            f"NOISE-FLOOR: realized N exceeds strict Bailey E=1.0 bound "
            f"({n_strict_e10}). Survivors require explicit noise-floor disclosure "
            "per Criterion 2."
        )
    if not verdict:
        verdict.append("OK: realized N within both strict and operational bounds.")

    lines.append("Verdict:")
    for v in verdict:
        lines.append(f"  - {v}")
    lines.append("")
    lines.append(
        "Authority: pre_registered_criteria.md Criterion 2; "
        "literature/bailey_et_al_2013_pseudo_mathematics.md Theorem 1."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bailey 2013 Theorem 1 retro-report for a discovery run.")
    parser.add_argument("--instrument", required=True, help="MNQ, MES, or MGC")
    parser.add_argument(
        "--discovery-date",
        default=None,
        help="ISO date (e.g. 2026-05-04) -- informational stamp.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Realized trial count. If omitted, queried from gold.db.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to gold.db (default: pipeline.paths.GOLD_DB_PATH).",
    )
    args = parser.parse_args()

    if args.n_trials is None:
        import duckdb

        from pipeline.paths import GOLD_DB_PATH

        db_path = Path(args.db) if args.db else GOLD_DB_PATH
        if not Path(db_path).exists():
            raise SystemExit(f"gold.db not found at {db_path}")
        with duckdb.connect(str(db_path), read_only=True) as con:
            n_trials = query_n_trials(con, args.instrument)
    else:
        n_trials = args.n_trials

    print(build_report(args.instrument, n_trials, args.discovery_date))
    return 0


if __name__ == "__main__":
    sys.exit(main())
