#!/usr/bin/env python3
"""Phase 2.5 — portfolio-wide subset-t + lift-vs-unfiltered sweep.

Applies the MES-composite-audit diagnostic (commit 56fb46e4) portfolio-wide to
all active ``validated_setups`` lanes. Catches the arithmetic-illusion pattern
that Rules 8.1 (extreme-fire) and 8.2 (arithmetic-only) do not catch.

Origin: Phase 2.4 third-pass reframe found a "+0.20 R/trade lift vs unfiltered"
claim on MES composite that collapsed under C1-C12 audit (p=0.668, t=0.43).
Generalized into .claude/rules/backtesting-methodology.md RULE 8.3
ARITHMETIC_LIFT (appended 2026-04-19).

Literature grounding:
  - Bailey-LdP 2014 DSR (docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md)
    — subset-level significance rigor
  - Chordia 2018 (docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md)
    — t ≥ 3.00 with-theory threshold
  - Harvey-Liu 2015 Exhibit 4 (docs/institutional/literature/harvey_liu_2015_backtesting.md)
    — N ≥ 100 deployable
  - Aronson Ch 6 data-mining bias (per .claude/rules/quant-audit-protocol.md)
    — "lift vs noise-baseline" is not evidence without subset significance

Canonical delegations:
  - ``load_active_setups`` + ``compute_mode_a`` from
    ``research.mode_a_revalidation_active_setups``
  - ``filter_signal`` from ``research.filter_utils``
  - ``HOLDOUT_SACRED_FROM`` from ``trading_app.holdout_policy``
  - ``GOLD_DB_PATH`` from ``pipeline.paths``
  - Chordia ``C4_T_WITH_THEORY`` from ``research.mode_a_revalidation_active_setups``

Outputs:
  - ``research/output/phase_2_5_portfolio_subset_t_sweep.csv`` — one row per lane
  - stdout ranked summary with flag breakdown
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dst import SESSION_CATALOG  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.mode_a_revalidation_active_setups import (  # noqa: E402
    C4_T_WITH_THEORY,
    compute_mode_a,
    direction_from_execution_spec,
    load_active_setups,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

# Per integrity-guardian rule #7 (never trust metadata): the orb_label value
# sourced from validated_setups is flowed into an f-string inside fetch_break_universe's
# SQL. Whitelist against the canonical session catalog before trusting the value.
_VALID_SESSION_NAMES: frozenset[str] = frozenset(SESSION_CATALOG.keys())

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"

# Rule 8.3 flag thresholds (literature-derived)
LIFT_THRESHOLD: float = 0.10          # |lift| > 0.10 R/trade triggers Rule 8.3 consideration
SUBSET_T_THRESHOLD: float = C4_T_WITH_THEORY  # Chordia with-theory 3.00


def fetch_break_universe(con: duckdb.DuckDBPyConnection, spec: dict[str, Any]) -> pd.DataFrame:
    """Fetch the full break-direction universe for this lane under Mode A IS.

    Same SQL as ``compute_mode_a`` minus the filter mask — the denominator for
    fire-rate, the baseline for lift-vs-unfiltered.
    """
    sess = spec["orb_label"]
    if sess not in _VALID_SESSION_NAMES:
        raise ValueError(
            f"orb_label={sess!r} not in pipeline.dst.SESSION_CATALOG — "
            f"refusing to inject untrusted session name into SQL"
        )
    direction = direction_from_execution_spec(spec.get("execution_spec"))
    sql = f"""
        SELECT o.trading_day, o.pnl_r, o.outcome, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND d.orb_{sess}_break_dir = ?
          AND o.pnl_r IS NOT NULL
          AND o.trading_day < ?
        ORDER BY o.trading_day
    """
    return con.execute(
        sql,
        [
            spec["instrument"], sess, spec["orb_minutes"],
            spec["entry_model"], spec["confirm_bars"], spec["rr_target"],
            direction, HOLDOUT_SACRED_FROM,
        ],
    ).df()


def subset_t_stat(expr: float | None, sd: float | None, n: int) -> float:
    """Chordia t = ExpR / (sd / sqrt(N)). NaN when inputs missing/degenerate."""
    if n < 2 or expr is None or sd is None or sd == 0:
        return float("nan")
    return float(expr) / (float(sd) / math.sqrt(n))


def classify_lane(
    *,
    n_on: int,
    n_universe: int,
    subset_expr: float | None,
    unfiltered_expr: float | None,
    subset_t: float,
    fire_rate: float,
) -> tuple[list[str], str]:
    """Return (flags, primary_flag).

    Flags are non-exclusive; a lane can carry multiple. Primary_flag is the
    most severe for single-column reporting.
    """
    flags: list[str] = []

    # N_on == N_universe means filter fires on every break day = not a filter.
    # This is Rule 8.1 territory (extreme-fire). We report but don't call it
    # ARITHMETIC_LIFT — that's specifically for filters that DO reduce sample.
    if n_on == n_universe and n_universe > 0:
        flags.append("FILTER_NOT_ACTIVE")
    elif n_universe > 0 and fire_rate >= 0.95:
        flags.append("EXTREME_FIRE_HIGH")
    elif n_universe > 0 and fire_rate <= 0.05:
        flags.append("EXTREME_FIRE_LOW")

    # N_on == 0: filter never fires in Mode A IS. Usually a data gap.
    if n_on == 0:
        flags.append("ZERO_FIRE")

    # Rule 8.3 ARITHMETIC_LIFT — the MES-class pattern.
    # Narrow interpretation per the addendum text: positive lift (filter
    # INFLATES apparent edge) without subset-level significance.
    if (
        subset_expr is not None
        and unfiltered_expr is not None
        and n_on > 0
        and n_on < n_universe
    ):
        lift = subset_expr - unfiltered_expr
        if lift > LIFT_THRESHOLD and not math.isnan(subset_t):
            if abs(subset_t) < SUBSET_T_THRESHOLD:
                flags.append("ARITHMETIC_LIFT")
        # Negative lift — filter REMOVES edge (not Rule 8.3 but worth surfacing).
        elif lift < -LIFT_THRESHOLD and not math.isnan(subset_t):
            flags.append("FILTER_REMOVES_EDGE")

    # Subset t-stat below Chordia with-theory threshold and N deployable.
    if n_on >= 100 and not math.isnan(subset_t) and abs(subset_t) < SUBSET_T_THRESHOLD:
        flags.append("SUBSET_T_BELOW_CHORDIA")

    # N below Harvey-Liu deployable floor.
    if n_on > 0 and n_on < 100:
        flags.append("N_BELOW_DEPLOYABLE")

    # UNEVALUABLE: cannot compute subset_t (e.g., N<2 or sd missing)
    if n_on >= 2 and math.isnan(subset_t):
        flags.append("SUBSET_T_UNEVALUABLE")

    if not flags:
        primary = "PASS"
    elif "ARITHMETIC_LIFT" in flags:
        primary = "ARITHMETIC_LIFT"
    elif "SUBSET_T_BELOW_CHORDIA" in flags:
        primary = "SUBSET_T_BELOW_CHORDIA"
    elif "ZERO_FIRE" in flags:
        primary = "ZERO_FIRE"
    elif "FILTER_NOT_ACTIVE" in flags:
        primary = "FILTER_NOT_ACTIVE"
    elif any(f.startswith("EXTREME_FIRE") for f in flags):
        primary = next(f for f in flags if f.startswith("EXTREME_FIRE"))
    else:
        primary = flags[0]

    return flags, primary


def audit_lane(con: duckdb.DuckDBPyConnection, spec: dict[str, Any]) -> dict[str, Any]:
    """Run the full subset-t + lift-vs-unfiltered audit for one lane."""
    # Subset-filtered Mode A stats via canonical compute_mode_a
    try:
        n_on, expr_on, sharpe_ann, wr_on, _year_break, sd_on = compute_mode_a(con, spec)
        del _year_break  # explicit: year_break dict intentionally discarded
    except Exception as e:  # noqa: BLE001
        return {
            "strategy_id": spec["strategy_id"],
            "instrument": spec["instrument"],
            "orb_label": spec["orb_label"],
            "rr_target": spec["rr_target"],
            "filter_type": spec.get("filter_type"),
            "error": f"compute_mode_a: {e}",
            "flags": "ERROR",
            "primary_flag": "ERROR",
        }

    # Unfiltered baseline on same (inst, session, orb, entry, confirm, rr, dir) universe
    try:
        df_univ = fetch_break_universe(con, spec)
    except Exception as e:  # noqa: BLE001
        return {
            "strategy_id": spec["strategy_id"],
            "instrument": spec["instrument"],
            "orb_label": spec["orb_label"],
            "rr_target": spec["rr_target"],
            "filter_type": spec.get("filter_type"),
            "n_on": n_on,
            "expr_on": expr_on,
            "error": f"fetch_break_universe: {e}",
            "flags": "ERROR",
            "primary_flag": "ERROR",
        }

    n_universe = len(df_univ)
    unfiltered_expr = (
        float(df_univ["pnl_r"].astype(float).mean()) if n_universe else None
    )
    fire_rate = (n_on / n_universe) if n_universe else 0.0

    lift = None
    if expr_on is not None and unfiltered_expr is not None:
        lift = expr_on - unfiltered_expr

    subset_t = subset_t_stat(expr_on, sd_on, n_on)

    flags, primary_flag = classify_lane(
        n_on=n_on,
        n_universe=n_universe,
        subset_expr=expr_on,
        unfiltered_expr=unfiltered_expr,
        subset_t=subset_t,
        fire_rate=fire_rate,
    )

    return {
        "strategy_id": spec["strategy_id"],
        "instrument": spec["instrument"],
        "orb_label": spec["orb_label"],
        "orb_minutes": spec["orb_minutes"],
        "rr_target": spec["rr_target"],
        "entry_model": spec["entry_model"],
        "filter_type": spec.get("filter_type"),
        "direction": direction_from_execution_spec(spec.get("execution_spec")),
        "n_universe": n_universe,
        "n_on": n_on,
        "fire_rate": fire_rate,
        "expr_unfiltered": unfiltered_expr,
        "expr_on": expr_on,
        "lift": lift,
        "subset_t": subset_t if not math.isnan(subset_t) else None,
        "subset_sd": sd_on,
        "wr_on": wr_on,
        "sharpe_ann": sharpe_ann,
        "flags": ";".join(flags) if flags else "PASS",
        "primary_flag": primary_flag,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})

    try:
        setups = load_active_setups(con)
        print(f"Loaded {len(setups)} active validated_setups lanes")

        rows: list[dict[str, Any]] = []
        for i, spec in enumerate(setups, 1):
            print(f"  [{i}/{len(setups)}] {spec['strategy_id']}")
            rows.append(audit_lane(con, spec))
    finally:
        con.close()

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "phase_2_5_portfolio_subset_t_sweep.csv"
    df.to_csv(csv_path, index=False)

    # Summary breakdown by primary flag
    print()
    print("=" * 70)
    print("PORTFOLIO SUBSET-T + LIFT-VS-UNFILTERED SWEEP")
    print(f"Mode A cutoff: trading_day < {HOLDOUT_SACRED_FROM}")
    print(f"Rule 8.3 lift threshold: |lift| > {LIFT_THRESHOLD}")
    print(f"Chordia t threshold: {SUBSET_T_THRESHOLD} (with-theory)")
    print("=" * 70)
    print()

    flag_counts = df["primary_flag"].value_counts().sort_index()
    print("Primary flag breakdown:")
    for flag, count in flag_counts.items():
        print(f"  {flag:<30} {count:>3}")
    print()

    # Top concerning lanes
    concerning = df[df["primary_flag"].isin(["ARITHMETIC_LIFT", "SUBSET_T_BELOW_CHORDIA", "ZERO_FIRE"])].copy()
    if len(concerning):
        concerning = concerning.sort_values(
            by=["primary_flag", "n_on"],
            key=lambda s: s if s.name == "primary_flag" else -s,
        )
        print(f"Concerning lanes ({len(concerning)}):")
        print(concerning[[
            "strategy_id", "n_on", "n_universe", "fire_rate",
            "expr_unfiltered", "expr_on", "lift", "subset_t", "primary_flag"
        ]].to_string(index=False))
    print()

    try:
        print(f"Written: {csv_path.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"Written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
