"""OOS reconciliation diligence runner for MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100.

Diligence audit, not new discovery. Reconciles three pre-existing OOS values for
the same nominal lane:

- A: ``validated_setups.oos_exp_r`` = +0.2029 (promoted 2026-04-11, LEGACY)
- B: ``docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md``
     OOS ExpR = +0.1658 (canonical strict-unlock 2026-05-02)
- C: ad-hoc raw query +0.069 (rejected — bypassed canonical filter delegation)

Computes a fourth canonical value via canonical machinery and emits a diff table.

Canonical machinery delegated (no re-encoding per
`.claude/rules/integrity-guardian.md` Rule 4):

- Holdout cut: ``trading_app.holdout_policy.HOLDOUT_SACRED_FROM``
- IS lower bound: ``trading_app.config.WF_START_OVERRIDE['MNQ']``
- Cohort load: ``research.chordia_strict_unlock_v1._load_universe``
- Filter fire: ``research.filter_utils.filter_signal``
- Cell shape: ``research.chordia_strict_unlock_v1.Cell``
- Stats: ``research.chordia_strict_unlock_v1._evaluate_split``

Read-only against ``gold.db``. No writes to ``validated_setups``,
``experimental_strategies``, ``chordia_audit_log.yaml``, or
``lane_allocation.json``.

Pre-reg:
    docs/audit/hypotheses/2026-05-05-oos-reconcile-ovnrng100.yaml

Result:
    docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.md (written by hand
    after running this script, citing this script's stdout).

Run:
    python research/oos_reconcile_ovnrng100.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH
from research.chordia_strict_unlock_v1 import (
    Cell,
    _evaluate_split,
    _load_universe,
)
from trading_app.config import WF_START_OVERRIDE
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM


LANE = Cell(
    hypothesis_file=Path(
        "docs/audit/hypotheses/2026-05-05-oos-reconcile-ovnrng100.yaml"
    ),
    hypothesis_name="oos_reconcile_ovnrng100_rr15_v1",
    strategy_id="MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
    instrument="MNQ",
    orb_label="COMEX_SETTLE",
    orb_minutes=5,
    entry_model="E2",
    confirm_bars=1,
    rr_target=1.5,
    filter_key="OVNRNG_100",
    has_theory=False,
    theory_mode="CLASS_GROUNDED_ONLY",
    result_md=Path("docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.md"),
    result_csv=Path("docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.csv"),
)

# Pre-existing OOS values for diff. Sourced from authoritative locations only.
PRIOR_VALIDATED_SETUPS_OOS_EXPR = 0.2029  # validated_setups.oos_exp_r
PRIOR_STRICT_UNLOCK_OOS_EXPR = 0.1658  # csv at docs/audit/results/2026-05-02-...
PRIOR_VALIDATED_SETUPS_IS_EXPR = 0.2151  # validated_setups.expectancy_r
PRIOR_STRICT_UNLOCK_N_IS = 522
PRIOR_STRICT_UNLOCK_N_OOS = 66

# Acceptance tolerances per pre-reg kill_criteria.
TOL_EXPR = 0.001
TOL_N = 2


def _query_validated_setups_row(con: duckdb.DuckDBPyConnection) -> dict:
    """Read the one validated_setups row this audit reconciles against.

    Read-only. No writes. The row's oos_exp_r is the primary diligence target
    (Number A in the pre-reg).
    """
    row = con.execute(
        """
        SELECT
            strategy_id,
            sample_size,
            expectancy_r,
            oos_exp_r,
            promoted_at,
            promotion_provenance,
            n_trials_at_discovery,
            dsr_score
        FROM validated_setups
        WHERE strategy_id = ?
        """,
        [LANE.strategy_id],
    ).fetchone()
    if row is None:
        raise SystemExit(
            f"REFUSE: validated_setups row {LANE.strategy_id!r} not found. "
            "This runner audits a deployed lane; without the row the diff is undefined."
        )
    cols = [c[0] for c in con.description]
    return dict(zip(cols, row))


def _format_split(label: str, result: dict) -> str:
    return (
        f"  {label:>4s}: N_universe={result['n_universe']:>5d} "
        f"N_fired={result['n_fired']:>4d} "
        f"fire_rate={result['fire_rate'] * 100:>5.2f}% "
        f"ExpR={result['expr']:+.4f} "
        f"Sharpe={result['sharpe']:+.4f} "
        f"t={result['t']:+.3f} "
        f"p_two={result['p_two_sided']:.5f} "
        f"scratch={result['scratch_n']} "
        f"null_non_scratch={result['null_non_scratch_n']}"
    )


def _check_acceptance(
    runner_is: dict, runner_oos: dict, vs_row: dict
) -> list[str]:
    """Return list of failed acceptance criteria. Empty list means PASS."""
    fails: list[str] = []

    # Acceptance 1: N_IS == strict-unlock N_IS within TOL_N
    delta_n_is = abs(runner_is["n_fired"] - PRIOR_STRICT_UNLOCK_N_IS)
    if delta_n_is > TOL_N:
        fails.append(
            f"N_IS_fired={runner_is['n_fired']} differs from strict-unlock CSV "
            f"{PRIOR_STRICT_UNLOCK_N_IS} by {delta_n_is} > tolerance {TOL_N}"
        )

    # Acceptance 2: N_OOS == strict-unlock N_OOS within TOL_N
    delta_n_oos = abs(runner_oos["n_fired"] - PRIOR_STRICT_UNLOCK_N_OOS)
    if delta_n_oos > TOL_N:
        fails.append(
            f"N_OOS_fired={runner_oos['n_fired']} differs from strict-unlock CSV "
            f"{PRIOR_STRICT_UNLOCK_N_OOS} by {delta_n_oos} > tolerance {TOL_N}"
        )

    # Acceptance 3: OOS_ExpR == strict-unlock OOS_ExpR within TOL_EXPR
    delta_oos_expr = abs(runner_oos["expr"] - PRIOR_STRICT_UNLOCK_OOS_EXPR)
    if delta_oos_expr > TOL_EXPR:
        fails.append(
            f"OOS_ExpR={runner_oos['expr']:.4f} differs from strict-unlock CSV "
            f"{PRIOR_STRICT_UNLOCK_OOS_EXPR:.4f} by {delta_oos_expr:.4f} > tolerance {TOL_EXPR}"
        )

    # Acceptance 4: IS_ExpR == validated_setups.expectancy_r within TOL_EXPR
    delta_is_expr = abs(runner_is["expr"] - PRIOR_VALIDATED_SETUPS_IS_EXPR)
    if delta_is_expr > TOL_EXPR:
        fails.append(
            f"IS_ExpR={runner_is['expr']:.4f} differs from validated_setups.expectancy_r "
            f"{PRIOR_VALIDATED_SETUPS_IS_EXPR:.4f} by {delta_is_expr:.4f} > tolerance {TOL_EXPR}"
        )

    return fails


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypothesis-file",
        default="docs/audit/hypotheses/2026-05-05-oos-reconcile-ovnrng100.yaml",
        help="Pre-reg path (informational; runner does not parse the yaml).",
    )
    parser.parse_args(argv)

    print("=" * 78)
    print("OOS RECONCILIATION RUNNER — MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100")
    print("=" * 78)
    print()
    print(f"Canonical DB: {GOLD_DB_PATH}")
    print(f"Pre-reg:      docs/audit/hypotheses/2026-05-05-oos-reconcile-ovnrng100.yaml")
    print()
    print("Canonical constants:")
    print(f"  HOLDOUT_SACRED_FROM         = {HOLDOUT_SACRED_FROM!r}")
    print(f"  WF_START_OVERRIDE['MNQ']    = {WF_START_OVERRIDE.get('MNQ')!r}")
    print()
    print("Lane:")
    print(f"  strategy_id   = {LANE.strategy_id}")
    print(f"  instrument    = {LANE.instrument}")
    print(f"  orb_label     = {LANE.orb_label}")
    print(f"  orb_minutes   = {LANE.orb_minutes}")
    print(f"  entry_model   = {LANE.entry_model}")
    print(f"  confirm_bars  = {LANE.confirm_bars}")
    print(f"  rr_target     = {LANE.rr_target}")
    print(f"  filter_key    = {LANE.filter_key}")
    print()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # Pre-existing values for the diff target.
    vs_row = _query_validated_setups_row(con)
    print("validated_setups row:")
    for k, v in vs_row.items():
        print(f"  {k} = {v}")
    print()

    # Canonical IS recompute via _load_universe + _evaluate_split.
    is_universe = _load_universe(con, LANE, is_only=True)
    is_result, _is_mask, _is_fired = _evaluate_split(
        is_universe, LANE, sample_label="IS"
    )

    # Canonical OOS recompute via the same helpers; is_only=False = trading_day >= HOLDOUT_SACRED_FROM.
    oos_universe = _load_universe(con, LANE, is_only=False)
    oos_result, _oos_mask, _oos_fired = _evaluate_split(
        oos_universe, LANE, sample_label="OOS"
    )

    print("Runner output (canonical recompute):")
    print(_format_split("IS", is_result))
    print(_format_split("OOS", oos_result))
    print()

    # Three-number diff table.
    print("=" * 78)
    print("RECONCILIATION DIFF TABLE")
    print("=" * 78)
    print()
    print("OOS_ExpR comparison:")
    print(
        f"  A  validated_setups.oos_exp_r       = {PRIOR_VALIDATED_SETUPS_OOS_EXPR:+.4f}  "
        "(promoted 2026-04-11, promotion_provenance=LEGACY)"
    )
    print(
        f"  B  strict-unlock CSV OOS ExpR        = {PRIOR_STRICT_UNLOCK_OOS_EXPR:+.4f}  "
        "(docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md)"
    )
    print(
        f"  R  runner canonical recompute OOS    = {oos_result['expr']:+.4f}  "
        "(this script, canonical machinery)"
    )
    print()
    print(f"  R - A delta = {oos_result['expr'] - PRIOR_VALIDATED_SETUPS_OOS_EXPR:+.4f}")
    print(f"  R - B delta = {oos_result['expr'] - PRIOR_STRICT_UNLOCK_OOS_EXPR:+.4f}")
    print()
    print("IS_ExpR comparison:")
    print(
        f"  A  validated_setups.expectancy_r     = {PRIOR_VALIDATED_SETUPS_IS_EXPR:+.4f}"
    )
    print(f"  R  runner canonical recompute IS     = {is_result['expr']:+.4f}")
    print(f"  R - A delta = {is_result['expr'] - PRIOR_VALIDATED_SETUPS_IS_EXPR:+.4f}")
    print()
    print("N comparison (post-filter fired):")
    print(
        f"  strict-unlock CSV N_IS  = {PRIOR_STRICT_UNLOCK_N_IS}   "
        f"runner N_IS  = {is_result['n_fired']}   "
        f"delta = {is_result['n_fired'] - PRIOR_STRICT_UNLOCK_N_IS:+d}"
    )
    print(
        f"  strict-unlock CSV N_OOS = {PRIOR_STRICT_UNLOCK_N_OOS}    "
        f"runner N_OOS = {oos_result['n_fired']}    "
        f"delta = {oos_result['n_fired'] - PRIOR_STRICT_UNLOCK_N_OOS:+d}"
    )
    print()

    # Acceptance gate.
    print("=" * 78)
    print("ACCEPTANCE CRITERIA (per pre-reg kill_criteria)")
    print("=" * 78)
    fails = _check_acceptance(is_result, oos_result, vs_row)
    if not fails:
        print("PASS — runner reproduces strict-unlock CSV within tolerance.")
        print()
        print("Diligence verdict: RECONCILED. The validated_setups.oos_exp_r=+0.2029")
        print("LEGACY value diverges from the canonical recompute; the strict-unlock")
        print("CSV value (+0.1658) is the canonical OOS for this lane.")
        return_code = 0
    else:
        print("FAIL — runner output diverges from prior canonical values:")
        for fail in fails:
            print(f"  - {fail}")
        print()
        print(
            "Diligence verdict: NOT RECONCILED. Investigate root cause before any "
            "allocator decision on the deployed lane."
        )
        return_code = 1

    print()
    return return_code


if __name__ == "__main__":
    sys.exit(main())
