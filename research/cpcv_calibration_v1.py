"""CPCV calibration — H1/H2/H3 per 2026-04-21-cpcv-infrastructure-v1.yaml.

Runs the three pre-registered calibration hypotheses against synthetic
known-state data to validate the CPCV implementation in
``trading_app.cpcv``.  Writes a postmortem markdown at
``docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1-postmortem.md``
recording numeric pass/fail against the pre-registered thresholds.

Usage:
    python research/cpcv_calibration_v1.py

This is a one-off calibration script, not a production path.  It does
NOT consume any real orb_outcomes data and does NOT write to gold.db.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Allow running from repo root without PYTHONPATH setup.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading_app.cpcv import cpcv_evaluate  # noqa: E402

# Pre-registered parameters (from pre-reg § axes)
N_SEEDS = 10
N_TRADES = 2000
N_SPLITS = 6
N_TEST_SPLITS = 2
DEFAULT_EMBARGO = 5
ALPHA = 0.05


def _gen_gaussian(rng: random.Random, n: int, mean: float, sd: float) -> list[float]:
    return [rng.gauss(mean, sd) for _ in range(n)]


def _gen_ar1(rng: random.Random, n: int, rho: float, sd: float) -> list[float]:
    """AR(1) series with mean 0, persistence `rho`, innovation sd matched
    so total sample sd ≈ `sd` (unit variance when sd=1).
    """
    innov_sd = sd * (1.0 - rho ** 2) ** 0.5
    out: list[float] = [rng.gauss(0.0, sd)]  # stationary start
    for _ in range(1, n):
        out.append(rho * out[-1] + rng.gauss(0.0, innov_sd))
    return out


def run_h1_known_null() -> dict:
    """H1: CPCV pass rate on known-null Gaussian matches nominal alpha.

    Pass threshold: mean reject_fraction across seeds in [0.025, 0.10].
    Kill threshold: > 0.15 or < 0.01.
    """
    reject_fractions: list[float] = []
    for seed in range(N_SEEDS):
        rng = random.Random(1000 + seed)
        returns = _gen_gaussian(rng, N_TRADES, mean=0.0, sd=1.0)
        r = cpcv_evaluate(
            returns,
            n_splits=N_SPLITS,
            n_test_splits=N_TEST_SPLITS,
            embargo_trades=DEFAULT_EMBARGO,
            alpha=ALPHA,
        )
        reject_fractions.append(r["reject_fraction"])
    mean_rf = sum(reject_fractions) / len(reject_fractions)
    pass_band = (0.025, 0.10)
    kill = mean_rf > 0.15 or mean_rf < 0.01
    passed = pass_band[0] <= mean_rf <= pass_band[1]
    return {
        "hypothesis_id": "H1",
        "name": "known-null reject rate near alpha",
        "mean_reject_fraction": mean_rf,
        "per_seed_reject_fractions": reject_fractions,
        "pass_band": pass_band,
        "passed": passed,
        "killed": kill,
        "verdict": "PASS" if passed else ("KILL" if kill else "MARGINAL"),
    }


def _theoretical_power(mean: float, sd: float, n: int, alpha: float = 0.05) -> float:
    """Analytical power at the fold level using non-central t.

    Matches ``trading_app.strategy_validator._estimate_oos_power``.
    """
    import math

    from scipy import stats

    if n < 2 or sd <= 0:
        return 0.0
    df = n - 1
    ncp = mean * (n ** 0.5) / sd
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    pu = float(stats.nct.sf(t_crit, df, ncp))
    pl = float(stats.nct.cdf(-t_crit, df, ncp))
    if math.isnan(pu):
        pu = 0.0
    if math.isnan(pl):
        pl = 0.0
    return max(0.0, min(1.0, pu + pl))


def run_h2_known_edge() -> dict:
    """H2: CPCV recovers +0.15 R known edge at near-theoretical power.

    Pass threshold: observed mean fraction within ±0.10 of theoretical
    power at the equivalent per-fold N.
    Kill threshold: deviates from theoretical by > 0.20.
    """
    effect = 0.15
    reject_fractions: list[float] = []
    fold_ns: list[int] = []
    for seed in range(N_SEEDS):
        rng = random.Random(2000 + seed)
        returns = _gen_gaussian(rng, N_TRADES, mean=effect, sd=1.0)
        r = cpcv_evaluate(
            returns,
            n_splits=N_SPLITS,
            n_test_splits=N_TEST_SPLITS,
            embargo_trades=DEFAULT_EMBARGO,
            alpha=ALPHA,
        )
        reject_fractions.append(r["reject_fraction"])
        # Per-fold N is the same across folds within a seed (up to ±1
        # from remainder chunks).  Use the first fold's n_test as the
        # representative value.
        fold_ns.append(r["folds"][0]["n_test"])
    mean_rf = sum(reject_fractions) / len(reject_fractions)
    mean_fold_n = sum(fold_ns) / len(fold_ns)
    theoretical = _theoretical_power(mean=effect, sd=1.0, n=int(mean_fold_n), alpha=ALPHA)
    gap = abs(mean_rf - theoretical)
    passed = gap <= 0.10
    killed = gap > 0.20
    return {
        "hypothesis_id": "H2",
        "name": "known-edge power matches theoretical",
        "effect_size": effect,
        "mean_reject_fraction": mean_rf,
        "mean_per_fold_n": mean_fold_n,
        "theoretical_power": theoretical,
        "abs_gap": gap,
        "per_seed_reject_fractions": reject_fractions,
        "passed": passed,
        "killed": killed,
        "verdict": "PASS" if passed else ("KILL" if killed else "MARGINAL"),
    }


def run_h3_embargo_sensitivity() -> dict:
    """H3: Zero-embargo inflates reject rate under AR(1) serial correlation.

    Pass threshold:
      (a) embargo=0 reject rate > embargo=5 by >= 0.03
      (b) embargo ∈ {5, 10, 20} all within [0.025, 0.10]
    Kill threshold: neither (a) nor (b) satisfied, OR embargo=0 <= embargo=5.
    """
    rho = 0.15
    embargo_values = [0, 5, 10, 20]
    per_embargo: dict[int, list[float]] = {e: [] for e in embargo_values}
    for seed in range(N_SEEDS):
        rng = random.Random(3000 + seed)
        returns = _gen_ar1(rng, N_TRADES, rho=rho, sd=1.0)
        for e in embargo_values:
            # Re-seed RNG NOT needed — returns are fixed for this seed;
            # only the embargo parameter varies. That IS the sensitivity test.
            r = cpcv_evaluate(
                returns,
                n_splits=N_SPLITS,
                n_test_splits=N_TEST_SPLITS,
                embargo_trades=e,
                alpha=ALPHA,
            )
            per_embargo[e].append(r["reject_fraction"])
    mean_per_embargo = {e: sum(v) / len(v) for e, v in per_embargo.items()}
    # Check (a): embargo=0 > embargo=5 by >= 0.03
    gap_0_vs_5 = mean_per_embargo[0] - mean_per_embargo[5]
    a_passed = gap_0_vs_5 >= 0.03
    # Check (b): embargo in {5, 10, 20} within [0.025, 0.10]
    b_passed = all(0.025 <= mean_per_embargo[e] <= 0.10 for e in (5, 10, 20))
    # Deterministic production embargo choice: smallest embargo passing (b)
    # with margin >= 0.02 from band edges.
    production_embargo: int | None = None
    for e in (5, 10, 20):
        mrf = mean_per_embargo[e]
        if 0.025 + 0.02 <= mrf <= 0.10 - 0.02:
            production_embargo = e
            break
    passed = a_passed and b_passed and production_embargo is not None
    # Kill: neither (a) nor (b) holds, OR embargo=0 <= embargo=5 (purge already sufficient).
    killed = (not a_passed) and (not b_passed)
    if mean_per_embargo[0] <= mean_per_embargo[5]:
        killed = True
    return {
        "hypothesis_id": "H3",
        "name": "embargo sensitivity on AR(1) serial correlation",
        "rho": rho,
        "mean_reject_fraction_by_embargo": mean_per_embargo,
        "gap_0_vs_5": gap_0_vs_5,
        "check_a_passed": a_passed,
        "check_b_passed": b_passed,
        "production_embargo": production_embargo,
        "passed": passed,
        "killed": killed,
        "verdict": "PASS" if passed else ("KILL" if killed else "MARGINAL"),
    }


def _fmt_float(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


def write_postmortem(h1: dict, h2: dict, h3: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# CPCV Infrastructure Calibration Postmortem (v1)")
    lines.append("")
    lines.append("**Date:** 2026-04-21")
    lines.append("**Pre-reg:** `docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1.yaml`")
    lines.append("**Authority:** `docs/institutional/pre_registered_criteria.md` Amendment 3.2")
    lines.append(f"**Seeds per hypothesis:** {N_SEEDS}")
    lines.append(
        f"**Trades per seed:** {N_TRADES} | n_splits={N_SPLITS} | "
        f"n_test_splits={N_TEST_SPLITS} | alpha={ALPHA}"
    )
    lines.append("")
    overall = "PASS" if all(h["passed"] for h in (h1, h2, h3)) else "FAIL"
    lines.append(f"## Overall verdict: **{overall}**")
    lines.append("")

    # H1
    lines.append("## H1 — Known-null reject rate matches alpha")
    lines.append("")
    lines.append(f"- Mean reject fraction: **{_fmt_float(h1['mean_reject_fraction'])}**")
    lines.append(f"- Pass band: [{h1['pass_band'][0]}, {h1['pass_band'][1]}]")
    lines.append(f"- Verdict: **{h1['verdict']}**")
    lines.append(
        f"- Per-seed reject fractions: {[round(v, 4) for v in h1['per_seed_reject_fractions']]}"
    )
    lines.append("")

    # H2
    lines.append("## H2 — Known-edge recovery matches theoretical power")
    lines.append("")
    lines.append(f"- Effect size: +{h2['effect_size']} R per trade, sd=1.0")
    lines.append(f"- Mean per-fold N: {h2['mean_per_fold_n']}")
    lines.append(f"- Observed mean reject fraction: **{_fmt_float(h2['mean_reject_fraction'])}**")
    lines.append(f"- Theoretical power at equivalent N: **{_fmt_float(h2['theoretical_power'])}**")
    lines.append(f"- |gap|: {_fmt_float(h2['abs_gap'])} (pass ≤ 0.10, kill > 0.20)")
    lines.append(f"- Verdict: **{h2['verdict']}**")
    lines.append("")

    # H3
    lines.append("## H3 — Embargo sensitivity on AR(1)")
    lines.append("")
    lines.append(f"- AR(1) ρ: {h3['rho']}")
    lines.append(f"- embargo=0:  mean reject = **{_fmt_float(h3['mean_reject_fraction_by_embargo'][0])}**")
    lines.append(f"- embargo=5:  mean reject = **{_fmt_float(h3['mean_reject_fraction_by_embargo'][5])}**")
    lines.append(f"- embargo=10: mean reject = **{_fmt_float(h3['mean_reject_fraction_by_embargo'][10])}**")
    lines.append(f"- embargo=20: mean reject = **{_fmt_float(h3['mean_reject_fraction_by_embargo'][20])}**")
    lines.append(f"- gap (embargo=0 − embargo=5): {_fmt_float(h3['gap_0_vs_5'])} (pass ≥ 0.03)")
    lines.append(f"- check (a) embargo=0 > embargo=5: **{h3['check_a_passed']}**")
    lines.append(f"- check (b) embargo∈{{5,10,20}} within [0.025, 0.10]: **{h3['check_b_passed']}**")
    lines.append(f"- production embargo chosen: **{h3['production_embargo']}**")
    lines.append(f"- Verdict: **{h3['verdict']}**")
    lines.append("")

    # Integration decision
    lines.append("## Integration decision")
    lines.append("")
    if overall == "PASS":
        lines.append(
            f"All three pre-registered calibration hypotheses passed. The CPCV "
            f"implementation in `trading_app/cpcv.py` is validated for wiring "
            f"into `strategy_validator._check_criterion_8_oos` as an opt-in "
            f"`cpcv_fallback` kwarg. Production embargo: **{h3['production_embargo']}**."
        )
        lines.append("")
        lines.append(
            "Integration itself is a FOLLOW-ON stage (not this commit) and requires "
            "a separate design-proposal-gate pass since it changes validator behavior."
        )
    else:
        failed = [h for h in (h1, h2, h3) if not h["passed"]]
        lines.append(
            "One or more calibration hypotheses failed. The CPCV implementation is "
            "PARKED — do NOT wire into strategy_validator."
        )
        lines.append("")
        for h in failed:
            lines.append(f"- {h['hypothesis_id']}: {h['verdict']} — {h['name']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print("=" * 60)
    print("CPCV CALIBRATION v1")
    print("pre-reg: docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1.yaml")
    print("=" * 60)
    print()

    print("Running H1 (known-null)...")
    h1 = run_h1_known_null()
    print(f"  mean reject fraction: {h1['mean_reject_fraction']:.4f}  verdict: {h1['verdict']}")
    print()

    print("Running H2 (known-edge)...")
    h2 = run_h2_known_edge()
    print(
        f"  observed={h2['mean_reject_fraction']:.4f}  "
        f"theoretical={h2['theoretical_power']:.4f}  "
        f"gap={h2['abs_gap']:.4f}  verdict: {h2['verdict']}"
    )
    print()

    print("Running H3 (embargo sensitivity)...")
    h3 = run_h3_embargo_sensitivity()
    print(f"  embargo means: {h3['mean_reject_fraction_by_embargo']}")
    print(
        f"  gap(0 vs 5): {h3['gap_0_vs_5']:.4f}  "
        f"production embargo: {h3['production_embargo']}  "
        f"verdict: {h3['verdict']}"
    )
    print()

    out_path = REPO_ROOT / "docs" / "audit" / "hypotheses" / "2026-04-21-cpcv-infrastructure-v1-postmortem.md"
    write_postmortem(h1, h2, h3, out_path)
    print(f"Postmortem written to: {out_path.relative_to(REPO_ROOT)}")

    overall_pass = all(h["passed"] for h in (h1, h2, h3))
    print()
    print("=" * 60)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
