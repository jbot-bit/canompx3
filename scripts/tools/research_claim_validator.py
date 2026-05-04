"""Validate research claims against RESEARCH_RULES.md thresholds.

Takes a research claim (N, p_value, bh_k, wfe, mechanism, time_span)
and returns a GovernanceDecision with reasons.

Callable as a Python function or CLI tool.

@research-source RESEARCH_RULES.md
@revalidated-for E1/E2 event-based (2026-04-03)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.trace import GovernanceDecision

# ---------------------------------------------------------------------------
# Thresholds from RESEARCH_RULES.md — do not change without updating that doc
# ---------------------------------------------------------------------------

# Sample size classes (RESEARCH_RULES.md § Statistical Rigor)
N_INVALID_THRESHOLD = 30  # < 30 = INVALID
N_REGIME_THRESHOLD = 100  # 30-99 = REGIME_ONLY
N_HIGH_CONF_THRESHOLD = 500  # 500+ = HIGH-CONFIDENCE

# p-value thresholds (RESEARCH_RULES.md § Significance Testing)
P_NOTE = 0.05  # minimum to note
P_RECOMMEND = 0.01  # minimum to recommend
P_DISCOVERY = 0.005  # minimum for new discovery

# Walk-forward efficiency (RESEARCH_RULES.md § Significance Testing)
WFE_MINIMUM = 0.50  # minimum for "reality" claims


@dataclass
class ValidationResult:
    """Result of validating a research claim."""

    decision: GovernanceDecision
    sample_class: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "sample_class": self.sample_class,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


def classify_sample_size(n: int) -> str:
    """Classify sample size per RESEARCH_RULES.md thresholds."""
    if n < N_INVALID_THRESHOLD:
        return "INVALID"
    if n < N_REGIME_THRESHOLD:
        return "REGIME"
    if n < 200:
        return "PRELIMINARY"
    if n < N_HIGH_CONF_THRESHOLD:
        return "CORE"
    return "HIGH-CONFIDENCE"


def validate_claim(
    n: int,
    p_value: float,
    bh_k: int | None = None,
    wfe: float | None = None,
    mechanism: str | None = None,
    time_span_years: float | None = None,
    sensitivity_passed: bool | None = None,
) -> ValidationResult:
    """Validate a research claim against RESEARCH_RULES.md thresholds.

    Args:
        n: Sample size (number of trades).
        p_value: Two-tailed p-value.
        bh_k: Number of tests for BH FDR correction.
        wfe: Walk-forward efficiency (0-1).
        mechanism: Structural market explanation.
        time_span_years: Data span in years.
        sensitivity_passed: Whether ±20% parameter change survived.

    Returns:
        ValidationResult with decision, sample class, reasons, warnings.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    decision = GovernanceDecision.VALID

    # --- 1. Sample size ---
    sample_class = classify_sample_size(n)
    if sample_class == "INVALID":
        reasons.append(f"N={n} < {N_INVALID_THRESHOLD}: below minimum sample size")
        decision = GovernanceDecision.INVALID
    elif sample_class == "REGIME":
        reasons.append(f"N={n}: REGIME class (30-99), conditional use only")
        decision = GovernanceDecision.REGIME_ONLY

    # --- 2. p-value ---
    if p_value >= P_NOTE:
        reasons.append(f"p={p_value:.4f} >= {P_NOTE}: not significant")
        decision = GovernanceDecision.INVALID
    elif p_value >= P_RECOMMEND:
        warnings.append(f"p={p_value:.4f}: significant but above recommendation threshold ({P_RECOMMEND})")
    elif p_value >= P_DISCOVERY:
        warnings.append(f"p={p_value:.4f}: below recommendation but above discovery threshold ({P_DISCOVERY})")

    # --- 3. BH FDR K ---
    if bh_k is not None and bh_k > 1:
        # BH correction: strategy must survive at its rank
        # We can't compute the full BH here without all p-values,
        # but we flag the K for context
        warnings.append(f"BH FDR K={bh_k}: verify adjusted p-value survives BH correction at this K")

    # --- 4. WFE ---
    if wfe is not None:
        if wfe < WFE_MINIMUM:
            reasons.append(f"WFE={wfe:.2f} < {WFE_MINIMUM}: walk-forward efficiency too low")
            if decision in (GovernanceDecision.VALID, GovernanceDecision.REGIME_ONLY):
                decision = GovernanceDecision.UNSUPPORTED
        elif wfe < 0.65:
            warnings.append(f"WFE={wfe:.2f}: acceptable but moderate (50-65%)")

    # --- 5. Mechanism ---
    if mechanism is None or mechanism.strip() == "":
        reasons.append("No mechanism provided: UNSUPPORTED without structural explanation")
        if decision in (GovernanceDecision.VALID, GovernanceDecision.REGIME_ONLY):
            decision = GovernanceDecision.UNSUPPORTED

    # --- 6. Time span ---
    if time_span_years is not None and time_span_years < 3.0:
        warnings.append(f"Time span {time_span_years:.1f}yr: may not capture regime diversity")

    # --- 7. Sensitivity ---
    if sensitivity_passed is False:
        reasons.append("Sensitivity FAILED: ±20% parameter change kills finding (curve-fitted)")
        if decision in (GovernanceDecision.VALID, GovernanceDecision.REGIME_ONLY):
            decision = GovernanceDecision.INVALID

    return ValidationResult(
        decision=decision,
        sample_class=sample_class,
        reasons=reasons,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate research claims against RESEARCH_RULES.md")
    parser.add_argument("--n", type=int, required=True, help="Sample size (N trades)")
    parser.add_argument("--p", type=float, required=True, help="p-value (two-tailed)")
    parser.add_argument("--k", type=int, default=None, help="BH FDR K (number of tests)")
    parser.add_argument("--wfe", type=float, default=None, help="Walk-forward efficiency (0-1)")
    parser.add_argument("--mechanism", type=str, default=None, help="Mechanism description")
    parser.add_argument("--years", type=float, default=None, help="Time span in years")
    parser.add_argument(
        "--sensitivity-passed",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Did ±20%% sensitivity pass?",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    sens = None
    if args.sensitivity_passed is not None:
        sens = args.sensitivity_passed == "true"

    result = validate_claim(
        n=args.n,
        p_value=args.p,
        bh_k=args.k,
        wfe=args.wfe,
        mechanism=args.mechanism,
        time_span_years=args.years,
        sensitivity_passed=sens,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Decision: {result.decision.value}")
        print(f"Sample class: {result.sample_class}")
        if result.reasons:
            print("Reasons:")
            for r in result.reasons:
                print(f"  - {r}")
        if result.warnings:
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w}")

    # Exit code: 0=VALID/REGIME_ONLY, 1=INVALID/BLOCKED, 2=UNSUPPORTED/STALE
    if result.decision in (GovernanceDecision.VALID, GovernanceDecision.REGIME_ONLY):
        sys.exit(0)
    elif result.decision in (GovernanceDecision.INVALID, GovernanceDecision.BLOCKED):
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
