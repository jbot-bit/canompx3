"""Estimate Bailey 2013 MinBTL K-budget for a pre-registered hypothesis file.

Gates every new pre-reg under ``docs/audit/hypotheses/`` against the locked
Criterion 2 MinBTL bound. Pure math + a single YAML read. Read-only.

Authority chain (single source of truth — do not re-encode):
- ``scripts.tools.minbtl_retro_report.strict_bailey_n`` — canonical formula
  ``MinBTL = 2*Ln[N] / E[max_N]^2`` (loose upper bound per Bailey 2013
  Theorem 1, Eq. 6 right side; matches ``pre_registered_criteria.md``
  Criterion 2 wording).
- ``scripts.tools.minbtl_retro_report.CLEAN_YEARS_BY_INSTRUMENT`` — per-
  instrument clean-data horizons (Amendment 2.8, 2026-04-09).
- ``docs/institutional/pre_registered_criteria.md`` Criterion 2 — locked
  operational caps (N <= 300 clean, N <= 2000 proxy).
- ``docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md``
  Theorem 1 — primary literature anchor.

Loose vs tight form (see ``docs/institutional/literature/`` Eq. 6):
Bailey's Eq. 6 quotes both the tight middle expression and the looser
right side ``2*Ln[N]/E^2`` that bounds it from above. Project doctrine
(Criterion 2) cites the loose form, so this tool reports the loose-form
MinBTL. The tighter middle form gives smaller required horizons — using
the looser bound is the conservative choice (more demanding on the
researcher), which is the institutional posture.

Usage:
    # Inline numbers
    python scripts/tools/estimate_k_budget.py --instrument MNQ --n-trials 12
    # Read a pre-reg yaml
    python scripts/tools/estimate_k_budget.py \\
        --hypothesis docs/audit/hypotheses/2026-04-09-mnq-comprehensive.yaml
    # Override E[max_N] from default 1.0
    python scripts/tools/estimate_k_budget.py --instrument MGC --n-trials 4 --e-max 1.2

Exit codes:
    0 = PASS (N within MinBTL bound for given instrument horizon)
    1 = FAIL (N exceeds bound; pre-reg refused)
    2 = USAGE error
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Canonical math + horizons — delegate, do not re-encode.
from scripts.tools.minbtl_retro_report import (
    CLEAN_YEARS_BY_INSTRUMENT,
    LOCKED_OPERATIONAL_CAP,
    strict_bailey_n,
)

# Locked operational ceilings from pre_registered_criteria.md Criterion 2.
# Operational > strict Bailey at E=1.0; documented as institutional ceiling
# with explicit noise-floor disclosure required when exceeded.
LOCKED_PROXY_CAP = 2000


@dataclass(frozen=True)
class KBudgetReport:
    """Structured result from a MinBTL K-budget check.

    All fields populated regardless of verdict so callers can render a
    consistent report. ``passed`` is the single boolean gate; ``verdict``
    carries human-readable framing.
    """

    instrument: str
    clean_years: float
    n_trials: int
    e_max: float
    minbtl_years_required: float
    n_max_at_horizon: int
    operational_cap: int
    passed: bool
    verdict: str
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "clean_years": self.clean_years,
            "n_trials": self.n_trials,
            "e_max": self.e_max,
            "minbtl_years_required": self.minbtl_years_required,
            "n_max_at_horizon": self.n_max_at_horizon,
            "operational_cap": self.operational_cap,
            "passed": self.passed,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def required_minbtl_years(n_trials: int, e_max: float = 1.0) -> float:
    """Bailey 2013 Theorem 1 (Eq. 6, loose upper bound).

    MinBTL = 2 * Ln[N] / E[max_N]^2

    Returns the minimum backtest horizon (in years) needed so the best
    observed IS Sharpe across N independent trials is not pure selection
    luck at the chosen E[max_N] noise floor.

    Edge cases:
    - n_trials == 0: returns 0.0 (audit/no-discovery marker — no
      hypothesis enumeration to bound, used in audit-only pre-regs).
    - n_trials == 1: returns 0.0 (theory-driven Pathway B K=1; see
      pre_registered_criteria.md Amendment 3.0).
    - n_trials < 0 or e_max <= 0: ValueError.

    Inverse of ``strict_bailey_n``: that gives N_max for a horizon; this
    gives horizon required for an N.
    """
    if n_trials < 0:
        raise ValueError(f"n_trials must be >= 0, got {n_trials}")
    if e_max <= 0:
        raise ValueError(f"e_max must be > 0, got {e_max}")
    if n_trials <= 1:
        return 0.0
    return 2.0 * math.log(n_trials) / (e_max**2)


def estimate_k_budget(
    instrument: str,
    n_trials: int,
    e_max: float = 1.0,
    proxy_extended: bool = False,
) -> KBudgetReport:
    """Compute MinBTL budget verdict for a hypothesis configuration.

    Args:
        instrument: One of CLEAN_YEARS_BY_INSTRUMENT keys (MNQ/MES/MGC).
        n_trials: Total expected trial count for the pre-reg.
        e_max: Noise-floor target Sharpe (default 1.0, Bailey default).
        proxy_extended: If True, uses N<=2000 operational cap (price-based
            filters on GC proxy per Amendment 2.7); otherwise N<=300.

    Returns:
        KBudgetReport with pass/fail verdict and structured fields.

    Raises:
        ValueError: instrument unknown, n_trials<1, or e_max<=0.
    """
    if instrument not in CLEAN_YEARS_BY_INSTRUMENT:
        raise ValueError(
            f"Unknown instrument {instrument!r}. Known: "
            f"{sorted(CLEAN_YEARS_BY_INSTRUMENT)}. Extend "
            "minbtl_retro_report.CLEAN_YEARS_BY_INSTRUMENT after a "
            "Criterion 2 amendment."
        )
    clean_years = CLEAN_YEARS_BY_INSTRUMENT[instrument]
    minbtl_required = required_minbtl_years(n_trials, e_max=e_max)
    n_max_horizon = strict_bailey_n(clean_years, e_max=e_max)
    op_cap = LOCKED_PROXY_CAP if proxy_extended else LOCKED_OPERATIONAL_CAP

    horizon_ok = minbtl_required <= clean_years
    cap_ok = n_trials <= op_cap
    passed = horizon_ok and cap_ok

    notes: list[str] = []
    if not horizon_ok:
        notes.append(
            f"N={n_trials} requires {minbtl_required:.2f}yr of clean data at "
            f"E[max_N]={e_max}, but {instrument} has only {clean_years:.2f}yr. "
            f"Reduce N to <= {n_max_horizon} or accept noise-floor contamination."
        )
    if not cap_ok:
        cap_label = "proxy-extended (N<=2000)" if proxy_extended else "clean (N<=300)"
        notes.append(f"N={n_trials} exceeds locked operational cap {cap_label}. Criterion 2 banned without amendment.")
    if passed:
        headroom = clean_years - minbtl_required
        notes.append(
            f"N={n_trials} fits within {clean_years:.2f}yr horizon "
            f"(requires {minbtl_required:.2f}yr, {headroom:.2f}yr headroom)."
        )

    if not horizon_ok and not cap_ok:
        verdict = "FAIL — horizon and operational cap both violated"
    elif not horizon_ok:
        verdict = "FAIL — Bailey horizon violated"
    elif not cap_ok:
        verdict = "FAIL — operational cap violated"
    else:
        verdict = "PASS"

    return KBudgetReport(
        instrument=instrument,
        clean_years=clean_years,
        n_trials=n_trials,
        e_max=e_max,
        minbtl_years_required=minbtl_required,
        n_max_at_horizon=n_max_horizon,
        operational_cap=op_cap,
        passed=passed,
        verdict=verdict,
        notes=tuple(notes),
    )


# YAML parsing — kept zero-dep so the drift check / MCP tool never pull PyYAML
# into a cold-start path. We only need three scalars; the project's existing
# yaml fixtures use the unambiguous ``key: value`` form for these keys.
_SCALAR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^\n#]+?)\s*(?:#.*)?$")


def _parse_yaml_scalars(text: str) -> dict[str, str]:
    """Extract top-level + simply-nested ``key: value`` scalars from YAML.

    Sufficient for the hypothesis-file fields we need:
    - ``total_expected_trials`` (top-level or under ``metadata:``)
    - ``testing_mode`` (under ``metadata:``)
    - First ``instruments: [...]`` list value in scope blocks

    Not a general YAML parser — fails-open by returning ``{}`` for the
    missing key, which the caller turns into a clear error message.
    """
    scalars: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _SCALAR_RE.match(stripped)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip().strip('"').strip("'")
        # First occurrence wins — top-level beats nested if both exist.
        scalars.setdefault(key, value)
    return scalars


_INSTRUMENT_RE = re.compile(r"instruments\s*:\s*\[\s*([A-Za-z0-9_,\s]+)\s*\]")


def _extract_instruments(text: str) -> list[str]:
    """Collect every instrument referenced in any ``instruments: [...]`` list.

    Hypothesis yamls use inline-list form (e.g. ``instruments: [MNQ]``)
    consistently across all current files. Returns sorted unique list.
    """
    seen: set[str] = set()
    for match in _INSTRUMENT_RE.finditer(text):
        for token in match.group(1).split(","):
            token = token.strip()
            if token:
                seen.add(token)
    return sorted(seen)


@dataclass(frozen=True)
class HypothesisSummary:
    """Minimal hypothesis-file metadata needed for the K-budget check."""

    path: Path
    n_trials: int | None
    testing_mode: str | None
    pathway: str | None
    instruments: tuple[str, ...]
    proxy_extended: bool


_TRIAL_COUNT_KEYS = (
    "total_expected_trials",  # legacy + most common (Phase 0 corpus)
    "primary_selection_trials",  # 2026-04-18 vwap family scan schema
    "n_trials",  # Pathway B individual / portfolio-test schema
)


def load_hypothesis(path: Path) -> HypothesisSummary:
    """Read a hypothesis YAML and extract MinBTL-relevant fields.

    Tolerant of the project's mixed yaml conventions:
    - Trial count: first hit among ``total_expected_trials``,
      ``primary_selection_trials``, ``n_trials`` (in that priority).
    - ``testing_mode`` may be ``family`` or ``individual``
    - ``pathway`` may be present (Pathway A/B) on Amendment-3.0+ files
    - ``instruments`` is collected from every scope block
    - ``proxy_extended`` heuristic: ``proxy`` token in filename or
      ``data_source`` field; conservative default False.

    Priority order matters: ``total_expected_trials`` wins when both
    appear (legacy authoritative key); other keys fill in only when the
    canonical key is absent.
    """
    if not path.exists():
        raise FileNotFoundError(f"Hypothesis file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    scalars = _parse_yaml_scalars(text)
    n_trials: int | None = None
    for key in _TRIAL_COUNT_KEYS:
        raw_n = scalars.get(key)
        if raw_n is None:
            continue
        try:
            n_trials = int(raw_n)
            break
        except ValueError:
            continue
    instruments = tuple(_extract_instruments(text))
    name = path.name.lower()
    proxy_extended = "proxy" in name or "proxy" in (scalars.get("data_source") or "").lower()
    return HypothesisSummary(
        path=path,
        n_trials=n_trials,
        testing_mode=scalars.get("testing_mode"),
        pathway=scalars.get("pathway"),
        instruments=instruments,
        proxy_extended=proxy_extended,
    )


def check_hypothesis_file(path: Path, e_max: float = 1.0) -> list[KBudgetReport]:
    """Compute per-instrument K-budget reports for a hypothesis file.

    Returns one KBudgetReport per instrument in the file. Empty list means
    the file declared no instruments we recognize (skip silently — not
    every YAML in the hypotheses directory is an instrument-scoped pre-reg).

    Pathway B / individual / K=1 files are reported with N=1 (MinBTL=0,
    trivially PASS) per Criterion 2 Amendment 3.0.
    """
    summary = load_hypothesis(path)
    if summary.n_trials is None:
        # No declared trial count — cannot evaluate. Drift check turns this
        # into a hard error; the library returns empty for callers that
        # want to skip silently (e.g., the MCP tool).
        return []
    instruments = [i for i in summary.instruments if i in CLEAN_YEARS_BY_INSTRUMENT]
    if not instruments:
        return []
    reports: list[KBudgetReport] = []
    for instrument in instruments:
        report = estimate_k_budget(
            instrument=instrument,
            n_trials=summary.n_trials,
            e_max=e_max,
            proxy_extended=summary.proxy_extended,
        )
        reports.append(report)
    return reports


def format_report(report: KBudgetReport) -> str:
    """Single-screen human-readable report (CLI default)."""
    lines = [
        "MinBTL K-budget gate",
        "====================",
        f"Instrument:           {report.instrument}",
        f"Clean horizon (yr):   {report.clean_years:.2f}",
        f"N (declared trials):  {report.n_trials}",
        f"E[max_N] noise floor: {report.e_max}",
        "",
        f"MinBTL required:      {report.minbtl_years_required:.2f} yr (= 2*Ln[N] / E^2)",
        f"N_max at this horizon: {report.n_max_at_horizon} (strict Bailey @ E={report.e_max})",
        f"Operational cap:      N <= {report.operational_cap}",
        "",
        f"Verdict: {report.verdict}",
    ]
    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"  - {note}")
    lines.append("")
    lines.append(
        "Authority: pre_registered_criteria.md Criterion 2 (loose-form 2*Ln[N]/E^2 per Bailey 2013 Eq. 6 RHS)."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bailey 2013 MinBTL K-budget gate for a pre-reg hypothesis.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--hypothesis",
        type=Path,
        help="Path to a pre-reg yaml under docs/audit/hypotheses/.",
    )
    src.add_argument(
        "--instrument",
        choices=sorted(CLEAN_YEARS_BY_INSTRUMENT),
        help="Instrument symbol (paired with --n-trials).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Total trial count (required with --instrument).",
    )
    parser.add_argument(
        "--e-max",
        type=float,
        default=1.0,
        help="E[max_N] noise floor target Sharpe (Bailey default 1.0).",
    )
    parser.add_argument(
        "--proxy-extended",
        action="store_true",
        help="Use N<=2000 operational cap (price-based filters on GC proxy).",
    )
    args = parser.parse_args(argv)

    reports: list[KBudgetReport]
    if args.hypothesis is not None:
        reports = check_hypothesis_file(args.hypothesis, e_max=args.e_max)
        if not reports:
            print(
                f"No evaluable instrument scope or trial count in "
                f"{args.hypothesis}. (Pathway B K=1 files pass trivially.)",
                file=sys.stderr,
            )
            return 2
    else:
        if args.n_trials is None:
            parser.error("--n-trials is required with --instrument")
        reports = [
            estimate_k_budget(
                instrument=args.instrument,
                n_trials=args.n_trials,
                e_max=args.e_max,
                proxy_extended=args.proxy_extended,
            )
        ]

    any_fail = False
    for i, report in enumerate(reports):
        if i > 0:
            print()
        print(format_report(report))
        if not report.passed:
            any_fail = True

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
