"""
Chordia t-statistic gate — Chordia, Goyal, Saretto (2018).

Computes the t-statistic of a strategy's mean per-trade return against the null
hypothesis of zero mean. Used as Criterion 4 of the locked institutional
criteria in `docs/institutional/pre_registered_criteria.md`.

The mathematical identity used here:
    sharpe_per_trade = mean[R] / std[R]
    t_statistic     = mean[R] / (std[R] / sqrt(N))
                    = (mean[R] / std[R]) * sqrt(N)
                    = sharpe_per_trade * sqrt(N)

So when the validator stores `sharpe_ratio` as the per-trade Sharpe (which it
does, see `trading_app/strategy_validator.py` and `validated_setups` schema),
the t-statistic is simply `sharpe_ratio * sqrt(sample_size)`. No separate std
column is required.

@research-source: docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md
@canonical-source: trading_app/chordia.py
@revalidated-for: Phase 4 Stage 4.0 (2026-04-08, criterion 4 enforcement)

Thresholds (locked in `pre_registered_criteria.md` Criterion 4):
- t >= 3.00 for strategies with strong pre-registered economic theory support
  (Harvey-Liu-Zhu 2015 hurdle)
- t >= 3.79 for strategies without such theoretical support
  (Chordia et al 2018 hurdle, K=2 million strategies tested)

These are HARD GATES — no post-hoc relaxation allowed per the locked criteria
file. If a strategy is borderline, the institutional response is to declare it
research-provisional, not to lower the threshold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

_LOG = logging.getLogger(__name__)

# Locked thresholds — sourced from docs/institutional/literature/.
# Modifying these requires an amendment to pre_registered_criteria.md
# Criterion 4 with explicit literature citation.
CHORDIA_T_WITH_THEORY: float = 3.00
CHORDIA_T_WITHOUT_THEORY: float = 3.79


def chordia_threshold(has_theory: bool) -> float:
    """Return the locked Chordia t-statistic threshold for the given theory state.

    Parameters
    ----------
    has_theory
        True if the strategy has a pre-registered economic theory citation in
        its hypothesis file. False if not (the discovery stands purely on
        empirical fit).

    Returns
    -------
    float
        3.00 if has_theory else 3.79.
    """
    return CHORDIA_T_WITH_THEORY if has_theory else CHORDIA_T_WITHOUT_THEORY


def compute_chordia_t(sharpe_ratio: float, sample_size: int) -> float:
    """Compute the Chordia t-statistic from per-trade Sharpe and sample size.

    Uses the identity ``t = sharpe_per_trade * sqrt(N)`` which follows directly
    from ``sharpe_per_trade = mean[R] / std[R]`` and the standard t-statistic
    definition ``t = mean[R] / (std[R] / sqrt(N))``.

    Parameters
    ----------
    sharpe_ratio
        Per-trade Sharpe ratio (mean of per-trade R-multiples divided by their
        standard deviation). This is the canonical form stored in
        ``validated_setups.sharpe_ratio`` and ``experimental_strategies.sharpe_ratio``.
    sample_size
        Number of trades in the sample. Must be at least 2 to compute the
        t-statistic; the standard error of the mean is undefined for N < 2.

    Returns
    -------
    float
        The t-statistic. Larger absolute values indicate stronger evidence
        against the null (mean per-trade return = 0).

    Raises
    ------
    ValueError
        If ``sample_size < 2``. The Chordia gate cannot be applied to a sample
        with fewer than 2 trades.
    """
    if sample_size < 2:
        raise ValueError(
            f"Chordia t-statistic requires sample_size >= 2, got {sample_size}. "
            f"The standard error of the mean is undefined for N < 2."
        )
    return sharpe_ratio * math.sqrt(sample_size)


def chordia_gate(
    sharpe_ratio: float,
    sample_size: int,
    has_theory: bool,
) -> tuple[bool, float, float]:
    """DEPRECATED — use ``chordia_verdict_label`` and ``chordia_verdict_allows_deploy``.

    The 5-state taxonomy (PASS_CHORDIA / PASS_PROTOCOL_A / FAIL_CHORDIA /
    FAIL_BOTH / MISSING) is the canonical interface as of the
    allocator_chordia_gate stage (2026-05-01). This 3-tuple boolean form
    is retained only for the existing ``test_chordia.py`` boundary tests.
    Production code MUST NOT call this function; the allocator gate refuses
    DEPLOY for FAIL_CHORDIA which the boolean form cannot distinguish from
    PASS_PROTOCOL_A. New callers will get the wrong policy.

    Apply the Chordia t-statistic gate to a strategy.

    Parameters
    ----------
    sharpe_ratio
        Per-trade Sharpe ratio.
    sample_size
        Number of trades.
    has_theory
        True if a pre-registered theory citation exists for the hypothesis
        family this strategy belongs to.

    Returns
    -------
    tuple[bool, float, float]
        ``(passed, t_statistic, threshold)`` where ``passed`` is True iff
        ``t_statistic >= threshold``. The threshold is selected by
        ``chordia_threshold(has_theory)``.

    Raises
    ------
    ValueError
        If ``sample_size < 2`` (propagated from ``compute_chordia_t``).
    """
    threshold = chordia_threshold(has_theory)
    t_stat = compute_chordia_t(sharpe_ratio, sample_size)
    return (t_stat >= threshold, t_stat, threshold)


# ---------------------------------------------------------------------------
# Audit-log reader — used by the allocator Chordia gate.
# ---------------------------------------------------------------------------
# Doctrine YAML lives at `docs/runtime/chordia_audit_log.yaml`. It carries
# (a) per-strategy `has_theory` grants with literature citations, and
# (b) per-strategy audit entries with verdict + audit_date.
#
# The allocator does NOT trust the YAML's verdict field for gate decisions —
# the verdict is recomputed live from `validated_setups.sharpe_ratio` +
# `sample_size` via chordia_verdict_label. The YAML is the source of truth
# ONLY for the `has_theory` flag (a doctrine choice, not a computation) and
# for the `audit_date` (when a human revalidated; needed for staleness).

CHORDIA_AUDIT_LOG_PATH = Path(__file__).resolve().parents[1] / "docs" / "runtime" / "chordia_audit_log.yaml"
CHORDIA_AUDIT_FRESHNESS_DAYS_DEFAULT = 90


@dataclass(frozen=True)
class ChordiaAuditEntry:
    """One strategy's chordia doctrine state.

    ``has_theory`` is a doctrine flag that selects the t-stat hurdle.
    ``audit_date`` is when the strategy was last revalidated (drives the
    staleness check). ``verdict`` is the verdict recorded at that audit; the
    live gate recomputes the verdict from current ``validated_setups`` data
    and ignores this field for selection — retained for traceability.
    """

    strategy_id: str
    has_theory: bool
    audit_date: date | None
    verdict: str | None
    theory_ref: str | None


@dataclass(frozen=True)
class ChordiaAuditLog:
    """Parsed chordia_audit_log.yaml. Index by strategy_id; query helpers below."""

    default_has_theory: bool
    audit_freshness_days: int
    entries: dict[str, ChordiaAuditEntry]

    def has_theory(self, strategy_id: str) -> bool:
        entry = self.entries.get(strategy_id)
        if entry is None:
            return self.default_has_theory
        return entry.has_theory

    def audit_date(self, strategy_id: str) -> date | None:
        entry = self.entries.get(strategy_id)
        return entry.audit_date if entry is not None else None

    def audit_age_days(self, strategy_id: str, today: date) -> int | None:
        d = self.audit_date(strategy_id)
        if d is None:
            return None
        return (today - d).days


def load_chordia_audit_log(path: str | Path | None = None) -> ChordiaAuditLog:
    """Read and parse the chordia audit-log YAML.

    Fail-closed: if the file is missing OR malformed (YAML parse error,
    not a mapping at the root), returns an empty log with
    default_has_theory=False and audit_freshness_days set to the
    institutional default (90). The allocator gate then treats every
    strategy as 'missing audit' -> PAUSED, which is the strict prior.

    Malformed YAML is logged at WARNING — silent fail-closed without a
    log line would mask doctrine corruption from the operator.
    """
    p = Path(path) if path is not None else CHORDIA_AUDIT_LOG_PATH
    if not p.exists():
        return ChordiaAuditLog(
            default_has_theory=False,
            audit_freshness_days=CHORDIA_AUDIT_FRESHNESS_DAYS_DEFAULT,
            entries={},
        )

    try:
        parsed = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        _LOG.warning(
            "chordia audit log %s could not be parsed (%s) — "
            "fail-closed: every strategy will be PAUSED until the YAML is fixed",
            p,
            exc,
        )
        return ChordiaAuditLog(
            default_has_theory=False,
            audit_freshness_days=CHORDIA_AUDIT_FRESHNESS_DAYS_DEFAULT,
            entries={},
        )
    if not isinstance(parsed, dict):
        _LOG.warning(
            "chordia audit log %s did not parse to a mapping (got %s) — "
            "fail-closed: every strategy will be PAUSED until the YAML is fixed",
            p,
            type(parsed).__name__,
        )
        return ChordiaAuditLog(
            default_has_theory=False,
            audit_freshness_days=CHORDIA_AUDIT_FRESHNESS_DAYS_DEFAULT,
            entries={},
        )
    raw = parsed
    default_has_theory = bool(raw.get("default_has_theory", False))
    freshness = int(raw.get("audit_freshness_days", CHORDIA_AUDIT_FRESHNESS_DAYS_DEFAULT))

    # Build the theory-grant map first. Only entries with explicit theory_ref
    # earn has_theory=True; bare has_theory:true without a citation is
    # demoted to False per the YAML's own doctrine.
    theory_map: dict[str, tuple[bool, str | None]] = {}
    for grant in raw.get("theory_grants", []) or []:
        sid = grant.get("strategy_id")
        if not sid:
            continue
        wants_theory = bool(grant.get("has_theory", False))
        ref = grant.get("theory_ref")
        effective_theory = wants_theory and bool(ref)
        theory_map[sid] = (effective_theory, ref if effective_theory else None)

    # Then merge audit entries.
    entries: dict[str, ChordiaAuditEntry] = {}
    for audit in raw.get("audits", []) or []:
        sid = audit.get("strategy_id")
        if not sid:
            continue
        d = audit.get("audit_date")
        if isinstance(d, str):
            d = date.fromisoformat(d)
        elif not isinstance(d, date) and d is not None:
            d = None
        has_theory, theory_ref = theory_map.get(sid, (default_has_theory, None))
        entries[sid] = ChordiaAuditEntry(
            strategy_id=sid,
            has_theory=has_theory,
            audit_date=d,
            verdict=audit.get("verdict"),
            theory_ref=theory_ref,
        )

    # Strategies with a theory grant but no audit row still appear in
    # entries so has_theory() returns the granted value rather than the
    # default. The gate will mark them missing-audit -> PAUSED, but any
    # live recomputation uses the correct hurdle (3.00) for them.
    for sid, (has_theory, theory_ref) in theory_map.items():
        if sid in entries:
            continue
        entries[sid] = ChordiaAuditEntry(
            strategy_id=sid,
            has_theory=has_theory,
            audit_date=None,
            verdict=None,
            theory_ref=theory_ref,
        )

    return ChordiaAuditLog(
        default_has_theory=default_has_theory,
        audit_freshness_days=freshness,
        entries=entries,
    )


def chordia_verdict_label(
    sharpe_ratio: float | None,
    sample_size: int | None,
    has_theory: bool,
) -> str:
    """Compute the verdict label used by the allocator gate.

    Returns one of:
      - PASS_CHORDIA     : t >= 3.79 (strict; sizing-up eligible)
      - PASS_PROTOCOL_A  : 3.00 <= t < 3.79 with has_theory=True
      - FAIL_CHORDIA     : 3.00 <= t < 3.79 with has_theory=False
      - FAIL_BOTH        : t < 3.00 (always)
      - MISSING          : sharpe_ratio or sample_size is None / N < 2

    Allocator policy:
      - PASS_CHORDIA, PASS_PROTOCOL_A -> may DEPLOY
      - FAIL_CHORDIA, FAIL_BOTH, MISSING -> refuse DEPLOY (PAUSED)
    """
    if sharpe_ratio is None or sample_size is None or sample_size < 2:
        return "MISSING"
    t = compute_chordia_t(sharpe_ratio, sample_size)
    if t >= CHORDIA_T_WITHOUT_THEORY:
        return "PASS_CHORDIA"
    if t >= CHORDIA_T_WITH_THEORY:
        return "PASS_PROTOCOL_A" if has_theory else "FAIL_CHORDIA"
    return "FAIL_BOTH"


def chordia_verdict_allows_deploy(verdict: str) -> bool:
    """Policy: which verdicts permit DEPLOY status."""
    return verdict in ("PASS_CHORDIA", "PASS_PROTOCOL_A")


__all__ = [
    "CHORDIA_T_WITH_THEORY",
    "CHORDIA_T_WITHOUT_THEORY",
    "CHORDIA_AUDIT_LOG_PATH",
    "CHORDIA_AUDIT_FRESHNESS_DAYS_DEFAULT",
    "ChordiaAuditEntry",
    "ChordiaAuditLog",
    "chordia_threshold",
    "compute_chordia_t",
    "chordia_gate",
    "chordia_verdict_label",
    "chordia_verdict_allows_deploy",
    "load_chordia_audit_log",
]
