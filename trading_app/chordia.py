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
from datetime import date, datetime
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
# (b) per-strategy strict-replay audit entries with verdict + audit_date.
#
# The allocator MUST trust the YAML's verdict field for live deployment
# decisions. Using `validated_setups.sharpe_ratio` + `sample_size` as a live
# proxy would reintroduce the exact Mode B / derived-layer doctrine violation
# that Phase 0 research truth protocol forbids. `validated_setups` is a
# strict-IS provenance shelf for VALIDATOR_NATIVE rows: every column
# (sample_size, win_rate, expectancy_r, sharpe_ratio, first_trade_day,
# last_trade_day, trade_day_count) pairs to one consistent population —
# trades on signal with trading_day < HOLDOUT_SACRED_FROM, frozen at
# promotion-time. Refresh via backfill_validated_trade_windows.py heals
# window-column drift back to the same strict-IS scope (2026-05-21). It is
# NOT a live trailing-performance view; consumers needing recent fire rates
# should query orb_outcomes directly with the strategy's filter applied.

CHORDIA_AUDIT_LOG_PATH = Path(__file__).resolve().parents[1] / "docs" / "runtime" / "chordia_audit_log.yaml"
CHORDIA_AUDIT_FRESHNESS_DAYS_DEFAULT = 90


@dataclass(frozen=True)
class ChordiaAuditEntry:
    """One strategy's chordia doctrine state.

    ``has_theory`` is a doctrine flag that selects the t-stat hurdle.
    ``audit_date`` is when the strategy was last revalidated (drives the
    staleness check). ``verdict`` is the strict-replay verdict recorded at
    that audit; the live allocator uses this field directly and fails closed
    to ``MISSING`` when no audit verdict exists.

    Optional addendum fields (``t_stat``, ``t_stat_source``,
    ``t_stat_csv_recompute``, ``oos_n``, ``oos_power``,
    ``audit_reaffirmed_date``) carry per-row reproducibility and OOS-power
    context introduced by PR #213 / PR #221. They are not consumed by the
    live allocator gate; they are loaded so future audit-trail tools can
    surface them without YAML-vs-code schema drift.
    """

    strategy_id: str
    has_theory: bool
    audit_date: date | None
    verdict: str | None
    theory_ref: str | None
    t_stat: float | None = None
    t_stat_source: str | None = None
    t_stat_csv_recompute: float | None = None
    oos_n: int | None = None
    oos_power: float | None = None
    audit_reaffirmed_date: date | None = None


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

    def verdict(self, strategy_id: str) -> str | None:
        entry = self.entries.get(strategy_id)
        return entry.verdict if entry is not None else None

    def audit_age_days(self, strategy_id: str, today: date) -> int | None:
        d = self.audit_date(strategy_id)
        if d is None:
            return None
        return (today - d).days


def _coerce_audit_date(raw: object) -> date | None:
    """Coerce a YAML date-like value to ``date`` or ``None``.

    YAML loaders may yield a string (quoted), a native ``date`` (unquoted
    ISO date), or a native ``datetime`` (unquoted ISO timestamp). The
    ``datetime`` branch precedes the ``date`` branch because ``datetime``
    is a subclass of ``date``: ``isinstance(datetime.now(), date)`` is
    ``True``, so an unquoted timestamp would otherwise return as
    ``datetime`` into a ``date | None`` field, breaking
    ``ChordiaAuditLog.audit_age_days`` arithmetic. Anything else is
    treated as missing.
    """
    if isinstance(raw, str):
        return date.fromisoformat(raw)
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    return None


def _coerce_optional_float(raw: object) -> float | None:
    """Coerce a YAML scalar to ``float`` or ``None``; warns on bad type.

    Booleans are rejected because Python treats ``True``/``False`` as
    ``int`` subclasses; an audit row with ``t_stat: true`` is a doctrine
    error, not a numeric value.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        _LOG.warning("chordia audit: expected float, got bool (%s) — coerced to None", raw)
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            _LOG.warning("chordia audit: cannot parse %r as float — coerced to None", raw)
            return None
    _LOG.warning("chordia audit: unsupported type for float field (%s) — coerced to None", type(raw).__name__)
    return None


def _coerce_optional_int(raw: object) -> int | None:
    """Coerce a YAML scalar to ``int`` or ``None``; warns on bad type."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        _LOG.warning("chordia audit: expected int, got bool (%s) — coerced to None", raw)
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            _LOG.warning("chordia audit: cannot parse %r as int — coerced to None", raw)
            return None
    _LOG.warning("chordia audit: unsupported type for int field (%s) — coerced to None", type(raw).__name__)
    return None


def _coerce_optional_str(raw: object) -> str | None:
    """Coerce a YAML scalar to ``str`` or ``None``."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    return str(raw)


def _coerce_oos_power(raw: object, strategy_id: str) -> float | None:
    """Load ``oos_power`` with range-check log.warning per stage spec.

    Out-of-range values (outside ``[0.0, 1.0]``) emit a warning but the
    value is accepted — the allocator gate does not consume this field,
    so range violations must NOT fail-closed. Pattern matches
    ``load_chordia_audit_log`` parse-error handling at lines 240-261:
    loud-but-non-fatal for non-allocator-load-bearing fields.
    """
    value = _coerce_optional_float(raw)
    if value is not None and not 0.0 <= value <= 1.0:
        _LOG.warning(
            "chordia audit: oos_power=%s for %s outside [0.0, 1.0] — accepted as-is",
            value,
            strategy_id,
        )
    return value


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
        d = _coerce_audit_date(audit.get("audit_date"))
        reaffirmed = _coerce_audit_date(audit.get("audit_reaffirmed_date"))
        oos_power = _coerce_oos_power(audit.get("oos_power"), sid)
        has_theory, theory_ref = theory_map.get(sid, (default_has_theory, None))
        entries[sid] = ChordiaAuditEntry(
            strategy_id=sid,
            has_theory=has_theory,
            audit_date=d,
            verdict=audit.get("verdict"),
            theory_ref=theory_ref,
            t_stat=_coerce_optional_float(audit.get("t_stat")),
            t_stat_source=_coerce_optional_str(audit.get("t_stat_source")),
            t_stat_csv_recompute=_coerce_optional_float(audit.get("t_stat_csv_recompute")),
            oos_n=_coerce_optional_int(audit.get("oos_n")),
            oos_power=oos_power,
            audit_reaffirmed_date=reaffirmed,
        )

    # Strategies with a theory grant but no audit row still appear in
    # entries so has_theory() returns the granted value rather than the
    # default. The live gate still treats them as missing-audit -> PAUSED
    # until a strict replay verdict exists.
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
