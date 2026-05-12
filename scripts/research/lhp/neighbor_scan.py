"""Neighbor / family scan — surface sibling cells around a candidate.

Given a candidate (e.g. ``MNQ CME_PRECLOSE ATR_P30``), enumerate adjacent
cells that share most of the spec but vary one axis:

- **Threshold neighbours:** ±N% of the filter's numeric threshold
  (e.g. ATR_P30 → ATR_P24, ATR_P36 at ±20%).
- **Session neighbours:** other ORB sessions on the same instrument.
- **Aperture neighbours:** O5 / O15 / O30 variants of the same (session,
  filter, rr).

For each neighbour we run two reads in parallel:

  1. **Mode A strict OOS screen** (`adjacency.screen_candidate_mode_a`) —
     would the neighbour pass Criterion 8 today.
  2. **Graveyard check** (`graveyard.check_graveyard`) — any prior KILL /
     PARK / NO-GO verdict on this neighbour.

The result is annotated context for the LLM and a counter
``siblings_killed`` that the static-check gate uses to flag suspect families.

Performance: each neighbour costs ~1 SQL roundtrip + 1 catalogue call.
Capped at 12 neighbours per candidate to keep wall time under ~10s.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical session list — sourced from pipeline.dst.SESSION_CATALOG to obey
# the "never hardcode session names" rule (integrity-guardian.md § 2).
# ---------------------------------------------------------------------------


def _canonical_sessions() -> list[str]:
    try:
        from pipeline.dst import SESSION_CATALOG  # type: ignore[import-not-found]
    except ImportError:
        # Fallback only fires during static analysis / hermetic tests where
        # pipeline.dst can't be imported. Caller should treat this as a
        # warning. We do NOT hardcode the canonical 12 here — empty list
        # forces neighbour scan to skip session axis, which is honest.
        logger.warning("neighbor_scan: pipeline.dst.SESSION_CATALOG unavailable; session axis skipped")
        return []
    try:
        return [str(s) for s in SESSION_CATALOG]  # type: ignore[union-attr]
    except AttributeError:
        return list(SESSION_CATALOG)  # type: ignore[arg-type]


_APERTURE_NEIGHBOURS: tuple[int, ...] = (5, 15, 30)


# ---------------------------------------------------------------------------
# Threshold-neighbour generation
# ---------------------------------------------------------------------------

_NUMERIC_SUFFIX_RE = re.compile(r"^(?P<root>[A-Z_]+?)(?P<num>\d+)(?P<tail>K?)$")


def _threshold_neighbours(filter_type: str, *, pct: float = 0.20) -> list[str]:
    """Generate ±pct neighbours for a filter with a numeric threshold.

    >>> _threshold_neighbours("ATR_P30")
    ['ATR_P24', 'ATR_P36']
    >>> _threshold_neighbours("ORB_VOL_16K")
    ['ORB_VOL_12K', 'ORB_VOL_19K']
    >>> _threshold_neighbours("ATR_VEL_GE105")
    ['ATR_VEL_GE84', 'ATR_VEL_GE126']
    >>> _threshold_neighbours("OVNRNG_100")
    ['OVNRNG_80', 'OVNRNG_120']
    >>> _threshold_neighbours("NONUMERIC_FILTER")
    []
    """
    match = _NUMERIC_SUFFIX_RE.match(filter_type)
    if not match:
        return []
    root = match.group("root")
    num = int(match.group("num"))
    tail = match.group("tail")
    lo = max(1, int(round(num * (1 - pct))))
    hi = max(lo + 1, int(round(num * (1 + pct))))
    if lo == num:
        lo = max(1, num - 1)
    if hi == num:
        hi = num + 1
    out: list[str] = []
    if lo != num:
        out.append(f"{root}{lo}{tail}")
    if hi != num:
        out.append(f"{root}{hi}{tail}")
    return out


# ---------------------------------------------------------------------------
# Sibling enumeration
# ---------------------------------------------------------------------------


def _enumerate_siblings(
    candidate: dict[str, Any],
    *,
    include_aperture: bool,
    include_session: bool,
    session_cap: int,
    threshold_pct: float,
) -> list[dict[str, Any]]:
    """Build the list of sibling spec dicts. Excludes the candidate itself."""
    base = dict(candidate)
    siblings: list[dict[str, Any]] = []

    # Threshold neighbours (same session, same aperture).
    filter_type = str(base.get("filter_type") or "")
    for nt in _threshold_neighbours(filter_type, pct=threshold_pct):
        s = dict(base)
        s["filter_type"] = nt
        s["_axis"] = "threshold"
        siblings.append(s)

    # Aperture neighbours (same session, same filter).
    if include_aperture:
        try:
            current_apt = int(base.get("orb_minutes") or 0)
        except (TypeError, ValueError):
            current_apt = 0
        for apt in _APERTURE_NEIGHBOURS:
            if apt == current_apt:
                continue
            s = dict(base)
            s["orb_minutes"] = apt
            s["_axis"] = "aperture"
            siblings.append(s)

    # Session neighbours (same instrument, same filter, same aperture).
    if include_session:
        current_session = str(base.get("orb_label") or base.get("session") or "")
        canonical = _canonical_sessions()
        for sess in canonical[:session_cap]:
            if sess == current_session:
                continue
            s = dict(base)
            s["orb_label"] = sess
            s["_axis"] = "session"
            siblings.append(s)

    return siblings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_neighbors(
    candidate: dict[str, Any],
    *,
    db_path: str | None = None,
    threshold_pct: float = 0.20,
    include_aperture: bool = True,
    include_session: bool = False,
    session_cap: int = 6,
    max_total_siblings: int = 12,
) -> dict[str, Any]:
    """Score adjacent cells around the candidate.

    By default the scan covers threshold (±20%) and aperture axes; session
    axis is opt-in because it multiplies the Mode A SQL hits by ~12 and
    pushes wall time over 30s. The LLM prompt can call with
    ``include_session=True`` when family-wide context matters more than
    speed.

    Parameters
    ----------
    candidate
        validated_setups-shaped dict. Needs at least instrument, orb_label,
        orb_minutes, entry_model, confirm_bars, rr_target, filter_type,
        expectancy_r.
    threshold_pct
        Fraction for threshold neighbours (default 0.20 = ±20% per
        RESEARCH_RULES.md § Sensitivity Analysis).
    include_aperture, include_session, session_cap, max_total_siblings
        Scope controls.

    Returns
    -------
    dict with:
        - ``siblings``: list of per-sibling reports
          {spec, axis, mode_a, graveyard_blocking, graveyard_warning, summary}
        - ``siblings_tested``: count after capping
        - ``siblings_killed``: count where Mode A failed (passes_criterion_8=False)
        - ``siblings_blocked_by_graveyard``: count with prior NO-GO/KILL/DEAD
        - ``family_health``: aggregate label
          - ``HOSTILE`` if ≥60% siblings_killed or any graveyard block
          - ``MIXED`` if ≥30%
          - ``CLEAN`` otherwise
        - ``summary``: one-line log message
    """
    from scripts.research.lhp.adjacency import screen_candidate_mode_a
    from scripts.research.lhp.graveyard import check_graveyard

    siblings = _enumerate_siblings(
        candidate,
        include_aperture=include_aperture,
        include_session=include_session,
        session_cap=session_cap,
        threshold_pct=threshold_pct,
    )
    siblings = siblings[:max_total_siblings]

    # Look up which filters are actually registered so we don't count
    # synthetic threshold variants (e.g. ATR_P24 when only ATR_P30/P50/P70
    # exist in ALL_FILTERS) as Mode-A failures. Those are *unknown to the
    # platform*, not *known to be dead* — different signal.
    try:
        from trading_app.config import ALL_FILTERS  # type: ignore[import-not-found]

        registered_filters = frozenset(ALL_FILTERS.keys())
    except ImportError:
        logger.warning("neighbor_scan: ALL_FILTERS unavailable; treating all neighbours as registered")
        registered_filters = frozenset()

    reports: list[dict[str, Any]] = []
    killed = 0
    gv_blocked = 0
    unregistered = 0
    for sib in siblings:
        axis = sib.pop("_axis", "?")
        sib_filter = str(sib.get("filter_type") or "")
        is_registered = (not registered_filters) or sib_filter in registered_filters

        mode_a: dict[str, Any]
        if not is_registered:
            # Skip Mode A — would just return "filter_type not in registry".
            # Mark as informational-only so family_health isn't skewed.
            mode_a = {
                "passes_criterion_8": None,
                "reason": f"not_in_registry: {sib_filter!r} not in ALL_FILTERS",
                "oos_is_ratio": None,
                "n_oos": None,
            }
            unregistered += 1
        else:
            try:
                mode_a = screen_candidate_mode_a(sib, db_path=db_path, strict_oos_n=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("neighbor_scan: Mode A failed for %s: %r", sib_filter, exc)
                mode_a = {"passes_criterion_8": None, "reason": f"mode_a_unavailable: {exc!r}"}
        try:
            gv = check_graveyard(sib)
        except Exception as exc:  # noqa: BLE001
            logger.warning("neighbor_scan: graveyard failed for %s: %r", sib_filter, exc)
            gv = {"has_blocking_verdict": False, "has_warning": False, "summary": f"graveyard_unavailable: {exc!r}"}

        passes = mode_a.get("passes_criterion_8")
        if passes is False:
            killed += 1
        if gv.get("has_blocking_verdict"):
            gv_blocked += 1
        reports.append(
            {
                "spec": {
                    "instrument": sib.get("instrument"),
                    "orb_label": sib.get("orb_label"),
                    "orb_minutes": sib.get("orb_minutes"),
                    "filter_type": sib_filter,
                    "rr_target": sib.get("rr_target"),
                    "entry_model": sib.get("entry_model"),
                    "confirm_bars": sib.get("confirm_bars"),
                },
                "axis": axis,
                "in_registry": is_registered,
                "mode_a_passes": passes,
                "mode_a_reason": mode_a.get("reason"),
                "mode_a_oos_is_ratio": mode_a.get("oos_is_ratio"),
                "mode_a_n_oos": mode_a.get("n_oos"),
                "graveyard_blocking": gv.get("has_blocking_verdict"),
                "graveyard_warning": gv.get("has_warning"),
                "graveyard_summary": gv.get("summary"),
            }
        )

    # family_health denominator excludes unregistered neighbours so a
    # threshold-axis sweep over ATR_P24/P36 (neither registered) doesn't
    # claim HOSTILE when only registered apertures were tested.
    tested = len(reports)
    evaluable = tested - unregistered
    if evaluable == 0:
        family_health = "UNKNOWN_NO_EVALUABLE_SIBLINGS"
    else:
        kill_pct = killed / evaluable
        if kill_pct >= 0.60 or gv_blocked > 0:
            family_health = "HOSTILE"
        elif kill_pct >= 0.30:
            family_health = "MIXED"
        else:
            family_health = "CLEAN"

    summary = (
        f"{family_health}: {tested} siblings enumerated "
        f"({evaluable} evaluable, {unregistered} not in registry); "
        f"{killed} fail Mode A, {gv_blocked} blocked by graveyard"
    )

    return {
        "siblings": reports,
        "siblings_tested": tested,
        "siblings_evaluable": evaluable,
        "siblings_unregistered": unregistered,
        "siblings_killed": killed,
        "siblings_blocked_by_graveyard": gv_blocked,
        "family_health": family_health,
        "summary": summary,
    }


__all__ = [
    "scan_neighbors",
]
