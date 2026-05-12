"""Graveyard check — auto-detect prior KILL/PARK/NO-GO verdicts on a candidate.

Two-layer architecture:

  1. **Primary:** delegate to canonical
     ``scripts.tools.research_catalog_mcp_server._search_research_catalog``,
     the same function the ``/nogo`` skill and the research-catalog MCP
     surface. It already indexes:
       - ``docs/audit/results/*.md`` (front-matter ``verdict:`` field, body
         ``**Verdict:**`` line, filename suffix)
       - ``docs/STRATEGY_BLUEPRINT.md`` §5 NO-GO Registry (with reopen
         criteria)
       - pre-reg hypotheses tagged with kill verdicts
       - literature sources
     Returns scored, verdict-tagged hits with ``reopen_criteria`` fields.

  2. **Fallback:** narrow file-grep across ``docs/audit/results/*.md`` and
     the auto-memory ``nogo_*.md`` files. Only fires when the canonical
     module fails to import (CI / hermetic test). Same-file-verdict +
     filter-token + session/instrument required; no broader matching.

Caller is the LLM hypothesis proposer; it must not silently swallow MCP
errors — log + degrade to fallback, never claim CLEAR on a failure.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical verdict tag set — same list as ``.claude/commands/nogo.md``.
_BLOCKING_VERDICTS: frozenset[str] = frozenset({"NO-GO", "KILL", "DEAD"})
_WARNING_VERDICTS: frozenset[str] = frozenset({"PARK", "REJECTED", "UNSUPPORTED", "DECAY", "STALE"})
_VERDICT_TAGS = sorted(_BLOCKING_VERDICTS | _WARNING_VERDICTS)


# ---------------------------------------------------------------------------
# Filter-family normalisation (used to build the search query AND to label
# fallback hits — siblings cross-match because we query the family root).
# ---------------------------------------------------------------------------

_FAMILY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^ATR_P\d+$"), "ATR_P"),
    (re.compile(r"^ATR_VEL_GE\d+$"), "ATR_VEL"),
    (re.compile(r"^ORB_VOL_\d+K?$"), "ORB_VOL"),
    (re.compile(r"^ORB_G\d+$"), "ORB_G"),
    (re.compile(r"^COST_LT\d+$"), "COST_LT"),
    (re.compile(r"^OVNRNG_\d+$"), "OVNRNG"),
    (re.compile(r"^VOL_RV\d*$"), "VOL_RV"),
    (re.compile(r"^VWAP_.*$"), "VWAP"),
)


def normalise_filter_family(filter_type: str) -> str:
    """Strip threshold tokens to reveal the family root.

    >>> normalise_filter_family("ATR_P30")
    'ATR_P'
    >>> normalise_filter_family("ORB_VOL_16K")
    'ORB_VOL'
    >>> normalise_filter_family("CUSTOM")
    'CUSTOM'
    """
    for pattern, family in _FAMILY_PATTERNS:
        if pattern.match(filter_type):
            return family
    return filter_type


def _candidate_query(candidate: dict[str, Any]) -> str:
    """Build the search query the way ``/nogo`` would be invoked manually."""
    instrument = str(candidate.get("instrument") or candidate.get("symbol") or "").upper()
    session = str(candidate.get("orb_label") or candidate.get("session") or "").upper()
    filter_type = str(candidate.get("filter_type") or "").upper()
    family = normalise_filter_family(filter_type)
    parts = [t for t in (instrument, session, filter_type, family) if t]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return " ".join(uniq)


def _candidate_tokens(candidate: dict[str, Any]) -> dict[str, str]:
    instrument = str(candidate.get("instrument") or candidate.get("symbol") or "").upper()
    session = str(candidate.get("orb_label") or candidate.get("session") or "").upper()
    filter_type = str(candidate.get("filter_type") or "").upper()
    family = normalise_filter_family(filter_type)
    return {
        "instrument": instrument,
        "session": session,
        "filter_type": filter_type,
        "family": family,
    }


# ---------------------------------------------------------------------------
# Primary path — canonical research-catalog
# ---------------------------------------------------------------------------


def _query_canonical_catalog(
    candidate: dict[str, Any],
    *,
    limit: int = 10,
) -> tuple[list[dict[str, Any]], str | None]:
    """Call the canonical search function. Returns (items, error_message).

    On any import or runtime failure, returns ([], reason). The caller is
    expected to degrade to the file-grep fallback and emit a WARN log so
    no candidate is silently passed as CLEAR after a backend failure.
    """
    try:
        from scripts.tools.research_catalog_mcp_server import _search_research_catalog
    except ImportError as exc:  # pragma: no cover - environment-specific
        return [], f"canonical_import_failed: {exc!r}"

    query = _candidate_query(candidate)
    if not query:
        return [], "empty_query"

    try:
        out = _search_research_catalog(
            query=query,
            verdict_tags=_VERDICT_TAGS,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 — explicit logging, no silent pass
        return [], f"canonical_query_failed: {exc!r}"

    items = out.get("items", []) if isinstance(out, dict) else []
    return list(items), None


def _classify_canonical_hit(item: dict[str, Any], tokens: dict[str, str]) -> dict[str, Any]:
    """Filter a canonical catalogue item to keep only fields the proposer needs.

    Adds an ``applies_to_candidate`` heuristic: catalogue hits often surface
    NO-GO items that are tangentially relevant (the query terms appeared in
    the same file). We mark a hit as ``applies_to_candidate=True`` only if
    the title/snippet/path contains BOTH the candidate's family token AND
    its session token. Other hits stay informational.
    """
    blob_parts: list[str] = []
    for key in ("title", "summary", "snippet", "path", "section"):
        val = item.get(key)
        if isinstance(val, str):
            blob_parts.append(val)
    blob = " ".join(blob_parts).upper()

    family_hit = bool(tokens["family"]) and tokens["family"] in blob
    filter_hit = bool(tokens["filter_type"]) and tokens["filter_type"] in blob
    session_hit = bool(tokens["session"]) and tokens["session"] in blob
    instrument_hit = bool(tokens["instrument"]) and tokens["instrument"] in blob

    return {
        "source": "research_catalog",
        "verdict": str(item.get("verdict") or "?").upper(),
        "title": item.get("title", "?"),
        "path": item.get("path", "?"),
        "date": item.get("date"),
        "score": item.get("score"),
        "reopen_criteria": item.get("reopen_criteria"),
        "snippet": (item.get("snippet") or "")[:240],
        "applies_to_candidate": (filter_hit or family_hit) and (session_hit or instrument_hit),
        "_match_flags": {
            "filter": filter_hit,
            "family": family_hit,
            "session": session_hit,
            "instrument": instrument_hit,
        },
    }


# ---------------------------------------------------------------------------
# Fallback — narrow file-grep (used only when the canonical path fails)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUDIT_RESULTS_DIR_DEFAULT = _REPO_ROOT / "docs" / "audit" / "results"
_MEMORY_DIR_DEFAULT = Path.home() / ".claude" / "projects" / "C--Users-joshd-canompx3" / "memory"


def _resolve_memory_dir() -> Path:
    override = os.environ.get("CANOMPX3_MEMORY_DIR")
    return Path(override) if override else _MEMORY_DIR_DEFAULT


_VERDICT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bNO[-\s_]?GO\b", re.IGNORECASE), "NO-GO"),
    (re.compile(r"\bKILL(?:ED)?\b", re.IGNORECASE), "KILL"),
    (re.compile(r"\bDEAD\b", re.IGNORECASE), "DEAD"),
    (re.compile(r"\bPARK(?:ED)?\b", re.IGNORECASE), "PARK"),
    (re.compile(r"\bREJECT(?:ED)?\b", re.IGNORECASE), "REJECTED"),
)


def _classify_verdict(line: str) -> str | None:
    for pattern, label in _VERDICT_PATTERNS:
        if pattern.search(line):
            return label
    return None


def _scan_file_fallback(
    path: Path,
    tokens: dict[str, str],
    *,
    source_label: str,
) -> list[dict[str, Any]]:
    """Narrow fallback — file must contain filter+session+verdict, return
    at most 3 representative lines."""
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    text_upper = text.upper()
    filter_token = tokens["filter_type"]
    family_token = tokens["family"]
    session_token = tokens["session"]
    instrument_token = tokens["instrument"]
    if not ((filter_token and filter_token in text_upper) or (family_token and family_token in text_upper)):
        return []
    if not ((session_token and session_token in text_upper) or (instrument_token and instrument_token in text_upper)):
        return []
    file_verdict: str | None = None
    for pattern, label in _VERDICT_PATTERNS:
        if pattern.search(text):
            file_verdict = label
            break
    if file_verdict is None:
        return []
    hits: list[dict[str, Any]] = []
    for idx, line in enumerate(text.splitlines()):
        line_upper = line.upper()
        if (filter_token and filter_token in line_upper) or (family_token and family_token in line_upper):
            hits.append(
                {
                    "source": source_label,
                    "verdict": file_verdict,
                    "path": str(path),
                    "line_no": idx + 1,
                    "matched_line": line.strip()[:240],
                    "applies_to_candidate": True,
                }
            )
            if len(hits) >= 3:
                break
    return hits


def _fallback_scan(
    candidate: dict[str, Any],
    *,
    audit_results_dir: Path | None,
    memory_dir: Path | None,
) -> list[dict[str, Any]]:
    tokens = _candidate_tokens(candidate)
    audit_dir = audit_results_dir or _AUDIT_RESULTS_DIR_DEFAULT
    mem_dir = memory_dir or _resolve_memory_dir()
    hits: list[dict[str, Any]] = []
    if audit_dir.is_dir():
        for path in sorted(audit_dir.glob("*.md"))[:500]:
            hits.extend(_scan_file_fallback(path, tokens, source_label="audit_result_fallback"))
    if mem_dir.is_dir():
        for path in sorted(mem_dir.glob("nogo_*.md"))[:200]:
            hits.extend(_scan_file_fallback(path, tokens, source_label="memory_nogo_fallback"))
    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_graveyard(
    candidate: dict[str, Any],
    *,
    audit_results_dir: Path | None = None,
    memory_dir: Path | None = None,
    catalog_limit: int = 10,
) -> dict[str, Any]:
    """Return prior-verdict signals for the candidate's (instrument, session, family).

    Always tries the canonical research-catalog first. Only falls back to
    file-grep if the canonical module fails to import or raises. The fallback
    branch is marked in the return so the caller can log degraded-mode.

    Returns
    -------
    dict with:
        - ``has_blocking_verdict``: True if any catalogue hit with verdict in
          {NO-GO, KILL, DEAD} ALSO ``applies_to_candidate`` (family+session
          co-occur in the hit text).
        - ``has_warning``: True if there are any warning-class hits or any
          non-applying blocking hits (manual review nudge).
        - ``hits``: list of hit dicts (sorted blocking-first, then by score).
        - ``family``: the normalised filter family used in the search.
        - ``backend``: ``"research_catalog"`` or ``"fallback_grep"``.
        - ``backend_error``: None on success, else the reason the canonical
          path declined.
        - ``summary``: one-line human-readable description.
    """
    tokens = _candidate_tokens(candidate)
    items, err = _query_canonical_catalog(candidate, limit=catalog_limit)
    if err is not None:
        logger.warning("graveyard canonical path failed (%s); degrading to file-grep", err)
        fallback_hits = _fallback_scan(
            candidate,
            audit_results_dir=audit_results_dir,
            memory_dir=memory_dir,
        )
        blocking = any(h["verdict"] in _BLOCKING_VERDICTS for h in fallback_hits)
        warning = any(h["verdict"] in _WARNING_VERDICTS for h in fallback_hits) or (
            not blocking and bool(fallback_hits)
        )
        return {
            "has_blocking_verdict": blocking,
            "has_warning": warning,
            "hits": fallback_hits,
            "family": tokens["family"],
            "backend": "fallback_grep",
            "backend_error": err,
            "summary": _build_summary(tokens, blocking, warning, len(fallback_hits)),
        }

    classified = [_classify_canonical_hit(item, tokens) for item in items]
    severity = {"NO-GO": 0, "KILL": 1, "DEAD": 2, "REJECTED": 3, "UNSUPPORTED": 4, "DECAY": 5, "STALE": 6, "PARK": 7}
    classified.sort(key=lambda h: (severity.get(h["verdict"], 99), -(h.get("score") or 0)))

    applying = [h for h in classified if h["applies_to_candidate"]]
    blocking = any(h["verdict"] in _BLOCKING_VERDICTS for h in applying)
    warning = (not blocking and any(h["verdict"] in _WARNING_VERDICTS for h in applying)) or (
        not blocking and bool(classified) and any(h["verdict"] in _BLOCKING_VERDICTS for h in classified)
    )

    return {
        "has_blocking_verdict": blocking,
        "has_warning": warning,
        "hits": classified,
        "family": tokens["family"],
        "backend": "research_catalog",
        "backend_error": None,
        "summary": _build_summary(tokens, blocking, warning, len(applying)),
    }


def _build_summary(tokens: dict[str, str], blocking: bool, warning: bool, n: int) -> str:
    descr = f"{tokens['instrument']} {tokens['session']} {tokens['family']}"
    if blocking:
        return f"BLOCKING: {n} candidate-applying verdict(s) on {descr}"
    if warning:
        return f"WARN: prior verdicts on adjacent cells of {descr} (manual review)"
    return f"CLEAR: no prior verdicts found applying to {descr}"


__all__ = [
    "check_graveyard",
    "normalise_filter_family",
]
