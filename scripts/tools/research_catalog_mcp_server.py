"""MCP server for repo-local research grounding and audit artifact lookup.

Exposes read-only tools via stdio (fastmcp):
  - list_literature_sources: list committed local literature extracts
  - get_literature_excerpt: return a bounded excerpt from a literature source
  - list_open_hypotheses: list prereg artifacts without an exact result-stem match
  - get_hypothesis_artifact: return a bounded hypothesis artifact payload
  - get_audit_result: return a bounded audit-result payload
  - search_research_catalog: search literature, hypotheses, and result docs;
    optional verdict_tags filter (NO-GO/PARK/KILL/UNSUPPORTED/DECAY/STALE)
    surfaces kill verdicts from filename stems, markdown front-matter, and
    docs/STRATEGY_BLUEPRINT.md §5 NO-GO Registry table rows
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ArtifactKind = Literal["literature", "hypothesis", "result"]
VALID_SCOPES: tuple[ArtifactKind, ...] = ("literature", "hypothesis", "result")

LITERATURE_ROOT = PROJECT_ROOT / "docs" / "institutional" / "literature"
HYPOTHESES_ROOT = PROJECT_ROOT / "docs" / "audit" / "hypotheses"
RESULTS_ROOT = PROJECT_ROOT / "docs" / "audit" / "results"
BLUEPRINT_PATH = PROJECT_ROOT / "docs" / "STRATEGY_BLUEPRINT.md"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Recognized verdict tags. NO-GO equivalents (DEAD, PURGED, RETIRED, REJECTED)
# normalize to NO-GO so callers can filter by canonical kill verdict without
# enumerating every legacy spelling. PARK / KILL / UNSUPPORTED / DECAY / STALE
# keep their raw form because they map 1:1 to the project's audit-result
# vocabulary (RESEARCH_RULES.md verdict tokens).
# Verdict normalization map.
#
# Source-of-truth note: project verdict vocabulary is informally distributed
# across docs/STRATEGY_BLUEPRINT.md §5 (NO-GO Registry verdicts) and
# docs/audit/results/*.md headers. There is no formal canonical list in
# RESEARCH_RULES.md or pre_registered_criteria.md as of 2026-05-12. This
# server is the de-facto verdict registry — tracked as a follow-up to
# formalize a doctrine source. Until then, additions here MUST be backed
# by `grep` evidence of production usage (not invented).
#
# NO-GO equivalents (DEAD/PURGED/RETIRED/REJECTED/REFUTED/GUILTY/INVESTIGATED/
# ARITHMETIC_ONLY) collapse to NO-GO so callers filter by canonical kill
# verdict without enumerating every legacy spelling. KILL / PARK /
# UNSUPPORTED / DECAY / STALE / NULL keep their raw form because they map
# 1:1 to project audit-result tokens. Non-kill outcomes (VALIDATED /
# CONFIRMED / CONTINUE / DOWNSIZE / HOLDING / PROMISING / MARGINAL / PASS /
# CONDITIONAL / EDGE_WITH_CAVEAT) are recognized so the body-line scanner
# does not misclassify them as kill — they normalize to themselves.
_VERDICT_NORMALIZE = {
    # Kill class -> NO-GO
    "NO-GO": "NO-GO",
    "NOGO": "NO-GO",
    "DEAD": "NO-GO",
    "PURGED": "NO-GO",
    "RETIRED": "NO-GO",
    "REJECTED": "NO-GO",
    "REFUTED": "NO-GO",
    "GUILTY": "NO-GO",
    "INVESTIGATED": "NO-GO",
    "ARITHMETIC_ONLY": "NO-GO",
    "ARITHMETIC-ONLY": "NO-GO",
    # Distinct kill-flavor outcomes
    "PARK": "PARK",
    "PARK_PENDING_OOS_POWER": "PARK",
    "PARKPENDINGOOSPOWER": "PARK",
    "KILL": "KILL",
    "KILL_DOWNGRADE": "KILL",
    "KILLDOWNGRADE": "KILL",
    "UNSUPPORTED": "UNSUPPORTED",
    "DECAY": "DECAY",
    "STALE": "STALE",
    "NULL": "NULL",
    # Non-kill outcomes (recognized so scanner doesn't fall through to a
    # later "DEAD" qualifier in narrative text — e.g., "VALIDATED but DEAD
    # in OOS" must read as VALIDATED, not NO-GO).
    "PROMISING": "PROMISING",
    "VALIDATED": "VALIDATED",
    "CONFIRMED": "CONFIRMED",
    "CONDITIONAL": "CONDITIONAL",
    "CONTINUE": "CONTINUE",
    "DOWNSIZE": "DOWNSIZE",
    "HOLDING": "HOLDING",
    "MARGINAL": "MARGINAL",
    "PASS": "PASS",
    "EDGE_WITH_CAVEAT": "EDGE_WITH_CAVEAT",
    "EDGEWITHCAVEAT": "EDGE_WITH_CAVEAT",
    "RESEARCH_PROVISIONAL": "RESEARCH_PROVISIONAL",
    "RESEARCHPROVISIONAL": "RESEARCH_PROVISIONAL",
    "WEAK": "WEAK",
    "REDESIGN": "REDESIGN",
    "FIX": "FIX",
    "CLOSED": "CLOSED",
}
# Canonical tags exposed to callers via verdict_tags. Excludes the underlying
# raw aliases (DEAD, PURGED, etc.) — callers filter by canonical tag only.
VALID_VERDICT_TAGS: frozenset[str] = frozenset(_VERDICT_NORMALIZE.values())

# Tokens scanned in priority order when a verdict cell carries qualifiers
# (e.g., "DEAD (impractical)", "INVESTIGATED — NO-GO", "**DEAD — PERMANENT**").
# Order matters: kill-verdict tokens come BEFORE neutral / non-kill ones so
# "INVESTIGATED — NO-GO" resolves to NO-GO (kill wins), and so a compound
# verdict like "**DEAD — PERMANENT**" cannot accidentally hit "PERMANENT".
# Within the kill class, more-specific spellings come first.
_VERDICT_PRIORITY_TOKENS = (
    "NO-GO",
    "NOGO",
    "DEAD",
    "PURGED",
    "RETIRED",
    "REJECTED",
    "REFUTED",
    "KILL_DOWNGRADE",
    "KILLDOWNGRADE",
    "KILL",
    "PARK_PENDING_OOS_POWER",
    "PARKPENDINGOOSPOWER",
    "PARK",
    "UNSUPPORTED",
    "DECAY",
    "STALE",
    "NULL",
    "GUILTY",
    "ARITHMETIC_ONLY",
    "ARITHMETIC-ONLY",
    "INVESTIGATED",
    "EDGE_WITH_CAVEAT",
    "EDGEWITHCAVEAT",
    "RESEARCH_PROVISIONAL",
    "RESEARCHPROVISIONAL",
    "VALIDATED",
    "CONFIRMED",
    "CONDITIONAL",
    "CONTINUE",
    "DOWNSIZE",
    "HOLDING",
    "MARGINAL",
    "PROMISING",
    "REDESIGN",
    "PASS",
    "FIX",
    "CLOSED",
    "WEAK",
)

_VERDICT_FILENAME_RE = re.compile(
    r"-(nogo|dead|park|kill|unsupported|decay|stale)$",
    re.IGNORECASE,
)
# YAML-style front-matter / inline scalar marker.
# Matches:  ``verdict: NO-GO`` , ``  verdict: "KILL"``
_VERDICT_FRONTMATTER_RE = re.compile(
    r"^\s*verdict\s*:\s*[\"']?([A-Za-z\-_]+)[\"']?\s*$",
    re.MULTILINE,
)
# Body-line marker as written across docs/audit/results/. Production format
# is ``**Verdict:** KILL per K1.`` or ``**VERDICT: KILL**`` — the prefix
# carries markdown bold markers and the value is followed by free text the
# regex must NOT capture. We extract the FIRST whitespace-delimited token
# after the prefix (stripping leading punctuation / backticks / bold), and
# defer normalization to ``_normalize_verdict`` which handles aliases.
#
# Verified 2026-05-12 against 44 production audit-result files using the
# ``^**Verdict...:**`` family of prefixes (`grep -E "^\*\*[Vv]erdict[^:]*:\*\*"`).
_VERDICT_BODY_RE = re.compile(
    r"""
    ^                           # line start
    [\s>*_`-]*                  # optional list/quote/bold leaders
    [Vv][Ee][Rr][Dd][Ii][Cc][Tt]
    [^\n:]*                     # optional qualifier ("Verdict on X")
    :                           # colon delimiter
    [\s*_`]*                    # eat trailing bold/backticks before value
    ([A-Za-z][A-Za-z0-9_\-]*)   # the verdict token (first word)
    """,
    re.MULTILINE | re.VERBOSE,
)
_BLUEPRINT_NOGO_HEADING = "5. NO-GO Registry"


@dataclass(frozen=True)
class Artifact:
    kind: ArtifactKind
    artifact_id: str
    path: Path
    title: str
    summary: str
    date: str | None
    metadata: dict[str, object]
    text: str
    search_blob: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]{2,}", text.lower())


def _bounded(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _extract_markdown_meta(text: str, label: str) -> str | None:
    match = re.search(rf"^\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_yaml_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"^\s{{2}}{re.escape(key)}:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return value.strip("\"'") if value else None


def _extract_first_paragraph(text: str) -> str:
    lines = text.splitlines()
    paragraph_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if paragraph_lines:
                break
            continue
        if line.startswith("#"):
            continue
        if line.startswith("**") and line.endswith("**"):
            continue
        if line.startswith("**") and ":" in line:
            continue
        if line.startswith(">"):
            continue
        if line.startswith("|"):
            continue
        if line.startswith("<!--"):
            continue
        paragraph_lines.append(line)
    return " ".join(paragraph_lines).strip()


def _normalize_verdict(raw: str | None) -> str | None:
    """Normalize a verdict cell to a canonical tag.

    Handles three shapes:
    1. Bare canonical token: ``"NO-GO"`` -> ``"NO-GO"``.
    2. Bare alias: ``"DEAD"`` -> ``"NO-GO"``, ``"REJECTED"`` -> ``"NO-GO"``.
    3. Compound qualifier: ``"DEAD (reversed)"``, ``"INVESTIGATED -- NO-GO"``,
       ``"**DEAD -- PERMANENT**"`` -> scans for the first known token in
       :data:`_VERDICT_PRIORITY_TOKENS` order; the most-specific token wins.

    Returns ``None`` only when the input is empty / unparseable. Callers
    that get ``None`` should treat the row as "no verdict declared" — not
    a silent kill verdict. Adding a new spelling = extend
    :data:`_VERDICT_NORMALIZE` and :data:`_VERDICT_PRIORITY_TOKENS`.
    """
    if not raw:
        return None
    cleaned = re.sub(r"\*+", "", raw).strip()
    if not cleaned:
        return None
    upper = cleaned.upper()
    # Exact-match fast path (covers bare tokens with no qualifier).
    bare = upper.replace("_", "-")
    if bare in _VERDICT_NORMALIZE:
        return _VERDICT_NORMALIZE[bare]
    # Token scan for compound verdicts. Word-boundary regex prevents
    # partial-word matches (e.g., 'PARK' should not fire inside 'SPARK').
    for token in _VERDICT_PRIORITY_TOKENS:
        pattern = rf"(?<![A-Z]){re.escape(token)}(?![A-Z])"
        if re.search(pattern, upper):
            return _VERDICT_NORMALIZE[token]
    return None


def _detect_verdict(stem: str, text: str) -> str | None:
    """Detect verdict from front-matter, body marker, or filename suffix.

    Precedence (high -> low):
        1. YAML front-matter / inline ``verdict: <X>`` scalar
        2. Body marker ``**Verdict:** <TOKEN>`` (canonical form across
           docs/audit/results/ — verified 2026-05-12 against 44 prod files)
        3. Filename slug suffix (``*-nogo.md`` / ``*-park.md`` / etc.)

    Explicit declarations beat conventions. Body-marker scan walks every
    matching line and returns the FIRST token that ``_normalize_verdict``
    recognizes — handles audit-results that carry a `Verdict on X:` or
    `Verdict trace:` informational header before the load-bearing one.
    """
    fm_match = _VERDICT_FRONTMATTER_RE.search(text)
    if fm_match:
        verdict = _normalize_verdict(fm_match.group(1))
        if verdict is not None:
            return verdict
    for body_match in _VERDICT_BODY_RE.finditer(text):
        verdict = _normalize_verdict(body_match.group(1))
        if verdict is not None:
            return verdict
    name_match = _VERDICT_FILENAME_RE.search(stem)
    if name_match:
        return _normalize_verdict(name_match.group(1))
    return None


def _split_table_row(line: str) -> list[str]:
    """Split a markdown table row into cell text, stripping leading/trailing pipes."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def _slugify(text: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return base or "row"


def _parse_blueprint_nogo_artifacts(path: Path) -> list[Artifact]:
    """Parse `## 5. NO-GO Registry` table rows into per-row Artifacts.

    Each table data row (skipping the header + separator) becomes one
    `result`-kind artifact tagged `metadata["verdict"]` derived from the
    Verdict column. Returns [] if the file or section is absent — the
    caller treats this as advisory, not fatal.
    """
    if not path.exists():
        return []
    text = _read_text(path)
    windows = _heading_windows(text)
    section_text: str | None = None
    for title, start, end in windows:
        if _normalized(title) == _normalized(_BLUEPRINT_NOGO_HEADING):
            section_text = text[start:end]
            break
    if not section_text:
        return []

    artifacts: list[Artifact] = []
    seen_header = False
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = _split_table_row(stripped)
        # First pipe-row is the header; second is the |---|---| separator.
        if not seen_header:
            seen_header = True
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if len(cells) < 2:
            continue
        path_label = cells[0]
        verdict_raw = cells[1]
        evidence = cells[2] if len(cells) > 2 else ""
        reopen = cells[3] if len(cells) > 3 else ""
        verdict = _normalize_verdict(verdict_raw)
        if verdict is None:
            continue
        clean_label = re.sub(r"\*+", "", path_label).strip()
        slug = _slugify(clean_label)
        artifact_id = f"blueprint-nogo-{slug}"
        title = f"BLUEPRINT NO-GO: {clean_label}"
        summary = _bounded(evidence, 400) if evidence else verdict_raw
        body_blob = " | ".join([path_label, verdict_raw, evidence, reopen])
        metadata = {
            "verdict": verdict,
            "source": "STRATEGY_BLUEPRINT.md",
            "section": "5. NO-GO Registry",
            "path": _display_path(path),
            "raw_verdict": verdict_raw,
            "reopen_criteria": _bounded(reopen, 300) if reopen else None,
        }
        search_blob = " ".join(filter(None, [title, summary, body_blob]))
        artifacts.append(
            Artifact(
                kind="result",
                artifact_id=artifact_id,
                path=path,
                title=title,
                summary=summary,
                date=None,
                metadata=metadata,
                text=body_blob,
                search_blob=search_blob,
            )
        )
    return artifacts


def _heading_windows(text: str) -> list[tuple[str, int, int]]:
    matches = list(HEADING_RE.finditer(text))
    windows: list[tuple[str, int, int]] = []
    for index, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        windows.append((title, start, end))
    return windows


def _section_excerpt(text: str, section_hint: str | None, max_chars: int) -> str:
    if section_hint:
        hint = _normalized(section_hint)
        for title, start, end in _heading_windows(text):
            if hint and hint in _normalized(title):
                return _bounded(text[start:end], max_chars)
    return _bounded(text, max_chars)


def _snippet_for_search(text: str, query_terms: set[str], max_chars: int = 220) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) < 20:
            continue
        lowered = line.lower()
        if any(term in lowered for term in query_terms):
            return _bounded(line, max_chars)
    return _bounded(_extract_first_paragraph(text), max_chars)


def _parse_literature(path: Path) -> Artifact:
    text = _read_text(path)
    title = _first_heading(text) or path.stem
    summary = _extract_first_paragraph(text)
    source = _extract_markdown_meta(text, "Source")
    authors = _extract_markdown_meta(text, "Authors")
    date = _extract_markdown_meta(text, "Date")
    criticality = _extract_markdown_meta(text, "Criticality")
    metadata = {
        "source": source,
        "authors": authors,
        "criticality": criticality,
        "path": _display_path(path),
    }
    search_blob = " ".join(filter(None, [title, summary, source, authors, criticality, text[:8000]]))
    return Artifact(
        kind="literature",
        artifact_id=path.stem,
        path=path,
        title=title,
        summary=summary,
        date=date,
        metadata=metadata,
        text=text,
        search_blob=search_blob,
    )


def _parse_hypothesis(path: Path, result_stems: set[str]) -> Artifact:
    text = _read_text(path)
    summary = ""
    date = None
    metadata: dict[str, object]
    if path.suffix == ".yaml":
        title = _extract_yaml_scalar(text, "name") or path.stem
        summary = _extract_yaml_scalar(text, "purpose") or ""
        date = _extract_yaml_scalar(text, "date_locked")
        metadata = {
            "holdout_date": _extract_yaml_scalar(text, "holdout_date"),
            "testing_mode": _extract_yaml_scalar(text, "testing_mode"),
            "total_expected_trials": _extract_yaml_scalar(text, "total_expected_trials"),
            "path": _display_path(path),
        }
    else:
        title = _first_heading(text) or path.stem
        summary = _extract_first_paragraph(text)
        metadata = {"path": _display_path(path)}
    metadata["exact_result_match"] = path.stem in result_stems
    metadata["open_status_basis"] = "No exact filename stem match in docs/audit/results."
    search_blob = " ".join(filter(None, [title, summary, text[:8000]]))
    return Artifact(
        kind="hypothesis",
        artifact_id=path.stem,
        path=path,
        title=title,
        summary=summary,
        date=date,
        metadata=metadata,
        text=text,
        search_blob=search_blob,
    )


def _parse_result(path: Path) -> Artifact:
    text = _read_text(path)
    title = _first_heading(text) or path.stem
    summary = _extract_first_paragraph(text)
    date = _extract_markdown_meta(text, "Date")
    prereg = _extract_markdown_meta(text, "Pre-registration")
    verdict = _detect_verdict(path.stem, text)
    metadata: dict[str, object] = {
        "pre_registration": prereg,
        "path": _display_path(path),
        "verdict": verdict,
    }
    search_blob = " ".join(filter(None, [title, summary, prereg, text[:8000]]))
    return Artifact(
        kind="result",
        artifact_id=path.stem,
        path=path,
        title=title,
        summary=summary,
        date=date,
        metadata=metadata,
        text=text,
        search_blob=search_blob,
    )


_ArtifactIndex = dict[str, Any]
_index_cache: dict[str, Any] = {}


def _root_fingerprint() -> tuple[tuple[str, int, int], ...]:
    """Cheap fingerprint of (relative_path, size, mtime_ns) for every catalog file.

    Used as the cache key for `_artifact_index` so the in-process cache
    invalidates automatically when any literature, hypothesis, or result file
    is added, removed, or modified mid-session. Avoids serving stale catalog
    payloads in long-running MCP processes — matches the project's
    "no stale shit" rule for read-only knowledge surfaces.
    """
    entries: list[tuple[str, int, int]] = []
    for root in (LITERATURE_ROOT, HYPOTHESES_ROOT, RESULTS_ROOT):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            entries.append((str(path), stat.st_size, stat.st_mtime_ns))
    if BLUEPRINT_PATH.exists() and BLUEPRINT_PATH.is_file():
        bp_stat = BLUEPRINT_PATH.stat()
        entries.append((str(BLUEPRINT_PATH), bp_stat.st_size, bp_stat.st_mtime_ns))
    return tuple(entries)


def _artifact_index_cache_clear() -> None:
    """Reset the artifact-index cache. Used by tests after fixture mutation."""
    _index_cache.clear()


def _artifact_index() -> _ArtifactIndex:
    fingerprint = _root_fingerprint()
    if _index_cache.get("fingerprint") == fingerprint and "index" in _index_cache:
        return _index_cache["index"]

    result_paths = sorted(
        path for path in RESULTS_ROOT.iterdir() if path.is_file() and path.suffix in {".md", ".csv", ".json"}
    )
    result_stems = {path.stem for path in result_paths}
    literature = [_parse_literature(path) for path in sorted(LITERATURE_ROOT.glob("*.md"))]
    hypotheses = [
        _parse_hypothesis(path, result_stems)
        for path in sorted(HYPOTHESES_ROOT.iterdir())
        if path.is_file() and path.suffix in {".yaml", ".md", ".txt"}
    ]
    results = [_parse_result(path) for path in result_paths if path.suffix == ".md"]
    blueprint_nogos = _parse_blueprint_nogo_artifacts(BLUEPRINT_PATH)
    results = [*results, *blueprint_nogos]
    by_kind: dict[str, dict[str, Artifact]] = {scope: {} for scope in VALID_SCOPES}
    for artifact in [*literature, *hypotheses, *results]:
        by_kind[artifact.kind][artifact.artifact_id.lower()] = artifact
    index: _ArtifactIndex = {
        "literature": tuple(literature),
        "hypothesis": tuple(hypotheses),
        "result": tuple(results),
        "by_kind": by_kind,
    }
    _index_cache["fingerprint"] = fingerprint
    _index_cache["index"] = index
    return index


def _artifact_payload(artifact: Artifact) -> dict[str, object]:
    payload = {
        "kind": artifact.kind,
        "artifact_id": artifact.artifact_id,
        "title": artifact.title,
        "date": artifact.date,
        "summary": artifact.summary,
        "path": _display_path(artifact.path),
    }
    payload.update(artifact.metadata)
    return payload


def _lookup_artifact(kind: ArtifactKind, artifact_id: str) -> Artifact:
    if not artifact_id or not artifact_id.strip():
        raise ValueError("artifact_id is required.")
    index = _artifact_index()["by_kind"][kind]
    exact = index.get(artifact_id.strip().lower())
    if exact is not None:
        return exact
    normalized_id = _normalized(artifact_id)
    for artifact in index.values():
        if normalized_id in {
            _normalized(artifact.artifact_id),
            _normalized(artifact.title),
            _normalized(artifact.path.name),
        }:
            return artifact
    raise ValueError(f"Unknown {kind} artifact_id: {artifact_id}")


def _list_literature_sources(limit: int = 50, query: str | None = None) -> dict[str, object]:
    items = list(_artifact_index()["literature"])
    if query and query.strip():
        query_terms = set(_tokenize(query))
        items = [
            artifact
            for artifact in items
            if query.lower() in artifact.search_blob.lower()
            or query_terms.intersection(_tokenize(artifact.search_blob))
        ]
    bounded = items[: max(limit, 1)]
    return {
        "total_sources": len(items),
        "items": [_artifact_payload(artifact) for artifact in bounded],
    }


def _get_literature_excerpt(
    source_id: str, section_hint: str | None = None, max_chars: int = 2000
) -> dict[str, object]:
    artifact = _lookup_artifact("literature", source_id)
    return {
        **_artifact_payload(artifact),
        "section_hint": section_hint,
        "excerpt": _section_excerpt(artifact.text, section_hint=section_hint, max_chars=max_chars),
    }


def _list_open_hypotheses(limit: int = 50, query: str | None = None) -> dict[str, object]:
    items = [
        artifact
        for artifact in _artifact_index()["hypothesis"]
        if not bool(artifact.metadata.get("exact_result_match"))
    ]
    if query and query.strip():
        query_terms = set(_tokenize(query))
        items = [
            artifact
            for artifact in items
            if query.lower() in artifact.search_blob.lower()
            or query_terms.intersection(_tokenize(artifact.search_blob))
        ]
    bounded = items[: max(limit, 1)]
    return {
        "status_basis": "Open means no exact filename stem match in docs/audit/results.",
        "total_open_hypotheses": len(items),
        "items": [_artifact_payload(artifact) for artifact in bounded],
    }


def _get_hypothesis_artifact(
    hypothesis_id: str, section_hint: str | None = None, max_chars: int = 2200
) -> dict[str, object]:
    artifact = _lookup_artifact("hypothesis", hypothesis_id)
    return {
        **_artifact_payload(artifact),
        "section_hint": section_hint,
        "excerpt": _section_excerpt(artifact.text, section_hint=section_hint, max_chars=max_chars),
    }


def _get_audit_result(result_id: str, section_hint: str | None = None, max_chars: int = 2400) -> dict[str, object]:
    artifact = _lookup_artifact("result", result_id)
    return {
        **_artifact_payload(artifact),
        "section_hint": section_hint,
        "excerpt": _section_excerpt(artifact.text, section_hint=section_hint, max_chars=max_chars),
    }


def _search_research_catalog(
    query: str,
    scopes: list[ArtifactKind] | None = None,
    limit: int = 10,
    verdict_tags: list[str] | None = None,
) -> dict[str, object]:
    if not query or not query.strip():
        raise ValueError("query is required.")
    if scopes is None:
        scopes = list(VALID_SCOPES)
    invalid_scopes = sorted({scope for scope in scopes if scope not in VALID_SCOPES})
    if invalid_scopes:
        raise ValueError(f"Unknown scope(s): {', '.join(invalid_scopes)}")

    normalized_verdict_filter: set[str] | None = None
    if verdict_tags is not None:
        normalized_pairs = [(tag, _normalize_verdict(tag)) for tag in verdict_tags]
        unknown = sorted({tag for tag, norm in normalized_pairs if norm is None})
        if unknown:
            raise ValueError(f"Unknown verdict_tag(s): {', '.join(unknown)}. Valid tags: {sorted(VALID_VERDICT_TAGS)}")
        normalized_verdict_filter = {norm for _tag, norm in normalized_pairs if norm is not None}

    query_terms = set(_tokenize(query))
    scored: list[tuple[int, Artifact]] = []
    for scope in scopes:
        for artifact in _artifact_index()[scope]:
            if normalized_verdict_filter is not None:
                artifact_verdict = artifact.metadata.get("verdict")
                if artifact_verdict not in normalized_verdict_filter:
                    continue
            lowered = artifact.search_blob.lower()
            score = 0
            if query.lower() in lowered:
                score += 40
            score += 8 * len(query_terms.intersection(_tokenize(artifact.search_blob)))
            if query.lower() in artifact.title.lower():
                score += 20
            if query.lower() in artifact.artifact_id.lower():
                score += 20
            if score:
                scored.append((score, artifact))

    scored.sort(key=lambda item: (-item[0], item[1].artifact_id))
    items = [
        {
            **_artifact_payload(artifact),
            "score": score,
            "snippet": _snippet_for_search(artifact.text, query_terms),
        }
        for score, artifact in scored[: max(limit, 1)]
    ]
    return {
        "query": query,
        "scopes": scopes,
        "verdict_tags": sorted(normalized_verdict_filter) if normalized_verdict_filter else None,
        "items": items,
    }


def _build_server():
    from scripts.tools.simple_mcp_stdio import StdioToolServer, ToolSpec

    instructions = (
        "Repo-local read-only research grounding surface for canompx3. "
        "Use it to search committed literature extracts, prereg hypotheses, and audit results. "
        "Prefer bounded excerpts and shallow catalog views before loading whole docs. "
        "Treat hypothesis open/closed status as a filename-stem heuristic unless a deeper audit confirms it."
    )

    return StdioToolServer(
        "research-catalog",
        instructions=instructions,
        tools=[
            ToolSpec(
                "list_literature_sources",
                "List committed literature extracts from docs/institutional/literature.",
                {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 50},
                        "query": {"type": ["string", "null"], "default": None},
                    },
                    "additionalProperties": False,
                },
                _list_literature_sources,
            ),
            ToolSpec(
                "get_literature_excerpt",
                "Return a bounded excerpt from a literature extract.",
                {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "section_hint": {"type": ["string", "null"], "default": None},
                        "max_chars": {"type": "integer", "default": 2000},
                    },
                    "required": ["source_id"],
                    "additionalProperties": False,
                },
                _get_literature_excerpt,
            ),
            ToolSpec(
                "list_open_hypotheses",
                "List prereg artifacts without an exact filename-stem match in docs/audit/results.",
                {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 50},
                        "query": {"type": ["string", "null"], "default": None},
                    },
                    "additionalProperties": False,
                },
                _list_open_hypotheses,
            ),
            ToolSpec(
                "get_hypothesis_artifact",
                "Return a bounded hypothesis artifact payload for a prereg doc.",
                {
                    "type": "object",
                    "properties": {
                        "hypothesis_id": {"type": "string"},
                        "section_hint": {"type": ["string", "null"], "default": None},
                        "max_chars": {"type": "integer", "default": 2200},
                    },
                    "required": ["hypothesis_id"],
                    "additionalProperties": False,
                },
                _get_hypothesis_artifact,
            ),
            ToolSpec(
                "get_audit_result",
                "Return a bounded audit-result payload from docs/audit/results.",
                {
                    "type": "object",
                    "properties": {
                        "result_id": {"type": "string"},
                        "section_hint": {"type": ["string", "null"], "default": None},
                        "max_chars": {"type": "integer", "default": 2400},
                    },
                    "required": ["result_id"],
                    "additionalProperties": False,
                },
                _get_audit_result,
            ),
            ToolSpec(
                "search_research_catalog",
                (
                    "Search the committed literature, prereg, and result catalog. "
                    "Optional verdict_tags filter (e.g., NO-GO, PARK, KILL, UNSUPPORTED, "
                    "DECAY, STALE) restricts results to artifacts carrying that verdict — "
                    "detected from filename stem, markdown front-matter, or "
                    "STRATEGY_BLUEPRINT.md §5 NO-GO Registry rows."
                ),
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "scopes": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "default": None,
                        },
                        "limit": {"type": "integer", "default": 10},
                        "verdict_tags": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "default": None,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                _search_research_catalog,
            ),
        ],
    )


def main() -> None:
    server = _build_server()
    server.run()


if __name__ == "__main__":
    main()
