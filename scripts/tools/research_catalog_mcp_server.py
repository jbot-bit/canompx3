"""MCP server for repo-local research grounding and audit artifact lookup.

Exposes read-only tools via stdio (fastmcp):
  - list_literature_sources: list committed local literature extracts
  - get_literature_excerpt: return a bounded excerpt from a literature source
  - list_open_hypotheses: list prereg artifacts without an exact result-stem match
  - get_hypothesis_artifact: return a bounded hypothesis artifact payload
  - get_audit_result: return a bounded audit-result payload
  - search_research_catalog: search literature, hypotheses, and result docs
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

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


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
    metadata = {
        "pre_registration": prereg,
        "path": _display_path(path),
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
) -> dict[str, object]:
    if not query or not query.strip():
        raise ValueError("query is required.")
    if scopes is None:
        scopes = list(VALID_SCOPES)
    invalid_scopes = sorted({scope for scope in scopes if scope not in VALID_SCOPES})
    if invalid_scopes:
        raise ValueError(f"Unknown scope(s): {', '.join(invalid_scopes)}")

    query_terms = set(_tokenize(query))
    scored: list[tuple[int, Artifact]] = []
    for scope in scopes:
        for artifact in _artifact_index()[scope]:
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
        "items": items,
    }


def _build_server():
    from fastmcp import FastMCP

    mcp = FastMCP(
        "research-catalog",
        instructions=(
            "Repo-local read-only research grounding surface for canompx3. "
            "Use it to search committed literature extracts, prereg hypotheses, and audit results. "
            "Prefer bounded excerpts and shallow catalog views before loading whole docs. "
            "Treat hypothesis open/closed status as a filename-stem heuristic unless a deeper audit confirms it."
        ),
    )

    @mcp.tool()
    def list_literature_sources(limit: int = 50, query: str | None = None) -> dict[str, object]:
        """List committed literature extracts from docs/institutional/literature."""

        return _list_literature_sources(limit=limit, query=query)

    @mcp.tool()
    def get_literature_excerpt(
        source_id: str,
        section_hint: str | None = None,
        max_chars: int = 2000,
    ) -> dict[str, object]:
        """Return a bounded excerpt from a literature extract.

        Args:
            source_id: File stem from list_literature_sources.
            section_hint: Optional heading fragment to narrow the excerpt.
            max_chars: Maximum excerpt size in characters.
        """

        return _get_literature_excerpt(source_id=source_id, section_hint=section_hint, max_chars=max_chars)

    @mcp.tool()
    def list_open_hypotheses(limit: int = 50, query: str | None = None) -> dict[str, object]:
        """List prereg artifacts without an exact filename-stem match in docs/audit/results."""

        return _list_open_hypotheses(limit=limit, query=query)

    @mcp.tool()
    def get_hypothesis_artifact(
        hypothesis_id: str,
        section_hint: str | None = None,
        max_chars: int = 2200,
    ) -> dict[str, object]:
        """Return a bounded hypothesis artifact payload for a prereg doc."""

        return _get_hypothesis_artifact(hypothesis_id=hypothesis_id, section_hint=section_hint, max_chars=max_chars)

    @mcp.tool()
    def get_audit_result(
        result_id: str,
        section_hint: str | None = None,
        max_chars: int = 2400,
    ) -> dict[str, object]:
        """Return a bounded audit-result payload from docs/audit/results."""

        return _get_audit_result(result_id=result_id, section_hint=section_hint, max_chars=max_chars)

    @mcp.tool()
    def search_research_catalog(
        query: str,
        scopes: list[ArtifactKind] | None = None,
        limit: int = 10,
    ) -> dict[str, object]:
        """Search the committed literature, prereg, and result catalog."""

        return _search_research_catalog(query=query, scopes=scopes, limit=limit)

    return mcp


def main() -> None:
    server = _build_server()
    server.run()


if __name__ == "__main__":
    main()
