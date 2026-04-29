"""CRG (code-review-graph) helpers for pipeline/check_drift.py.

Calls CRG via its internal Python API (``code_review_graph.analysis`` and
``code_review_graph.tools.query``).  We deliberately bypass the
``code_review_graph.tools.analysis_tools.*_func`` wrappers because v2.1.0 of
those wrappers have a known bug: ``store = _get_store(str(root))`` stores a
``(GraphStore, Path)`` tuple in ``store`` but then passes that tuple to
``find_*`` calls that expect just the ``GraphStore``.  Going via
``analysis.find_*(store, ...)`` directly works around it.

All functions are fail-open: when CRG is unavailable (package missing, graph
DB absent, runtime error in any underlying call) they return the sentinel
``CRG_UNAVAILABLE`` so callers can emit an ADVISORY and exit 0.

Module isolation: ``check_drift.py`` never imports ``code_review_graph``
directly. Removing CRG requires only deleting this file and the five D1-D5
check functions in ``check_drift.py``.

Authority: ``docs/plans/2026-04-29-crg-integration-spec.md`` Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Sentinel returned when CRG is unavailable (package missing, graph DB absent,
# or runtime error). Callers check ``is CRG_UNAVAILABLE``.
CRG_UNAVAILABLE: object = object()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _graph_db_exists() -> bool:
    return (_PROJECT_ROOT / ".code-review-graph" / "graph.db").exists()


def crg_is_available() -> bool:
    """Return True iff the ``code_review_graph`` package and graph DB are both present."""
    if not _graph_db_exists():
        return False
    try:
        import code_review_graph  # noqa: F401
    except ImportError:
        return False
    return True


def _open_store():
    """Open CRG's graph store. Raises on failure; callers wrap in try/except."""
    from code_review_graph.tools._common import _get_store

    store, _root = _get_store(str(_PROJECT_ROOT))
    return store


# ── public API — one function per Phase 2 drift check ─────────────────────


def get_surprising_connections(top_n: int = 50) -> list[dict[str, Any]] | object:
    """Return surprising cross-layer edges from CRG's analysis layer.

    Each entry has keys: ``source``, ``source_qualified``, ``target``,
    ``target_qualified``, ``edge_kind``, ``surprise_score``, ``reasons``,
    ``source_community``, ``target_community``.

    Returns ``CRG_UNAVAILABLE`` on any error.
    """
    try:
        from code_review_graph.analysis import find_surprising_connections

        store = _open_store()
        result = find_surprising_connections(store, top_n=top_n)
    except Exception:
        return CRG_UNAVAILABLE
    return result if isinstance(result, list) else CRG_UNAVAILABLE


def query_tests_for(qualified_target: str) -> dict[str, Any] | object:
    """Return TESTS_FOR results for a CRG qualified-name target.

    The target MUST be in CRG qualified-name format
    (``<relative/path>::<symbol>``), not Python dotted notation.

    Returns a tagged dict with ``status`` ∈ {"ok", "empty", "error"} and:
      - ``status == "ok"`` → ``tests`` is a non-empty list of test-node dicts.
      - ``status == "empty"`` → CRG ran cleanly but returned no edges.
      - ``status == "error"`` → CRG returned a non-ok response shape
        (``not_found`` / ``ambiguous`` / unexpected). Caller can distinguish
        this from a genuine "no tests" finding.

    Returns ``CRG_UNAVAILABLE`` only on a hard exception during the call
    (CRG package missing or runtime crash), distinct from ``status="error"``
    above.

    Audit-trail (2026-04-29): earlier version collapsed ``not_found`` /
    ``ambiguous`` / ``error`` into ``[]``, indistinguishable from the
    legitimate empty-tests case. Tagged status was added to let D5
    surface a separate ADVISORY when CRG itself is uncertain about a target.
    """
    try:
        from code_review_graph.tools.query import query_graph

        result = query_graph(
            pattern="tests_for",
            target=qualified_target,
            repo_root=str(_PROJECT_ROOT),
            detail_level="minimal",
        )
    except Exception:
        return CRG_UNAVAILABLE
    if not isinstance(result, dict):
        return {"status": "error", "tests": [], "raw_status": "non-dict-response"}
    raw_status = result.get("status")
    if raw_status != "ok":
        return {"status": "error", "tests": [], "raw_status": str(raw_status)}
    nodes = result.get("results", [])
    if not isinstance(nodes, list):
        return {"status": "error", "tests": [], "raw_status": "non-list-results"}
    return {"status": "ok" if nodes else "empty", "tests": nodes, "raw_status": "ok"}


def find_large_functions(min_lines: int = 200, limit: int = 100) -> list[dict[str, Any]] | object:
    """Return all functions exceeding ``min_lines`` (no path filter).

    CRG's ``file_path_pattern`` is a substring match against the stored path,
    which on Windows uses backslashes — making cross-platform pattern matching
    brittle. We instead fetch ALL large functions and let callers filter
    client-side via ``Path.parts``.

    Each entry has: ``relative_path``, ``file_path``, ``name``,
    ``qualified_name``, ``line_count``.

    Returns list of function dicts or ``CRG_UNAVAILABLE``.
    """
    try:
        from code_review_graph.tools.query import find_large_functions as _fl

        result = _fl(
            min_lines=min_lines,
            kind="Function",
            file_path_pattern=None,
            limit=limit,
            repo_root=str(_PROJECT_ROOT),
        )
    except Exception:
        return CRG_UNAVAILABLE
    if not isinstance(result, dict):
        return CRG_UNAVAILABLE
    nodes = result.get("results", [])
    return nodes if isinstance(nodes, list) else CRG_UNAVAILABLE


def get_bridge_nodes(top_n: int = 10) -> list[dict[str, Any]] | object:
    """Return top-N betweenness-centrality bridge nodes.

    Each entry has: ``name``, ``qualified_name``, ``kind``, ``file``,
    ``betweenness``, ``community_id``.  ``file`` is an absolute path on the
    indexing host; ``qualified_name`` is in CRG format (``<path>::<symbol>``).

    Returns list of bridge-node dicts or ``CRG_UNAVAILABLE``.
    """
    try:
        from code_review_graph.analysis import find_bridge_nodes

        store = _open_store()
        result = find_bridge_nodes(store, top_n=top_n)
    except Exception:
        return CRG_UNAVAILABLE
    return result if isinstance(result, list) else CRG_UNAVAILABLE


def get_affected_flows(
    changed_files: list[str] | None = None,
    base: str = "HEAD~1",
) -> list[dict[str, Any]] | object:
    """Return execution flows affected by ``changed_files`` (or auto-detected diff).

    Used by /crg-lineage when given a file path.

    Returns list of flow dicts or ``CRG_UNAVAILABLE``.
    """
    try:
        from code_review_graph.tools.review import get_affected_flows_func

        result = get_affected_flows_func(
            changed_files=changed_files,
            base=base,
            repo_root=str(_PROJECT_ROOT),
        )
    except Exception:
        return CRG_UNAVAILABLE
    if not isinstance(result, dict):
        return CRG_UNAVAILABLE
    flows = result.get("affected_flows")
    return flows if isinstance(flows, list) else CRG_UNAVAILABLE
