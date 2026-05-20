"""Injection tests for check_fast_lane_state_graph_node_parity (Check #166).

Covers the three required mutation paths plus clean cases:

  Direction (a) ORPHAN-NODE:
    1. Spec names a node whose path does not exist on disk → VIOLATION.
    2. Spec names a glob path with zero matches → VIOLATION.

  Direction (b) ORPHAN-FILE:
    3. On-disk derived-state file exists but no spec node names it → VIOLATION.

  Malformed-doc fail-closed:
    4. Spec missing the '## 2. Node Inventory' heading → VIOLATION (no silent pass).
    5. Spec heading present but YAML fenced block is malformed → VIOLATION.
    6. Spec doc absent entirely → VIOLATION.

  Clean cases:
    7. Active node points at existing file → no violation.
    8. Reserved (proposed: true) node with absent path → no violation.
"""

from __future__ import annotations

import textwrap

from pipeline.check_drift import check_fast_lane_state_graph_node_parity


def _write_spec(spec_path, yaml_block: str, include_heading: bool = True):
    """Write a minimal spec doc with the Node Inventory section.

    Built without ``textwrap.dedent`` because interpolating a multi-line
    string (with embedded newlines but no leading whitespace) into a dedented
    template defeats dedent's common-prefix detection -- the heading would
    end up indented and the parser regex (anchored at ``^##``) would miss it.
    """
    heading = "## 2. Node Inventory" if include_heading else "## something else"
    lines = [
        "# Fast-Lane State Graph (test fixture)",
        "",
        "Pre-content prose.",
        "",
        heading,
        "",
        "```yaml",
        yaml_block.rstrip("\n"),
        "```",
        "",
        "Trailing prose.",
        "",
    ]
    spec_path.write_text("\n".join(lines), encoding="utf-8")
    return spec_path


def _make_runtime_tree(tmp_path):
    """Build a minimal runtime tree matching the chain's known glob roots."""
    (tmp_path / "docs" / "runtime").mkdir(parents=True)
    (tmp_path / "docs" / "audit" / "hypotheses" / "drafts").mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Direction (a) — ORPHAN-NODE: spec names non-existent file
# ---------------------------------------------------------------------------


def test_orphan_node_missing_file(tmp_path):
    """Active node names a path that does not exist on disk → VIOLATION."""
    runtime = _make_runtime_tree(tmp_path)
    spec = tmp_path / "spec.md"
    _write_spec(
        spec,
        textwrap.dedent("""\
        nodes:
          - id: ghost_node
            path: docs/runtime/does_not_exist.yaml
            writer: scripts/nowhere.py
            schema_version: 1
    """),
    )

    violations = check_fast_lane_state_graph_node_parity(spec_path=spec, runtime_dir=runtime)
    assert violations, "Expected ORPHAN-NODE violation but got none"
    combined = "\n".join(violations)
    assert "ORPHAN-NODE" in combined
    assert "ghost_node" in combined
    assert "does_not_exist.yaml" in combined


def test_orphan_node_glob_no_match(tmp_path):
    """Active node declares a glob with zero on-disk matches → VIOLATION."""
    runtime = _make_runtime_tree(tmp_path)
    spec = tmp_path / "spec.md"
    _write_spec(
        spec,
        textwrap.dedent("""\
        nodes:
          - id: ranking_csv_pattern
            path: docs/runtime/cherry_pick_ranking_*.csv
            writer: scripts/research/cherry_pick_ranker.py
            schema_version: 1
    """),
    )

    violations = check_fast_lane_state_graph_node_parity(spec_path=spec, runtime_dir=runtime)
    assert violations, "Expected ORPHAN-NODE violation for unmatched glob"
    combined = "\n".join(violations)
    assert "ORPHAN-NODE" in combined
    assert "ranking_csv_pattern" in combined


# ---------------------------------------------------------------------------
# Direction (b) — ORPHAN-FILE: file on disk, no spec node
# ---------------------------------------------------------------------------


def test_orphan_file_promote_queue_unnamed(tmp_path):
    """promote_queue.yaml exists on disk but no spec node names it → VIOLATION."""
    runtime = _make_runtime_tree(tmp_path)
    # Create an on-disk derived-state file that the chain knows about.
    (runtime / "docs" / "runtime" / "promote_queue.yaml").write_text(
        "schema_version: 1\nentries: []\n", encoding="utf-8"
    )
    spec = tmp_path / "spec.md"
    # Spec is empty of nodes — every on-disk file is orphaned.
    _write_spec(spec, "nodes: []\n")

    violations = check_fast_lane_state_graph_node_parity(spec_path=spec, runtime_dir=runtime)
    assert violations, "Expected ORPHAN-FILE violation but got none"
    combined = "\n".join(violations)
    assert "ORPHAN-FILE" in combined
    assert "promote_queue.yaml" in combined


# ---------------------------------------------------------------------------
# Malformed doc — fail-closed paths
# ---------------------------------------------------------------------------


def test_malformed_doc_missing_heading_fails_closed(tmp_path):
    """Spec missing '## 2. Node Inventory' heading → VIOLATION (no silent pass)."""
    runtime = _make_runtime_tree(tmp_path)
    spec = tmp_path / "spec.md"
    _write_spec(
        spec,
        "nodes: []\n",
        include_heading=False,  # heading replaced with '## something else'
    )

    violations = check_fast_lane_state_graph_node_parity(spec_path=spec, runtime_dir=runtime)
    assert violations, "Expected fail-closed violation on missing heading"
    combined = "\n".join(violations)
    assert "Node Inventory" in combined


def test_malformed_yaml_block_fails_closed(tmp_path):
    """Spec has heading but YAML block is unparseable → VIOLATION."""
    runtime = _make_runtime_tree(tmp_path)
    spec = tmp_path / "spec.md"
    # Inject genuinely broken YAML (unclosed inline map).
    spec.write_text(
        "## 2. Node Inventory\n```yaml\nnodes: [{id: x, path: '\n```\n",
        encoding="utf-8",
    )

    violations = check_fast_lane_state_graph_node_parity(spec_path=spec, runtime_dir=runtime)
    assert violations, "Expected fail-closed violation on unparseable YAML"
    combined = "\n".join(violations)
    # Either the YAML parse error or the structure-mismatch path may fire,
    # depending on which token the parser trips on first; both are valid
    # fail-closed messages from this check.
    assert "failed to parse Node Inventory" in combined or "is not a mapping" in combined or "is not a list" in combined


def test_missing_spec_doc_fails_closed(tmp_path):
    """Spec doc absent entirely → VIOLATION (no silent pass)."""
    runtime = _make_runtime_tree(tmp_path)
    spec = tmp_path / "does_not_exist.md"

    violations = check_fast_lane_state_graph_node_parity(spec_path=spec, runtime_dir=runtime)
    assert violations, "Expected fail-closed violation when spec doc absent"
    combined = "\n".join(violations)
    assert "spec doc not found" in combined


# ---------------------------------------------------------------------------
# Clean cases — should produce zero violations
# ---------------------------------------------------------------------------


def test_clean_active_node_resolves(tmp_path):
    """Active node whose path resolves on disk produces no violation."""
    runtime = _make_runtime_tree(tmp_path)
    (runtime / "docs" / "runtime" / "promote_queue.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (runtime / "docs" / "runtime" / "cherry_pick_journal.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (runtime / "docs" / "runtime" / "cherry_pick_ranking_2026-01-01.csv").write_text(
        "strategy_id,score\n", encoding="utf-8"
    )
    spec = tmp_path / "spec.md"
    _write_spec(
        spec,
        textwrap.dedent("""\
        nodes:
          - id: promote_queue
            path: docs/runtime/promote_queue.yaml
            writer: scripts/research/fast_lane_promote_queue.py
            schema_version: 1
          - id: cherry_pick_journal
            path: docs/runtime/cherry_pick_journal.yaml
            writer: scripts/research/cherry_pick_ranker.py
            schema_version: 1
          - id: cherry_pick_ranking_csv
            path: docs/runtime/cherry_pick_ranking_*.csv
            writer: scripts/research/cherry_pick_ranker.py
            schema_version: 1
          - id: heavyweight_drafts_dir
            path: docs/audit/hypotheses/drafts/
            writer: scripts/research/fast_lane_to_heavyweight_bridge.py
            schema_version: 1
    """),
    )

    violations = check_fast_lane_state_graph_node_parity(spec_path=spec, runtime_dir=runtime)
    assert violations == [], f"Expected clean pass, got: {violations}"


def test_proposed_node_with_absent_path_excluded(tmp_path):
    """Reserved (proposed: true) node with absent path → no violation.

    Build an isolated runtime tree without the drafts/ dir so the only thing
    under test is the proposed-node exclusion behavior.
    """
    runtime = tmp_path
    (runtime / "docs" / "runtime").mkdir(parents=True)
    spec = tmp_path / "spec.md"
    _write_spec(
        spec,
        textwrap.dedent("""\
        nodes:
          - id: future_status_rollup
            path: docs/runtime/not_yet_implemented.yaml
            writer: scripts/tools/future_writer.py
            schema_version: 1
            proposed: true
    """),
    )

    violations = check_fast_lane_state_graph_node_parity(spec_path=spec, runtime_dir=runtime)
    assert violations == [], f"Expected proposed-node to be excluded from parity, got: {violations}"
