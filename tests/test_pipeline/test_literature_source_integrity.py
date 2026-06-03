"""Literature source-integrity drift check — two-mode verification tests.

The check `check_literature_source_integrity` reads the machine-readable manifest
(docs/audit/research_grounding_source_manifest.yaml) and verifies grounding
*integrity* — that every literature extract resolves to a declared source whose
status is honestly recorded, and (strict-local only) that present PDFs match the
extract's cited page count.

Two modes, by design (mirrors check_drift's _skip_db_check_for_ci idiom):
  CI-safe (default)  : manifest <-> extract consistency. Does NOT require any PDF
                       on disk -> passes in a fresh CI clone. MUST NEVER emit a
                       green that implies PDFs were verified when they weren't.
  strict-local       : additionally verify present PDFs exist + fitz page-count
                       matches the extract's declared `expected_pages`.

These tests prove (per plan §E):
  - real repo state passes CI-safe (the committed manifest is internally consistent)
  - CI-safe passes when untracked PDFs are absent
  - strict-local FAILS when a declared local PDF is absent (but only for rows it
    is asked to verify locally — present-file gate)
  - strict-local FAILS when a present PDF's page count != declared expected_pages
  - URL/web-derived rows pass only when explicitly declared WEB_DERIVED
  - a stub passes only when declared STUB_LEDGERED
  - chatgpt_bundle/LIT_*.md + literature/ are both covered; an extract missing
    from the manifest is a violation (false-completeness guard)
  - files without a Source: line (e.g. PENDING_ACQUISITION_*) are skipped, not
    flagged as missing-source
  - generic title/text overlap is NEVER treated as source proof (G12/Pardo class)

Fixtures use pytest tmp_path (OS-native Windows temp, NOT bash /tmp — avoids the
cross-env empty-read footgun) and inject roots via the function's parameters.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pipeline.check_drift import check_literature_source_integrity


# --- helpers ---------------------------------------------------------------


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def _make_repo(
    tmp_path: Path,
    *,
    manifest: str,
    extracts: dict[str, str],
) -> Path:
    """Build a minimal fake repo: a manifest + the extract files it references."""
    _write(tmp_path / "docs" / "audit" / "research_grounding_source_manifest.yaml", manifest)
    for rel, body in extracts.items():
        _write(tmp_path / rel, body)
    return tmp_path


# A self-consistent minimal manifest: one MISSING_LOCAL PDF row + one extract.
_GOOD_MANIFEST = """\
    sources:
      - source_id: demo_paper
        extract_files:
          - docs/institutional/literature/demo_paper.md
        expected_resource: resources/Demo_Paper.pdf
        source_tier: peer-reviewed
        tracked: n
        ci_verifiable: y
        gap_id: null
        status: MISSING_LOCAL
        expected_pages: 10
    out_of_scope: []
"""

_GOOD_EXTRACT = {
    "docs/institutional/literature/demo_paper.md": """\
        # Demo Paper

        **Source:** `resources/Demo_Paper.pdf` (10 pages)
        **Pages read:** Full text, 10 pages.
    """
}


# --- real repo state -------------------------------------------------------


def test_real_repo_passes_ci_safe():
    """The committed manifest + extracts must be internally consistent (CI-safe)."""
    violations = check_literature_source_integrity(strict_local=False)
    assert violations == [], "CI-safe mode must pass on real repo state: " + "; ".join(violations)


# --- CI-safe core ----------------------------------------------------------


def test_ci_safe_passes_when_untracked_pdf_absent(tmp_path):
    """CI-safe must pass even though resources/Demo_Paper.pdf does not exist."""
    root = _make_repo(tmp_path, manifest=_GOOD_MANIFEST, extracts=_GOOD_EXTRACT)
    assert check_literature_source_integrity(strict_local=False, project_root=root) == []


def test_ci_safe_fails_when_extract_missing_from_manifest(tmp_path):
    """False-completeness guard: an extract on disk with a Source: line but no
    manifest row is drift."""
    extracts = dict(_GOOD_EXTRACT)
    extracts["docs/institutional/literature/orphan.md"] = """\
        # Orphan

        **Source:** `resources/Orphan.pdf` (5 pages)
    """
    root = _make_repo(tmp_path, manifest=_GOOD_MANIFEST, extracts=extracts)
    violations = check_literature_source_integrity(strict_local=False, project_root=root)
    assert any("orphan.md" in v and "not in manifest" in v for v in violations), violations


def test_ci_safe_covers_chatgpt_bundle(tmp_path):
    """chatgpt_bundle/LIT_*.md is a grounding surface too — an uncovered bundle
    extract with a Source: line is drift (Blocker B)."""
    extracts = dict(_GOOD_EXTRACT)
    extracts["chatgpt_bundle/LIT_demo_bundle.md"] = """\
        # Demo bundle

        **Source:** `resources/Demo_Bundle.pdf` (8 pages)
    """
    root = _make_repo(tmp_path, manifest=_GOOD_MANIFEST, extracts=extracts)
    violations = check_literature_source_integrity(strict_local=False, project_root=root)
    assert any("LIT_demo_bundle.md" in v and "not in manifest" in v for v in violations), violations


def test_source_line_without_manifest_for_pending_doc_is_skipped(tmp_path):
    """A literature/ file with NO Source: line (e.g. PENDING_ACQUISITION_*) is a
    tracking doc, not an extract — must NOT be flagged as missing-source."""
    extracts = dict(_GOOD_EXTRACT)
    extracts["docs/institutional/literature/PENDING_ACQUISITION_demo.md"] = """\
        # Pending Acquisition

        **Status:** awaiting source. No Source line here.
    """
    root = _make_repo(tmp_path, manifest=_GOOD_MANIFEST, extracts=extracts)
    assert check_literature_source_integrity(strict_local=False, project_root=root) == []


def test_ci_safe_fails_on_inconsistent_status(tmp_path):
    """Declared status must be internally consistent: a WEB_DERIVED row whose
    expected_resource is a resources/*.pdf (not a URL) is dishonest -> drift."""
    bad_manifest = """\
        sources:
          - source_id: demo_paper
            extract_files:
              - docs/institutional/literature/demo_paper.md
            expected_resource: resources/Demo_Paper.pdf
            source_tier: web
            tracked: y
            ci_verifiable: y
            gap_id: G5
            status: WEB_DERIVED
        out_of_scope: []
    """
    root = _make_repo(tmp_path, manifest=bad_manifest, extracts=_GOOD_EXTRACT)
    violations = check_literature_source_integrity(strict_local=False, project_root=root)
    assert any("WEB_DERIVED" in v for v in violations), violations


def test_web_derived_passes_when_declared_with_url(tmp_path):
    """A URL source declared WEB_DERIVED is honest grounding -> passes."""
    manifest = """\
        sources:
          - source_id: web_paper
            extract_files:
              - docs/institutional/literature/web_paper.md
            expected_resource: https://example.com/paper
            source_tier: web
            tracked: y
            ci_verifiable: y
            gap_id: G5
            status: WEB_DERIVED
        out_of_scope: []
    """
    extracts = {
        "docs/institutional/literature/web_paper.md": """\
            # Web Paper

            **Source:** https://example.com/paper
        """
    }
    root = _make_repo(tmp_path, manifest=manifest, extracts=extracts)
    assert check_literature_source_integrity(strict_local=False, project_root=root) == []


def test_stub_passes_only_when_ledgered(tmp_path):
    """A STUB_LEDGERED row passes; the same stub declared as a normal source whose
    PDF can't be verified must not silently pass under a wrong status."""
    manifest = """\
        sources:
          - source_id: stub_paper
            extract_files:
              - docs/institutional/literature/stub_paper.md
            expected_resource: resources/Stub.pdf
            source_tier: book
            tracked: n
            ci_verifiable: y
            gap_id: G6
            status: STUB_LEDGERED
        out_of_scope: []
    """
    extracts = {
        "docs/institutional/literature/stub_paper.md": """\
            # Stub Paper

            **Source:** `resources/Stub.pdf`
            **Status:** STUB — content not yet extracted.
        """
    }
    root = _make_repo(tmp_path, manifest=manifest, extracts=extracts)
    assert check_literature_source_integrity(strict_local=False, project_root=root) == []


# --- strict-local ----------------------------------------------------------


def test_strict_local_fails_when_declared_pdf_absent(tmp_path):
    """strict-local with a flag that forces verification must FAIL when a row's
    PDF is absent — this is the divergence from CI-safe."""
    root = _make_repo(tmp_path, manifest=_GOOD_MANIFEST, extracts=_GOOD_EXTRACT)
    # CI-safe passes ...
    assert check_literature_source_integrity(strict_local=False, project_root=root) == []
    # ... strict-local with require_all forces the present-file gate and fails.
    violations = check_literature_source_integrity(strict_local=True, require_all=True, project_root=root)
    assert any("Demo_Paper.pdf" in v and "absent" in v.lower() for v in violations), violations


def test_strict_local_passes_when_pdf_present_and_pages_match(tmp_path):
    """strict-local verifies a present PDF whose page count matches expected_pages."""
    fitz = pytest.importorskip("fitz")
    root = _make_repo(tmp_path, manifest=_GOOD_MANIFEST, extracts=_GOOD_EXTRACT)
    # Synthesize a real 10-page PDF so fitz reports the matching count.
    doc = fitz.open()
    for _ in range(10):
        doc.new_page()
    (root / "resources").mkdir(parents=True, exist_ok=True)
    doc.save(str(root / "resources" / "Demo_Paper.pdf"))
    doc.close()
    assert check_literature_source_integrity(strict_local=True, project_root=root) == []


def test_strict_local_fails_on_page_count_mismatch(tmp_path):
    """strict-local fails when a present PDF's page count != declared expected_pages
    (the 'cited pages exist' rule; guards the AFML 3pp-stub-vs-393pp trap)."""
    fitz = pytest.importorskip("fitz")
    root = _make_repo(tmp_path, manifest=_GOOD_MANIFEST, extracts=_GOOD_EXTRACT)
    doc = fitz.open()
    for _ in range(3):  # declared 10, actual 3 — the sampler/stub trap
        doc.new_page()
    (root / "resources").mkdir(parents=True, exist_ok=True)
    doc.save(str(root / "resources" / "Demo_Paper.pdf"))
    doc.close()
    violations = check_literature_source_integrity(strict_local=True, project_root=root)
    assert any("page" in v.lower() and "Demo_Paper.pdf" in v for v in violations), violations


def test_committed_verified_local_for_untracked_absent_pdf_is_drift(tmp_path):
    """Honesty rule: VERIFIED_LOCAL is a RUNTIME status, never a committed claim
    for an untracked PDF. A committed VERIFIED_LOCAL whose (tracked:n) PDF is
    absent on disk asserts a verification a CI clone cannot reproduce -> drift,
    even in CI-safe mode. (This is the core 'never imply PDFs verified when they
    weren't' guarantee, made self-enforcing.)"""
    manifest = """\
        sources:
          - source_id: demo_paper
            extract_files:
              - docs/institutional/literature/demo_paper.md
            expected_resource: resources/Demo_Paper.pdf
            source_tier: peer-reviewed
            tracked: n
            ci_verifiable: y
            gap_id: G1
            status: VERIFIED_LOCAL
            expected_pages: 10
        out_of_scope: []
    """
    root = _make_repo(tmp_path, manifest=manifest, extracts=_GOOD_EXTRACT)
    violations = check_literature_source_integrity(strict_local=False, project_root=root)
    assert any("VERIFIED_LOCAL" in v and "demo_paper" in v for v in violations), violations


def test_verified_local_is_valid_when_pdf_actually_present(tmp_path):
    """When the PDF IS on disk, VERIFIED_LOCAL is a reproducible runtime fact ->
    no drift (the status is allowed to be true when it is verifiable here)."""
    fitz = pytest.importorskip("fitz")
    manifest = """\
        sources:
          - source_id: demo_paper
            extract_files:
              - docs/institutional/literature/demo_paper.md
            expected_resource: resources/Demo_Paper.pdf
            source_tier: peer-reviewed
            tracked: n
            ci_verifiable: y
            gap_id: G1
            status: VERIFIED_LOCAL
            expected_pages: 10
        out_of_scope: []
    """
    root = _make_repo(tmp_path, manifest=manifest, extracts=_GOOD_EXTRACT)
    doc = fitz.open()
    for _ in range(10):
        doc.new_page()
    (root / "resources").mkdir(parents=True, exist_ok=True)
    doc.save(str(root / "resources" / "Demo_Paper.pdf"))
    doc.close()
    assert check_literature_source_integrity(strict_local=False, project_root=root) == []


# --- generic-title-overlap false-positive guard (G12 / Pardo class) --------


def test_generic_title_overlap_never_counts_as_source_proof(tmp_path):
    """If a different PDF whose title merely overlaps the extract's words is on
    disk, strict-local must NOT accept it as the source — only the exact declared
    filename + page count counts. Guards the G12/Pardo false-positive class."""
    fitz = pytest.importorskip("fitz")
    root = _make_repo(tmp_path, manifest=_GOOD_MANIFEST, extracts=_GOOD_EXTRACT)
    (root / "resources").mkdir(parents=True, exist_ok=True)
    # A decoy with an overlapping title but the WRONG filename + WRONG pages.
    doc = fitz.open()
    doc.new_page()
    doc.set_metadata({"title": "Demo Paper — a totally different optimization book"})
    doc.save(str(root / "resources" / "Some_Other_Demo.pdf"))
    doc.close()
    # The declared resources/Demo_Paper.pdf is still absent.
    violations = check_literature_source_integrity(strict_local=True, require_all=True, project_root=root)
    assert any("Demo_Paper.pdf" in v and "absent" in v.lower() for v in violations), violations
