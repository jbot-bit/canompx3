#!/usr/bin/env python
"""Check local PDF extraction and OCR tooling for resources-grounded work."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = PROJECT_ROOT / "resources"
INDEX_PATH = RESOURCES_DIR / "INDEX.md"


def _resource_roots() -> list[Path]:
    roots: list[Path] = [RESOURCES_DIR]
    env_root = os.environ.get("CANOMPX3_RESOURCE_ROOT")
    if env_root:
        roots.append(Path(env_root))

    primary = Path("C:/Users/joshd/canompx3/resources")
    if primary.exists() and primary not in roots:
        roots.append(primary)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(root)
    return deduped


def _index_pdf_names() -> list[str]:
    if not INDEX_PATH.exists():
        return []
    text = INDEX_PATH.read_text(encoding="utf-8", errors="replace")
    names = re.findall(r"`([^`]+\.pdf)`", text)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if "/" in name or "\\" in name:
            continue
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _first_pdf() -> Path | None:
    for root in _resource_roots():
        if not root.exists():
            continue
        for name in _index_pdf_names():
            path = root / name
            if path.is_file():
                return path
        for path in sorted(root.glob("*.pdf")):
            if path.is_file():
                return path
    return None


def _check_pymupdf(sample_pdf: Path | None) -> tuple[bool, list[str]]:
    lines: list[str] = []
    try:
        import fitz  # type: ignore[import-untyped]  # PyMuPDF
    except Exception as exc:
        return False, [f"FAIL pymupdf import: {exc}"]

    version = getattr(fitz, "version", None)
    lines.append(f"OK pymupdf import: {version or 'version unknown'}")

    if sample_pdf is None:
        lines.append("WARN no top-level resources/*.pdf sample found to open")
        return True, lines

    try:
        with fitz.open(str(sample_pdf)) as doc:
            page_count = doc.page_count
            first_text = doc[0].get_text("text") if page_count else ""
    except Exception as exc:
        return False, [*lines, f"FAIL pymupdf open/read {sample_pdf}: {exc}"]

    try:
        display = str(sample_pdf.relative_to(PROJECT_ROOT))
    except ValueError:
        display = str(sample_pdf)
    lines.append(f"OK opened sample PDF: {display} pages={page_count}")
    if len(first_text.strip()) < 50:
        lines.append("WARN sample first page has low text yield; scanned PDFs may require OCR")
    else:
        lines.append(f"OK sample first-page text yield: {len(first_text.strip())} chars")
    return True, lines


def _check_ocrmypdf(require_ocr: bool) -> tuple[bool, list[str]]:
    exe = shutil.which("ocrmypdf")
    if not exe:
        status = "FAIL" if require_ocr else "WARN"
        return not require_ocr, [f"{status} ocrmypdf not found on PATH"]

    try:
        result = subprocess.run(
            [exe, "--version"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        status = "FAIL" if require_ocr else "WARN"
        return not require_ocr, [f"{status} ocrmypdf --version failed: {exc}"]

    version = (result.stdout or result.stderr).strip().splitlines()
    if result.returncode != 0:
        status = "FAIL" if require_ocr else "WARN"
        detail = version[0] if version else f"exit={result.returncode}"
        return not require_ocr, [f"{status} ocrmypdf unusable: {detail}"]

    detail = version[0] if version else "version unknown"
    return True, [f"OK ocrmypdf available: {detail}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-ocr",
        action="store_true",
        help="Fail if ocrmypdf is not installed and runnable.",
    )
    args = parser.parse_args(argv)

    sample_pdf = _first_pdf()
    ok_pdf, pdf_lines = _check_pymupdf(sample_pdf)
    ok_ocr, ocr_lines = _check_ocrmypdf(args.require_ocr)

    for line in [*pdf_lines, *ocr_lines]:
        print(line)

    return 0 if ok_pdf and ok_ocr else 1


if __name__ == "__main__":
    raise SystemExit(main())
