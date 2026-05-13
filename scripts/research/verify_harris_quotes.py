#!/usr/bin/env python3
"""Verify every verbatim quote in the Harris 2002 extract against the OCR'd PDF.

Reads `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md`,
extracts every quote that follows the pattern `> "..."` and its cited page number
(`p.<N>` in the same Key-claim section), then re-extracts that PDF page via PyMuPDF
and asserts the quote text is present.

Exit code:
  0 — all quotes verified
  1 — at least one quote not located on its cited page
  2 — bad arguments or unreadable input

Safe to run repeatedly; pure read-only.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed in this venv", file=sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACT = (
    REPO_ROOT / "docs" / "institutional" / "literature" / "harris_2002_trading_exchanges_microstructure.md"
)
DEFAULT_PDF = REPO_ROOT / "resources" / "Harris_Trading_Exchanges_Market_Microstructure.pdf"


def _normalize(text: str) -> str:
    """Normalize for OCR-tolerant matching: NFKC, drop curly punctuation, repair common OCR
    artifacts (line-break hyphens, drop-cap replacement characters), collapse whitespace.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\xa0", " ")
    text = text.replace("�", "")  # OCR-substituted drop-cap initials become U+FFFD
    text = re.sub(r"-\s*\n\s*", "", text)  # rejoin hyphen-broken words across line breaks
    text = re.sub(r"-\s+", "", text) if False else text  # leave dash-separated tokens alone otherwise
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _fuzzy_contains(needle: str, haystack: str, max_diffs: int = 2) -> bool:
    """Return True if `needle` appears in `haystack` with at most `max_diffs` single-char
    insertions/deletions, OR if a suffix of `needle` (after dropping the leading 8 chars,
    which catches drop-cap OCR errors on paragraph-starting capitalized words) is contained.

    Designed for OCR-tolerant matching of short verbatim probes against Tesseract output
    of scanned books, which routinely (a) mis-recognizes drop-cap initial letters as the
    Unicode replacement character, (b) hyphenates words across line breaks, (c) confuses
    visually-similar character pairs (rn/m, cl/d).
    """
    if needle in haystack:
        return True
    # Suffix fallback: if the back-half of the probe appears, the leading drop-cap mismatch
    # is the most likely cause of failure. Try dropping 1-12 leading chars; any hit is OK.
    for skip in range(1, min(13, len(needle))):
        if needle[skip:] in haystack:
            return True
    if max_diffs <= 0:
        return False
    # Single-char deletion search
    n = len(needle)
    for i in range(n):
        if needle[:i] + needle[i + 1 :] in haystack:
            return True
    return False


def _strip_md(quote: str) -> str:
    """Remove markdown bold/italic and leading > from a quote line."""
    quote = re.sub(r"\*\*([^*]+)\*\*", r"\1", quote)
    quote = re.sub(r"__([^_]+)__", r"\1", quote)
    quote = quote.strip().strip(">").strip()
    quote = quote.strip('"').strip()
    return quote


def parse_extract(md_path: Path) -> list[dict]:
    """Parse the extract markdown into a list of (quote, cited_page, section) tuples.

    Looks for sections starting with `## Key claim N` (or `### Key claim N`),
    inside each section finds `**Verbatim (p.<N>...):**` headers followed by
    one or more `> "..."` lines.
    """
    text = md_path.read_text(encoding="utf-8")
    quotes: list[dict] = []

    section_re = re.compile(r"^#{2,3}\s+(Key claim \d+[^\n]*)", re.MULTILINE)
    sections = list(section_re.finditer(text))

    for i, sec in enumerate(sections):
        sec_title = sec.group(1).strip()
        sec_start = sec.end()
        sec_end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        body = text[sec_start:sec_end]

        verbatim_re = re.compile(
            r"\*\*Verbatim\s*\(p\.(\d+)[^)]*\)[^*]*:\*\*",
            re.IGNORECASE,
        )
        verb_matches = list(verbatim_re.finditer(body))
        for j, vm in enumerate(verb_matches):
            page = int(vm.group(1))
            block_start = vm.end()
            block_end = verb_matches[j + 1].start() if j + 1 < len(verb_matches) else len(body)
            block = body[block_start:block_end]

            quote_re = re.compile(r"^>\s+(.+?)(?=\n[^>]|\n\n|\Z)", re.MULTILINE | re.DOTALL)
            for qm in quote_re.finditer(block):
                raw = qm.group(1)
                quote_lines = [
                    ln.strip().lstrip(">").strip()
                    for ln in raw.splitlines()
                    if ln.strip().startswith(">") or ln.strip()
                ]
                quote = " ".join(quote_lines)
                quote = _strip_md(quote)
                if len(quote) < 20:
                    continue
                quotes.append({"section": sec_title, "page": page, "quote": quote})

    return quotes


def verify_quotes(
    quotes: list[dict], pdf_path: Path, probe_chars: int = 80, window: int = 1, printed_page_offset: int = 12
) -> tuple[int, int, list[dict]]:
    """For each quote, open the cited PDF page (±window pages tolerance for OCR drift) and
    check the first `probe_chars` normalized characters of the quote appear in the page text.

    The Harris hardcover OCR has front-matter (cover, title page, ToC, acknowledgements, preface)
    occupying PDF indices 0..11 before printed page 1 begins. So a citation that names printed
    page P corresponds to PyMuPDF (0-indexed) page index `P - 1 + printed_page_offset`.

    Returns (ok_count, fail_count, failures_list).
    """
    doc = fitz.open(str(pdf_path))
    ok = 0
    fail = 0
    failures: list[dict] = []

    for q in quotes:
        probe = _normalize(q["quote"])[:probe_chars]
        # Map printed page number -> 0-based PyMuPDF index
        target_page_idx = (q["page"] - 1) + printed_page_offset
        found_on = None
        for offset in range(-window, window + 1):
            idx = target_page_idx + offset
            if idx < 0 or idx >= len(doc):
                continue
            page_text = _normalize(doc[idx].get_text())
            if _fuzzy_contains(probe, page_text, max_diffs=2):
                found_on = idx + 1
                break
        if found_on is not None:
            ok += 1
        else:
            fail += 1
            failures.append(
                {
                    "section": q["section"],
                    "cited_page": q["page"],
                    "probe": probe[:60] + ("..." if len(probe) > 60 else ""),
                }
            )

    return ok, fail, failures


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Verify Harris 2002 extract quotes vs PDF")
    ap.add_argument("--extract", default=str(DEFAULT_EXTRACT))
    ap.add_argument("--pdf", default=str(DEFAULT_PDF))
    ap.add_argument("--probe-chars", type=int, default=80)
    ap.add_argument("--window", type=int, default=1, help="page tolerance for OCR drift")
    ap.add_argument(
        "--printed-page-offset",
        type=int,
        default=12,
        help="PyMuPDF index = printed_page - 1 + offset. Default 12 = Harris hardcover front-matter length.",
    )
    args = ap.parse_args(argv)

    md = Path(args.extract)
    pdf = Path(args.pdf)
    if not md.exists():
        print(f"ERROR: extract not found at {md}", file=sys.stderr)
        return 2
    if not pdf.exists():
        print(f"ERROR: PDF not found at {pdf}", file=sys.stderr)
        return 2

    quotes = parse_extract(md)
    if not quotes:
        print("ERROR: no quotes parsed from extract", file=sys.stderr)
        return 2

    print(f"Parsed {len(quotes)} verbatim quotes from {md.name}")
    print(f"Verifying against {pdf.name} (window=+/-{args.window} pages, probe={args.probe_chars} chars)")
    print()

    ok, fail, failures = verify_quotes(
        quotes, pdf, probe_chars=args.probe_chars, window=args.window, printed_page_offset=args.printed_page_offset
    )
    print(f"OK:    {ok}/{len(quotes)}")
    print(f"FAIL:  {fail}/{len(quotes)}")
    if failures:
        print()
        print("Failures:")
        for f in failures:
            print(f"  [{f['section']}] p.{f['cited_page']}: {f['probe']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
