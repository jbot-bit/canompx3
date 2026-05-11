"""Literature corpus indexer for the LLM hypothesis proposer.

Loads ``docs/institutional/literature/*.md`` into immutable ``LiteratureEntry``
objects so the proposer can (a) build a corpus summary to send to the LLM and
(b) check at validation time that each proposed ``theory_citation`` actually
maps to a file on disk.

This module is read-only. It never writes to the literature directory.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_MAX_FILE_BYTES = 200_000  # Hard ceiling; a corpus file above this is malformed.
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class LiteratureEntry:
    """One extracted literature file. Immutable for safe sharing."""

    path: Path
    slug: str
    title: str
    text: str
    blurb: str  # First non-empty paragraph after the H1, ≤500 chars.


def _extract_title(text: str, fallback: str) -> str:
    match = _H1_RE.search(text)
    if match is None:
        return fallback
    return match.group(1).strip()


def _extract_blurb(text: str) -> str:
    """Return the first non-empty paragraph after any H1 line, truncated."""
    lines = text.splitlines()
    after_h1 = False
    paragraph: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("#"):
            if paragraph:
                break
            after_h1 = True
            continue
        if not after_h1:
            # Pre-H1 frontmatter or blank — skip.
            continue
        if not line.strip():
            if paragraph:
                break
            continue
        paragraph.append(line.strip())
    blurb = " ".join(paragraph).strip()
    return blurb[:500]


def load_corpus(literature_dir: Path) -> Sequence[LiteratureEntry]:
    """Load every ``*.md`` in ``literature_dir`` (non-recursive).

    Files larger than ``_MAX_FILE_BYTES`` are skipped with no entry produced —
    the corpus is meant for short extracts, not full PDFs.
    """
    if not literature_dir.is_dir():
        raise FileNotFoundError(
            f"Literature directory not found: {literature_dir}. Expected docs/institutional/literature/ to exist."
        )
    entries: list[LiteratureEntry] = []
    for path in sorted(literature_dir.glob("*.md")):
        if path.stat().st_size > _MAX_FILE_BYTES:
            continue
        text = path.read_text(encoding="utf-8")
        title = _extract_title(text, fallback=path.stem)
        blurb = _extract_blurb(text)
        entries.append(
            LiteratureEntry(
                path=path.resolve(),
                slug=path.stem,
                title=title,
                text=text,
                blurb=blurb,
            )
        )
    return tuple(entries)


def citation_exists(corpus: Sequence[LiteratureEntry], cite_str: str) -> bool:
    """Return True iff ``cite_str`` matches at least one corpus entry.

    Match rule: case-insensitive substring of ``cite_str`` is found in any of
    ``slug``, ``path.name``, or ``title``. We do NOT require the user to write
    a fully-qualified path — only that a recognizable token from the file is
    present. Fabrications ("Smith 2099", "FakePaper") will not match anything.
    """
    if not isinstance(cite_str, str) or not cite_str.strip():
        return False
    needle = cite_str.lower()
    for entry in corpus:
        if entry.slug.lower() in needle:
            return True
        if entry.path.name.lower() in needle:
            return True
        # Title match: split title into words ≥4 chars and require at least
        # two such tokens appear in the citation. Prevents "Crabel 1990" from
        # matching every Crabel-mentioning file by accident.
        title_tokens = [t for t in re.findall(r"[A-Za-z]{4,}", entry.title.lower())]
        if len(title_tokens) >= 2:
            matched = sum(1 for tok in title_tokens if tok in needle)
            if matched >= 2:
                return True
    return False


def corpus_summary_for_llm(corpus: Sequence[LiteratureEntry], max_bytes: int = 8000) -> str:
    """Build a compact corpus summary suitable for the LLM context.

    Format per entry: ``- <slug> :: <title> :: <blurb>``. We truncate the
    output to ``max_bytes`` bytes (UTF-8) so a sprawling corpus cannot blow the
    LLM input budget.
    """
    parts: list[str] = []
    used = 0
    for entry in corpus:
        line = f"- {entry.slug} :: {entry.title} :: {entry.blurb}\n"
        nbytes = len(line.encode("utf-8"))
        if used + nbytes > max_bytes:
            break
        parts.append(line)
        used += nbytes
    return "".join(parts)


__all__ = [
    "LiteratureEntry",
    "citation_exists",
    "corpus_summary_for_llm",
    "load_corpus",
]
