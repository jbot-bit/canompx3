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


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "over",
        "than",
        "that",
        "the",
        "this",
        "those",
        "to",
        "was",
        "were",
        "which",
        "while",
        "with",
        "without",
        "would",
        "should",
        "must",
        "where",
        "when",
        "what",
        "such",
        "shows",
        "demonstrates",
        "filter",
        "filters",
        "framework",
        "implies",
        "regime",
        "regimes",
        "exceeds",
        "exceeding",
        "exceed",
        "above",
        "below",
        "higher",
        "lower",
        "etc",
    }
)


def _tokenize(text: str, *, min_len: int = 4) -> list[str]:
    """Lowercase alpha tokens >= min_len, stopwords removed."""
    return [t.lower() for t in re.findall(r"[A-Za-z]+", text) if len(t) >= min_len and t.lower() not in _STOPWORDS]


def find_citation_entries(corpus: Sequence[LiteratureEntry], cite_str: str) -> list[LiteratureEntry]:
    """Return all corpus entries that match ``cite_str``.

    Same match rule as ``citation_exists`` but returns the matching entries
    rather than a boolean. ``theory_citation`` strings often join multiple
    slugs with an em-dash or comma; this surfaces all matched files so the
    content-verification step can scan each one for the economic_basis terms.
    """
    if not isinstance(cite_str, str) or not cite_str.strip():
        return []
    needle = cite_str.lower()
    matches: list[LiteratureEntry] = []
    seen: set[Path] = set()
    for entry in corpus:
        already = entry.path in seen
        slug_hit = entry.slug.lower() in needle
        name_hit = entry.path.name.lower() in needle
        title_tokens = [t for t in re.findall(r"[A-Za-z]{4,}", entry.title.lower())]
        title_hit = False
        if len(title_tokens) >= 2:
            matched_tokens = sum(1 for tok in title_tokens if tok in needle)
            title_hit = matched_tokens >= 2
        if not already and (slug_hit or name_hit or title_hit):
            matches.append(entry)
            seen.add(entry.path)
    return matches


def verify_citation_content(
    corpus: Sequence[LiteratureEntry],
    cite_str: str,
    economic_basis: str,
    *,
    min_term_overlap: int = 3,
    min_term_overlap_pct: float = 0.30,
) -> dict[str, object]:
    """Verify that the cited literature ACTUALLY supports ``economic_basis``.

    Catches the Carver-sizing-cited-for-entry-filter class bug: file exists,
    filename appears in cite_str, but the file body doesn't address the
    claimed mechanism. Yesterday's 3 LLM-drafted pre-regs all passed file-
    existence and would have failed this content check.

    Algorithm:
        1. Tokenize ``economic_basis`` (lowercase alpha, ≥4 chars, stopwords
           removed). Call this set ``E``.
        2. Find the entries cited (``find_citation_entries``).
        3. For each entry, tokenize its body the same way. Call this ``B``.
        4. Compute ``overlap = E ∩ B``. Compute ``coverage = |overlap| / |E|``.
        5. Pass if EITHER ``|overlap| >= min_term_overlap`` OR
           ``coverage >= min_term_overlap_pct``.

    Why both thresholds: a 3-word ``economic_basis`` (rare but possible)
    needs coverage; a 50-word one needs absolute count or it becomes too
    strict to satisfy.

    Returns
    -------
    dict with keys:
        - ``passes``: bool
        - ``cited_files``: list[str] of slugs matched
        - ``no_cited_file_found``: bool — caller should reject outright
        - ``per_file``: list[dict] with slug, overlap_count, coverage, overlap_terms
        - ``reason``: human-readable verdict line
    """
    cited = find_citation_entries(corpus, cite_str)
    if not cited:
        return {
            "passes": False,
            "cited_files": [],
            "no_cited_file_found": True,
            "per_file": [],
            "reason": f"no corpus entry matches theory_citation={cite_str!r}",
        }

    basis_tokens = set(_tokenize(economic_basis))
    if not basis_tokens:
        return {
            "passes": False,
            "cited_files": [e.slug for e in cited],
            "no_cited_file_found": False,
            "per_file": [],
            "reason": "economic_basis has no content tokens after stopword removal",
        }

    per_file: list[dict[str, object]] = []
    any_pass = False
    for entry in cited:
        body_tokens = set(_tokenize(entry.text))
        overlap = basis_tokens & body_tokens
        coverage = len(overlap) / max(1, len(basis_tokens))
        file_pass = len(overlap) >= min_term_overlap or coverage >= min_term_overlap_pct
        any_pass = any_pass or file_pass
        per_file.append(
            {
                "slug": entry.slug,
                "overlap_count": len(overlap),
                "basis_token_count": len(basis_tokens),
                "coverage": coverage,
                "overlap_terms": sorted(overlap)[:20],
                "file_passes": file_pass,
            }
        )

    reason = (
        "citation_content_OK: at least one cited file shares enough mechanism terms with economic_basis"
        if any_pass
        else f"citation_content_MISMATCH: none of {[e.slug for e in cited]} shares "
        f">= {min_term_overlap} or {min_term_overlap_pct:.0%} of {len(basis_tokens)} basis tokens"
    )
    return {
        "passes": any_pass,
        "cited_files": [e.slug for e in cited],
        "no_cited_file_found": False,
        "per_file": per_file,
        "reason": reason,
    }


def search_corpus(
    corpus: Sequence[LiteratureEntry],
    query: str,
    *,
    top_k: int = 5,
    min_overlap: int = 2,
) -> list[dict[str, object]]:
    """Targeted top-K extracts ranked by query-token overlap with each body.

    Offline equivalent of ``mcp__research-catalog__search_research_catalog``
    for the LHP proposer pipeline: tokenize ``query`` the same way
    ``verify_citation_content`` tokenizes ``economic_basis``, score each entry
    by intersection size with its body tokens, return the top-K sorted by
    score desc (entries below ``min_overlap`` are excluded — better to return
    fewer hits than to surface unrelated extracts as "the literature").

    Used by ``llm_hypothesis_proposer.py --ground-via-mcp`` to pre-load the
    LLM context with the most relevant on-disk extracts BEFORE drafting.
    This converts the failure class from "LLM hallucinated a citation" to
    "LLM paraphrased a real extract" (per the Improvement 2 plan, 2026-05-19).

    Returns
    -------
    list of dicts with keys: ``slug``, ``title``, ``blurb``, ``overlap_count``,
    ``overlap_terms`` (sorted, capped at 12), ``score`` (= overlap_count for
    transparency). Empty list when query has no usable tokens or no entry
    scores above ``min_overlap``.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []
    scored: list[tuple[int, LiteratureEntry, set[str]]] = []
    for entry in corpus:
        body_tokens = set(_tokenize(entry.text))
        overlap = query_tokens & body_tokens
        if len(overlap) < min_overlap:
            continue
        scored.append((len(overlap), entry, overlap))
    scored.sort(key=lambda t: (-t[0], t[1].slug))
    return [
        {
            "slug": entry.slug,
            "title": entry.title,
            "blurb": entry.blurb,
            "overlap_count": score,
            "overlap_terms": sorted(overlap)[:12],
            "score": score,
        }
        for score, entry, overlap in scored[:top_k]
    ]


def format_search_results_for_llm(
    results: Sequence[dict[str, object]], *, max_bytes: int = 4000
) -> str:
    """Render top-K search hits as a compact LLM-context block.

    Format per hit:
        - <slug> :: <title> [overlap=N: term1, term2, ...]
          <blurb>

    Truncates total output to ``max_bytes`` so the targeted block stays
    bounded even on a hypothetically deeper corpus. Order is preserved
    from ``search_corpus`` (rank-descending).
    """
    if not results:
        return ""
    lines = ["TARGETED LITERATURE SEARCH (top-K extracts ranked by overlap with query)"]
    lines.append("=" * 80)
    used = sum(len(line.encode("utf-8")) for line in lines) + 2
    for r in results:
        terms_val = r.get("overlap_terms") or []
        if isinstance(terms_val, (list, tuple)):
            terms = ", ".join(str(t) for t in terms_val)
        else:
            terms = str(terms_val)
        block = (
            f"- {r.get('slug')} :: {r.get('title')} "
            f"[overlap={r.get('overlap_count')}: {terms}]\n"
            f"  {r.get('blurb')}\n"
        )
        nbytes = len(block.encode("utf-8"))
        if used + nbytes > max_bytes:
            break
        lines.append(block.rstrip())
        used += nbytes
    return "\n".join(lines)


__all__ = [
    "LiteratureEntry",
    "citation_exists",
    "corpus_summary_for_llm",
    "find_citation_entries",
    "format_search_results_for_llm",
    "load_corpus",
    "search_corpus",
    "verify_citation_content",
]
