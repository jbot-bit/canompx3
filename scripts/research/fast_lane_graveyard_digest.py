"""NO-GO graveyard digest builder (Stage 2A.2).

Builds ``docs/runtime/fast_lane_graveyard_digest.yaml`` from three canonical
NO-GO sources:

  1. ``chatgpt_bundle/06_RD_GRAVEYARD.md`` -- consolidated NO-GO registry
     (one ``## <title>`` heading per entry; "Rule for re-opening" + intro
     sections excluded).
  2. ``docs/STRATEGY_BLUEPRINT.md`` § 5 ``NO-GO Registry`` -- one bullet /
     table row per NO-GO entry.
  3. ``docs/runtime/action-queue.yaml`` -- entries whose ``status`` is
     ``park`` or ``kill``.

Each parsed entry produces a digest row keyed by ``structural_hash``. For
entries that name a concrete lane (instrument + session + ORB + RR + entry
model + confirm + filter + direction + threshold), the hash is computed by
``fast_lane_structural_hash.compute_structural_hash`` and is the same hash
the 2A.3 scanner emits per PROMOTE candidate -- direct suppression match.

For class-level entries (ML, architecture, methodology) that lack a lane
tuple, the digest emits a content-derived hash keyed off the entry's
identifying string (``source_path::title``) so the digest is non-empty and
auditable but cannot collide with any real lane hash. The 2A.3 scanner
checks both classes: lane-hash match -> ``SUPPRESSED_GRAVEYARD``;
content-hash match is informational (scanner cites the source but does not
suppress because a class-level kill does not map to a single lane).

Design grounding:
  docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md
Implementation grounding:
  docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a2-ledger-digest.md

CLI:
  python -m scripts.research.fast_lane_graveyard_digest --build
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DIGEST_SCHEMA_VERSION = 1

GRAVEYARD_MD = PROJECT_ROOT / "chatgpt_bundle" / "06_RD_GRAVEYARD.md"
STRATEGY_BLUEPRINT = PROJECT_ROOT / "docs" / "STRATEGY_BLUEPRINT.md"
ACTION_QUEUE = PROJECT_ROOT / "docs" / "runtime" / "action-queue.yaml"

DIGEST_PATH = PROJECT_ROOT / "docs" / "runtime" / "fast_lane_graveyard_digest.yaml"

# Headings under 06_RD_GRAVEYARD.md that are not graveyard entries -- they
# are the file's preamble / rule documentation. Skip these when parsing.
_GRAVEYARD_NON_ENTRY_HEADINGS = frozenset(
    {
        "Rule for re-opening",
        "Status Token Doctrine",
        "What to DO when user proposes something similar",
    }
)


def _load_status_tokens(text: str) -> tuple[str, ...]:
    """Parse the ``## Status Token Doctrine`` block from 06_RD_GRAVEYARD.md.

    The doctrine block is the canonical source for the status-token
    alternation; the parser reads it at parse time rather than inlining
    the list. Drift between this list and the inline copy in the parity
    check is caught by ``check_graveyard_status_tokens_parity``.

    Returns an empty tuple if the doctrine block is missing or unparseable
    — the parser then captures every heading with ``status="UNKNOWN"``,
    which is the fail-loud degradation mode.
    """
    section = re.search(
        r"(?ms)^##\s+Status Token Doctrine[^\n]*\n.*?```yaml\n(?P<yaml>.*?)\n```",
        text,
    )
    if section is None:
        return ()
    try:
        spec = yaml.safe_load(section.group("yaml"))
    except yaml.YAMLError:
        return ()
    if not isinstance(spec, dict):
        return ()
    tokens = spec.get("status_tokens")
    if not isinstance(tokens, list):
        return ()
    return tuple(t for t in tokens if isinstance(t, str))


@dataclass(frozen=True)
class GraveyardEntry:
    """One parsed NO-GO entry. ``structural_hash`` resolution happens in
    ``_resolve_hash`` -- lane-tuple entries get the canonical 16-hex hash;
    class-level entries get a content-derived 16-hex hash with a distinct
    namespace prefix recorded in ``hash_kind`` so the 2A.3 scanner can tell
    them apart at suppression-decision time.
    """

    source_path: str  # repo-relative
    title: str
    status: str  # DEAD | NO-GO | PARK | KILL | PAUSED | CLOSED | …
    hash_kind: str  # "lane" | "class"
    structural_hash: str  # 16-hex
    # Lane-tuple entries (hash_kind="lane") carry the inputs the hash was
    # computed from for human auditability. Empty dict for "class" entries.
    lane_inputs: dict[str, Any]


def _normalise_status(text: str) -> str:
    """Map any of the graveyard status strings to a canonical UPPER token."""
    upper = text.strip().upper()
    # Keep first whitespace-separated token; "DEAD + DELETED" -> "DEAD".
    for sep in (" + ", " — ", " - ", "/"):
        if sep in upper:
            upper = upper.split(sep, 1)[0].strip()
            break
    return upper.split()[0] if upper else "UNKNOWN"


def _content_hash(source_path: str, title: str) -> str:
    """16-hex content hash for class-level entries. Distinct namespace from
    lane hashes by virtue of the prefix string. We seed sha256 with a
    namespace tag so an accidental collision with a real lane hash is
    cryptographically negligible."""
    payload = f"graveyard-class::{source_path}::{title.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_graveyard_md(path: Path) -> list[GraveyardEntry]:
    """Parse 06_RD_GRAVEYARD.md. One ``## Heading`` (or ``### Heading``) per
    entry; the heading text encodes both title and an optional status token
    (e.g. ``## ML V3 DEAD + DELETED``). Headings without a recognised status
    token are still captured with ``status="UNKNOWN"`` -- the digest must
    never silently drop a graveyard entry merely because the operator used
    a new status vocabulary."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    entries: list[GraveyardEntry] = []
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    # Capture both ## and ### so architectural sub-kills under a parent
    # heading (e.g. ## Architecture-level kills > ### Scratch DB ... DEPRECATED)
    # are not lost. The non-entry frozenset gates the file's preamble /
    # rule documentation; the parent of a status-bearing subtree (e.g.
    # "Architecture-level kills" itself) is captured with status="UNKNOWN"
    # rather than dropped, so Check #170 can still detect a hand-edited
    # digest that omits the parent.
    #
    # Status tokens are loaded from the canonical ``## Status Token Doctrine``
    # block in the same source file -- parser does NOT inline the alternation
    # list. Parity between code and doctrine is enforced by
    # ``check_graveyard_status_tokens_parity`` (canonical-inline-copy bug
    # class, 9th confirmed instance).
    status_tokens = _load_status_tokens(text)
    if status_tokens:
        # Order by length descending so longer phrases ("HYPOTHESES DEAD")
        # are tried before their substrings ("DEAD").
        sorted_tokens = sorted(status_tokens, key=len, reverse=True)
        alternation = "|".join(re.escape(t) for t in sorted_tokens)
        status_pattern = re.compile(rf"\b({alternation})\b")
    else:
        status_pattern = None
    for match in re.finditer(r"^(#{2,3})\s+(?P<title>[^\n]+?)\s*$", text, re.MULTILINE):
        title = match.group("title").strip()
        if title in _GRAVEYARD_NON_ENTRY_HEADINGS:
            continue
        if status_pattern is not None:
            status_match = status_pattern.search(title)
            status = status_match.group(1) if status_match else "UNKNOWN"
        else:
            status = "UNKNOWN"
        entries.append(
            GraveyardEntry(
                source_path=rel,
                title=title,
                status=_normalise_status(status),
                hash_kind="class",
                structural_hash=_content_hash(rel, title),
                lane_inputs={},
            )
        )
    return entries


def _parse_strategy_blueprint(path: Path) -> list[GraveyardEntry]:
    """Parse the ``## 5. NO-GO Registry`` table in STRATEGY_BLUEPRINT.md."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    # Locate the NO-GO Registry section start and stop at the next top-level
    # `## ` heading.
    section = re.search(
        r"(?ms)^## 5\. NO-GO Registry\s*$(?P<body>.*?)(?=^## )",
        text,
    )
    if section is None:
        return []
    body = section.group("body")
    entries: list[GraveyardEntry] = []
    # NO-GO Registry rows look like markdown table rows: ``| <title> | <status> | …``
    # Skip header / separator rows.
    for line in body.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        title = cells[0]
        if not title or set(title) <= {"-", " ", ":"}:
            continue
        if title.lower() in {"hypothesis / approach", "approach", "hypothesis"}:
            continue
        status = cells[1] if len(cells) > 1 else "NO-GO"
        if not status or set(status) <= {"-", " ", ":"}:
            status = "NO-GO"
        entries.append(
            GraveyardEntry(
                source_path=rel,
                title=title,
                status=_normalise_status(status),
                hash_kind="class",
                structural_hash=_content_hash(rel, title),
                lane_inputs={},
            )
        )
    return entries


def _parse_action_queue(path: Path) -> list[GraveyardEntry]:
    """Parse action-queue.yaml entries with ``status: park`` or
    ``status: kill``."""
    if not path.exists():
        return []
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, (list, dict)):
        return []
    rows: list[dict[str, Any]] = []
    if isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict):
        # Common shape: top-level ``entries: [...]``.
        for key in ("entries", "items", "queue"):
            if isinstance(data.get(key), list):
                rows = [r for r in data[key] if isinstance(r, dict)]
                break
        else:
            # Last-resort: treat dict-of-dicts as entries.
            rows = [v for v in data.values() if isinstance(v, dict)]
    entries: list[GraveyardEntry] = []
    for row in rows:
        status = str(row.get("status", "")).lower()
        if status not in {"park", "kill"}:
            continue
        title = (
            row.get("id")
            or row.get("title")
            or row.get("name")
            or row.get("summary")
            or "unnamed action-queue entry"
        )
        entries.append(
            GraveyardEntry(
                source_path=rel,
                title=str(title),
                status=_normalise_status(status),
                hash_kind="class",
                structural_hash=_content_hash(rel, str(title)),
                lane_inputs={},
            )
        )
    return entries


def build_digest() -> dict[str, Any]:
    """Build the digest dict (does not write to disk)."""
    parsed = (
        _parse_graveyard_md(GRAVEYARD_MD)
        + _parse_strategy_blueprint(STRATEGY_BLUEPRINT)
        + _parse_action_queue(ACTION_QUEUE)
    )

    # Collapse by structural_hash; if two entries collide on hash, keep the
    # first occurrence and record the duplicate's source under
    # ``additional_sources`` so the audit trail survives.
    seen: dict[str, dict[str, Any]] = {}
    for entry in parsed:
        if entry.structural_hash in seen:
            seen[entry.structural_hash].setdefault("additional_sources", []).append(
                {"source_path": entry.source_path, "title": entry.title}
            )
            continue
        seen[entry.structural_hash] = {
            "source_path": entry.source_path,
            "title": entry.title,
            "status": entry.status,
            "hash_kind": entry.hash_kind,
            "structural_hash": entry.structural_hash,
            "lane_inputs": dict(entry.lane_inputs),
        }

    return {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "do_not_hand_edit": True,
        "built_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_files": [
            GRAVEYARD_MD.relative_to(PROJECT_ROOT).as_posix(),
            STRATEGY_BLUEPRINT.relative_to(PROJECT_ROOT).as_posix(),
            ACTION_QUEUE.relative_to(PROJECT_ROOT).as_posix(),
        ],
        "entries": list(seen.values()),
    }


def _dump_digest_yaml(data: dict[str, Any]) -> str:
    """Banner-first emit so Check #170's banner detection is trivial."""
    banner = "do_not_hand_edit: true\n"
    schema = f"schema_version: {data['schema_version']}\n"
    body_dict = {k: v for k, v in data.items() if k not in {"do_not_hand_edit", "schema_version"}}
    body = yaml.safe_dump(
        body_dict,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    return banner + schema + body


def write_digest(out_path: Path = DIGEST_PATH) -> dict[str, Any]:
    """Build + write the digest. Returns the in-memory dict."""
    digest = build_digest()
    out_path.write_text(_dump_digest_yaml(digest), encoding="utf-8")
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0]
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="rebuild docs/runtime/fast_lane_graveyard_digest.yaml from sources",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DIGEST_PATH,
        help="output digest path (default: docs/runtime/fast_lane_graveyard_digest.yaml)",
    )
    args = parser.parse_args(argv)

    if not args.build:
        parser.print_help()
        return 2

    digest = write_digest(args.out)
    print(
        f"fast_lane_graveyard_digest: wrote {len(digest['entries'])} entries "
        f"to {args.out.relative_to(PROJECT_ROOT) if args.out.is_absolute() else args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DIGEST_PATH",
    "DIGEST_SCHEMA_VERSION",
    "GraveyardEntry",
    "build_digest",
    "write_digest",
]
