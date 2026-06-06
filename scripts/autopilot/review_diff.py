#!/usr/bin/env python3
"""Post-build diff review for the headless autopilot runner.

Reviews the working-tree (or a commit-range) diff and emits a JSON findings
object the runner consumes to decide whether a repair pass is needed and whether
the work is safe to commit.

What it does:
- `git diff --stat` + changed-file list.
- Risk label per file via `tier_guard.classify_path` ("A" reversible /
  "B" high-risk capital/schema-touching).
- Full diff for high-risk (Tier-B) files; a compact snippet for normal files.
- Dedupe by diff hash: each file's per-hunk content is hashed; an identical hunk
  seen again (across the build + repair passes) is skipped, so the runner's
  second review does not re-print unchanged work.

Output JSON shape:
{
  "files": [ {"path","tier","reason","added","removed","snippet"} ],
  "high_risk": ["path", ...],          # Tier-B files present in the diff
  "new_hunks": int,                     # hunks not seen in --seen-hashes
  "findings": ["..."],                  # human-readable review notes
  "commit_safe": bool                   # False if any high_risk file present
}

The runner persists the emitted hunk hashes and feeds them back via
`--seen-hashes` on the post-repair review so dedupe spans passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "autopilot"))

SNIPPET_MAX_LINES = 20  # normal (Tier-A) files: cap the printed snippet


def _git(args: list[str]) -> str:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return r.stdout if r.returncode == 0 else ""
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def _untracked_files() -> list[str]:
    """New (untracked) files. `git diff HEAD` does NOT show these, but the
    runner commits with `git add -A`, so they MUST be reviewed too."""
    out = _git(["ls-files", "--others", "--exclude-standard"])
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _changed_files(diff_base: str) -> list[str]:
    out = _git(["diff", "--name-only", diff_base])
    tracked = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # Include untracked files (deduped, order-stable: tracked first).
    seen = set(tracked)
    result = list(tracked)
    for u in _untracked_files():
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _file_diff(diff_base: str, path: str) -> str:
    out = _git(["diff", diff_base, "--", path])
    if out.strip():
        return out
    # Untracked file: synthesize a simple all-added unified hunk by reading the
    # file. (git diff --no-index against os.devnull is unreliable on Windows —
    # it emits a non-UTF8 byte that crashes the subprocess decoder — so we build
    # the hunk directly, which is portable and keeps @@/snippet/dedupe working.)
    try:
        p = PROJECT_ROOT / path
        if not p.is_file():
            return ""
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return ""
        body = "\n".join(f"+{ln}" for ln in lines)
        return f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}"
    except OSError:
        return ""


def _numstat(diff_base: str, path: str) -> tuple[int, int]:
    out = _git(["diff", "--numstat", diff_base, "--", path])
    for ln in out.splitlines():
        parts = ln.split("\t")
        if len(parts) >= 3:
            added = 0 if parts[0] == "-" else int(parts[0] or 0)
            removed = 0 if parts[1] == "-" else int(parts[1] or 0)
            return added, removed
    # Untracked file: count lines as added.
    try:
        p = PROJECT_ROOT / path
        if p.is_file():
            n = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))
            return n, 0
    except OSError:
        pass
    return 0, 0


def _hunks(file_diff: str) -> list[str]:
    """Split a unified file diff into per-hunk text blocks (each starts @@)."""
    hunks: list[str] = []
    current: list[str] = []
    for line in file_diff.splitlines():
        if line.startswith("@@"):
            if current:
                hunks.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        hunks.append("\n".join(current))
    return hunks


def _hunk_hash(path: str, hunk: str) -> str:
    h = hashlib.sha256()
    h.update(path.encode("utf-8"))
    h.update(b"\0")
    h.update(hunk.encode("utf-8"))
    return h.hexdigest()[:16]


def review(diff_base: str, seen_hashes: set[str]) -> dict:
    from tier_guard import classify_path  # type: ignore

    files_out: list[dict] = []
    high_risk: list[str] = []
    findings: list[str] = []
    emitted_hashes: list[str] = []
    new_hunk_count = 0

    for path in _changed_files(diff_base):
        tier, reason = classify_path(path)
        added, removed = _numstat(diff_base, path)
        fdiff = _file_diff(diff_base, path)

        # Dedupe by hunk hash.
        fresh_hunks: list[str] = []
        for hunk in _hunks(fdiff):
            hh = _hunk_hash(path, hunk)
            if hh in seen_hashes:
                continue
            seen_hashes.add(hh)
            emitted_hashes.append(hh)
            fresh_hunks.append(hunk)
            new_hunk_count += 1

        if not fresh_hunks:
            continue  # nothing new for this file across passes

        fresh_diff = "\n".join(fresh_hunks)
        if tier == "B":
            high_risk.append(path)
            snippet = fresh_diff  # full diff for high-risk files
            findings.append(
                f"HIGH-RISK: {path} ({reason}) changed +{added}/-{removed}. "
                "Autopilot should NOT have edited a Tier-B file — review before commit."
            )
        else:
            lines = fresh_diff.splitlines()
            snippet = "\n".join(lines[:SNIPPET_MAX_LINES])
            if len(lines) > SNIPPET_MAX_LINES:
                snippet += f"\n... (+{len(lines) - SNIPPET_MAX_LINES} more lines)"

        files_out.append(
            {
                "path": path,
                "tier": tier,
                "reason": reason,
                "added": added,
                "removed": removed,
                "snippet": snippet,
            }
        )

    commit_safe = len(high_risk) == 0
    if not files_out and new_hunk_count == 0:
        findings.append("No new changes to review (all hunks already seen).")

    return {
        "files": files_out,
        "high_risk": high_risk,
        "new_hunks": new_hunk_count,
        "emitted_hashes": emitted_hashes,
        "findings": findings,
        "commit_safe": commit_safe,
    }


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Autopilot post-build diff review.")
    ap.add_argument(
        "--diff-base",
        default="HEAD",
        help="git diff base (default HEAD = uncommitted working tree)",
    )
    ap.add_argument(
        "--seen-hashes",
        default=None,
        help="Path to a JSON file of previously-emitted hunk hashes (dedupe across passes).",
    )
    args = ap.parse_args(argv)

    seen: set[str] = set()
    if args.seen_hashes and Path(args.seen_hashes).exists():
        try:
            seen = set(json.loads(Path(args.seen_hashes).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            seen = set()

    result = review(args.diff_base, seen)

    # Persist the (now-grown) seen set back for the next pass.
    if args.seen_hashes:
        try:
            Path(args.seen_hashes).write_text(json.dumps(sorted(seen)), encoding="utf-8")
        except OSError:
            pass

    print(json.dumps(result, indent=2))
    # Exit 2 signals "repair needed / not commit-safe" for shell consumers.
    return 0 if result["commit_safe"] else 2


if __name__ == "__main__":
    sys.exit(_main())
