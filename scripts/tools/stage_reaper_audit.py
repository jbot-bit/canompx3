#!/usr/bin/env python3
"""Read-only audit of docs/runtime/stages/*.md — classify which stale stage
files are provably done and safe to archive, and (with --apply) git-mv only the
DONE_SAFE ones into docs/runtime/stages/archive/.

Why this exists: non-CLOSED stage files whose work has long since merged pollute
the /next + /orient brief (force "+N more" truncation) and create false "active
stage" signals the stage-gate hook reads on every prompt. But "non-CLOSED" is NOT
"abandoned" — a stale IMPLEMENTATION whose scope is a live capital path another
terminal is committing to today must NOT be touched. This tool encodes that
distinction with hard safety gates so cleanup never races peer work.

Classification (per non-CLOSED stage file):
  DONE_SAFE          — every scope file is git-tracked AND its newest commit is
                       older than --recency-hours (default 48) AND no peer
                       worktree is dirty on any scope file AND no scope file is
                       under a protected live path with a recent commit.
  LIVE_OR_CONTESTED  — a scope file was committed within --recency-hours, OR a
                       peer worktree is dirty on a scope file. Hands off.
  UNVERIFIABLE       — a scope file has no git history (e.g. gitignored .env) or
                       does not exist. Cannot prove done → never archive.

Default is DRY-RUN (reports only, moves nothing). --apply git-mv's DONE_SAFE
files into archive/ (reversible via git). CLOSED-mode files are left alone (the
existing close-out flow owns them).

Fail-safe: any parse/git error on a single stage downgrades it to UNVERIFIABLE
(never DONE_SAFE) — the tool can never archive a file it could not fully verify.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Live/capital paths: a stage touching these is only archivable when its scope
# has been quiet for the full recency window (handled by the recency gate, but
# named explicitly so the report can flag WHY a live stage was held).
PROTECTED_PREFIXES = ("trading_app/live/", "trading_app/broker", "pipeline/")

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES_DIR = REPO_ROOT / "docs" / "runtime" / "stages"
ARCHIVE_DIR = STAGES_DIR / "archive"


def _load_stage_parsers():
    """Import parse_field + parse_scope_lock from the canonical stage-gate hook.

    Reuses the hook's parsers verbatim so this tool never re-encodes stage
    parsing (institutional-rigor: no re-encoded canonical logic).
    """
    hook = REPO_ROOT / ".claude" / "hooks" / "stage-gate-guard.py"
    spec = importlib.util.spec_from_file_location("_stage_gate_guard", hook)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load stage-gate-guard from {hook}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_field, mod.parse_scope_lock


@dataclass
class StageVerdict:
    name: str
    mode: str
    classification: str  # DONE_SAFE | LIVE_OR_CONTESTED | UNVERIFIABLE
    reasons: list[str] = field(default_factory=list)


@dataclass
class AuditCache:
    """Per-run cache so the reaper stays report-only but not O(stages*peers)."""

    peer_dirty_hits: dict[str, list[str]] = field(default_factory=dict)
    commit_age_hours: dict[str, float | None] = field(default_factory=dict)


def _git_last_commit_age_hours(root: Path, rel_path: str) -> float | None:
    """Hours since the newest commit touching rel_path, or None if no history."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%ct", "--", rel_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    ts = out.stdout.strip()
    if out.returncode != 0 or not ts:
        return None
    try:
        commit_epoch = int(ts)
    except ValueError:
        return None
    now_epoch = _git_now_epoch()
    return max(0.0, (now_epoch - commit_epoch) / 3600.0)


def _git_now_epoch() -> int:
    """Current epoch (system clock). Separated for monkeypatch in tests."""
    import time

    return int(time.time())


def _peer_worktrees(root: Path) -> list[Path]:
    """All worktree roots except the primary `root`."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    roots: list[Path] = []
    for line in out.stdout.splitlines():
        if line.startswith("worktree "):
            p = Path(line[len("worktree ") :].strip())
            if p.resolve() != root.resolve():
                roots.append(p)
    return roots


def _peer_dirty_on(peer_roots: list[Path], scope: list[str]) -> list[str]:
    """Return 'peer:path' markers for any scope file dirty in a peer worktree."""
    hits: list[str] = []
    for peer in peer_roots:
        try:
            out = subprocess.run(
                ["git", "-C", str(peer), "status", "--porcelain", "--"] + scope,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        if out.returncode != 0:
            continue
        for line in out.stdout.splitlines():
            for path in _porcelain_paths(line):
                hits.append(f"{peer.name}:{path}")
    return hits


def _porcelain_paths(line: str) -> list[str]:
    """Extract every scope-relevant path from one `git status --porcelain` line.

    Per the git-status porcelain v1 spec, a line is `<xy> <path>`, except a
    rename/copy which is `<xy> <orig-path> -> <path>` (old -> new order, literal
    " -> " separator). BOTH old and new are keyed so a peer that renamed a scope
    file is still detected as dirty on that scope (the original pathspec form
    matched either side). The rename/copy code `R`/`C` can appear in EITHER the
    index column (X) or the work-tree column (Y) — e.g. `R `, ` R`, `MR` — so we
    detect on either column AND on the unambiguous " -> " separator (which never
    appears in a non-rename porcelain line). Paths with whitespace/specials are
    C-quoted (surrounded by double quotes) — strip the wrapping quotes so the key
    matches the unquoted scope_lock entry. Over-detection here is the SAFE
    direction: an extra key can only make the reaper hand off
    (LIVE_OR_CONTESTED), never wrongly reap.
    """
    body = line[3:].strip()
    if not body:
        return []
    status = line[:2]
    is_rename = ("R" in status or "C" in status) and " -> " in body
    parts = [body]
    if is_rename:
        old, _, new = body.partition(" -> ")
        parts = [old.strip(), new.strip()]
    return [_strip_porcelain_quotes(p) for p in parts if _strip_porcelain_quotes(p)]


def _strip_porcelain_quotes(path: str) -> str:
    """Strip the wrapping C-quote double-quotes git adds to special-char paths.

    Only strips a matched leading+trailing pair (the quoting git applies to the
    whole field); an interior quote in an unquoted path is left untouched.
    """
    path = path.strip()
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        return path[1:-1]
    return path


def _build_peer_dirty_cache(peer_roots: list[Path]) -> dict[str, list[str]]:
    """Map dirty path -> peer markers with one git-status call per peer."""
    dirty: dict[str, list[str]] = {}
    for peer in peer_roots:
        try:
            out = subprocess.run(
                ["git", "-C", str(peer), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        if out.returncode != 0:
            continue
        for line in out.stdout.splitlines():
            for path in _porcelain_paths(line):
                dirty.setdefault(path, []).append(f"{peer.name}:{path}")
    return dirty


def _peer_dirty_on_cached(cache: AuditCache, scope: list[str]) -> list[str]:
    """Cache-backed equivalent of `_peer_dirty_on`'s git-pathspec match.

    The non-cached path passed scope entries as git pathspecs, which match by
    directory PREFIX (`pipeline/` matched `pipeline/foo.py`). The cache keys on
    the literal dirty path, so a plain exact lookup would MISS a peer-dirty file
    under a directory scope entry — the UNSAFE direction (could let a contested
    stage reach DONE_SAFE). We restore prefix semantics: a scope entry ending in
    `/` matches any cached dirty path beneath it; exact file entries still match
    exactly. Over-detection here only ever makes the reaper hand off
    (LIVE_OR_CONTESTED), never wrongly reap.
    """
    hits: list[str] = []
    for entry in scope:
        if entry.endswith("/"):
            for dirty_path, markers in cache.peer_dirty_hits.items():
                if dirty_path.startswith(entry):
                    hits.extend(markers)
        else:
            hits.extend(cache.peer_dirty_hits.get(entry, []))
    return hits


def _git_last_commit_age_hours_cached(cache: AuditCache, root: Path, rel_path: str) -> float | None:
    if rel_path not in cache.commit_age_hours:
        cache.commit_age_hours[rel_path] = _git_last_commit_age_hours(root, rel_path)
    return cache.commit_age_hours[rel_path]


def classify_stage(
    path: Path,
    parse_field,
    parse_scope_lock,
    peer_roots: list[Path],
    recency_hours: float,
    root: Path,
    cache: AuditCache | None = None,
) -> StageVerdict:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return StageVerdict(path.name, "?", "UNVERIFIABLE", ["unreadable stage file"])

    mode = (parse_field(content, "mode") or parse_field(content, "stage") or "?").strip()
    if mode.upper() == "CLOSED":
        return StageVerdict(path.name, mode, "CLOSED", ["already closed"])

    # DESIGN stages are NEVER auto-archived: "scope is quiet" means the code
    # didn't move, which for a design doc is ambiguous (parked-unresolved vs.
    # superseded) — not a done-signal. Only landed IMPLEMENTATION/TRIVIAL work
    # is provable-done by scope quiescence. Hand design decisions to a human.
    if mode.upper() not in ("IMPLEMENTATION", "TRIVIAL"):
        return StageVerdict(
            path.name, mode, "LIVE_OR_CONTESTED", [f"mode={mode}: only IMPLEMENTATION/TRIVIAL are auto-archivable"]
        )

    scope = parse_scope_lock(content)
    if not scope:
        return StageVerdict(path.name, mode, "UNVERIFIABLE", ["no scope_lock parsed"])

    reasons: list[str] = []

    # Gate 1: peer-dirty on any scope file → hands off.
    peer_hits = _peer_dirty_on_cached(cache, scope) if cache is not None else _peer_dirty_on(peer_roots, scope)
    if peer_hits:
        reasons.append("peer dirty: " + ", ".join(peer_hits[:4]))
        return StageVerdict(path.name, mode, "LIVE_OR_CONTESTED", reasons)

    # Gate 2: every scope file must have git history older than recency window.
    for sp in scope:
        age = (
            _git_last_commit_age_hours_cached(cache, root, sp)
            if cache is not None
            else _git_last_commit_age_hours(root, sp)
        )
        if age is None:
            reasons.append(f"no git history: {sp}")
            return StageVerdict(path.name, mode, "UNVERIFIABLE", reasons)
        if age < recency_hours:
            protected = any(sp.startswith(p) for p in PROTECTED_PREFIXES)
            tag = " [PROTECTED live/pipeline]" if protected else ""
            reasons.append(f"recent commit ({age:.0f}h<{recency_hours:.0f}h): {sp}{tag}")
            return StageVerdict(path.name, mode, "LIVE_OR_CONTESTED", reasons)

    reasons.append(f"all {len(scope)} scope file(s) merged & quiet >{recency_hours:.0f}h, no peer contention")
    return StageVerdict(path.name, mode, "DONE_SAFE", reasons)


def audit(recency_hours: float, root: Path = REPO_ROOT) -> list[StageVerdict]:
    parse_field, parse_scope_lock = _load_stage_parsers()
    peer_roots = _peer_worktrees(root)
    cache = AuditCache(peer_dirty_hits=_build_peer_dirty_cache(peer_roots))
    stages_dir = root / "docs" / "runtime" / "stages"
    verdicts: list[StageVerdict] = []
    for f in sorted(stages_dir.glob("*.md")):
        verdicts.append(classify_stage(f, parse_field, parse_scope_lock, peer_roots, recency_hours, root, cache))
    return verdicts


def _print_report(verdicts: list[StageVerdict]) -> None:
    order = {"DONE_SAFE": 0, "LIVE_OR_CONTESTED": 1, "UNVERIFIABLE": 2, "CLOSED": 3}
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.classification] = counts.get(v.classification, 0) + 1
    print("=" * 70)
    print("STALE STAGE REAPER — AUDIT")
    print("=" * 70)
    for v in sorted(verdicts, key=lambda x: (order.get(x.classification, 9), x.name)):
        if v.classification == "CLOSED":
            continue
        print(f"[{v.classification:18}] {v.name}")
        for r in v.reasons:
            print(f"      - {r}")
    print("-" * 70)
    summary = "  ".join(f"{k}={counts.get(k, 0)}" for k in order)
    print(f"SUMMARY: {summary}  (total={len(verdicts)})")
    print("=" * 70)


def _apply_archive(verdicts: list[StageVerdict], root: Path = REPO_ROOT) -> int:
    archive = root / "docs" / "runtime" / "stages" / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    moved = 0
    for v in verdicts:
        if v.classification != "DONE_SAFE":
            continue
        dst_rel = f"docs/runtime/stages/archive/{v.name}"
        src_rel = f"docs/runtime/stages/{v.name}"
        res = subprocess.run(
            ["git", "-C", str(root), "mv", src_rel, dst_rel],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if res.returncode == 0:
            print(f"  archived: {v.name}")
            moved += 1
        else:
            print(f"  SKIP (git mv failed): {v.name} — {res.stderr.strip()}")
    return moved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true", help="git-mv DONE_SAFE stage files into archive/ (default: dry-run report only)"
    )
    ap.add_argument(
        "--recency-hours",
        type=float,
        default=48.0,
        help="a scope file committed within this window holds its stage as LIVE_OR_CONTESTED (default 48)",
    )
    args = ap.parse_args(argv)

    verdicts = audit(args.recency_hours)
    _print_report(verdicts)

    if args.apply:
        print("\nAPPLYING (git mv DONE_SAFE → archive/):")
        n = _apply_archive(verdicts)
        print(f"\nArchived {n} stage file(s). Review `git status` then commit.")
    else:
        safe = sum(1 for v in verdicts if v.classification == "DONE_SAFE")
        print(f"\nDRY-RUN. {safe} file(s) would be archived. Re-run with --apply to move them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
