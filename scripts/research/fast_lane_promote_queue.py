"""FAST_LANE v5.1 PROMOTE queue scanner.

Reconstructs the PROMOTE-queue state from on-disk artifacts:
  - PROMOTE result MDs under ``docs/audit/results/*fast-lane*.md``
  - revocation sidecars (``<base>.revocation.md`` next to the result MD)
  - heavyweight pre-regs under ``docs/audit/hypotheses/`` whose
    ``scope.strategy_id`` matches a PROMOTE result's strategy_id
  - park entries in ``docs/runtime/action-queue.yaml``

The queue is **derived state**. The cache file ``docs/runtime/promote_queue.yaml``
is rebuilt from these sources on every run; ``--write`` refreshes the cache,
``--dry-run`` (default) prints the would-be queue and the diff vs the cache.

Drift check ``check_fast_lane_promote_orphans`` (Check 157, added in this
landing) reconstructs independently and fails the build if the cache is
stale, hand-edited, or any entry is ERROR.

Per-direction sanity gate
-------------------------
Re-applies v5.1 thresholds to the ``## Directional breakdown`` table that
every v5.1 result MD carries.  When a pooled PROMOTE has BOTH per-direction
sub-stats failing v5.1 gates as standalone (t<2.5, N<50, or fire-rate
outside [0.05, 0.95]), the pooled PROMOTE is a sample-doubling artifact and
the cell is flagged ``REVOKE_RECOMMENDED``.  Operator authors a
``.revocation.md`` sidecar; on next scan the entry moves QUEUED -> REVOKED.

Status enum (no UNKNOWN)
------------------------
QUEUED      PROMOTE + no revocation sidecar + no heavyweight prereg +
            no park entry + per-direction sanity gate PASS
ESCALATED   PROMOTE + matching heavyweight prereg under docs/audit/hypotheses/
REVOKED     PROMOTE + revocation sidecar present
PARKED      PROMOTE + action-queue.yaml entry naming this strategy_id with park
ERROR       PROMOTE + (missing/unparseable directional breakdown
                       OR per-direction sanity gate fires REVOKE_RECOMMENDED
                       with no revocation sidecar yet)

ERROR forces operator attention - never silently lingers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "docs" / "audit" / "results"
HYPOTHESES_DIR = REPO_ROOT / "docs" / "audit" / "hypotheses"
ACTION_QUEUE = REPO_ROOT / "docs" / "runtime" / "action-queue.yaml"
QUEUE_CACHE = REPO_ROOT / "docs" / "runtime" / "promote_queue.yaml"

FAST_LANE_RESULT_GLOB = "*fast-lane*.md"

# v5.1 thresholds re-applied to per-direction sub-stats.
# Mirrors docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml lines 102-145.
T_PROMOTE_FLOOR = 3.0
T_KILL_FLOOR = 2.5
EXPR_FLOOR = 0.0
N_FLOOR = 50
FIRE_MIN = 0.05
FIRE_MAX = 0.95


STATUS_VALUES = ("QUEUED", "ESCALATED", "REVOKED", "PARKED", "ERROR")


@dataclass
class PromoteEntry:
    result_md: str
    strategy_id: str
    direction: str
    pooled_t: float
    pooled_expr: float
    pooled_n: int
    pooled_fire: float
    long_n: int
    long_expr: float
    long_t: float
    short_n: int
    short_expr: float
    short_t: float
    pooled_universe_n: int
    long_fire: float
    short_fire: float
    long_side_verdict: str
    short_side_verdict: str
    pooling_artifact: bool
    revocation_sidecar: str | None
    heavyweight_prereg: str | None
    park_entry: str | None
    status: str
    error_reason: str | None


_TITLE_RE = re.compile(r"^#\s+Chordia strict unlock audit\s+\S\s+(?P<sid>\S+)\s*$", re.MULTILINE)
_VERDICT_RE = re.compile(r"^\*\*FAST_LANE verdict:\*\*\s+`(?P<v>PROMOTE|KILL|NEEDS-MORE)`\s*$", re.MULTILINE)
_DIR_FOOTER_RE = re.compile(r"_Scope direction at screen:\s*`'(?P<dir>pooled|long|short)'`", re.MULTILINE)

_SPLIT_IS_RE = re.compile(
    r"^\|\s*IS\s*\|\s*(?P<nu>[-+0-9.]+)\s*\|\s*(?P<nf>[-+0-9.]+)\s*\|\s*(?P<fp>[-+0-9.]+)%\s*"
    r"\|\s*\S+\s*\|\s*\S+\s*\|\s*(?P<expr>[-+0-9.eE]+)\s*\|\s*\S+\s*\|\s*\S+\s*\|\s*(?P<t>[-+0-9.eE]+)\s*\|",
    re.MULTILINE,
)

_DIR_IS_RE = re.compile(
    r"^\|\s*IS\s*\|\s*(?P<ln>[-+0-9.]+)\s*\|\s*(?P<lex>[-+0-9.eE]+)\s*\|\s*(?P<lt>[-+0-9.eEnNaA]+)\s*"
    r"\|\s*(?P<sn>[-+0-9.]+)\s*\|\s*(?P<sex>[-+0-9.eEnNaA]+)\s*\|\s*(?P<st>[-+0-9.eEnNaA]+)\s*\|",
    re.MULTILINE,
)


def _parse_float(s: str) -> float:
    s = s.strip().lower()
    if s in {"nan", "", "n/a"}:
        return float("nan")
    return float(s)


def _parse_int(s: str) -> int:
    s = s.strip()
    if s == "":
        return 0
    return int(float(s))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rel_to_repo(path: Path) -> str:
    """Return repo-relative path string; fall back to absolute when outside
    REPO_ROOT (test fixtures under tmp_path)."""
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def parse_result_md(path: Path) -> dict[str, Any] | None:
    text = _read_text(path)
    verdict_match = _VERDICT_RE.search(text)
    title_match = _TITLE_RE.search(text)
    direction_match = _DIR_FOOTER_RE.search(text)
    if not verdict_match or not title_match:
        return None
    return {
        "result_md": _rel_to_repo(path),
        "strategy_id": title_match.group("sid"),
        "verdict": verdict_match.group("v"),
        "direction": direction_match.group("dir") if direction_match else "pooled",
    }


def parse_promote_stats(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    text = _read_text(path)
    is_m = _SPLIT_IS_RE.search(text)
    dir_m = _DIR_IS_RE.search(text)
    if not is_m:
        return None, "Split summary IS row not found"
    if not dir_m:
        return None, "Directional breakdown IS row not found"
    try:
        pooled_universe_n = _parse_int(is_m.group("nu"))
        stats = {
            "pooled_universe_n": pooled_universe_n,
            "pooled_n": _parse_int(is_m.group("nf")),
            "pooled_fire": _parse_float(is_m.group("fp")) / 100.0,
            "pooled_expr": _parse_float(is_m.group("expr")),
            "pooled_t": _parse_float(is_m.group("t")),
            "long_n": _parse_int(dir_m.group("ln")),
            "long_expr": _parse_float(dir_m.group("lex")),
            "long_t": _parse_float(dir_m.group("lt")),
            "short_n": _parse_int(dir_m.group("sn")),
            "short_expr": _parse_float(dir_m.group("sex")),
            "short_t": _parse_float(dir_m.group("st")),
        }
    except ValueError as exc:
        return None, f"Numeric parse failure: {exc}"
    if pooled_universe_n <= 0:
        return None, "pooled_universe_n <= 0; cannot compute per-direction fire-rate"
    stats["long_fire"] = stats["long_n"] / pooled_universe_n
    stats["short_fire"] = stats["short_n"] / pooled_universe_n
    return stats, None


def _side_verdict(t: float, n: int, expr: float, fire: float) -> str:
    if n == 0 or math.isnan(t):
        return "N_A"
    if n < N_FLOOR or fire < FIRE_MIN or fire > FIRE_MAX or expr <= EXPR_FLOOR:
        return "KILL_AS_STANDALONE"
    if t < T_KILL_FLOOR:
        return "KILL_AS_STANDALONE"
    if t < T_PROMOTE_FLOOR:
        return "NEEDS_MORE_AS_STANDALONE"
    return "PROMOTE_AS_STANDALONE"


def per_direction_sanity_gate(stats: dict[str, Any], direction: str) -> tuple[str, str, bool]:
    lv = _side_verdict(stats["long_t"], stats["long_n"], stats["long_expr"], stats["long_fire"])
    sv = _side_verdict(stats["short_t"], stats["short_n"], stats["short_expr"], stats["short_fire"])
    if direction == "pooled" and lv == "KILL_AS_STANDALONE" and sv == "KILL_AS_STANDALONE":
        return lv, sv, True
    return lv, sv, False


def find_revocation_sidecar(result_md: Path) -> Path | None:
    sidecar = result_md.with_name(result_md.stem + ".revocation.md")
    return sidecar if sidecar.exists() else None


_HEAVYWEIGHT_TEMPLATE_EXCLUDES = {"fast_lane_v5.1"}


def find_heavyweight_prereg(strategy_id: str, hypotheses_dir: Path = HYPOTHESES_DIR) -> Path | None:
    if not hypotheses_dir.exists():
        return None
    for candidate in sorted(hypotheses_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        scope = data.get("scope") or {}
        if scope.get("strategy_id") != strategy_id:
            continue
        meta = data.get("metadata") or {}
        template = meta.get("template_version")
        if template in _HEAVYWEIGHT_TEMPLATE_EXCLUDES:
            continue
        return candidate
    return None


def find_park_entry(strategy_id: str, action_queue: Path = ACTION_QUEUE) -> str | None:
    if not action_queue.exists():
        return None
    try:
        data = yaml.safe_load(action_queue.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    entries = data.get("entries") or data.get("items") or []
    if not isinstance(entries, list):
        return None
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if (entry.get("status") or "").lower() != "park":
            continue
        sid = entry.get("strategy_id") or entry.get("lane_id")
        if sid == strategy_id:
            return f"action-queue#{idx}"
    return None


def classify(entry: PromoteEntry) -> tuple[str, str | None]:
    if entry.revocation_sidecar is not None:
        return "REVOKED", None
    if entry.heavyweight_prereg is not None:
        return "ESCALATED", None
    if entry.park_entry is not None:
        return "PARKED", None
    if entry.pooling_artifact:
        return (
            "ERROR",
            "per-direction sanity gate flags pooling artifact "
            "(both directions KILL_AS_STANDALONE); revocation sidecar required",
        )
    return "QUEUED", None


def build_entry(
    path: Path,
    *,
    hypotheses_dir: Path = HYPOTHESES_DIR,
    action_queue: Path = ACTION_QUEUE,
) -> PromoteEntry | None:
    parsed = parse_result_md(path)
    if parsed is None:
        return None
    if parsed["verdict"] != "PROMOTE":
        return None

    stats, parse_err = parse_promote_stats(path)
    if stats is None:
        return PromoteEntry(
            result_md=parsed["result_md"],
            strategy_id=parsed["strategy_id"],
            direction=parsed["direction"],
            pooled_t=float("nan"),
            pooled_expr=float("nan"),
            pooled_n=0,
            pooled_fire=float("nan"),
            long_n=0,
            long_expr=float("nan"),
            long_t=float("nan"),
            short_n=0,
            short_expr=float("nan"),
            short_t=float("nan"),
            pooled_universe_n=0,
            long_fire=float("nan"),
            short_fire=float("nan"),
            long_side_verdict="N_A",
            short_side_verdict="N_A",
            pooling_artifact=False,
            revocation_sidecar=None,
            heavyweight_prereg=None,
            park_entry=None,
            status="ERROR",
            error_reason=f"result MD parse failure: {parse_err}",
        )

    lv, sv, artifact = per_direction_sanity_gate(stats, parsed["direction"])
    sidecar = find_revocation_sidecar(path)
    heavy = find_heavyweight_prereg(parsed["strategy_id"], hypotheses_dir)
    park = find_park_entry(parsed["strategy_id"], action_queue)

    entry = PromoteEntry(
        result_md=parsed["result_md"],
        strategy_id=parsed["strategy_id"],
        direction=parsed["direction"],
        pooled_t=stats["pooled_t"],
        pooled_expr=stats["pooled_expr"],
        pooled_n=stats["pooled_n"],
        pooled_fire=stats["pooled_fire"],
        long_n=stats["long_n"],
        long_expr=stats["long_expr"],
        long_t=stats["long_t"],
        short_n=stats["short_n"],
        short_expr=stats["short_expr"],
        short_t=stats["short_t"],
        pooled_universe_n=stats["pooled_universe_n"],
        long_fire=stats["long_fire"],
        short_fire=stats["short_fire"],
        long_side_verdict=lv,
        short_side_verdict=sv,
        pooling_artifact=artifact,
        revocation_sidecar=(_rel_to_repo(sidecar) if sidecar is not None else None),
        heavyweight_prereg=(_rel_to_repo(heavy) if heavy is not None else None),
        park_entry=park,
        status="ERROR",
        error_reason=None,
    )
    entry.status, entry.error_reason = classify(entry)
    return entry


def scan(
    results_dir: Path | None = None,
    *,
    hypotheses_dir: Path | None = None,
    action_queue: Path | None = None,
) -> list[PromoteEntry]:
    # Resolve module-level defaults at call time so monkeypatching from tests
    # works against scripts.research.fast_lane_promote_queue.RESULTS_DIR etc.
    rd = results_dir if results_dir is not None else RESULTS_DIR
    hd = hypotheses_dir if hypotheses_dir is not None else HYPOTHESES_DIR
    aq = action_queue if action_queue is not None else ACTION_QUEUE
    entries: list[PromoteEntry] = []
    for path in sorted(rd.glob(FAST_LANE_RESULT_GLOB)):
        entry = build_entry(path, hypotheses_dir=hd, action_queue=aq)
        if entry is not None:
            entries.append(entry)
    return entries


def _entry_to_dict(entry: PromoteEntry) -> dict[str, Any]:
    d = asdict(entry)
    for k, v in list(d.items()):
        if isinstance(v, float) and math.isnan(v):
            d[k] = None
    return d


def serialize_queue(entries: list[PromoteEntry]) -> str:
    payload = {
        "schema_version": 1,
        "source": "scripts/research/fast_lane_promote_queue.py",
        "warning": (
            "DERIVED STATE - do not hand-edit. Rebuilt from result MDs + "
            "revocation sidecars + heavyweight preregs + action-queue.yaml "
            "on every scanner run. Drift check #157 reconstructs and diffs."
        ),
        "entries": [_entry_to_dict(e) for e in entries],
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def diff_against_cache(entries: list[PromoteEntry], cache_path: Path = QUEUE_CACHE) -> list[str]:
    if not cache_path.exists():
        return ["(no cache file on disk; first run)"]
    try:
        cached = yaml.safe_load(cache_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return [f"(cache unreadable: {exc})"]
    cached_entries = {e.get("strategy_id"): e for e in (cached.get("entries") or [])}
    fresh_entries = {e.strategy_id: _entry_to_dict(e) for e in entries}

    lines: list[str] = []
    for sid in sorted(set(cached_entries) | set(fresh_entries)):
        c = cached_entries.get(sid)
        f = fresh_entries.get(sid)
        if c is None and f is not None:
            lines.append(f"ADDED   {sid} status={f['status']}")
        elif f is None and c is not None:
            lines.append(f"REMOVED {sid}")
        elif c is not None and f is not None and c.get("status") != f.get("status"):
            lines.append(f"CHANGED {sid} {c.get('status')} -> {f.get('status')}")
    return lines or ["(cache up to date)"]


def render_report(entries: list[PromoteEntry]) -> str:
    lines = [
        "FAST_LANE v5.1 PROMOTE queue",
        "============================",
        f"Total PROMOTE results scanned: {len(entries)}",
        "",
    ]
    by_status: dict[str, list[PromoteEntry]] = {s: [] for s in STATUS_VALUES}
    for e in entries:
        by_status[e.status].append(e)
    for status in STATUS_VALUES:
        bucket = by_status[status]
        lines.append(f"## {status} ({len(bucket)})")
        for e in bucket:
            lines.append(f"  - {e.strategy_id}")
            lines.append(f"      result_md: {e.result_md}")
            lines.append(
                f"      direction={e.direction} pooled_t={e.pooled_t:.3f} "
                f"pooled_n={e.pooled_n} pooled_fire={e.pooled_fire:.4f}"
            )
            lines.append(
                f"      long: t={e.long_t!r} n={e.long_n} fire={e.long_fire:.4f} -> {e.long_side_verdict}"
            )
            lines.append(
                f"      short: t={e.short_t!r} n={e.short_n} fire={e.short_fire:.4f} -> {e.short_side_verdict}"
            )
            if e.revocation_sidecar:
                lines.append(f"      revocation_sidecar: {e.revocation_sidecar}")
            if e.heavyweight_prereg:
                lines.append(f"      heavyweight_prereg: {e.heavyweight_prereg}")
            if e.park_entry:
                lines.append(f"      park_entry: {e.park_entry}")
            if e.error_reason:
                lines.append(f"      ERROR: {e.error_reason}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Refresh docs/runtime/promote_queue.yaml from current on-disk state.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="(default) Print the would-be queue and the diff vs cache; do not write.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (for drift-check integration).",
    )
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--cache-path", default=str(QUEUE_CACHE))
    parser.add_argument("--hypotheses-dir", default=str(HYPOTHESES_DIR))
    parser.add_argument("--action-queue", default=str(ACTION_QUEUE))
    args = parser.parse_args(argv)

    if args.write and args.dry_run:
        parser.error("--write and --dry-run are mutually exclusive")

    entries = scan(
        Path(args.results_dir),
        hypotheses_dir=Path(args.hypotheses_dir),
        action_queue=Path(args.action_queue),
    )

    if args.json:
        sys.stdout.write(json.dumps([_entry_to_dict(e) for e in entries], indent=2))
        sys.stdout.write("\n")
        return 0

    sys.stdout.write(render_report(entries))
    sys.stdout.write("\n--- diff vs cache ---\n")
    for line in diff_against_cache(entries, Path(args.cache_path)):
        sys.stdout.write(line + "\n")

    if args.write:
        Path(args.cache_path).write_text(serialize_queue(entries), encoding="utf-8")
        sys.stdout.write(f"\nwrote cache: {args.cache_path}\n")

    if any(e.status == "ERROR" for e in entries):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
