"""Fast-lane status roll-up writer.

See ``docs/specs/fast_lane_state_graph.md`` for the canonical fast-lane chain definition.

Stage 2 of ``docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md``.

Walks every artifact in the fast-lane prereg chain and emits a per-strategy_id
status roll-up at ``docs/runtime/fast_lane_status.yaml``. The roll-up is
**derived state** — rebuilt from source artifacts on every run; hand-edits
fail Check #168 on the next walk.

Inputs (read-only)
------------------
- ``docs/audit/hypotheses/*.yaml`` (active preregs; ``drafts/`` excluded by glob)
- ``docs/audit/hypotheses/drafts/`` (bridge drafts + grounded sidecars + rejections)
- ``docs/audit/results/*.md`` (every fast-lane + heavyweight result MD)
- ``docs/runtime/promote_queue.yaml`` (already-derived PROMOTE-queue cache)
- ``docs/runtime/cherry_pick_journal.yaml`` (append-only journal)
- ``docs/runtime/cherry_pick_ranking_*.csv`` (per-day ranking snapshots — newest mtime wins per strategy_id)

Outputs
-------
- ``docs/runtime/fast_lane_status.yaml`` (rebuilt; one entry per observed strategy_id)

Capital-class boundary
----------------------
This writer NEVER opens ``docs/runtime/chordia_audit_log.yaml``,
``docs/runtime/lane_allocation.json``, ``validated_setups``, or anything under
``trading_app/live/`` for write. The capital-class read-only assertion is
enforced by ``check_fast_lane_status_rollup_reconstruction_parity`` (Check #168)
via a greppable static check on this file plus the test fixture's mock-fs run.

Stage enum (canonical — mirrors fast_lane_state_graph.md § 5.1)
---------------------------------------------------------------
ACTIVE_PREREG          active prereg YAML exists; no result MD yet
FAST_LANE_RUN          fast-lane result MD exists; no PROMOTE verdict
PROMOTE_QUEUED         result MD verdict=PROMOTE; queue status=QUEUED (no draft, no park, no revocation)
RANKED                 PROMOTE_QUEUED + journal entry exists (cherry_pick_ranker scored it)
BRIDGED                draft YAML exists under hypotheses/drafts/
GROUNDED               .grounded.yaml sibling exists alongside the draft
HEAVYWEIGHT_PENDING    grounded or strict-acceptance draft has been moved to active hypotheses/
HEAVYWEIGHT_COMPLETE   heavyweight result MD exists (non-fast-lane verdict landed)
ENRICHED               journal entry's heavyweight_verdict field is populated
REVOKED                queue status=REVOKED (pooling artifact, revocation sidecar)
PARKED                 queue status=PARKED (action-queue.yaml park entry)
REJECTED_OOS_UNPOWERED queue status=REJECTED_OOS_UNPOWERED (RULE 3.3 pre-flight)
ERROR                  any upstream parse failure or status=ERROR from queue scanner
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
HYPOTHESES_DIR = REPO_ROOT / "docs" / "audit" / "hypotheses"
DRAFTS_DIR = HYPOTHESES_DIR / "drafts"
RESULTS_DIR = REPO_ROOT / "docs" / "audit" / "results"
RUNTIME_DIR = REPO_ROOT / "docs" / "runtime"
PROMOTE_QUEUE_CACHE = RUNTIME_DIR / "promote_queue.yaml"
JOURNAL_PATH = RUNTIME_DIR / "cherry_pick_journal.yaml"
RANKING_GLOB = "cherry_pick_ranking_*.csv"
STATUS_ROLLUP_PATH = RUNTIME_DIR / "fast_lane_status.yaml"

SCHEMA_VERSION = 1

# Canonical stage enum — mirrors docs/specs/fast_lane_state_graph.md § 5.1.
# Order is precedence (later wins): if a strategy_id is observed at multiple
# stages simultaneously the downstream-most stage is recorded as current_stage.
STAGE_PRECEDENCE: tuple[str, ...] = (
    "ACTIVE_PREREG",
    "FAST_LANE_RUN",
    "PROMOTE_QUEUED",
    "RANKED",
    "BRIDGED",
    "GROUNDED",
    "HEAVYWEIGHT_PENDING",
    "HEAVYWEIGHT_COMPLETE",
    "ENRICHED",
    # Terminal / off-path stages (no further advancement expected):
    "REVOKED",
    "PARKED",
    "REJECTED_OOS_UNPOWERED",
    "ERROR",
)

_TERMINAL_STAGES = frozenset({"REVOKED", "PARKED", "REJECTED_OOS_UNPOWERED"})

# Direct mapping of promote_queue.yaml entry.status → roll-up current_stage.
# Anything that maps to PROMOTE_QUEUED here may be overridden later by RANKED /
# BRIDGED / GROUNDED / etc. as those downstream artifacts are detected.
_QUEUE_STATUS_TO_STAGE: dict[str, str] = {
    "QUEUED": "PROMOTE_QUEUED",
    "ESCALATED": "HEAVYWEIGHT_PENDING",
    "REVOKED": "REVOKED",
    "PARKED": "PARKED",
    "REJECTED_OOS_UNPOWERED": "REJECTED_OOS_UNPOWERED",
    "ERROR": "ERROR",
}

# Next-action token per terminal-eligible stage. Mirrors design § 2.5
# Connector 4 ("next operator action" footer) per-stage routing.
_NEXT_ACTION_BY_STAGE: dict[str, str] = {
    "ACTIVE_PREREG": "run_fast_lane",
    "FAST_LANE_RUN": "rebuild_promote_queue",
    "PROMOTE_QUEUED": "run_cherry_pick_ranker",
    "RANKED": "run_fast_lane_to_heavyweight_bridge",
    "BRIDGED": "run_cherry_pick_grounder_or_accept_strict_t",
    "GROUNDED": "operator_promote_draft_to_active",
    "HEAVYWEIGHT_PENDING": "run_chordia_strict_unlock_v1",
    "HEAVYWEIGHT_COMPLETE": "run_cherry_pick_journal_enricher",
    "ENRICHED": "operator_deployment_decision",
    "REVOKED": "no_action_required",
    "PARKED": "no_action_required",
    "REJECTED_OOS_UNPOWERED": "operator_pick_remedy_cpcv_haircut_pool_or_park",
    "ERROR": "operator_resolve_error",
}

# Override token for HEAVYWEIGHT_COMPLETE entries that lack fast-lane lineage.
# Background: heavyweight Chordia preregs authored directly (predating the
# 2026-05-19 cherry-pick loop landing) reach HEAVYWEIGHT_COMPLETE without ever
# being scored by the ranker, so the cherry-pick journal has no row for the
# enricher to update. The enricher (cherry_pick_journal_enricher.py) is
# update-only against existing journal entries; it cannot create new entries.
# Emitting the enricher token for these is a misclassification — the result MD
# already carries the heavyweight verdict and the next operator action is the
# deployment decision, identical to the ENRICHED stage's downstream gate.
#
# Lineage signal: a strategy_id has fast-lane lineage iff a journal entry
# exists for it (ranker writes the entry at score-time, BEFORE the heavyweight
# verdict lands). queue_entry presence is NOT a fast-lane lineage signal — a
# heavyweight prereg can backfill into promote_queue.yaml via the PROMOTE/PARK
# pathways without ever being ranked.
_NEXT_ACTION_HEAVYWEIGHT_COMPLETE_NO_LINEAGE: str = "operator_deployment_decision"


@dataclass
class StatusEntry:
    """One row per observed strategy_id in the roll-up."""

    strategy_id: str
    current_stage: str
    age_days: int
    next_action_token: str
    upstream_artifact_path: str | None
    downstream_artifact_path: str | None
    observed_at: dict[str, str | None] = field(default_factory=dict)


def _rel(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _file_age_days(path: Path, *, today: date | None = None) -> int:
    """Filesystem-mtime based age in calendar days.

    Falls back to 0 when the file is missing or mtime is in the future
    (clock skew). Design § 4.1 ("Age-staleness mtime spoofing") notes
    git-log fallback is a Stage 3 concern, not Stage 2.
    """
    if not path.exists():
        return 0
    today_d = today if today is not None else date.today()
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()
    except OSError:
        return 0
    delta = (today_d - mtime).days
    return max(delta, 0)


def _safe_load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return None


def collect_active_preregs(
    hypotheses_dir: Path = HYPOTHESES_DIR,
) -> dict[str, Path]:
    """Map strategy_id → active prereg path. Excludes drafts/ by glob.

    Falls back silently for prereg files whose scope.strategy_id is absent;
    they remain in the chain as unaddressable orphans and are surfaced by
    other drift checks (e.g. check_chordia_audit_log_orphan_prereg).
    """
    out: dict[str, Path] = {}
    if not hypotheses_dir.exists():
        return out
    for path in sorted(hypotheses_dir.glob("*.yaml")):
        if path.name.startswith("TEMPLATE-"):
            continue  # Skip skeleton templates whose strategy_id is a placeholder.
        data = _safe_load_yaml(path)
        if not isinstance(data, dict):
            continue
        scope = data.get("scope") or {}
        sid = scope.get("strategy_id") if isinstance(scope, dict) else None
        if isinstance(sid, str) and sid:
            out.setdefault(sid, path)
    return out


def collect_drafts(
    drafts_dir: Path = DRAFTS_DIR,
) -> dict[str, dict[str, Path]]:
    """Map strategy_id → {draft, grounded} paths under drafts/.

    Parses each draft.yaml for scope.strategy_id. Grounded siblings detected
    by filename pattern (<stem>.grounded.yaml). Rejections (.rejected.txt)
    are not stage-advancing and are ignored here.
    """
    out: dict[str, dict[str, Path]] = {}
    if not drafts_dir.exists():
        return out
    for path in sorted(drafts_dir.glob("*.draft.yaml")):
        data = _safe_load_yaml(path)
        if not isinstance(data, dict):
            continue
        scope = data.get("scope") or {}
        sid = scope.get("strategy_id") if isinstance(scope, dict) else None
        if not isinstance(sid, str) or not sid:
            continue
        entry = out.setdefault(sid, {})
        entry["draft"] = path
        grounded = path.with_name(path.stem.replace(".draft", "") + ".grounded.yaml")
        if grounded.exists():
            entry["grounded"] = grounded
    return out


def collect_promote_queue_entries(
    queue_cache: Path = PROMOTE_QUEUE_CACHE,
) -> dict[str, dict[str, Any]]:
    """Map strategy_id → {status, result_md, ...} from promote_queue.yaml.

    Returns empty dict when the cache is missing (e.g. first-run). Drift
    check #157 already covers cache freshness; this writer never rebuilds
    the cache itself.
    """
    data = _safe_load_yaml(queue_cache)
    if not isinstance(data, dict):
        return {}
    entries = data.get("entries") or []
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("strategy_id")
        if isinstance(sid, str) and sid:
            out[sid] = entry
    return out


def collect_journal_entries(
    journal_path: Path = JOURNAL_PATH,
) -> dict[str, dict[str, Any]]:
    """Map strategy_id → latest journal entry dict. Last write wins.

    Journal is append-only with multiple entries per strategy_id possible
    (re-runs). The latest entry by iter is the current state of record.
    """
    data = _safe_load_yaml(journal_path)
    if not isinstance(data, dict):
        return {}
    entries = data.get("entries") or []
    if not isinstance(entries, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("strategy_id")
        if not isinstance(sid, str) or not sid:
            continue
        prior = out.get(sid)
        if prior is None or (entry.get("iter") or 0) > (prior.get("iter") or 0):
            out[sid] = entry
    return out


def collect_ranking_csvs(
    runtime_dir: Path = RUNTIME_DIR,
) -> set[str]:
    """Set of strategy_ids ranked at least once in any ranking CSV.

    Multiple CSVs exist (one per ranker invocation date). We union across
    all of them — being ranked once is sufficient evidence of stage RANKED.
    """
    out: set[str] = set()
    for path in sorted(runtime_dir.glob(RANKING_GLOB)):
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    sid = row.get("strategy_id")
                    if isinstance(sid, str) and sid:
                        out.add(sid)
        except OSError:
            continue
    return out


def collect_heavyweight_results(
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Path]:
    """Map strategy_id → most-recent heavyweight result MD path.

    Heavyweight result MDs are everything under docs/audit/results/ that is
    NOT a fast-lane result (the "*fast-lane*.md" glob covers fast-lane).
    Strategy_id is parsed from the canonical title line — same regex as
    fast_lane_promote_queue._TITLE_RE.
    """
    out: dict[str, Path] = {}
    if not results_dir.exists():
        return out
    import re

    title_re = re.compile(r"^#\s+Chordia strict unlock audit\s+\S\s+(?P<sid>\S+)\s*$", re.MULTILINE)
    for path in sorted(results_dir.glob("*.md")):
        if "fast-lane" in path.name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        m = title_re.search(text)
        if m:
            sid = m.group("sid")
            existing = out.get(sid)
            if existing is None or path.stat().st_mtime > existing.stat().st_mtime:
                out[sid] = path
    return out


def _classify_stage(
    *,
    active_prereg: Path | None,
    queue_entry: dict[str, Any] | None,
    journal_entry: dict[str, Any] | None,
    is_ranked: bool,
    drafts: dict[str, Path] | None,
    heavyweight_result: Path | None,
) -> str:
    """Pick the downstream-most stage observed for a strategy_id.

    Terminal queue stages (REVOKED, PARKED, REJECTED_OOS_UNPOWERED, ERROR)
    short-circuit downstream progression — a revoked PROMOTE that later
    happened to get a draft authored is still REVOKED.
    """
    if queue_entry is not None:
        q_status = queue_entry.get("status")
        if q_status in _TERMINAL_STAGES or q_status == "ERROR":
            return _QUEUE_STATUS_TO_STAGE.get(q_status, "ERROR")

    # Walk precedence forward, downstream wins.
    candidates: list[str] = []
    if active_prereg is not None:
        candidates.append("ACTIVE_PREREG")
    if queue_entry is not None:
        candidates.append("FAST_LANE_RUN")
        mapped = _QUEUE_STATUS_TO_STAGE.get(queue_entry.get("status") or "")
        if mapped:
            candidates.append(mapped)
    if is_ranked or journal_entry is not None:
        candidates.append("RANKED")
    if drafts and "draft" in drafts:
        candidates.append("BRIDGED")
    if drafts and "grounded" in drafts:
        candidates.append("GROUNDED")
    if heavyweight_result is not None:
        candidates.append("HEAVYWEIGHT_COMPLETE")
    if journal_entry is not None and journal_entry.get("heavyweight_verdict"):
        candidates.append("ENRICHED")

    if not candidates:
        return "ERROR"
    # Pick latest in precedence order.
    return max(candidates, key=lambda s: STAGE_PRECEDENCE.index(s))


def _next_action_for(
    stage: str,
    *,
    journal_entry: dict[str, Any] | None,
) -> str:
    """Resolve the next-action token for a stage, with lineage-qualified override.

    Special case: HEAVYWEIGHT_COMPLETE without a journal entry has no
    fast-lane lineage — the enricher cannot create journal entries, only
    update them. Emit the deployment-decision token instead so the operator
    is pointed at the heavyweight result MD rather than at a stage script
    that will silently no-op.
    """
    if stage == "HEAVYWEIGHT_COMPLETE" and journal_entry is None:
        return _NEXT_ACTION_HEAVYWEIGHT_COMPLETE_NO_LINEAGE
    return _NEXT_ACTION_BY_STAGE.get(stage, "operator_resolve_error")


def build_status_entries(
    *,
    hypotheses_dir: Path = HYPOTHESES_DIR,
    drafts_dir: Path = DRAFTS_DIR,
    results_dir: Path = RESULTS_DIR,
    runtime_dir: Path = RUNTIME_DIR,
    queue_cache: Path = PROMOTE_QUEUE_CACHE,
    journal_path: Path = JOURNAL_PATH,
    today: date | None = None,
) -> list[StatusEntry]:
    """Walk every artifact in the chain; return one entry per strategy_id."""
    active = collect_active_preregs(hypotheses_dir)
    drafts = collect_drafts(drafts_dir)
    queue = collect_promote_queue_entries(queue_cache)
    journal = collect_journal_entries(journal_path)
    ranked = collect_ranking_csvs(runtime_dir)
    heavy = collect_heavyweight_results(results_dir)

    sids: set[str] = set(active) | set(drafts) | set(queue) | set(journal) | ranked | set(heavy)

    entries: list[StatusEntry] = []
    for sid in sorted(sids):
        prereg = active.get(sid)
        q_entry = queue.get(sid)
        j_entry = journal.get(sid)
        d_entry = drafts.get(sid)
        h_result = heavy.get(sid)

        stage = _classify_stage(
            active_prereg=prereg,
            queue_entry=q_entry,
            journal_entry=j_entry,
            is_ranked=sid in ranked,
            drafts=d_entry,
            heavyweight_result=h_result,
        )

        # Upstream artifact = whatever produced the current_stage advancement.
        # Downstream artifact = the artifact the next action will produce.
        upstream: Path | None
        downstream: Path | None
        if stage == "ACTIVE_PREREG":
            upstream, downstream = prereg, None
        elif stage in ("FAST_LANE_RUN", "PROMOTE_QUEUED", "RANKED"):
            upstream = REPO_ROOT / (q_entry.get("result_md") or "") if q_entry and q_entry.get("result_md") else prereg
            downstream = None
        elif stage in ("BRIDGED", "GROUNDED"):
            upstream = (d_entry or {}).get("grounded") or (d_entry or {}).get("draft")
            downstream = None
        elif stage == "HEAVYWEIGHT_PENDING":
            upstream = prereg
            downstream = None
        elif stage in ("HEAVYWEIGHT_COMPLETE", "ENRICHED"):
            upstream = h_result
            downstream = None
        elif stage in _TERMINAL_STAGES:
            upstream = REPO_ROOT / (q_entry.get("result_md") or "") if q_entry and q_entry.get("result_md") else None
            downstream = None
        else:  # ERROR
            upstream = None
            downstream = None

        # Age: based on whichever artifact represents current_stage.
        age_source = upstream or prereg
        age = _file_age_days(age_source, today=today) if age_source else 0

        draft_p: Path | None = (d_entry or {}).get("draft") if d_entry else None
        grounded_p: Path | None = (d_entry or {}).get("grounded") if d_entry else None
        observed = {
            "active_prereg": _rel(prereg) if prereg else None,
            "promote_queue_status": q_entry.get("status") if q_entry else None,
            "journal_iter": (j_entry or {}).get("iter") if j_entry else None,
            "ranked_in_csv": sid in ranked,
            "draft_path": _rel(draft_p) if draft_p else None,
            "grounded_path": _rel(grounded_p) if grounded_p else None,
            "heavyweight_result_md": _rel(h_result) if h_result else None,
        }

        entries.append(
            StatusEntry(
                strategy_id=sid,
                current_stage=stage,
                age_days=age,
                next_action_token=_next_action_for(stage, journal_entry=j_entry),
                upstream_artifact_path=_rel(upstream) if upstream else None,
                downstream_artifact_path=_rel(downstream) if downstream else None,
                observed_at=observed,
            )
        )
    return entries


def serialize_rollup(entries: list[StatusEntry], *, today: date | None = None) -> str:
    """Render the YAML payload. Stable key order; entries sorted by strategy_id."""
    today_d = today if today is not None else date.today()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": today_d.isoformat(),
        "source": "scripts/tools/fast_lane_status.py",
        "do_not_hand_edit": True,
        "warning": (
            "DERIVED STATE — do not hand-edit. Rebuilt from every fast-lane "
            "chain artifact on every run. Drift check "
            "check_fast_lane_status_rollup_reconstruction_parity (#168) "
            "reconstructs independently and fails the build if this file "
            "disagrees with the canonical sources or has been hand-edited."
        ),
        "entries": [asdict(e) for e in entries],
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fast_lane_status",
        description=(
            "Walk the fast-lane prereg chain and emit a per-strategy_id status "
            "roll-up. Read-only over every source artifact; writes only "
            "docs/runtime/fast_lane_status.yaml."
        ),
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Refresh docs/runtime/fast_lane_status.yaml from on-disk state.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="(default) Print the would-be roll-up to stdout; do not write.",
    )
    p.add_argument("--output", default=str(STATUS_ROLLUP_PATH))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.write and args.dry_run:
        print("ERROR: --write and --dry-run are mutually exclusive", file=sys.stderr)
        return 2
    entries = build_status_entries()
    payload = serialize_rollup(entries)
    if args.write:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"wrote roll-up: {_rel(out)} ({len(entries)} entries)")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
