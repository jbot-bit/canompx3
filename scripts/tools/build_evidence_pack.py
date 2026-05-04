#!/usr/bin/env python3
"""Evidence-pack generator — CLI entry point.

Authority: docs/plans/2026-04-28-evidence-pack-generator-design.md
Stage:     docs/runtime/stages/evidence-pack-generator-stage1.md

Composes existing pre-reg YAMLs + result MDs + canonical computation
modules into a single audit-grade evidence pack. Read-only against
canonical sources. Writes ONLY to ``reports/evidence_packs/<slug>/<run>/``.

Outputs (PR1, text-only):
  - manifest.json      — full provenance, deterministic
  - decision_card.md   — one-page verdict
  - gate_table.json    — twelve GateResult rows
  - report.md          — full pack with SQL text + SHA-256 + rerun

Reserved but inert in PR1: --html, --include-extracts.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make the parent ``scripts/tools/`` importable as a package root for the
# nested ``evidence_pack`` package without polluting sys.path globally.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

PROJECT_ROOT = _THIS_DIR.parent.parent

from evidence_pack import PACK_VERSION  # noqa: E402
from evidence_pack.gates import evaluate_all  # noqa: E402
from evidence_pack.manifest import (  # noqa: E402
    Contamination,
    Manifest,
)
from evidence_pack.renderers import (  # noqa: E402
    render_decision_card_md,
    render_gate_table_json,
    render_manifest_json,
    render_report_md,
)

CONTAMINATION_GLOB = "docs/audit/**/*e2*lookahead*contamination*.md"
# pathlib.Path.rglob does not auto-prepend ``*`` to a filename pattern, so
# the date-prefixed real file name (``2026-04-28-e2-lookahead-...``) only
# matches when the leading ``*`` is explicit. Tested 2026-04-28.
DERIVED_LAYERS = frozenset({"validated_setups", "edge_families", "live_config"})
HOLDOUT_SACRED_FROM = "2026-01-01"
PACK_ROOT = "reports/evidence_packs"


# ─────────────────────────── Argument parsing ────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_evidence_pack",
        description="Compile a deterministic evidence pack for one candidate.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--prereg",
        help="Path or slug of a pre-reg YAML under docs/audit/hypotheses/.",
    )
    src.add_argument(
        "--validated-setup",
        help="Identity tuple of a validated_setups row "
        "(instrument:session:orb_minutes:entry_model:rr:filter:direction).",
    )
    src.add_argument(
        "--experimental",
        help="Row id of an experimental_strategies entry.",
    )
    p.add_argument(
        "--out",
        help="Output directory. Default: reports/evidence_packs/<slug>/<run-iso8601>/",
    )
    p.add_argument(
        "--allow-fingerprint-drift",
        action="store_true",
        help="Allow re-runs with a different DB fingerprint for the same git SHA.",
    )
    p.add_argument(
        "--update-snapshots",
        action="store_true",
        help="Test-fixture flag; writes golden snapshots. Not for production use.",
    )
    p.add_argument(
        "--html",
        action="store_true",
        help="Reserved (PR4). Currently inert.",
    )
    p.add_argument(
        "--include-extracts",
        action="store_true",
        help="Reserved (PR2). Currently inert.",
    )
    return p


# ─────────────────────────── Slug resolution ─────────────────────────────


def resolve_prereg(arg: str, project_root: Path) -> tuple[Path | None, str]:
    """Return (prereg_path, slug). Path is None on miss; slug is best-effort.

    Three behaviours:
    - arg is an existing file path → use it directly.
    - arg looks like a date-prefixed slug (YYYY-MM-DD-...) → glob the
      hypotheses dir.
    - arg is a bare slug (no date) → glob with date wildcard.
    """

    candidate = Path(arg)
    if candidate.is_absolute() and candidate.exists():
        return candidate, candidate.stem
    rel = project_root / arg
    if rel.exists():
        return rel, rel.stem

    hyp_dir = project_root / "docs" / "audit" / "hypotheses"
    if not hyp_dir.is_dir():
        return None, arg

    # Look for *.yaml ending with the slug
    matches = sorted(hyp_dir.glob(f"*{arg}*.yaml"))
    if not matches:
        return None, arg
    return matches[0], matches[0].stem


# ────────────────────────── YAML / MD parsing ────────────────────────────


def load_prereg_frontmatter(path: Path) -> dict[str, Any]:
    """Return the prereg YAML's structured metadata.

    The pre-reg files in this repo are plain YAML documents (not MD with
    frontmatter), so this is just yaml.safe_load wrapped with permissive
    fallbacks.
    """

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover — pyyaml in lockfile
        raise RuntimeError("pyyaml is required (lockfile pin)") from exc

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def parse_result_md(path: Path) -> dict[str, Any]:
    """Extract result MD frontmatter + recognisable scalar bullets.

    Result MDs in this repo follow a loose convention. We extract:
    - YAML frontmatter (if present between leading ``---`` fences).
    - Bullet-line scalars like ``- Git HEAD: `<sha>``` and
      ``- DB schema fingerprint: `<hash>``` and ``- K = N`` and
      ``- Bootstrap seed: N``.
    """

    out: dict[str, Any] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out

    # Frontmatter
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = text[3:end].strip()
            try:
                import yaml  # type: ignore

                fm_data = yaml.safe_load(fm)
                if isinstance(fm_data, dict):
                    out.update(fm_data)
            except Exception:
                pass

    # Bullet scalars
    import re

    patterns = [
        ("git_head", r"^- Git HEAD:\s*`([0-9a-f]{6,40})`"),
        ("db_fingerprint", r"^- DB schema fingerprint:\s*`([0-9a-f]{6,40})`"),
        ("bootstrap_seed", r"^- Bootstrap seed:\s*(\d+)"),
        ("verdict_text", r"^- \*\*VERDICT:\*\*\s*([A-Z_]+)"),
    ]
    for name, pat in patterns:
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            out[name] = m.group(1)

    # K declared
    k_match = re.search(r"K\s*=\s*(\d+)\s*declared", text)
    if k_match:
        out["k_global"] = int(k_match.group(1))

    return out


# ────────────────────── Contamination registry glob ──────────────────────


def discover_contamination_registries(project_root: Path) -> tuple[Path, ...]:
    """Sorted recursive glob for the E2 look-ahead contamination registry.

    Per Python docs, ``Path.rglob`` order is not guaranteed; results MUST
    be sorted before use. Stage acceptance criterion #9.
    """

    docs_root = project_root / "docs" / "audit"
    if not docs_root.is_dir():
        return ()
    matches = list(docs_root.rglob("*e2*lookahead*contamination*.md"))
    return tuple(sorted(matches, key=lambda p: str(p)))


def evaluate_contamination(project_root: Path, result_commit: str | None) -> Contamination:
    """Build the Contamination block from the discovered registries.

    - No registry → registry_status=MISSING, status=UNCOMPUTED, amber banner.
    - Registry present, commit not in any tainted list → status=CLEAN.
    - Registry present, commit in tainted list → status=TAINTED, hits[].
    """

    registries = discover_contamination_registries(project_root)
    paths = tuple(str(p.relative_to(project_root)) for p in registries)

    if not registries:
        return Contamination(
            registry_paths=(),
            registry_status="MISSING",
            hits=(),
            status="UNCOMPUTED",
            expected_glob=CONTAMINATION_GLOB,
            note=(
                "Sorted recursive glob returned no match. Canonical registry "
                "exists on unmerged research branches as of 2026-04-28."
            ),
        )

    if not result_commit:
        return Contamination(
            registry_paths=paths,
            registry_status="PRESENT",
            hits=(),
            status="UNCOMPUTED",
            expected_glob=CONTAMINATION_GLOB,
            note="Result MD commit unknown; cannot match against tainted list.",
        )

    hits: list[str] = []
    for reg in registries:
        try:
            content = reg.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # A 'hit' is the commit prefix appearing inside the registry text.
        # Match on >=7 char prefix to avoid false-positives on 6-char hex.
        prefix = result_commit[:7] if len(result_commit) >= 7 else result_commit
        if prefix and prefix in content:
            hits.append(prefix)

    if hits:
        return Contamination(
            registry_paths=paths,
            registry_status="PRESENT",
            hits=tuple(sorted(set(hits))),
            status="TAINTED",
            expected_glob=CONTAMINATION_GLOB,
        )
    return Contamination(
        registry_paths=paths,
        registry_status="PRESENT",
        hits=(),
        status="CLEAN",
        expected_glob=CONTAMINATION_GLOB,
    )


# ─────────────────────────── Verdict logic ───────────────────────────────


def compute_verdict(
    *,
    prereg_resolved: bool,
    holdout_ok: bool,
    derived_truth_violation: bool,
    fingerprint_drift: bool,
    pooled_misuse: bool,
) -> tuple[str, tuple[str, ...]]:
    """Apply the 5 fail-closed gates and return (verdict, reasons).

    PASS only when all five hold and there is no other failure. Anything
    else downgrades to INCOMPLETE_EVIDENCE or LANE_NOT_SUPPORTED_BY_POOLED.
    Note: this PR1 generator returns INCOMPLETE_EVIDENCE for fail-closed
    failures; PASS / CONDITIONAL / KILL distinctions depend on richer
    inputs deferred to follow-up PRs.
    """

    reasons: list[str] = []
    if not prereg_resolved:
        reasons.append("Hypothesis pre-reg path did not resolve (fail-closed gate 1).")
    if not holdout_ok:
        reasons.append("Holdout not declared as 2026-01-01 in pre-reg (fail-closed gate 2).")
    if derived_truth_violation:
        reasons.append(
            "Result MD names a derived layer (validated_setups/edge_families/"
            "live_config) as a discovery-truth source (fail-closed gate 3)."
        )
    if fingerprint_drift:
        reasons.append(
            "DB fingerprint drift vs prior run for the same git_sha "
            "(fail-closed gate 4). Pass --allow-fingerprint-drift to override."
        )
    if pooled_misuse:
        return "LANE_NOT_SUPPORTED_BY_POOLED", (
            "Result MD is pooled (pooled_finding=true) and the requested lane "
            "appears in the per-lane breakdown with a sign-flip vs pooled "
            "(fail-closed gate 5).",
        )

    if reasons:
        return "INCOMPLETE_EVIDENCE", tuple(reasons)
    # PR1 default verdict for a clean run is CONDITIONAL — we do not have
    # enough inputs to claim PASS without per-criterion thresholds.
    return "CONDITIONAL", ("PR1 generator does not auto-PASS; auditor review required.",)


# ─────────────────────────── DB fingerprint ──────────────────────────────


def db_fingerprint(db_path: Path) -> str | None:
    """Return a short schema fingerprint or None if DB is unreachable.

    Cheap: hashes the sorted (table, column, type) triples from
    ``information_schema.columns``. Read-only.
    """

    if not db_path.exists():
        return None
    try:
        import duckdb  # type: ignore
    except ImportError:
        return None
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                "SELECT table_name, column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_schema = 'main' "
                "ORDER BY table_name, column_name"
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return None
    digest = hashlib.sha256()
    for r in rows:
        digest.update("|".join(str(x) for x in r).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:32]


def repo_git_sha(project_root: Path) -> str | None:
    """Return repo HEAD SHA via pipeline.audit_log if importable, else None."""

    try:
        sys.path.insert(0, str(project_root))
        from pipeline.audit_log import get_git_sha  # type: ignore

        return get_git_sha()
    except Exception:
        return None


# ─────────────────────────── Orchestrator ────────────────────────────────


def build_manifest(
    *,
    slug: str,
    prereg_path: Path | None,
    project_root: Path,
    git_sha: str | None,
    fingerprint: str | None,
    db_path_str: str,
    allow_fingerprint_drift: bool,
    out_dir: Path,
) -> Manifest:
    """Build the Manifest, computing all gates and the verdict."""

    prereg_data = load_prereg_frontmatter(prereg_path) if prereg_path else {}
    metadata = prereg_data.get("metadata") if isinstance(prereg_data, dict) else None
    if not isinstance(metadata, dict):
        metadata = prereg_data if isinstance(prereg_data, dict) else {}

    prereg_sha = metadata.get("commit_sha") or metadata.get("commit_hash")
    if isinstance(prereg_sha, str):
        prereg_sha = prereg_sha.strip() or None
    holdout_in_prereg = metadata.get("holdout_date")
    if hasattr(holdout_in_prereg, "isoformat"):
        holdout_in_prereg = holdout_in_prereg.isoformat()
    holdout_ok = isinstance(holdout_in_prereg, str) and holdout_in_prereg.strip() == HOLDOUT_SACRED_FROM

    # Try to find the matching result MD by slug stem.
    result_dir = project_root / "docs" / "audit" / "results"
    result_path: Path | None = None
    if prereg_path and result_dir.is_dir():
        stem_no_prereg = prereg_path.stem.replace("-prereg", "")
        candidates = sorted(result_dir.glob(f"{stem_no_prereg}*.md"))
        if candidates:
            result_path = candidates[0]

    result_data = parse_result_md(result_path) if result_path else {}

    # Derived-layer truth violation: scan the result MD for direct truth
    # claims against derived layers.
    derived_truth_violation = False
    if result_path and result_path.exists():
        try:
            content = result_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = ""
        # Heuristic: a line of the form "Truth source: <derived>" or
        # "discovery-truth from validated_setups" trips this. Bullet
        # references in caveats / audit-trail are allowed.
        import re

        for layer in DERIVED_LAYERS:
            if re.search(rf"discovery[- ]truth\s+from\s+`?{layer}`?", content, re.IGNORECASE):
                derived_truth_violation = True
                break
            if re.search(rf"truth\s+source\s*[:=]\s*`?{layer}`?", content, re.IGNORECASE):
                derived_truth_violation = True
                break

    # Pooled-finding misuse: handled as a soft check in PR1 (no per-lane
    # slug parsing yet). Always False in PR1 unless an explicit flag is set
    # by tests via a synthetic frontmatter field.
    pooled_misuse = bool(result_data.get("_pooled_misuse_test_flag"))

    # Fingerprint drift: check prior runs under same slug.
    fingerprint_drift = False
    pack_dir = project_root / PACK_ROOT / slug
    if pack_dir.is_dir() and git_sha and fingerprint:
        for prior in sorted(pack_dir.iterdir()):
            prior_manifest = prior / "manifest.json"
            if not prior_manifest.is_file():
                continue
            try:
                import json

                prior_data = json.loads(prior_manifest.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                prior_data.get("git_sha") == git_sha
                and prior_data.get("db_fingerprint")
                and prior_data["db_fingerprint"] != fingerprint
            ):
                fingerprint_drift = True
                break
    if allow_fingerprint_drift:
        fingerprint_drift = False

    contamination = evaluate_contamination(project_root, result_data.get("git_head"))

    verdict, reasons = compute_verdict(
        prereg_resolved=prereg_path is not None and prereg_path.exists(),
        holdout_ok=holdout_ok,
        derived_truth_violation=derived_truth_violation,
        fingerprint_drift=fingerprint_drift,
        pooled_misuse=pooled_misuse,
    )

    gates = evaluate_all(
        {
            "prereg_path": str(prereg_path.relative_to(project_root)) if prereg_path else None,
            "prereg_sha": prereg_sha,
            "total_trials": metadata.get("total_expected_trials"),
            "k_global": metadata.get("k_global") or result_data.get("k_global"),
        }
    )

    manifest = Manifest(
        pack_version=PACK_VERSION,
        slug=slug,
        run_iso8601=_deterministic_run_iso(out_dir),
        git_sha=git_sha,
        db_fingerprint=fingerprint,
        db_path=db_path_str,
        holdout_date=holdout_in_prereg if isinstance(holdout_in_prereg, str) else "(undeclared)",
        hypothesis={
            "path": str(prereg_path.relative_to(project_root)) if prereg_path else "(unresolved)",
            "commit_sha": prereg_sha or "(missing)",
            "k_global": metadata.get("k_global"),
            "total_expected_trials": metadata.get("total_expected_trials"),
            "testing_mode": metadata.get("testing_mode"),
        },
        result={
            "path": str(result_path.relative_to(project_root)) if result_path else "(none)",
            "git_head": result_data.get("git_head"),
            "db_fingerprint_in_md": result_data.get("db_fingerprint"),
            "verdict_in_md": result_data.get("verdict_text"),
            "pooled_finding": bool(result_data.get("pooled_finding", False)),
            "flip_rate_pct": result_data.get("flip_rate_pct"),
            "heterogeneity_ack": result_data.get("heterogeneity_ack"),
        },
        validated_setups=None,  # PR1 stops here; richer DB read is PR2 territory.
        tables_used=(),  # PR1 records nothing it did not actually query.
        is_oos_split={
            "holdout_sacred_from": HOLDOUT_SACRED_FROM,
            "is_window": "trading_day < 2026-01-01",
            "oos_window": "trading_day >= 2026-01-01",
        },
        k_framings={
            "K_global": metadata.get("k_global") or result_data.get("k_global") or "UNCOMPUTED",
            "K_family": "UNCOMPUTED",
            "K_lane": "UNCOMPUTED",
            "K_session": "UNCOMPUTED",
            "K_instrument": "UNCOMPUTED",
            "K_feature": "UNCOMPUTED",
        },
        gates=gates,
        queries=(),  # PR1 does not run any SQL itself.
        contamination=contamination,
        verdict=verdict,
        verdict_reasons=reasons,
    )
    return manifest


def _deterministic_run_iso(out_dir: Path) -> str:
    """Return a stable run timestamp.

    For deterministic output we honour an explicit ``out_dir`` whose final
    path component is an ISO-8601 timestamp (or any string); otherwise we
    use the current UTC time. The orchestrator passes the chosen ``out_dir``
    here so that two runs with the same target directory produce
    byte-identical manifests.
    """

    name = out_dir.name
    if name and name != ".":
        return name
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


# ─────────────────────────── Write outputs ───────────────────────────────


def write_pack(out_dir: Path, manifest: Manifest) -> dict[str, Path]:
    """Write the four PR1 artefacts. Returns a mapping of name → written path."""

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest.json": out_dir / "manifest.json",
        "decision_card.md": out_dir / "decision_card.md",
        "gate_table.json": out_dir / "gate_table.json",
        "report.md": out_dir / "report.md",
    }
    paths["manifest.json"].write_bytes(render_manifest_json(manifest))
    paths["gate_table.json"].write_bytes(render_gate_table_json(manifest))
    paths["decision_card.md"].write_text(render_decision_card_md(manifest), encoding="utf-8")
    paths["report.md"].write_text(render_report_md(manifest), encoding="utf-8")
    return paths


# ─────────────────────────── Main entrypoint ─────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.html or args.include_extracts:
        print(
            "warning: --html and --include-extracts are reserved for follow-up PRs; ignored in PR1.",
            file=sys.stderr,
        )

    project_root = PROJECT_ROOT
    db_path: Path
    try:
        sys.path.insert(0, str(project_root))
        from pipeline.paths import GOLD_DB_PATH  # type: ignore

        db_path = GOLD_DB_PATH
    except Exception:
        db_path = project_root / "gold.db"

    if args.prereg:
        prereg_path, slug = resolve_prereg(args.prereg, project_root)
    elif args.validated_setup or args.experimental:
        # PR1 supports the resolver-shape for these modes via slug only;
        # the richer DB lookup is PR2 territory but the slug routes the
        # rest of the pipeline correctly.
        slug = (args.validated_setup or args.experimental).replace(":", "_")
        prereg_path = None
    else:  # pragma: no cover — argparse enforces required group
        parser.error("one of --prereg / --validated-setup / --experimental required")
        return 2

    git_sha = repo_git_sha(project_root)
    fingerprint = db_fingerprint(db_path)

    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = project_root / out_dir
    else:
        run_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        out_dir = project_root / PACK_ROOT / slug / run_iso

    manifest = build_manifest(
        slug=slug,
        prereg_path=prereg_path,
        project_root=project_root,
        git_sha=git_sha,
        fingerprint=fingerprint,
        db_path_str=str(db_path),
        allow_fingerprint_drift=args.allow_fingerprint_drift,
        out_dir=out_dir,
    )

    written = write_pack(out_dir, manifest)
    print(f"evidence-pack written: {out_dir}", file=sys.stderr)
    for name, p in sorted(written.items()):
        print(f"  - {name}: {p}", file=sys.stderr)
    print(f"verdict: {manifest.verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
