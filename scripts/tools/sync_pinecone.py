#!/usr/bin/env python3
"""Sync project knowledge to Pinecone Assistant.

Collects files across 5 content tiers (static, living, memory, research_output,
generated), hashes them with SHA256, and uploads only changed files to the
Pinecone Assistant. Snapshots are regenerated every run.

Usage:
    python scripts/tools/sync_pinecone.py              # full sync
    python scripts/tools/sync_pinecone.py --dry-run    # show what would upload
    python scripts/tools/sync_pinecone.py --force      # upload everything
"""

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root + imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scripts.tools.pinecone_snapshots import (
    generate_fitness_report_snapshot,
    generate_live_config_snapshot,
    generate_portfolio_state_snapshot,
    generate_research_index_snapshot,
    save_snapshot,
)

MANIFEST_PATH = PROJECT_ROOT / "scripts" / "tools" / "pinecone_manifest.json"
STATE_PATH = PROJECT_ROOT / "scripts" / "tools" / ".pinecone_sync_state.json"
ASSISTANT_ID_PATH = PROJECT_ROOT / "scripts" / "tools" / ".pinecone_assistant_id"
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"

# ---------------------------------------------------------------------------
# Research output topic prefixes → bundle names
# ---------------------------------------------------------------------------

# Order matters: longer/more-specific prefixes first to avoid mis-grouping.
RESEARCH_BUNDLE_GROUPS: list[tuple[str, list[str]]] = [
    ("a0_analysis", ["a0_"]),
    ("dalton_research", ["dalton_"]),
    ("leadlag_research", ["lead_lag_", "fast_lead_lag_", "fast_proxy_"]),
    ("forward_gate", ["forward_gate_"]),
    ("session_research", ["session_"]),
    ("shinies_research", ["shinies_"]),
    ("wide_regime", ["wide_"]),
    ("dst_research", ["dst_", "DST_"]),
]


# ---------------------------------------------------------------------------
# Manifest + state helpers
# ---------------------------------------------------------------------------


def load_manifest() -> dict:
    """Load the pinecone manifest from disk."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state() -> dict:
    """Load previous sync state (file hashes). Returns empty dict on first run."""
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    """Save sync state to disk."""
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def hash_file(path: Path) -> str:
    """SHA256 hash of file content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_assistant_name() -> str:
    """Read assistant name from .pinecone_assistant_id file."""
    if ASSISTANT_ID_PATH.exists():
        return ASSISTANT_ID_PATH.read_text(encoding="utf-8").strip()
    # Fallback to manifest
    manifest = load_manifest()
    return manifest.get("assistant_name", "orb-research")


# ---------------------------------------------------------------------------
# Research output bundling (Issue 1: 100-file quota)
# ---------------------------------------------------------------------------


def _classify_research_file(filename: str) -> str:
    """Return the bundle topic for a research output filename."""
    lower = filename.lower()
    for topic, prefixes in RESEARCH_BUNDLE_GROUPS:
        for prefix in prefixes:
            if lower.startswith(prefix.lower()):
                return topic
    return "misc_research"


def bundle_research_output(
    research_files: list[tuple[Path, str]],
) -> list[tuple[Path, str]]:
    """Bundle research output files by topic prefix into fewer combined files.

    Groups files by their topic prefix (before first underscore or known
    pattern), concatenates each group into a single markdown document with
    clear section headers, and writes to scripts/tools/_bundle_*.md.

    Returns new list of (abs_path, rel_key) for the bundled files.
    """
    from collections import defaultdict

    groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)

    for abs_path, rel_key in research_files:
        topic = _classify_research_file(abs_path.name)
        groups[topic].append((abs_path, rel_key))

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    bundled: list[tuple[Path, str]] = []

    for topic in sorted(groups.keys()):
        files_in_group = groups[topic]
        bundle_name = f"_bundle_{topic}.md"
        bundle_path = TOOLS_DIR / bundle_name
        rel_key = f"scripts/tools/{bundle_name}"

        lines = [
            f"# Research Bundle: {topic.replace('_', ' ').title()}",
            "",
            f"Generated: {timestamp}",
            f"Files included: {len(files_in_group)}",
            "",
        ]

        for abs_path, _original_key in sorted(files_in_group, key=lambda x: x[1]):
            lines.append("---")
            lines.append("")
            lines.append(f"## {abs_path.name}")
            lines.append("")
            try:
                content = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = abs_path.read_bytes().decode("utf-8", errors="replace")
            lines.append(content.rstrip())
            lines.append("")

        lines.append("---")

        bundle_path.write_text("\n".join(lines), encoding="utf-8")
        bundled.append((bundle_path, rel_key))
        print(f"  Bundled {len(files_in_group)} files -> {bundle_name}")

    return bundled


# ---------------------------------------------------------------------------
# Living file .py → .md conversion (Issue 2: .py rejected)
# ---------------------------------------------------------------------------


def prepare_living_as_md(
    living_files: list[tuple[Path, str]],
) -> list[tuple[Path, str]]:
    """Convert .py config files to .md for Pinecone upload.

    For each .py file, writes a markdown wrapper to
    scripts/tools/_living_{basename}.md with the code in a fenced block.
    Non-.py files are passed through unchanged.

    Returns new list of (abs_path, rel_key).
    """
    result: list[tuple[Path, str]] = []

    for abs_path, rel_key in living_files:
        if abs_path.suffix == ".py":
            md_name = f"_living_{abs_path.stem}.md"
            md_path = TOOLS_DIR / md_name
            md_rel_key = f"scripts/tools/{md_name}"

            try:
                code = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                code = abs_path.read_bytes().decode("utf-8", errors="replace")

            content = f"# {abs_path.name} (Source of Truth)\n\nSource: `{rel_key}`\n\n```python\n{code}\n```\n"
            md_path.write_text(content, encoding="utf-8")
            result.append((md_path, md_rel_key))
            print(f"  Converted {rel_key} -> {md_name}")
        else:
            result.append((abs_path, rel_key))

    return result


# ---------------------------------------------------------------------------
# UTF-8 sanitization (Issue 3: encoding errors)
# ---------------------------------------------------------------------------


def ensure_utf8(file_path: Path) -> Path:
    """Ensure file is valid UTF-8. Returns path to clean file (may be temp copy)."""
    try:
        file_path.read_text(encoding="utf-8")
        return file_path  # Already valid
    except UnicodeDecodeError:
        # Read with replacement and write clean copy
        content = file_path.read_bytes().decode("utf-8", errors="replace")
        clean_path = TOOLS_DIR / f"_clean_{file_path.name}"
        clean_path.write_text(content, encoding="utf-8")
        print(f"  Sanitized UTF-8: {file_path.name} -> _clean_{file_path.name}")
        return clean_path


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_coaching_files(*, data_dir: Path = PROJECT_ROOT / "data") -> list[tuple[Path, str]]:
    """Collect coaching files for Pinecone sync."""
    files = []
    for filename in ("trader_profile.json", "coaching_digests.jsonl"):
        path = data_dir / filename
        if path.exists():
            files.append((path, f"coaching/{filename}"))
    return files


def collect_files(manifest: dict) -> dict[str, list[tuple[Path, str]]]:
    """Collect all files from manifest, grouped by tier.

    Returns {tier_name: [(absolute_path, relative_key), ...]}.
    The relative_key is used as the identifier for change detection.
    """
    tiers = manifest["content_tiers"]
    collected: dict[str, list[tuple[Path, str]]] = {}

    # --- Static files ---
    static_files = []
    for rel_path in tiers["static"]["files"]:
        abs_path = PROJECT_ROOT / rel_path
        if abs_path.exists():
            static_files.append((abs_path, rel_path))
        else:
            print(f"  WARNING: static file missing: {rel_path}")
    collected["static"] = static_files

    # --- Living files (convert .py → .md for Pinecone) ---
    raw_living = []
    for rel_path in tiers["living"]["files"]:
        abs_path = PROJECT_ROOT / rel_path
        if abs_path.exists():
            raw_living.append((abs_path, rel_path))
        else:
            print(f"  WARNING: living file missing: {rel_path}")
    collected["living"] = prepare_living_as_md(raw_living)

    # --- Memory files (absolute path + glob) ---
    memory_cfg = tiers["memory"]
    base_path = Path(memory_cfg["base_path"])
    pattern = memory_cfg["glob_pattern"]
    memory_files = []
    if base_path.exists():
        for f in sorted(base_path.glob(pattern)):
            if f.is_file():
                # Use memory/<filename> as the relative key
                rel_key = f"memory/{f.name}"
                memory_files.append((f, rel_key))
    else:
        print(f"  WARNING: memory base_path missing: {base_path}")
    collected["memory"] = memory_files

    # --- Research output (bundle into ~10 topic files) ---
    research_cfg = tiers["research_output"]
    raw_research: list[tuple[Path, str]] = []
    seen_paths: set[Path] = set()
    for glob_pattern in research_cfg["glob_patterns"]:
        for f in sorted(PROJECT_ROOT.glob(glob_pattern)):
            if f.is_file() and f not in seen_paths:
                seen_paths.add(f)
                try:
                    rel_key = str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
                except ValueError:
                    rel_key = f.name
                raw_research.append((f, rel_key))
    collected["research_output"] = bundle_research_output(raw_research)

    # --- Generated snapshots ---
    generated_cfg = tiers["generated"]
    output_dir = PROJECT_ROOT / generated_cfg["output_dir"]
    generated_files = []
    for filename in generated_cfg["files"]:
        abs_path = output_dir / filename
        if abs_path.exists():
            rel_key = f"{generated_cfg['output_dir']}/{filename}"
            generated_files.append((abs_path, rel_key))
        else:
            print(f"  WARNING: generated snapshot missing: {filename}")
    collected["generated"] = generated_files

    # --- Coaching files ---
    if "coaching" in tiers:
        collected["coaching"] = collect_coaching_files()

    return collected


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------


def detect_changes(
    collected: dict[str, list[tuple[Path, str]]],
    previous_hashes: dict[str, str],
    force: bool = False,
) -> tuple[dict[str, list[tuple[Path, str, str]]], dict[str, str]]:
    """Compare file hashes to previous state.

    Returns:
        changed: {tier: [(abs_path, rel_key, sha256), ...]} — files to upload
        current_hashes: {rel_key: sha256} — full state for saving
    """
    changed: dict[str, list[tuple[Path, str, str]]] = {}
    current_hashes: dict[str, str] = {}

    for tier, files in collected.items():
        tier_changed = []
        for abs_path, rel_key in files:
            file_hash = hash_file(abs_path)
            current_hashes[rel_key] = file_hash

            if force or previous_hashes.get(rel_key) != file_hash:
                tier_changed.append((abs_path, rel_key, file_hash))

        if tier_changed:
            changed[tier] = tier_changed

    return changed, current_hashes


# ---------------------------------------------------------------------------
# Pinecone upload
# ---------------------------------------------------------------------------


def upload_to_pinecone(
    changed: dict[str, list[tuple[Path, str, str]]],
    assistant_name: str,
) -> tuple[int, int]:
    """Upload changed files to Pinecone Assistant.

    Returns (success_count, failure_count).
    """
    from pinecone import Pinecone

    pc = Pinecone()  # reads PINECONE_API_KEY from env
    assistant = pc.assistant.Assistant(assistant_name=assistant_name)

    # Get existing files for conflict detection
    existing_files = assistant.list_files()
    existing_names = {f.name: f.id for f in existing_files}

    success = 0
    failure = 0

    for tier, files in changed.items():
        for abs_path, rel_key, _ in files:
            # Ensure valid UTF-8 before upload (Issue 3)
            upload_path = ensure_utf8(abs_path)
            basename = upload_path.name

            # Check for name conflict — delete old version first
            if basename in existing_names:
                try:
                    print(f"  Replacing existing file: {basename}")
                    assistant.delete_file(file_id=existing_names[basename])
                except Exception as e:
                    print(f"  WARNING: failed to delete old {basename}: {e}")

            try:
                print(f"  Uploading: {rel_key} (tier={tier})")
                assistant.upload_file(
                    file_path=str(upload_path),
                    metadata={"tier": tier, "rel_key": rel_key},
                    timeout=None,
                )
                success += 1
            except Exception as e:
                print(f"  ERROR: failed to upload {rel_key}: {e}")
                failure += 1

    return success, failure


# ---------------------------------------------------------------------------
# Basename conflict checker
# ---------------------------------------------------------------------------


def check_basename_conflicts(
    collected: dict[str, list[tuple[Path, str]]],
) -> list[str]:
    """Check for files with the same basename across tiers."""
    seen: dict[str, list[str]] = {}
    for _tier, files in collected.items():
        for abs_path, rel_key in files:
            basename = abs_path.name
            seen.setdefault(basename, []).append(rel_key)

    conflicts = []
    for basename, keys in seen.items():
        if len(keys) > 1:
            conflicts.append(f"  {basename}: {keys}")
    return conflicts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sync project knowledge to Pinecone Assistant")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload all files regardless of hash changes",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Pinecone Assistant Sync")
    print("=" * 60)
    print()

    # 1. Load manifest
    print("[1/6] Loading manifest...")
    manifest = load_manifest()
    assistant_name = get_assistant_name()
    print(f"  Assistant: {assistant_name}")

    # 2. Load previous state
    print("[2/6] Loading sync state...")
    previous_state = load_state()
    previous_hashes = previous_state.get("hashes", {})
    last_sync = previous_state.get("last_sync")
    if last_sync:
        print(f"  Last sync: {last_sync}")
    else:
        print("  First run — no previous state")

    # 3. Generate snapshots
    print("[3/6] Generating snapshots...")
    snapshot_generators = {
        "_snapshot_portfolio_state.md": generate_portfolio_state_snapshot,
        "_snapshot_fitness_report.md": generate_fitness_report_snapshot,
        "_snapshot_live_config.md": generate_live_config_snapshot,
        "_snapshot_research_index.md": generate_research_index_snapshot,
    }
    for filename, gen_func in snapshot_generators.items():
        content = gen_func()
        save_snapshot(content, filename)
        print(f"  Generated: {filename}")

    # 4. Collect files
    print("[4/6] Collecting files...")
    collected = collect_files(manifest)
    total_files = sum(len(files) for files in collected.values())
    for tier, files in collected.items():
        print(f"  {tier}: {len(files)} files")
    print(f"  TOTAL: {total_files} files")

    # Check basename conflicts
    conflicts = check_basename_conflicts(collected)
    if conflicts:
        print()
        print("  WARNING: basename conflicts detected:")
        for c in conflicts:
            print(c)
        print("  Pinecone uses basename — last upload wins for duplicates.")
        print()

    # 5. Detect changes
    print("[5/6] Detecting changes...")
    changed, current_hashes = detect_changes(collected, previous_hashes, force=args.force)
    total_changed = sum(len(files) for files in changed.values())

    if total_changed == 0:
        print("  No changes detected. Nothing to upload.")
        print()
        print("Done.")
        return

    print(f"  {total_changed} file(s) changed:")
    for tier, files in changed.items():
        for _, rel_key, _ in files:
            print(f"    [{tier}] {rel_key}")

    # 6. Upload (or dry-run)
    if args.dry_run:
        print()
        print("[6/6] DRY RUN — skipping upload")
        print(f"  Would upload {total_changed} file(s) to '{assistant_name}'")
    else:
        print()
        print(f"[6/6] Uploading {total_changed} file(s) to '{assistant_name}'...")
        success, failure = upload_to_pinecone(changed, assistant_name)
        print()
        print(f"  Uploaded: {success}, Failed: {failure}")

        if failure > 0:
            print()
            print("ERROR: Some uploads failed. State NOT saved.")
            print("Re-run to retry failed uploads.")
            sys.exit(1)

        # Save state only on full success
        state = {
            "last_sync": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "assistant_name": assistant_name,
            "hashes": current_hashes,
            "file_count": total_files,
        }
        save_state(state)
        print("  State saved.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
