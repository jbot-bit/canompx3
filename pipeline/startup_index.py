"""Generated startup index for packet-driven orientation."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from context.registry import FILE_PACKETS, TASKS, VERIFICATION_PROFILES, resolve_task

STARTUP_INDEX_PATH = Path(".canompx3-runtime/startup_index.json")


def build_startup_index_payload() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "file_packets": {packet.path: asdict(packet) for packet in FILE_PACKETS.values()},
        "verification_profiles": {key: asdict(value) for key, value in VERIFICATION_PROFILES.items()},
        "task_routes": {
            task_id: {
                "task_id": task_id,
                "doctrine_files": list(resolve_task(task_id).doctrine_files),
                "canonical_files": list(resolve_task(task_id).canonical_files),
                "live_views": [view.id for view in resolve_task(task_id).live_views],
                "verification_profile": resolve_task(task_id).verification.id,
            }
            for task_id in TASKS
        },
    }


def startup_index_path(root: Path) -> Path:
    return root / STARTUP_INDEX_PATH


def write_startup_index(root: Path) -> Path:
    path = startup_index_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_startup_index_payload(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_startup_index(root: Path) -> dict[str, Any]:
    path = startup_index_path(root)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return build_startup_index_payload()
