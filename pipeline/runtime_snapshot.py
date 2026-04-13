"""Materialized runtime snapshot contract for fast orientation surfaces."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNTIME_SNAPSHOT_PATH = Path(".canompx3-runtime/runtime_snapshot.json")
DEFAULT_RUNTIME_SNAPSHOT_MAX_AGE_SECONDS = 900


def runtime_snapshot_path(root: Path) -> Path:
    return root / RUNTIME_SNAPSHOT_PATH


def read_runtime_snapshot(root: Path) -> dict[str, Any] | None:
    path = runtime_snapshot_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_runtime_snapshot(root: Path, payload: dict[str, Any]) -> Path:
    path = runtime_snapshot_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def snapshot_age_seconds(payload: dict[str, Any], now: datetime | None = None) -> int | None:
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str):
        return None
    try:
        generated = datetime.fromisoformat(generated_at)
    except ValueError:
        return None
    current = now or datetime.now(UTC)
    return max(0, int((current - generated).total_seconds()))


def snapshot_is_fresh(payload: dict[str, Any], *, max_age_seconds: int = DEFAULT_RUNTIME_SNAPSHOT_MAX_AGE_SECONDS) -> bool:
    age = snapshot_age_seconds(payload)
    return age is not None and age <= max_age_seconds
