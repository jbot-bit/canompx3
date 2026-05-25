"""DuckDB connection helpers for lock-contention-safe open operations."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from pipeline.log import get_logger

logger = get_logger(__name__)

_LOCK_ERROR_MARKERS = (
    "being used by another process",
    "could not set lock",
)


def _is_lock_class_ioerror(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _LOCK_ERROR_MARKERS)


def _retry_delay(attempt_index: int, *, base_delay: float, max_delay: float) -> float:
    base = min(base_delay * (2**attempt_index), max_delay)
    return base * (0.75 + 0.5 * random.random())


def _open_with_retry(
    db_path: str | Path,
    *,
    read_only: bool,
    attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Any:
    """Open a DuckDB connection, retrying only documented lock-class IOErrors."""
    import duckdb

    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    path = str(db_path)
    last_exc: Exception | None = None
    mode = "read-only" if read_only else "writer"

    for attempt_index in range(attempts):
        try:
            if read_only:
                return duckdb.connect(path, read_only=True)
            return duckdb.connect(path)
        except duckdb.IOException as exc:
            if not _is_lock_class_ioerror(exc):
                raise
            last_exc = exc
            if attempt_index == attempts - 1:
                break
            delay = _retry_delay(attempt_index, base_delay=base_delay, max_delay=max_delay)
            logger.warning(
                "[duckdb-%s-retry] gold.db locked (attempt %s/%s); sleeping %.1fs",
                mode,
                attempt_index + 1,
                attempts,
                delay,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def open_read_only_with_retry(
    db_path: str | Path,
    *,
    attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Any:
    """Open DuckDB read-only, retrying peer-process file-lock contention."""
    return _open_with_retry(
        db_path,
        read_only=True,
        attempts=attempts,
        base_delay=base_delay,
        max_delay=max_delay,
    )


def open_writer_with_retry(
    db_path: str | Path,
    *,
    attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Any:
    """Open DuckDB in default write mode, retrying peer-process file locks."""
    return _open_with_retry(
        db_path,
        read_only=False,
        attempts=attempts,
        base_delay=base_delay,
        max_delay=max_delay,
    )
