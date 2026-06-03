"""Canonical writer/reader for the dashboard's planned-launch surface.

Single source of truth for "what is about to launch" — written by both
START_BOT.bat (via `python -m trading_app.live.planned_launch write …`) and
the CLI codepath in `scripts/run_live_session.py` at session boot.

The artifact lives at `data/bot_planned_launch.json` and is read by the
dashboard's `/api/planned-launch` endpoint. It is gitignored runtime state.

Schema (v1):
    {
      "schema_version": 1,
      "profile_id": "topstep_50k_mnq_auto",
      "mode": "SIGNAL" | "DEMO" | "LIVE",
      "copies": 1,
      "instruments": ["MNQ"],
      "broker_accounts_count": 1,
      "source": "START_BOT.bat" | "CLI" | "dashboard",
      "ts": "2026-05-28T03:14:15+00:00"
    }

Fail-visible doctrine: if the file is missing, malformed, or older than
``STALE_AFTER_SECONDS``, the reader returns a structured "unknown" record
rather than guessing. The dashboard renders that explicitly.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PLANNED_LAUNCH_PATH = Path(__file__).parent.parent.parent / "data" / "bot_planned_launch.json"
SCHEMA_VERSION = 1
STALE_AFTER_SECONDS = 24 * 60 * 60

VALID_MODES = frozenset({"SIGNAL", "DEMO", "LIVE"})
VALID_SOURCES = frozenset({"START_BOT.bat", "CLI", "dashboard"})


def write_planned_launch(
    *,
    profile_id: str,
    mode: str,
    source: str,
    copies: int | None = None,
    instruments: list[str] | None = None,
    broker_accounts_count: int | None = None,
) -> dict[str, Any]:
    """Write the planned-launch artifact atomically.

    Looks up ``copies`` and ``instruments`` from
    ``trading_app.prop_profiles.ACCOUNT_PROFILES`` when not supplied — keeps
    callers honest by deriving from the canonical profile registry.

    Returns the payload that was written (useful for test assertions and for
    callers that want to echo to stdout).

    Raises ValueError if mode or source is not in the allowed set, or if
    profile_id is not registered. Never silently writes bad data.
    """
    mode_normalized = (mode or "").upper()
    if mode_normalized not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}")

    if copies is None or instruments is None:
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        if profile_id not in ACCOUNT_PROFILES:
            raise ValueError(
                f"profile_id {profile_id!r} not in ACCOUNT_PROFILES — "
                f"add it to trading_app/prop_profiles.py or pass copies/instruments explicitly"
            )
        profile = ACCOUNT_PROFILES[profile_id]
        if copies is None:
            copies = profile.copies
        if instruments is None:
            instruments = sorted(profile.allowed_instruments) if profile.allowed_instruments else []

    if broker_accounts_count is None:
        broker_accounts_count = copies

    payload = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "mode": mode_normalized,
        "copies": int(copies),
        "instruments": list(instruments),
        "broker_accounts_count": int(broker_accounts_count),
        "source": source,
        "ts": datetime.now(UTC).isoformat(),
    }

    PLANNED_LAUNCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PLANNED_LAUNCH_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(PLANNED_LAUNCH_PATH))
    return payload


def read_planned_launch() -> dict[str, Any]:
    """Read the planned-launch artifact with fail-visible staleness handling.

    Returns one of three shapes, always with a top-level ``status`` field:

    - ``{"status": "ok", ...payload, "age_seconds": int}`` — fresh and valid.
    - ``{"status": "stale", ...payload, "age_seconds": int}`` — older than
      ``STALE_AFTER_SECONDS``. Payload is included so the dashboard can show
      what it was, but UI must treat it as untrustworthy.
    - ``{"status": "unknown", "reason": "..."}`` — file missing, malformed,
      or schema mismatch.

    Never raises. Never guesses.
    """
    if not PLANNED_LAUNCH_PATH.exists():
        return {"status": "unknown", "reason": "no planned launch file"}
    try:
        raw = PLANNED_LAUNCH_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unknown", "reason": f"read_error:{type(exc).__name__}"}

    if not isinstance(payload, dict):
        return {"status": "unknown", "reason": "payload not a dict"}
    if payload.get("schema_version") != SCHEMA_VERSION:
        return {
            "status": "unknown",
            "reason": f"schema_version mismatch: file={payload.get('schema_version')!r} expected={SCHEMA_VERSION}",
        }
    required = {"profile_id", "mode", "copies", "instruments", "broker_accounts_count", "source", "ts"}
    missing = required - set(payload)
    if missing:
        return {"status": "unknown", "reason": f"missing fields: {sorted(missing)}"}
    if payload["mode"] not in VALID_MODES:
        return {"status": "unknown", "reason": f"invalid mode {payload['mode']!r}"}

    try:
        ts = datetime.fromisoformat(payload["ts"])
    except (ValueError, TypeError):
        return {"status": "unknown", "reason": "ts not ISO-8601"}
    age = (datetime.now(UTC) - ts).total_seconds()
    payload["age_seconds"] = int(age)

    if age > STALE_AFTER_SECONDS:
        payload["status"] = "stale"
    else:
        payload["status"] = "ok"
    return payload


def clear_planned_launch() -> None:
    """Remove the artifact. Called by orchestrator on clean shutdown so the
    next pre-start view reflects "no launch planned" rather than stale state.
    """
    PLANNED_LAUNCH_PATH.unlink(missing_ok=True)


def _cli_main(argv: list[str]) -> int:
    """Minimal CLI so START_BOT.bat can invoke without inlining the schema.

    Usage:
        python -m trading_app.live.planned_launch write \\
            --profile <id> --mode <SIGNAL|DEMO|LIVE> --source <START_BOT.bat|CLI|dashboard> \\
            [--copies <n>] [--instrument <symbol>]

        python -m trading_app.live.planned_launch read
        python -m trading_app.live.planned_launch clear
    """
    if not argv or argv[0] == "read":
        print(json.dumps(read_planned_launch(), indent=2))
        return 0
    if argv[0] == "clear":
        clear_planned_launch()
        return 0
    if argv[0] != "write":
        print(f"unknown subcommand {argv[0]!r}", file=sys.stderr)
        return 2

    args = argv[1:]
    kwargs: dict[str, str] = {}
    instruments: list[str] = []
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            key = args[i][2:]
            if key == "instrument":
                instruments.append(args[i + 1])
            else:
                kwargs[key] = args[i + 1]
            i += 2
        else:
            print(f"bad arg {args[i]!r}", file=sys.stderr)
            return 2

    try:
        payload = write_planned_launch(
            profile_id=kwargs["profile"],
            mode=kwargs["mode"],
            source=kwargs["source"],
            copies=int(kwargs["copies"]) if "copies" in kwargs else None,
            instruments=instruments or None,
        )
    except KeyError as exc:
        print(f"missing required flag: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main(sys.argv[1:]))
