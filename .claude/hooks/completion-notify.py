#!/usr/bin/env python3
"""Completion notification hook: plays a sound when Claude finishes a task or needs input.

Fires on: Notification, Stop events.
Windows: uses winsound beep. Cross-platform fallback: terminal bell.

Dedupe: 2-second cooldown prevents double-beep when a Notification and Stop
fire back-to-back on the same completion (common pattern).
"""

import json
import sys
import time
from pathlib import Path

_COOLDOWN_FILE = Path(__file__).parent / ".completion-notify-last"
_COOLDOWN_SECONDS = 2.0


def _in_cooldown() -> bool:
    try:
        age = time.time() - _COOLDOWN_FILE.stat().st_mtime
        return age < _COOLDOWN_SECONDS
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _touch_cooldown() -> None:
    try:
        _COOLDOWN_FILE.touch()
    except OSError:
        pass


def main():
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as exc:
        print(f"[completion-notify] unexpected: {exc}", file=sys.stderr)
        sys.exit(0)

    hook_event = event.get("hook_event_name", "")

    # Only notify on meaningful completions
    if hook_event not in ("Notification", "Stop"):
        sys.exit(0)

    # Dedupe: Notification + Stop often fire in quick succession for the same
    # completion. Suppress the second one.
    if _in_cooldown():
        sys.exit(0)
    _touch_cooldown()

    # Play sound
    try:
        import winsound
        # Two short beeps: task done
        winsound.Beep(800, 200)
        winsound.Beep(1000, 200)
    except (ImportError, RuntimeError):
        # Fallback: terminal bell
        print("\a", end="", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
