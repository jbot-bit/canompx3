#!/usr/bin/env python3
"""Completion notification hook: plays a sound when Claude finishes a task or needs input.

Fires on: Notification, Stop events.
Windows: uses winsound beep. Cross-platform fallback: terminal bell.
"""

import json
import sys


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
