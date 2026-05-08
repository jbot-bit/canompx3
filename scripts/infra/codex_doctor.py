#!/usr/bin/env python3
"""Diagnostic for `codex.bat` smart-mode WSL clone path.

Prints a short report explaining whether the smart launcher will engage the
WSL-home clone (fast path) or fall back to the Windows checkout (slow path),
and which precondition is failing if it falls back.

Read-only. Reuses helpers from `windows_agent_launch` so the verdict can never
drift from what the smart codepath actually probes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Same-dir import: this module lives next to windows_agent_launch.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from windows_agent_launch import (  # noqa: E402
    repo_root,
    wsl_home_clone_available,
)


def _git(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return 1, f"git not runnable: {exc}"
    return result.returncode, (result.stdout or result.stderr or "").strip()


def _wsl(script: str) -> tuple[int, str]:
    # Multi-line bash via `wsl.exe bash -lc <text>` loses statements at the
    # Win32->WSL argv boundary (same class as the bug documented in
    # windows_agent_launch.run_wsl). Write to a tempfile in .claude/scratch
    # and execute as a real script file.
    import tempfile

    scratch_dir = repo_root() / ".claude" / "scratch"
    tmp_dir = str(scratch_dir) if scratch_dir.exists() else str(repo_root())
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            prefix="codex_doctor_",
            dir=tmp_dir,
            delete=False,
            newline="\n",
            encoding="utf-8",
        ) as tmp:
            tmp.write(script)
            tmp_name = tmp.name

        full = str(Path(tmp_name).resolve())
        drive = full[0].lower()
        rest = full[2:].replace("\\", "/")
        wsl_path = f"/mnt/{drive}{rest}"

        result = subprocess.run(
            ["wsl.exe", "bash", wsl_path],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
    except OSError as exc:
        return 1, f"wsl.exe not runnable: {exc}"
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass

    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    return result.returncode, out or err


def _wsl_clone_root_script() -> str:
    # Mirror of the resolution logic inside `wsl_home_clone_available`. Kept in
    # sync by routing the actual READY/BLOCKED verdict through that function;
    # this script just surfaces the resolved path for the human-readable report.
    return "\n".join(
        [
            "set -eu",
            'ROOT_INPUT="${CANOMPX3_CODEX_WSL_ROOT:-}"',
            'if [ -z "$ROOT_INPUT" ] || [ "$ROOT_INPUT" = "." ] || [ "$ROOT_INPUT" = "./" ]; then',
            '  ROOT="$HOME/canompx3"',
            'elif [ "$ROOT_INPUT" = "~" ]; then',
            '  ROOT="$HOME"',
            'elif [ "${ROOT_INPUT#~/}" != "$ROOT_INPUT" ]; then',
            '  ROOT="$HOME/${ROOT_INPUT#~/}"',
            'elif [ "${ROOT_INPUT#/}" != "$ROOT_INPUT" ]; then',
            '  ROOT="$ROOT_INPUT"',
            "else",
            '  echo "INVALID:$ROOT_INPUT"',
            "  exit 0",
            "fi",
            'echo "$ROOT"',
        ]
    )


def run() -> int:
    print("=== codex.bat smart-mode doctor ===")
    print()

    # 1. Windows checkout
    win_root = repo_root()
    print(f"Windows checkout: {win_root}")
    rc, branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=win_root)
    win_branch = branch if rc == 0 else "<unknown>"
    rc, status = _git(["status", "--short"], cwd=win_root)
    win_dirty = bool(status) if rc == 0 else False
    rc, win_head = _git(["rev-parse", "HEAD"], cwd=win_root)
    win_head_short = win_head[:12] if rc == 0 else "<unknown>"
    print(f"  branch:  {win_branch}")
    print(f"  HEAD:    {win_head_short}")
    print(f"  status:  {'dirty' if win_dirty else 'clean'}")
    print()

    # 2. WSL reachable + tools
    print("WSL toolchain:")
    rc_wsl, _ = _wsl("true")
    if rc_wsl != 0:
        print("  wsl.exe: UNREACHABLE")
        print()
        print("SMART_PATH=BLOCKED: wsl.exe not reachable")
        return 0
    print("  wsl.exe: reachable")
    for tool in ("uv", "git", "python3"):
        # Use `bash -lc` so the user's ~/.profile / ~/.bashrc PATH additions
        # are applied; this matches what codex-wsl-sync.sh sees at launch
        # time. Single-line probes survive the Win32->WSL argv boundary.
        try:
            result = subprocess.run(
                ["wsl.exe", "bash", "-lc", f"command -v {tool} >/dev/null 2>&1 && echo present || echo missing"],
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
            out = (result.stdout or "").strip() or "unknown"
        except OSError:
            out = "unknown"
        print(f"  {tool}: {out}")
    print()

    # 3. WSL clone presence
    env_root = os.environ.get("CANOMPX3_CODEX_WSL_ROOT", "")
    print(f"CANOMPX3_CODEX_WSL_ROOT: {env_root or '(unset)'}")
    rc, resolved_root = _wsl(_wsl_clone_root_script())
    if rc != 0 or not resolved_root:
        print("  resolved root: <could not resolve>")
        clone_present = False
        wsl_branch = "<n/a>"
        wsl_head_short = "<n/a>"
    elif resolved_root.startswith("INVALID:"):
        print(f"  resolved root: INVALID ({resolved_root.split(':', 1)[1]})")
        print("  CANOMPX3_CODEX_WSL_ROOT must be empty, '~', '~/...', or absolute (/...)")
        clone_present = False
        wsl_branch = "<n/a>"
        wsl_head_short = "<n/a>"
    else:
        print(f"  resolved root: {resolved_root}")
        clone_present = wsl_home_clone_available()
        if clone_present:
            rc, wsl_branch = _wsl(f'git -C "{resolved_root}" rev-parse --abbrev-ref HEAD')
            if rc != 0:
                wsl_branch = "<unknown>"
            rc, wsl_head = _wsl(f'git -C "{resolved_root}" rev-parse HEAD')
            wsl_head_short = wsl_head[:12] if rc == 0 else "<unknown>"
            print(f"  clone:   present  ({wsl_branch} @ {wsl_head_short})")
        else:
            wsl_branch = "<absent>"
            wsl_head_short = "<absent>"
            print("  clone:   ABSENT (no .git directory at resolved root)")
    print()

    # 4. Verdict
    if not clone_present:
        print("SMART_PATH=BLOCKED: WSL clone missing")
        print()
        print("To bootstrap (run once):")
        print("  wsl.exe bash -lc 'cd ~ && git clone <origin-url> canompx3'")
        print()
        print("Confirm origin URL with:")
        print(f"  git -C {win_root} remote get-url origin")
        return 0

    if win_dirty:
        print("SMART_PATH=BLOCKED: Windows checkout dirty (codex-wsl-sync.sh requires clean tree)")
        return 0

    if win_branch != wsl_branch:
        print(f"SMART_PATH=BLOCKED: branch mismatch (windows={win_branch}, wsl={wsl_branch})")
        return 0

    if win_branch in ("HEAD", "<unknown>") or wsl_branch in ("HEAD", "<unknown>"):
        print("SMART_PATH=BLOCKED: detached HEAD on one side")
        return 0

    print("SMART_PATH=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
