#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

VALID_MODES = {
    "claude",
    "codex",
    "codex-search",
    "list",
    "close",
    "close-pick",
    "resume",
    "menu",
    "prune",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def manager_py() -> Path:
    return repo_root() / "scripts" / "tools" / "worktree_manager.py"


def windows_to_git_bash(path_value: Path) -> str:
    full = str(path_value.resolve())
    drive = full[0].lower()
    rest = full[2:].replace("\\", "/")
    return f"/{drive}{rest}"


def windows_to_wsl(path_value: Path) -> str:
    full = str(path_value.resolve())
    drive = full[0].lower()
    rest = full[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def pick_python() -> list[str]:
    root = repo_root()
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return [str(venv_python)]
    if shutil.which("py"):
        return ["py", "-3"]
    if shutil.which("python"):
        return ["python"]
    raise RuntimeError(
        "No usable Python launcher found for AI Workstreams. Expected .venv\\Scripts\\python.exe, `py -3`, or `python`."
    )


def invoke_manager(arguments: list[str]) -> tuple[bool, str]:
    cmd = pick_python() + [str(manager_py())] + arguments
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        return False, f"Manager failed to start: {exc}"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    if result.returncode == 0:
        return True, output
    message = error or output or f"Manager exited with code {result.returncode}"
    return False, message


def get_managed_workstreams() -> list[dict[str, Any]]:
    success, output = invoke_manager(["list", "--managed-only", "--json"])
    if not success:
        print()
        print(output)
        return []
    if not output:
        return []
    data = json.loads(output)
    return data if isinstance(data, list) else []


def get_existing_purpose(tool_name: str, workstream_name: str) -> str | None:
    success, output = invoke_manager(["show", "--tool", tool_name, "--name", workstream_name])
    if not success or not output or output == "{}":
        return None
    try:
        meta = json.loads(output)
    except json.JSONDecodeError:
        return None
    purpose = meta.get("purpose")
    return purpose if isinstance(purpose, str) and purpose else None


def show_managed_workstreams(workstreams: list[dict[str, Any]]) -> None:
    if not workstreams:
        print()
        print("No active workstreams found.")
        return
    print()
    print("Active workstreams")
    print("------------------")
    for index, workstream in enumerate(workstreams, start=1):
        status = "dirty" if workstream.get("dirty") else "clean"
        purpose = workstream.get("purpose") or "No purpose saved"
        opened = workstream.get("last_opened_at") or workstream.get("created_at") or "-"
        branch = workstream.get("branch") or "-"
        name = workstream.get("name") or "-"
        tool = workstream.get("tool") or "-"
        print(f"[{index}] {name} | {tool} | {status}")
        print(f"     Purpose: {purpose}")
        print(f"     Last used: {opened}")
        print(f"     Branch: {branch}")


def select_managed_workstream() -> dict[str, Any] | None:
    workstreams = get_managed_workstreams()
    show_managed_workstreams(workstreams)
    if not workstreams:
        return None
    choice = input("Pick a workstream number: ").strip()
    if not choice:
        return None
    try:
        index = int(choice)
    except ValueError as exc:
        raise RuntimeError("Invalid workstream selection.") from exc
    if index < 1 or index > len(workstreams):
        raise RuntimeError("Workstream selection out of range.")
    return workstreams[index - 1]


def git_bash_path() -> str:
    candidates = [
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("PROGRAMW6432", ""), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Git", "bin", "bash.exe"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError("Git Bash not found. Install Git for Windows or adjust the launcher.")


def run_git_bash(command_text: str) -> int:
    return subprocess.call([git_bash_path(), "-lc", command_text])


def run_wsl(command_text: str) -> int:
    return subprocess.call(["wsl.exe", "bash", "-lc", command_text])


def open_claude_workstream(workstream_name: str, purpose: str | None) -> int:
    import shlex

    root = windows_to_git_bash(repo_root())
    saved_purpose = get_existing_purpose("claude", workstream_name)
    if saved_purpose:
        purpose = saved_purpose
    command_parts = [f"cd {shlex.quote(root)} &&"]
    if purpose:
        command_parts.append(f"CANOMPX3_WORKSTREAM_PURPOSE={shlex.quote(purpose)}")
    command_parts.append(f"exec ./scripts/infra/claude-worktree.sh open {shlex.quote(workstream_name)}")
    return run_git_bash(" ".join(command_parts))


def open_codex_workstream(workstream_name: str, purpose: str | None, search_mode: bool = False) -> int:
    import shlex

    root = windows_to_wsl(repo_root())
    saved_purpose = get_existing_purpose("codex", workstream_name)
    if saved_purpose:
        purpose = saved_purpose
        if saved_purpose == "Investigate / search":
            search_mode = True
    script_mode = "search" if search_mode else "open"
    command_parts = [f"cd {shlex.quote(root)} &&"]
    if purpose:
        command_parts.append(f"CANOMPX3_WORKSTREAM_PURPOSE={shlex.quote(purpose)}")
    command_parts.append(f"exec ./scripts/infra/codex-worktree.sh {script_mode} {shlex.quote(workstream_name)}")
    return run_wsl(" ".join(command_parts))


def prompt_workstream_name() -> str:
    value = input("Workstream name: ").strip()
    if not value:
        raise RuntimeError("Workstream name required.")
    return value


def select_workstream_purpose() -> dict[str, Any]:
    print()
    print("Pick the purpose")
    print("[1] Build / edit")
    print("[2] Investigate / search")
    print("[3] Review / verify")
    print()
    choice = input(">>> ").strip()
    if choice == "1":
        return {"label": "Build / edit", "recommended_tool": "codex", "search_mode": False}
    if choice == "2":
        return {
            "label": "Investigate / search",
            "recommended_tool": "codex",
            "search_mode": True,
        }
    if choice == "3":
        return {
            "label": "Review / verify",
            "recommended_tool": "claude",
            "search_mode": False,
        }
    raise RuntimeError("Invalid purpose selection.")


def select_agent_for_purpose(purpose: dict[str, Any]) -> str:
    print()
    print(f"Recommended agent: {purpose['recommended_tool']}")
    print("[1] Use recommended")
    print("[2] Claude")
    print("[3] Codex")
    print()
    choice = input(">>> ").strip()
    if choice in {"", "1"}:
        return str(purpose["recommended_tool"])
    if choice == "2":
        return "claude"
    if choice == "3":
        return "codex"
    raise RuntimeError("Invalid agent selection.")


def clear_screen() -> None:
    os.system("cls")


def run_menu() -> int:
    while True:
        clear_screen()
        print()
        print("============================================================")
        print(" AI WORKSTREAMS")
        print("============================================================")
        print()
        print("Purpose: run one problem per isolated workstream so Claude and Codex do not stomp on each other.")
        print()
        print("[1] Start new workstream")
        print("[2] Continue workstream")
        print("[3] Finish workstream")
        print("[4] Show active workstreams")
        print("[5] Clean stale workstream records")
        print("[Q] Quit")
        print()
        choice = input(">>> ").strip().lower()
        if choice == "1":
            name = prompt_workstream_name()
            purpose = select_workstream_purpose()
            agent = select_agent_for_purpose(purpose)
            if agent == "claude":
                return open_claude_workstream(name, str(purpose["label"]))
            return open_codex_workstream(
                name,
                str(purpose["label"]),
                bool(purpose["search_mode"]),
            )
        if choice == "2":
            workstream = select_managed_workstream()
            if workstream is None:
                input("Press Enter to continue")
                continue
            if workstream.get("tool") == "claude":
                return open_claude_workstream(
                    str(workstream.get("name", "")),
                    str(workstream.get("purpose") or ""),
                )
            if workstream.get("tool") == "codex":
                purpose = str(workstream.get("purpose") or "")
                return open_codex_workstream(
                    str(workstream.get("name", "")),
                    purpose,
                    purpose == "Investigate / search",
                )
            raise RuntimeError(f"Unsupported workstream tool: {workstream.get('tool')}")
        if choice == "3":
            workstream = select_managed_workstream()
            if workstream is None:
                input("Press Enter to continue")
                continue
            success, output = invoke_manager(
                [
                    "close",
                    "--tool",
                    str(workstream.get("tool")),
                    "--name",
                    str(workstream.get("name")),
                    "--force",
                    "--drop-branch",
                ]
            )
            if not success:
                raise RuntimeError(output)
            if output:
                print(output)
            input("Press Enter to continue")
            return 0
        if choice == "4":
            show_managed_workstreams(get_managed_workstreams())
            input("Press Enter to continue")
            continue
        if choice == "5":
            success, output = invoke_manager(["prune"])
            if not success:
                raise RuntimeError(output)
            if output:
                print(output)
            input("Press Enter to continue")
            return 0
        if choice == "q":
            return 0
        print("Invalid choice.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows AI workstream launcher")
    parser.add_argument("--mode", required=True, choices=sorted(VALID_MODES))
    parser.add_argument("--task", default="")
    parser.add_argument("--tool", default="")
    args = parser.parse_args()

    if args.mode == "menu":
        return run_menu()
    if args.mode == "claude":
        task = args.task or prompt_workstream_name()
        return open_claude_workstream(task, "Build / edit")
    if args.mode == "codex":
        task = args.task or prompt_workstream_name()
        return open_codex_workstream(task, "Build / edit", False)
    if args.mode == "codex-search":
        task = args.task or prompt_workstream_name()
        return open_codex_workstream(task, "Investigate / search", True)
    if args.mode == "list":
        show_managed_workstreams(get_managed_workstreams())
        return 0
    if args.mode == "resume":
        workstream = select_managed_workstream()
        if workstream is None:
            return 0
        if workstream.get("tool") == "claude":
            return open_claude_workstream(
                str(workstream.get("name", "")),
                str(workstream.get("purpose") or ""),
            )
        if workstream.get("tool") == "codex":
            purpose = str(workstream.get("purpose") or "")
            return open_codex_workstream(
                str(workstream.get("name", "")),
                purpose,
                purpose == "Investigate / search",
            )
        raise RuntimeError(f"Unsupported workstream tool: {workstream.get('tool')}")
    if args.mode == "close":
        if not args.task:
            raise RuntimeError("Workstream name required for close.")
        if not args.tool:
            raise RuntimeError("Tool required for close. Use claude or codex.")
        success, output = invoke_manager(
            [
                "close",
                "--tool",
                args.tool,
                "--name",
                args.task,
                "--force",
                "--drop-branch",
            ]
        )
        if not success:
            raise RuntimeError(output)
        if output:
            print(output)
        return 0
    if args.mode == "close-pick":
        workstream = select_managed_workstream()
        if workstream is None:
            return 0
        success, output = invoke_manager(
            [
                "close",
                "--tool",
                str(workstream.get("tool")),
                "--name",
                str(workstream.get("name")),
                "--force",
                "--drop-branch",
            ]
        )
        if not success:
            raise RuntimeError(output)
        if output:
            print(output)
        return 0
    if args.mode == "prune":
        success, output = invoke_manager(["prune"])
        if not success:
            raise RuntimeError(output)
        if output:
            print(output)
        return 0
    raise RuntimeError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AI Workstreams error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
