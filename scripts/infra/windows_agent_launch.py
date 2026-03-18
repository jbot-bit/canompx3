#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(force_terminal=True)

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

AGENT_STYLES = {
    "claude": "bold cyan",
    "codex": "bold green",
}


# ---------------------------------------------------------------------------
# Plumbing (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def relative_time(iso_str: str | None) -> str:
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        return f"{seconds // 86400}d ago"
    except (ValueError, TypeError):
        return "-"


def _supports_unicode() -> bool:
    try:
        "●".encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_BULLET = "●" if _supports_unicode() else "*"


def status_text(dirty: bool) -> Text:
    if dirty:
        return Text(f"{_BULLET} dirty", style="bold yellow")
    return Text(f"{_BULLET} clean", style="dim green")


def agent_text(tool: str) -> Text:
    style = AGENT_STYLES.get(tool, "bold white")
    return Text(tool, style=style)


def build_workstream_table(workstreams: list[dict[str, Any]], numbered: bool = True) -> Table | None:
    if not workstreams:
        return None
    table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    if numbered:
        table.add_column("#", style="bold white", width=3, justify="right")
    table.add_column("Name", style="bold white", min_width=16)
    table.add_column("Agent", min_width=8)
    table.add_column("Last Used", style="dim", min_width=8)
    table.add_column("Status", min_width=6)
    for i, ws in enumerate(workstreams, start=1):
        name = ws.get("name") or "-"
        tool = ws.get("tool") or "-"
        opened = ws.get("last_opened_at") or ws.get("created_at")
        dirty = bool(ws.get("dirty"))
        row: list[Any] = []
        if numbered:
            row.append(str(i))
        row.extend([name, agent_text(tool), relative_time(opened), status_text(dirty)])
        table.add_row(*row)
    return table


def clear_screen() -> None:
    subprocess.run(["cmd", "/c", "cls"], check=False)


def prompt(label: str, default: str = "") -> str:
    try:
        if default:
            console.print(f"  {label} [dim]\\[{default}][/]: ", end="")
        else:
            console.print(f"  {label}: ", end="")
        value = input().strip()
        return value or default
    except (EOFError, KeyboardInterrupt):
        return default


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------


def run_menu() -> int:
    while True:
        clear_screen()
        workstreams = get_managed_workstreams()
        most_recent = workstreams[0] if workstreams else None

        # Header
        console.print()
        console.print(
            Panel(
                Text("AI WORKSTREAMS", style="bold white", justify="center"),
                border_style="bright_blue",
                expand=True,
                padding=(0, 1),
            )
        )

        # Active workstreams
        if workstreams:
            table = build_workstream_table(workstreams)
            if table:
                console.print()
                console.print("  [bold bright_blue]Active[/]")
                console.print(table)
        else:
            console.print()
            console.print("  [dim]No active workstreams[/]")

        # Actions
        console.print()
        actions = Text()
        actions.append("  ")
        actions.append("[N]", style="bold white")
        actions.append(" New   ", style="dim")
        actions.append("[O]", style="bold white")
        actions.append(" Orient   ", style="dim")
        actions.append("[F]", style="bold white")
        actions.append(" Finish   ", style="dim")
        actions.append("[P]", style="bold white")
        actions.append(" Prune   ", style="dim")
        actions.append("[Q]", style="bold white")
        actions.append(" Quit", style="dim")
        console.print(actions)

        # Default action hint
        if most_recent:
            name = most_recent.get("name", "")
            tool = most_recent.get("tool", "")
            console.print(f"  [dim]Enter = resume [bold]{name}[/bold] ({tool})[/dim]")

        console.print()
        choice = prompt(">>>").lower()

        # Default: resume most recent
        if not choice and most_recent:
            return _launch_workstream(most_recent)

        # Resume by number
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(workstreams):
                return _launch_workstream(workstreams[idx - 1])
            console.print("  [red]Invalid number[/]")
            input("  Press Enter")
            continue

        if choice == "n":
            result = _new_workstream()
            if result == 0:
                return result  # Successfully launched agent
            continue  # Bad input — back to menu

        if choice == "o":
            _run_pulse()
            input("  Press Enter to continue")
            continue

        if choice == "f":
            _finish_workstream(workstreams)
            continue  # Back to menu after finish

        if choice == "p":
            success, output = invoke_manager(["prune"])
            if output:
                console.print(f"  {output}")
            input("  Press Enter")
            continue

        if choice == "q":
            return 0

        # Unrecognized input — redraw


def _launch_workstream(ws: dict[str, Any]) -> int:
    name = str(ws.get("name", ""))
    tool = str(ws.get("tool", ""))
    purpose = str(ws.get("purpose") or "")
    console.print(f"  [dim]Opening[/] [bold]{name}[/] [dim]with[/] [{AGENT_STYLES.get(tool, '')}]{tool}[/]...")
    if tool == "claude":
        return open_claude_workstream(name, purpose)
    if tool == "codex":
        return open_codex_workstream(name, purpose, purpose == "Investigate / search")
    console.print(f"  [red]Unknown agent: {tool}[/]")
    return 1


def _new_workstream() -> int:
    console.print()
    name = prompt("Name")
    if not name:
        console.print("  [red]Name required[/]")
        return 1

    console.print()
    console.print("  [bold bright_blue]Agent[/]")
    console.print("  [bold cyan]1[/] Claude       [dim]review, verify, complex reasoning[/]")
    console.print("  [bold green]2[/] Codex        [dim]build, edit, implement[/]")
    console.print("  [bold green]3[/] Codex search [dim]investigate, research with web[/]")
    console.print()
    agent_choice = prompt("Agent", "1")

    if agent_choice in {"1", "claude", "c"}:
        console.print(f"  [dim]Opening[/] [bold]{name}[/] [dim]with[/] [bold cyan]claude[/]...")
        return open_claude_workstream(name, "Build / edit")
    if agent_choice in {"2", "codex", "x"}:
        console.print(f"  [dim]Opening[/] [bold]{name}[/] [dim]with[/] [bold green]codex[/]...")
        return open_codex_workstream(name, "Build / edit", False)
    if agent_choice in {"3", "search", "s"}:
        console.print(f"  [dim]Opening[/] [bold]{name}[/] [dim]with[/] [bold green]codex search[/]...")
        return open_codex_workstream(name, "Investigate / search", True)

    console.print("  [red]Invalid agent choice[/]")
    return 1


def _finish_workstream(workstreams: list[dict[str, Any]]) -> None:
    if not workstreams:
        console.print("  [dim]Nothing to finish[/]")
        input("  Press Enter")
        return

    console.print()
    console.print("  [bold bright_blue]Finish which?[/]")
    table = build_workstream_table(workstreams)
    if table:
        console.print(table)
    console.print()
    choice = prompt("#")
    if not choice or not choice.isdigit():
        return
    idx = int(choice)
    if idx < 1 or idx > len(workstreams):
        console.print("  [red]Invalid number[/]")
        input("  Press Enter")
        return
    ws = workstreams[idx - 1]
    name = str(ws.get("name", ""))
    tool = str(ws.get("tool", ""))
    success, output = invoke_manager(
        [
            "close",
            "--tool",
            tool,
            "--name",
            name,
            "--force",
            "--drop-branch",
        ]
    )
    if success:
        console.print(f"  [green]Closed[/] [bold]{name}[/]")
    else:
        console.print(f"  [red]Failed:[/] {output}")
    input("  Press Enter")


def _run_pulse() -> None:
    pulse_script = repo_root() / "scripts" / "tools" / "project_pulse.py"
    if pulse_script.exists():
        subprocess.run(pick_python() + [str(pulse_script), "--fast"], check=False)
    else:
        console.print("  [red]project_pulse.py not found[/]")


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows AI workstream launcher")
    parser.add_argument("--mode", required=True, choices=sorted(VALID_MODES))
    parser.add_argument("--task", default="")
    parser.add_argument("--tool", default="")
    args = parser.parse_args()

    if args.mode == "menu":
        return run_menu()
    if args.mode == "claude":
        task = args.task or prompt("Workstream name")
        if not task:
            return 1
        return open_claude_workstream(task, "Build / edit")
    if args.mode == "codex":
        task = args.task or prompt("Workstream name")
        if not task:
            return 1
        return open_codex_workstream(task, "Build / edit", False)
    if args.mode == "codex-search":
        task = args.task or prompt("Workstream name")
        if not task:
            return 1
        return open_codex_workstream(task, "Investigate / search", True)
    if args.mode == "list":
        workstreams = get_managed_workstreams()
        table = build_workstream_table(workstreams, numbered=False)
        if table:
            console.print(table)
        else:
            console.print("[dim]No active workstreams[/]")
        return 0
    if args.mode == "resume":
        workstreams = get_managed_workstreams()
        if not workstreams:
            console.print("[dim]No active workstreams[/]")
            return 0
        if len(workstreams) == 1:
            return _launch_workstream(workstreams[0])
        table = build_workstream_table(workstreams)
        if table:
            console.print(table)
        choice = prompt("#")
        if not choice or not choice.isdigit():
            return 0
        idx = int(choice)
        if idx < 1 or idx > len(workstreams):
            return 0
        return _launch_workstream(workstreams[idx - 1])
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
            console.print(output)
        return 0
    if args.mode == "close-pick":
        workstreams = get_managed_workstreams()
        if not workstreams:
            console.print("[dim]No active workstreams[/]")
            return 0
        table = build_workstream_table(workstreams)
        if table:
            console.print(table)
        choice = prompt("#")
        if not choice or not choice.isdigit():
            return 0
        idx = int(choice)
        if idx < 1 or idx > len(workstreams):
            return 0
        ws = workstreams[idx - 1]
        success, output = invoke_manager(
            [
                "close",
                "--tool",
                str(ws.get("tool")),
                "--name",
                str(ws.get("name")),
                "--force",
                "--drop-branch",
            ]
        )
        if not success:
            raise RuntimeError(output)
        if output:
            console.print(output)
        return 0
    if args.mode == "prune":
        success, output = invoke_manager(["prune"])
        if not success:
            raise RuntimeError(output)
        if output:
            console.print(output)
        return 0
    raise RuntimeError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        console.print(f"[red]AI Workstreams error:[/] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
