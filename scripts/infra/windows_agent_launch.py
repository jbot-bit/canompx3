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

import readchar
from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(force_terminal=True)

VALID_MODES = {
    "claude",
    "codex",
    "codex-search",
    "codex-project",
    "codex-project-gold-db",
    "codex-project-search-gold-db",
    "codex-project-smart",
    "codex-project-smart-gold-db",
    "codex-project-smart-search-gold-db",
    "codex-project-smart-power",
    "codex-project-power",
    "codex-project-linux",
    "codex-project-linux-gold-db",
    "codex-project-linux-search-gold-db",
    "codex-project-linux-power",
    "green-codex",
    "green-claude",
    "handoff",
    "list",
    "close",
    "close-pick",
    "resume",
    "ship",
    "menu",
    "prune",
    "doctor",
    "cleanup",
}

AGENT_STYLES = {
    "claude": "bold cyan",
    "codex": "bold green",
}

UP = readchar.key.UP
DOWN = readchar.key.DOWN
ENTER = readchar.key.ENTER


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def manager_py() -> Path:
    return repo_root() / "scripts" / "tools" / "worktree_manager.py"


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


def prepare_task_route_packet(
    root: Path,
    *,
    tool: str,
    task_text: str | None = None,
    briefing_level: str = "mutating",
    clear: bool = False,
) -> None:
    packet_writer = root / "scripts" / "tools" / "task_route_packet.py"
    if not packet_writer.exists():
        return
    command = pick_python() + [str(packet_writer), "--root", str(root)]
    if clear or not task_text:
        command.append("--clear")
    else:
        command.extend(["--tool", tool, "--task", task_text, "--briefing-level", briefing_level])
    subprocess.run(command, cwd=root, check=False)


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


def get_workstream_metadata(tool_name: str, workstream_name: str) -> dict[str, Any] | None:
    success, output = invoke_manager(["show", "--tool", tool_name, "--name", workstream_name])
    if not success or not output or output == "{}":
        return None
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def classify_source_target_relation(
    source_branch: str,
    source_head: str,
    target_branch: str,
    target_head: str,
    *,
    source_contains_target: bool | None = None,
    target_contains_source: bool | None = None,
) -> str:
    """Classify the sync relation between source and target repos.

    The smart launcher treats the Windows checkout as the source of truth and
    the WSL-home clone as the target that may be fast-forwarded from it.
    """

    if source_branch != target_branch:
        return "branch_mismatch"
    if source_branch in {"HEAD", "<unknown>"} or target_branch in {"HEAD", "<unknown>"}:
        return "detached_head"
    if source_head in {"", "<unknown>"} or target_head in {"", "<unknown>"}:
        return "unknown"
    if source_head == target_head:
        return "aligned"
    if source_contains_target is True:
        return "target_behind_source"
    if target_contains_source is True:
        return "source_behind_target"
    if source_contains_target is False and target_contains_source is False:
        return "diverged"
    return "unknown"


def run_wsl(command_text: str) -> int:
    import tempfile

    # Multi-line bash scripts passed via `wsl.exe bash -lc <text>` lose
    # variable assignments at the Win32→WSL argv boundary (lines execute,
    # `set -x` echoes them, but assignments don't stick). Write the script
    # to a temp file inside the repo and run it as a real file via its
    # /mnt/c WSL path. LF newlines preserved explicitly.
    scratch_dir = repo_root() / ".claude" / "scratch"
    tmp_dir = str(scratch_dir) if scratch_dir.exists() else str(repo_root())
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".sh",
        prefix="codex_launch_",
        dir=tmp_dir,
        delete=False,
        newline="\n",
        encoding="utf-8",
    ) as tmp:
        tmp.write(command_text)
        tmp_name = tmp.name
    try:
        wsl_path = windows_to_wsl(Path(tmp_name))
        return subprocess.call(
            [
                "wsl.exe",
                "bash",
                "-lc",
                f"status=0; bash {wsl_path} || status=$?; rm -f {wsl_path}; exit $status",
            ]
        )
    finally:
        # Best-effort cleanup if WSL never ran (e.g., wsl.exe missing)
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass


def wsl_home_clone_available() -> bool:
    # Multi-line scripts via `wsl.exe bash -lc <text>` lose statements at the
    # Win32->WSL argv boundary (same class as the bug documented in run_wsl).
    # Write the probe to a tempfile and execute it as a real script file.
    import tempfile

    root_probe = "\n".join(
        [
            "set -euo pipefail",
            'ROOT_INPUT="${CANOMPX3_CODEX_WSL_ROOT:-}"',
            'if [[ -z "$ROOT_INPUT" || "$ROOT_INPUT" == "." || "$ROOT_INPUT" == "./" ]]; then',
            '  ROOT="$HOME/canompx3"',
            'elif [[ "$ROOT_INPUT" == "~" ]]; then',
            '  ROOT="$HOME"',
            'elif [[ "$ROOT_INPUT" == "~/"* ]]; then',
            '  ROOT="$HOME/${ROOT_INPUT#~/}"',
            'elif [[ "$ROOT_INPUT" == /* ]]; then',
            '  ROOT="$ROOT_INPUT"',
            "else",
            "  exit 1",
            "fi",
            '[[ -d "$ROOT/.git" ]]',
        ]
    )

    scratch_dir = repo_root() / ".claude" / "scratch"
    tmp_dir = str(scratch_dir) if scratch_dir.exists() else str(repo_root())
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            prefix="codex_probe_",
            dir=tmp_dir,
            delete=False,
            newline="\n",
            encoding="utf-8",
        ) as tmp:
            tmp.write(root_probe)
            tmp_name = tmp.name
        wsl_path = windows_to_wsl(Path(tmp_name))
        result = subprocess.run(
            ["wsl.exe", "bash", wsl_path],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    except OSError:
        return False
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
    return result.returncode == 0


def build_codex_wsl_command(
    root_wsl: str,
    workstream_name: str,
    purpose: str | None,
    search_mode: bool,
) -> str:
    import shlex

    script_mode = "search" if search_mode else "open"
    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(root_wsl)}",
        f"python3 scripts/tools/wsl_mount_guard.py --root {shlex.quote(root_wsl)}",
        "export JOBLIB_MULTIPROCESSING=0",
        "if ! command -v uv >/dev/null 2>&1; then",
        "  echo 'ERROR: uv is not installed in WSL PATH.' >&2",
        "  exit 1",
        "fi",
        "export UV_PROJECT_ENVIRONMENT=.venv-wsl",
        "export UV_CACHE_DIR=/tmp/uv-cache",
        "export UV_PYTHON_INSTALL_DIR=/tmp/uv-python",
        "export UV_LINK_MODE=copy",
        'mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"',
        "uv sync --frozen --python 3.13 --group dev",
    ]
    if purpose:
        lines.append(f"export CANOMPX3_WORKSTREAM_PURPOSE={shlex.quote(purpose)}")
    lines.append(
        f"exec ./scripts/infra/codex-worktree.sh {script_mode} {shlex.quote(workstream_name)} -- --no-alt-screen"
    )
    return "\n".join(lines)


def build_codex_project_wsl_command(
    root_wsl: str,
    search_mode: bool = False,
    use_linux_home: bool = False,
    profile: str | None = None,
    sync_windows_checkout: bool = False,
    enable_gold_db: bool = False,
) -> str:
    import shlex

    if enable_gold_db and search_mode:
        script_name = "codex-project-search-gold-db.sh"
    elif enable_gold_db:
        script_name = "codex-project-gold-db.sh"
    elif search_mode:
        script_name = "codex-project-search.sh"
    else:
        script_name = "codex-project.sh"

    lines = ["set -euo pipefail"]
    if profile:
        lines.append(f"export CANOMPX3_CODEX_PROFILE={shlex.quote(profile)}")

    if use_linux_home:
        lines.extend(
            [
                'ROOT_INPUT="${CANOMPX3_CODEX_WSL_ROOT:-}"',
                'if [[ -z "$ROOT_INPUT" || "$ROOT_INPUT" == "." || "$ROOT_INPUT" == "./" ]]; then',
                '  ROOT="$HOME/canompx3"',
                'elif [[ "$ROOT_INPUT" == "~" ]]; then',
                '  ROOT="$HOME"',
                'elif [[ "$ROOT_INPUT" == "~/"* ]]; then',
                '  ROOT="$HOME/${ROOT_INPUT#~/}"',
                'elif [[ "$ROOT_INPUT" == /* ]]; then',
                '  ROOT="$ROOT_INPUT"',
                "else",
                '  echo "ERROR: CANOMPX3_CODEX_WSL_ROOT must be an absolute WSL path, not: $ROOT_INPUT" >&2',
                '  echo "Unset it to use ~/canompx3, or set it to a path like /home/joshd/canompx3." >&2',
                "  exit 1",
                "fi",
                'if [[ ! -d "$ROOT" ]]; then',
                '  echo "ERROR: WSL Codex repo not found at $ROOT." >&2',
                '  echo "Set CANOMPX3_CODEX_WSL_ROOT or clone canompx3 into ~/canompx3." >&2',
                "  exit 1",
                "fi",
            ]
        )
        if sync_windows_checkout:
            lines.append(
                f'bash {shlex.quote(root_wsl)}/scripts/infra/codex-wsl-sync.sh --source {shlex.quote(root_wsl)} --target "$ROOT"'
            )
        lines.extend(
            [
                'cd "$ROOT"',
                # Bootstrap .venv-wsl on first launch. codex-project.sh exits
                # immediately if .venv-wsl/bin/python is missing, which would
                # otherwise leave a fresh WSL-home clone unusable on first run.
                'if [[ ! -x ".venv-wsl/bin/python" ]]; then',
                "  if ! command -v uv >/dev/null 2>&1; then",
                "    echo 'ERROR: uv is not installed in WSL PATH.' >&2",
                "    exit 1",
                "  fi",
                "  export UV_PROJECT_ENVIRONMENT=.venv-wsl",
                "  export UV_CACHE_DIR=/tmp/uv-cache",
                "  export UV_PYTHON_INSTALL_DIR=/tmp/uv-python",
                "  export UV_LINK_MODE=copy",
                '  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"',
                "  uv sync --frozen --python 3.13 --group dev",
                "fi",
                f"exec bash ./scripts/infra/{script_name} --no-alt-screen",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"cd {shlex.quote(root_wsl)}",
            f"exec ./scripts/infra/{script_name} --no-alt-screen",
        ]
    )
    return "\n".join(lines)


def ensure_managed_worktree(tool_name: str, workstream_name: str, purpose: str | None) -> tuple[Path, str | None]:
    saved_purpose = get_existing_purpose(tool_name, workstream_name)
    final_purpose = saved_purpose or purpose
    arguments = ["create", "--tool", tool_name, "--name", workstream_name, "--json"]
    if final_purpose:
        arguments.extend(["--purpose", final_purpose])
    success, output = invoke_manager(arguments)
    if not success:
        raise RuntimeError(output or f"Unable to open {tool_name} workstream {workstream_name}.")
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid worktree manager output: {output}") from exc
    path_value = payload.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"Worktree manager did not return a path: {output}")
    return Path(path_value), final_purpose


def run_preflight(worktree_path: Path, claim_tool: str, context: str = "generic", mode: str = "mutating") -> None:
    preflight = worktree_path / "scripts" / "tools" / "session_preflight.py"
    if not preflight.exists():
        return
    env = os.environ.copy()
    env.setdefault("CANOMPX3_SESSION_OWNER", f"launcher-{os.getpid()}")
    result = subprocess.run(
        pick_python() + [str(preflight), "--quiet", "--context", context, "--claim", claim_tool, "--mode", mode],
        cwd=worktree_path,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Session preflight blocked launch for {claim_tool} ({mode})")


def find_claude_cli() -> str:
    for candidate in ("claude", "claude.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Claude CLI not found on PATH. Expected `claude` or `claude.exe`.")


def open_claude_workstream(workstream_name: str, purpose: str | None) -> int:
    worktree_path, _ = ensure_managed_worktree("claude", workstream_name, purpose)
    run_preflight(worktree_path, claim_tool="claude", context="generic", mode="mutating")
    task_text = f"{purpose}: {workstream_name}" if purpose else workstream_name
    prepare_task_route_packet(worktree_path, tool="claude", task_text=task_text, briefing_level="mutating")
    return subprocess.call([find_claude_cli(), "-C", str(worktree_path)])


def open_codex_workstream(workstream_name: str, purpose: str | None, search_mode: bool = False) -> int:
    root = windows_to_wsl(repo_root())
    saved_purpose = get_existing_purpose("codex", workstream_name)
    if saved_purpose:
        purpose = saved_purpose
        if saved_purpose == "Investigate / search":
            search_mode = True
    return run_wsl(build_codex_wsl_command(root, workstream_name, purpose, search_mode))


def open_codex_project(search_mode: bool = False, enable_gold_db: bool = False) -> int:
    root = windows_to_wsl(repo_root())
    return run_wsl(build_codex_project_wsl_command(root, search_mode=search_mode, enable_gold_db=enable_gold_db))


def open_codex_project_power() -> int:
    root = windows_to_wsl(repo_root())
    return run_wsl(build_codex_project_wsl_command(root, profile="canompx3_power"))


def open_codex_project_linux_home(search_mode: bool = False, enable_gold_db: bool = False) -> int:
    root = windows_to_wsl(repo_root())
    return run_wsl(
        build_codex_project_wsl_command(
            root,
            search_mode=search_mode,
            use_linux_home=True,
            enable_gold_db=enable_gold_db,
        )
    )


def open_codex_project_linux_home_power() -> int:
    root = windows_to_wsl(repo_root())
    return run_wsl(
        build_codex_project_wsl_command(
            root,
            use_linux_home=True,
            profile="canompx3_power",
        )
    )


def open_codex_project_linux_home_synced(search_mode: bool = False, enable_gold_db: bool = False) -> int:
    root = windows_to_wsl(repo_root())
    return run_wsl(
        build_codex_project_wsl_command(
            root,
            search_mode=search_mode,
            use_linux_home=True,
            sync_windows_checkout=True,
            enable_gold_db=enable_gold_db,
        )
    )


def open_codex_project_linux_home_power_synced() -> int:
    root = windows_to_wsl(repo_root())
    return run_wsl(
        build_codex_project_wsl_command(
            root,
            use_linux_home=True,
            profile="canompx3_power",
            sync_windows_checkout=True,
        )
    )


def _fallback_notice() -> None:
    print("INFO: WSL-home Codex repo unavailable; falling back to the current Windows checkout.")


def open_codex_project_smart(search_mode: bool = False, enable_gold_db: bool = False) -> int:
    if wsl_home_clone_available():
        return open_codex_project_linux_home_synced(search_mode=search_mode, enable_gold_db=enable_gold_db)
    _fallback_notice()
    return open_codex_project(search_mode=search_mode, enable_gold_db=enable_gold_db)


def open_codex_project_smart_power() -> int:
    if wsl_home_clone_available():
        return open_codex_project_linux_home_power_synced()
    _fallback_notice()
    return open_codex_project_power()


def cleanup_codex_wsl_clone() -> int:
    root = windows_to_wsl(repo_root())
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S")
    stash_name = f"wsl-codex-dirty-before-launch-{stamp}"
    command = "\n".join(
        [
            "set -euo pipefail",
            'ROOT_INPUT="${CANOMPX3_CODEX_WSL_ROOT:-}"',
            'if [[ -z "$ROOT_INPUT" || "$ROOT_INPUT" == "." || "$ROOT_INPUT" == "./" ]]; then',
            '  ROOT="$HOME/canompx3"',
            'elif [[ "$ROOT_INPUT" == "~" ]]; then',
            '  ROOT="$HOME"',
            'elif [[ "$ROOT_INPUT" == "~/"* ]]; then',
            '  ROOT="$HOME/${ROOT_INPUT#~/}"',
            'elif [[ "$ROOT_INPUT" == /* ]]; then',
            '  ROOT="$ROOT_INPUT"',
            "else",
            '  echo "ERROR: CANOMPX3_CODEX_WSL_ROOT must be empty, ~, ~/..., or absolute (/...)." >&2',
            "  exit 2",
            "fi",
            'if [[ ! -d "$ROOT/.git" ]]; then',
            '  echo "ERROR: WSL Codex repo not found: $ROOT" >&2',
            "  exit 1",
            "fi",
            'cd "$ROOT"',
            'echo "WSL Codex repo: $ROOT"',
            "git status --short --branch",
            'if [[ -z "$(git status --short)" ]]; then',
            '  echo "WSL Codex repo is already clean."',
            "  exit 0",
            "fi",
            'echo "Stashing WSL Codex repo changes so codex.bat can launch safely."',
            f'git stash push -u -m "{stash_name}"',
            'echo "Stash saved. Relaunch from PowerShell with: .\\codex.bat"',
            "echo \"To inspect later: wsl -e bash -lc 'cd ~/canompx3 && git stash list'\"",
        ]
    )
    console.print(f"[bold]Cleaning WSL Codex clone via {root} front door[/]")
    return run_wsl(command)


def _green_baseline_worktree() -> Path:
    worktree_path = repo_root() / ".worktrees" / "tasks" / "green-baseline"
    if not worktree_path.exists():
        raise RuntimeError(f"Clean green-baseline worktree not found: {worktree_path}")
    return worktree_path


def open_codex_green_baseline() -> int:
    worktree_path = _green_baseline_worktree()
    return run_wsl(build_codex_project_wsl_command(windows_to_wsl(worktree_path)))


def open_claude_green_baseline() -> int:
    worktree_path = _green_baseline_worktree()
    prepare_task_route_packet(worktree_path, tool="claude", clear=True)
    return subprocess.call([find_claude_cli(), "-C", str(worktree_path)])


def handoff_workstream(
    name: str,
    current_tool: str,
    target_tool: str,
    purpose: str | None,
    note: str | None = None,
) -> tuple[bool, str]:
    arguments = [
        "handoff",
        "--name",
        name,
        "--tool",
        current_tool,
        "--target-tool",
        target_tool,
    ]
    if purpose:
        arguments.extend(["--purpose", purpose])
    if note is not None:
        arguments.extend(["--note", note])
    return invoke_manager(arguments)


def ship_workstream(name: str, tool: str, commit_message: str | None = None) -> tuple[bool, str]:
    arguments = ["ship", "--name", name, "--tool", tool]
    if commit_message:
        arguments.extend(["--commit-message", commit_message])
    return invoke_manager(arguments)


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


_UNI = _supports_unicode()
_BULLET = "●" if _UNI else "*"
_ARROW = "▸" if _UNI else ">"


def status_text(dirty: bool) -> Text:
    if dirty:
        return Text(f"{_BULLET} dirty", style="bold yellow")
    return Text(f"{_BULLET} clean", style="dim green")


def agent_text(tool: str) -> Text:
    style = AGENT_STYLES.get(tool, "bold white")
    return Text(tool, style=style)


def state_text(state: str | None) -> Text:
    normalized = (state or "active").strip().lower() or "active"
    if normalized == "handoff":
        return Text("handoff", style="bold magenta")
    return Text(normalized, style="dim cyan")


def build_workstream_table(
    workstreams: list[dict[str, Any]],
    cursor: int = -1,
) -> Table | None:
    if not workstreams:
        return None
    table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    table.add_column("Sel", width=2)
    table.add_column("Name", style="bold white", min_width=16, ratio=1)
    table.add_column("Owner", min_width=8)
    table.add_column("State", min_width=8)
    table.add_column("Last Used", min_width=8)
    table.add_column("Git", min_width=8)
    for i, ws in enumerate(workstreams):
        name = ws.get("name") or "-"
        tool = ws.get("tool") or "-"
        state = str(ws.get("state") or "active")
        opened = ws.get("last_opened_at") or ws.get("created_at")
        dirty = bool(ws.get("dirty"))
        selected = i == cursor
        sel = Text(f" {_ARROW}", style="bold bright_blue") if selected else Text("  ")
        row_style = "on grey15" if selected else ""
        table.add_row(
            sel,
            Text(name, style=f"bold white {row_style}"),
            Text(tool, style=f"{AGENT_STYLES.get(tool, '')} {row_style}"),
            state_text(state),
            Text(relative_time(opened), style=f"dim {row_style}"),
            status_text(dirty),
            style=row_style,
        )
    return table


def build_footer() -> Text:
    keys = Text()
    keys.append(" ↑↓ " if _UNI else " Up/Dn ", style="bold white on grey30")
    keys.append(" Navigate ", style="dim")
    keys.append(" Enter " if _UNI else " Enter ", style="bold white on grey30")
    keys.append(" Launch ", style="dim")
    keys.append(" N ", style="bold white on grey30")
    keys.append(" New ", style="dim")
    keys.append(" H ", style="bold white on grey30")
    keys.append(" Handoff ", style="dim")
    keys.append(" S ", style="bold white on grey30")
    keys.append(" Ship ", style="dim")
    keys.append(" O ", style="bold white on grey30")
    keys.append(" Orient ", style="dim")
    keys.append(" F ", style="bold white on grey30")
    keys.append(" Drop ", style="dim")
    keys.append(" P ", style="bold white on grey30")
    keys.append(" Prune ", style="dim")
    keys.append(" Q ", style="bold white on grey30")
    keys.append(" Quit ", style="dim")
    return keys


def build_header() -> Panel:
    title = Text()
    title.append(" AI WORKSTREAMS ", style="bold white")
    title.append("  ", style="dim")
    title.append("canompx3", style="dim cyan")
    return Panel(title, border_style="bright_blue", expand=True, padding=(0, 1))


class MenuRenderable:
    """Full-screen menu layout rendered on each keypress."""

    def __init__(self, workstreams: list[dict[str, Any]], cursor: int, message: str = "") -> None:
        self.workstreams = workstreams
        self.cursor = cursor
        self.message = message

    def __rich_console__(self, rconsole: Console, options: ConsoleOptions) -> RenderResult:
        yield Text()
        yield build_header()
        yield Text()

        if self.workstreams:
            yield Text("  Active", style="bold bright_blue")
            yield Text()
            table = build_workstream_table(self.workstreams, cursor=self.cursor)
            if table:
                yield table
        else:
            yield Text()
            yield Text("  No active workstreams — press N to start one", style="dim")
            yield Text()

        yield Text()
        yield build_footer()

        if self.message:
            yield Text()
            yield Text(f"  {self.message}", style="dim yellow")

        if self.workstreams and 0 <= self.cursor < len(self.workstreams):
            selected = self.workstreams[self.cursor]
            note = str(selected.get("handoff_note") or "").strip()
            if note:
                yield Text()
                yield Text(f"  Note: {note}", style="dim")


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


def wait_for_key(label: str = "Press Enter") -> None:
    try:
        input(f"  {label}")
    except (EOFError, KeyboardInterrupt):
        pass


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------


def run_menu() -> int:
    cursor = 0
    message = ""

    while True:
        workstreams = get_managed_workstreams()
        if cursor >= len(workstreams):
            cursor = max(0, len(workstreams) - 1)

        renderable = MenuRenderable(workstreams, cursor, message)
        message = ""

        # Render to alternate screen
        subprocess.run(["cmd", "/c", "cls"], check=False)
        console.print(renderable)

        try:
            key = readchar.readkey()
        except (EOFError, KeyboardInterrupt):
            return 0

        if key == UP:
            cursor = max(0, cursor - 1)
        elif key == DOWN:
            cursor = min(max(0, len(workstreams) - 1), cursor + 1)
        elif key == ENTER:
            if workstreams:
                return _launch_workstream(workstreams[cursor])
        elif key.lower() == "n":
            result = _new_workstream()
            if result == 0:
                return result
        elif key.lower() == "h":
            if workstreams:
                result = _handoff_workstream(workstreams, cursor)
                if result is not None:
                    return result
            else:
                message = "Nothing to hand off"
        elif key.lower() == "s":
            if workstreams:
                _ship_workstream(workstreams, cursor)
            else:
                message = "Nothing to ship"
        elif key.lower() == "o":
            _run_pulse()
            wait_for_key("Press Enter to continue")
        elif key.lower() == "f":
            if workstreams:
                _finish_workstream(workstreams, cursor)
            else:
                message = "Nothing to finish"
        elif key.lower() == "p":
            success, output = invoke_manager(["prune"])
            message = output if output else "Pruned"
        elif key.lower() == "q":
            return 0


def _launch_workstream(ws: dict[str, Any]) -> int:
    name = str(ws.get("name", ""))
    tool = str(ws.get("tool", ""))
    purpose = str(ws.get("purpose") or "")
    console.print(f"\n  [dim]Opening[/] [bold]{name}[/] [dim]with[/] [{AGENT_STYLES.get(tool, '')}]{tool}[/]...\n")
    if tool == "claude":
        return open_claude_workstream(name, purpose)
    if tool == "codex":
        return open_codex_workstream(name, purpose, purpose == "Investigate / search")
    console.print(f"  [red]Unknown agent: {tool}[/]")
    return 1


def _new_workstream() -> int:
    subprocess.run(["cmd", "/c", "cls"], check=False)
    console.print()
    console.print(
        Panel(Text("NEW WORKSTREAM", style="bold white", justify="center"), border_style="bright_blue", expand=True)
    )
    console.print()
    name = prompt("Name")
    if not name:
        return 1

    console.print()
    console.print("  [bold bright_blue]Agent[/]")
    console.print()
    console.print("  [bold cyan]1[/]  Claude        [dim]review, verify, complex reasoning[/]")
    console.print("  [bold green]2[/]  Codex         [dim]build, edit, implement[/]")
    console.print("  [bold green]3[/]  Codex search  [dim]investigate, research with web[/]")
    console.print()
    agent_choice = prompt("Agent", "1")

    if agent_choice in {"1", "claude", "c"}:
        console.print(f"\n  [dim]Opening[/] [bold]{name}[/] [dim]with[/] [bold cyan]claude[/]...\n")
        return open_claude_workstream(name, "Build / edit")
    if agent_choice in {"2", "codex", "x"}:
        console.print(f"\n  [dim]Opening[/] [bold]{name}[/] [dim]with[/] [bold green]codex[/]...\n")
        return open_codex_workstream(name, "Build / edit", False)
    if agent_choice in {"3", "search", "s"}:
        console.print(f"\n  [dim]Opening[/] [bold]{name}[/] [dim]with[/] [bold green]codex search[/]...\n")
        return open_codex_workstream(name, "Investigate / search", True)

    return 1


def _finish_workstream(workstreams: list[dict[str, Any]], cursor: int = 0) -> None:
    if not workstreams:
        return
    ws = workstreams[cursor]
    name = str(ws.get("name", ""))
    tool = str(ws.get("tool", ""))

    subprocess.run(["cmd", "/c", "cls"], check=False)
    console.print()
    console.print(f"  Drop [bold]{name}[/] ({tool}) without merging? [dim]y/n[/]")
    try:
        key = readchar.readkey()
    except (EOFError, KeyboardInterrupt):
        return
    if key.lower() != "y":
        return

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
        console.print(f"  [green]Dropped[/] [bold]{name}[/]")
    else:
        console.print(f"  [red]Failed:[/] {output}")
    wait_for_key()


def _handoff_workstream(workstreams: list[dict[str, Any]], cursor: int = 0) -> int | None:
    ws = workstreams[cursor]
    name = str(ws.get("name", ""))
    current_tool = str(ws.get("tool", ""))
    current_purpose = str(ws.get("purpose") or "")

    subprocess.run(["cmd", "/c", "cls"], check=False)
    console.print()
    console.print(f"  Handoff [bold]{name}[/] from [bold]{current_tool}[/]")
    console.print()
    console.print("  [bold cyan]1[/]  Claude")
    console.print("  [bold green]2[/]  Codex")
    console.print("  [bold green]3[/]  Codex search")
    console.print()

    default_choice = "2" if current_tool == "claude" else "1"
    choice = prompt("Target", default_choice)
    note = prompt("Note", "")

    target_tool = ""
    purpose = current_purpose or "Build / edit"
    launch_search = False

    if choice in {"1", "claude", "c"}:
        target_tool = "claude"
        if current_purpose == "Investigate / search":
            purpose = "Review / verify"
        else:
            purpose = current_purpose or "Review / verify"
    elif choice in {"2", "codex", "x"}:
        target_tool = "codex"
        purpose = "Build / edit" if current_purpose == "Investigate / search" else (current_purpose or "Build / edit")
    elif choice in {"3", "search", "s"}:
        target_tool = "codex"
        purpose = "Investigate / search"
        launch_search = True
    else:
        return None

    success, output = handoff_workstream(
        name, current_tool=current_tool, target_tool=target_tool, purpose=purpose, note=note
    )
    if not success:
        console.print(f"  [red]Failed:[/] {output}")
        wait_for_key()
        return None

    console.print(f"  [green]Handoff set[/] [bold]{name}[/] -> [bold]{target_tool}[/]")
    if target_tool == "claude":
        return open_claude_workstream(name, purpose)
    return open_codex_workstream(name, purpose, search_mode=launch_search)


def _ship_workstream(workstreams: list[dict[str, Any]], cursor: int = 0) -> None:
    ws = workstreams[cursor]
    name = str(ws.get("name", ""))
    tool = str(ws.get("tool", ""))
    dirty = bool(ws.get("dirty"))

    subprocess.run(["cmd", "/c", "cls"], check=False)
    console.print()
    console.print(f"  Ship [bold]{name}[/] into [bold]main[/] and close the worktree? [dim]y/n[/]")
    try:
        key = readchar.readkey()
    except (EOFError, KeyboardInterrupt):
        return
    if key.lower() != "y":
        return

    commit_message = None
    if dirty:
        commit_message = prompt("Commit message", f"workstream: {name}")
        if not commit_message:
            return

    success, output = ship_workstream(name, tool=tool, commit_message=commit_message)
    if success:
        console.print(f"  [green]Shipped[/] [bold]{name}[/] into [bold]main[/]")
    else:
        console.print(f"  [red]Failed:[/] {output}")
    wait_for_key()


def _run_pulse() -> None:
    subprocess.run(["cmd", "/c", "cls"], check=False)
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
    if args.mode == "codex-project":
        return open_codex_project()
    if args.mode == "codex-project-gold-db":
        return open_codex_project(enable_gold_db=True)
    if args.mode == "codex-project-search-gold-db":
        return open_codex_project(search_mode=True, enable_gold_db=True)
    if args.mode == "codex-project-smart":
        return open_codex_project_smart()
    if args.mode == "codex-project-smart-gold-db":
        return open_codex_project_smart(enable_gold_db=True)
    if args.mode == "codex-project-smart-search-gold-db":
        return open_codex_project_smart(search_mode=True, enable_gold_db=True)
    if args.mode == "codex-project-smart-power":
        return open_codex_project_smart_power()
    if args.mode == "codex-project-power":
        return open_codex_project_power()
    if args.mode == "codex-project-linux":
        return open_codex_project_linux_home()
    if args.mode == "codex-project-linux-gold-db":
        return open_codex_project_linux_home(enable_gold_db=True)
    if args.mode == "codex-project-linux-search-gold-db":
        return open_codex_project_linux_home(search_mode=True, enable_gold_db=True)
    if args.mode == "codex-project-linux-power":
        return open_codex_project_linux_home_power()
    if args.mode == "green-codex":
        return open_codex_green_baseline()
    if args.mode == "green-claude":
        return open_claude_green_baseline()
    if args.mode == "codex-search":
        task = args.task or prompt("Workstream name")
        if not task:
            return 1
        return open_codex_workstream(task, "Investigate / search", True)
    if args.mode == "handoff":
        if not args.task:
            raise RuntimeError("Workstream name required for handoff.")
        if not args.tool:
            raise RuntimeError("Target tool required for handoff.")
        current = get_workstream_metadata("claude", args.task) or get_workstream_metadata("codex", args.task)
        if current is None:
            raise RuntimeError(f"Unable to find workstream {args.task}.")
        current_tool = str(current.get("tool") or "")
        purpose = str(current.get("purpose") or "")
        success, output = handoff_workstream(
            args.task, current_tool=current_tool, target_tool=args.tool, purpose=purpose
        )
        if not success:
            raise RuntimeError(output)
        if args.tool == "claude":
            return open_claude_workstream(args.task, purpose)
        return open_codex_workstream(args.task, purpose, purpose == "Investigate / search")
    if args.mode == "list":
        workstreams = get_managed_workstreams()
        table = build_workstream_table(workstreams)
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
    if args.mode == "ship":
        if not args.task:
            raise RuntimeError("Workstream name required for ship.")
        current = get_workstream_metadata("claude", args.task) or get_workstream_metadata("codex", args.task)
        if current is None:
            raise RuntimeError(f"Unable to find workstream {args.task}.")
        success, output = ship_workstream(args.task, tool=str(current.get("tool") or ""))
        if not success:
            raise RuntimeError(output)
        if output:
            console.print(output)
        return 0
    if args.mode == "doctor":
        from codex_doctor import run as run_doctor

        return run_doctor()
    if args.mode == "cleanup":
        return cleanup_codex_wsl_clone()
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
