import ast
import re
from pathlib import Path


def _extract_codex_bat_modes(codex_bat: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r'set "MODE=([^"]+)"', codex_bat)
        if match.group(1).startswith("codex-project") or match.group(1).startswith("green-")
    }


def _extract_powershell_validate_set(script_text: str) -> set[str]:
    match = re.search(r"\[ValidateSet\((.*?)\)\]", script_text, re.DOTALL)
    assert match, "windows-agent-launch.ps1 is missing a ValidateSet declaration"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def _extract_python_valid_modes(script_text: str) -> set[str]:
    module = ast.parse(script_text)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "VALID_MODES":
                return {
                    elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }
    raise AssertionError("windows_agent_launch.py is missing VALID_MODES")


def test_wsl_launcher_scripts_call_mount_guard() -> None:
    root = Path(__file__).resolve().parents[2]
    project = (root / "scripts" / "infra" / "codex-project.sh").read_text(encoding="utf-8")
    search = (root / "scripts" / "infra" / "codex-project-search.sh").read_text(encoding="utf-8")
    review = (root / "scripts" / "infra" / "codex-review.sh").read_text(encoding="utf-8")
    worktree = (root / "scripts" / "infra" / "codex-worktree.sh").read_text(encoding="utf-8")
    sync_guard = (root / "scripts" / "infra" / "codex-wsl-sync.sh").read_text(encoding="utf-8")

    assert 'wsl_mount_guard.py" --root "$ROOT"' in project
    assert 'wsl_mount_guard.py" --root "$ROOT"' in search
    assert "task_route_packet.py" in project
    assert "task_route_packet.py" in search
    assert 'CANOMPX3_SESSION_OWNER="pid:$$"' in project
    assert 'CANOMPX3_SESSION_OWNER="pid:$$"' in search
    assert 'wsl_mount_guard.py" --root "$ROOT"' in review
    assert 'python3 "$ROOT/scripts/tools/wsl_mount_guard.py" --root "$ROOT"' in worktree
    assert "task_route_packet.py" in worktree
    assert '--related-root "$SOURCE_ROOT"' in sync_guard
    assert "--claim codex" in sync_guard
    assert "--mode mutating" in sync_guard


def test_codex_bat_modes_are_supported_by_windows_launchers() -> None:
    root = Path(__file__).resolve().parents[2]
    codex_bat = (root / "codex.bat").read_text(encoding="utf-8")
    ps_launcher = (root / "scripts" / "infra" / "windows-agent-launch.ps1").read_text(encoding="utf-8")
    py_launcher = (root / "scripts" / "infra" / "windows_agent_launch.py").read_text(encoding="utf-8")

    codex_modes = _extract_codex_bat_modes(codex_bat)
    powershell_modes = _extract_powershell_validate_set(ps_launcher)
    python_modes = _extract_python_valid_modes(py_launcher)

    missing_in_powershell = codex_modes - powershell_modes
    missing_in_python = codex_modes - python_modes

    assert not missing_in_powershell, f"codex.bat exposes unsupported PowerShell modes: {sorted(missing_in_powershell)}"
    assert not missing_in_python, f"codex.bat exposes unsupported Python launcher modes: {sorted(missing_in_python)}"


def test_linux_modes_route_to_wsl_home_clone() -> None:
    root = Path(__file__).resolve().parents[2]
    py_launcher = (root / "scripts" / "infra" / "windows_agent_launch.py").read_text(encoding="utf-8")

    assert py_launcher.count("use_linux_home=True") >= 2
