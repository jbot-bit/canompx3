#!/usr/bin/env python3
"""Repo-owned setup, actions, and cleanup for Codex local environments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_SUBTREES = {
    ".claude",
    ".codex",
    ".git",
    ".venv",
    ".venv-wsl",
    ".worktrees",
    "node_modules",
}
REMOVABLE_DIRS = {
    ".cache",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "htmlcov",
}
REMOVABLE_FILES = {
    ".coverage",
}
EXPECTED_PRIMARY_MODEL = "gpt-5.4"
SHARED_CODEX_HOME_OVERRIDE_ENV = "CANOMPX3_SHARED_CODEX_HOME"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex local-environment helpers")
    parser.add_argument(
        "command",
        choices=["setup", "cleanup", "status", "doctor", "lint", "tests", "drift"],
        help="Action to run",
    )
    parser.add_argument(
        "--platform",
        choices=["windows", "wsl"],
        required=True,
        help="Runtime platform for env and interpreter selection",
    )
    return parser.parse_args(argv)


def env_for_platform(platform: str) -> dict[str, str]:
    env = os.environ.copy()
    env["JOBLIB_MULTIPROCESSING"] = "0"
    if platform == "wsl":
        env["UV_PROJECT_ENVIRONMENT"] = ".venv-wsl"
        env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
        env.setdefault("UV_PYTHON_INSTALL_DIR", "/tmp/uv-python")
        env.setdefault("UV_LINK_MODE", "copy")
        local_codex_home = default_local_codex_home()
        shared_codex_home = default_shared_codex_home()
        forced_shared_home = os.environ.get(SHARED_CODEX_HOME_OVERRIDE_ENV)
        if forced_shared_home:
            forced_path = Path(forced_shared_home)
            if forced_path.exists():
                env.setdefault("CODEX_HOME", str(forced_path))
        elif not (local_codex_home and local_codex_home.exists()):
            if shared_codex_home and shared_codex_home.exists():
                env.setdefault("CODEX_HOME", str(shared_codex_home))
    else:
        env["UV_PROJECT_ENVIRONMENT"] = ".venv"
    return env


def safe_path_exists(path: Path) -> tuple[bool, str]:
    try:
        return path.exists(), str(path)
    except OSError as exc:
        return False, f"{path} ({exc})"


def platform_python(platform: str) -> list[str]:
    if platform == "wsl":
        venv_python = ROOT / ".venv-wsl" / "bin" / "python"
        venv_exists, _ = safe_path_exists(venv_python)
        if venv_exists:
            return [str(venv_python)]
        return ["python3"]

    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    venv_exists, _ = safe_path_exists(venv_python)
    if venv_exists:
        return [str(venv_python)]
    if shutil.which("py"):
        return ["py", "-3"]
    return ["python"]


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(command))
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return result.returncode


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    exit_code = run_command(command, env=env)
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_setup(platform: str) -> None:
    env = env_for_platform(platform)
    python_cmd = platform_python(platform)

    if platform == "wsl":
        if not is_wsl_native_root(ROOT):
            raise SystemExit(
                "\n".join(
                    [
                        f"ERROR: Codex WSL setup is blocked for non-native repo root: {ROOT}",
                        "",
                        "Why launch is blocked:",
                        "  - Codex app sessions on /mnt/... are fallback-only in this repo.",
                        "  - The supported daily-driver path is a WSL-home clone such as ~/canompx3.",
                        "  - This avoids /mnt/c mount instability, slow I/O, and Codex session crashes.",
                        "",
                        "Recovery:",
                        "  1. Launch the WSL-home clone instead, for example with `codex.bat linux`.",
                        "  2. Or set CANOMPX3_CODEX_WSL_ROOT to your WSL-side clone path.",
                        "  3. Keep /mnt/c launches for compatibility only, not Codex app daily use.",
                    ]
                )
            )
        run_checked(python_cmd + ["scripts/tools/wsl_mount_guard.py", "--root", str(ROOT)], env=env)

    run_checked(["uv", "sync", "--frozen", "--python", "3.13", "--group", "dev"], env=env)

    preflight = ROOT / "scripts" / "tools" / "session_preflight.py"
    if preflight.exists():
        context = "codex-wsl" if platform == "wsl" else "generic"
        run_checked(
            ["uv", "run", "--frozen", "python"]
            + [
                str(preflight),
                "--quiet",
                "--context",
                context,
                "--claim",
                "codex",
                "--mode",
                "mutating",
            ],
            env=env,
        )


def run_status(platform: str) -> None:
    del platform
    run_checked(["git", "status", "--short", "--branch"])
    run_checked(["git", "worktree", "list"])


def run_lint(platform: str) -> None:
    env = env_for_platform(platform)
    run_checked(
        ["uv", "run", "--frozen", "ruff", "check", "pipeline", "trading_app", "ui", "scripts", "tests"], env=env
    )


def run_tests(platform: str) -> None:
    env = env_for_platform(platform)
    run_checked(["uv", "run", "--frozen", "python", "-m", "pytest", "tests/", "-x", "-q"], env=env)


def run_drift(platform: str) -> None:
    env = env_for_platform(platform)
    run_checked(["uv", "run", "--frozen", "python", "pipeline/check_drift.py"], env=env)


def is_wsl_native_root(root: Path) -> bool:
    root_text = root.as_posix()
    return root_text.startswith("/") and not root_text.startswith("/mnt/")


def capture_command(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=str(exc))


def _doctor_check(label: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    return label, ok, detail


def _doctor_status(label: str, status: str, detail: str) -> tuple[str, str, str]:
    return label, status, detail


def _tail_text(path: Path, *, max_bytes: int = 65536) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        handle.seek(max(file_size - max_bytes, 0))
        return handle.read().decode("utf-8", errors="replace")


def _parse_mount_guard_detail(platform: str, env: dict[str, str]) -> tuple[str, str]:
    mount_guard = capture_command(
        platform_python(platform) + ["scripts/tools/wsl_mount_guard.py", "--root", str(ROOT), "--json"],
        env=env,
    )
    if mount_guard.returncode not in {0, 2}:
        return "FAIL", mount_guard.stdout.strip() or mount_guard.stderr.strip() or f"exit {mount_guard.returncode}"
    try:
        payload = json.loads(mount_guard.stdout)
    except json.JSONDecodeError:
        return "FAIL", mount_guard.stdout.strip() or mount_guard.stderr.strip() or "invalid JSON from mount guard"

    state = str(payload.get("state", "unknown"))
    fatal_issues = [str(item) for item in payload.get("fatal_issues", [])]
    warnings = [str(item) for item in payload.get("warnings", [])]
    if state == "healthy":
        return "PASS", "repo root is writable and no unhealthy nested mounts were detected"
    if state == "sandbox-protected":
        joined = "; ".join(warnings) if warnings else "protected paths remain read-only inside workspace-write"
        return "WARN", joined
    joined = "; ".join(fatal_issues) if fatal_issues else "WSL repo mount is unhealthy"
    return "FAIL", joined


def _detect_recent_wsl_reset() -> str | None:
    journal = capture_command(["journalctl", "-b", "--no-pager"])
    text = f"{journal.stdout}\n{journal.stderr}"
    if "Operation canceled @p9io.cpp:258 (AcceptAsync)" in text and "unmounting filesystem" in text:
        return (
            "current-boot journal shows a WSL reset signature "
            "(`AcceptAsync` plus distro filesystem unmount/remount); if terminals are still dying, run `wsl --shutdown`, "
            "then `wsl --update`, and use the Microsoft VHD repair flow if the issue repeats"
        )
    if "uncleanly shut down" in text and "unmounting filesystem" in text:
        return "current-boot journal shows an unclean WSL shutdown/remount sequence"
    return None


def _effective_codex_home(env: dict[str, str] | None = None) -> Path | None:
    if env:
        codex_home = env.get("CODEX_HOME")
        if codex_home:
            return Path(codex_home)
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home)
    local_home = default_local_codex_home()
    if local_home and local_home.exists():
        return local_home
    shared_codex_home = default_shared_codex_home()
    if shared_codex_home and shared_codex_home.exists():
        return shared_codex_home
    return None


def default_local_codex_home() -> Path | None:
    home = Path.home()
    return home / ".codex" if home.exists() else None


def _detect_model_mismatch_warning(env: dict[str, str] | None = None) -> str | None:
    codex_home = _effective_codex_home(env)
    if codex_home is None:
        return None
    if _read_toml_model(codex_home / "config.toml") == EXPECTED_PRIMARY_MODEL:
        return None
    log_path = codex_home / "log" / "codex-tui.log"
    if not log_path.exists():
        return None
    try:
        recent = _tail_text(log_path)
    except OSError as exc:
        return f"could not inspect Codex log for recent model/build mismatches ({exc})"
    if "requires a newer version of Codex" in recent:
        return (
            "recent Codex log shows a configured model that requires a newer app/CLI build; "
            "update Codex or pin the primary workflow to a supported model"
        )
    return None


def default_shared_codex_home() -> Path | None:
    override = os.environ.get(SHARED_CODEX_HOME_OVERRIDE_ENV)
    if override:
        return Path(override)
    user = os.environ.get("USER")
    if not user:
        return None
    return Path("/mnt/c/Users") / user / ".codex"


def _shared_codex_home_status() -> tuple[str, str]:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return "PASS", f"CODEX_HOME is already set to `{codex_home}`"
    local_codex_home = default_local_codex_home()
    if local_codex_home and local_codex_home.exists():
        return (
            "PASS",
            f"managed WSL launchers will use the local user-level Codex home `{local_codex_home}` by default",
        )
    shared_codex_home = default_shared_codex_home()
    if shared_codex_home and shared_codex_home.exists():
        return (
            "PASS",
            "no local `~/.codex` detected, so managed WSL launchers will fall back to "
            f"`CODEX_HOME={shared_codex_home}`",
        )
    return (
        "WARN",
        "no Codex home was found for WSL; create `~/.codex` or set "
        f"`{SHARED_CODEX_HOME_OVERRIDE_ENV}` / `CODEX_HOME` explicitly",
    )


def _read_toml_model(path: Path) -> str | None:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    model = payload.get("model")
    return model if isinstance(model, str) else None


def _detect_primary_model_drift(env: dict[str, str] | None = None) -> str | None:
    codex_home = _effective_codex_home(env)
    if codex_home is None:
        return None
    user_model = _read_toml_model(codex_home / "config.toml")
    if user_model and user_model != EXPECTED_PRIMARY_MODEL:
        return (
            f"effective Codex home `{codex_home}` defaults to `{user_model}`; this repo's stable primary path is "
            f"`{EXPECTED_PRIMARY_MODEL}` until a newer Codex build is verified"
        )
    return None


def _codex_login_looks_healthy(env: dict[str, str] | None = None) -> bool | None:
    codex_path = shutil.which("codex")
    if not codex_path:
        return None
    status = capture_command([codex_path, "login", "status"], env=env)
    text = f"{status.stdout}\n{status.stderr}".strip().lower()
    if status.returncode == 0 and "logged in" in text:
        return True
    if "not logged in" in text:
        return False
    return None


def _detect_auth_state_warning(env: dict[str, str] | None = None) -> str | None:
    codex_home = _effective_codex_home(env)
    if codex_home is None:
        return None
    auth_path = codex_home / "auth.json"
    log_path = codex_home / "log" / "codex-tui.log"
    if not log_path.exists():
        return None
    try:
        recent = _tail_text(log_path)
    except OSError as exc:
        return f"could not inspect Codex auth state at `{log_path}` ({exc})"

    if "refresh token was already used" in recent or "token_expired" in recent:
        try:
            auth_is_newer = auth_path.exists() and auth_path.stat().st_mtime_ns > log_path.stat().st_mtime_ns
        except OSError:
            auth_is_newer = False
        if auth_is_newer and _codex_login_looks_healthy(env) is True:
            return None
        return (
            f"effective Codex home `{codex_home}` has stale auth in `log/codex-tui.log`; "
            "log out and sign in again in the Windows Codex app/CLI before retrying `codex.bat`"
        )
    return None


def run_doctor(platform: str) -> None:
    env = env_for_platform(platform)
    checks: list[tuple[str, str, str]] = []

    if platform == "wsl":
        venv_exists, venv_detail = safe_path_exists(ROOT / ".venv-wsl" / "bin" / "python")
        checks.append(
            _doctor_status(
                "WSL-native repo root",
                "PASS" if is_wsl_native_root(ROOT) else "FAIL",
                str(ROOT),
            )
        )
        checks.append(
            _doctor_status(
                ".venv-wsl present",
                "PASS" if venv_exists else "FAIL",
                venv_detail,
            )
        )
        codex_path = shutil.which("codex")
        checks.append(
            _doctor_status(
                "Codex binary available",
                "PASS" if codex_path else "FAIL",
                codex_path or "codex not found on PATH",
            )
        )
        checks.append(
            _doctor_status(
                "WSL mount guard",
                *_parse_mount_guard_detail(platform, env),
            )
        )
        reset_warning = _detect_recent_wsl_reset()
        if reset_warning:
            checks.append(_doctor_status("Recent WSL reset evidence", "WARN", reset_warning))
        model_warning = _detect_model_mismatch_warning(env)
        if model_warning:
            checks.append(_doctor_status("Codex model compatibility", "WARN", model_warning))
        primary_model_warning = _detect_primary_model_drift(env)
        if primary_model_warning:
            checks.append(_doctor_status("Primary model drift", "WARN", primary_model_warning))
        auth_warning = _detect_auth_state_warning(env)
        if auth_warning:
            checks.append(_doctor_status("Codex auth state", "WARN", auth_warning))
        shared_home_status, shared_home_detail = _shared_codex_home_status()
        checks.append(_doctor_status("Shared CODEX_HOME", shared_home_status, shared_home_detail))
        preflight_context = "codex-wsl"
    else:
        venv_exists, venv_detail = safe_path_exists(ROOT / ".venv" / "Scripts" / "python.exe")
        checks.append(
            _doctor_status(
                ".venv present",
                "PASS" if venv_exists else "FAIL",
                venv_detail,
            )
        )
        launcher = ROOT / "codex.bat"
        checks.append(
            _doctor_status(
                "Codex launcher available",
                "PASS" if launcher.exists() else "FAIL",
                str(launcher),
            )
        )
        preflight_context = "generic"

    preflight = ROOT / "scripts" / "tools" / "session_preflight.py"
    preflight_exists, preflight_detail = safe_path_exists(preflight)
    if preflight_exists:
        preflight_result = capture_command(
            [
                "uv",
                "run",
                "--frozen",
                "python",
                str(preflight),
                "--quiet",
                "--context",
                preflight_context,
                "--claim",
                "codex-search",
                "--mode",
                "read-only",
            ],
            env=env,
        )
        checks.append(
            _doctor_status(
                "Session preflight",
                "PASS" if preflight_result.returncode == 0 else "FAIL",
                preflight_result.stdout.strip()
                or preflight_result.stderr.strip()
                or f"exit {preflight_result.returncode}",
            )
        )
    else:
        checks.append(_doctor_status("Session preflight", "FAIL", f"missing {preflight_detail}"))

    worktrees = capture_command(["git", "worktree", "list"], env=env)
    worktree_detail = worktrees.stdout.strip().splitlines()
    checks.append(
        _doctor_status(
            "Git worktree visibility",
            "PASS" if worktrees.returncode == 0 else "FAIL",
            worktree_detail[0] if worktree_detail else worktrees.stderr.strip() or f"exit {worktrees.returncode}",
        )
    )

    failures = 0
    for label, status, detail in checks:
        print(f"[{status}] {label}: {detail}")
        if status == "FAIL":
            failures += 1

    if failures:
        raise SystemExit(1)


def cleanup_paths(root: Path) -> list[Path]:
    removed: list[Path] = []
    for current_root, dirs, files in os.walk(root, topdown=True):
        current = Path(current_root)
        dirs[:] = [name for name in dirs if name not in SKIP_SUBTREES]

        for dirname in list(dirs):
            candidate = current / dirname
            if dirname == "__pycache__" or dirname in REMOVABLE_DIRS or dirname.endswith(".egg-info"):
                shutil.rmtree(candidate, ignore_errors=True)
                removed.append(candidate)
                dirs.remove(dirname)

        for filename in files:
            candidate = current / filename
            if filename in REMOVABLE_FILES or filename.endswith((".pyc", ".pyo")):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    continue
                removed.append(candidate)
    return removed


def run_cleanup(platform: str) -> None:
    del platform
    removed = cleanup_paths(ROOT)
    if not removed:
        print("Nothing to clean.")
        return

    print(f"Removed {len(removed)} paths:")
    for path in removed[:40]:
        print(f"- {path.relative_to(ROOT)}")
    if len(removed) > 40:
        print(f"- ... and {len(removed) - 40} more")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "setup":
        run_setup(args.platform)
        return 0
    if args.command == "cleanup":
        run_cleanup(args.platform)
        return 0
    if args.command == "status":
        run_status(args.platform)
        return 0
    if args.command == "doctor":
        run_doctor(args.platform)
        return 0
    if args.command == "lint":
        run_lint(args.platform)
        return 0
    if args.command == "tests":
        run_tests(args.platform)
        return 0
    if args.command == "drift":
        run_drift(args.platform)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
