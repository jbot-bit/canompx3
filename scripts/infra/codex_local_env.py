#!/usr/bin/env python3
"""Repo-owned setup, actions, and cleanup for Codex local environments."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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
    else:
        env["UV_PROJECT_ENVIRONMENT"] = ".venv"
    return env


def platform_python(platform: str) -> list[str]:
    if platform == "wsl":
        venv_python = ROOT / ".venv-wsl" / "bin" / "python"
        if venv_python.exists():
            return [str(venv_python)]
        return ["python3"]

    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
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


def _doctor_check(label: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    return label, ok, detail


def run_doctor(platform: str) -> None:
    env = env_for_platform(platform)
    checks: list[tuple[str, bool, str]] = []

    if platform == "wsl":
        checks.append(
            _doctor_check(
                "WSL-native repo root",
                is_wsl_native_root(ROOT),
                str(ROOT),
            )
        )
        checks.append(
            _doctor_check(
                ".venv-wsl present",
                (ROOT / ".venv-wsl" / "bin" / "python").exists(),
                str(ROOT / ".venv-wsl" / "bin" / "python"),
            )
        )
        codex_path = shutil.which("codex")
        checks.append(
            _doctor_check(
                "Codex binary available",
                bool(codex_path),
                codex_path or "codex not found on PATH",
            )
        )
        mount_guard = capture_command(
            platform_python(platform) + ["scripts/tools/wsl_mount_guard.py", "--root", str(ROOT)],
            env=env,
        )
        checks.append(
            _doctor_check(
                "WSL mount guard",
                mount_guard.returncode == 0,
                mount_guard.stdout.strip() or mount_guard.stderr.strip() or f"exit {mount_guard.returncode}",
            )
        )
        preflight_context = "codex-wsl"
    else:
        checks.append(
            _doctor_check(
                ".venv present",
                (ROOT / ".venv" / "Scripts" / "python.exe").exists(),
                str(ROOT / ".venv" / "Scripts" / "python.exe"),
            )
        )
        launcher = ROOT / "codex.bat"
        checks.append(
            _doctor_check(
                "Codex launcher available",
                launcher.exists(),
                str(launcher),
            )
        )
        preflight_context = "generic"

    preflight = ROOT / "scripts" / "tools" / "session_preflight.py"
    if preflight.exists():
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
            _doctor_check(
                "Session preflight",
                preflight_result.returncode == 0,
                preflight_result.stdout.strip()
                or preflight_result.stderr.strip()
                or f"exit {preflight_result.returncode}",
            )
        )
    else:
        checks.append(_doctor_check("Session preflight", False, f"missing {preflight}"))

    worktrees = capture_command(["git", "worktree", "list"], env=env)
    worktree_detail = worktrees.stdout.strip().splitlines()
    checks.append(
        _doctor_check(
            "Git worktree visibility",
            worktrees.returncode == 0,
            worktree_detail[0] if worktree_detail else worktrees.stderr.strip() or f"exit {worktrees.returncode}",
        )
    )

    failures = 0
    for label, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: {detail}")
        if not ok:
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
