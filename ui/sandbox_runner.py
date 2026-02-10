"""
Sandbox command runner for SOP-compliant execution.

Rules (from sysopgates.txt):
- Read-only commands run directly against main DB
- DB-write commands auto-sandbox to C:\\db\\gold_sandbox.db
- Only allowlisted executables: python, duckdb
- No shell pipes, eval, or background execution
- All paths resolved to absolute before execution
"""

import shutil
import subprocess
from pathlib import Path

from pipeline.paths import GOLD_DB_PATH

SANDBOX_DB_PATH = Path(r"C:\db\gold_sandbox.db")

# Commands that only read -- safe to run against main DB
READ_ONLY_COMMANDS = [
    "pytest",
    "check_drift.py",
    "health_check.py",
    "check_db.py",
    "dashboard.py",
    "audit_bars_coverage.py",
]

# Commands that write to DB -- must sandbox
WRITE_COMMANDS = [
    "outcome_builder.py",
    "strategy_discovery.py",
    "strategy_validator.py",
    "build_bars_5m.py",
    "build_daily_features.py",
    "ingest_dbn_mgc.py",
    "run_pipeline.py",
]

# Executable allowlist
ALLOWED_EXECUTABLES = {"python", "python3", "duckdb", "pytest"}


def _is_read_only(command: str) -> bool:
    """Check if a command is read-only based on known patterns."""
    for pattern in READ_ONLY_COMMANDS:
        if pattern in command:
            return True
    return False


def _is_write_command(command: str) -> bool:
    """Check if a command writes to the database."""
    for pattern in WRITE_COMMANDS:
        if pattern in command:
            return True
    return False


def _validate_command(command: str) -> tuple[bool, str]:
    """Validate command against allowlist. Returns (valid, reason)."""
    parts = command.strip().split()
    if not parts:
        return False, "Empty command"

    exe = Path(parts[0]).name.lower()
    # Handle "python -m pytest" style
    if exe in ("python", "python3", "python.exe"):
        return True, ""
    if exe in ALLOWED_EXECUTABLES:
        return True, ""

    return False, f"Executable '{exe}' not in allowlist: {sorted(ALLOWED_EXECUTABLES)}"


def _has_shell_injection(command: str) -> bool:
    """Reject commands with shell metacharacters."""
    dangerous = ["|", "&&", "||", ";", "`", "$(", ">>", "<<", "eval ", "exec "]
    return any(d in command for d in dangerous)


def create_sandbox() -> Path:
    """Copy main DB to sandbox location. Returns sandbox path."""
    SANDBOX_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(GOLD_DB_PATH), str(SANDBOX_DB_PATH))
    return SANDBOX_DB_PATH


def promote_sandbox() -> bool:
    """Copy sandbox DB back to main DB. Returns True on success."""
    if not SANDBOX_DB_PATH.exists():
        return False
    shutil.copy2(str(SANDBOX_DB_PATH), str(GOLD_DB_PATH))
    return True


def discard_sandbox() -> bool:
    """Delete the sandbox DB. Returns True on success."""
    if SANDBOX_DB_PATH.exists():
        SANDBOX_DB_PATH.unlink()
        return True
    return False


def rewrite_for_sandbox(command: str) -> str:
    """Rewrite a command to use the sandbox DB path instead of main."""
    # Replace common DB path references
    cmd = command.replace(str(GOLD_DB_PATH), str(SANDBOX_DB_PATH))
    cmd = cmd.replace("gold.db", str(SANDBOX_DB_PATH))
    return cmd


def run_sandboxed(
    command: str,
    needs_db_write: bool | None = None,
    timeout_seconds: int = 300,
) -> tuple[int, str, bool]:
    """Run a command safely. Auto-detects read-only vs write.

    Args:
        command: Shell command to execute
        needs_db_write: Override auto-detection. None = auto-detect.
        timeout_seconds: Max execution time

    Returns:
        (exit_code, output, was_sandboxed)
    """
    # Validate
    valid, reason = _validate_command(command)
    if not valid:
        return 1, f"BLOCKED: {reason}", False

    if _has_shell_injection(command):
        return 1, "BLOCKED: Shell metacharacters detected. No pipes or chaining allowed.", False

    # Determine if sandboxing needed
    if needs_db_write is None:
        needs_db_write = _is_write_command(command) and not _is_read_only(command)

    was_sandboxed = False
    run_cmd = command

    if needs_db_write:
        create_sandbox()
        run_cmd = rewrite_for_sandbox(command)
        was_sandboxed = True

    # Execute
    try:
        result = subprocess.run(
            run_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(GOLD_DB_PATH.parent),
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr
        return result.returncode, output.strip(), was_sandboxed
    except subprocess.TimeoutExpired:
        return 1, f"TIMEOUT: Command exceeded {timeout_seconds}s limit", was_sandboxed
    except Exception as e:
        return 1, f"ERROR: {e}", was_sandboxed
