#!/usr/bin/env python3
"""Behavioral anti-pattern scanner — catches common AI/human coding mistakes.

Each check returns list[str] violations. Exit code 0 = all passed, 1 = failures.
Self-reports check count dynamically.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
SELF_PATH = Path(__file__).resolve()

# Directories to scan for each check
PIPELINE_DIRS = [PROJECT_ROOT / "pipeline", PROJECT_ROOT / "trading_app", PROJECT_ROOT / "scripts" / "tools"]
INSTRUMENT_SCAN_DIRS = [PROJECT_ROOT / "pipeline", PROJECT_ROOT / "scripts" / "tools"]
CI_DIRS = [PROJECT_ROOT / ".github" / "workflows"]

# Files allowed to contain hardcoded instrument lists (they ARE config/definition sources
# or one-off analysis scripts not part of the production pipeline)
INSTRUMENT_ALLOWLIST = {
    "check_drift.py",
    "config.py",
    "asset_configs.py",
    "cost_model.py",
    "hypothesis_test.py",
    "audit_15m30m.py",
}
# Directories whose files are always allowed
INSTRUMENT_ALLOWLIST_DIRS = {"tests", "docs", "research"}


def _python_files(dirs: list[Path]) -> list[Path]:
    """Collect .py files from given directories."""
    files = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob("*.py"))
    return sorted(files)


def _text_files(dirs: list[Path], exts: tuple[str, ...] = (".py", ".yml", ".yaml", ".md")) -> list[Path]:
    """Collect text files from given directories."""
    files = []
    for d in dirs:
        if d.exists():
            for ext in exts:
                files.extend(d.rglob(f"*{ext}"))
    return sorted(set(files))


# ── Check 1: Hardcoded check counts ──────────────────────────────────

# Pattern: "all <number> checks" — counts should be computed at runtime
HARDCODED_COUNT_PATTERN = re.compile(r'\ball\s+\d+\s+checks\b', re.IGNORECASE)


def check_hardcoded_check_counts() -> list[str]:
    """Detect hardcoded check counts — these should be computed dynamically."""
    violations = []
    scan_dirs = PIPELINE_DIRS + CI_DIRS
    for f in _text_files(scan_dirs):
        if f.resolve() == SELF_PATH:
            continue
        try:
            text = f.read_text(encoding='utf-8')
        except (UnicodeDecodeError, PermissionError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if HARDCODED_COUNT_PATTERN.search(line):
                rel = f.relative_to(PROJECT_ROOT)
                violations.append(f"  {rel}:{i}: hardcoded check count: {line.strip()[:80]}")
    return violations


# ── Check 2: Hardcoded instrument lists ──────────────────────────────

# Detects 3+ instrument symbols in a Python list literal or SQL IN clause.
# These should import from pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS instead.
_INST = r'(?:MGC|MNQ|MES|M2K|MCL|SIL|M6E)'
PY_INSTRUMENT_LIST = re.compile(
    rf"""\[[\s'"]*{_INST}['"][\s,'"]*{_INST}['"][\s,'"]*{_INST}""",
    re.IGNORECASE,
)
SQL_INSTRUMENT_IN = re.compile(
    rf"""IN\s*\(\s*'{_INST}'\s*,\s*'{_INST}'""",
    re.IGNORECASE,
)


def _is_allowlisted(filepath: Path) -> bool:
    """Check if file is in the allowlist for instrument lists."""
    if filepath.name in INSTRUMENT_ALLOWLIST:
        return True
    for d in INSTRUMENT_ALLOWLIST_DIRS:
        try:
            filepath.relative_to(PROJECT_ROOT / d)
            return True
        except ValueError:
            pass
    return False


def check_hardcoded_instrument_lists() -> list[str]:
    """Detect hardcoded instrument lists in non-allowlisted files."""
    violations = []
    for f in _python_files(INSTRUMENT_SCAN_DIRS):
        if f.resolve() == SELF_PATH or _is_allowlisted(f):
            continue
        try:
            text = f.read_text(encoding='utf-8')
        except (UnicodeDecodeError, PermissionError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if PY_INSTRUMENT_LIST.search(line) or SQL_INSTRUMENT_IN.search(line):
                rel = f.relative_to(PROJECT_ROOT)
                violations.append(f"  {rel}:{i}: hardcoded instrument list: {line.strip()[:80]}")
    return violations


# ── Check 3: Broad except returning success ──────────────────────────

BROAD_EXCEPT_PATTERN = re.compile(r'except\s+(?:Exception|BaseException)\b')
SUCCESS_RETURN_PATTERN = re.compile(r'return\s+(?:True|0|None)\b')

# Only check health/integrity/audit paths where swallowing exceptions is dangerous
EXCEPT_SCAN_GLOBS = [
    "pipeline/health_check.py",
    "scripts/tools/audit_*.py",
]


def check_broad_except_success() -> list[str]:
    """Detect 'except Exception' followed by 'return True/0' in health/audit code."""
    violations = []
    files = []
    for pattern in EXCEPT_SCAN_GLOBS:
        files.extend(PROJECT_ROOT.glob(pattern))

    for f in sorted(set(files)):
        try:
            lines = f.read_text(encoding='utf-8').splitlines()
        except (UnicodeDecodeError, PermissionError):
            continue
        for i, line in enumerate(lines):
            if BROAD_EXCEPT_PATTERN.search(line):
                # Check next 3 lines for success return
                for j in range(i + 1, min(i + 4, len(lines))):
                    if SUCCESS_RETURN_PATTERN.search(lines[j]):
                        rel = f.relative_to(PROJECT_ROOT)
                        violations.append(
                            f"  {rel}:{i+1}: broad except + success return: "
                            f"{line.strip()[:40]} ... {lines[j].strip()[:40]}"
                        )
                        break
    return violations


# ── Check 4: CLI arg drift (WARNING only) ────────────────────────────

def check_cli_arg_drift() -> list[str]:
    """Detect new CLI args in recent diff with no matching docs/test reference.

    WARNING only — best-effort heuristic that fails open.
    Returns warnings (not violations) so it never blocks.
    """
    warnings = []
    try:
        # Get recent changes (staged + unstaged)
        result = subprocess.run(
            ["git", "diff", "HEAD", "--unified=0"],
            capture_output=True, text=True, timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            return []

        diff_text = result.stdout
        # Find added add_argument lines
        new_args = []
        current_file = None
        for line in diff_text.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("+") and not line.startswith("+++"):
                if "add_argument(" in line and current_file:
                    # Extract arg name
                    match = re.search(r'["\'](-[-\w]+)["\']', line)
                    if match:
                        new_args.append((current_file, match.group(1)))

        if not new_args:
            return []

        # Check if matching docs or test reference exists in diff
        for filepath, arg_name in new_args:
            arg_base = arg_name.lstrip('-').replace('-', '_')
            if arg_base not in diff_text.replace(filepath, ''):
                warnings.append(
                    f"  WARNING: {filepath}: new arg '{arg_name}' — no doc/test reference in diff"
                )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Fails open
    return warnings


# ── Check registry ───────────────────────────────────────────────────

CHECKS = [
    ("1. Hardcoded check counts", check_hardcoded_check_counts, False),
    ("2. Hardcoded instrument lists", check_hardcoded_instrument_lists, False),
    ("3. Broad except returning success", check_broad_except_success, False),
    ("4. CLI arg drift (warning only)", check_cli_arg_drift, True),  # warning_only
]


def main():
    print('=' * 70)
    print('BEHAVIORAL AUDIT — ANTI-PATTERN SCANNER')
    print('=' * 70)

    all_violations = []

    for label, check_fn, warning_only in CHECKS:
        print(f'\n--- {label} ---')
        results = check_fn()
        if results:
            tag = "WARNING" if warning_only else "FAILED"
            print(f'  {tag}:')
            for line in results:
                print(line)
            if not warning_only:
                all_violations.extend(results)
        else:
            print('  OK')

    print('\n' + '=' * 70)
    if all_violations:
        print(f'BEHAVIORAL AUDIT FAILED: {len(all_violations)} violation(s)')
        print('=' * 70)
        sys.exit(1)
    else:
        print(f'BEHAVIORAL AUDIT PASSED: all {len(CHECKS)} checks clean')
        print('=' * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
