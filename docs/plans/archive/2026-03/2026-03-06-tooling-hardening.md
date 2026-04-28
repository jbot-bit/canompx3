---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Tooling Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Bring the project to Bloomberg-grade professional Python tooling — type checking, formatting, expanded linting, reproducible builds, security scanning, coverage enforcement.

**Architecture:** All changes are config/tooling files — no production logic changes except auto-fixed lint/format. Each task produces one atomic commit. Tasks are ordered by dependency: formatting first (normalizes diffs), then linting (catches bugs), then deps (reproducibility), then type checking (catches type bugs), then CI/editor/hooks (enforcement).

**Tech Stack:** pyright (type checking), ruff (lint + format), uv (dependency management), pip-audit (security), pytest-cov (coverage)

**Design Doc:** `docs/plans/2026-03-06-tooling-hardening-design.md`

---

### Task 0: Measure coverage baseline + add .python-version

**Files:**
- Create: `.python-version`
- Reference: `pyproject.toml` (read only — verify `requires-python`)

**Step 1: Measure current coverage**

Run:
```bash
python -m pytest tests/ -x -q --cov=pipeline --cov=trading_app --cov-report=term 2>&1 | tail -5
```

Record the total coverage percentage. This becomes the baseline for Task 5.

**Step 2: Create `.python-version`**

```
3.13
```

One line, no trailing newline needed. This is read by pyenv, uv, mise, asdf.

**Step 3: Verify `pyproject.toml` matches**

Confirm `requires-python = ">=3.13"` already exists in `pyproject.toml`. No change needed.

**Step 4: Commit**

```bash
git add .python-version
git commit -m "chore: add .python-version for Python 3.13 pinning"
```

---

### Task 1: Run ruff format across entire codebase

**Files:**
- Modify: ALL `.py` files in `pipeline/`, `trading_app/`, `ui/`, `scripts/`, `tests/`
- Modify: `ruff.toml` (add `[format]` section)

**Step 1: Add format config to `ruff.toml`**

Append to `ruff.toml`:
```toml

[format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

**Step 2: Preview format changes**

Run:
```bash
ruff format --check pipeline/ trading_app/ ui/ scripts/ tests/ 2>&1 | tail -20
```

This shows which files WOULD be changed. Review the count.

**Step 3: Apply formatting**

Run:
```bash
ruff format pipeline/ trading_app/ ui/ scripts/ tests/
```

**Step 4: Verify no test breakage**

Run:
```bash
python -m pytest tests/ -x -q 2>&1 | tail -5
```

Expected: same pass count as before (formatting never changes behavior).

**Step 5: Verify ruff lint still passes**

Run:
```bash
ruff check pipeline/ trading_app/
```

Expected: no errors.

**Step 6: Commit**

```bash
git add -A
git commit -m "style: apply ruff format across entire codebase

One-time formatting pass. All future code follows this format.
No behavioral changes — formatting only."
```

---

### Task 2: Expand ruff lint rules (I, B, UP, SIM)

**Files:**
- Modify: `ruff.toml` (expand `select`, update `ignore`, update `per-file-ignores`)
- Modify: various `.py` files (auto-fixed violations)

**Step 1: Update `ruff.toml`**

Replace the full `ruff.toml` with:
```toml
# Ruff linter + formatter config — professional-grade rules
line-length = 120
target-version = "py313"

[lint]
# F = pyflakes, E = pycodestyle errors, W = pycodestyle warnings
# I = isort (import sorting), B = bugbear (real bug patterns)
# UP = pyupgrade (modernize to 3.13), SIM = simplify
select = ["F", "E", "W", "I", "B", "UP", "SIM"]
ignore = [
    "E402",   # module-level import not at top (required: sys.path.insert before imports)
    "E501",   # line too long (handled by line-length setting)
    "E731",   # lambda assignment (common pattern in this codebase)
    "UP015",  # redundant open mode (noisy, low value)
    "SIM108", # ternary instead of if/else (readability preference)
    "SIM117", # multiple with statements (readability preference)
]

[lint.per-file-ignores]
"tests/*" = ["F401", "F811"]  # unused imports + redefined names OK in tests (fixtures)
"scripts/*" = ["E402"]        # scripts often manipulate sys.path before imports
"research/*" = ["E402", "F401"]  # research scripts are experimental

[lint.isort]
known-first-party = ["pipeline", "trading_app", "ui"]

[format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

**Step 2: Run ruff check with auto-fix on safe rules**

Run:
```bash
ruff check --fix pipeline/ trading_app/ ui/ scripts/ 2>&1 | head -30
```

`I` (isort) and `UP` (pyupgrade) violations are auto-fixable. `B` (bugbear) violations require manual review.

**Step 3: Check remaining violations that need manual fixes**

Run:
```bash
ruff check pipeline/ trading_app/ ui/ scripts/ 2>&1
```

For any remaining `B` (bugbear) violations: review each one. Common fixes:
- `B006` mutable default argument: use `None` + `if arg is None: arg = []`
- `B007` unused loop variable: prefix with `_`
- `B904` raise without `from` in except: add `from err` or `from None`

Fix each manually.

**Step 4: Re-format after auto-fixes**

Run:
```bash
ruff format pipeline/ trading_app/ ui/ scripts/ tests/
```

Auto-fixes may have introduced formatting inconsistencies.

**Step 5: Run tests**

Run:
```bash
python -m pytest tests/ -x -q 2>&1 | tail -5
```

Expected: same pass count. `UP` and `I` changes are purely syntactic.

**Step 6: Run drift checks**

Run:
```bash
python pipeline/check_drift.py 2>&1 | tail -3
```

Expected: all checks pass.

**Step 7: Commit**

```bash
git add -A
git commit -m "refactor: expand ruff rules — isort, bugbear, pyupgrade, simplify

Added I (import sorting), B (bugbear — real bug patterns), UP (pyupgrade —
modernize to 3.13 syntax), SIM (simplify). Extended linting to scripts/.
All auto-fixable violations resolved. Manual bugbear fixes applied."
```

---

### Task 3: Consolidate dependencies into pyproject.toml + uv lock

**Files:**
- Modify: `pyproject.toml` (add `[project.dependencies]` and `[project.optional-dependencies]`)
- Regenerate: `uv.lock` (via `uv lock`)
- Keep: `requirements.txt` (as fallback until CI confirmed)

**Step 1: Update `pyproject.toml` with dependencies**

Add after the `requires-python` line (before `[tool.setuptools...]`):

```toml
dependencies = [
    "databento>=0.70.0",
    "duckdb>=1.4.4",
    "pandas>=2.3.3",
    "numpy>=2.2.6",
    "pyarrow>=23.0.0",
    "zstandard>=0.25.0",
    "arch>=8.0.0",
    "websockets>=14.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "pydantic>=2.0.0",
    "streamlit>=1.33.0",
    "plotly>=5.0.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.4",
    "pytest-cov>=6.1.1",
    "ruff>=0.11.6",
    "pyright>=1.1.400",
    "pip-audit>=2.7.0",
]
```

**Step 2: Regenerate uv.lock**

Run:
```bash
uv lock
```

This resolves all dependencies and writes exact versions to `uv.lock`.

**Step 3: Verify uv sync works**

Run:
```bash
uv sync --frozen 2>&1
```

Expected: installs all packages from lock file without re-resolving.

**Step 4: Verify tests still pass with uv-managed environment**

Run:
```bash
python -m pytest tests/ -x -q 2>&1 | tail -5
```

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: consolidate deps into pyproject.toml + uv lock

Moved all dependencies from requirements.txt to pyproject.toml.
Added dev dependencies (pytest, ruff, pyright, pip-audit).
Generated uv.lock for reproducible builds.
requirements.txt kept as fallback until CI migration confirmed."
```

---

### Task 4: Add pyright basic mode + pyrightconfig.json

**Files:**
- Create: `pyrightconfig.json`
- Reference: `pipeline/`, `trading_app/`, `ui/` (type check targets)

**Step 1: Create `pyrightconfig.json`**

```json
{
    "pythonVersion": "3.13",
    "typeCheckingMode": "basic",
    "include": [
        "pipeline",
        "trading_app",
        "ui"
    ],
    "exclude": [
        "**/__pycache__",
        "venv",
        ".venv",
        "research",
        "openclaw",
        "llm-code-scanner",
        ".auto-claude"
    ],
    "reportMissingModuleSource": "warning",
    "reportMissingTypeStubs": "warning",
    "reportUnusedImport": false,
    "reportUnusedVariable": false
}
```

Notes:
- `reportMissingModuleSource` and `reportMissingTypeStubs` are `warning` not `error` — duckdb/databento stubs are incomplete
- `reportUnusedImport`/`reportUnusedVariable` are `false` — ruff handles these (avoid duplicate errors)
- `scripts/` excluded initially — add later once `pipeline/` and `trading_app/` are clean
- `research/` always excluded (experimental)

**Step 2: Run pyright and assess baseline**

Run:
```bash
pyright 2>&1 | tail -20
```

Record the error count. This is informational — we will NOT require zero errors initially.

**Step 3: If error count is manageable (<50), fix critical errors**

Common pyright basic-mode errors that should be fixed:
- `reportGeneralClassIssues` — wrong number of arguments to class constructor
- `reportReturnType` — function returns wrong type
- `reportArgumentType` — wrong argument type passed

If >50 errors: skip fixes for now. The goal is to GET pyright running, not to fix everything.

**Step 4: Commit**

```bash
git add pyrightconfig.json
git commit -m "chore: add pyrightconfig.json — pyright basic mode

Type checking enabled in basic mode for pipeline/, trading_app/, ui/.
Missing stubs (duckdb, databento) set to warning, not error.
scripts/ and research/ excluded from type checking initially."
```

---

### Task 5: Add pip-audit + coverage threshold to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Update CI workflow**

Replace the full `.github/workflows/ci.yml` with:

```yaml
name: Pipeline CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  checks:
    runs-on: windows-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync --frozen

      - name: Ruff format check
        run: ruff format --check pipeline/ trading_app/ ui/ scripts/ tests/

      - name: Ruff lint check
        run: ruff check pipeline/ trading_app/ ui/ scripts/

      - name: Pyright type check
        run: pyright
        continue-on-error: true

      - name: Drift check (static analysis)
        run: python pipeline/check_drift.py

      - name: Behavioral audit
        run: python scripts/tools/audit_behavioral.py

      - name: Check file encoding (UTF-8)
        run: |
          python -c "
          from pathlib import Path
          bad = []
          for ext in ('*.md', '*.txt', '*.py'):
              for f in Path('.').rglob(ext):
                  if '.venv' in f.parts or 'openclaw' in f.parts or 'llm-code-scanner' in f.parts:
                      continue
                  try:
                      f.read_text(encoding='utf-8')
                  except (UnicodeDecodeError, ValueError):
                      bad.append(str(f))
          if bad:
              print('Non-UTF-8 files found:')
              for b in bad: print(f'  {b}')
              raise SystemExit(1)
          print('All tracked text files are UTF-8.')
          "

      - name: Security audit (advisory)
        run: pip-audit --desc on 2>&1 || true
        continue-on-error: true

      - name: Tests with coverage
        run: python -m pytest tests/ -v --ignore=tests/test_trader_logic.py --cov=pipeline --cov=trading_app --cov-report=term-missing --cov-fail-under=BASELINE
```

**IMPORTANT:** Replace `BASELINE` in the last line with the coverage number measured in Task 0, minus 3 (rounded down). For example, if measured coverage is 68%, set `--cov-fail-under=65`.

Notes:
- `pyright` runs with `continue-on-error: true` — advisory until baseline is clean
- `pip-audit` runs with `continue-on-error: true` — advisory until CVEs are cleared
- `ruff format --check` added before lint (catches unformatted code)
- `ruff check` scope extended to `scripts/`
- `-n auto --dist loadscope` removed from pytest (was causing issues with `--cov`)
- `uv sync --frozen` replaces `pip install -r requirements.txt`

**Step 2: Verify CI YAML syntax**

Run:
```bash
python -c "
import pathlib, re
content = pathlib.Path('.github/workflows/ci.yml').read_text()
# Basic YAML structure check
assert content.startswith('name:'), 'Missing name field'
assert 'jobs:' in content, 'Missing jobs field'
assert 'steps:' in content, 'Missing steps field'
print('CI YAML structure looks valid')
"
```

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add pyright, pip-audit, ruff format, coverage threshold

- Install deps via uv sync --frozen (reproducible builds)
- Added ruff format --check (formatting gate)
- Extended ruff check to scripts/
- Added pyright type check (advisory — continue-on-error)
- Added pip-audit security scan (advisory — continue-on-error)
- Added --cov-fail-under=BASELINE (coverage regression gate)"
```

---

### Task 6: Add .vscode/settings.json + editor config

**Files:**
- Create: `.vscode/settings.json`

**Step 1: Create `.vscode/settings.json`**

```json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingModuleSource": "warning",
        "reportMissingTypeStubs": "warning"
    },
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    },
    "ruff.lint.args": ["--config=ruff.toml"],
    "ruff.format.args": ["--config=ruff.toml"],
    "files.exclude": {
        "**/__pycache__": true,
        "venv": true,
        ".venv": true,
        "*.egg-info": true
    }
}
```

**Step 2: Check `.gitignore` does not exclude `.vscode/`**

Run:
```bash
grep -n "vscode" .gitignore 2>&1 || echo ".vscode not in .gitignore — good"
```

If `.vscode/` is in `.gitignore`, remove that line. We WANT settings.json committed.

**Step 3: Commit**

```bash
git add .vscode/settings.json
git commit -m "chore: add VS Code settings — ruff format-on-save, pyright basic

Editor config for consistent development experience:
- Ruff as default Python formatter with format-on-save
- Pyright basic mode matching pyrightconfig.json
- Import organization on save"
```

---

### Task 7: Add ruff format check to pre-commit hook

**Files:**
- Modify: `.githooks/pre-commit`

**Step 1: Add ruff format check after the lint check**

Insert after the ruff lint block (after `echo "  PASSED"` on line 57) and before the drift check (line 59). Also update all step numbering from `[0/4]` to `[0/6]` etc:

New step to insert:
```bash

# 1. Ruff format check (~1 second)
echo "[1/6] Ruff format..."
if ! "$RUFF" format --check pipeline/ trading_app/ ui/ scripts/ tests/ > /dev/null 2>&1; then
    echo "BLOCKED: Unformatted code detected. Run: ruff format pipeline/ trading_app/ ui/ scripts/ tests/"
    "$RUFF" format --check pipeline/ trading_app/ ui/ scripts/ tests/ 2>&1 | head -20
    exit 1
fi
echo "  PASSED"
```

Updated numbering for all steps:
- `[0/6]` Ruff lint
- `[1/6]` Ruff format
- `[2/6]` Drift check
- `[3/6]` Fast tests
- `[4/6]` Behavioral audit
- `[5/6]` Syntax check
- `[6/6]` M2.5 scan

**Step 2: Verify hook runs under 30 seconds**

Run:
```bash
time bash .githooks/pre-commit 2>&1 | tail -10
```

Expected: <30 seconds total. `ruff format --check` adds <1 second.

**Step 3: Commit**

```bash
git add .githooks/pre-commit
git commit -m "chore: add ruff format check to pre-commit hook

Added format gate before lint. Hook now enforces both formatting
and lint rules. Step numbering updated to [0/6] through [6/6]."
```

---

### Task 8: Add drift checks for tooling config consistency

**Files:**
- Modify: `pipeline/check_drift.py`

**Step 1: Add tooling drift check functions**

Add these check functions before the `CHECKS` list at the bottom of `check_drift.py`:

```python
def check_pyright_config_exists(project_root: Path) -> list[str]:
    """Ensure pyrightconfig.json exists and has basic mode."""
    config_path = project_root / "pyrightconfig.json"
    if not config_path.exists():
        return ["pyrightconfig.json missing — type checking not configured"]
    import json
    config = json.loads(config_path.read_text())
    mode = config.get("typeCheckingMode", "off")
    if mode not in ("basic", "standard", "strict"):
        return [f"pyrightconfig.json typeCheckingMode={mode}, expected basic/standard/strict"]
    return []


def check_ruff_rules_minimum(project_root: Path) -> list[str]:
    """Ensure ruff.toml has minimum required rule sets."""
    ruff_path = project_root / "ruff.toml"
    if not ruff_path.exists():
        return ["ruff.toml missing"]
    content = ruff_path.read_text()
    required = ["I", "B", "UP"]
    missing = [r for r in required if f'"{r}"' not in content]
    if missing:
        return [f"ruff.toml missing required rule sets: {missing}"]
    return []


def check_python_version_file(project_root: Path) -> list[str]:
    """Ensure .python-version exists and matches pyproject.toml."""
    pv_path = project_root / ".python-version"
    if not pv_path.exists():
        return [".python-version file missing"]
    version = pv_path.read_text().strip()
    if not version.startswith("3.13"):
        return [f".python-version says {version}, expected 3.13"]
    return []


def check_uv_lock_exists(project_root: Path) -> list[str]:
    """Ensure uv.lock exists and is not a skeleton."""
    lock_path = project_root / "uv.lock"
    if not lock_path.exists():
        return ["uv.lock missing — run 'uv lock' to generate"]
    content = lock_path.read_text()
    if content.count("[[package]]") < 5:
        return ["uv.lock appears to be a skeleton — run 'uv lock' to regenerate"]
    return []
```

Then add entries to the `CHECKS` list:

```python
    ("pyrightconfig.json exists with basic+ mode",
     lambda: check_pyright_config_exists(PROJECT_ROOT), False, False),
    ("ruff.toml has minimum required rules (I, B, UP)",
     lambda: check_ruff_rules_minimum(PROJECT_ROOT), False, False),
    (".python-version file exists and matches 3.13",
     lambda: check_python_version_file(PROJECT_ROOT), False, False),
    ("uv.lock exists and is not a skeleton",
     lambda: check_uv_lock_exists(PROJECT_ROOT), False, False),
```

**Step 2: Run drift checks to verify new checks pass**

Run:
```bash
python pipeline/check_drift.py 2>&1 | tail -10
```

Expected: all checks pass (including the 4 new ones).

**Step 3: Commit**

```bash
git add pipeline/check_drift.py
git commit -m "chore: add drift checks for tooling config consistency

4 new checks:
- pyrightconfig.json exists with basic+ mode
- ruff.toml has minimum required rules (I, B, UP)
- .python-version file exists and matches 3.13
- uv.lock exists and is not a skeleton"
```

---

### Task 9: Final verification + downstream updates

**Files:**
- Modify: `CLAUDE.md` (update Key Commands section)
- Modify: `requirements.txt` (add deprecation notice)
- Run: `python scripts/tools/gen_repo_map.py` (regenerate REPO_MAP)

**Step 1: Run full drift checks**

Run:
```bash
python pipeline/check_drift.py 2>&1
```

Expected: all checks pass (including new tooling checks).

**Step 2: Run full test suite**

Run:
```bash
python -m pytest tests/ -x -q 2>&1 | tail -10
```

Expected: same pass count as before (only the ML memory test may fail — pre-existing).

**Step 3: Run pyright and record baseline**

Run:
```bash
pyright 2>&1 | tail -10
```

Record error count for memory file.

**Step 4: Run ruff (lint + format)**

Run:
```bash
ruff check pipeline/ trading_app/ ui/ scripts/ 2>&1
ruff format --check pipeline/ trading_app/ ui/ scripts/ tests/ 2>&1
```

Expected: zero violations on both.

**Step 5: Install and run pip-audit**

Run:
```bash
pip install pip-audit && pip-audit --desc on 2>&1
```

Record any CVEs — advisory only.

**Step 6: Time the pre-commit hook**

Run:
```bash
time bash .githooks/pre-commit 2>&1 | tail -5
```

Expected: <30 seconds.

**Step 7: Verify uv sync**

Run:
```bash
uv sync --frozen 2>&1
```

Expected: clean install from lock file.

**Step 8: Add deprecation notice to requirements.txt**

Add to top of `requirements.txt`:
```
# DEPRECATED — dependencies now managed in pyproject.toml
# This file is kept as fallback only. Use: uv sync --frozen
# To update: uv lock && uv export > requirements.txt
```

**Step 9: Update CLAUDE.md Key Commands**

Add to the Key Commands section:
```bash
# Tooling
ruff format pipeline/ trading_app/ ui/ scripts/ tests/  # Format all code
ruff check pipeline/ trading_app/ ui/ scripts/           # Lint all code
ruff check --fix pipeline/ trading_app/ ui/ scripts/     # Auto-fix lint issues
pyright                                                   # Type check (basic mode)
uv sync --frozen                                          # Install from lock file
uv lock                                                   # Regenerate lock file
pip-audit --desc on                                       # Security scan
```

**Step 10: Regenerate REPO_MAP**

Run:
```bash
python scripts/tools/gen_repo_map.py
```

**Step 11: Create memory file for tooling decisions**

Record pyright baseline errors, coverage baseline, key decisions in a new memory topic file.

**Step 12: Commit everything**

```bash
git add -A
git commit -m "chore: tooling hardening complete — downstream updates

- CLAUDE.md updated with new tooling commands
- requirements.txt marked as deprecated
- REPO_MAP.md regenerated
- All tools verified: pyright, ruff, uv, pip-audit, coverage"
```

---

## Post-Implementation Downstream Checklist

- [ ] `CLAUDE.md` Key Commands updated with tooling commands
- [ ] `REPO_MAP.md` regenerated
- [ ] `.gitignore` verified: `.vscode/` NOT excluded
- [ ] `requirements.txt` marked deprecated
- [ ] Pinecone assistant synced (`python scripts/tools/sync_pinecone.py`)
- [ ] Memory file created: `tooling_hardening.md` with pyright baseline, coverage baseline, decisions
