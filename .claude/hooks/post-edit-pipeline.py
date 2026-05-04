#!/usr/bin/env python3
"""Post-edit hook: runs drift check + targeted tests after pipeline/trading_app file edits."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_DEBOUNCE_FILE = Path(__file__).parent / ".last_drift_ok"
_DEBOUNCE_SECONDS = 30
# Resolve project root for subprocess cwd and PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# PYTHONIOENCODING=utf-8 prevents Windows cp1252 UnicodeEncodeError on check labels
# containing non-ASCII chars (e.g. arrows, ellipsis). subprocess.run(capture_output=True)
# pipes via the system locale by default, which on Windows is cp1252.
_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": str(_PROJECT_ROOT),
    "PYTHONIOENCODING": "utf-8",
}


def _resolve_python() -> str:
    """Prefer project `.venv` Python over `sys.executable`.

    The hook is wired in .claude/settings.json as `python <hookscript>`, which
    resolves to the system Python — often lacking project deps like `anthropic`
    (see Stages 1-4 of claude-api-modernization). Using `.venv/Scripts/python`
    when present guarantees drift checks see the same deps the tests do.
    Falls back to `sys.executable` if no venv is available.
    """
    venv_win = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_win.exists():
        return str(venv_win)
    venv_unix = _PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_unix.exists():
        return str(venv_unix)
    return sys.executable


_HOOK_PYTHON = _resolve_python()

# Map edited modules to their test files for targeted testing
TEST_MAP = {
    "pipeline/build_daily_features.py": "tests/test_pipeline/test_build_daily_features.py",
    "pipeline/build_bars_5m.py": "tests/test_pipeline/test_build_bars_5m.py",
    "pipeline/ingest_dbn.py": "tests/test_pipeline/test_ingest.py",
    "pipeline/check_drift.py": "tests/test_pipeline/test_check_drift.py",
    "pipeline/dst.py": "tests/test_pipeline/test_dst.py",
    "pipeline/init_db.py": "tests/test_pipeline/test_schema.py",
    "pipeline/asset_configs.py": "tests/test_pipeline/test_asset_configs.py",
    "trading_app/outcome_builder.py": "tests/test_trading_app/test_outcome_builder.py",
    "trading_app/strategy_discovery.py": "tests/test_trading_app/test_strategy_discovery.py",
    "trading_app/strategy_validator.py": "tests/test_trading_app/test_strategy_validator.py",
    "trading_app/entry_rules.py": "tests/test_trading_app/test_entry_rules.py",
    "trading_app/paper_trader.py": "tests/test_trading_app/test_paper_trader.py",
    "trading_app/config.py": "tests/test_trading_app/test_config.py",
    "trading_app/prop_portfolio.py": "tests/test_trading_app/test_prop_portfolio.py",
    "trading_app/prop_profiles.py": "tests/test_trading_app/test_prop_profiles.py",
    "trading_app/pre_session_check.py": "tests/test_trading_app/test_pre_session_check.py",
    "trading_app/lane_ctl.py": "tests/test_trading_app/test_lane_ctl.py",
    "trading_app/live/trade_journal.py": "tests/test_trading_app/test_trade_journal.py",
    "trading_app/live/session_orchestrator.py": "tests/test_trading_app/test_session_orchestrator.py",
    "trading_app/live/projectx/order_router.py": "tests/test_trading_app/test_projectx_429_retry.py",
    "trading_app/execution_engine.py": "tests/test_trading_app/test_execution_engine.py",
    "trading_app/strategy_fitness.py": "tests/test_trading_app/test_strategy_fitness.py",
    "trading_app/live_config.py": "tests/test_trading_app/test_live_config.py",
    "trading_app/walkforward.py": "tests/test_trading_app/test_walkforward.py",
    "trading_app/risk_manager.py": "tests/test_trading_app/test_risk_manager.py",
    "trading_app/setup_detector.py": "tests/test_trading_app/test_setup_detector.py",
    "trading_app/mcp_server.py": "tests/test_trading_app/test_mcp_server.py",
    "pipeline/cost_model.py": "tests/test_pipeline/test_cost_model.py",
    "pipeline/health_check.py": "tests/test_pipeline/test_health_check.py",
    "scripts/tools/audit_behavioral.py": "tests/test_tools/test_audit_behavioral.py",
}


def normalize_path(p):
    """Normalize to forward-slash relative path for matching."""
    fwd = p.replace("\\", "/")
    # Strip project root prefix to get relative path
    project_root = str(_PROJECT_ROOT).replace("\\", "/")
    if fwd.startswith(project_root):
        return fwd[len(project_root) :].lstrip("/")
    # Fallback: split on last known directory marker
    for marker in ("pipeline/", "trading_app/", "scripts/"):
        idx = fwd.rfind(marker)
        if idx >= 0:
            return fwd[idx:]
    return fwd


def _crg_canonical_root() -> Path:
    """Resolve the canonical CRG repo root for this worktree.

    Mirrors `.githooks/pre-commit` lines 222–225 (sibling-detection):
    if this worktree is `<parent>/canompx3-<descriptor>` and the canonical
    sibling `<parent>/canompx3` exists with a `.code-review-graph/` dir,
    return the sibling so all worktrees share ONE full graph rather than
    fragmenting into per-worktree partials. Otherwise return _PROJECT_ROOT
    (this worktree).

    Source-of-truth: `.githooks/pre-commit` step 3b. If that logic changes,
    update both sites.
    """
    name = _PROJECT_ROOT.name
    if name.startswith("canompx3-"):
        sibling = _PROJECT_ROOT.parent / "canompx3"
        if (sibling / ".code-review-graph").exists():
            return sibling
    return _PROJECT_ROOT


def _crg_update(file_path: str) -> None:
    """Run code-review-graph update after edits to canonical paths.

    Timeout 5s; fail-silent. Per spec F3: PostToolUse auto-update.
    Covers pipeline/, trading_app/, scripts/, research/, tests/.

    Pins CRG_REPO_ROOT to the canonical sibling so worktree edits refresh
    the single shared 1052-file graph instead of a 4-file fragment.
    """
    norm = normalize_path(file_path)
    _CRG_PREFIXES = ("pipeline/", "trading_app/", "scripts/", "research/", "tests/")
    if not any(norm.startswith(pfx) for pfx in _CRG_PREFIXES):
        return
    canonical_root = _crg_canonical_root()
    env = {**os.environ, "CRG_REPO_ROOT": str(canonical_root)}
    try:
        subprocess.run(
            ["code-review-graph", "update", "--skip-flows"],
            cwd=str(canonical_root),
            capture_output=True,
            timeout=5,
            check=False,
            env=env,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass  # fail-silent per spec


def main():
    input_data = json.load(sys.stdin)
    file_path = input_data.get("tool_input", {}).get("file_path", "")

    # CRG incremental update fires for the broader prefix set (pipeline/, trading_app/,
    # scripts/, research/, tests/) per spec F3. Must run BEFORE the pipeline|trading_app
    # early-exit below — otherwise edits to scripts/research/tests/ silently skip CRG
    # updates despite _crg_update declaring those prefixes. Fail-silent; <5s cost.
    _crg_update(file_path)

    # Only run drift/tests/behavioral-audit for pipeline or trading_app Python files
    if not (("pipeline" in file_path or "trading_app" in file_path) and file_path.endswith(".py")):
        sys.exit(0)

    # --- Phase 1: Drift check (FAST tier, ~3-5s) with 30s debounce ---
    # Pre-commit hook + CI run the FULL drift check (no `--fast`) for end-to-end coverage.
    # Hook uses `--fast` to skip the 19 checks measured >0.3s by profile_check_drift.py
    # (full check is 50-130s — far exceeds the 30s hook timeout).
    _skip_drift = False
    if _DEBOUNCE_FILE.exists():
        try:
            age = time.time() - _DEBOUNCE_FILE.stat().st_mtime
            if age < _DEBOUNCE_SECONDS:
                _skip_drift = True
        except OSError:
            pass  # race condition — file deleted between exists() and stat()

    if not _skip_drift:
        try:
            result = subprocess.run(
                [_HOOK_PYTHON, str(_PROJECT_ROOT / "pipeline" / "check_drift.py"), "--fast"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(_PROJECT_ROOT),
                env=_SUBPROCESS_ENV,
            )
        except subprocess.TimeoutExpired:
            # Fast tier should never exceed 30s. If it does, surface it loudly —
            # it means SLOW_CHECK_LABELS in pipeline/check_drift.py is out of date.
            _DEBOUNCE_FILE.unlink(missing_ok=True)
            print(
                f"DRIFT CHECK FAST TIER TIMED OUT (>30s) for {file_path}.\n"
                f"  Re-profile with: python -m scripts.tools.profile_check_drift\n"
                f"  Add new slow checks to SLOW_CHECK_LABELS in pipeline/check_drift.py.",
                file=sys.stderr,
            )
            sys.exit(2)
        if result.returncode != 0:
            # Invalidate debounce on failure
            _DEBOUNCE_FILE.unlink(missing_ok=True)
            print(f"DRIFT DETECTED after editing {file_path}", file=sys.stderr)
            # Emit ONLY the failed check blocks + the final summary line, not the
            # full PASSED/ADVISORY listing (~100 lines = ~8k tokens per hook fire).
            # Parses `check_drift.py --fast` output: each check is
            #   "Check N: <label>..."
            #   "  PASSED [OK]" | "  FAILED:" (with details lines following until blank)
            stdout_lines = result.stdout.splitlines()
            emit: list[str] = []
            i = 0
            while i < len(stdout_lines):
                line = stdout_lines[i]
                if line.startswith("Check ") and line.rstrip().endswith("..."):
                    # Look ahead: is next non-empty line "  FAILED:"?
                    j = i + 1
                    while j < len(stdout_lines) and not stdout_lines[j].strip():
                        j += 1
                    if j < len(stdout_lines) and stdout_lines[j].strip().startswith("FAILED:"):
                        # Emit from the header through the end of this block
                        # (next blank line, or next "Check ", whichever comes first)
                        emit.append(line)
                        k = i + 1
                        while k < len(stdout_lines):
                            nxt = stdout_lines[k]
                            if nxt.startswith("Check "):
                                break
                            emit.append(nxt)
                            k += 1
                        i = k
                        continue
                # Always keep the separator + final summary (last ~5 lines)
                if line.startswith("====") or line.startswith("DRIFT ") or line.startswith("NO DRIFT"):
                    emit.append(line)
                i += 1
            filtered = "\n".join(emit).strip()
            print(filtered if filtered else result.stdout, file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            sys.exit(2)
        # Mark successful drift check
        _DEBOUNCE_FILE.touch()

    # --- Phase 2: Targeted tests (~10-20s) ---
    norm = normalize_path(file_path)
    test_file = TEST_MAP.get(norm)

    if test_file and os.path.exists(test_file):
        try:
            result = subprocess.run(
                [
                    _HOOK_PYTHON,
                    "-m",
                    "pytest",
                    test_file,
                    "-x",
                    "-q",
                    "--no-header",
                    "--tb=short",
                    "-k",
                    "not Integration and not integration and not Idempotent",
                ],
                capture_output=True,
                text=True,
                timeout=45,
                cwd=str(_PROJECT_ROOT),
                env=_SUBPROCESS_ENV,
            )
            if result.returncode != 0:
                print(f"TESTS FAILED after editing {file_path}", file=sys.stderr)
                print(f"Test file: {test_file}", file=sys.stderr)
                # Cap stdout at last 80 lines: pytest's failure summary lives at
                # the tail (FAILED block + final counts). Full stdout on a wide
                # failure can be 100s of lines = thousands of context tokens
                # per hook fire. User can re-run pytest manually for full output.
                stdout_lines = result.stdout.splitlines()
                if len(stdout_lines) > 80:
                    print(f"... ({len(stdout_lines) - 80} earlier lines truncated)", file=sys.stderr)
                    print("\n".join(stdout_lines[-80:]), file=sys.stderr)
                else:
                    print(result.stdout, file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                sys.exit(2)
        except subprocess.TimeoutExpired:
            # Don't block on slow integration tests — pre-commit will catch them
            pass

    # --- Phase 3: Behavioral audit (only for pipeline/ and scripts/tools/ edits, ~1s) ---
    if "pipeline/" in norm or "scripts/tools/" in norm:
        try:
            result = subprocess.run(
                [_HOOK_PYTHON, str(_PROJECT_ROOT / "scripts" / "tools" / "audit_behavioral.py")],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(_PROJECT_ROOT),
                env=_SUBPROCESS_ENV,
            )
            if result.returncode != 0:
                print(f"BEHAVIORAL AUDIT FAILED after editing {file_path}", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
                sys.exit(2)
        except subprocess.TimeoutExpired:
            pass  # Don't block on timeout

    # --- Phase 4: CRG incremental update (background, fail-silent, ~1-5s) ---
    _crg_update(file_path)

    sys.exit(0)


if __name__ == "__main__":
    main()
