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
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}

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


def main():
    input_data = json.load(sys.stdin)
    file_path = input_data.get("tool_input", {}).get("file_path", "")

    # Only run for pipeline or trading_app Python files
    if not (("pipeline" in file_path or "trading_app" in file_path) and file_path.endswith(".py")):
        sys.exit(0)

    # --- Phase 1: Drift check (fast, ~2s) with 30s debounce ---
    # Pre-commit hook still runs full drift check (last line of defense)
    _skip_drift = False
    if _DEBOUNCE_FILE.exists():
        try:
            age = time.time() - _DEBOUNCE_FILE.stat().st_mtime
            if age < _DEBOUNCE_SECONDS:
                _skip_drift = True
        except OSError:
            pass  # race condition — file deleted between exists() and stat()

    if not _skip_drift:
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "pipeline" / "check_drift.py")],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_PROJECT_ROOT),
            env=_SUBPROCESS_ENV,
        )
        if result.returncode != 0:
            # Invalidate debounce on failure
            _DEBOUNCE_FILE.unlink(missing_ok=True)
            print(f"DRIFT DETECTED after editing {file_path}", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
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
                    sys.executable,
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
                [sys.executable, str(_PROJECT_ROOT / "scripts" / "tools" / "audit_behavioral.py")],
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

    sys.exit(0)


if __name__ == "__main__":
    main()
