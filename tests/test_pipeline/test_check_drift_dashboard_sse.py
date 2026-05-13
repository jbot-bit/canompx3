"""Tests for cockpit-v3 Stage 2.1 drift checks:

- check_dashboard_localhost_only_binding: refuses 0.0.0.0 / public binds
- check_dashboard_sse_single_worker: refuses uvicorn.run workers>1

Both checks are mutation-proven: pass on canonical file, fail loudly on a
synthetic mutation. This is the test pattern from the C1 kill-switch race
postmortem — assertions on green-path only let bugs ship; mutation probes
prove the check actually detects the violation it claims to detect.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.check_drift import (
    check_dashboard_localhost_only_binding,
    check_dashboard_sse_single_worker,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRADING_APP_DIR = REPO_ROOT / "trading_app"


def test_dashboard_localhost_check_passes_on_canonical():
    assert check_dashboard_localhost_only_binding(TRADING_APP_DIR) == []


def test_dashboard_localhost_check_tolerates_missing_file(tmp_path):
    """A path with no trading_app/live/bot_dashboard.py must not raise."""
    assert check_dashboard_localhost_only_binding(tmp_path) == []


def test_dashboard_localhost_check_catches_public_default(tmp_path):
    """Mutation: argparse default → 0.0.0.0 must trip the check."""
    live = tmp_path / "live"
    live.mkdir()
    (live / "bot_dashboard.py").write_text(
        'def run_dashboard(host: str = "0.0.0.0", port: int = 8080) -> None:\n'
        '    if host not in {"127.0.0.1", "localhost", "::1"}:\n'
        '        raise RuntimeError("Refusing to start dashboard on non-localhost host")\n'
        "    uvicorn.run(app, host=host, port=port, workers=1)\n"
        '    parser.add_argument("--host", default="127.0.0.1")\n',
        encoding="utf-8",
    )
    violations = check_dashboard_localhost_only_binding(tmp_path)
    assert any("0.0.0.0" in v for v in violations), violations


def test_dashboard_localhost_check_catches_missing_guard(tmp_path):
    """Mutation: RuntimeError guard removed → must trip the check."""
    live = tmp_path / "live"
    live.mkdir()
    (live / "bot_dashboard.py").write_text(
        'def run_dashboard(host: str = "127.0.0.1", port: int = 8080) -> None:\n'
        "    uvicorn.run(app, host=host, port=port, workers=1)\n"
        '    parser.add_argument("--host", default="127.0.0.1")\n',
        encoding="utf-8",
    )
    violations = check_dashboard_localhost_only_binding(tmp_path)
    assert any("RuntimeError guard" in v for v in violations), violations


def test_dashboard_localhost_check_catches_missing_signature(tmp_path):
    """Mutation: run_dashboard signature deleted → must trip the check."""
    live = tmp_path / "live"
    live.mkdir()
    (live / "bot_dashboard.py").write_text(
        "# No run_dashboard function at all\n"
        'uvicorn.run(app, host="127.0.0.1", port=8080, workers=1)\n'
        "Refusing to start dashboard on non-localhost host  # in a comment\n",
        encoding="utf-8",
    )
    violations = check_dashboard_localhost_only_binding(tmp_path)
    assert any("not found" in v for v in violations), violations


def test_dashboard_sse_workers_check_passes_on_canonical():
    assert check_dashboard_sse_single_worker(TRADING_APP_DIR) == []


def test_dashboard_sse_workers_check_tolerates_missing_file(tmp_path):
    assert check_dashboard_sse_single_worker(tmp_path) == []


def test_dashboard_sse_workers_check_catches_multi_worker(tmp_path):
    """Mutation: workers=4 must trip the check."""
    live = tmp_path / "live"
    live.mkdir()
    (live / "bot_dashboard.py").write_text(
        'uvicorn.run(app, host="127.0.0.1", port=8080, workers=4)\n',
        encoding="utf-8",
    )
    violations = check_dashboard_sse_single_worker(tmp_path)
    assert any("workers=4" in v for v in violations), violations


def test_dashboard_sse_workers_check_catches_missing_workers_arg(tmp_path):
    """Mutation: workers= argument removed → must trip the check."""
    live = tmp_path / "live"
    live.mkdir()
    (live / "bot_dashboard.py").write_text(
        'uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")\n',
        encoding="utf-8",
    )
    violations = check_dashboard_sse_single_worker(tmp_path)
    assert any("missing explicit workers=1" in v for v in violations), violations
