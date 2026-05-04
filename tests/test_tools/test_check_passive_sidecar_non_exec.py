from pathlib import Path

from scripts import check_passive_sidecar_non_exec


def test_non_exec_scanner_passes_clean_file(tmp_path: Path) -> None:
    pkg = tmp_path / "passive_sidecar"
    pkg.mkdir()
    file_path = pkg / "clean.py"
    file_path.write_text("def ok() -> None:\n    return None\n", encoding="utf-8")

    violations = check_passive_sidecar_non_exec.scan_paths([pkg])

    assert violations == []


def test_non_exec_scanner_fails_for_forbidden_import(tmp_path: Path) -> None:
    pkg = tmp_path / "passive_sidecar"
    pkg.mkdir()
    file_path = pkg / "bad.py"
    file_path.write_text("from trading_app.live.projectx.order_router import ProjectXOrderRouter\n", encoding="utf-8")

    violations = check_passive_sidecar_non_exec.scan_paths([pkg])

    assert violations
    assert "ProjectXOrderRouter" in violations[0]


def test_non_exec_scanner_fails_for_forbidden_endpoint(tmp_path: Path) -> None:
    pkg = tmp_path / "passive_sidecar"
    pkg.mkdir()
    file_path = pkg / "bad_endpoint.py"
    file_path.write_text('ENDPOINT = "/api/Order/place"\n', encoding="utf-8")

    violations = check_passive_sidecar_non_exec.scan_paths([pkg])

    assert violations
    assert "/api/Order/place" in violations[0]


def test_non_exec_scanner_fails_for_module_import_path(tmp_path: Path) -> None:
    pkg = tmp_path / "passive_sidecar"
    pkg.mkdir()
    file_path = pkg / "bad_module.py"
    file_path.write_text(
        "import trading_app.live.projectx.order_router as order_router\n",
        encoding="utf-8",
    )

    violations = check_passive_sidecar_non_exec.scan_paths([pkg])

    assert violations
    assert "trading_app.live.projectx.order_router" in violations[0]
