"""Mutation-proof tests for ``check_code_fingerprint_registries_pinned``.

Capital review (2026-06-07) blast scan: the cached-state code-fingerprint
staleness pattern is a CLASS OF FOUR (`_criterion11_code_paths`,
`_lane_code_paths`, `_sr_code_paths`, `_overlay_code_paths`). None was pinned by
a drift check, so a new live-risk file added OUTSIDE a registry list would let a
stale cached PASS survive — the capital-critical case being the C11 survival
gate. This check pins each registry's FLOOR entry count; these tests prove the
check actually catches a shrink (integrity-guardian.md § 7 — never trust a check
without injection).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.check_drift import (
    _CODE_FINGERPRINT_REGISTRIES,
    check_code_fingerprint_registries_pinned,
)


def _write_registry(root: Path, rel_path: str, fn_name: str, n_entries: int) -> None:
    """Write a fake module defining ``fn_name`` returning ``n_entries`` paths."""
    module = root / rel_path
    module.parent.mkdir(parents=True, exist_ok=True)
    entries = "\n".join(f"        Path('f{i}.py')," for i in range(n_entries))
    module.write_text(
        f"from pathlib import Path\n\n\ndef {fn_name}():\n    return [\n{entries}\n    ]\n",
        encoding="utf-8",
    )


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A fake project tree with every registry at EXACTLY its floor count."""
    for rel_path, fn_name, floor in _CODE_FINGERPRINT_REGISTRIES:
        _write_registry(tmp_path, rel_path, fn_name, floor)
    return tmp_path


def test_real_tree_passes() -> None:
    """The REAL repo registries all meet their floors → zero violations."""
    violations = check_code_fingerprint_registries_pinned()
    assert violations == [], violations


def test_fake_tree_at_floor_passes(fake_root: Path) -> None:
    """Each registry at exactly its floor count → passes."""
    violations = check_code_fingerprint_registries_pinned(project_root=fake_root)
    assert violations == [], violations


def test_shrunk_registry_fails(fake_root: Path) -> None:
    """Mutation probe: removing a file from the C11 registry triggers the check."""
    rel_path, fn_name, floor = _CODE_FINGERPRINT_REGISTRIES[0]
    assert fn_name == "_criterion11_code_paths"  # the capital-critical one
    _write_registry(fake_root, rel_path, fn_name, floor - 1)
    violations = check_code_fingerprint_registries_pinned(project_root=fake_root)
    assert len(violations) == 1, violations
    assert fn_name in violations[0]
    assert "below the recorded floor" in violations[0]


def test_missing_function_fails(fake_root: Path) -> None:
    """Mutation probe: renaming/removing a registry function fails closed."""
    rel_path, fn_name, _floor = _CODE_FINGERPRINT_REGISTRIES[1]
    (fake_root / rel_path).write_text(
        "from pathlib import Path\n\n\ndef _renamed_away():\n    return [Path('x.py')]\n",
        encoding="utf-8",
    )
    violations = check_code_fingerprint_registries_pinned(project_root=fake_root)
    assert any(fn_name in v and "no longer defines" in v for v in violations), violations


def test_missing_module_fails(fake_root: Path) -> None:
    """Mutation probe: a deleted registry module fails closed."""
    rel_path, _fn, _floor = _CODE_FINGERPRINT_REGISTRIES[2]
    (fake_root / rel_path).unlink()
    violations = check_code_fingerprint_registries_pinned(project_root=fake_root)
    assert any(rel_path in v and "missing module" in v for v in violations), violations


def test_non_list_return_fails(fake_root: Path) -> None:
    """Mutation probe: a registry that stops returning a list literal fails closed."""
    rel_path, fn_name, _floor = _CODE_FINGERPRINT_REGISTRIES[3]
    (fake_root / rel_path).write_text(
        f"from pathlib import Path\n\n\ndef {fn_name}(spec):\n    return some_dynamic_thing()\n",
        encoding="utf-8",
    )
    violations = check_code_fingerprint_registries_pinned(project_root=fake_root)
    assert any(fn_name in v and "no longer returns a list literal" in v for v in violations), violations
