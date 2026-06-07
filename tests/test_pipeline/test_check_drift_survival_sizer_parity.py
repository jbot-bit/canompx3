"""Drift check: survival sim and live engine must import ONE canonical sizer.

D-3 seam Stage 1 (2026-06-07). The account-survival gate now sizes each trade
like the live execution engine. Both paths MUST resolve contracts through the
same canonical helper (``compute_position_size_vol_scaled`` from
``trading_app.portfolio``). If a future edit re-implements sizing in one path,
the gate would prove DD at a contract count the engine never trades — a silent
fork. This check fails closed on that fork.
"""

import textwrap
from pathlib import Path

from pipeline.check_drift import check_survival_engine_sizer_parity


def test_check_survival_engine_sizer_parity_passes_on_clean_tree():
    violations = check_survival_engine_sizer_parity()
    assert violations == [], violations


def test_check_survival_engine_sizer_parity_detects_reencoded_sizer(tmp_path: Path):
    """Inject a module that does NOT import the canonical sizer → must flag."""
    survival = tmp_path / "account_survival.py"
    engine = tmp_path / "execution_engine.py"
    # survival imports canonically; engine re-encodes (no canonical import).
    survival.write_text(
        textwrap.dedent(
            """
            from trading_app.portfolio import compute_position_size_vol_scaled
            def f():
                return compute_position_size_vol_scaled(1, 2, 3, None, 1.0)
            """
        ),
        encoding="utf-8",
    )
    engine.write_text(
        textwrap.dedent(
            """
            def compute_position_size_vol_scaled(*a):  # re-encoded fork, NOT canonical
                return 1
            """
        ),
        encoding="utf-8",
    )
    violations = check_survival_engine_sizer_parity(survival_path=survival, engine_path=engine)
    assert violations, "must detect the engine re-encoding the sizer"
    assert any("execution_engine" in v for v in violations)


def test_check_survival_engine_sizer_parity_fails_closed_on_missing_file(tmp_path: Path):
    """A missing source file means parity is unprovable → fail closed."""
    violations = check_survival_engine_sizer_parity(
        survival_path=tmp_path / "nope.py", engine_path=tmp_path / "also_nope.py"
    )
    assert violations, "missing files must fail closed, not silently pass"
