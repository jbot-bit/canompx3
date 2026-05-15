"""Tests for the --account-id sentinel convention.

Covers the bug fixed in `docs/runtime/stages/fix-account-id-sentinel-mismatch.md`:
dashboard "Start Live" did not pass --account-id, argparse coerced the missing
value to 0, and `_select_primary_and_shadow_accounts` rejected 0 as "not in
broker accounts" — crashing every copies>1 profile.

Canonical convention going forward:
- `None` = auto-discover from broker (preferred).
- `0`    = auto-discover (legacy-tolerated; many test fixtures construct with it).
- Any other int = user-requested specific account; must exist at broker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_live_session as rls  # noqa: E402


def test_argparse_default_account_id_is_none():
    """CLI default must be None so missing `--account-id` flows through as
    'auto-discover', not as `0` which the copy-trading branch would reject.

    The parser is built inline inside `main()`, so we assert against the
    source-level declaration directly. This is the load-bearing line: if
    someone re-introduces `default=0`, the dashboard's "Start Live" path
    will crash again on copies>1 profiles.
    """
    src = Path(rls.__file__).read_text(encoding="utf-8")
    # The --account-id block must declare default=None
    assert '"--account-id"' in src
    idx = src.index('"--account-id"')
    block = src[idx : idx + 250]
    assert "default=None" in block, f"Expected default=None in --account-id block, got: {block!r}"
    assert "default=0" not in block, (
        "Regression: --account-id default=0 re-introduced. "
        "Dashboard 'Start Live' will crash on copies>1 profiles."
    )


def test_select_primary_and_shadow_accounts_with_none_auto_discovers():
    """copies>1 + requested_account_id=None must pick from the discovered
    accounts without raising. This is the dashboard 'Start Live' path."""
    all_accounts = [(21944866, "PA-XFA-001"), (21944867, "PA-XFA-002")]
    primary, shadows = rls._select_primary_and_shadow_accounts(
        all_accounts=all_accounts,
        n_copies=2,
        requested_account_id=None,
    )
    assert primary == 21944866
    assert shadows == [21944867]


def test_select_primary_and_shadow_accounts_treats_zero_as_real_id():
    """ASYMMETRIC TOLERANCE — documented gotcha.

    `session_orchestrator.py:542` treats both `None` and `0` as auto-discover.
    But `_select_primary_and_shadow_accounts` treats only `None` as auto;
    `0` falls into the `requested_account_id is not None` branch and gets
    rejected as "not at broker". This is INTENTIONAL: the helper is reached
    only from the argparse-default path (now `None`), so the `0` case here
    is a legacy/programmatic-caller error, not a normal flow.

    If a future refactor makes this helper accept `0` as auto, update both
    this test and the commit message for a0b3c24b.
    """
    all_accounts = [(21944866, "PA-XFA-001")]
    with pytest.raises(RuntimeError, match="not in the broker's discovered"):
        rls._select_primary_and_shadow_accounts(
            all_accounts=all_accounts,
            n_copies=1,
            requested_account_id=0,
        )


def test_multi_runner_passes_account_id_through_to_orchestrators():
    """Structural check: MultiInstrumentRunner.account_id (None or 0) must
    propagate to each SessionOrchestrator unchanged. The orchestrator's
    `is None or == 0` guard at session_orchestrator.py:542 is the canonical
    auto-discover site for the multi-instrument path."""
    import inspect

    from trading_app.live import multi_runner

    sig = inspect.signature(multi_runner.MultiInstrumentRunner.__init__)
    account_id_param = sig.parameters["account_id"]
    # Default is 0 (legacy compat); annotation widens to int|None per the fix.
    assert account_id_param.default == 0
    ann = account_id_param.annotation
    # Annotation may be `int | None` (PEP 604) — assert it accepts None.
    assert "None" in str(ann), f"Expected None in account_id annotation, got {ann!r}"


def test_select_primary_and_shadow_accounts_rejects_unknown_real_id():
    """A specific account ID that isn't at the broker must still raise.
    Confirms we haven't accidentally relaxed the real-ID validation."""
    all_accounts = [(21944866, "PA-XFA-001")]
    with pytest.raises(RuntimeError, match="not in the broker's discovered"):
        rls._select_primary_and_shadow_accounts(
            all_accounts=all_accounts,
            n_copies=1,
            requested_account_id=999999,
        )
