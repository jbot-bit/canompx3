"""Tests for `scripts.run_live_session._select_primary_and_shadow_accounts`.

Bug context (2026-04-25): pre-fix, the inline selection in `main()` sliced
`all_accounts[:n_copies]` BEFORE checking `args.account_id` membership. With
`profile.copies=2` and the user wanting the 3rd-discovered XFA (account at
index 2), the membership check fell through and the code routed to
`account_ids[0]` — the WRONG account. Live session ran on a TC instead of
the XFA, no error, no warning.

Audit gap closure: previous fix had no test (it shipped under Ralph 174's
absorbed commit). These probes pin both the validation behavior and the
move-to-front guarantee.
"""

import pytest


def _accounts():
    """Mirror the user's broker layout from 2026-04-25 02:00 Brisbane:
    2 TC accounts + 1 XFA. The XFA (21944866) is the user's intended
    primary for the middle-stage live runs.
    """
    return [
        (20859313, "50KTC-V2-451890-20372221"),
        (21390438, "50KTC-V2-451890-67605663"),
        (21944866, "EXPRESS-V2-451890-53179846"),
    ]


def test_account_id_in_third_position_is_routed_as_primary_when_copies_is_2():
    """The original bug: copies=2, user wants index-2 account, slice excludes
    it. Verify the fix routes the user's choice as primary regardless of
    discovery order or slice horizon.
    """
    from scripts.run_live_session import _select_primary_and_shadow_accounts

    primary, shadows = _select_primary_and_shadow_accounts(
        all_accounts=_accounts(),
        n_copies=2,
        requested_account_id=21944866,
    )
    assert primary == 21944866, "User-specified XFA must be primary"
    assert shadows == [20859313], "Shadow must be the next account, not the unwanted XFA"


def test_account_id_not_in_discovered_raises_runtime_error():
    """Audit-mandated hard-fail: pre-fix this silently routed to accounts[0].
    Now must raise RuntimeError so the operator sees the wrong account ID
    instead of trading on the wrong account silently.
    """
    from scripts.run_live_session import _select_primary_and_shadow_accounts

    with pytest.raises(RuntimeError, match="not in the broker's discovered accounts"):
        _select_primary_and_shadow_accounts(
            all_accounts=_accounts(),
            n_copies=2,
            requested_account_id=99999999,  # nonexistent
        )


def test_no_account_id_routes_first_discovered_as_primary():
    """Backward-compat: when the operator does NOT pass --account-id, the first
    discovered account becomes primary (legacy auto-discover behavior).
    """
    from scripts.run_live_session import _select_primary_and_shadow_accounts

    primary, shadows = _select_primary_and_shadow_accounts(
        all_accounts=_accounts(),
        n_copies=2,
        requested_account_id=None,
    )
    assert primary == 20859313, "Auto-discover picks the first account"
    assert shadows == [21390438]


def test_account_id_at_index_1_picked_with_copies_2():
    """User-specified account already inside the slice horizon — must still be
    primary, not shadow. Tests the move-to-front idempotency."""
    from scripts.run_live_session import _select_primary_and_shadow_accounts

    primary, shadows = _select_primary_and_shadow_accounts(
        all_accounts=_accounts(),
        n_copies=2,
        requested_account_id=21390438,
    )
    assert primary == 21390438
    assert shadows == [20859313]


def test_copies_equals_total_accounts_uses_all_as_shadows():
    """copies=3, 3 accounts available, user picks the XFA → all other accounts
    become shadows. Validates the upper bound."""
    from scripts.run_live_session import _select_primary_and_shadow_accounts

    primary, shadows = _select_primary_and_shadow_accounts(
        all_accounts=_accounts(),
        n_copies=3,
        requested_account_id=21944866,
    )
    assert primary == 21944866
    assert sorted(shadows) == [20859313, 21390438], "All non-primary accounts become shadows when copies == total"


def test_copies_greater_than_accounts_truncates_silently():
    """copies=5, 3 accounts available — caller already log.warning'd; this
    helper must not raise. Slice naturally truncates to len(all_accounts)."""
    from scripts.run_live_session import _select_primary_and_shadow_accounts

    primary, shadows = _select_primary_and_shadow_accounts(
        all_accounts=_accounts(),
        n_copies=5,  # more than available
        requested_account_id=None,
    )
    assert primary == 20859313
    assert sorted(shadows) == [21390438, 21944866]


def test_single_account_no_shadows_with_account_id():
    """Single discovered account + user picks it → no shadows."""
    from scripts.run_live_session import _select_primary_and_shadow_accounts

    primary, shadows = _select_primary_and_shadow_accounts(
        all_accounts=[(21944866, "EXPRESS-V2")],
        n_copies=2,
        requested_account_id=21944866,
    )
    assert primary == 21944866
    assert shadows is None
