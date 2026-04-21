"""Targeted tests for live-session preflight helpers."""

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "run_live_session.py"
SPEC = importlib.util.spec_from_file_location("run_live_session_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_live_session = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_live_session)


class _FakeAuth:
    pass


class _FakeContracts:
    def __init__(self, auth, demo):
        self.auth = auth
        self.demo = demo


class _CleanPositions:
    def __init__(self, auth):
        self.auth = auth

    def query_account_metadata(self, account_id: int):
        return {"name": "EXPRESS-53179846", "canTrade": True}

    def query_open(self, account_id: int):
        return []


class _OrphanedPositions(_CleanPositions):
    def query_open(self, account_id: int):
        return [{"contract_id": "MNQ", "side": "long", "size": 1}]


def _components(positions_class):
    return {
        "auth": _FakeAuth(),
        "contracts_class": _FakeContracts,
        "positions_class": positions_class,
    }


def test_account_binding_check_resolves_suffix_and_reports_clean_state(monkeypatch):
    monkeypatch.setattr(
        run_live_session,
        "_select_broker_account",
        lambda contracts, explicit_account_id=0, account_suffix=None: (21944866, "EXPRESS-V2-451890-53179846"),
    )

    result = run_live_session._run_account_binding_check(
        _components(_CleanPositions),
        demo=False,
        account_suffix="846",
    )

    assert result == {
        "account_id": 21944866,
        "account_name": "EXPRESS-V2-451890-53179846",
        "can_trade": True,
        "orphans": 0,
    }


def test_account_binding_check_fails_closed_when_orphans_exist(monkeypatch):
    monkeypatch.setattr(
        run_live_session,
        "_select_broker_account",
        lambda contracts, explicit_account_id=0, account_suffix=None: (21944866, "EXPRESS-V2-451890-53179846"),
    )

    with pytest.raises(RuntimeError, match="open position"):
        run_live_session._run_account_binding_check(
            _components(_OrphanedPositions),
            demo=False,
            account_suffix="846",
        )
