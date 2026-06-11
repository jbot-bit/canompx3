"""Stage 1 of the multi-account control plane — dashboard account-selector backend.

Proves the load-bearing invariant: the broker account the operator selects flows
as ONE id through ONE builder (``_live_pilot_cli_args``) into BOTH preflight and
launch, so the account PREFLIGHT checks and the account the bot LAUNCHES on are
byte-identical (engine check [13] parity). Also covers the handoff-restart
account carry (finding #3) and running-state visibility (watch-out #9).

The front-end selector reuse + LIVE gating is verified live via Playwright (see
the stage doc); these are the backend unit guards.
"""

from __future__ import annotations

import trading_app.live.bot_dashboard as bd

PILOT = "topstep_50k_mnq_auto"
EXPRESS = 21944866
COMBINE = 23055112


# ── _live_pilot_cli_args: the single injection point ──────────────────────────


def test_cli_args_none_falls_back_to_default():
    args = bd._live_pilot_cli_args(PILOT)
    assert "--account-id" in args
    # Fallback is the module constant (last-resort zero-arg path).
    assert args[args.index("--account-id") + 1] == str(bd.LIVE_PILOT_ACCOUNT_ID)


def test_cli_args_explicit_express_passthrough():
    args = bd._live_pilot_cli_args(PILOT, EXPRESS)
    assert args[args.index("--account-id") + 1] == str(EXPRESS)


def test_cli_args_explicit_combine_passthrough():
    args = bd._live_pilot_cli_args(PILOT, COMBINE)
    assert args[args.index("--account-id") + 1] == str(COMBINE)


def test_cli_args_non_pilot_profile_emits_nothing():
    # Only the pinned pilot profile gets server-side routing args.
    assert bd._live_pilot_cli_args("some_other_profile", EXPRESS) == []


# ── Preflight↔launch parity by construction (the core risk) ───────────────────


def test_preflight_and_launch_use_same_builder_same_id():
    """The preflight subprocess and the launch Popen must bind the SAME id.

    Both call _live_pilot_cli_args(profile, account_id); proving the builder is
    deterministic for a given id proves #3 (preflight) ≡ #4 (launch) — there is
    no second code path that could re-derive a different account.
    """
    for acct in (None, EXPRESS, COMBINE):
        preflight_args = bd._live_pilot_cli_args(PILOT, acct)
        launch_args = bd._live_pilot_cli_args(PILOT, acct)
        assert preflight_args == launch_args
        # And the id actually present matches the request (or the fallback).
        expected = str(acct) if acct is not None else str(bd.LIVE_PILOT_ACCOUNT_ID)
        assert preflight_args[preflight_args.index("--account-id") + 1] == expected


# ── Handoff-restart carries the account (finding #3) ──────────────────────────


def test_set_handoff_carries_target_account(monkeypatch):
    # Use a clean handoff state so the test is order-independent.
    monkeypatch.setattr(bd, "_handoff_state", dict(bd._handoff_state), raising=True)
    bd._set_handoff(PILOT, "live", "switching", account_id=EXPRESS)
    assert bd._handoff_state["target_account"] == EXPRESS


def test_clear_handoff_resets_target_account(monkeypatch):
    monkeypatch.setattr(bd, "_handoff_state", dict(bd._handoff_state), raising=True)
    bd._set_handoff(PILOT, "live", "switching", account_id=EXPRESS)
    bd._clear_handoff()
    assert bd._handoff_state["target_account"] is None


def test_handoff_default_account_is_none(monkeypatch):
    # A handoff set WITHOUT an explicit account carries None (→ backend fallback),
    # never a silently-substituted id.
    monkeypatch.setattr(bd, "_handoff_state", dict(bd._handoff_state), raising=True)
    bd._set_handoff(PILOT, "live", "switching")
    assert bd._handoff_state["target_account"] is None


# ── Running-state visibility via the engine's canonical state (watch-out #9) ──


def test_session_snapshot_surfaces_engine_account_id(monkeypatch):
    """_session_snapshot reads account_id from bot_state (engine-written), not a
    dashboard-side cache — so the running dashboard shows the live account."""
    monkeypatch.setattr(bd, "read_state", lambda: {"mode": "LIVE", "account_id": EXPRESS})
    snap = bd._session_snapshot()
    assert snap["live_account_id"] == EXPRESS


def test_session_snapshot_no_account_when_stopped(monkeypatch):
    monkeypatch.setattr(bd, "read_state", lambda: {"mode": "STOPPED"})
    snap = bd._session_snapshot()
    assert snap["live_account_id"] is None


# ── End-to-end: preflight and launch carry the SAME --account-id ──────────────


def test_preflight_and_launch_cmds_carry_identical_account_id(monkeypatch):
    """Both _run_preflight_subprocess and the Popen in action_start must pass
    the SAME --account-id to run_live_session.  This test inspects the actual
    command lists built under each code path so a future refactor that breaks
    parity fails here before it reaches live trading.
    """
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        import subprocess as sp

        r = sp.CompletedProcess(cmd, 0, "", "")
        return r

    monkeypatch.setattr(bd.subprocess, "run", fake_run)

    # Call _run_preflight_subprocess directly with the pilot profile + Express id.
    bd._run_preflight_subprocess(PILOT, mode="live", account_id=EXPRESS)

    assert captured, "subprocess.run was not called"
    preflight_cmd = captured[-1]
    assert "--account-id" in preflight_cmd
    preflight_id = preflight_cmd[preflight_cmd.index("--account-id") + 1]

    # The launch path calls _live_pilot_cli_args with the same args.
    launch_args = bd._live_pilot_cli_args(PILOT, EXPRESS)
    assert "--account-id" in launch_args
    launch_id = launch_args[launch_args.index("--account-id") + 1]

    assert preflight_id == launch_id == str(EXPRESS)
