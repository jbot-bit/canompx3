import ast
import re
from pathlib import Path


def _extract_codex_bat_modes(codex_bat: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r'set "MODE=([^"]+)"', codex_bat)
        if match.group(1).startswith("codex-project") or match.group(1).startswith("green-")
    }


def _extract_powershell_validate_set(script_text: str) -> set[str]:
    match = re.search(r"\[ValidateSet\((.*?)\)\]", script_text, re.DOTALL)
    assert match, "windows-agent-launch.ps1 is missing a ValidateSet declaration"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def _extract_python_valid_modes(script_text: str) -> set[str]:
    module = ast.parse(script_text)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "VALID_MODES":
                return {
                    elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }
    raise AssertionError("windows_agent_launch.py is missing VALID_MODES")


def test_wsl_launcher_scripts_call_mount_guard() -> None:
    root = Path(__file__).resolve().parents[2]
    shared_home = (root / "scripts" / "infra" / "codex_shared_home.sh").read_text(encoding="utf-8")
    sticky = (root / "scripts" / "infra" / "windows-sticky-launch.ps1").read_text(encoding="utf-8")
    project = (root / "scripts" / "infra" / "codex-project.sh").read_text(encoding="utf-8")
    search = (root / "scripts" / "infra" / "codex-project-search.sh").read_text(encoding="utf-8")
    review = (root / "scripts" / "infra" / "codex-review.sh").read_text(encoding="utf-8")
    capital_review = (root / "scripts" / "infra" / "codex-capital-review.sh").read_text(encoding="utf-8")
    worktree = (root / "scripts" / "infra" / "codex-worktree.sh").read_text(encoding="utf-8")
    sync_guard = (root / "scripts" / "infra" / "codex-wsl-sync.sh").read_text(encoding="utf-8")

    assert "setup_shared_codex_home()" in shared_home
    assert "resolve_local_codex_home()" in shared_home
    assert "append_codex_profile_arg()" in shared_home
    assert 'source "$SHARED_CODEX_HOME_HELPER"' in project
    assert "setup_shared_codex_home" in project
    assert 'append_codex_profile_arg "$PROFILE" CODEX_ARGS' in project
    assert 'wsl_mount_guard.py" --root "$ROOT"' in project
    assert 'PROFILE="${CANOMPX3_CODEX_PROFILE:-canompx3}"' in project
    assert 'source "$SHARED_CODEX_HOME_HELPER"' in search
    assert "setup_shared_codex_home" in search
    assert 'append_codex_profile_arg "$PROFILE" CODEX_ARGS' in search
    assert 'wsl_mount_guard.py" --root "$ROOT"' in search
    assert "task_route_packet.py" in project
    assert "task_route_packet.py" in search
    assert 'CANOMPX3_SESSION_OWNER="pid:$$"' in project
    assert 'CANOMPX3_SESSION_OWNER="pid:$$"' in search
    assert 'source "$SHARED_CODEX_HOME_HELPER"' in review
    assert "setup_shared_codex_home" in review
    assert 'append_codex_profile_arg "$PROFILE" CODEX_ARGS' in review
    assert 'PROFILE="${CANOMPX3_CODEX_PROFILE:-canompx3_power}"' in review
    assert 'PROFILE="${CANOMPX3_CODEX_PROFILE:-canompx3_power}"' in capital_review
    assert 'wsl_mount_guard.py" --root "$ROOT"' in review
    assert 'python3 "$ROOT/scripts/tools/wsl_mount_guard.py" --root "$ROOT"' in worktree
    assert "task_route_packet.py" in worktree
    assert '--related-root "$SOURCE_ROOT"' in sync_guard
    assert "--claim codex" in sync_guard
    assert "--mode mutating" in sync_guard
    assert '$quickExitExemptModes = @("doctor", "cleanup")' in sticky
    assert "$Mode -notin $quickExitExemptModes" in sticky
    assert "suspiciousQuickExit" in sticky
    assert "Start-Process powershell.exe" in sticky
    assert '"-File", $launcherPs1' in sticky
    assert "& powershell.exe @launcherArgs" in sticky
    assert "$interactiveHoldModes" in sticky
    assert "$Mode -in $interactiveHoldModes" in sticky
    assert "Codex session exited with code $exitCode." in sticky


def test_wsl_sync_dirty_clone_error_is_operator_actionable() -> None:
    root = Path(__file__).resolve().parents[2]
    sync_guard = (root / "scripts" / "infra" / "codex-wsl-sync.sh").read_text(encoding="utf-8")

    assert "WSL Codex repo has uncommitted changes" in sync_guard
    assert "MEASURED: dirty WSL Codex clone" in sync_guard
    assert "This is a fail-closed guard, not a Codex install failure." in sync_guard
    assert "cd ~/canompx3" in sync_guard
    assert "git status --short --branch" in sync_guard
    assert "codex.bat task <name>" in sync_guard
    assert "Microsoft WSL and OpenAI Codex both recommend keeping Linux-tool repos under /home" in sync_guard


def test_wsl_sync_is_churn_aware_and_discards_handoff_before_ffmerge() -> None:
    """The sync guard must (a) classify dirt via the canonical churn predicate so
    routine operational churn (live_journal.db / HANDOFF.md) no longer false-blocks,
    and (b) discard the regenerable WSL-side churn stamp BEFORE the ff-merge so a
    restamped HANDOFF.md cannot abort the fast-forward. Mirrors the live-arm drift
    gate's churn-ignore intent (run_live_session.py _check_repo_clean)."""
    root = Path(__file__).resolve().parents[2]
    sync_guard = (root / "scripts" / "infra" / "codex-wsl-sync.sh").read_text(encoding="utf-8")

    # (a) The dirty-check routes porcelain through the canonical churn CLI, not a
    #     raw `status --short` block. Reuse, not a re-encoded path list.
    assert "scripts/tools/_worktree_churn.py" in sync_guard, (
        "dirty-check must reuse the canonical churn predicate, not re-encode the list"
    )
    assert 'target_dirty="$(material_dirt "$TARGET_ROOT")"' in sync_guard, (
        "target_dirty must come from the churn-aware filter, not raw status --short"
    )
    # The churn filter must FAIL CLOSED to raw porcelain if the predicate is
    # unreachable (a tooling gap must never silently wave dirt past the guard).
    assert "printf '%s' \"$porcelain\"" in sync_guard

    # (b) Between the ancestor check and the ff-merge, the guard discards tracked
    #     churn so the fast-forward is not aborted by a restamped HANDOFF.md.
    discard_idx = sync_guard.index('checkout -- "$churn_path"')
    ffmerge_idx = sync_guard.index("merge --ff-only --quiet FETCH_HEAD")
    assert discard_idx < ffmerge_idx, "churn discard must precede the ff-merge"
    assert "diff --name-only" in sync_guard, "discard must target tracked-modified paths"

    # The new error header must name the ignored churn so the operator understands
    # WHY a previously-blocking state now passes (or what real dirt remains).
    assert "operational churn such as live_journal.db / HANDOFF.md is ignored" in sync_guard


def test_operator_docs_explain_wsl_home_launcher_recovery() -> None:
    root = Path(__file__).resolve().parents[2]
    handbook = (root / "docs" / "reference" / "codex-operator-handbook.md").read_text(encoding="utf-8")
    setup = (root / "docs" / "reference" / "codex-claude-operator-setup.md").read_text(encoding="utf-8")

    assert "Observed failure pattern" in handbook
    assert "cd ~/canompx3" in handbook
    assert "git status --short --branch" in handbook
    assert "codex.bat task <name>" in handbook
    assert "Microsoft Learn: Working across Windows and Linux file systems" in setup
    assert "OpenAI Codex Windows guide" in setup


def test_codex_bat_modes_are_supported_by_windows_launchers() -> None:
    root = Path(__file__).resolve().parents[2]
    codex_bat = (root / "codex.bat").read_text(encoding="utf-8")
    ps_launcher = (root / "scripts" / "infra" / "windows-agent-launch.ps1").read_text(encoding="utf-8")
    py_launcher = (root / "scripts" / "infra" / "windows_agent_launch.py").read_text(encoding="utf-8")

    codex_modes = _extract_codex_bat_modes(codex_bat)
    powershell_modes = _extract_powershell_validate_set(ps_launcher)
    python_modes = _extract_python_valid_modes(py_launcher)

    missing_in_powershell = codex_modes - powershell_modes
    missing_in_python = codex_modes - python_modes

    assert not missing_in_powershell, f"codex.bat exposes unsupported PowerShell modes: {sorted(missing_in_powershell)}"
    assert not missing_in_python, f"codex.bat exposes unsupported Python launcher modes: {sorted(missing_in_python)}"


def test_linux_modes_route_to_wsl_home_clone() -> None:
    root = Path(__file__).resolve().parents[2]
    py_launcher = (root / "scripts" / "infra" / "windows_agent_launch.py").read_text(encoding="utf-8")

    assert py_launcher.count("use_linux_home=True") >= 2


def test_start_bot_prints_checkout_identity_before_launching() -> None:
    root = Path(__file__).resolve().parents[2]
    start_bot = (root / "START_BOT.bat").read_text(encoding="utf-8")

    banner = start_bot.index("This shortcut runs the Windows checkout above")
    launch = start_bot.index(".venv\\Scripts\\python.exe -m trading_app.live.bot_dashboard")

    assert banner < launch
    assert "WSL/Codex branch pushes do not change this app until merged or pulled here" in start_bot
    assert "git branch --show-current" not in start_bot
    assert "git rev-list --left-right --count" not in start_bot


def test_start_bot_is_signal_only_dashboard_entrypoint() -> None:
    root = Path(__file__).resolve().parents[2]
    start_bot = (root / "START_BOT.bat").read_text(encoding="utf-8")

    # Safety invariant (the real thing to protect): the DEFAULT mode is
    # signal-only — a bare launch with no positional arg never arms capital.
    # The default-assignment must precede any demo/live override.
    default_idx = start_bot.index("set BOT_MODE_FLAGS=--signal-only")
    assert default_idx >= 0

    # demo/live modes exist (84f1ba3f — front-end arming) but are gated behind an
    # EXPLICIT positional arg (`%~1`), not the default. Assert that gating, rather
    # than forbidding the strings outright (which broke when the feature landed).
    for marker in ('if /i "%~1"=="demo"', 'if /i "%~1"=="live"'):
        assert marker in start_bot, f"expected mode arg gated behind explicit %~1: {marker}"
        assert start_bot.index(marker) > default_idx, "default signal-only must precede mode overrides"

    # No naked `set BOT_MODE_FLAGS=--demo`/`--live` assignment outside the %~1 gate
    # (those would change the default and arm capital on a bare launch).
    for line in start_bot.splitlines():
        stripped = line.strip()
        if stripped.startswith("set BOT_MODE_FLAGS=") and ("--demo" in stripped or "--live" in stripped):
            raise AssertionError(f"ungated capital-mode default assignment: {stripped!r}")

    assert "--source START_BOT.bat --copies 1 --instrument MNQ" in start_bot
    assert not (root / "START_LIVE_PILOT.bat").exists()


def test_start_bot_live_runs_blocking_strict_preflight_before_orchestrator() -> None:
    """Direct `START_BOT.bat live` must run the strict-zero-warn preflight and
    BLOCK before launching the orchestrator, closing the direct-live bypass of
    the dashboard's mode=live warn-blocking arm guard (parity fix 2026-06-08)."""
    root = Path(__file__).resolve().parents[2]
    start_bot = (root / "START_BOT.bat").read_text(encoding="utf-8")

    # The strict preflight invocation must exist and carry --strict-zero-warn.
    assert "--live --preflight --strict-zero-warn" in start_bot, (
        "live direct launch must run the strict-zero-warn preflight"
    )

    preflight_idx = start_bot.index("--live --preflight --strict-zero-warn")
    orchestrator_idx = start_bot.index('start "ORB Orchestrator')
    # Preflight must run BEFORE the orchestrator process is started.
    assert preflight_idx < orchestrator_idx, "strict preflight must precede orchestrator launch"

    # It must be gated to live-only (not run for signal/demo) and block on failure.
    live_block = start_bot[:orchestrator_idx]
    assert 'if /i "%BOT_MODE_FLAGS%"=="--live"' in live_block, "preflight must be gated to --live"
    assert "errorlevel 1" in start_bot[preflight_idx:orchestrator_idx], "must check preflight exit code"
    assert "exit /b 1" in start_bot[preflight_idx:orchestrator_idx], "must abort launch on preflight failure"
