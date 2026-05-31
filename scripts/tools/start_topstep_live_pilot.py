#!/usr/bin/env python3
"""Operator-safe launcher for the Topstep MNQ single-copy live pilot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LivePilotConfig:
    profile_id: str = "topstep_50k_mnq_auto"
    instrument: str = "MNQ"
    copies: int = 1


@dataclass(frozen=True)
class CommandStep:
    name: str
    argv: tuple[str, ...]


DEFAULT_CONFIG = LivePilotConfig()


def _py(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)


def build_gate_steps(config: LivePilotConfig, *, refresh_control_state: bool = True) -> list[CommandStep]:
    steps: list[CommandStep] = []
    if refresh_control_state:
        steps.append(
            CommandStep(
                "Refresh C11/C12 control state",
                _py("scripts/tools/refresh_control_state.py", "--profile", config.profile_id),
            )
        )
    steps.extend(
        [
            CommandStep(
                "Strict live readiness",
                _py(
                    "scripts/tools/live_readiness_report.py",
                    "--profile",
                    config.profile_id,
                    "--copies",
                    str(config.copies),
                    "--strict-zero-warn",
                ),
            ),
            CommandStep(
                "Live-session preflight",
                _py(
                    "-m",
                    "scripts.run_live_session",
                    "--profile",
                    config.profile_id,
                    "--instrument",
                    config.instrument,
                    "--live",
                    "--copies",
                    str(config.copies),
                    "--preflight",
                ),
            ),
        ]
    )
    return steps


def build_launch_step(config: LivePilotConfig) -> CommandStep:
    return CommandStep(
        "Launch live pilot",
        _py(
            "-m",
            "scripts.run_live_session",
            "--profile",
            config.profile_id,
            "--instrument",
            config.instrument,
            "--live",
            "--copies",
            str(config.copies),
        ),
    )


def _format_command(argv: tuple[str, ...]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in argv)


def _run_step(step: CommandStep) -> int:
    print(f"\n==> {step.name}", flush=True)
    print(_format_command(step.argv), flush=True)
    result = subprocess.run(step.argv, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nBLOCKED: {step.name} failed with exit code {result.returncode}.")
    return result.returncode


def run_live_pilot(
    *,
    config: LivePilotConfig = DEFAULT_CONFIG,
    refresh_control_state: bool = True,
    preflight_only: bool = False,
    dry_run: bool = False,
) -> int:
    gate_steps = build_gate_steps(config, refresh_control_state=refresh_control_state)
    launch_step = build_launch_step(config)

    if dry_run:
        print("Topstep MNQ live pilot command plan:")
        for step in [*gate_steps, launch_step]:
            print(f"- {step.name}: {_format_command(step.argv)}")
        return 0

    print("Topstep MNQ live pilot launcher")
    print(f"Profile: {config.profile_id}")
    print(f"Instrument: {config.instrument}")
    print(f"Copies: {config.copies}")
    print("Mode: LIVE after gates pass; canonical runner still requires typing CONFIRM.")

    for step in gate_steps:
        rc = _run_step(step)
        if rc != 0:
            return rc

    print("\nGATES GREEN for the single-copy pilot.")
    if preflight_only:
        print("Preflight-only mode requested; not launching live session.")
        return 0

    print("Handing off to the canonical live runner. Type CONFIRM there to arm real-money orders.")
    return _run_step(launch_step)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Topstep MNQ single-copy live-pilot gate and launcher")
    parser.add_argument("--dry-run", action="store_true", help="Print the command plan without running anything")
    parser.add_argument("--preflight-only", action="store_true", help="Run gates and stop before live launch")
    parser.add_argument("--skip-control-refresh", action="store_true", help="Do not refresh C11/C12 before gates")
    args = parser.parse_args(argv)

    return run_live_pilot(
        refresh_control_state=not args.skip_control_refresh,
        preflight_only=args.preflight_only,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
