#!/usr/bin/env python3
"""Front door for pre-registered research runs.

Purpose:
- restore an easy operator surface for the fast prereg -> run -> verify loop
- make the pipeline branch explicit from the hypothesis file itself
- stop operators from guessing whether a prereg writes to
  ``experimental_strategies`` or remains a bounded docs-only / runner-only
  study

This tool is intentionally conservative:
- ``standalone_edge`` preregs route to ``trading_app.strategy_discovery``
- ``conditional_role`` preregs are treated as bounded research unless an
  explicit runner is provided

It does NOT turn bounded research into generic discovery and it does NOT
promote anything automatically. Promotion remains:
``experimental_strategies`` -> ``strategy_validator`` -> ``validated_setups``
with deployment and ``paper_trades`` as optional later gates.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _preferred_repo_python() -> Path | None:
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv-wsl" / "bin" / "python"
    return candidate if candidate.exists() else None


def _preferred_repo_prefix(expected_python: Path) -> Path:
    return expected_python.parent.parent.resolve()


def _ensure_repo_python() -> None:
    if "pytest" in sys.modules:
        return
    expected_python = _preferred_repo_python()
    if expected_python is None:
        return
    current_prefix = Path(sys.prefix).resolve()
    expected_prefix = _preferred_repo_prefix(expected_python)
    if current_prefix == expected_prefix or os.environ.get("CANOMPX3_BOOTSTRAP_DONE") == "1":
        return

    env = os.environ.copy()
    env["CANOMPX3_BOOTSTRAP_DONE"] = "1"
    env.setdefault("CANOMPX3_BOOTSTRAPPED_FROM", str(Path(sys.executable).resolve()))
    raise SystemExit(
        subprocess.call(
            [str(expected_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    )


_ensure_repo_python()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from trading_app.hypothesis_loader import load_hypothesis_metadata  # noqa: E402


@dataclass(frozen=True)
class RouteDecision:
    hypothesis_file: str
    research_question_type: str
    execution_mode: str
    holdout_date: str
    instruments_declared: list[str]
    orb_minutes_declared: list[int]
    instrument: str | None
    orb_minutes: int | None
    writes_to: list[str]
    next_surface: str
    execution_entrypoint: str | None
    notes: list[str]


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _unique_sorted(values: list[Any]) -> list[Any]:
    try:
        return sorted(set(values))
    except TypeError:
        seen: list[Any] = []
        for value in values:
            if value not in seen:
                seen.append(value)
        return seen


def _collect_scope_values(hypotheses: list[dict[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for hypothesis in hypotheses:
        scope = hypothesis.get("scope", {})
        if not isinstance(scope, dict):
            continue
        values.extend(_coerce_list(scope.get(key)))
    return _unique_sorted(values)


def _infer_single(values: list[Any], override: Any | None) -> Any | None:
    if override is not None:
        return override
    if len(values) == 1:
        return values[0]
    return None


def build_route_decision(
    hypothesis_file: Path,
    *,
    instrument_override: str | None = None,
    orb_minutes_override: int | None = None,
    runner_override: str | None = None,
) -> RouteDecision:
    meta = load_hypothesis_metadata(hypothesis_file)
    body = yaml.safe_load(hypothesis_file.read_text(encoding="utf-8"))
    metadata = body.get("metadata", {}) if isinstance(body, dict) else {}
    hypotheses = body.get("hypotheses", []) if isinstance(body, dict) else []
    execution = body.get("execution", {}) if isinstance(body, dict) else {}
    if not isinstance(execution, dict):
        execution = {}

    research_question_type = str(metadata.get("research_question_type", "standalone_edge"))
    instruments = [str(v) for v in _collect_scope_values(hypotheses, "instruments")]
    orb_minutes_values = [int(v) for v in _collect_scope_values(hypotheses, "orb_minutes")]

    instrument = _infer_single(instruments, instrument_override)
    orb_minutes = _infer_single(orb_minutes_values, orb_minutes_override)

    notes: list[str] = []
    execution_mode = str(execution.get("mode") or "")
    if research_question_type == "standalone_edge":
        if not execution_mode:
            execution_mode = "grid_discovery"
        writes_to = ["experimental_strategies"]
        next_surface = (
            "experimental_strategies -> strategy_validator -> validated_setups "
            "(optional deployment/profile routing -> optional paper_trades)"
        )
        if instrument is None:
            notes.append("Instrument is ambiguous; pass --instrument to execute.")
        if orb_minutes is None:
            notes.append("ORB aperture is ambiguous or missing; pass --orb-minutes to execute.")
        if len(instruments) > 1 or len(orb_minutes_values) > 1:
            notes.append(
                "Multi-slice prereg: execute once per declared instrument/aperture slice. "
                "The single-use gate is scoped by hypothesis SHA + instrument + orb_minutes."
            )
    elif research_question_type == "conditional_role":
        if not execution_mode:
            execution_mode = "bounded_runner"
        writes_to = ["docs/audit/results (or bounded research artifact only)"]
        next_surface = "bounded result doc -> explicit role decision -> optional translation stage (no auto-promotion)"
        notes.append("Conditional-role preregs do not auto-write to experimental_strategies.")
    else:
        execution_mode = execution_mode or "unknown"
        writes_to = ["UNSUPPORTED"]
        next_surface = "UNSUPPORTED"
        notes.append(f"Unsupported research_question_type={research_question_type!r}.")

    entrypoint = runner_override or execution.get("entrypoint")
    if isinstance(entrypoint, Path):
        entrypoint = str(entrypoint)

    if research_question_type == "conditional_role" and not entrypoint:
        notes.append("No bounded runner specified. Use --runner or add execution.entrypoint to the prereg.")

    return RouteDecision(
        hypothesis_file=str(hypothesis_file),
        research_question_type=research_question_type,
        execution_mode=execution_mode,
        holdout_date=meta["holdout_date"].isoformat(),
        instruments_declared=instruments,
        orb_minutes_declared=orb_minutes_values,
        instrument=instrument,
        orb_minutes=orb_minutes,
        writes_to=writes_to,
        next_surface=next_surface,
        execution_entrypoint=str(entrypoint) if entrypoint else None,
        notes=notes,
    )


def _repo_python() -> str:
    preferred = _preferred_repo_python()
    return str(preferred) if preferred is not None else sys.executable


def _text_render(route: RouteDecision) -> str:
    lines = [
        "Pre-reg route",
        f"  file: {route.hypothesis_file}",
        f"  type: {route.research_question_type}",
        f"  mode: {route.execution_mode}",
        f"  holdout: {route.holdout_date}",
        f"  declared instruments: {route.instruments_declared or ['UNSPECIFIED']}",
        f"  declared orb_minutes: {route.orb_minutes_declared or ['UNSPECIFIED']}",
        f"  resolved instrument: {route.instrument or 'REQUIRED_AT_EXECUTION'}",
        f"  resolved orb_minutes: {route.orb_minutes if route.orb_minutes is not None else 'REQUIRED_AT_EXECUTION'}",
        f"  writes to: {', '.join(route.writes_to)}",
        f"  next: {route.next_surface}",
    ]
    if route.execution_entrypoint:
        lines.append(f"  runner: {route.execution_entrypoint}")
    if route.notes:
        lines.append("  notes:")
        lines.extend(f"    - {note}" for note in route.notes)
    return "\n".join(lines)


def execute_route(
    route: RouteDecision,
    *,
    hypothesis_file: Path,
    start: str | None,
    end: str | None,
    db: str | None,
    dry_run: bool,
    runner_args: list[str],
) -> int:
    if route.research_question_type == "standalone_edge":
        if route.instrument is None or route.orb_minutes is None:
            raise SystemExit(
                "This prereg cannot execute yet: instrument and orb_minutes must resolve to one value. "
                "Pass --instrument / --orb-minutes or narrow the prereg."
            )
        cmd = [
            _repo_python(),
            "-m",
            "trading_app.strategy_discovery",
            "--instrument",
            route.instrument,
            "--orb-minutes",
            str(route.orb_minutes),
            "--holdout-date",
            route.holdout_date,
            "--hypothesis-file",
            str(hypothesis_file),
        ]
        if start:
            cmd.extend(["--start", start])
        if end:
            cmd.extend(["--end", end])
        if db:
            cmd.extend(["--db", db])
        if dry_run:
            cmd.append("--dry-run")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        return proc.returncode

    if route.research_question_type == "conditional_role":
        if not route.execution_entrypoint:
            raise SystemExit(
                "This prereg is a bounded conditional-role study. No generic discovery engine will run it. "
                "Provide --runner or add execution.entrypoint/default_args to the prereg."
            )
        cmd = [_repo_python()]
        entrypoint = route.execution_entrypoint
        if entrypoint.endswith(".py"):
            cmd.append(str(PROJECT_ROOT / entrypoint if not Path(entrypoint).is_absolute() else entrypoint))
        else:
            cmd.extend(shlex.split(entrypoint))

        execution = yaml.safe_load(hypothesis_file.read_text(encoding="utf-8")).get("execution", {})
        default_args = execution.get("default_args", []) if isinstance(execution, dict) else []
        if not isinstance(default_args, list):
            raise SystemExit("execution.default_args must be a list of strings.")
        cmd.extend(str(arg) for arg in default_args)
        cmd.extend(runner_args)
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        return proc.returncode

    raise SystemExit(f"Unsupported prereg route: {route.research_question_type}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or execute a pre-registered research route.")
    parser.add_argument("--hypothesis-file", type=Path, required=True, help="Path to the prereg YAML file.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--execute", action="store_true", help="Execute the resolved route.")
    parser.add_argument("--instrument", default=None, help="Override instrument when the prereg is multi-instrument.")
    parser.add_argument(
        "--orb-minutes",
        type=int,
        default=None,
        help="Override ORB aperture when the prereg declares multiple apertures or omits it.",
    )
    parser.add_argument("--start", default=None, help="Optional discovery start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Optional discovery end date (YYYY-MM-DD).")
    parser.add_argument("--db", default=None, help="Optional DB path for discovery.")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to the underlying discovery path.")
    parser.add_argument(
        "--runner",
        default=None,
        help="Bounded-study runner entrypoint for conditional_role preregs. Overrides execution.entrypoint.",
    )
    parser.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        help="Additional arg passed to the bounded runner. May be repeated.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    hypothesis_file = args.hypothesis_file
    if not hypothesis_file.is_absolute():
        hypothesis_file = (PROJECT_ROOT / hypothesis_file).resolve()

    route = build_route_decision(
        hypothesis_file,
        instrument_override=args.instrument,
        orb_minutes_override=args.orb_minutes,
        runner_override=args.runner,
    )

    if args.format == "json":
        print(json.dumps(asdict(route), indent=2, sort_keys=True))
    else:
        print(_text_render(route))

    if args.execute:
        return execute_route(
            route,
            hypothesis_file=hypothesis_file,
            start=args.start,
            end=args.end,
            db=args.db,
            dry_run=args.dry_run,
            runner_args=args.runner_arg,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
