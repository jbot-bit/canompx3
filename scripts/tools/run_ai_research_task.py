#!/usr/bin/env python3
"""Run a bounded OpenRouter research task with repo-native packet grounding."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

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


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.ai.openrouter_runtime import run_openrouter_task  # noqa: E402
from trading_app.ai.provider_registry import list_openrouter_research_profiles  # noqa: E402
from trading_app.ai.schema_registry import list_schemas  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded OpenRouter research task.")
    parser.add_argument("--task", required=True, help="Natural-language research/planning task.")
    parser.add_argument("--profile", default="deepseek_planning", choices=list_openrouter_research_profiles())
    parser.add_argument("--schema", choices=list_schemas())
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument(
        "--execute", action="store_true", help="Call OpenRouter instead of returning a dry-run envelope."
    )
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_openrouter_task(
        task_text=args.task,
        profile_id=args.profile,
        root=Path(args.root).resolve(),
        db_path=Path(args.db_path).resolve(),
        schema_name=args.schema,
        max_turns=args.max_turns,
        execute=args.execute,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    _ensure_repo_python()
    raise SystemExit(main())
