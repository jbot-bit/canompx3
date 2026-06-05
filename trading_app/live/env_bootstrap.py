"""Runtime environment loading for live broker paths.

The live launchers can run from linked worktrees that intentionally do not
carry the canonical ``.env``. This module loads only the untracked runtime
``.env`` from the canonical runtime root, never ``.env.example``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:  # pragma: no cover - production dependency, shell env fallback
    dotenv_values = None  # type: ignore[assignment]
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _discover_git_common_root(project_root: Path) -> Path | None:
    """Return the shared repo root when this checkout is a linked worktree."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except Exception:
        return None

    stdout = result.stdout.strip()
    if not stdout:
        return None

    common_dir = Path(stdout)
    if not common_dir.is_absolute():
        common_dir = (project_root / common_dir).resolve()
    if common_dir.name != ".git":
        return None

    common_root = common_dir.parent
    if not (common_root / "pipeline").exists():
        return None
    return common_root


CANONICAL_RUNTIME_ROOT = _discover_git_common_root(PROJECT_ROOT) or PROJECT_ROOT


PROJECTX_SECRET_ENV_KEYS = ("PROJECTX_USERNAME", "PROJECTX_USER", "PROJECTX_API_KEY")
_PLACEHOLDER_MARKERS = (
    "<",
    ">",
    "...",
    "change_me",
    "changeme",
    "dummy",
    "example",
    "placeholder",
    "replace",
    "todo",
    "your_",
    "your-",
)


@dataclass(frozen=True)
class RuntimeEnvLoadResult:
    env_path: Path | None
    loaded: bool
    attempted_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class EnvExampleSecretWarning:
    path: Path
    keys: tuple[str, ...]

    def format_message(self) -> str:
        key_list = ", ".join(self.keys)
        return (
            f"tracked {self.path.name} contains secret-shaped values for {key_list}; "
            "runtime ignores .env.example; keep real ProjectX credentials in untracked .env"
        )


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _candidate_env_paths() -> list[Path]:
    return _unique_paths(
        [
            CANONICAL_RUNTIME_ROOT / ".env",
            PROJECT_ROOT / ".env",
        ]
    )


def load_runtime_env(env_path: Path | str | None = None) -> RuntimeEnvLoadResult:
    """Load the runtime ``.env`` without overriding shell-provided values."""

    candidates = [Path(env_path)] if env_path is not None else _candidate_env_paths()
    attempted_paths = tuple(candidate.expanduser().resolve() for candidate in candidates)
    if load_dotenv is None:
        return RuntimeEnvLoadResult(env_path=None, loaded=False, attempted_paths=attempted_paths)

    for candidate in candidates:
        if not candidate.exists():
            continue
        loaded = bool(load_dotenv(candidate, override=False))
        return RuntimeEnvLoadResult(env_path=candidate.resolve(), loaded=loaded, attempted_paths=attempted_paths)
    return RuntimeEnvLoadResult(env_path=None, loaded=False, attempted_paths=attempted_paths)


def _is_git_tracked(root: Path, relative_path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative_path],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _clean_env_value(value: object) -> str:
    return str(value).strip().strip('"').strip("'")


def _looks_secret_shaped(key: str, value: object) -> bool:
    text = _clean_env_value(value)
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return False
    if key == "PROJECTX_API_KEY":
        return len(text) >= 20
    return len(text) >= 4


def detect_tracked_env_example_secret_shapes(root: Path | None = None) -> EnvExampleSecretWarning | None:
    """Return a redacted warning when tracked ``.env.example`` has real-looking secrets."""

    repo_root = (root or CANONICAL_RUNTIME_ROOT).resolve()
    env_example = repo_root / ".env.example"
    if not env_example.exists():
        return None
    if not _is_git_tracked(repo_root, ".env.example"):
        return None
    if dotenv_values is None:
        return None

    values = dotenv_values(env_example)
    secret_keys = tuple(
        key for key in PROJECTX_SECRET_ENV_KEYS if key in values and _looks_secret_shaped(key, values.get(key))
    )
    if not secret_keys:
        return None
    return EnvExampleSecretWarning(path=env_example, keys=secret_keys)
