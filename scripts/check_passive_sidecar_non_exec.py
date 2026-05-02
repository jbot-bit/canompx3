"""Fail closed if passive-sidecar code drifts into execution behavior."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = PROJECT_ROOT / "trading_app" / "live" / "passive_sidecar"

FORBIDDEN_MARKERS = (
    "ProjectXOrderRouter",
    "CopyOrderRouter",
    "BrokerRouter",
    "webhook_server",
    "trading_app.live.projectx.order_router",
    "trading_app.live.copy_order_router",
    "trading_app.live.broker_connections",
    "/api/Order/place",
    "/api/Order/cancel",
    "/api/Order/modify",
    "build_order_spec",
    "build_exit_spec",
    "cancel_bracket_orders",
)


def scan_paths(paths: list[Path]) -> list[str]:
    violations: list[str] = []
    for root in paths:
        if root.is_file():
            files = [root]
        else:
            files = sorted(path for path in root.rglob("*.py") if path.is_file())
        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "submit(" in line or "cancel(" in line:
                    violations.append(f"{file_path}:{lineno}: forbidden call marker")
                for marker in FORBIDDEN_MARKERS:
                    if marker in line:
                        violations.append(f"{file_path}:{lineno}: forbidden marker {marker!r}")
    return violations


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    targets = [Path(arg).resolve() for arg in args] if args else [DEFAULT_TARGET]
    violations = scan_paths(targets)
    if violations:
        print("PASSIVE SIDECAR NON-EXEC CHECK FAILED")
        for violation in violations:
            print(violation)
        return 1
    print("PASSIVE SIDECAR NON-EXEC CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
