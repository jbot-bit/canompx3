"""Lightweight trace system for audit and research operations.

Captures what happened during an operation (files read, queries run,
findings, decisions) and writes structured JSON to logs/traces/.

Integrates with existing infrastructure:
- Complements pipeline/audit_log.py (which tracks DB operations)
- Writes to TRACES_DIR from pipeline/paths.py
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pipeline.paths import TRACES_DIR


class GovernanceDecision(str, Enum):
    """Decision labels for audit/research findings.

    Unifies existing labels across the codebase:
    - Fitness: FIT/WATCH/DECAY/STALE
    - Classification: CORE/REGIME/INVALID
    - Audits: PASS/FAIL/WARN
    These are the governance-level decisions that wrap those domain labels.
    """

    VALID = "VALID"
    INVALID = "INVALID"
    REGIME_ONLY = "REGIME_ONLY"
    STALE = "STALE"
    UNSUPPORTED = "UNSUPPORTED"
    BLOCKED = "BLOCKED"


@dataclass
class Finding:
    """Single audit or research finding."""

    check: str
    status: str  # PASS, FAIL, WARN
    detail: str
    data: dict[str, Any] | None = None


@dataclass
class TraceReport:
    """Structured output from any audit/research run.

    Every audit, research validation, or governed build should produce
    one of these. The trace is written to JSON for replay and evidence.
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_state: dict[str, Any] = field(default_factory=dict)
    truth_chain: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    queries_run: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    governance_decision: str = GovernanceDecision.VALID.value
    caveats: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def add_finding(
        self,
        check: str,
        status: str,
        detail: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.findings.append(Finding(check=check, status=status, detail=detail, data=data))

    def has_failures(self) -> bool:
        return any(f.status == "FAIL" for f in self.findings)

    def has_warnings(self) -> bool:
        return any(f.status == "WARN" for f in self.findings)

    def summary(self) -> str:
        n_pass = sum(1 for f in self.findings if f.status == "PASS")
        n_fail = sum(1 for f in self.findings if f.status == "FAIL")
        n_warn = sum(1 for f in self.findings if f.status == "WARN")
        return f"{n_pass} PASS, {n_fail} FAIL, {n_warn} WARN — {self.governance_decision}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, directory: Path | None = None) -> Path:
        """Write trace to JSON file. Returns the file path."""
        out_dir = directory or TRACES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        slug = self.task.replace(" ", "-")[:40].lower()
        # Remove characters that are unsafe in filenames
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        filename = f"{ts}-{slug}-{self.trace_id}.json"
        path = out_dir / filename
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


def get_git_state() -> dict[str, Any]:
    """Capture current git state for trace context."""
    result: dict[str, Any] = {}
    try:
        branch_out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if branch_out.returncode != 0:
            result["error"] = "not a git repo or git unavailable"
            return result
        result["branch"] = branch_out.stdout.strip()

        commit_out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["commit"] = commit_out.stdout.strip() if commit_out.returncode == 0 else "unknown"

        status_out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["dirty"] = bool(status_out.stdout.strip()) if status_out.returncode == 0 else True
    except Exception:
        result["error"] = "git state unavailable"
    return result
