#!/usr/bin/env python3
"""Shared audit utilities — output formatting, tags, DB helper, finding registry.

Every phase script imports from here for consistent output and exit behavior.
Pattern follows existing audit_integrity.py / audit_behavioral.py.
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH, PROJECT_ROOT

# Re-export for convenience
__all__ = [
    "Severity",
    "Finding",
    "AuditPhase",
    "PROJECT_ROOT",
    "GOLD_DB_PATH",
    "db_connect",
]


class Severity(str, Enum):
    """From SYSTEM_AUDIT.md Section 3 — Issue Register severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Canonical tag vocabulary from SYSTEM_AUDIT.md Section 3
VALID_TAGS = {
    # Doc drift
    "DOC_STALE",
    "DOC_WRONG",
    "BOTH_WRONG",
    # Config drift
    "CONFIG_DRIFT",
    "COST_MODEL_DRIFT",
    "THRESHOLD_DRIFT",
    "SESSION_CONFIG_DRIFT",
    "SESSION_LABEL_DRIFT",
    "MCP_ALLOWLIST_STALE",
    # Schema
    "SCHEMA_DRIFT",
    "SCHEMA_BEHIND",
    "ORPHAN_COLUMN",
    # Data integrity
    "DATA_INTEGRITY_VIOLATION",
    "AGGREGATION_ERROR",
    "FEATURE_ERROR",
    "OUTCOME_ERROR",
    "COUNT_ANOMALY",
    "DATA_LOSS_RISK",
    # Write scope
    "SCOPED",
    "OVER_DELETE",
    "UNDER_DELETE",
    # Join safety
    "TRIPLE_JOIN_VIOLATION",
    "TRIPLE_JOIN_TRAP_RISK",
    "FILTER_LEAK",
    # Ghosts
    "PHANTOM_FILTER",
    "PHANTOM_SESSION",
    "ZOMBIE_E0",
    "ZOMBIE_STRATEGY",
    "ORPHAN_STRATEGY",
    "ORPHAN_FAMILY",
    "NO_GO_ZOMBIE",
    "DEAD_FILTER",
    # Staleness
    "STALE_CHECK",
    "STALE_TEST",
    "STALE_OUTCOMES",
    "STALE_RESEARCH_SCRIPT",
    "REPO_MAP_STALE",
    "BUILD_GAP",
    "BUILD_CHAIN_GAP",
    "REBUILD_NEEDED",
    # Gates
    "FAIL_CLOSED",
    "FAIL_OPEN",
    "GATE_MISSING",
    "TOOTHLESS",
    "TOOTHLESS_GATE",
    # Parity
    "FEATURE_PARITY_VIOLATION",
    "PARITY_VIOLATION",
    "IDENTICAL_PATH",
    "EQUIVALENT_LOGIC",
    "DIVERGENT",
    # Testing
    "UNTESTED",
    "UNTESTED_MODULE",
    "BRITTLE_TEST",
    "SMOKE_TEST_FAILURE",
    # Research
    "DATA_LEAK_RISK",
    "SPEC_VIOLATION",
    "UNDOCUMENTED",
    # Verdicts
    "MATCH",
    "CLEAN",
    "VERIFIED",
    "ENFORCED",
    "CONTAINED",
    "ALL_SAFE",
    "FULL_COVERAGE",
    "ACCURATE",
    "UP_TO_DATE",
    "NO_DUPLICATES",
    # Process
    "HALLUCINATED_PASS",
    "REQUIRES_HUMAN_DECISION",
    "SKIPPED",
    # Fix types (also used as tags)
    "DOC_FIX",
    "CODE_FIX",
    "CONFIG_FIX",
    "DATA_FIX",
    "DELETE",
}


@dataclass
class Finding:
    """A single audit finding from SYSTEM_AUDIT.md Section 3 schema."""

    severity: Severity
    tag: str
    claimed: str  # What the doc/config claims
    actual: str  # What reality shows
    evidence: str  # File:line, query result, or command output
    fix_type: str  # DOC_FIX / CODE_FIX / CONFIG_FIX / DATA_FIX / REBUILD_NEEDED / DELETE


@dataclass
class AuditPhase:
    """Base for all phase scripts — consistent output and exit behavior."""

    phase_num: int
    name: str
    findings: list[Finding] = field(default_factory=list)
    _check_count: int = 0

    def add_finding(
        self,
        severity: Severity,
        tag: str,
        claimed: str,
        actual: str,
        evidence: str,
        fix_type: str = "CODE_FIX",
    ):
        self.findings.append(
            Finding(
                severity=severity,
                tag=tag,
                claimed=claimed,
                actual=actual,
                evidence=evidence,
                fix_type=fix_type,
            )
        )

    def check_passed(self, label: str):
        """Record a passing check."""
        self._check_count += 1
        print(f"  [OK] {label}")

    def check_failed(self, label: str, detail: str = ""):
        """Record a failing check (finding must be added separately)."""
        self._check_count += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    def check_info(self, label: str, detail: str = ""):
        """Record an informational check (no pass/fail)."""
        self._check_count += 1
        msg = f"  [INFO] {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    def print_header(self):
        print("=" * 70)
        print(f"PHASE {self.phase_num} — {self.name.upper()}")
        print("=" * 70)

    def print_summary(self) -> int:
        """Print summary and return exit code (1 if CRITICAL, 0 otherwise)."""
        counts = {s: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity] += 1

        print("\n" + "=" * 70)
        if self.findings:
            parts = []
            for s in Severity:
                if counts[s]:
                    parts.append(f"{counts[s]} {s.value}")
            print(f"PHASE {self.phase_num} — {len(self.findings)} finding(s): {', '.join(parts)}")
            print()
            for f in self.findings:
                print(f"  [{f.severity.value}] {f.tag}: {f.actual}")
                print(f"         evidence: {f.evidence}")
        else:
            print(f"PHASE {self.phase_num} PASSED: {self._check_count} checks clean")
        print("=" * 70)

        return 1 if counts[Severity.CRITICAL] > 0 else 0

    def run_and_exit(self):
        """Print summary and sys.exit with appropriate code."""
        code = self.print_summary()
        sys.exit(code)


def db_connect():
    """Read-only DuckDB connection via canonical path."""
    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)
