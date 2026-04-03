"""Scan docs for claims that may be stale compared to live code/DB state.

Targets patterns that have caused real bugs in this project:
- Strategy counts (changed 5x in 3 months: 470 → 747 → 124 → ...)
- Instrument active/dead status (M2K went dead Mar 2026)
- Entry model references (E0 purged Feb 2026, E3 retired)
- "as of YYYY-MM-DD" dates older than configurable threshold
- Hardcoded drift check counts

Does NOT scan:
- .claude/memory/ files (historical by design, expected to contain old data)
- Code files (covered by check_drift.py and audit_behavioral.py)

@research-source CLAUDE.md volatile data rule
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Canonical sources — imported at scan time to get live values
_SCAN_TARGETS = [
    "docs/*.md",
    "docs/plans/*.md",
    "docs/specs/*.md",
    "HANDOFF.md",
    "ROADMAP.md",
]

# Files to skip (historical by design, or auto-generated)
_SKIP_PATTERNS = [
    ".claude/",
    "REPO_MAP.md",
    "docs/runtime/STAGE_STATE.md",
    "node_modules/",
    ".git/",
]

# How many days before an "as of" date is flagged
STALE_DAYS_THRESHOLD = 60


@dataclass
class StaleClaim:
    """A potentially stale claim found in a document."""

    file: str
    line: int
    claim: str
    actual: str
    severity: str  # STALE, WARN, INFO


@dataclass
class ScanResult:
    """Result of scanning all docs."""

    claims: list[StaleClaim] = field(default_factory=list)
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, file: str, line: int, claim: str, actual: str, severity: str = "STALE") -> None:
        self.claims.append(StaleClaim(file=file, line=line, claim=claim, actual=actual, severity=severity))

    @property
    def stale_count(self) -> int:
        return sum(1 for c in self.claims if c.severity == "STALE")

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.claims if c.severity == "WARN")

    def summary(self) -> str:
        return (
            f"Scanned {self.files_scanned} files: "
            f"{self.stale_count} STALE, {self.warn_count} WARN, "
            f"{len(self.errors)} errors"
        )


def _get_active_instruments() -> set[str]:
    """Get active instruments from canonical source."""
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        return set(ACTIVE_ORB_INSTRUMENTS)
    except ImportError:
        return set()


def _get_session_names() -> set[str]:
    """Get valid session names from canonical source."""
    try:
        from pipeline.dst import SESSION_CATALOG

        return set(SESSION_CATALOG.keys())
    except ImportError:
        return set()


def _get_dead_instruments() -> set[str]:
    """Get dead instruments from canonical source."""
    try:
        from pipeline.asset_configs import DEAD_ORB_INSTRUMENTS

        return set(DEAD_ORB_INSTRUMENTS)
    except ImportError:
        return set()


def _should_skip(filepath: Path) -> bool:
    """Check if file should be skipped."""
    rel = str(filepath.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return any(skip in rel for skip in _SKIP_PATTERNS)


def _collect_files() -> list[Path]:
    """Collect all markdown files to scan."""
    files = []
    for pattern in _SCAN_TARGETS:
        files.extend(PROJECT_ROOT.glob(pattern))
    return [f for f in files if f.is_file() and not _should_skip(f)]


# ---------------------------------------------------------------------------
# Check functions — each scans a line and may add findings
# ---------------------------------------------------------------------------


def check_as_of_dates(line: str, line_num: int, filepath: str, result: ScanResult) -> None:
    """Flag 'as of YYYY-MM-DD' dates older than threshold."""
    pattern = r"[Aa]s of (\d{4}-\d{2}-\d{2})"
    for match in re.finditer(pattern, line):
        try:
            date = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - date).days
            if age_days > STALE_DAYS_THRESHOLD:
                result.add(
                    file=filepath,
                    line=line_num,
                    claim=f"'as of {match.group(1)}' ({age_days} days old)",
                    actual=f"Exceeds {STALE_DAYS_THRESHOLD}-day threshold",
                    severity="WARN",
                )
        except ValueError:
            pass


def check_instrument_status(
    line: str,
    line_num: int,
    filepath: str,
    result: ScanResult,
    dead_instruments: set[str],
) -> None:
    """Flag dead instruments referenced as active."""
    if not dead_instruments:
        return
    # Build pattern dynamically from canonical source
    dead_pat = "|".join(re.escape(i) for i in sorted(dead_instruments))
    # Pattern 1: "Active ... : ... <DEAD>" (lists)
    list_pattern = rf"[Aa]ctive.*?:.*?\b({dead_pat})\b"
    # Pattern 2: "<DEAD> ... active" or "active ... <DEAD>" (prose)
    prose_pattern = rf"\b({dead_pat})\b.*\bactive\b|\bactive\b.*\b({dead_pat})\b"
    for match in re.finditer(list_pattern, line):
        inst = match.group(1)
        result.add(
            file=filepath,
            line=line_num,
            claim=f"{inst} referenced as active",
            actual=f"{inst} is DEAD for ORB",
            severity="STALE",
        )
        return  # One finding per line is enough
    for match in re.finditer(prose_pattern, line, re.IGNORECASE):
        inst = match.group(1) or match.group(2)
        # Skip lines that say the instrument is dead/removed
        if re.search(r"(?:dead|removed|retired|killed|dropped)", line, re.IGNORECASE):
            continue
        result.add(
            file=filepath,
            line=line_num,
            claim=f"{inst} referenced as active",
            actual=f"{inst} is DEAD for ORB",
            severity="STALE",
        )
        return


def check_e0_references(line: str, line_num: int, filepath: str, result: ScanResult) -> None:
    """Flag E0 entry model referenced as active/usable."""
    # E0 is purged — any reference suggesting it's active is stale
    # But historical references ("E0 was purged") are fine
    patterns = [
        r"\bE0\b.*(?:active|deploy|use|trade|recommend)",
        r"(?:active|deploy|use|trade).*\bE0\b",
        r"entry.model.*E0(?!.*(?:purge|dead|remove|retire))",
    ]
    for pat in patterns:
        if re.search(pat, line, re.IGNORECASE):
            # Skip lines that mention E0 as dead/purged
            if re.search(r"(?:purge|dead|remove|retire|replaced)", line, re.IGNORECASE):
                continue
            result.add(
                file=filepath,
                line=line_num,
                claim="E0 entry model referenced as active",
                actual="E0 was purged Feb 2026 (3 compounding biases)",
                severity="STALE",
            )
            break


def check_hardcoded_counts(line: str, line_num: int, filepath: str, result: ScanResult) -> None:
    """Flag hardcoded check counts (volatile data)."""
    # Pattern: "all N checks" or "N drift checks" with specific numbers
    pattern = r"(?:all\s+)?(\d+)\s+(?:drift\s+)?checks?"
    for match in re.finditer(pattern, line, re.IGNORECASE):
        count = int(match.group(1))
        # Only flag if it looks like a specific check count (not generic "2 checks")
        if count > 10:
            result.add(
                file=filepath,
                line=line_num,
                claim=f"Hardcoded count '{count} checks'",
                actual="Check counts are dynamic — verify against runtime",
                severity="WARN",
            )


def check_strategy_counts(line: str, line_num: int, filepath: str, result: ScanResult) -> None:
    """Flag specific strategy count claims that may be stale."""
    # Pattern: "N validated" or "N strategies" with 3+ digit numbers
    pattern = r"(\d{3,})\s+(?:validated|strategies|survivors|families)"
    for match in re.finditer(pattern, line, re.IGNORECASE):
        count = int(match.group(1))
        result.add(
            file=filepath,
            line=line_num,
            claim=f"Strategy count claim: {count}",
            actual="Strategy counts are volatile — query gold.db for current",
            severity="WARN",
        )


def scan_docs(verbose: bool = False) -> ScanResult:
    """Scan all target docs for stale claims."""
    result = ScanResult()
    dead_instruments = _get_dead_instruments()

    files = _collect_files()
    result.files_scanned = len(files)

    for filepath in files:
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            result.errors.append(f"{filepath}: {e}")
            continue

        rel_path = str(filepath.relative_to(PROJECT_ROOT)).replace("\\", "/")

        for line_num, line in enumerate(text.splitlines(), 1):
            # Skip empty lines and markdown headers for some checks
            stripped = line.strip()
            if not stripped:
                continue

            check_as_of_dates(stripped, line_num, rel_path, result)
            check_instrument_status(stripped, line_num, rel_path, result, dead_instruments)
            check_e0_references(stripped, line_num, rel_path, result)
            check_hardcoded_counts(stripped, line_num, rel_path, result)
            check_strategy_counts(stripped, line_num, rel_path, result)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan docs for stale claims")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all claims")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = scan_docs(verbose=args.verbose)

    if args.json:
        import json

        output = {
            "summary": result.summary(),
            "files_scanned": result.files_scanned,
            "claims": [
                {
                    "file": c.file,
                    "line": c.line,
                    "claim": c.claim,
                    "actual": c.actual,
                    "severity": c.severity,
                }
                for c in result.claims
            ],
            "errors": result.errors,
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.summary())
        print()

        if result.claims:
            # Group by severity
            for severity in ["STALE", "WARN", "INFO"]:
                claims = [c for c in result.claims if c.severity == severity]
                if claims:
                    print(f"--- {severity} ({len(claims)}) ---")
                    for c in claims:
                        print(f"  {c.file}:{c.line}")
                        print(f"    Claim: {c.claim}")
                        print(f"    Actual: {c.actual}")
                        print()

        if result.errors:
            print(f"--- ERRORS ({len(result.errors)}) ---")
            for e in result.errors:
                print(f"  {e}")

    # Exit: 0 = no STALE, 1 = STALE found, 2 = errors only
    if result.stale_count > 0:
        sys.exit(1)
    elif result.errors:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
