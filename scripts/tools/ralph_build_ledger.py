"""Build structured JSON ledger from ralph-loop-history.md.

Reads the semi-structured markdown iteration log and produces
docs/ralph-loop/ralph-ledger.json with aggregated stats.
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_PATH = PROJECT_ROOT / "docs" / "ralph-loop" / "ralph-loop-history.md"
LEDGER_PATH = PROJECT_ROOT / "docs" / "ralph-loop" / "ralph-ledger.json"


# ── Finding type classification ──────────────────────────────────────
# Maps keyword patterns (applied to finding text, lowercased) to a type.
# Order matters: first match wins.
FINDING_TYPE_RULES: list[tuple[str, str]] = [
    (r"hardcod|canonical.?violat|should (use|import|derive|reference)|duplicat.*(canonical|source)|instead of.*canonical|not using.*variable|missing from session.?order", "canonical_violation"),
    (r"silent(ly)?|fail[- ]?open|no (log|warning|diagnostic)|invisible|no-op|silently (return|drop|skip|arm|fall|omit|discard)", "silent_failure"),
    (r"bare\s+except|except\s+exception|broad\s+except|narrowed? to", "broad_except"),
    (r"dead\b.*\b(variable|assignment|code|import|function)|orphan|never (referenced|used|read|imported)|unused.*(import|variable|assignment|loop var)|removed.*dead|PROJECT_ROOT.*dead|F841", "dead_code"),
    (r"connection.?leak|\.close\(\).*not in.*finally|not in.*finally.*block", "connection_leak"),
    (r"annotation|@research[- ]?source|missing.*provenance|unannotat", "annotation_debt"),
    (r"(stale|outdated|wrong).*(comment|docstring|count|stat|number)|volatile.?data|mislead", "stale_metadata"),
    (r"timing[- ]?safe|hmac|secret|auth|security|compare_digest", "security"),
    (r"oom|memory|unbounded|OOM|blocking ci", "resource"),
    (r"ruff|F541|I001|B007|B023|B905|E702|f-string|import sort|lint", "lint"),
    (r"pyright|type.?error|reportOptional|ufunc mismatch|None guard|total_seconds", "type_error"),
    (r"aperture.*missing|orb_minutes.*missing|query.*missing.*filter|missing.*AND|inflat", "query_bug"),
    (r"falsy[- ]?zero|or.*pattern|antipattern.*or|0\.0.*treated as none", "falsy_zero"),
    (r"fail[- ]?closed|inverted|unknown.*filter|blocked", "fail_closed_fix"),
    (r"drift.?check|regex.*(hardening|gap|only matched)", "drift_check_fix"),
    (r"unreachable|key.*mismatch|lookup.*silently|preflight", "unreachable_code"),
    (r"size_multiplier|friction|inflat.*dollar|cost", "cost_model_bug"),
    (r"left join.*inner|IS NULL fallback|diverge", "query_alignment"),
    (r"test|coverage|unit test", "test_gap"),
]


def classify_finding(text: str) -> str:
    """Classify a finding into a type based on keyword matching."""
    lower = text.lower()
    for pattern, ftype in FINDING_TYPE_RULES:
        if re.search(pattern, lower):
            return ftype
    return "other"


def infer_severity(finding_text: str, classification: str, phase: str) -> str:
    """Infer severity from the finding text and metadata."""
    lower = finding_text.lower()
    # Explicit severity markers in the text
    if re.search(r"\bcritical\b", lower):
        return "CRITICAL"
    if re.search(r"\bhigh\b", lower):
        return "HIGH"
    if re.search(r"\bmedium\b", lower):
        return "MEDIUM"
    if re.search(r"\blow\b", lower):
        return "LOW"
    # Infer from context
    if phase == "audit-only":
        return "LOW"
    if any(kw in lower for kw in ["security", "oom", "crash", "timing-safe"]):
        return "HIGH"
    if any(kw in lower for kw in ["fail-open", "silent", "leak", "cost", "inflat"]):
        return "MEDIUM"
    return "LOW"


def parse_field(block: str, field: str) -> str:
    """Extract a field value from an iteration block."""
    # Match "- Field: value" or "- Field value" patterns
    pattern = rf"^-\s*{re.escape(field)}:\s*(.+?)$"
    m = re.search(pattern, block, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def extract_files_from_target(target: str) -> list[str]:
    """Pull file paths from a Target line."""
    files = []
    # Match patterns like path/to/file.py or path/to/file.py:123
    for m in re.finditer(r"([\w/]+\.py)(?::\d+)?", target):
        files.append(m.group(1))
    return files


def parse_iterations(text: str) -> list[dict]:
    """Parse all iteration blocks from the markdown text."""
    # Split on iteration headers
    pattern = r"##\s*Iteration\s+(\d+)\s*(?:—|--|-)\s*(\d{4}-\d{2}-\d{2})"
    headers = list(re.finditer(pattern, text))
    iterations = []
    seen_iters: set[int] = set()

    for i, header in enumerate(headers):
        iter_num = int(header.group(1))
        iter_date = header.group(2)

        # Get block text up to next header or end
        start = header.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]

        # Skip duplicates (some iterations appear twice in the file)
        if iter_num in seen_iters:
            continue
        seen_iters.add(iter_num)

        phase = parse_field(block, "Phase").split("(")[0].strip().lower()
        # Normalize phase
        if "audit" in phase and "fix" not in phase:
            phase = "audit-only"
        elif "fix" in phase or "test" in phase:
            phase = "fix"
        elif "audit" in phase:
            phase = "fix"  # audit+fix = fix

        classification_raw = parse_field(block, "Classification")
        classification = "audit-only"
        if "mechanical" in classification_raw.lower():
            classification = "mechanical"
        elif "judgment" in classification_raw.lower():
            classification = "judgment"
        elif phase == "audit-only":
            classification = "audit-only"
        elif phase == "fix":
            # Some early iterations lack Classification field
            classification = "judgment"

        target = parse_field(block, "Target")
        finding_text = parse_field(block, "Finding")
        action = parse_field(block, "Action")
        blast_radius = parse_field(block, "Blast radius")
        verification = parse_field(block, "Verification")
        commit = parse_field(block, "Commit")

        # Determine verdict
        if phase == "audit-only":
            verdict = "CLEAN" if "clean" in finding_text.lower() or "0 findings" in finding_text.lower() else "NOTED"
        elif commit and commit.upper() not in ("NONE", "PENDING", ""):
            verdict = "FIXED"
        elif commit and commit.upper() == "PENDING":
            verdict = "PENDING"
        else:
            verdict = "ACCEPT"

        finding_type = classify_finding(finding_text) if finding_text else "other"
        severity = infer_severity(finding_text, classification, phase)

        # Extract files
        files = extract_files_from_target(target)

        entry = {
            "iter": iter_num,
            "date": iter_date,
            "phase": phase,
            "classification": classification,
            "target": target,
            "finding_type": finding_type,
            "severity": severity,
            "verdict": verdict,
            "commit": commit if commit else None,
            "files": files,
        }
        iterations.append(entry)

    # Sort by iteration number
    iterations.sort(key=lambda x: x["iter"])
    return iterations


def compute_findings_by_type(iterations: list[dict]) -> dict:
    """Aggregate finding counts by type."""
    types: dict[str, dict] = {}
    for it in iterations:
        ft = it["finding_type"]
        if ft not in types:
            types[ft] = {"found": 0, "fixed": 0, "deferred": 0}
        types[ft]["found"] += 1
        if it["verdict"] in ("FIXED", "ACCEPT"):
            types[ft]["fixed"] += 1
        elif it["verdict"] in ("NOTED", "PENDING"):
            types[ft]["deferred"] += 1

    # Add fix_rate
    for v in types.values():
        v["fix_rate"] = round(v["fixed"] / v["found"], 2) if v["found"] > 0 else 0.0

    # Sort by found count descending
    return dict(sorted(types.items(), key=lambda kv: -kv[1]["found"]))


def compute_findings_by_severity(iterations: list[dict]) -> dict:
    """Count iterations by severity."""
    counts: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for it in iterations:
        sev = it["severity"]
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def compute_classifications(iterations: list[dict]) -> dict:
    """Count iterations by classification."""
    counts: dict[str, int] = {}
    for it in iterations:
        c = it["classification"]
        counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def compute_files_audited(iterations: list[dict]) -> dict:
    """Build per-file audit stats."""
    files: dict[str, dict] = {}
    for it in iterations:
        for f in it["files"]:
            if f not in files:
                files[f] = {"audit_count": 0, "last_iter": 0, "findings": 0}
            files[f]["audit_count"] += 1
            files[f]["last_iter"] = max(files[f]["last_iter"], it["iter"])
            if it["verdict"] in ("FIXED", "ACCEPT", "PENDING"):
                files[f]["findings"] += 1
    # Sort by audit_count descending
    return dict(sorted(files.items(), key=lambda kv: -kv[1]["audit_count"]))


def compute_consecutive_low(iterations: list[dict]) -> tuple[int, int]:
    """Scan from most recent iteration backwards.

    Returns (consecutive_low_only, last_high_finding_iter).
    'low' means severity is LOW and phase is not audit-only with no finding.
    """
    consecutive = 0
    last_high_iter = 0
    # Walk from highest iter to lowest
    for it in reversed(iterations):
        sev = it["severity"]
        if sev in ("CRITICAL", "HIGH", "MEDIUM"):
            if last_high_iter == 0:
                last_high_iter = it["iter"]
            break
        consecutive += 1

    # If we never found a high, set last_high_iter to 0
    if last_high_iter == 0:
        # Scan all for the last one
        for it in reversed(iterations):
            if it["severity"] in ("CRITICAL", "HIGH", "MEDIUM"):
                last_high_iter = it["iter"]
                break

    return consecutive, last_high_iter


def build_ledger(text: str) -> dict:
    """Build the full ledger structure."""
    iterations = parse_iterations(text)

    consecutive_low, last_high_iter = compute_consecutive_low(iterations)

    ledger = {
        "total_iterations": len(iterations),
        "last_updated": date.today().isoformat(),
        "consecutive_low_only": consecutive_low,
        "last_high_finding_iter": last_high_iter,
        "findings_by_type": compute_findings_by_type(iterations),
        "findings_by_severity": compute_findings_by_severity(iterations),
        "classifications": compute_classifications(iterations),
        "files_audited": compute_files_audited(iterations),
        "iterations": [
            {
                "iter": it["iter"],
                "date": it["date"],
                "phase": it["phase"],
                "classification": it["classification"],
                "target": it["target"],
                "finding_type": it["finding_type"],
                "severity": it["severity"],
                "verdict": it["verdict"],
                "commit": it["commit"],
            }
            for it in iterations
        ],
    }
    return ledger


def print_summary(ledger: dict) -> None:
    """Print a concise summary of the ledger."""
    print(f"=== Ralph Loop Ledger Summary ===")
    print(f"Total iterations parsed: {ledger['total_iterations']}")
    print(f"Consecutive LOW-only (tail): {ledger['consecutive_low_only']}")
    print(f"Last HIGH+ finding: iter {ledger['last_high_finding_iter']}")
    print()

    print("Severity distribution:")
    for sev, count in ledger["findings_by_severity"].items():
        print(f"  {sev:>10s}: {count}")
    print()

    print("Classification distribution:")
    for cls, count in ledger["classifications"].items():
        print(f"  {cls:>15s}: {count}")
    print()

    print("Top finding types:")
    for ft, stats in list(ledger["findings_by_type"].items())[:10]:
        print(f"  {ft:<25s}  found={stats['found']:>3d}  fixed={stats['fixed']:>3d}  rate={stats['fix_rate']:.0%}")
    print()

    print(f"Unique files audited: {len(ledger['files_audited'])}")
    top_files = list(ledger["files_audited"].items())[:10]
    print("Most-audited files:")
    for f, stats in top_files:
        print(f"  {f:<55s}  audits={stats['audit_count']}  findings={stats['findings']}  last_iter={stats['last_iter']}")
    print()

    dates = sorted(set(it["date"] for it in ledger["iterations"] if it.get("date")))
    if dates:
        print(f"Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")


def main() -> None:
    if not HISTORY_PATH.exists():
        print(f"ERROR: {HISTORY_PATH} not found", file=sys.stderr)
        sys.exit(1)

    text = HISTORY_PATH.read_text(encoding="utf-8")
    ledger = build_ledger(text)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"Wrote {LEDGER_PATH} ({len(ledger['iterations'])} iterations)\n")

    print_summary(ledger)


if __name__ == "__main__":
    main()
