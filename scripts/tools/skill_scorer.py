#!/usr/bin/env python3
"""Immutable skill assessment scorer.

LOCKED FILE — the skill-improve loop agent MUST NOT modify this file.
This is the equivalent of Karpathy's prepare.py: the agent can only edit
the skill under test, never the scoring mechanism.

Usage:
    python scripts/tools/skill_scorer.py <assessment.json> --transcript <file>
    echo "transcript text" | python scripts/tools/skill_scorer.py <assessment.json>
"""

import json
import re
import sys
from pathlib import Path


def load_assessment(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def check_assertion(assertion: dict, text: str) -> dict:
    """Run a single binary assertion against transcript text. Returns result dict."""
    atype = assertion["type"]
    result = {
        "id": assertion["id"],
        "description": assertion.get("description", ""),
        "type": atype,
        "passed": False,
    }

    if atype == "text_contains":
        target = assertion["value"]
        case_sensitive = assertion.get("case_sensitive", False)
        if case_sensitive:
            result["passed"] = target in text
        else:
            result["passed"] = target.lower() in text.lower()

    elif atype == "text_not_contains":
        target = assertion["value"]
        case_sensitive = assertion.get("case_sensitive", False)
        if case_sensitive:
            result["passed"] = target not in text
        else:
            result["passed"] = target.lower() not in text.lower()

    elif atype == "regex_match":
        pattern = assertion["pattern"]
        flags = re.IGNORECASE if not assertion.get("case_sensitive", False) else 0
        result["passed"] = bool(re.search(pattern, text, flags))

    elif atype == "regex_not_match":
        pattern = assertion["pattern"]
        flags = re.IGNORECASE if not assertion.get("case_sensitive", False) else 0
        result["passed"] = not bool(re.search(pattern, text, flags))

    elif atype == "command_ran":
        pattern = assertion["pattern"]
        result["passed"] = bool(re.search(pattern, text))

    elif atype == "line_count_gte":
        pattern = assertion.get("pattern", ".")
        threshold = assertion["threshold"]
        matches = len(re.findall(pattern, text, re.MULTILINE))
        result["passed"] = matches >= threshold
        result["actual_count"] = matches

    elif atype == "line_count_lte":
        pattern = assertion.get("pattern", ".")
        threshold = assertion["threshold"]
        matches = len(re.findall(pattern, text, re.MULTILINE))
        result["passed"] = matches <= threshold
        result["actual_count"] = matches

    elif atype == "occurrence_count_gte":
        target = assertion["value"]
        threshold = assertion["threshold"]
        actual = text.lower().count(target.lower())
        result["passed"] = actual >= threshold
        result["actual_count"] = actual

    elif atype == "word_count_lte":
        threshold = assertion["threshold"]
        actual = len(text.split())
        result["passed"] = actual <= threshold
        result["actual_count"] = actual

    elif atype == "starts_with":
        prefix = assertion["value"]
        stripped = text.strip()
        result["passed"] = stripped.lower().startswith(prefix.lower())

    elif atype == "ends_with":
        suffix = assertion["value"]
        stripped = text.strip()
        result["passed"] = stripped.lower().endswith(suffix.lower())

    elif atype == "all_rows_have_field":
        row_pattern = assertion["row_pattern"]
        field_pattern = assertion["field_pattern"]
        rows = re.findall(row_pattern, text, re.MULTILINE)
        if not rows:
            result["passed"] = False
            result["note"] = "no rows matched row_pattern"
        else:
            missing = [r for r in rows if not re.search(field_pattern, r)]
            result["passed"] = len(missing) == 0
            result["total_rows"] = len(rows)
            result["rows_missing_field"] = len(missing)

    else:
        result["passed"] = False
        result["error"] = f"unknown assertion type: {atype}"

    return result


def score_test(test: dict, text: str) -> dict:
    """Score a single test (prompt + assertions) against transcript text."""
    results = []
    for assertion in test["assertions"]:
        results.append(check_assertion(assertion, text))

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    return {
        "test_id": test["id"],
        "prompt": test["prompt"],
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "assertions": results,
    }


def score_all(assessment_def: dict, transcripts: dict[str, str]) -> dict:
    """Score all tests in an assessment definition.

    Args:
        assessment_def: The assessment.json content
        transcripts: Mapping of test_id -> transcript text
    """
    test_results = []
    total_assertions = 0
    total_passed = 0

    for test in assessment_def["tests"]:
        text = transcripts.get(test["id"], "")
        if not text:
            test_results.append(
                {
                    "test_id": test["id"],
                    "prompt": test["prompt"],
                    "total": len(test["assertions"]),
                    "passed": 0,
                    "failed": len(test["assertions"]),
                    "pass_rate": 0.0,
                    "skipped": True,
                    "assertions": [
                        {
                            "id": a["id"],
                            "description": a.get("description", ""),
                            "passed": False,
                            "note": "test not run",
                        }
                        for a in test["assertions"]
                    ],
                }
            )
            total_assertions += len(test["assertions"])
        else:
            result = score_test(test, text)
            test_results.append(result)
            total_assertions += result["total"]
            total_passed += result["passed"]

    return {
        "skill": assessment_def["skill"],
        "total_assertions": total_assertions,
        "passed": total_passed,
        "failed": total_assertions - total_passed,
        "pass_rate": total_passed / total_assertions if total_assertions > 0 else 0.0,
        "tests": test_results,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: skill_scorer.py <assessment.json> [--transcript <file>]", file=sys.stderr)
        print("       echo text | skill_scorer.py <assessment.json>", file=sys.stderr)
        sys.exit(1)

    assessment_path = sys.argv[1]
    assessment_def = load_assessment(assessment_path)

    transcript_file = None
    for i, arg in enumerate(sys.argv):
        if arg == "--transcript" and i + 1 < len(sys.argv):
            transcript_file = sys.argv[i + 1]

    if transcript_file:
        text = Path(transcript_file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    if "--test-id" in sys.argv:
        test_id_idx = sys.argv.index("--test-id") + 1
        test_id = sys.argv[test_id_idx]
        matching = [t for t in assessment_def["tests"] if t["id"] == test_id]
        if not matching:
            print(json.dumps({"error": f"test_id '{test_id}' not found"}))
            sys.exit(1)
        result = score_test(matching[0], text)
        print(json.dumps(result, indent=2))
    else:
        transcripts = {t["id"]: text for t in assessment_def["tests"]}
        result = score_all(assessment_def, transcripts)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
