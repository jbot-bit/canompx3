# Eval Schema Reference

## eval.json Structure

```json
{
  "skill": "skill-name",
  "version": 1,
  "tests": [
    {
      "id": "unique-test-id",
      "prompt": "The task to give the subagent",
      "context": "Optional setup context or constraints",
      "assertions": [
        {
          "id": "assertion-id",
          "type": "text_contains",
          "value": "expected text",
          "description": "Human-readable description of what this checks"
        }
      ]
    }
  ]
}
```

## Assertion Types

| Type | Required Fields | What It Checks |
|------|----------------|----------------|
| `text_contains` | `value` | Output contains substring |
| `text_not_contains` | `value` | Output does NOT contain substring |
| `regex_match` | `pattern` | Output matches regex |
| `regex_not_match` | `pattern` | Output does NOT match regex |
| `command_ran` | `pattern` | A command matching pattern appears in output |
| `line_count_gte` | `threshold`, opt `pattern` | >= N lines matching pattern |
| `line_count_lte` | `threshold`, opt `pattern` | <= N lines matching pattern |
| `occurrence_count_gte` | `value`, `threshold` | >= N occurrences of value |
| `word_count_lte` | `threshold` | Total word count <= N |
| `starts_with` | `value` | Output starts with value |
| `ends_with` | `value` | Output ends with value |
| `all_rows_have_field` | `row_pattern`, `field_pattern` | Every row matching row_pattern also matches field_pattern |

All string comparisons are case-insensitive by default. Set `"case_sensitive": true` to override.

## Writing Good Assertions

**Binary only.** Every assertion must be unambiguously true or false.

| Good (binary) | Bad (subjective) |
|---------------|------------------|
| Output contains "rr_target" | Output has useful information |
| No "PURGED" in output | Output is well-organized |
| Word count <= 500 | Output is concise |
| Command `check_drift.py` was run | Proper verification was done |

**For action-based skills**, check:
- Commands that MUST be run (`command_ran`)
- Fields that MUST appear in output (`text_contains`, `all_rows_have_field`)
- Things that MUST NOT appear (`text_not_contains`)
- Structural requirements (`line_count_gte`, `regex_match`)

## Test Prompt Design

Prompts should simulate real user requests. Use the same language users actually use:
- "what do I trade tonight" (not "display validated strategies")
- "is everything green?" (not "run verification gates")
- "where are we" (not "provide project orientation")

Include edge cases:
- Ambiguous requests that could trigger wrong behavior
- Requests for specific instruments/sessions
- Requests that historically caused failures
