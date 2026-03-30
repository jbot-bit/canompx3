# Large File Reading Protocol

The Read tool has a hard limit of 2000 lines per call. Files larger than this WILL be silently truncated.

## Rule

Before reading any production file that might exceed 2000 lines:
1. Check line count first: `wc -l <file>`
2. If > 1800 lines: read in targeted chunks using `offset` and `limit` parameters
3. Never assume you've read the entire file if you didn't check the line count
4. For exploration: use Grep to find the relevant section first, then Read with offset

## Common Large Files in This Project

These files regularly exceed 2000 lines — always chunk-read:
- `pipeline/check_drift.py` — drift checks (read specific check by grepping first)
- `trading_app/config.py` — trading config (grep for the specific section needed)
- `trading_app/strategy_discovery.py` — discovery logic
- `trading_app/strategy_validator.py` — validation logic
- `tests/` — many test files are large

## Pattern

```
# Step 1: Find what you need
grep -n "def function_name" file.py

# Step 2: Read just that section
Read(file_path, offset=150, limit=50)
```

Never read a 3000-line file from line 1 and assume you saw everything.
