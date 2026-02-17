# Ralph Plan

## Context
Full codebase audit against CLAUDE.md authority. Reviews each module area for compliance with documented rules, catches stale code, dead references, security issues, and drift from stated architecture. Covers uncommitted changes, pipeline, trading_app, tests, scripts, and cross-document consistency.

## Tasks

```json
[
  {
    "id": 1,
    "category": "verification",
    "description": "Review uncommitted changes against CLAUDE.md standards",
    "passes": true,
    "steps": [
      "Step 1: Run git diff to see all uncommitted changes across 15 modified files",
      "Step 2: Read CLAUDE.md for the authoritative rules",
      "Step 3: Check each changed file for: hardcoded symbols, import direction violations, timezone hygiene, schema consistency, security (no secrets, no injection)",
      "Step 4: Check TRADING_RULES.md changes for consistency with code changes",
      "Step 5: Run python pipeline/check_drift.py to verify drift checks pass",
      "Step 6: Run python -m pytest tests/ -x -q to verify tests pass",
      "Step 7: Write findings to ralph-audit-report.md under '## 1. Uncommitted Changes'"
    ],
    "acceptance": "All uncommitted changes reviewed, drift check passes, tests pass, findings logged"
  },
  {
    "id": 2,
    "category": "verification",
    "description": "Audit pipeline/ modules against CLAUDE.md architecture rules",
    "passes": true,
    "steps": [
      "Step 1: Read pipeline/ingest_dbn.py, pipeline/build_bars_5m.py, pipeline/build_daily_features.py",
      "Step 2: Verify fail-closed principle: validation failures abort immediately",
      "Step 3: Verify idempotent operations: INSERT OR REPLACE or DELETE+INSERT patterns",
      "Step 4: Verify one-way dependency: pipeline/ never imports from trading_app/",
      "Step 5: Check timezone hygiene: all DB timestamps UTC, Brisbane local only for display",
      "Step 6: Verify GC to MGC source_symbol handling per CLAUDE.md critical note",
      "Step 7: Check pipeline/paths.py DUCKDB_PATH env var handling",
      "Step 8: Read pipeline/init_db.py and verify schema matches documented tables",
      "Step 9: Append findings to ralph-audit-report.md under '## 2. Pipeline Modules'"
    ],
    "acceptance": "All pipeline modules reviewed, no CLAUDE.md violations found or violations documented"
  },
  {
    "id": 3,
    "category": "verification",
    "description": "Audit trading_app/ modules against CLAUDE.md and TRADING_RULES.md",
    "passes": true,
    "steps": [
      "Step 1: Read trading_app/outcome_builder.py — verify pre-computed outcomes pattern",
      "Step 2: Read trading_app/strategy_discovery.py — verify grid search, DST split compliance",
      "Step 3: Read trading_app/strategy_validator.py — verify walk-forward, DST columns, double-break exclusion",
      "Step 4: Read trading_app/config.py — verify classification thresholds (CORE>=100, REGIME 30-99, INVALID<30)",
      "Step 5: Read trading_app/db_manager.py — verify no concurrent write violations, connection handling",
      "Step 6: Check FIX5 trade day invariant compliance across all modules",
      "Step 7: Verify DST contamination rules: sessions 0900/1800/0030/2300 must split by DST regime",
      "Step 8: Append findings to ralph-audit-report.md under '## 3. Trading App Modules'"
    ],
    "acceptance": "All trading_app modules reviewed against both CLAUDE.md and TRADING_RULES.md, findings logged"
  },
  {
    "id": 4,
    "category": "verification",
    "description": "Audit tests/ for coverage gaps and compliance",
    "passes": true,
    "steps": [
      "Step 1: List all test files in tests/, tests/test_pipeline/, tests/test_trading_app/",
      "Step 2: Read key test files and verify they test documented behavior, not unbuilt features",
      "Step 3: Check for CLAUDE.md rule: Do NOT reference unbuilt features in code or tests",
      "Step 4: Verify test_dst.py covers DST contamination rules from CLAUDE.md",
      "Step 5: Check that validation gate tests exist for all 7 ingestion gates and 4 aggregation gates",
      "Step 6: Verify no hardcoded paths (OneDrive, C:\\db, etc.) in tests",
      "Step 7: Run full test suite: python -m pytest tests/ -v --tb=short 2>&1 | tail -30",
      "Step 8: Append findings to ralph-audit-report.md under '## 4. Test Coverage'"
    ],
    "acceptance": "Test suite passes, coverage gaps documented, no tests reference unbuilt features"
  },
  {
    "id": 5,
    "category": "verification",
    "description": "Audit scripts/ and infrastructure for stale references and security",
    "passes": true,
    "steps": [
      "Step 1: List all scripts in scripts/ recursively",
      "Step 2: Read each script and check for: stale OneDrive paths, hardcoded DB paths, leaked API keys",
      "Step 3: Check .env handling — DATABENTO_API_KEY should never be committed",
      "Step 4: Review .githooks/pre-commit — verify it runs drift check + fast tests",
      "Step 5: Review .github/workflows/ci.yml — verify CI runs drift check + tests",
      "Step 6: Check scripts/infra/ files for correct paths after project move",
      "Step 7: Verify no sys.path.insert hacks remain (should use pip install -e . now)",
      "Step 8: Append findings to ralph-audit-report.md under '## 5. Scripts and Infrastructure'"
    ],
    "acceptance": "All scripts reviewed, no stale paths, no security issues, findings logged"
  },
  {
    "id": 6,
    "category": "verification",
    "description": "Cross-reference CLAUDE.md, TRADING_RULES.md, ROADMAP.md, and REPO_MAP.md for consistency",
    "passes": false,
    "steps": [
      "Step 1: Read CLAUDE.md, TRADING_RULES.md, ROADMAP.md",
      "Step 2: Verify ROADMAP.md phase status matches what is actually built in code",
      "Step 3: Check CLAUDE.md data flow diagram matches actual module imports",
      "Step 4: Verify document authority table — no conflicts between docs",
      "Step 5: Check that DST contamination section in CLAUDE.md matches actual dst.py implementation",
      "Step 6: Verify key commands section — all commands actually work",
      "Step 7: Check REPO_MAP.md is up to date by running python scripts/tools/gen_repo_map.py",
      "Step 8: Append findings to ralph-audit-report.md under '## 6. Documentation Consistency'"
    ],
    "acceptance": "All docs cross-referenced, inconsistencies documented, REPO_MAP verified"
  },
  {
    "id": 7,
    "category": "verification",
    "description": "Security and guardrail audit",
    "passes": false,
    "steps": [
      "Step 1: Search for SQL injection risks: grep for string concatenation in SQL queries",
      "Step 2: Search for command injection: grep for subprocess, os.system, eval, exec calls",
      "Step 3: Verify all DB access uses parameterized queries or DuckDB built-in escaping",
      "Step 4: Check for hardcoded credentials or API keys in any file",
      "Step 5: Run python pipeline/check_drift.py and verify all checks pass",
      "Step 6: Check .gitignore covers: .env, gold.db, __pycache__, *.egg-info, credentials",
      "Step 7: Verify connection leak prevention — all DuckDB connections use context managers",
      "Step 8: Append findings to ralph-audit-report.md under '## 7. Security and Guardrails'"
    ],
    "acceptance": "No SQL injection, no leaked secrets, all connections managed, drift check passes"
  },
  {
    "id": 8,
    "category": "cleanup",
    "description": "Generate final audit summary with PASS/FAIL verdicts",
    "passes": false,
    "steps": [
      "Step 1: Read ralph-audit-report.md with all findings from tasks 1-7",
      "Step 2: Create '## Summary' section at the top with overall PASS/FAIL per area",
      "Step 3: List critical findings that need immediate attention",
      "Step 4: List minor findings that are nice-to-fix",
      "Step 5: List areas that are fully compliant with CLAUDE.md",
      "Step 6: Add timestamp and commit count to the summary"
    ],
    "acceptance": "ralph-audit-report.md has complete summary with per-area verdicts"
  }
]
```

## Notes
- Authority hierarchy: CLAUDE.md for code, TRADING_RULES.md for trading logic, RESEARCH_RULES.md for research
- Database is at project root: gold.db (913MB, 6.5M bars)
- 4 instruments: MGC, MCL, MES, MNQ
- DST contamination is a known issue — check compliance with documented rules, don't re-discover it
- Project was recently moved from OneDrive path — watch for stale path references
- Key files: pipeline/dst.py for DST resolvers, trading_app/config.py for thresholds
- Untracked files in git status may include work-in-progress research — don't flag as issues
- The gen_repo_map.py script is at scripts/tools/gen_repo_map.py
