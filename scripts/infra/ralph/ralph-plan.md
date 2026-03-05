# Ralph Plan — Code Guardian (CG) Deep Audit

## Context
Code Guardian 2-pass style: DISCOVER everything first, then FIX verified issues.
Every task follows:
  Pass 1 (Discovery): Read code, understand intent, find the real issue, articulate PURPOSE.
  Pass 2 (Implementation): Fix → verify (drift + tests) → commit.

Authority hierarchy: CLAUDE.md > TRADING_RULES.md > RESEARCH_RULES.md.
M2.5 findings are UNVERIFIED SUGGESTIONS — verify against actual code before acting.
Never fix a false positive. Never skip tests.

Audit scope: gaps, logic flaws, silences (unexplained/undocumented behaviors),
killers (fatal bugs that corrupt data or produce wrong results), stale code,
dead references, error-handling holes, and architecture drift.

## Tasks

```json
[
  {
    "id": 1,
    "category": "discovery",
    "description": "CG Pass 1 — M2.5 audit all core pipeline files, triage findings",
    "passes": true,
    "steps": [
      "Step 1: Run python scripts/tools/m25_audit.py pipeline/ingest_dbn.py pipeline/build_bars_5m.py pipeline/build_daily_features.py --mode bugs --output scripts/infra/ralph/m25_pipeline.md",
      "Step 2: Run python scripts/tools/m25_audit.py pipeline/dst.py pipeline/cost_model.py pipeline/asset_configs.py --mode bugs --output scripts/infra/ralph/m25_pipeline2.md",
      "Step 3: Read each output file. For every finding: read the ACTUAL code at cited line numbers, trace execution path, check existing guards, classify as TRUE/FALSE-POSITIVE/WORTH-EXPLORING",
      "Step 4: Write triage results to scripts/infra/ralph/ralph-audit-report.md under '## CG-1: Pipeline Findings'"
    ],
    "acceptance": "All pipeline M2.5 findings triaged with TRUE/FALSE-POSITIVE verdict and code evidence. Report written."
  },
  {
    "id": 2,
    "category": "discovery",
    "description": "CG Pass 1 — M2.5 audit trading_app core files, triage findings",
    "passes": true,
    "steps": [
      "Step 1: Run python scripts/tools/m25_audit.py trading_app/outcome_builder.py trading_app/strategy_discovery.py --mode bias --output scripts/infra/ralph/m25_trading1.md",
      "Step 2: Run python scripts/tools/m25_audit.py trading_app/strategy_validator.py trading_app/config.py --mode bias --output scripts/infra/ralph/m25_trading2.md",
      "Step 3: Read each output. Triage every finding: read actual code, verify or reject claim with line-level evidence",
      "Step 4: Write triage to ralph-audit-report.md under '## CG-2: Trading App Findings'"
    ],
    "acceptance": "All trading_app M2.5 findings triaged with code evidence. Report written."
  },
  {
    "id": 3,
    "category": "discovery",
    "description": "CG Pass 1 — M2.5 audit ML module files, triage findings",
    "passes": false,
    "steps": [
      "Step 1: Run python scripts/tools/m25_audit.py trading_app/ml/meta_label.py trading_app/ml/cpcv.py --mode bias --output scripts/infra/ralph/m25_ml1.md",
      "Step 2: Run python scripts/tools/m25_audit.py trading_app/ml/features.py trading_app/ml/evaluate.py trading_app/ml/predict_live.py --mode bugs --output scripts/infra/ralph/m25_ml2.md",
      "Step 3: Triage: read actual code at cited lines. ML 4-gate system (delta_r >= 0, calibration, threshold, importance) is correct by design — do NOT flag individual gates as issues",
      "Step 4: Write triage to ralph-audit-report.md under '## CG-3: ML Module Findings'"
    ],
    "acceptance": "All ML M2.5 findings triaged. Known M2.5 blind spots (4-gate system, atexit, -999 sentinel) not acted on. Report written."
  },
  {
    "id": 4,
    "category": "discovery",
    "description": "CG Pass 1 — gaps audit: cross-file interfaces and silent contracts",
    "passes": false,
    "steps": [
      "Step 1: Run python scripts/tools/m25_audit.py pipeline/build_daily_features.py trading_app/outcome_builder.py --mode joins --output scripts/infra/ralph/m25_gaps.md",
      "Step 2: Manually check: does orb_outcomes always join daily_features on ALL 3 columns (trading_day, symbol, orb_minutes)? Search codebase for any bare 2-column joins.",
      "Step 3: Check for silence killers: any path that returns success after an exception, any counter hardcoded instead of computed from len(CHECKS), any pipeline step that swallows errors",
      "Step 4: Check for double_break look-ahead use anywhere outside of post-session analysis",
      "Step 5: Write gaps findings to ralph-audit-report.md under '## CG-4: Gaps and Silences'"
    ],
    "acceptance": "Cross-file interface gaps checked. No join traps, no silent successes, no look-ahead leaks found or documented."
  },
  {
    "id": 5,
    "category": "implementation",
    "description": "CG Pass 2 — fix all TRUE findings from CG-1 (pipeline)",
    "passes": false,
    "steps": [
      "Step 1: Read ralph-audit-report.md section '## CG-1: Pipeline Findings'",
      "Step 2: For each TRUE finding: implement the minimal fix (do NOT over-engineer, no new abstractions unless necessary)",
      "Step 3: After each fix: run python pipeline/check_drift.py",
      "Step 4: Run python -m pytest tests/ -x -q after all pipeline fixes",
      "Step 5: Stage only changed files and commit with message 'fix: CG pipeline audit findings'",
      "Step 6: Update ralph-audit-report.md to mark each fixed finding as FIXED"
    ],
    "acceptance": "All TRUE pipeline findings fixed. Drift check passes. All tests pass. Commit made."
  },
  {
    "id": 6,
    "category": "implementation",
    "description": "CG Pass 2 — fix all TRUE findings from CG-2 (trading_app) and CG-3 (ML)",
    "passes": false,
    "steps": [
      "Step 1: Read ralph-audit-report.md sections '## CG-2' and '## CG-3'",
      "Step 2: For each TRUE finding: implement minimal fix",
      "Step 3: Run python pipeline/check_drift.py after fixes",
      "Step 4: Run python -m pytest tests/ -x -q",
      "Step 5: Stage changed files and commit with message 'fix: CG trading_app and ML audit findings'",
      "Step 6: Mark fixed findings in ralph-audit-report.md as FIXED"
    ],
    "acceptance": "All TRUE trading_app/ML findings fixed. Tests pass. Commit made."
  },
  {
    "id": 7,
    "category": "implementation",
    "description": "CG Pass 2 — fix all TRUE gap findings from CG-4",
    "passes": false,
    "steps": [
      "Step 1: Read ralph-audit-report.md section '## CG-4: Gaps and Silences'",
      "Step 2: Fix any confirmed join traps, silent successes, look-ahead leaks, or hardcoded counters",
      "Step 3: Run python pipeline/check_drift.py",
      "Step 4: Run python -m pytest tests/ -x -q",
      "Step 5: Commit any fixes with message 'fix: CG gap audit findings'",
      "Step 6: Mark fixed findings as FIXED in report"
    ],
    "acceptance": "All TRUE gap findings fixed or confirmed as non-issues. Tests pass."
  },
  {
    "id": 8,
    "category": "verification",
    "description": "Architecture drift check — CLAUDE.md vs live code",
    "passes": false,
    "steps": [
      "Step 1: Read CLAUDE.md data flow diagram section",
      "Step 2: Verify one-way dependency (pipeline/ never imports trading_app/): grep -r 'from trading_app' pipeline/",
      "Step 3: Verify canonical source usage — no hardcoded instrument lists, session lists, or cost numbers in pipeline/ or trading_app/ (should import from asset_configs/dst/cost_model)",
      "Step 4: Run python pipeline/check_drift.py — count must be computed dynamically, never hardcoded",
      "Step 5: Verify REPO_MAP.md is current: python scripts/tools/gen_repo_map.py",
      "Step 6: Write findings to ralph-audit-report.md under '## CG-8: Architecture Drift'"
    ],
    "acceptance": "No one-way dependency violations. No hardcoded canonical values. Drift check passes. REPO_MAP current."
  },
  {
    "id": 9,
    "category": "cleanup",
    "description": "Final CG report — summary with PASS/FAIL per area, WORTH-EXPLORING queue",
    "passes": false,
    "steps": [
      "Step 1: Read complete ralph-audit-report.md",
      "Step 2: Write '## FINAL SUMMARY' at top with per-area verdict: PASS/FAIL/CLEAN",
      "Step 3: List any WORTH-EXPLORING suggestions (not bugs — improvements for future research)",
      "Step 4: List any findings that were FALSE POSITIVES (so we know M2.5 blind spots)",
      "Step 5: Run final python -m pytest tests/ -x -q to confirm clean state",
      "Step 6: Commit updated report: 'chore: CG audit complete — final report'"
    ],
    "acceptance": "ralph-audit-report.md has complete CG summary. All tests pass. Final commit made."
  }
]
```

## Notes
- CLAUDE.md is the authority — any M2.5 finding that contradicts CLAUDE.md is wrong
- Known M2.5 false positive patterns: DuckDB replacement scans, ML 4-gate system, atexit except pass, -999.0 sentinel, DELETE+INSERT pattern, cross-file guards
- daily_features JOIN: ALWAYS 3-column (trading_day + symbol + orb_minutes). Missing orb_minutes triples rows.
- double_break is LOOK-AHEAD — flag any use as a pre-entry filter as a TRUE bug
- Active instruments: MGC, MNQ, MES, M2K (dead: MCL, SIL, M6E, MBT)
- Entry models: E1, E2 active. E0 purged Feb 2026. E3 soft-retired.
- DB path: gold.db at project root. C:/db/gold.db is scratch only.
- Cost models live in pipeline/cost_model.py COST_SPECS — never hardcode
- Tests take ~8 min — only run after a batch of fixes, not after each individual fix
