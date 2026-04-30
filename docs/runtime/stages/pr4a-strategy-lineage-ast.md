---
task: pr4a-strategy-lineage-ast
mode: IMPLEMENTATION
scope_lock:
  - trading_app/strategy_lineage_ast.py
  - tests/test_trading_app/test_strategy_lineage_ast.py
blast_radius: NEW leaf module. trading_app/strategy_lineage_ast.py exports ColumnRef NamedTuple + scan_python_for_column_refs(path, allowlist) -> list[ColumnRef]. Pure function, no DB, no I/O beyond reading the supplied path, no callers in pipeline/ or trading_app/ runtime (verified by grep — file does not yet exist). PR-4b (post-MVP) will be the first importer; until then the module is invokable only via `python -m trading_app.strategy_lineage_ast --scan <path>`. Tests are isolated unit tests using tmp_path fixtures, no gold.db touch. Halt-condition for the broader CRG-maximization effort already PASSED (76.7% median token savings on PR-3 eval baseline 2026-04-30, MRR 0.325). No production data flow, no schema change, no canonical layer touched.
agent: claude
updated: 2026-04-30
---

# PR-4a: AST scanner (pure function, leaf module)

**Status:** IMPLEMENTATION
**Date:** 2026-04-30
**Worktree:** `canompx3-strategy-lineage-ast` on `feature/strategy-lineage-ast`
**Plan:** `docs/plans/2026-04-30-crg-maximization-v2.md` § PR-4a (commit `1e868ac1`)

## Why this stage exists

PR-4a is the leaf-module half of v1's PR-4 split. It ships a single
pure-function module that scans Python source for canonical column references
using a hybrid AST + regex algorithm. Independently useful (contamination
detection — e.g., catching E2 cohort scripts that touch `break_dir` /
`break_bar*` post-entry) and independently revertable. PR-4b imports it later.

## Approach

Hybrid algorithm per v2 plan:

1. Compile a regex alternation from the column allowlist; if the file contains
   no allowlisted token, skip AST entirely (cheap fast-path).
2. `ast.walk` over the parsed tree, classify three node kinds:
   - `Subscript` with str-`Constant` slice → `df["col"]` hit (`ast_literal`).
   - `Attribute` whose `.attr` is in the allowlist → `df.col` hit (`ast_literal`).
   - `Constant` of `str` type → run the regex on the literal to catch SQL
     fragments and assigned column-name strings (`ast_literal`).
3. If `ast.parse` raises `SyntaxError`, fall back to line-by-line regex and
   emit `evidence="regex_fallback"` rows so weaker evidence is labeled, never
   silently merged with strong evidence (institutional rigor §4).

CLI: `python -m trading_app.strategy_lineage_ast --scan <path>` walks `*.py`
files, prints `<file>:<lineno>\t<column>\t<evidence>` per hit. Default
allowlist is a small built-in baseline (`atr_20`, `rsi_14`, `break_dir`,
`break_bar_high`, `break_bar_low`) so the CLI smoke-tests without DB access;
real callers (PR-4b) will supply the allowlist from `DESCRIBE daily_features`.

## Acceptance criteria (v2 plan § PR-4a)

- 6 unit tests pass (`pytest tests/test_trading_app/test_strategy_lineage_ast.py`)
- `python pipeline/check_drift.py` exit 0
- `python -m trading_app.strategy_lineage_ast --scan research/` exit 0 with
  column-ref report on stdout
- `grep -rn "INSERT\|UPDATE\|DELETE\|CREATE" trading_app/strategy_lineage_ast.py`
  returns empty (no DB writes)
- Self-review: AST disagreement-with-regex case covered (per Bias 3 in v2 plan
  § Bias audit — regex_fallback edges remain visible and labeled)
