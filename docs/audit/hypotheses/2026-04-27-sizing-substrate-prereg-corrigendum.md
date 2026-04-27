# Corrigendum — Sizing-Substrate Pre-Reg `testing_mode` Value

**Companion to:** `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml`
**Triggered by:** institutional code+quant audit 2026-04-27 Finding H (LOW)
**Status:** doctrine-formalization gap acknowledged; pre-reg integrity preserved (no edits to locked YAML)

## The deviation

The pre-reg YAML declares:

```yaml
testing_mode: "diagnostic_descriptive"  # Per pre_registered_criteria.md RULE 10 carve-out: descriptive diagnostics that do not write to validated_setups
```

The canonical enum in `docs/institutional/pre_registered_criteria.md` (line ~106 / Amendment 3.0) defines exactly two values:

- `testing_mode: family` (Pathway A — BH-FDR over the family)
- `testing_mode: individual` (Pathway B — theory-driven K=1)

`diagnostic_descriptive` is not a canonical value. The inline comment cites "RULE 10 carve-out", which actually lives in `.claude/rules/backtesting-methodology.md` RULE 10:

> "Confirmatory audits (T0-T8 on prior survivors) do NOT require new pre-reg — they're not new discovery."

RULE 10 exempts certain diagnostics from *needing* a pre-reg. It does not formally define a third `testing_mode` enum value. The Stage-1 pre-reg author extended the enum unilaterally to flag the descriptive-diagnostic intent.

## Why the deviation does not invalidate the result

The Stage-1 diagnostic operates *within* the spirit of all canonical doctrine:

1. **Read-only over `gold.db`** — `boundary.read_only: true` in the pre-reg. No writes to `validated_setups`, `experimental_strategies`, or `lane_allocation`. `boundary.no_writes_to_validated_setups: true` etc.
2. **Holdout discipline** — `holdout_date: "2026-01-01"`, `raises_on_holdout_row: true`, double-layer enforcement (SQL `WHERE` + Python `RuntimeError`).
3. **Literature grounding** — Carver Ch.7/Ch.8 + Bailey false-strategy + LdP ML4AM, all verified against extracts in `docs/institutional/literature/`.
4. **K-budget pre-committed** — K=48 declared, BH-FDR q=0.05.
5. **Single-pass discipline** — `boundary.single_pass: true`, `reopen_requires_new_mechanism: true`.
6. **Pre-registered before run** — `commit_sha: 57f72f337ce69b8a7b4efde606f2c21509381677`.

The diagnostic is a Pathway-A-style family test (K=48 with BH-FDR) that explicitly does not promote anything to deployment. The substantively-correct canonical declaration would have been `testing_mode: family` with a separate `boundary.diagnostic_only: true` flag or equivalent. The functional methodology was correct; only the metadata label was non-canonical.

## Why this corrigendum is the right close (not doctrine amendment)

Three options were considered:

1. **Amend the pre-reg YAML** — REJECTED. The pre-reg is locked at `commit_sha: 57f72f33` per single-pass discipline. Editing locked pre-regs after run breaks the reproducibility chain and violates `boundary.single_pass: true`.
2. **Amend `pre_registered_criteria.md` to add `diagnostic_descriptive` as a third canonical value** — DEFERRED. A doctrine amendment requires Amendment 3.x semantic-versioning per the file's existing convention, plus updates to `prereg-writer-prompt.md`, plus possibly a drift check. One occurrence of the deviation is not a pattern justifying doctrine churn (per `institutional-rigor.md` §3 "Refactor when you see a pattern of bugs"). Future occurrences may justify formalization; this single instance does not.
3. **Corrigendum subdoc (this file)** — ACCEPTED. Documents the deviation, explains why the result is not invalidated, sets a pointer for future sessions encountering the same gap. Lightweight, honest, audit-trail-compatible.

## Future-session guidance

If a future pre-reg author considers `testing_mode: diagnostic_descriptive`:

- **Prefer `testing_mode: family` + an explicit `boundary.diagnostic_only: true` flag** to express the same intent within the canonical enum.
- If the canonical enum is genuinely insufficient for an emerging diagnostic-test pattern, escalate via Amendment 3.x rather than extending the enum unilaterally.
- The `commit_sha`-locked pre-reg cannot be retroactively edited; document deviations in companion corrigenda like this one.

## Audit closure status

| Finding | Severity | Status |
|---|---|---|
| H — `testing_mode: diagnostic_descriptive` non-canonical | LOW | **CLOSED** via this corrigendum (option 3 above) |

## References

- Pre-reg: `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml`
- Result: `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md`
- Cross-walk: `docs/audit/results/2026-04-27-sizing-substrate-vs-pr51-cross-walk.md`
- Canonical doctrine: `docs/institutional/pre_registered_criteria.md` Amendment 3.0 (lines ~104-106, 672-680)
- Carve-out citation: `.claude/rules/backtesting-methodology.md` RULE 10
- Single-pass principle: `.claude/rules/research-truth-protocol.md` § Pre-reg writer gate
