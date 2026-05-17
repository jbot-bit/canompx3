# Doctrine Amendment Design — Admit No-Theory Pathway-B Preregs Without the Field-Presence Trap

**Date:** 2026-05-17
**Status:** DESIGN PROPOSAL — awaiting user confirmation before implementation
**Author:** Claude (Opus 4.7)
**Scope:** `trading_app/hypothesis_loader.py` + `scripts/research/lhp/static_checks.py` + `pre_registered_criteria.md` **Amendment 3.3** (next unused; 3.0/3.1/3.2 all taken — verified `grep` 2026-05-17)
**Stage:** RESEARCH/DESIGN (no code edits yet)
**Related memory:** `feedback_chordia_theory_citation_field_presence_trap.md`, `feedback_lhp_validator_vs_field_presence_trap_n1.md`
**Drift check:** `check_chordia_result_threshold_matches_prereg` (commit `79f94508`, 2026-05-12) remains the post-run backstop — unchanged.

---

## Context

The hypothesis loader infers a `has_theory: bool` flag from the **field-presence** of `theory_citation` on any hypothesis (`trading_app/hypothesis_loader.py:265-269`). That flag controls the Chordia threshold gate (`trading_app/chordia.py:53-68`): `True → 3.00`, `False → 3.79`.

Two opposing failure modes have been observed:

1. **Prose-flips-strict-to-lenient (2026-05-12 MGC LONDON_METALS):** author wrote `"No filter-mechanism theory citation available, this is empirical..."` into `theory_citation`. Loader read non-empty string → `has_theory=True` → threshold silently relaxed `3.79 → 3.00`. Drift check `check_chordia_result_threshold_matches_prereg` (added same day) now catches the prereg↔result mismatch **post-run**, but the prereg itself still commits.

2. **Omission-blocks-load (2026-05-13 locked K=1 prereg):** author correctly omitted `theory_citation` to claim the strict 3.79 threshold for an honest no-theory Pathway-B test (MNQ_USDATA1000 VWAPMID O30 RR1.0 Chordia unlock). Loader logic at `hypothesis_loader.py:288-295` (Amendment 3.0) **requires** `theory_citation` on every hypothesis when `testing_mode='individual'`, and `static_checks.check_citations_exist` (`scripts/research/lhp/static_checks.py:324-369`) emits fatal `CITATION_MISSING` + `CITATION_NOT_FOUND`. The prereg was rejected to `drafts/*.rejected.txt`; current route-around is the `.draft.yaml` quarantine. The locked K=1 test cannot execute as-authored.

The amendment must thread the needle: **admit no-theory Pathway-B preregs at the strict 3.79 threshold without re-opening the prose-flip trap**. The doctrinal answer is an **explicit boolean metadata field**, not field-presence inference.

## Decision rule (the amendment in one line)

> Replace `has_theory` field-presence inference with an explicit `metadata.theory_grant: bool` field. **No shim. No grandfather inference. No default.** Loader fail-closes if the field is missing or non-bool. All 203 existing preregs are migrated **in the same change** that lands the loader.

This is the **allow-list approach (fail-closed)**. Rejected alternatives:

- **One-cycle shim with deprecation warning:** rejected per user direction 2026-05-17 — silent legacy paths are the trap's habitat; the migration is the amendment's enforcement teeth.
- **Implicit default (`theory_grant=False` if absent):** rejected — recreates presence-inference at the schema level.
- **Enum `theory_basis: none|class_only|pre_registered`:** rejected — over-engineered for n=2 incidents; `class_only` has no consumer.

## What changes (atomic — single change set, no staged migration)

### 1. `trading_app/hypothesis_loader.py` — fail-closed explicit read

**Current (lines 262-295):** field-presence inference + Amendment 3.0 individual-mode citation requirement.

**Amended logic (replaces lines 262-295 entirely):**
```python
# Amendment 3.3: theory_grant is an EXPLICIT bool. No field-presence inference.
# No default. No legacy shim. Missing or non-bool fails closed.
if "theory_grant" not in metadata:
    raise HypothesisLoaderError(
        f"Hypothesis file {path} metadata.theory_grant is REQUIRED (Amendment 3.3, "
        f"2026-05-17). Must be explicit bool: true (theory-grounded, citation required) "
        f"or false (no-theory Pathway-B, strict t>=3.79). No default permitted."
    )
theory_grant = metadata["theory_grant"]
if not isinstance(theory_grant, bool):
    raise HypothesisLoaderError(
        f"Hypothesis file {path} metadata.theory_grant must be a bool, "
        f"got {type(theory_grant).__name__} ({theory_grant!r})."
    )
has_theory = theory_grant  # canonical assignment — no inference

testing_mode = metadata.get("testing_mode", "family")
# (testing_mode validation unchanged)

# Amendment 3.3: theory_grant=true requires theory_citation on every hypothesis
# (carries forward Amendment 3.0's intent; now triggered by explicit bool, not testing_mode).
if theory_grant:
    for i, h in enumerate(hypotheses):
        cite = h.get("theory_citation") if isinstance(h, dict) else None
        if not (isinstance(cite, str) and cite.strip()):
            raise HypothesisLoaderError(
                f"Hypothesis file {path} declares theory_grant=true but hypothesis {i+1} "
                f"is missing theory_citation. Amendment 3.3: theory_grant=true requires "
                f"theory_citation on every hypothesis."
            )

# Cross-rule: theory_grant=false MUST NOT carry any non-empty theory_citation
# (defends against prose-in-field reopening the trap).
if not theory_grant:
    for i, h in enumerate(hypotheses):
        cite = h.get("theory_citation") if isinstance(h, dict) else None
        if isinstance(cite, str) and cite.strip():
            raise HypothesisLoaderError(
                f"Hypothesis file {path} declares theory_grant=false but hypothesis {i+1} "
                f"carries a non-empty theory_citation. Either set theory_grant=true and "
                f"cite a real on-disk extract, or remove the theory_citation field entirely."
            )
```

The previous Amendment 3.0 individual-mode block is **subsumed** by the `theory_grant=true` check above — it no longer hinges on `testing_mode`. (Pathway-A files set `theory_grant=true` and continue to carry citations; nothing changes for them behaviorally.)

### 2. `scripts/research/lhp/static_checks.check_citations_exist` — gate on explicit field

```python
def check_citations_exist(parsed, corpus):
    metadata = parsed.get("metadata", {}) or {}
    theory_grant = metadata.get("theory_grant")

    # Amendment 3.3: theory_grant=false is a valid no-theory prereg. Loader
    # already enforces no-prose-in-field. Static check has nothing to verify.
    if theory_grant is False:
        return []

    # theory_grant=true OR field missing → fall through to existing fatal enforcement.
    # (The loader will fail-close on missing field at load time; this branch covers
    # the static-check pass which can run before load.)
    # (existing per-hypothesis loop + aggregate any_real check unchanged)
    ...
```

The CITATION_NOT_FOUND aggregate check at lines 360-368 keeps its semantics for `theory_grant=true` files — at least one citation must match a real on-disk extract.

### 3. `docs/institutional/pre_registered_criteria.md` — Amendment 3.3

Append a new amendment block (do NOT rewrite Amendment 3.0; supersession-banner pattern per `feedback_doctrine_supersession_banner_pattern.md`):

> **Amendment 3.3 (2026-05-17) — Explicit theory_grant declaration (fail-closed).** Supersedes the field-presence inference embedded in Amendment 3.0. Every prereg `metadata` block MUST declare `theory_grant: bool` **explicitly** — no default, no inference, no legacy shim. `theory_grant=true` requires `theory_citation` on every hypothesis (carrying forward Amendment 3.0's citation-exists requirement); the corpus-match check in `static_checks.check_citations_exist` continues to apply. `theory_grant=false` admits the prereg at the strict Chordia threshold `t ≥ 3.79` (Chordia 2018 empirical bound, no-theory case) and FORBIDS any non-empty `theory_citation` field on hypotheses. Loader rejects on (a) missing field, (b) non-bool value, (c) `theory_grant=false` with prose in any `theory_citation`, (d) `theory_grant=true` with missing/blank `theory_citation`. The 2026-05-12 MGC LONDON_METALS prose-flip and the 2026-05-13 K=1 omission-rejection are both eliminated. All preexisting preregs (203 files as of this amendment) are migrated to add explicit `theory_grant` in the same change set; no grandfather window exists.

### 4. `docs/institutional/hypothesis_registry_template.md` — schema row

Add `metadata.theory_grant: bool` to the required-field table: "Explicit bool, **required**. False = no-theory (strict 3.79). True = theory-grounded, requires `theory_citation` on every hypothesis. No default."

### 5. `docs/prompts/prereg-writer-prompt.md` — author guidance

Update § FORBIDDEN to add: "Omitting `metadata.theory_grant`. Writing prose into `theory_citation` when `theory_grant=false`. Citing extracts when `theory_grant=false`."

Update failure-mode table at line 303:
- `Any prereg missing metadata.theory_grant` → loader fail-closed (Amendment 3.3)
- `Pathway B with theory_grant=true but missing citation` → loader fail-closed
- `theory_grant=false with prose in theory_citation` → loader fail-closed (Amendment 3.3)

### 6. Explicit migration of all 203 existing preregs — same change set

Migration logic (deterministic, scripted — single commit):

| Existing state of prereg | `theory_grant` value to insert |
|---|---|
| At least one hypothesis carries a non-empty `theory_citation` that matches a `docs/institutional/literature/` extract | `true` |
| No hypothesis carries a non-empty `theory_citation` | `false` |
| Some hypothesis carries non-empty `theory_citation` that does **not** match any extract (prose / placeholder) | **HALT migration** — author must manually classify before commit; do not silently coerce. |

**Migration script:** new one-shot `scripts/tools/migrate_preregs_amendment_3_3.py` — reads each yaml, classifies via the table above, inserts `theory_grant` at top of `metadata`, **deletes** any `theory_citation` field on hypotheses when `theory_grant=false` (cross-rule compliance), writes back with stable key ordering preserved via `ruamel.yaml`. Runs once, lands in same commit, then the script is deleted (or marked DEPRECATED per `feedback_deprecation_notice_not_removal_2026_05_01.md` — preferred: delete). Any HALT cases are listed in the commit message and resolved before commit.

### 7. Locked 2026-05-13 K=1 prereg — promote in same change

The rejected draft at `drafts/2026-05-13-llm-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.rejected.txt` is promoted to `docs/audit/hypotheses/2026-05-13-llm-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.yaml` with `metadata.theory_grant: false` and any `theory_citation` fields removed. The locked K=1 test then loads cleanly. **Stage 2 family scan (K=20) is explicitly out of scope for this change.**

## Files touched

| File | Change | Reuses canonical |
|---|---|---|
| `trading_app/hypothesis_loader.py` | Replace lines 262-295 with fail-closed explicit `theory_grant` block | `HypothesisLoaderError`, existing metadata access |
| `scripts/research/lhp/static_checks.py` | `check_citations_exist` short-circuit on `theory_grant=False` | `CheckFailure`, `citation_exists` |
| `docs/institutional/pre_registered_criteria.md` | Append Amendment 3.3 (supersession-banner pattern) | — |
| `docs/institutional/hypothesis_registry_template.md` | Add `theory_grant` schema row | — |
| `docs/prompts/prereg-writer-prompt.md` | FORBIDDEN + failure-mode table | — |
| `tests/test_research/test_hypothesis_loader.py` | 6 unit tests covering each rejection path | Existing pytest harness |
| `tests/test_research/test_static_checks.py` | `theory_grant=false` short-circuit test | Existing harness |
| `scripts/tools/migrate_preregs_amendment_3_3.py` | One-shot migration (203 files); deleted in same commit | `ruamel.yaml` (already vendored) |
| `docs/audit/hypotheses/*.yaml` (203 files) | Add `metadata.theory_grant: bool` per migration table | — |
| `docs/audit/hypotheses/2026-05-13-llm-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.yaml` | Promoted from `drafts/`, `theory_grant: false` | — |

## Blast radius

- **Loader:** every `load_hypothesis_file` call site (Chordia runner, audit script, future K=1 verify scripts) starts requiring `theory_grant`. Migration commit guarantees no in-tree file fails to load.
- **Static check:** consumed by `propose-hypothesis` skill at draft-promotion time. New behavior: `theory_grant=false` admits without citation; everything else preserved.
- **Drift check `check_chordia_result_threshold_matches_prereg`:** unchanged. Post-run backstop verifies declared↔applied threshold consistency.
- **Downstream consumers of `has_theory`:** `trading_app/chordia.py:68`, `research/chordia_revalidation_deployed_2026_05_01.py`, `research/audit_chordia_queue_false_exclusions.py` — all read the loader's computed bool; no consumer changes needed.
- **No live capital impact.** Deployed Chordia lanes already cleared strict 3.79 per `docs/audit/results/2026-05-12-deployed-lanes-chordia-strict-379-exposure-audit.md`. Amendment can only admit honest no-theory preregs at the strict threshold; cannot lower any deployed gate.

## Verification

Per user direction: **stop after tests + drift + diff summary**. No execution of the unlocked K=1 prereg; no Stage 2 authoring.

1. **Migration dry-run:** run `scripts/tools/migrate_preregs_amendment_3_3.py --dry-run` — print classification per file (true/false/HALT), confirm zero HALT cases (or resolve them).
2. **Migration apply:** run without `--dry-run`; confirm 203 files modified, `git diff --stat docs/audit/hypotheses/` matches expectations.
3. **Unit tests pass:** `pytest tests/test_research/test_hypothesis_loader.py tests/test_research/test_static_checks.py -v` — all rejection paths green:
   - missing `theory_grant` → HypothesisLoaderError
   - non-bool `theory_grant` (string, int, None) → HypothesisLoaderError
   - `theory_grant=true` + missing citation → HypothesisLoaderError
   - `theory_grant=false` + prose in citation → HypothesisLoaderError
   - `theory_grant=true` + valid citation → loads, `has_theory=True`
   - `theory_grant=false` + no citation field → loads, `has_theory=False`
4. **Loader smoke test on all migrated preregs:** `python -c "from trading_app.hypothesis_loader import load_hypothesis_file; from pathlib import Path; [load_hypothesis_file(p) for p in Path('docs/audit/hypotheses').glob('*.yaml')]"` — zero errors.
5. **Drift check pass:** `python pipeline/check_drift.py` — full suite green; `check_chordia_result_threshold_matches_prereg` still fires on injected mismatch (synthetic test).
6. **Diff summary:** report net lines changed across (loader, static_check, doctrine, template, prompt, tests, migration script, 203 preregs, 1 promoted prereg). Stop. Report to user.

## What this design does NOT include

- No execution of the unlocked 2026-05-13 K=1 prereg (out of scope per user direction).
- No Stage 2 family scan authoring (out of scope per user direction).
- No shim, no grandfather window, no legacy inference path.
- No change to Chordia thresholds 3.00/3.79.
- No change to BH-FDR Pathway-A path.
- No change to the 12 locked pre-registered criteria (Amendment 3.3 adds a schema field; does not relax any criterion).
- No deletion of the existing `check_chordia_result_threshold_matches_prereg` drift check.

## Token-efficiency notes

- Single durable artifact in `docs/plans/` (this file).
- Implementation: ~80 production-diff lines + ~30 doctrine prose lines + 203 migrated yamls (mechanical insertion of one line each via script) + 1 promoted yaml.
- No agent fan-out for implementation.
- Stop-after-verify per user direction; no post-implementation research, no Stage 2 scan.
