---
task: Fix producer/consumer naming-convention drift between LLM hypothesis proposer and strategy_validator prereg-discipline gate
mode: IMPLEMENTATION
slug: fix-llm-prereg-naming-parity
created: 2026-05-11
updated: 2026-05-11
scope_lock:
  - scripts/research/lhp/yaml_emitter.py
  - trading_app/strategy_validator.py
  - pipeline/check_drift.py
  - tests/test_research/lhp/test_yaml_emitter.py
  - tests/test_trading_app/test_strategy_validator.py
  - tests/test_pipeline/test_check_drift.py
acceptance:
  - "LLM-drafted preregs (`<date>-llm-<slug>.yaml`) pass the validator's Criterion 1 prereg-discipline gate without --allow-legacy-prereg."
  - "Multi-instrument preregs (`scope.instruments: [MNQ, MES]`) are matched correctly by the validator for each instrument."
  - "New drift check refuses to commit a prereg yaml whose name does not satisfy the validator's match contract."
  - "Re-running the 3 audits committed at 82ab4c06 promotes them via the validator without flag overrides."
  - "All companion tests pass (yaml_emitter, strategy_validator, check_drift)."
---

## Blast Radius

- `scripts/research/lhp/yaml_emitter.py` — producer of LLM-drafted prereg filenames. Single function `default_draft_path` (~10 lines). One caller: `llm_hypothesis_proposer.py`. Test file: `tests/test_research/lhp/test_yaml_emitter.py`.
- `trading_app/strategy_validator.py` — consumer of prereg yamls. Function `_check_prereg_yaml_present_for_recent_discoveries` (around lines 2510-2589) globs `<date>-<instr>-*.yaml`. Read by every `--allow-legacy-prereg` decision. NEVER_TRIVIAL per stage-gate-guard.
- `pipeline/check_drift.py` — proposed new check (Check 145) that asserts every committed prereg yaml in `docs/audit/hypotheses/` satisfies BOTH conventions. NEVER_TRIVIAL.
- `docs/audit/hypotheses/2026-05-11-llm-*.yaml` — 3 yamls already committed at SHA 82ab4c06 with the producer's `<date>-llm-<slug>.yaml` naming. They MUST remain valid post-fix (the hypothesis_file_sha stamped in `experimental_strategies` is content-based; rename would not invalidate it, but we should not need to rename).
- Reads: gold.db `experimental_strategies` table (read-only) for the validator gate. Writes: none from this fix.

## Why

PR #259 (LLM hypothesis proposer, landed 2026-05-11) introduced a producer that writes prereg yamls as `docs/audit/hypotheses/<date>-llm-<slug>.yaml`. The pre-existing validator's prereg-discipline gate at `strategy_validator.py:2575` globs `<date>-<instr_lower>-*.yaml` to verify every recent `experimental_strategies` row has a matching prereg yaml. The two conventions disagree silently — every LLM-drafted prereg fails the validator gate even when the prereg is valid, well-formed, locked, and the experimental row carries a correct content-SHA.

This is a producer/consumer parity class bug per `memory/feedback_producer_consumer_parity_class_bug_2026_05_06.md`. Both surfaces are canonical-source code, neither is "wrong" in isolation, and the failure mode is silent until run-time.

The current workaround paths are unacceptable per `institutional-rigor.md`:
- `--allow-legacy-prereg` is documented as "legacy migration only" — using it for new LLM-drafted preregs defeats Criterion 1 of `pre_registered_criteria.md`.
- Renaming the LLM-drafted yamls to satisfy the validator's regex is an ad-hoc workaround, not a structural fix; the next LLM-drafted prereg will hit the same wall.

## Design Decision

**Adopt Option B + Option C from the prior trace:**

**Option B — Make the validator pattern-aware of the LLM-drafted convention AND multi-instrument scope.**

The validator currently encodes a constraint that does not exist in the prereg schema: that the filename includes the instrument. The schema allows `scope.instruments: [MNQ, MES]` (multi-instrument preregs), so encoding a single instrument in the filename is wrong by construction.

Replace the filename-pattern match with a content-aware match:

1. Glob ALL committed prereg yamls in `docs/audit/hypotheses/` for the relevant `<date>` (any naming convention, including `<date>-<instr>-*.yaml`, `<date>-llm-<slug>.yaml`, `<date>-<descriptor>.yaml`).
2. Parse each candidate yaml's `scope.instruments` (or equivalent canonical field per `hypothesis_loader`).
3. Match if `instrument` is in the prereg's declared `scope.instruments`.
4. Preserve the existing carve-outs (`HOLDOUT_GRANDFATHER_CUTOFF`, `--allow-legacy-prereg`).

This is the structurally correct fix: it delegates "does this prereg cover this instrument?" to the prereg's content (its declared scope), not its filename. Filenames remain free-form descriptors for human navigation.

**Option C — Add Check 145 to enforce parity going forward.**

Add `pipeline/check_drift.py::check_llm_prereg_validator_parity()` that:

1. For every committed yaml in `docs/audit/hypotheses/` matching `<date>-*.yaml` with `<date>` ≥ `HOLDOUT_GRANDFATHER_CUTOFF`,
2. Parse via the canonical `trading_app.hypothesis_loader.load_hypothesis_file()` (delegate, never re-encode).
3. For each instrument in `scope.instruments`, simulate the validator's gate: would the validator find this file via its glob?
4. If any instrument is unmatched, flag drift.

This catches the parity bug at commit time, before any future LLM-drafted prereg can fail at validator-run-time.

**Option A (encode instrument in filename) is rejected** because it cannot represent multi-instrument preregs without picking an arbitrary primary instrument, and that primacy would then drift from `scope.instruments` over time — a worse class of bug than the one we are fixing.

## Procedure

### Stage 1 — Validator content-aware match (`trading_app/strategy_validator.py`)

1.1. Read the existing `_check_prereg_yaml_present_for_recent_discoveries` function (`strategy_validator.py:2510-2589`) end-to-end.
1.2. Identify the canonical `hypothesis_loader` API for parsing a prereg file's declared `scope.instruments`.
1.3. Replace the `f"{ds}-{instr_lower}-*.yaml"` glob with a `<date>-*.yaml` glob followed by an in-Python instrument-membership check parsed from each candidate file's content.
1.4. Preserve all existing fail-open paths (DB missing, hyp_dir missing, allow_legacy=True).
1.5. Update or add unit tests in `tests/test_trading_app/test_strategy_validator.py` to cover:
   - Single-instrument prereg with old naming (`<date>-mnq-foo.yaml`) — must still match.
   - Single-instrument prereg with LLM naming (`<date>-llm-foo.yaml`) — must match if scope.instruments includes MNQ.
   - Multi-instrument prereg (`scope.instruments: [MNQ, MES]`) — must match for MNQ AND MES discovery.
   - Wrong-instrument prereg (`scope.instruments: [MES]` but discovery is MNQ) — must NOT match.

### Stage 2 — New drift check (`pipeline/check_drift.py`)

2.1. Add `check_llm_prereg_validator_parity` modelled on existing prereg-aware checks (e.g., `check_pre_registration_present`).
2.2. Use canonical `hypothesis_loader` to parse each committed prereg.
2.3. Register in the check sequence; increment the printed count.
2.4. Add unit test in `tests/test_pipeline/test_check_drift.py` covering: parity-clean repo (PASS), simulated yaml with no `scope.instruments` (drift), simulated yaml with instrument absent from validator's expected set (drift).

### Stage 3 — Re-run the 3 audits committed at 82ab4c06

3.1. With Stages 1 and 2 landed, run `python -m trading_app.strategy_validator --instrument MNQ --testing-mode individual` (per `.claude/rules/validation-workflow.md` flag conventions).
3.2. Confirm all 3 LLM-drafted preregs (`2026-05-11-llm-tokyo-open-atr-vel-ge105.yaml`, `2026-05-11-llm-cme-preclose-orb-vol-16k-o15.yaml`, `2026-05-11-llm-cme-preclose-atr-p30-o15.yaml`) pass the prereg-discipline gate.
3.3. Capture validator output: which strategies promote to `validated_setups` (Phase 4 institutional criteria), which downgrade or fail, and why.
3.4. Write a result doc to `docs/audit/results/2026-05-11-llm-prereg-trio-audit.md` summarizing pre-validation predictions vs actual outcomes (TOKYO_OPEN ATR_VEL_GE105 expected PASS, ORB_VOL_16K expected PASS, ATR_P30 expected PASS but flagged as thin-N).

### Stage 4 — Verification

4.1. Run `python pipeline/check_drift.py` — must show 145 checks passing including new Check 145.
4.2. Run targeted tests: `tests/test_research/lhp/`, `tests/test_trading_app/test_strategy_validator.py`, `tests/test_pipeline/test_check_drift.py`.
4.3. Self-review per `institutional-rigor.md` § 1.
4.4. Adversarial-audit gate per `.claude/rules/adversarial-audit-gate.md` — `strategy_validator.py` is in `trading_app/` truth-layer, classification is judgment, severity HIGH (changes a Phase 4 enforcement gate). Independent-context evidence-auditor pass required before next phase.

## Open questions for sign-off

1. **Hypothesis-loader API:** does `trading_app.hypothesis_loader` already expose a function that returns `scope.instruments` from a yaml path? If not, Stage 1 needs a tiny canonical helper; if yes, delegate to it (Rule #4).
2. **Filename freeform vs convention:** after this fix, the validator no longer requires any filename convention. Should we preserve `<date>-*.yaml` as a soft requirement (date prefix for sortability) and document it in `docs/audit/hypotheses/README.md`? Recommendation: yes, soft requirement, drift check enforces date-prefix only.
3. **Backwards compatibility:** all 124 existing rows in `validated_setups` that came through the old `<date>-<instr>-*.yaml` path still have valid prereg files matching that convention — they continue to pass. Confirmed by inspection of `docs/audit/hypotheses/`.

## What this stage forbids

- Any change to the 3 already-committed LLM-drafted yamls (their content-SHA is locked at 82ab4c06; renaming or editing breaks the audit trail).
- Use of `--allow-legacy-prereg` for the audit re-run.
- Any change to the LLM proposer's filename convention (Option A rejected — see Design Decision).
- Skipping Stage 4.4 adversarial audit (truth-layer enforcement gate change).

## Risks

- **R1 (LOW):** the new content-aware match runs YAML parse on every candidate file per validator invocation. Cost: O(N) parses where N = preregs in directory (~50 today, <500 over project lifetime). Negligible vs the actual validator work.
- **R2 (MEDIUM):** if `hypothesis_loader.load_hypothesis_file` raises on a malformed yaml, the validator gate could mis-fire (fail-open vs fail-closed). Decision: fail-closed — a malformed prereg in the directory is a real bug worth surfacing.
- **R3 (LOW):** the new drift check duplicates some logic already in the validator. Mitigation: the drift check imports from the same canonical helper, not a parallel implementation (Rule #4).
