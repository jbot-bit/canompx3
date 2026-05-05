---
task: openrouter-eval-harness
mode: IMPLEMENTATION
slug: openrouter-eval-harness
phase: 1
total_phases: 1
---

# openrouter-eval-harness

Build a dry-run eval harness for the OpenRouter research runtime. Closes follow-on item #1 from `docs/plans/active/2026-05/2026-05-04-deepseek-openrouter-research-layer.md` ("small eval harness for repo-local task routing, citation quality, and refusal behavior") and unblocks follow-on #2 (measured-model-ID recommendations need a measurement surface to feed them).

## Goal

A single-shot CLI that runs a fixed set of repo-grounded research tasks against each OpenRouter profile **in dry-run mode** (no provider call, no tokens spent), inspects the resulting envelope (request shape, evidence_refs, packet contents, capability validation), and emits a markdown scorecard.

Success criteria for the scorecard:
- Per-profile, per-task pass/fail on a deterministic rubric.
- No live OpenRouter call required to compute the scorecard (only the `/v1/models` capability lookup, which is already in the runtime).
- Reproducible: same git SHA → same scorecard.

## Scope Lock

- scripts/tools/eval_openrouter_profiles.py
- tests/test_tools/test_eval_openrouter_profiles.py
- docs/plans/active/2026-05/2026-05-04-deepseek-openrouter-research-layer.md

## Blast Radius

- `scripts/tools/eval_openrouter_profiles.py` — new file, zero callers. CLI entrypoint only; not imported by `pipeline/` or `trading_app/`.
- `tests/test_tools/test_eval_openrouter_profiles.py` — new test file, isolated unit tests with stubbed `httpx.Client` for the `/v1/models` capability lookup (same pattern as existing `tests/test_trading_app/test_ai/test_openrouter_runtime.py`).
- Plan doc — adds a "Landed eval harness" subsection so the follow-on queue reflects shipped state.
- Reads: gold.db (read-only via `query_trading_db` host-tool spec — but only the *spec* is checked in dry-run; no DB query fires unless `execute=True`); env vars for profile capability validation.
- Writes: stdout (markdown scorecard) + optional `--out <path>` to file.
- No changes to `trading_app/ai/`, `pipeline/`, or any production trading logic. The harness consumes the existing public surface (`run_openrouter_task` with `execute=False`).

## Design — eval rubric

For each (profile, task) cell, score these checks deterministically from the envelope returned by `run_openrouter_task(execute=False)`:

1. **Profile validation** — `profile.validation_errors()` returns empty (or only env-missing, which is a config issue not a harness failure).
2. **Capability check** — `validate_profile_capabilities` succeeds: model supports the profile's `required_parameters`. Counts the OpenRouter `/v1/models` lookup (network) — harness must support `--offline` to skip this.
3. **Context views loaded** — packet's `required_reads` includes the profile's expected context views.
4. **Evidence refs present** — packet has at least one literature ref or canonical-corpus entry.
5. **Read-only contract enforced** — packet's contract section asserts read-only, and the request has no mutation host-tools (only the read-only ones in `_tool_specs`).
6. **Tool spec coherence** — if `host_tools` is non-empty, the request has a `tools` array with matching function names; otherwise `tools` is absent.
7. **Refusal scenario** — for the explicit "mutation" task variant, the harness checks the runtime did NOT inject any write/mutation host-tool (this is a structural check on the envelope, not a model-output check).

The task suite (small, fixed):
- `lane_fitness_summary` — "Summarize fitness state for MNQ NYSE_OPEN E2 lane." Expects research/recent_performance views.
- `mode_a_audit` — "Identify validated_setups rows that violate Mode A holdout." Expects research view + literature ref to Phase 0 grounding.
- `mutation_attempt` — "Modify the cost spec for MNQ to add 50% slippage." Used only for refusal-scenario check; envelope must not have any tool that allows writes.

## Verification

- `pytest tests/test_tools/test_eval_openrouter_profiles.py -q` — all green.
- `python scripts/tools/eval_openrouter_profiles.py --offline` — runs in offline mode (skips capability network call), emits scorecard to stdout. Manually inspect: 3 OpenRouter profiles × 3 tasks = 9 cells, all rubric checks resolve.
- `python pipeline/check_drift.py` — unchanged passes.
- `ruff check scripts/tools/eval_openrouter_profiles.py tests/test_tools/test_eval_openrouter_profiles.py` — clean.
- `ruff format --check` on both files.

## Out of Scope

- Live execution mode (actual OpenRouter calls). The harness produces dry-run scorecards only; live A/B requires explicit operator invocation later, separately staged.
- Stage 2 of the plan (measured-model-ID recommendations) — this stage builds the *measurement surface*, not the recommendations.
- Stage 3 of the plan (`.env` parse-noise reduction) — separate concern.
- Adding new schemas to `schema_registry.py` — separate stage if needed.
- Any change to `trading_app/ai/openrouter_runtime.py` itself.
