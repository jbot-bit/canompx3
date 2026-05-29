# Project Review OOS Integrity Audit - 2026-05-30

## Scope

Audit the `2026-05-30-project-improvement-review.md` research-integrity BLOCKER findings against the source files, before treating them as real contamination.

## Verdict

MODIFY TOOLING, NOT RESEARCH.

The flagged research artifacts are not proven OOS-tuning violations. The reviewer v1 heuristic over-matched safe audit language:

- "No 2026 holdout tuning"
- "does NOT tune against it"
- "No post-hoc threshold changes"
- "does NOT rescue any of them to deployable status"
- "Powered-OOS graveyard re-sweep"

Those phrases are guardrail or audit labels, not evidence that an OOS result changed a threshold, filter, session, or lane-selection decision.

## Evidence

`docs/audit/hypotheses/drafts/2026-05-29-mgc-cpcv-methodology-audit-PASS1.md`

- Explicitly states CPCV does not carve a sacred 2026 window and does not tune against it.
- Carries K=1992 forward as the honest selection budget.
- States thresholds are unchanged/read-only constants from `pre_registered_criteria.md`.
- Verdict stays non-deployment-grade: `0 VALID, 6 CONDITIONAL, 0 UNVERIFIED, 0 WRONG`.

`docs/audit/results/2026-05-29-powered-oos-graveyard-resweep-and-mgc-wide-scan.md`

- Reports zero graveyard candidates rescued to deployable status.
- Reports zero MGC cells reaching directional-only powered OOS.
- Discloses K_family and warns broader K_global would only make the gate stricter.
- Frames signal-only SR shadow as a pre-reg-only, zero-capital path, not a live parameter change.

`research/powered_oos_graveyard_resweep.py`

- Read-only script; emits stdout only.
- Uses fixed `OOS_FRACTION = 0.30` and reports power tier.
- Does not write DB, profile, allocation, or live config.
- The title phrase "Powered-OOS re-sweep" is not tuning by itself.

`docs/runtime/action-queue.yaml`

- The "short rescue" text is a carried-forward caveat describing directional split behavior at `N_OOS=49`.
- It does not instruct a threshold, filter, session, or lane-selection change.

## Patch

Tightened `scripts/tools/project_improvement_review.py`:

- Removed `sweep` from the OOS-tuning blocker verb set.
- Suppressed explicit prohibition/safety contexts such as "does not", "not", "no", "without", and read-only threshold language.
- Kept direct blockers for actual OOS decision contamination, for example "tune the threshold against OOS to rescue this strategy for promotion".
- Required concrete decision-surface context before treating "rescue" as a blocker.
- Excluded generated `docs/runtime/project_reviews/` reports from rescanning, so stale report text cannot become the next report's evidence.
- Suppressed protected-surface mutation findings for docs, tests, and the read-only `live_readiness_report.py` surface while preserving code-path detection for real calls like `write_live_config(...)`.

## Stop Conditions

Stop and treat as a real research blocker if a future artifact shows any of:

- OOS or holdout data used to choose a threshold, filter, session, entry model, lane, or profile.
- A killed/no-go result reopened without a critique of the original verdict.
- A result promoted without pre-reg/K accounting.
- A live/profile/allocation change derived from these audit-only findings.

## Verification Plan

```powershell
uv run python -m pytest tests/test_tools/test_project_improvement_review.py -q
uv run python scripts/tools/project_improvement_review.py --out docs/runtime/project_reviews/2026-05-30-project-improvement-review.md
uv run python pipeline/check_drift.py --fast
uv run python -m ruff check scripts/tools/project_improvement_review.py tests/test_tools/test_project_improvement_review.py --quiet
```
