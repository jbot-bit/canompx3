# Certificate Templates — G1-G10

Non-negotiable gates from the 2026-04-21 Deployment-First Reset v3 directive. Every advancement decision (KEEP / DEGRADE / RETIRE / SHADOW_DEPLOY) in Phases B-E requires all applicable certificates.

## Structure

Each template is a self-contained markdown skeleton with:
- Header with canonical-data evidence requirements
- Gate-specific fields (live-query command + truncated output expected)
- Self-audit checklist
- Failure disposition (what to do if the gate fails)

## Templates

| Gate | Template | Use when |
|---|---|---|
| G1 | `G1-timing-validity.md` | Any pre-entry filter, sizer, or signal uses a computable variable |
| G2 | `G2-minbtl.md` | Any discovery scan / hypothesis family with K > 1 |
| G3 | `G3-dsr-neff.md` | Any candidate being promoted past RESEARCH_SURVIVOR |
| G4 | `G4-chordia-band.md` | Any pre-registered hypothesis with an IS t-stat gate |
| G5 | `G5-smell-test.md` | Any finding with `|t| > 7` or `Δ_IS > ±0.6` |
| G6 | `G6-holdout-integrity.md` | Any candidate with 2026 OOS evidence |
| G7 | `G7-negative-controls.md` | Any candidate advancing to shadow or deploy |
| G8 | `G8-mechanism-statement.md` | Every pre-reg + every deployed lane re-audit |
| G9 | `G9-kill-criteria.md` | Every pre-reg BEFORE the first run |
| G10 | `G10-pre-reg-commit-pin.md` | Every pre-reg that references pre_registered_criteria.md |

## Usage

1. Copy the relevant template(s) to `docs/audit/certificates/<YYYY-MM-DD>-<candidate-slug>/`
2. Fill each field with live-query evidence (command + truncated output) — NO memory, NO stale docs
3. Every "evidence required" row must be non-empty or marked UNAVAILABLE with explanation
4. Attach to the decision doc in `docs/decisions/<YYYY-MM-DD>-<slug>.md`
5. Reject any decision doc claiming "gates passed" without attached certificates — per v3 directive, that's performative self-review

## Literature anchors

Every certificate cites its backing literature extract. If the local extract doesn't exist under `docs/institutional/literature/`, the certificate is UNGROUNDED and the advancement is blocked per CLAUDE.md Local Academic / Project-Source Grounding Rule.

## Drift protection

When pre_registered_criteria.md is amended (e.g., Amendment 2.1 DSR cross-check, Amendment 2.4 banded deployability), templates inherit the amendment automatically because they cite the file by path + commit SHA pinned per G10.
