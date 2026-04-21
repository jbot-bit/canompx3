# MNQ rel_vol biphasic — observation record (SHELVED)

**Status:** SHELVED pending v3 Deployment-First Reset Phases A-C
**Date observed:** 2026-04-21
**Date shelved:** 2026-04-21

---

## Why this is an observation record, not a draft pre-reg

The 2026-04-21 Deployment-First Reset v3 directive (Codex terminal) declares:

> Operating mode: evidence-first. No research scans, no new MNQ variants, no statistical-methodology refinement, no shadow deploys, no config edits to ALL_FILTERS / StrategyFilter / live_config, until Phases A–C are complete.

A draft pre-reg proposing new MNQ variant tests would violate the directive's letter even if DRAFT_PENDING_REVIEW-flagged. This document instead preserves the observation for future use without proposing work.

## Observation

The PR #59 sizer re-audit (commit `ec8198f3`) raw OOS quintile table shows MNQ rel_vol quintile ExpR as **inverted-U**, distinct from MGC monotonic-up and MES binary-Q5 shapes:

| Instrument | Q1 | Q2 | Q3 | Q4 | Q5 | Shape |
|---|---:|---:|---:|---:|---:|---|
| MNQ | +0.016 | +0.035 | +0.064 | +0.142 | **+0.010** | Q4-peak, Q5-crash (inverted-U) |
| MES | -0.199 | -0.176 | -0.083 | -0.147 | +0.112 | binary (only Q5 positive) |
| MGC | -0.026 | -0.023 | +0.033 | +0.143 | +0.263 | monotonic-up |

The crash from Q4 (+0.142) to Q5 (+0.010) on MNQ is consistent with a **participation-exhaustion** mechanism — very high pre-ORB rel_vol signals the session's directional move is already priced in. Distinct from MGC's participation-continuation mechanism.

## Why this observation is NOT verified

- The OOS quintile table was produced by the PR #59 re-audit, which used the 2026-01-01..2026-04-21 OOS window.
- That window is now CONTAMINATED for confirmatory purposes — the shape was discovered IN the OOS data, not pre-registered.
- No IS-only (pre-2026) confirmation has been done. The shape could be a noise artifact at N=137 (MNQ Q5).
- Codex's filter-form runner (commit `96b9e358`) explicitly excluded MNQ from its MES/MGC filter-form pre-reg because MNQ's Spearman p=0.12 (linear rank signal not significant).

## Why it's worth preserving

The inverted-U shape is mechanistically distinct from the monotonic/binary shapes on MES/MGC. If MNQ rel_vol is revisited post-v3, the exhaustion mechanism is a different hypothesis class from the Q5-only filter that Codex is testing on MES/MGC. It would require:

1. **IS-only verification** — does the shape replicate on `trading_day < 2026-01-01`?
2. **Per-session heterogeneity check** — is the Q4-peak Q5-crash stable across sessions or is it a pooled artifact (one session driving)?
3. **Post-break-role reframing** (per v3 G1 template `docs/audit/remediations/gate-templates/G1-timing-validity.md`) — rel_vol uses break-bar volume, which is E2-incompatible as pre-entry filter. Any MNQ variant must be framed as post-break role (R3 size / R6 entry-model switch / R7 confluence / R8 post-break conditioner).

## Conditions for un-shelving

1. **v3 Phase A (truth-state snapshot) completed** with Codex's Phase A2 verifying whether MNQ rel_vol has already been closed or is still a live research-queue item.
2. **v3 Phase B deployed-lane verdicts issued** so we know what's live to route around.
3. **v3 Phase C vestigial sweep completed** including any ML-V residue or stale MNQ overlays that could contaminate a fresh MNQ pre-reg.
4. **v3 Phase E2 Q5 rel_vol pre-reg (MES/MGC only)** executed and verdicted, to see whether MNQ's distinct mechanism still warrants its own pre-reg or whether MNQ rel_vol closes entirely.

## What is NOT proposed by this document

- **No pre-reg** — this is shelved, not locked, not a draft to execute.
- **No script** — no `research/mnq_participation_exhaustion_*.py` written.
- **No variant** — no proposal to test H1/H2/H3 forms or any specific parameterization.
- **No deploy path** — well upstream of any deployment question.

## Related artifacts

- PR #59 re-audit doc: `docs/audit/results/2026-04-21-pr48-sizer-rule-skeptical-reaudit-v1.md` (contains the raw MNQ quintile table)
- Role-selection meta-pre-reg: `docs/audit/hypotheses/draft-2026-04-21-rel-vol-role-selection-meta-v1.yaml` (MNQ × FILTER cell lists this biphasic observation as the blocker)
- Sweep doc classifying the same-OOS peek contamination: `docs/audit/results/2026-04-21-post-hoc-rejection-sweep.md`

## Provenance

- Originally authored as `draft-2026-04-21-mnq-participation-exhaustion-v1.yaml` during the 2026-04-21 Claude-terminal session.
- Demoted and reframed as this observation record to respect v3 directive's "no new MNQ variants" clause.
- Original draft deleted from the working tree; this document supersedes it.
