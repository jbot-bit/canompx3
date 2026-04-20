# Stage: retroactive_heterogeneity_audit

**Created:** 2026-04-20
**Owner:** this session
**Classification:** research (non-production-code scope; no stage-gate production lock required)

## Intent

Execute CURRENT-C from `memory/next_session_mandates_2026_04_20.md`:
re-audit memory-cited pooled/universal findings per today's new RULE 14
(`.claude/rules/backtesting-methodology.md`) to surface any hidden
per-cell heterogeneity artefacts before they propagate further into
research or deployment decisions.

## Phases

1. **Enumerate** — table of every memory-cited pooled/universal claim
2. **Triage** — rank by impact × heterogeneity-plausibility; user gate
3. **Per-claim audit** — per-cell breakdown for each candidate
4. **Reconcile** — rewrite memory + result doc for each HETEROGENEOUS finding
5. **Commit** — single commit on `research/retroactive-heterogeneity-audit`
   branch (different branch from today's RULE 3.3/14 bundle)

## Pre-committed decision criteria (RULE 14)

- HETEROGENEOUS = ≥25% of cells show sign opposite to pooled aggregate
  (where cells = natural unit of independence: lane × direction, or
  instrument × session × direction for cross-instrument claims)
- Per-cell N floor = 30 trades to count a cell's sign. Cells with
  N < 30 labeled INSUFFICIENT, not counted as flip or match.
- Pooled delta near-zero (|delta| < 0.02 R) automatically HETEROGENEOUS-suspect
  and must emit breakdown regardless of sign-flip ratio.

## Anti-bias discipline

- Full enumeration in Phase 1 — no early-stop if first claims are clean.
- Expected-sign from theory stated BEFORE each cell's actual sign seen.
- Adversarial dual on each HETEROGENEOUS flag: argue both real-artefact
  and small-N-noise before picking verdict.
- Claims that re-test clean get logged as "CONFIRMED at original scope",
  not reframed.

## Scope boundaries

- Research-layer only (memory, `docs/audit/results/`, research/*.py).
- No production-code changes.
- No touch to other terminal's `research/orb-g5-cross-session-overlap`
  branch or its files.
- Branch: `research/retroactive-heterogeneity-audit` (fresh from origin/main
  after today's bull_short_avoidance commit lands).
