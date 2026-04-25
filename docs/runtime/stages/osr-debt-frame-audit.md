# Stage: osr-debt-frame-audit

mode: IMPLEMENTATION
date: 2026-04-25
mode_history:
  - 2026-04-25 RESEARCH (Stage 1 audit drafted + adversarial-audit-cleared + audit doc committed at 3745ba2a)
  - 2026-04-25 IMPLEMENTATION (Stage 2 closeout: decision-ledger entry + cusum_monitor.py docstring clarifier; bundled into PR #106 per user direction "finishing what we started" after verdict was already locked)
worktree: C:/Users/joshd/canompx3-osr-audit
branch: research/osr-shiryaev-roberts-audit
anchor_commit: 73329cd1

scope_lock:
  - docs/audit/results/2026-04-25-osr-debt-frame-audit.md
  - docs/runtime/stages/osr-debt-frame-audit.md
  - docs/plans/2026-04-25-osr-debt-frame-audit-design.md
  # Stage 2 closeout (added 2026-04-25 after Stage 1 verdict locked + auditor cleared):
  - docs/runtime/decision-ledger.md
  # Deferred from Stage 2 closeout (audit recommendation #3, explicitly "optional"):
  #   trading_app/live/cusum_monitor.py — docstring clarifier blocked by parallel-session
  #   RESEARCH/DESIGN stages in other worktrees (stage-gate-guard.py global mode rule).
  #   Defer to a later TRIVIAL fix when parallel stages clear. The decision-ledger entry
  #   is the durable canonical record; the docstring is a nice-to-have surface clarifier.

read_only_sources:
  - docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md
  - docs/institutional/pre_registered_criteria.md
  - trading_app/live/cusum_monitor.py
  - trading_app/live/sr_monitor.py
  - trading_app/sr_monitor.py
  - trading_app/live/performance_monitor.py
  - tests/test_trading_app/test_cusum_monitor.py
  - tests/test_trading_app/test_sr_monitor.py
  - scripts/tools/refresh_control_state.py
  - scripts/tools/project_pulse.py
  - trading_app/lifecycle_state.py
  - docs/runtime/decision-ledger.md
  - HANDOFF.md

## Task

Audit whether the HANDOFF "Next Steps - Active" line "O-SR debt -- `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`. Multi-stage; not autonomous." corresponds to actual missing canonical work, or is stale framing.

Produce one verdict doc at `docs/audit/results/2026-04-25-osr-debt-frame-audit.md` with the ten sections specified in `docs/plans/2026-04-25-osr-debt-frame-audit-design.md` Turn 3.

## Acceptance

1. Audit doc exists at the path above with ten sections: Scope, Literature anchor, Code anchor (CUSUM), Code anchor (SR), Consumer anchor, Criterion 12 anchor, Verdict, Action recommendation, Limitations, References.
2. Every factual claim in the doc has a `file:line` citation back to a primary source.
3. Anchor sections are drafted in full BEFORE the Verdict section is written (bias-prevention discipline).
4. The doc passes the claim-hygiene hook regex (Scope / Verdict / Outputs / Limitations headings present).
5. Isolated-context `evidence-auditor` is dispatched with anchor-by-anchor verification scope, and returns VERIFIED on all four anchors and on the verdict classification. Any REFUTED finding triggers a rewrite, not a patch.
6. Drift check passes idempotently on commit (no canonical surface touched).
7. Pre-commit gates pass (lint / format / drift / claim-hygiene / syntax).
8. Branch is pushed and PR is opened against `origin/main` describing the verdict.
9. Stage doc deletion happens only after audit doc is committed AND auditor has cleared it.

## Out of scope

- Any change to monitor implementations (cusum_monitor.py, sr_monitor.py).
- Any backtest or canonical-layer query against gold.db.
- Empirical adequacy of the ARL approximately 60 days target.
- Parameter-tuning quality of the SR baseline window or threshold.
- Integration smoke-test of either monitor.
- Any decision-ledger / HANDOFF / docstring write -- those belong to Stage 2 (separate design pass).

## Adversarial-audit gate (mandatory)

After draft, dispatch:

- Agent: evidence-auditor (read-only; isolated context).
- Scope: anchor-by-anchor verification of literature anchor, CUSUM code anchor, SR code anchor, consumer anchor, Criterion 12 anchor, and the verdict classification.
- Required output: VERIFIED / REFUTED / PARTIAL / UNVERIFIABLE per anchor, with file-and-line evidence; overall verdict on whether the doc may merge as-is, merge with listed minor fixes, or be pulled back for rework.

## Verification commands run before commit

1. `python pipeline/check_drift.py` (expect: pass; no canonical-surface change).
2. `pytest tests/test_trading_app/test_cusum_monitor.py tests/test_trading_app/test_sr_monitor.py -q` (expect: existing tests pass; confirms intent locks cited in doc are accurate at commit time).
3. Pre-commit gate sweep (lint / format / drift / claim-hygiene / syntax).

## Why RESEARCH not IMPLEMENTATION

This stage produces a verdict doc as its primary artefact. The Stage 2 closeout actions added below are mechanical post-verdict execution (decision-ledger entry locking the MISFRAMED finding + a 3-line docstring clarifier). They are added to scope_lock after the Stage 1 verdict was locked and adversarial-audit-cleared (no design freedom remaining for the action shape — verdict mechanically constrains it).

## Stage 2 closeout (rolled into the same PR)

Per user direction "finishing what we started" after Stage 1 verdict was locked and adversarial-audit-cleared:

- `docs/runtime/decision-ledger.md` — one entry recording the MISFRAMED verdict so the question cannot be reopened without primary-source evidence that a Criterion 12 parameter is actually failing. **Shipped in this PR.**

**Deliberately NOT touched in Stage 2 closeout:**

- `HANDOFF.md` line 30 — volatile surface (auto-written by post-commit hook, actively edited by parallel sessions). The decision-ledger is the canonical durable record per the `work-queue-is-canonical` decision; once the ledger entry exists, the HANDOFF line being stale is harmless and self-correcting.
- Monitor implementations themselves — the audit confirmed both are faithful to their stated intents.

**Deferred from Stage 2 closeout (audit recommendation #3, explicitly "optional"):**

- `trading_app/live/cusum_monitor.py` module docstring clarifier (3-line addition pointing at `sr_monitor.py` and the audit verdict). Blocked at edit time by `.claude/hooks/stage-gate-guard.py` because parallel-session worktrees hold RESEARCH/DESIGN-mode stages on other branches; the hook requires all active stages to be TRIVIAL or IMPLEMENTATION before any production-code edit anywhere. The blocking is a correct safety property (it prevents cross-stage contamination on shared production code), not a bug. The docstring clarifier is a nice-to-have surface fix — the durable closure of the question is the decision-ledger entry, which is on a doc-only path and shipped in this PR. When parallel RESEARCH/DESIGN stages clear, this becomes a one-line TRIVIAL fix.
