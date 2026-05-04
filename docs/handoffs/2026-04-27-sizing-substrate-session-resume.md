# Sizing-Substrate Session — Resume Note (2026-04-27)

**Use:** drop into next Claude session after terminal clear so context isn't lost. Shows what was done, what's queued, and where to pick up.

## Branch state
- Branch: `chore/freshness-bumps` — 12 ahead of `main`. NOT pushed, NOT PR'd.
- All commits passed pre-commit gauntlet (8/8 every time).

## What was done

ChatGPT pushed thesis: "convert binary filters into continuous sizing." Verified parts ground-truthed via direct repo+/resources/ pass; built Stage-1 falsifier; ran it.

### Verdict: SUBSTRATE_WEAK
- 2 of 6 lanes pass (EUROPE_FLOW + TOKYO_OPEN, both via `rel_vol_session`)
- Pre-reg required ≥3 lanes for global confirmation
- 0 cells stage-2-eligible (both PASS cells are UNSTABLE on Carver Ch. 7 fn 78 stability gate)
- Tier-A (deployed-filter substrates) all FAIL → deployed binary filters DO capture their substrates' predictive content
- 2 cells INVALID by RULE 1.2 lookahead audit (`overnight_range_pct` on TOKYO_OPEN/SINGAPORE_OPEN — pre-17:00 Brisbane sessions)

### Decision: park sizing thesis
- No Stage 2 unless someone writes fresh per-lane pre-reg with new mechanism citation
- NO-GO entry NOT added (SUBSTRATE_WEAK ≠ THESIS_KILLED)

## 12 commits (chronological)

```
caa0980c docs(plans): design v0.1 sizing-substrate diagnostic
c0d18bca docs(plans): design v0.2 — fix lit citations + add 6 gaps
df1cae8a docs(plans): implementation plan
571848e7 chore(audit): gitignore scratch extracts
57f72f33 research(sizing): lock pre-reg YAML (K=48)
1bcd65b7 research(sizing): record pre-reg commit_sha
d812d5cf research(sizing): pure-function gates + 20 tests (TDD)
e7ad48cb research(sizing): SQL loader + main runner
9f8a1873 fix(sizing): SQL DATE-param + cost_ratio_pct formula
9d63ea4c fix(sizing): enforce overnight_* lookahead validity gate (RULE 1.2)
60556b6d research(sizing): Stage-1 result — verdict=SUBSTRATE_WEAK
19a7a534 fix(sizing): post-review hardening (stage2_eligible flag + B align + tests)
```

## Files of record

- **Design** (canonical of approach): `docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md` v0.2
- **Implementation plan**: `docs/plans/2026-04-27-sizing-substrate-diagnostic-implementation-plan.md`
- **Pre-reg locked**: `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml`
- **Diagnostic script**: `research/audit_sizing_substrate_diagnostic.py` (~470 lines)
- **Tests**: `tests/test_research/test_audit_sizing_substrate_diagnostic.py` (29 tests, all green)
- **Result MD**: `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md`
- **Result JSON twin**: `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json`

## Key parameters (locked in pre-reg)

- K = 48 (6 lanes × 8 features)
- BH-FDR q = 0.05 (Pathway A canonical)
- Bootstrap seed = 42, B = 10000
- Power floor N ≥ 902 (for ρ=0.10 at t≥3.00, Harvey-Liu-Zhu with-theory)
- NULL coverage max 0.20
- Stability SD-variation max 0.50
- Linear-rank weights {0.6, 0.8, 1.0, 1.2, 1.4} mean=1.0 dollar-vol-matched
- Holdout sealed at 2026-01-01 (script raises on any 2026 row)

## Code review outcome (final)

APPROVE WITH CHANGES → all important findings landed in `19a7a534`:
- I-1/I-6: `stage2_eligible_flag` added; UNSTABLE PASS cells fail-closed for Stage-2
- I-2: split-half delta bootstrap aligned to pre-reg B=10000 (was hardcoded 1000)
- I-5: result MD now reports effective tested K = 46 (after lookahead INVALID gating)
- M-5 partial: 6 lookahead-gate tests + 3 stage2-eligibility tests added (29 total)

## QUEUED audits (user wants these next, after terminal clear)

User submitted two follow-up audit prompts before terminal-clear request. Verbatim text below.

### A. Institutional code + quant audit reviewer

```
Operate as an institutional code + quant audit reviewer.

Goal:
Audit this codebase/change for correctness, hidden logic gaps, bias, stale assumptions, and missing safeguards.

MANDATORY GROUNDING
- Actually read relevant project resources, not filenames/metadata.
- Start with:
  - CLAUDE.md
  - RESEARCH_RULES.md
  - TRADING_RULES.md
  - PROJECT_REFERENCE.md
  - HANDOFF.md
  - relevant docs/specs/prompts
  - relevant local literature extracts/resources if methodology is involved
- Quote/file-reference the exact passages used.
- If a source was not read, say NOT READ.
- If a claim is not grounded, mark UNSUPPORTED.

AUDIT ORDER

1. SOURCE-OF-TRUTH MAP
2. CODE PATH REVIEW
3. LOGIC + BIAS CHECK
4. RESOURCE GAP AUDIT
5. TEST + GUARDRAIL AUDIT
6. BLAST RADIUS

OUTPUT
1. VERDICT — PASS / PASS_WITH_RISKS / FAIL / BLOCKED
2. LOAD-BEARING FINDINGS
3. GAPS / SILENCES
4. TESTS TO ADD
5. NEXT ACTION

Rules: Be adversarial. No skimmed sources. No assumptions. No broad refactors. No code changes unless explicitly approved.
```

Recommended dispatch: `evidence-auditor` subagent (purpose-built for "main thread biased toward its own prior work"). Charter exactly maps to the 6 audit areas above.

### B. Institutional discovery-mode researcher

```
Operate as an institutional trading researcher in DISCOVERY mode.

Goal:
Explore honestly from first principles to find where edge may exist, where it does not, what is worth testing next, and what should be parked — without tunnel vision, bias, or premature narrowing.

MANDATORY GROUNDING
- Discovery truth must come from canonical layers only:
  - bars_1m
  - daily_features
  - orb_outcomes
- Derived layers (validated_setups, edge_families, live_config, docs, memory, summaries) may orient, but are NOT proof.
- Ground methodology in local project canon and local academic PDFs/resources first.
- Separate every statement into:
  - MEASURED repo truth
  - GROUNDED prior/literature
  - INFERRED hypothesis
- If not grounded, mark UNSUPPORTED.

STAGE 1 — DEFINE THE OBJECT CORRECTLY
1. Unit of analysis (pre-trade / post-trigger / execution / portfolio / retrospective label)
2. Information horizon (decision-time vs after-trigger vs after-trade vs after-session vs retrospective)
3. Role mapping (standalone / filter / conditioner / allocator / confluence / diagnostic-only)
4. Path type (current-stack / architecture-change / new-data / dead/tautological / impl-only)

STAGE 2 — FAIR-FIGHT EXPLORATION
1. Ask the right question (give 2–3 better-framed alternatives)
2. Alternative honest paths (3–5 framings + plausibility + evidence + park/test)
3. Smallest honest test family (K-budget honest)
4. Required controls (baseline, eligibility, costs, no-leakage, holdout, multiple-testing, mechanism)

STAGE 3 — WHERE EDGE MAY LIVE
- local vs global
- conditional vs standalone
- interaction vs isolated
- signal vs implementation
- execution vs alpha
- portfolio contribution vs cell-level prettiness

STAGE 4 — PREMATURE KILL / FALSE SURVIVOR CHECK
STAGE 5 — DISCOVERY DECISION

OUTPUT
1. Correct object (unit / horizon / role)
2. Discovery map (promising / non-promising / dead)
3. Honest next tests
4. Park / kill list
5. Final recommendation: CONTINUE / NARROW / REDESIGN / PARK / KILL

RULES
- No tunnel vision / post-hoc rescue / implementation optimism / derived-layer proof
- No collapsing "not standalone" into "dead"
- No collapsing "not yet proven" into "alive"
```

User explicitly said *clear terminal before* dispatching these. Don't auto-launch — wait for user prompt.

## Outstanding considerations

1. **Branch finalization** — `chore/freshness-bumps` has 12 commits unrelated to its original purpose. Either rename branch (e.g. `research/sizing-substrate-stage1-2026-04-27`) before PR, or squash + merge directly to main, or PR as-is with mixed commits.

2. **AFML acquisition optional** — sigmoid bet-sizer recipe needed for Stage 2 if substrate were ever confirmed. Currently NOT in `resources/`. Pre-reg explicitly defers; no action unless Stage 2 is opened on some lane in the future.

3. **MEMORY.md update** — Apr 27 entry added to `memory/recent_findings.md`. The user-level memory at `C:\Users\joshd\.claude\projects\C--Users-joshd-canompx3\memory\` is the source; the in-repo `memory/MEMORY.md` is project-tracked but the index is mostly maintained by hand.

## Resume instructions

1. Read this file.
2. Run the queued audit (A or B or both).
3. Decide branch finalization (rename / merge / leave).

User-level Brisbane time at session save: 2026-04-27.
