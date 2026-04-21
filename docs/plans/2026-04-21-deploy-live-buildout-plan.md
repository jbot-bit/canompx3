# Deploy-Live Buildout v1 — Stage-Gated Plan

**Date:** 2026-04-21
**Branch:** `deploy/live-trading-buildout-v1`
**Worktree:** `C:/Users/joshd/canompx3-deploy-live`
**Parent commit:** `f567cfe6` (origin/main — `Merge pull request #63 from jbot-bit/research/ovnrng-router-rolling-cv`)
**Directive:** User directive `2026-04-21 — Deploy-Live Buildout, Autonomous Execution` (see session transcript).
**Role:** DEPLOYMENT terminal only. Not research. Not audit. Two other terminals (pr48-sizer, ovnrng-router) run research on separate branches — NEVER touched from this worktree.
**Authority:** `docs/governance/document_authority.md`, `docs/institutional/pre_registered_criteria.md` @ `126ed6b883fbfa7d4930e3a7bdcd5b16afb63902` (pinned for G10), `CLAUDE.md`.
**Co-existence frozen heads (for rollback reference):**
- `origin/research/pr48-sizer-rule-oos-backtest` → `5e768af8`
- `origin/research/ovnrng-router-rolling-cv` → `265d07b1`

---

## Workstream scope (resolved from directive Q1)

- **A — XFA connection** (F-1 hard gate off dormant)
- **B — Broker stack build** (Rithmic → AMP / EdgeClear + Bulenox; account binding for TopStep, Bulenox, MFFU)
- **C — PR #48 participation-shape shadow deploy** (deployment-gating pre-reg, NOT fresh research)
- **E — Prop-rules refresh** (first-party fetches, provenance-logged; dependency for A and B)
- **D — Phase 6e monitoring** — **DEFERRED to v2** but flagged as the **hard prerequisite** for C's real-capital flip (shadow→live). C can shadow without D. C cannot go live without D.

---

## Non-negotiable gates (G1–G10)

Restated from directive. Each gate requires an attached certificate file (`docs/decisions/2026-04-21-pr48-<gate>.md` for Phase 3) OR equivalent evidence block in this plan's stage log. No certificate = no advancement. Performative (un-cited) gates = automatic rollback.

| Gate | Content | Authority |
|---|---|---|
| G1 | Timing-validity certificate — variables × source bar vs decision-time bar | `.claude/rules/daily-features-joins.md` § Look-Ahead Columns; `pipeline.dst.orb_utc_window`; 2026-04-07 postmortem |
| G2 | MinBTL certificate — N, T, E[max_N] from actual data horizon | `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` Theorem 1 |
| G3 | DSR + N̂ certificate — ρ̂ within hypothesis family, Eq. 9 + Eq. 2 | `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` |
| G4 | Chordia t-band — t ≥ 3.79 (no theory) / t ≥ 3.00 (with pre-reg'd literature-cited theory) | `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` |
| G5 | Smell-test clearance — if \|t\| > 7, written justification citing local-extract passages; why it is not look-ahead / scale artifact | `docs/institutional/edge-finding-playbook.md` §3 |
| G6 | Holdout-integrity certificate — pre-reg git SHA + locked evaluation date predating any 2026 OOS look | `docs/institutional/pre_registered_criteria.md` § Criterion 8 (Amendment 2.7) |
| G7 | Negative-controls record — destruction shuffle fails, RNG-null fails, positive control passes | `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` |
| G8 | Mechanism statement — cites local-extract path + page OR labels UNGROUNDED | `docs/institutional/mechanism_priors.md` R1–R8 role menu |
| G9 | Kill criteria pre-committed — written before any run | `docs/audit/hypotheses/2026-04-21-deploy-live-participation-shape-shadow-v1.yaml` |
| G10 | Pre-reg file pins commit SHA of `pre_registered_criteria.md` | pinned SHA `126ed6b8` (this file) |

---

## Stage 0 — Setup (COMPLETED)

**Purpose:** Scaffold the isolated worktree, freeze co-existence heads, emit plan + pre-reg skeleton, create durable task list.

**Blast radius:**
- Files touched: `docs/plans/2026-04-21-deploy-live-buildout-plan.md` (new), `docs/audit/hypotheses/2026-04-21-deploy-live-participation-shape-shadow-v1.yaml` (new, SKELETON).
- DB writes: none.
- Branches touched: `deploy/live-trading-buildout-v1` only.
- Reversibility: full — `git worktree remove` + `git branch -D`.

**Acceptance criteria:**
- [x] Clean tree on origin/main
- [x] Worktree created at `C:/Users/joshd/canompx3-deploy-live`
- [x] Branch `deploy/live-trading-buildout-v1` off `origin/main` (`f567cfe6`)
- [x] `git worktree list` shows all three worktrees; research branch HEADs unchanged
- [x] `context_resolver.py` run — no deterministic match; fallback reads acknowledged
- [x] Plan doc + pre-reg skeleton committed
- [x] Durable task list created via TaskCreate (IDs tracked)

---

## Stage 1 — Prop-rules refresh (workstream E)

**Purpose:** Fresh first-party fetches for TopStep, Bulenox, MFFU. Any rule relied on downstream in A or B MUST carry first-party provenance (URL + fetch date + SHA-256 of fetched HTML). No training-memory fallbacks.

**Literature grounding:**
- `docs/institutional/edge-finding-playbook.md` §10 — pressure-test discipline (no single source).
- N/A for trading literature; this is ops-rule verification.

**Steps:**
1. WebFetch (or Firecrawl when WebFetch returns restricted/blocked): official rules pages for TopStep, Bulenox, MFFU.
2. For each: record fetch URL, timestamp, first-party flag, content hash.
3. Diff against existing `resources/prop-firm-official-rules.md`; flag every divergence — especially: TopStep copier constraints (Live Funded restriction noted in existing file), Bulenox drawdown, any "no Live Funded copying" clause, news-embargo / session restrictions, contract-size / RR / stop caps.
4. Rewrite `resources/prop-firm-official-rules.md`. Every claim either carries first-party provenance OR is labeled `UNVERIFIED`.
5. Commit.

**Blast radius:**
- Files: `resources/prop-firm-official-rules.md` (rewrite).
- DB writes: none.
- Downstream: A (XFA wiring must respect rules), B (account binding).

**Acceptance criteria:**
- Three fetch provenance blocks present (URL + date + SHA-256) OR documented UNVERIFIABLE if fetch blocked.
- Rule diff committed with explicit conflict notes if any.

**Abort trigger:** TopStep copier rules conflict with multi-firm scaling plan — STOP, surface, wait.

---

## Stage 2 — XFA truth-state + root cause (workstream A step 1)

**Purpose:** Determine empirically why the F-1 hard gate is dormant. Do NOT fake a connection. Root cause has 6 possible classes per directive.

**Literature grounding:** N/A (this is ops forensics).

**Steps:**
1. Grep codebase for F-1 gate, XFA, copy_order_router, session_orchestrator gating logic.
2. Trace the gate's enable/disable paths. Identify whether dormancy is: (a) code gap, (b) flag/config switch, (c) credential issue, (d) risk gate deliberately paused, (e) ops dependency on broker build, (f) unknown.
3. Write `docs/decisions/2026-04-21-xfa-root-cause.md` with finding + evidence paths + line numbers.

**Blast radius:**
- Files read-only across `trading_app/live/*`, `trading_app/execution_engine.py`, relevant config.
- Files new: `docs/decisions/2026-04-21-xfa-root-cause.md`.
- DB writes: none.

**Acceptance criteria:**
- Root-cause class identified with code evidence.
- Decision doc committed.

**Abort trigger:**
- Root cause = (c) credentials → STOP, surface. Do not provision.
- Root cause = (d) deliberate pause → STOP, surface. Do not re-enable without explicit user decision.
- Root causes (a)/(b)/(e) → continue to Stage 4.

---

## Stage 3 — PR #48 gate audit (workstream C step 1 — MANDATORY BEFORE SHADOW)

**Purpose:** Re-verify every PR #48 participation-shape claim against live data. No audit metadata trusted. All ten gates G1–G10. G5 is critical — PR #48 t-stats of +7.54 / +9.59 / +11.80 sit squarely in the edge-finding-playbook's §3 "smell-test tripwire" range.

**Literature grounding (written citations required in gate certificates):**
- `docs/institutional/edge-finding-playbook.md` §3 (smell test) — applied in G5.
- `docs/institutional/edge-finding-playbook.md` §4 (single-factor first) — applied in mechanism framing.
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` Eq. 2, Eq. 9, Appendix A.3 — applied in G3.
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` Theorem 1 — applied in G2.
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` — applied in G4.
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` p.7 — walk-forward + WFE ≥ 0.5, applied in G1 + G6.
- `docs/institutional/mechanism_priors.md` R1–R8 — applied in G8.

**Steps:**
1. LIVE-QUERY via `gold-db` MCP (NOT raw SQL) every claim PR #48 makes: t-stats, sample sizes, K_family, WFE, N_fires, OOS perf, per-instrument decomposition (MNQ / MES / MGC), N̂ computation. Each claim tagged inline CONFIRM / CONTRADICT / UNVERIFIABLE.
2. G1 timing-validity: trace every variable "participation shape" uses to its source bar. If any input computed from break bar or later (Q5 class-of-error pattern) → DEMOTE workstream C to post-break-role pre-reg. Halt shadow.
3. Run G2 / G3 / G4 / G5 / G6 / G7 / G8. Each produces a separate certificate file: `docs/decisions/2026-04-21-pr48-<gate>.md`.
4. If all gates pass → fill pre-reg YAML (G9, G10). Pin pre_registered_criteria.md SHA = `126ed6b8` (pre-existing commit before this stage; re-pin if that file moves before C is shadow-deployed).
5. If any gate fails → rewrite YAML as post-break-role pre-reg per `mechanism_priors.md` §4 (R3 / R6 / R7). Halt workstream C. Surface.

**Blast radius:**
- Files new: `docs/decisions/2026-04-21-pr48-g1.md` through `-g10.md` (up to 10 cert files), `docs/audit/hypotheses/2026-04-21-deploy-live-participation-shape-shadow-v1.yaml` (fill from skeleton).
- DB writes: none (read-only MCP only).
- Downstream: workstream C gated on this stage.

**Acceptance criteria:**
- Ten certificates attached OR workstream C demoted with written reason.
- Each claim PR #48 made labeled C/C/U against live data.
- No certificate claim without evidence block.

**Abort trigger:** any gate fail → halt workstream C advance past Stage 6.

---

## Stage 4 — XFA wiring (workstream A step 2) — gated on Stage 2

**Purpose:** Implement the F-1 hard gate live. Round-trip test on broker practice account (paper).

**Prerequisite:** Stage 2 cleared (root cause ∈ {a, b, e}). If root cause ∈ {c, d, f}, Stage 4 does not run.

**Literature grounding:**
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` Ch 10 — position sizing at order-submission gate.
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` — gate framing (will be wired in Stage 6 monitoring, not here).

**Steps:**
1. Design Proposal Gate — write `docs/decisions/2026-04-21-xfa-wiring-design.md` (what / files / blast-radius / approach) BEFORE code. Commit. Proceed (no user approval wait — directive is autonomous).
2. Implement F-1 hard gate end-to-end.
3. Round-trip test on practice account. Capture order-submission log + fill log + local-state reconciliation.
4. Verify: `check_drift.py` passes, tests pass, `grep -r` dead-code sweep, self-review with evidence. All four per CLAUDE.md "Done" definition.

**Blast radius:**
- Files likely touched: `trading_app/live/broker_dispatcher.py`, `trading_app/live/copy_order_router.py`, `trading_app/live/session_orchestrator.py`, `trading_app/execution_engine.py`, config.
- DB writes: paper_trades only (never gold.db writes from this worktree).
- Reversibility: full on the branch; no main pushes until user reviews.

**Acceptance criteria:**
- F-1 gate live.
- Round-trip evidence captured in decision doc.
- `check_drift.py` green, tests green, dead code swept, self-review signed off with evidence.

**Abort trigger:** design doc reveals the wiring requires editing canonical code without separate design proposal → STOP.

---

## Stage 5 — Broker stack (workstream B)

**Purpose:** Wire Rithmic R|API for AMP, EdgeClear, Bulenox account slots. Order routing, fill capture, reconciliation, account binding.

**Literature grounding:**
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` Ch 10 — vol target position sizing (inputs to routing).
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` Appendix A.3 Eq. 9 — N̂ correlation computation for multi-firm duplication.

**Sub-phases (each with design gate + verify cycle):**

- **5.1a Rithmic R|API auth** — Bulenox, AMP, EdgeClear as **separate credential slots**. No credential sharing.
- **5.1b Order routing** — instrument → venue mapping per prop-firm rules from Stage 1.
- **5.1c Fill capture + local persistence** — paper_trades table only (no gold.db writes).
- **5.1d Reconciliation** — local intended state vs broker-reported fills vs gold-db outcomes. Any mismatch = HARD ABORT.
- **5.1e Account binding** — TopStep, Bulenox, MFFU → lane mapping.

**Cost/capital numbers rule:** cite `pipeline.cost_model.COST_SPECS` live. Do NOT quote $2,929/yr/contract NET or any other number from memory.

**N̂ multi-firm correlation gate (5.3 in directive):** before letting multiple firms run the same lane, compute ρ̂ across firm-lane pairs per Bailey-LdP Appendix A.3 Eq. 9. If N̂ across firms < N̂ assumed by the allocator → HALT, surface.

**Blast radius:**
- Files likely touched: `trading_app/live/rithmic/*`, `trading_app/live/projectx/*`, `trading_app/live/broker_factory.py`, `trading_app/live/broker_connections.py`, credential storage location (separate, not committed), `pipeline.cost_model` read-only.
- DB writes: paper_trades only.

**Acceptance criteria per sub-phase:**
- 5.1a: auth handshake logged, separate credential slots verified.
- 5.1b: routing table committed, cites Stage 1 prop rules.
- 5.1c: paper fill captured + persisted.
- 5.1d: reconciliation pass. Mismatch = abort trigger.
- 5.1e: account binding table committed.
- N̂ multi-firm gate: computed + passed OR halted.

**Abort triggers:**
- Reconciliation mismatch.
- N̂ multi-firm gate fail.
- Credential storage ambiguity (shared credentials).

---

## Stage 6 — Participation-shape shadow (workstream C step 2)

**Purpose:** Shadow-deploy PR #48 participation-shape = signal-only logging, NO real capital. Shiryaev-Roberts monitor active. Sharpe alarm wired. N_fires floor enforced.

**Prerequisites:**
- Stage 3 all ten gates passed.
- Stage 4 produced a live F-1 gate.
- Stage 5.1c producing fill-capture logs.

If any missing → HALT.

**Literature grounding:**
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` Eq. 11 + Eq. 17–18 — Shiryaev-Roberts alarm construction.
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` p.7 — walk-forward / forward-test discipline.
- `docs/institutional/literature/carver_2015_ch11_portfolios.md` — portfolio-level diversification implications.
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` Theorem 1 — MinBTL bound on N_fires floor.

**Steps:**
1. Confirm prerequisites.
2. Wire shadow lane → paper_trades table. NO real-capital flag.
3. Wire Shiryaev-Roberts monitor with pre-committed score function per Pepelyshev-Polunchenko Eq. 17–18.
4. Wire Sharpe alarm per pre-reg YAML thresholds.
5. Enforce N_fires floor per MinBTL.
6. Record start timestamp. Lock 2-week minimum observation window (Chan Ch 1 p.7 + Aronson forward-test discipline).

**Blast radius:**
- Files likely touched: `trading_app/live/sr_monitor.py` (wiring), `trading_app/paper_trader.py`, lane config.
- DB writes: paper_trades only.

**Acceptance criteria:**
- Shadow active, signal-only confirmed (no real-capital path).
- SR monitor + Sharpe alarm + N_fires floor wired with pre-reg thresholds.
- 2-week observation lock recorded.

**Live-capital flip block (D dependency):** Pre-reg YAML kill criteria MUST state:
> Real-capital flip from shadow → live is BLOCKED until `docs/plans/2026-02-08-phase6-live-trading-design.md` § 6e monitoring gap is delivered.

---

## Reporting contract

After each stage (not each sub-step), emit:
1. What was done (one line per sub-step).
2. Certificates produced (paths).
3. Files touched (paths + one-line purpose).
4. Live-query evidence for any claim made (command + truncated output).
5. Claims PR #48 / audit made that live data CONFIRMED / CONTRADICTED / UNVERIFIABLE.
6. What's next.
7. Abort triggers hit (or "none").

---

## Success criteria (run-level)

- Worktree created, branch off origin/main clean, research branches untouched. ✓ Stage 0.
- `resources/prop-firm-official-rules.md` refreshed with first-party provenance; rule conflicts surfaced. → Stage 1.
- XFA either (i) live with round-trip evidence logged, or (ii) blocked with a named root cause surfaced. → Stage 2 + Stage 4.
- Broker stack: each sub-phase with paper fill logged + reconciled. N̂ across firms computed + within allocator assumptions. → Stage 5.
- PR #48: either (i) ten certificates + YAML locked + shadow deploy active with SR monitor, OR (ii) demoted to post-break-role pre-reg with reason. → Stage 3 + Stage 6.
- D (Phase 6e) flagged as remaining blocker for live-capital flip. NOT built this run.
- Teardown doc written; NOT executed. Awaiting user command.

---

## Teardown (placeholder — populated by final stage)

Placeholder path: `docs/decisions/2026-04-21-deploy-live-teardown.md` — will contain the exact `git worktree remove` + `git branch -D` commands and state of paper_trades data at teardown time. User runs teardown manually.
