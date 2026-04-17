# Docs Authority & Overlap Audit — 2026-04-17

**Scope:** documents that claim authority over the same domain, or duplicate/overlap in ways that can actively mislead decisions *right now*.
**Explicitly out of scope:** stale/orphan cleanup (appendix only), doc-style nits.
**Action:** audit only — zero edits performed. Findings ranked by severity.

**Source of truth for authority roles:** `docs/governance/document_authority.md` (2026-04-11).
**Method:** three parallel read-only reviews of root-level MDs, research/institutional doctrine, and plan/handoff surfaces. Governance file read in full, target docs spot-checked via keyword grep + section reads.

---

## Severity Legend

- **CRITICAL** — Actively misleading trading or research decisions *today*. Fix before next material decision.
- **IMPORTANT** — Ambiguous single-source-of-truth; likely to cause drift within weeks.
- **CLEANUP** — Cosmetic or boundary issues; not currently dangerous.

---

## CRITICAL findings

### C1. Dual authority over "what is planned" — `ROADMAP.md` vs `PLAN_24H_EDGE_IMPLEMENTATION.md`
**Docs involved:**
- `ROADMAP.md` (registered in `document_authority.md` as the planning inventory)
- `PLAN_24H_EDGE_IMPLEMENTATION.md` (top-level, unregistered, untracked in git — 294 lines)

**Exact conflict:** Both describe ORB/edge discovery pipeline entry/filtering strategy. `ROADMAP.md` is the registered canonical surface for "planned but not yet built"; `PLAN_24H_EDGE_IMPLEMENTATION.md` is a parallel narrative of the same scope (Phase 5b-onward work). No git history on the PLAN file — it's a working draft that was never committed.

**Which doc should win:** `ROADMAP.md`. It is explicitly registered as the planning inventory and is the only "plan"-shaped doc bound by `document_authority.md`.

**Recommended fix:** Delete `PLAN_24H_EDGE_IMPLEMENTATION.md`. If any of its content is still active, append a new phase entry to `ROADMAP.md`. Same treatment for `PLAN.md` (top-level, untracked, 91 lines) and `PLAN_codex.md` (2026-03-24, superseded by `docs/audit/`).

---

### C2. ~~Amendment numbering non-monotonic — cross-references fragile~~ — **PARTIALLY RESOLVED 2026-04-17**

**Status:** Citation convention established + fragile `RESEARCH_RULES.md` citations migrated to stable criterion anchors. Amendment restructuring + drift check deferred.

**Applied fixes (2026-04-17):**
- `docs/institutional/pre_registered_criteria.md` header gained a **Citation convention** note: external docs MUST cite by the stable `## Criterion N` anchors, never by amendment number. Amendment subsections are revision history and their numbering may change.
- `RESEARCH_RULES.md` — four fragile amendment citations migrated:
  - `RESEARCH_RULES.md:14` — "(v2 with 5 Codex audit amendments)" → points to the *Version history* + *Amendment procedure* sections; adds the cross-reference convention inline.
  - `RESEARCH_RULES.md:16` — "See Amendment 2.7 in pre_registered_criteria.md" → "See `Criterion 8 — 2026 out-of-sample positive` (as revised by Amendment 2.7)".
  - `RESEARCH_RULES.md:38` — "Amendment 2.7 (2026-04-08) for the restoration" → "§ `Criterion 8 — 2026 out-of-sample positive` (as revised by Amendment 2.7 on 2026-04-08)".
  - `RESEARCH_RULES.md:47` — "research-provisional per Amendment 2.4" → "per the *Applying to current 5 deployed lanes* section (lane classification language codified by Amendment 2.4)".

**Deferred (out of this pass):**
- Flattening the amendment subsection structure in `pre_registered_criteria.md` to a simpler top-level list. Higher blast radius — needs a review of every inbound citation across `docs/plans/`, `HANDOFF.md`, `docs/audit/hypotheses/`, `docs/handoffs/`, memory files. Own workstream.
- Drift check in `pipeline/check_drift.py` greppting for `Amendment \d\.\d` citations outside the canonical doc and verifying each target resolves. Code change — explicitly excluded from the docs-only scope.

Both deferred items still needed for full C2 closure. The citation convention note now makes the policy explicit, and the RESEARCH_RULES migration removes the immediate fragility. A future contributor who cites an Amendment instead of a Criterion is now breaking a written rule, not an implicit one.

---

## IMPORTANT findings

### I1. ~~WFE threshold inconsistency: `>0.50` vs `≥0.50`~~ — **RESOLVED 2026-04-17**

**Applied fix:** `RESEARCH_RULES.md:80` changed from *"Walk-forward efficiency (WFE) > 50% = strategy likely real. < 50% = likely overfit."* → *"Walk-forward efficiency (WFE) ≥ 0.50 = strategy meets the locked binding threshold (see `docs/institutional/pre_registered_criteria.md` § *Criterion 6 — Walk-forward efficiency*). < 0.50 = likely overfit."* Operator aligned with Criterion 6; stable anchor pointer added. No change to `pre_registered_criteria.md`.

---

### I2. ~~REGIME-class strategy deployment — `RESEARCH_RULES.md` bans it, `TRADING_RULES.md` shows one deployed~~ — **RESOLVED 2026-04-17**

**Status:** Resolved by live-state query. Docs edit only — no doctrine breach.

**Live-state verification (2026-04-17 via gold-db):**
- All 4 active `MNQ TOKYO_OPEN` `validated_setups` rows are CORE (sample_size 918–1487), not REGIME.
- Portfolio-wide: 59 CORE, 2 REGIME (both GC, retired), 0 INVALID. Zero active REGIME-class strategies in the deployed book.
- `TRADING_RULES.md:74` "REGIME-class (WFE=0.53, p=0.010)" was stale wording from a prior backtest era when N was still below 100.

**Resolution:** Single-line edit to `TRADING_RULES.md:74` applied 2026-04-17 — replaced "TOKYO_OPEN is REGIME-class (WFE=0.53, p=0.010)" with "TOKYO_OPEN is CORE (active N=918-1487, WFE=0.525-0.823, p=0.001-0.011)" plus a note pointing to the 2026-04-17 verification. No change needed to `RESEARCH_RULES.md`. No change needed to the deployed book.

**Structural follow-up (not in scope of I2):** Carrying per-strategy N/WFE/p numbers in `TRADING_RULES.md` is a future staleness trap — every backtest rebuild invalidates them. The I4 fix (delete strategy-status tables from doctrine docs; point to live DB) applies here too; if implemented, the same class of bug disappears entirely.

---

### I3. Significance threshold cited inconsistently across three registered docs — **(a) RESOLVED 2026-04-17; (b) + (c) DEFERRED**

**Original problem (preserved for context):**
- `RESEARCH_RULES.md:65` cited *"p < 0.005 required for discovery claims (Harvey & Liu 2014)"*
- `pre_registered_criteria.md:69` defines *"t ≥ 3.79 (Chordia et al 2018)" OR "t ≥ 3.00 (HLZ)"*
- `TRADING_RULES.md:619` cites a deployed lane with `t=3.34, p=0.0016, p_bh=0.088` as PROMISING (passes t≥3.00, fails t≥3.79)

**(a) Applied 2026-04-17:** `RESEARCH_RULES.md:65` changed from *"- **p < 0.005:** Required for 'discovery' claims (per Harvey & Liu, 2014)."* → *"- **Discovery claims:** see the binding threshold in `docs/institutional/pre_registered_criteria.md` § *Criterion 4 — Chordia t-statistic threshold*."* RESEARCH_RULES no longer states a separate threshold; it points to the locked binding source.

**(b) DEFERRED:** Adding prior-theory citations per deployed lane with `t < 3.79` in `TRADING_RULES.md`. This is doctrine wording, not mechanical cleanup — picking the correct theory citation per lane (Crabel 1990, Fitschen 2013, etc.) is a research judgment call. Blanket citations rejected by user 2026-04-17. Own workstream.

**(c) DEFERRED:** Clarifying what passes the "PROMISING" vs "VALIDATED" bar. Also doctrine wording — requires deciding whether PROMISING is a named tier with a threshold, or only loose shorthand. Own workstream.

---

### I4. ~~Strategy status table is a dual source of truth~~ — **ALREADY SATISFIED (verified 2026-04-17)**

**Verification:** Re-reading `RESEARCH_RULES.md:244-270` on 2026-04-17 confirmed the file does NOT carry a parallel validated-and-deployed strategy table. Both relevant subsections are already single-line pointers:
- `RESEARCH_RULES.md:247` — *"### Validated and Deployed — See `TRADING_RULES.md` → Confirmed Edges table."*
- `RESEARCH_RULES.md:265` — *"### Confirmed NO-GOs (Do Not Revisit) — See `TRADING_RULES.md` → What Doesn't Work table."*

The audit's initial characterisation ("Both contain strategy-status tables") was inaccurate — `RESEARCH_RULES.md` already defers to `TRADING_RULES.md` for strategy status. The remaining content in that section (Cross-Instrument Stress Test Finding, Awaiting 10-Year Outcome Rebuild, Re-Validation Trigger) is **research commentary** about methodology-level findings, not a deployed-strategy list.

No edit applied. I4 closed.

**Follow-up note (not in scope):** The "Awaiting 10-Year Outcome Rebuild" subsection lists 4 items flagged Feb 2024 - Feb 2026. If any have since been validated or killed, that list is stale — but resolving that requires research judgment (query DB, compare to current validated_setups), not mechanical docs cleanup.

---

### I5. ~~Holdout CONFIRMATION gates buried — cross-doc dependency not obvious~~ — **RESOLVED (incidentally) 2026-04-17**

**Status:** Resolved as a side-effect of the C2 citation migration on 2026-04-17.

`RESEARCH_RULES.md:38` (the holdout section) now reads *"...and `docs/institutional/pre_registered_criteria.md` § `Criterion 8 — 2026 out-of-sample positive` (as revised by Amendment 2.7 on 2026-04-08) for the Mode A restoration..."* — exactly the stable-anchor pointer I5 asked for, placed in the holdout section. No further edit required.

---

## CLEANUP findings

### K1. `mechanism_priors.md` disclaims authority but is cited as authority
**Docs involved:** `docs/institutional/mechanism_priors.md`, `docs/institutional/edge-finding-playbook.md`
- `mechanism_priors.md:60` § 3 states: *"This is a prior-beliefs document, not a validated model."*
- `edge-finding-playbook.md:4` cites it as: *"Authority: Complementary to pre_registered_criteria.md."*

**Recommended fix:** Pick one framing. If priors are authority-complementary, remove the disclaimer in `mechanism_priors.md`. If priors are informational only, downgrade the citation in `edge-finding-playbook.md` from "Authority" to "Background reading."

### K2. `finite_data_framework.md` adds locked policy that isn't in `pre_registered_criteria.md` header
**Docs involved:** `docs/institutional/finite_data_framework.md:10, 126`
- Declares deference: *"Where the two disagree, follow the criteria file."*
- Adds a 5-era stability gate: *"ExpR ≥ −0.05 in each era with N ≥ 50."*
- That rule *is* in `pre_registered_criteria.md` Amendment 3.2 (Criterion 9), but `finite_data_framework.md` states it first — framework reads like doctrine.

**Recommended fix:** Reorder `finite_data_framework.md:126` to cite Criterion 9 as the source, not to state the rule. Or move the rule entirely to `pre_registered_criteria.md` and reference it from the framework doc.

### K3. `docs/institutional/HANDOFF.md` marked SUPERSEDED but still in the tree
**Docs involved:** `docs/institutional/HANDOFF.md:1` — first line: *"🟢 SUPERSEDED 2026-04-07."*
- Top-level `HANDOFF.md` is the canonical cross-tool baton per `document_authority.md`.

**Recommended fix:** Move to `docs/archive/` or delete. Leaving "SUPERSEDED" docs in the tree trains readers to ignore the banner on files that *aren't* superseded.

### K4. `@research-source` citation paths inconsistent in `TRADING_RULES.md`
**Docs involved:** `TRADING_RULES.md:148, 189, 256` — citations mix `research/…`, `scripts/…`, `trading_app/…` without a format convention.

**Recommended fix:** Add a one-liner to `TRADING_RULES.md` header defining the citation format (e.g., *"All `@research-source` paths are relative to repo root"*). Drift check #45 already validates these paths; consider adding a format-normalizer if paths drift again.

---

## Summary table

| ID | Severity | Docs | Recommended fix (one-liner) |
|---|---|---|---|
| C1 | CRITICAL | `ROADMAP.md` vs `PLAN_24H_EDGE_IMPLEMENTATION.md` (+2 orphan PLAN files) | Delete orphan PLAN files; ROADMAP is sole registered plan |
| C2 | ~~CRITICAL~~ **PARTIALLY RESOLVED 2026-04-17** | `RESEARCH_RULES.md`, `pre_registered_criteria.md` | Citation convention added + 4 fragile Amendment citations migrated to Criterion anchors. Amendment flattening + drift check deferred. |
| I1 | ~~IMPORTANT~~ **RESOLVED 2026-04-17** | `RESEARCH_RULES.md:80` | Operator changed `>` → `≥`; Criterion 6 pointer added |
| I2 | ~~IMPORTANT~~ **RESOLVED 2026-04-17** | `TRADING_RULES.md:74` | Single-line stale-wording fix applied; no doctrine breach (live query confirmed TOKYO_OPEN is CORE, N=918-1487) |
| I3 | **(a) RESOLVED 2026-04-17; (b)+(c) DEFERRED** | `RESEARCH_RULES.md:65` (done); `TRADING_RULES.md` lane citations (deferred); PROMISING/VALIDATED tier (deferred) | (a) RESEARCH_RULES threshold replaced with Criterion 4 pointer; (b)+(c) doctrine wording, requires research judgment |
| I4 | ~~IMPORTANT~~ **ALREADY SATISFIED (verified 2026-04-17)** | `RESEARCH_RULES.md:247, 265` | Both subsections already use single-line pointers to TRADING_RULES; no duplication to delete |
| I5 | ~~IMPORTANT~~ **RESOLVED (incidentally) 2026-04-17** | `RESEARCH_RULES.md:38` | C2 citation migration already placed the Criterion 8 pointer in the holdout section |
| K1 | CLEANUP | `mechanism_priors.md` vs `edge-finding-playbook.md` | Pick one framing (authority or informational) |
| K2 | CLEANUP | `finite_data_framework.md` | Reorder: cite Criterion 9 as source, not restate |
| K3 | CLEANUP | `docs/institutional/HANDOFF.md` | Move SUPERSEDED file to `docs/archive/` |
| K4 | CLEANUP | `TRADING_RULES.md` citation format | Document `@research-source` path convention |

---

## Appendix — stale/orphan items (out of main scope, flagged for later)

These surfaced during the review. None create authority conflicts today, but they are obvious candidates for a future cleanup pass.

| File | Status | Note |
|---|---|---|
| `HANDOFF_E2_CANONICAL_FIX.md` | STALE | E2 fix merged Apr 8 2026 per MEMORY.md. Move to `docs/archive/` or `docs/runtime/stages/e2-canonical-window-fix.md` |
| `PLAN.md` | ORPHAN | Gitignored + untracked (no history); 91 lines. Delete or commit to `docs/plans/` |
| `PLAN_24H_EDGE_IMPLEMENTATION.md` | ORPHAN | Gitignored + untracked; 294 lines. See C1 above |
| `PLAN_codex.md` | STALE | 2026-03-24; superseded by `docs/audit/` (38+ files) — tracked in git, needs `git rm` |
| `PIPELINE_AUDIT_2026-02-25.md` | STALE | Feb 2026 audit artifact, unreferenced |
| `PIPELINE_AUDIT_2026-02-27.md` | STALE | Feb 2026 audit artifact, unreferenced |
| `E3_REBUILD_PROMPT.md` | ORPHAN | Prompt fixture, untracked, unreferenced |
| `PROMPT_hardening_sprint.md` | ORPHAN | Transient sprint plan, unreferenced |
| `CHANGELOG.md` | STALE | Last touched 2026-02-20, unreferenced |
| `IDENTITY.md` | ORPHAN | Untracked, unreferenced |
| `CRYPTO_EDGE_RESEARCH_CATALOG.md` | ACTIVE-BUT-UNREGISTERED | 2026-04-15; research resource catalog; consider registering in `document_authority.md` or leave as informal catalog |
| `HEARTBEAT.md` | ACTIVE-BUT-UNREGISTERED | Referenced by `session_orchestrator.py`; if operational task-queue is formal, register in `document_authority.md` |
| `DB_PERFORMANCE_PLAN.md` | SEMI-LIVE | Referenced from `docs/plans/`; decide: migrate into `docs/plans/` or keep as top-level plan |
| `docs/plans/` (224 files) | HISTORICAL | Apr 2026 files are active design history; Feb–Mar 2026 files (~100+) are completed-phase records. Consider `docs/archive/completed_phases/` subdir |

---

## Notes on method & limitations

- Three parallel Explore agents produced the raw findings; this doc synthesizes and filters to the agreed scope (duplicates + authority conflicts only). Stale/orphan items were deferred to the appendix.
- Spot-checks only on docs/plans/ (3 oldest files). A deeper pass over the 224-file plan directory would likely find more references to renamed/removed tables/functions — deferred.
- Live DB values for I2 were queried 2026-04-17 via the gold-db MCP; result resolved I2 to a docs-only staleness fix (see I2 section).
- All remaining findings (C1, C2, I1, I3–I5, K1–K4) are actionable by docs-only edits.

**End of audit.** 11 findings total; I2 resolved; 10 actionable. Recommend addressing C1 + C2 first (they corrupt the authority model itself); I1, I3, I4, I5 second; K1–K4 last. Appendix items are their own cleanup sprint when you're ready.
