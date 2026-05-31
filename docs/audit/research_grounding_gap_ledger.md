# Research Grounding — Durable Gap Ledger

**Created:** 2026-05-31 · **Owner:** institutional-grounding integrity
**Machine-readable source of truth:** [`research_grounding_source_manifest.yaml`](research_grounding_source_manifest.yaml)
**Enforced by:** `pipeline/check_drift.py :: check_literature_source_integrity` (CI-safe default + strict-local)

## Purpose

Grounding has two halves: **retrieval** (we hold a page-cited extract — the
`research-catalog` MCP + `resources/INDEX.md` give us that) and **verification**
(the cited source is actually present and says what the extract claims). The
project was strong on retrieval and silent on verification: `resources/INDEX.md`
*counted files* without checking that each extract's cited source exists, so an
extract could cite a PDF that was never saved into the repo and nothing flagged
it. This ledger + manifest + drift check close that gap. The ledger is the
human-readable companion to the manifest; both key on the same `source_id` /
`gap_id` so they cannot silently disagree.

## Three locked rules (non-negotiable)

1. **No source ⇒ UNSUPPORTED.** A claim whose source is absent from disk and
   absent as recorded web provenance is `UNSUPPORTED`, never "probably fine".
2. **Page count ≠ completeness.** A source being present is not enough — the
   extract's *cited pages* must exist in the local file. Strict-local verifies
   `fitz` page count == the extract's declared `expected_pages` (catches the
   sampler/wrong-edition swap, e.g. the 3-page AFML stub vs the 393-page book).
3. **Current stats require a canonical query.** Volatile numbers (strategy
   counts, fitness, sessions, costs) are queried live from the canonical layer
   (gold-db MCP / `pipeline.*` registries), never cited from memory or docs.
   (CLAUDE.md § Volatile Data Rule.)

## Severity

- **HIGH** — a load-bearing source for a *live or candidate* decision is
  unverifiable; could invalidate a research conclusion if the source is wrong.
- **MEDIUM** — grounding link missing or a source is unreviewed/derived; bias or
  navigation cost, not an immediate decision risk.
- **LOW** — cosmetic / completeness note; no decision exposure.

---

## OPEN gaps

| gap_id | source / surface | sev | bite scenario | remediation |
|---|---|---|---|---|
| G5 | `yordanov_2026_nq_orb`, `topstep_2026_amt_intro` (WEB_DERIVED) | MEDIUM | A web working paper is revised or removed; the extract's claims can no longer be re-checked against a live page, yet a hypothesis leans on them. | Status `WEB_DERIVED` with retrieval date + verified-claim count recorded in-extract; treat as unreviewed (theory_grant:false). Snapshot HTML into `docs/research-input/` if a decision ever depends on it. |
| G6 | Chan ch5 currencies/futures (extract present, thin) | LOW | A future mean-reversion idea cites "Chan ch5" expecting depth the stub doesn't carry. | Extract exists and is manifest-covered; expand from the present `Algorithmic_Trading_Chan.pdf` when actually needed. Not STUB_LEDGERED (it has real content), so no stub gate. |
| G7 | `docs/STRATEGY_BLUEPRINT.md` inline cites (Fitschen, Carver, Chordia) | MEDIUM | A reader follows "Fitschen p91-101" with no path to the extract, re-derives from memory, and drifts from the page-cited claim. | **CLOSED below** by task F (literature/ cross-links added). |
| G8 | `TRADING_RULES.md` inline cites (Chordia t≥3.79) | LOW | Same as G7 for the NO-GO registry rows. | **CLOSED below** by task F. |
| G9 | `mechanism_priors.md` leans on unreviewed Howard/Tolušić/Yordanov | MEDIUM | A mechanism prior is treated as established when its only support is a 2026 unreviewed preprint corpus; over-weights a fragile claim. | Manifest tier-tags all three as `unreviewed-preprint`/`web`; priors must carry the unreviewed caveat and the t≥3.79 gate. Bias is exposed, not hidden. |

## CLOSED gaps

| gap_id | source | resolution |
|---|---|---|
| G1 | `howard_2026_value_area_breakouts_es` | Located PDF (SSRN 6350238, 23pp) copied into `resources/` under cited filename; fitz confirms 23pp. **Committed status `MISSING_LOCAL`** (PDF untracked → absent in CI; a committed `VERIFIED_LOCAL` would assert an unreproducible verification). strict-local UPGRADES to a runtime verified-PASS when present. Gap CLOSED. |
| G2 | `tolusic_2026_amt_inventory_dynamics` | Located PDF (SSRN 6616280, 15pp) copied; fitz confirms 15pp. Committed `MISSING_LOCAL`; runtime-upgraded by strict-local. Gap CLOSED. |
| G3 | `lopez_de_prado_2018_afml` | FULL 393pp book copied (NOT the 3pp Downloads stub — rule 2 trap avoided); fitz confirms 393pp. Committed `MISSING_LOCAL`; runtime-upgraded by strict-local. CROSSWALK key added; INDEX regenerated. Gap CLOSED. |
| G4 | `harvey_liu_zhu_2015_cross_section` | Derived note quoting the tracked Chordia extract; status `DERIVED_NOTE`, grounding resolves to a tracked file (CI-verifiable). Not a missing PDF. |
| G7 | STRATEGY_BLUEPRINT inline cites | literature/ cross-links added (task F) so each surname resolves to its extract row. |
| G8 | TRADING_RULES inline cites | literature/ cross-link note added (task F). |
| G10 | "pages ≠ completeness" | Encoded as **locked rule 2** + strict-local page-count check. |
| G11 | Pardo excerpt uncited | `rober prado` PDF is CROSSWALK topic-only (`stem=None`); not a literature extract, so not a manifest source row. No false grounding claim made. |
| G12 | CLAUDE.md regex false-positive | The check matches on the **exact declared filename + page count**, never generic title/text overlap — proven by `test_generic_title_overlap_never_counts_as_source_proof`. A decoy PDF with an overlapping title is never accepted as the source. |

---

## Enumerated grounding surfaces (no silent exclusions)

Every grounding surface in the repo, marked **covered** by the integrity check or
**out-of-scope-with-rationale**. Blocker B was: a literature-only check would lie
by ignoring the other surfaces. This is the explicit inventory.

| surface | status | rationale |
|---|---|---|
| `docs/institutional/literature/*.md` (extracts with `**Source:**`) | **COVERED** | Canonical citation source. Every such file must have a manifest row; verified by the check. |
| `chatgpt_bundle/LIT_*.md` | **COVERED** | Parallel bundle copy, same `**Source:**` convention. Same coverage gate (`test_ci_safe_covers_chatgpt_bundle`). |
| `docs/institutional/literature/PENDING_ACQUISITION_*.md` | OUT-OF-SCOPE | Acquisition-tracking docs, no own `Source:` line → not extracts. Skipped (not flagged missing). |
| `docs/audit/hypotheses/*.yaml` `*_authority:` blocks (219 files) | OUT-OF-SCOPE | `*_authority:` keys point at other repo paths (intra-repo provenance); literature is cited only by author surname. No `resources/*.pdf` to verify. Author→extract grounding is captured by manifest rows; config-value provenance is enforced separately by **drift #45** (`@research-source`). Boundary is explicit. |
| Inline author cites in `STRATEGY_BLUEPRINT.md`, `TRADING_RULES.md`, `mechanism_priors.md`, `RESEARCH_RULES.md`, `chatgpt_bundle/00_INDEX.md`, `01_OPERATING_RULES.md` | OUT-OF-SCOPE (links added) | Prose surname cites. PDF-verifying prose is out of scope; instead task F adds literature/ cross-links so a surname resolves to its (covered) extract. |
| `resources/INDEX.md` | NAVIGATION (not a source) | Generated manifest; `_topic_for` already fails safe on phantom stems. Not a citation source. |

## Boundary vs existing checks

- **drift #45** (`@research-source` / `@entry-models` / `@revalidated-for`) governs
  provenance of **config values** in `config.py`. This check governs **literature
  source presence** for extracts. Disjoint; no overlap.
- **`build_resources_index.py`** does *retrieval* navigation (topic → extract).
  This check does *verification* (extract → present, page-matched source). They
  are the two halves named in the Purpose.

## How to use this ledger

1. Before relying on a source, find its row in the manifest (by `source_id`) or
   here (by author/gap). Check `status`.
2. `VERIFIED_LOCAL` → cite from the page-cited extract. `WEB_DERIVED` → cite with
   the unreviewed caveat. `MISSING_LOCAL` → the extract is retrievable but the PDF
   isn't committed; do not claim you verified the PDF. `UNSUPPORTED` → do not cite.
3. If you add a new extract, add its manifest row in the same commit — the drift
   check will fail otherwise (false-completeness guard).
