# Plan — Next Literature Ingest (post-/clear baton)

**Created:** 2026-05-12 (immediately before user `/clear`)
**Status:** Awaiting fresh session — Harris PR (#265) is open separately
**Owner:** next Claude session

---

## Context (survives /clear)

The Harris 2002 hardcover ingestion landed on `feat/harris-full-text-ingestion` (commits `974465bf`, `fa8d2ee1`), PR **#265** open against main. That work proved the canonical ingestion pattern this repo should reuse: PDF stays gitignored in `resources/`, extract MD has YAML frontmatter + verbatim quotes + printed-page citations, and a companion verifier script programmatically confirms every quote is on its cited page.

The verifier (`scripts/research/verify_harris_quotes.py` on the Harris branch) is currently Harris-specific — it has the `printed_page_offset=12` baked in. The natural next move is to (a) generalize it into a reusable tool, and (b) use it on the next-priority unextracted PDF.

---

## Tasks for next session (priority order)

### Stage 1 — Generalize the verifier (do FIRST)

**Branch:** new from `origin/main` (Harris branch is in PR review; don't stack)

**File:** Rename / refactor `scripts/research/verify_harris_quotes.py` (currently only on Harris branch — wait until #265 merges, OR cherry-pick the script onto the new branch) into `scripts/research/verify_literature_extract.py`.

**Interface:**
```bash
python scripts/research/verify_literature_extract.py \
    --extract docs/institutional/literature/<source>.md \
    --pdf resources/<source>.pdf \
    --printed-page-offset <N>  # default 0; query the extract's YAML frontmatter for it
```

**Implementation notes:**
- Read `printed_page_offset` from the extract MD's YAML frontmatter when present (key: `printed_page_offset`), else fall back to CLI arg, else 0.
- Keep the parser, normalizer, and fuzzy_contains logic from `verify_harris_quotes.py` verbatim — those are battle-tested.
- Optional: add an `--all` mode that walks every `docs/institutional/literature/*.md` with a `verification_script: scripts/research/verify_literature_extract.py` frontmatter field and verifies the whole catalog.

**Wire into `pipeline/check_drift.py`** as an advisory check (not blocking — extracts without YAML frontmatter would noisily fail).

### Stage 2 — Ingest BH 1995 FDR (the most load-bearing missing extract)

**Source:** `resources/benjamini-and-Hochberg-1995-fdr.pdf` (text-based, no OCR needed).

**Why this one first:**
- Referenced inline across `docs/institutional/pre_registered_criteria.md` (Criterion 3: BH FDR q < 0.05)
- Referenced in `.claude/rules/backtesting-methodology.md` (Rule 4: per-family K framing for BH-FDR)
- Referenced in `.claude/rules/research-truth-protocol.md` (Phase 0 grounding §3 — every statistical method must cite a specific extract)
- **No standalone extract exists** — so every BH-FDR claim in the repo currently violates Rule 7's "cite from `docs/institutional/literature/`, not training memory" requirement.

**Target extract:** `docs/institutional/literature/benjamini_hochberg_1995_fdr.md`

**Required content (per Harris template):**
1. YAML frontmatter (source_pdf, year, criticality=HIGH, role=methodology_source, verification_script)
2. **Key claim 1** — Definition of False Discovery Rate (FDR) vs Family-Wise Error Rate (FWER). Verbatim quote + page.
3. **Key claim 2** — The step-up procedure (the BH algorithm itself). Verbatim quote of the procedure statement.
4. **Key claim 3** — Theorem 1 (the procedure controls FDR ≤ q under independence). Verbatim quote of the theorem statement.
5. **Key claim 4** — When BH applies under positive dependence (later authors' extension — Benjamini-Yekutieli 2001 covers this; cite Benjamini-Hochberg 1995 for the independence case and flag that the BY 2001 generalization is what `pipeline/` actually uses if independence is not asserted).
6. **Mechanism implication for canompx3** — point at `research/comprehensive_deployed_lane_scan.py::bh_fdr_multi_framing()` as the canonical implementation; cross-reference `harvey_liu_2015_backtesting.md` (which builds on BH 1995) and `bailey_et_al_2013_pseudo_mathematics.md` (MinBTL, complementary framework).

**Verifier expected output:** `OK: N/N` for whatever N quotes the extract contains.

### Stage 3 — Ingest Man Overfitting 2015 (second-priority miss)

**Source:** `resources/man_overfitting_2015.pdf`

**Why second:** cited in `backtesting-methodology.md` but has no standalone extract. Smaller payload than BH 1995 — likely 4-5 Key claims.

### Stage 4 — Quick audit pass on Aronson / Pardo / BRTS

For each, run `extract-before-dismiss` (per `institutional-rigor.md` corollary):
1. Extract TOC + 3 mid-doc sample pages
2. Decide: relevant-enough-to-extract (write a small extract MD) OR not-load-bearing (write a one-line `docs/institutional/literature/PENDING_<slug>_not_extracted.md` recording the decision and why, so it's not silently ignored forever)

The three are most likely **not load-bearing** for our current ORB-on-futures pipeline:
- Aronson: ANN/technical-indicator territory — we don't do ML
- Pardo: classic optimization-of-trading-systems — we use Bailey/Harvey-Liu/Chordia framework instead
- BRTS: generic trading-system design — overlaps with Carver who's already extracted

But the **extract-before-dismiss** rule says don't reject without reading the TOC + mid-doc.

---

## Token efficiency notes (cross-link to research done earlier this session)

Best-practice findings from the session that produced Harris PR #265:
- Progressive disclosure beats vector RAG for citation-grounded research (Anthropic Agent Skills doctrine).
- Per-chapter MD extracts > whole-book embeddings.
- Verbatim quotes + page numbers + canonical PDF path is the recommended grounding pattern.
- The verifier script is the load-bearing safety guard — it caught 17 distinct citation errors during Harris that would have shipped silently.

Apply the same pattern to BH 1995 / Man Overfitting / others.

---

## What NOT to do this session

- **Don't stack on the Harris branch** — wait for PR #265 to merge, OR start the new branch from `origin/main` and cherry-pick the verifier script onto it. (Cherry-pick is fine since the verifier is independent of the Harris extract content.)
- **Don't generalize the verifier AND ingest a book in the same commit** — split into two commits (refactor + new ingest) for clean review.
- **Don't ingest a PDF without first running text-extraction yield check** — if the PDF is scanned-image-only (like Harris hardcover was), pause and ask the user before kicking off OCR.

---

## Pointer to canonical references in this work

- Harris extract template (this session's deliverable): `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md` (lands when PR #265 merges)
- Harris verifier (template for generalization): `scripts/research/verify_harris_quotes.py` (lands when PR #265 merges)
- Institutional README (the index): `docs/institutional/README.md`
- Best-practices research findings: previous conversation context lost on /clear, but the gist is in the Harris PR description (#265) and the Stage-1 implementation notes above.
