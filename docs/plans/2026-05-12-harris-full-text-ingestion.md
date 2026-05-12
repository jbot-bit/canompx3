# Plan — Harris 2002 Full-Text Ingestion (continuation)

**Created:** 2026-05-12 (session before `/clear` for fresh context window)
**Status:** AWAITING full-text PDF download by user
**Owner:** next Claude session

---

## What this plan exists for

User is downloading the **full Harris 2002 hardcover** (~640 pp). The current session ingested the **March 2002 draft** (113 pp, outline-only on many sub-sections) as an interim artifact. Context is being cleared so the next session has a fresh window for the full-text pass.

---

## Verified state on disk (survives `/clear`)

| Path | State | Notes |
|---|---|---|
| `resources/Harris_Trading_Exchanges_Market_Microstructure.pdf` | DRAFT (March 2002, 113 pp) | Gitignored. Will be **overwritten** by full text. |
| `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md` | DRAFT-LEVEL extract (22 KB) | All 12 verbatim quotes verified against draft PDF source on 2026-05-12. All 8 page-number citations verified. Contains honest "What's in the draft but NOT yet usable" disclaimer section. |
| `docs/institutional/README.md` | UPDATED | Harris row added to key-findings table + directory listing. Staged but NOT committed (concurrent-terminal hold). |
| `HANDOFF.md` | MODIFIED by other terminal (Codex auto-baton) — NOT my work |
| `.claude/rules/institutional-rigor.md` | UNCHANGED — Rule 7 covered-papers list pending update (failed in prior session, file was injected as context not Read) |
| `MEMORY.md` | UNCHANGED — Harris not yet indexed |

---

## Verification audit performed before clear

The draft extract was audited line-by-line on 2026-05-12:

- **Verbatim quotes (12 total):** all 12 located in source PDF via PyMuPDF text extraction. Probe match on first 80 normalized chars: 12/12 OK.
- **Page-number citations (8 unique pages: 49, 59, 75, 76, 78, 79, 80, 107):** all 8 verified by re-extracting the cited page and confirming the quoted text appears on that exact page (no off-by-one drift). PyMuPDF is 0-indexed internally; extract cites PDF page numbers (1-indexed) consistently.
- **Interpretive claims (mechanism implications):** these are extrapolations from the verbatim quotes to our project lanes/cost-model. They are clearly labeled as "Mechanism implication for canompx3" and not as Harris's words. Honest framing.
- **Partial-source disclaimer section:** the extract explicitly lists chapters/sections that are outline-only in our draft and forbids citing Harris for prose that does not exist locally. This is the load-bearing safety guard.

**Conclusion:** the draft-level extract is **safe to use as an interim citation source**. Nothing fabricated. Disclaimers prevent over-citation. The full-text pass will *expand* what can be cited, not *replace* what's currently there.

---

## Tasks for next session

### Step 0 — orient

```bash
ls /c/Users/joshd/Downloads/*Harris* /c/Users/joshd/Downloads/*Trading*Exchange* 2>/dev/null
git -C /c/Users/joshd/canompx3 status --short
```

If full Harris PDF is not in Downloads, ask user where it landed.

### Step 1 — replace PDF, verify text extraction

```bash
# After confirming full PDF location:
cp "<full_pdf_path>" /c/Users/joshd/canompx3/resources/Harris_Trading_Exchanges_Market_Microstructure.pdf

# Verify text yield is high (text-based PDF, not scanned):
python -c "
import fitz
d = fitz.open(r'C:\Users\joshd\canompx3\resources\Harris_Trading_Exchanges_Market_Microstructure.pdf')
print(f'pages={len(d)}')
total_chars = sum(len(d[p].get_text()) for p in range(min(50, len(d))))
print(f'first-50-pages text chars: {total_chars} (expect >50000 for text-based)')
"
```

If text yield is low (< ~500 chars/page), the full version may be scanned. Run `ocrmypdf` (already installed at `C:\Users\joshd\AppData\Local\Programs\Python\Python311\Scripts\ocrmypdf`):

```bash
ocrmypdf --skip-text --output-type pdf "<input>" "<output>"
```

### Step 2 — extract full TOC, identify newly-substantive chapters

```bash
python -c "
import fitz
d = fitz.open(r'C:\Users\joshd\canompx3\resources\Harris_Trading_Exchanges_Market_Microstructure.pdf')
for lvl, title, pg in d.get_toc():
    if pg > 0:  # filter outline-only entries
        print(f'{\"  \"*(lvl-1)}{title}  (p.{pg})')
"
```

Compare against the draft's outline-only sections (listed in the existing extract's "What's in the draft but NOT yet usable" section):

- Ch 4 substantive: Stop Orders, Market-If-Touched Orders, Tick-Sensitive Orders
- Ch 5 substantive: Execution Systems, Market Information Systems
- Ch 10 substantive: Informed Trading Strategies, Styles of Informed Trading
- Ch 14 substantive: Equilibrium Spreads, Spread Components inner detail, Cross-sectional Spread Predictions
- Ch 21 substantive: Implicit Transaction Cost Estimation, Missed Trade Opportunity Costs
- Ch 22 substantive: Performance Evaluation Methods, Sample Selection Bias

These are the priority targets for new extraction.

### Step 3 — re-extract priority chapters with verbatim prose

Highest-value chapters for our deployed-lane work, in priority order:

1. **Ch 4 Stop Orders body** — direct mechanism source for E2 entry model. Currently outline-only.
2. **Ch 14 Spread Components body** — decomposes the slippage component of our `cost_specs` total_friction. Currently outline-only.
3. **Ch 21 Implicit Transaction Cost Estimation** — methodology source for empirical cost calibration. Currently outline-only.
4. **Ch 22 Sample Selection Bias** — methodology source complementing Bailey-LdP and Harvey-Liu. Currently outline-only.

For each chapter, extract verbatim passages with page citations and add new "Key claim N" sections to the existing extract file. Do NOT delete the existing claims 1-5 — they are verified and load-bearing.

### Step 4 — update the existing extract file

`docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md`:

1. Update **Source-state note** at top — change from "publicly-circulated March 2002 Draft Copy (113 pp)" to "Full Oxford 2002 hardcover (~640 pp)".
2. **Add new Key claims** for the previously-outline-only sections (Stop Orders body, Spread Components body, Transaction Cost Estimation, Sample Selection Bias).
3. **Remove** the "What's in the draft but NOT yet usable as canonical citation" section once those chapters have been extracted.
4. **Update** the "How to cite this extract" footer — drop the partial-source citation format since full prose is now available.

### Step 5 — finish pending index updates

Three pending updates (deferred from 2026-05-12 session due to concurrent-terminal commit hold):

1. `.claude/rules/institutional-rigor.md` Rule 7 covered-papers line — add Harris to the explicit list. Currently reads "... Carver 2015 Ch 9-10 (added 2026-04-15: ...)". The next session should `Read` the file first (this is why the prior session's Edit failed), then append the Harris entry.

2. `MEMORY.md` index — add one line under "### Statistical rigor" or a new "### Microstructure" subsection:
   ```
   - **Harris 2002 microstructure mechanism source** — stop cascade explains E2 breakout; adverse selection dominates slippage; volatility decomposes fundamental vs transitory; order anticipators are apex predators. → `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md`
   ```

3. `docs/institutional/mechanism_priors.md` — write Harris-cited per-lane mechanism priors for the 3 currently-deployed MNQ lanes (NYSE_OPEN E2 RR1.0, COMEX_SETTLE E2 RR1.5, US_DATA_1000 E2 RR1.5). The mechanism-map table at the bottom of the existing Harris extract is the seed for this — expand into a full priors entry.

### Step 6 — commit

When the concurrent-terminal hold has released (check `git status` for stale staged changes), commit in one cohesive change:

```bash
git add docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md \
        docs/institutional/README.md \
        docs/institutional/mechanism_priors.md \
        .claude/rules/institutional-rigor.md \
        MEMORY.md  # if user-memory updated — check first; user memory may live at ~/.claude/projects/...

git commit -m "docs(institutional): ingest Harris 2002 Trading and Exchanges as mechanism source"
```

DO NOT commit `HANDOFF.md` modifications unless those are also your work — the 2026-05-12 session noted that HANDOFF.md was modified by a concurrent terminal (Codex auto-baton), not by the Harris work.

---

## Falsification probes — confirm the extract is still load-bearing after expansion

After full-text expansion, run these sanity checks:

1. **Citation discoverability:** `mcp__research-catalog__list_literature_sources` should include the Harris entry. The MCP server discovers by directory scan on `docs/institutional/literature/`, so this is automatic.

2. **Citation excerpt:** `mcp__research-catalog__get_literature_excerpt` should return the Harris extract content when queried by name.

3. **Cross-reference integrity:** every citation of Harris elsewhere in the repo (e.g., `mechanism_priors.md`, future hypothesis YAMLs) should reference the canonical extract path. Grep for `harris_2002` to verify no stale paths.

---

## What the prior session deliberately did NOT do

- Did NOT commit the work because a concurrent terminal was committing on the same worktree (per `feedback_shared_worktree_concurrent_commits.md`).
- Did NOT fabricate prose from outline-only sections — the partial-source disclaimer is the load-bearing safety guard.
- Did NOT update `MEMORY.md` or `.claude/rules/institutional-rigor.md` — pending for the full-text pass.
- Did NOT touch `HANDOFF.md` — that modification was the other terminal's work.

---

## References

- Existing draft extract: `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md`
- Source PDF (draft): `resources/Harris_Trading_Exchanges_Market_Microstructure.pdf` (gitignored)
- Institutional README: `docs/institutional/README.md`
- Phase 0 grounding precedent: `docs/institutional/HANDOFF.md` (SUPERSEDED 2026-04-07) — shows the original 6-PDF ingestion pattern this work extends.
- Existing extract format template: `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`
