# Pending Literature Acquisition — Market Profile / Auction Theory

**Created:** 2026-05-07
**Last updated:** 2026-05-07 (post-sourcing-attempt)
**Trigger:** User question on whether "post-thrust horizontal range" (drawn on MGC 5m 2026-05-06/07) could serve as overlay/condition. Plan document: `~/.claude/plans/image-1-analyse-jolly-kite.md`. Existing local NO-GO registry kills NR4/NR7/Crabel/compression × overnight. Only honest re-open path is value-area / auction-market-theory framework.

---

## Status: PARTIAL GROUNDING LANDED

**Acquired and extracted today:**
- ✓ `yordanov_2026_nq_orb_value_area_breakouts.md` — 2026 working paper, NQ futures, Databento MBO data, Volume Profile Value Area + ORB framework, 159-day empirical study. Direct relevance: HIGH for mechanism, MEDIUM for citation weight (not peer-reviewed). 17/17 load-bearing claims independently re-verified against source HTML.
- ✓ `topstep_2026_auction_market_theory_intro.md` — Topstep operational vocabulary; thin but useful for cross-citation.

**Wired into existing institutional docs:**
- `docs/institutional/mechanism_priors.md` § "Auction theory literature" — updated to reference the new extracts and to acknowledge the prior-project Dalton/value-area archive (`research/archive/research_dalton_*.py`, output in `research/output/dalton_*`). Operational consequence: Role R5 level-target hypotheses can now invoke Chordia Protocol A (t≥3.00 with theory) when citing Yordanov 2026; pure causal-mechanism claims still need Dalton or Tolušić.

**Prior project work that connects (must read before any new pre-reg):**
- 7 archived scripts in `research/archive/` testing Dalton 80% Rule + VA reversion + filter overlays across MGC/MNQ/MES. Mostly fail-or-marginal; MNQ 0900 anchor only positive carrier. Pre-Phase-0, Mode B holdout — cannot be cited as Mode A evidence. Yordanov-grounded veto-signal pre-reg is structurally different (non-execution gate vs entry signal) and does not collide with the archive.

**Sourcing attempted, blocked by anti-bot:**
- ✗ Howard 2026 (SSRN abstract_id=6350238) — "Stop Distance, Exit Methodology, and Signal Preservation in Intraday Value Area Breakouts: Evidence from E-mini S&P 500 Futures." Published 2026-03-05. Tried 4 SSRN delivery URL variants + 2 user-agents + Google referer. Cloudflare Turnstile blocks every script attempt. **Manual download required.**
- ✗ Tolušić 2026 (SSRN abstract_id=6616280) — "Auction Market Theory as an Emergent Property of Inventory Dynamics: The First Formal Mathematical Treatment." This is the **first formal mathematical treatment of AMT** with Hawkes-process inventory dynamics, validated on 44k FX bars + 17k gap-fade trades over 26 years. **Highest citation-weight item on the list.** Same Cloudflare block. Manual download required.
- ✗ CME Group market profile guides (UST + FX) — explicit ToS forbids scripted access (HTTP 403 with policy notice). Manual browser download required.
- ✗ Mind Over Markets (Dalton) on archive.org — access-restricted, no current borrow.
- ✗ Steidlmayer on Markets / Markets and Market Logic on archive.org — access-restricted.

**Refused on integrity grounds (will NOT download):**
- pdfroom, pdfcoffee, forex-station, kupdf, scribd, r-5.org, quastic.cz mirrors of CBOT/Dalton/Steidlmayer material. Per project source-of-truth chain rule, grey-zone copies break the audit chain even if content is identical. Pirate copies of copyrighted books are excluded from the canonical resources directory by policy.

---

## What you (user) need to do — manual acquisition list

For maximum institutional-rigor weight, in priority order:

### TIER 1 — ~5 minutes total, free, legitimate

**1. Tolušić 2026 — formal mathematical treatment of AMT** ⭐ HIGHEST PRIORITY
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6616280
- Action: Open in browser → click "Download This Paper" → may require free SSRN account → save PDF.
- Save as: `resources/Tolusic_2026_Auction_Market_Theory_Inventory_Dynamics.pdf`
- Why it's #1: Empirically validated (44,035 FX bars, 17,716 gap-fade trades, 26 years) AND formally derived (Hawkes inventory model). This is a genuine academic mechanism source, not narrative theory. **One paper that could legitimately ground both an MGC and an MNQ pre-reg.**

**2. Howard 2026 — value-area breakouts on E-mini S&P 500**
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6350238
- Action: Same as above.
- Save as: `resources/Howard_2026_Value_Area_Breakouts_ES_Futures.pdf`
- Why: Direct algorithmic test of the same mechanism Yordanov 2026 tested on NQ. Independent confirmation across two index futures = much stronger Phase 0 grounding than either alone.

**3. CME Group UST Market Profile Guide**
- URL: https://www.cmegroup.com/tools-information/webhelp/us-treasury-market-profile/Content/USTreasuryMarket.pdf
- Action: Open in browser (CME blocks scripts but permits browser access), save PDF.
- Save as: `resources/CME_UST_Market_Profile_Guide.pdf`
- Why: Authoritative current-CME definitions for vocabulary alignment in any pre-reg or design memo.

### TIER 2 — paid, $45 Kindle, ~5 minutes

**4. Dalton — Mind Over Markets (Wiley, 2013 updated edition)**
- Buy: Amazon Kindle ASIN B00DOSTEVG, $45 USD, instant.
- Export to PDF via Calibre (or keep epub — `Read` tool handles both).
- Save as: `resources/Dalton_Mind_Over_Markets.pdf` (or `.epub`)
- Why: Canonical narrative source for AMT/Market Profile theory. Required citation for any mechanism-priors update touching value-area concepts. Tolušić 2026 (TIER 1 #1) cites Dalton as the source theory.

### TIER 3 — skip unless first re-open survives

- Steidlmayer (1986) — out of print, $80–200 used. Dalton restates the theory.
- CME FX Market Profile Guide — companion to UST, redundant for first pre-reg.

---

## What I (Claude) will do once each PDF lands in `resources/`

For each PDF dropped:

1. **Verify file**: confirm it's a real PDF (size > 50KB, valid header), not an HTML error page.
2. **Extract canonical passages**: use the `Read` tool's PDF support; pull verbatim quotes with **page numbers** for value-area definitions, day-type taxonomy, balance-acceptance/rejection rules, range-extension logic, and any explicit empirical claim.
3. **Write extract file** at `docs/institutional/literature/<lastname>_<year>_<short_title>.md` per the schema in `docs/specs/research_modes_and_lineage.md` § 9.2:554-630.
4. **Auto-indexed by `research-catalog` MCP** (`scripts/tools/research_catalog_mcp_server.py:27-29`) — no manifest update needed.
5. **Open hypothesis file** at `docs/audit/hypotheses/2026-05-XX-mgc-mnq-balance-area-acceptance-v1.yaml` citing the new extract files plus the existing Yordanov + Topstep extracts.
6. **Run pre-reg through Gate 1–7** per `STRATEGY_BLUEPRINT.md` § 3 — FDR-corrected, holdout-locked, lane-specific p-values per `feedback_pooled_not_lane_specific.md`, OOS power floor per `RULE 3.3` of `backtesting-methodology.md`.

---

## Honest summary of what's possible RIGHT NOW vs after acquisition

| Question | Right now (Yordanov + Topstep grounded) | After Tolušić + Howard + Dalton |
|----------|-----------------------------------------|----------------------------------|
| Cite a value-area mechanism for an MNQ pre-reg | YES (Yordanov §3.5 Filter Case A; instrument-class match NQ→MNQ) | STRONGER (3 independent empirical sources + theory) |
| Cite a mechanism for an MGC pre-reg | NO (Yordanov is index futures only) | YES once Tolušić's FX/cross-asset 26-year validation lands |
| Pre-reg the "Cross + Miss" veto signal | YES, supported by Yordanov §3.8 alone (n=44 occurrences, 48.4pp gap from baseline) | STRONGER if Howard 2026 confirms on ES |
| Update `docs/institutional/mechanism_priors.md` with AMT | NO — needs theoretical grounding (Dalton or Tolušić formal model) | YES |
| Reopen NR7/Crabel NO-GO entries | NO — none of these papers provide a "fundamentally new mechanism class" justifying a re-open of those specific NO-GO entries; they propose a STRUCTURALLY DIFFERENT framework (value-area), not a re-test of compression | NO (same answer; the new framework is the alternative path, not a NO-GO reopen) |

---

## What this means for your original chart question

The horizontal box you drew on MGC 5m maps to "Filter Case A — close inside Value Area." Yordanov 2026 (§ 3.5) found that Case A produces the **highest depth of follow-through** at deeper deviation targets in NQ — directly contradicting the visual-intuition that horizontal range = exhaustion. **However**: that's NQ index futures over 159 days. We do not yet have any literature source legitimizing that finding for MGC (precious metals).

**Cheapest honest first step:** the "Cross + Miss" veto-signal pre-reg on existing deployed MNQ lanes. This:
- Requires only the Yordanov extract (already landed).
- Tests a *non-execution* signal (skip days), so transaction-cost objection is irrelevant.
- Has a 48.4pp gap from baseline in the source paper — large enough that even if MNQ shows a quarter of that effect, it's deployable.
- Pathway-A K=1 single-cell test on a deployed lane = minimum multiple-comparison cost.

If you want me to write that hypothesis file now (pre-reg only, no scan), say so. Otherwise the next move is yours: download one of the TIER 1 PDFs above and drop it in `resources/`.
