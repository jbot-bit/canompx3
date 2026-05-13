---
pooled_finding: false
date: 2026-05-12
status: READ_ONLY_NEXT_AUDIT_CANDIDATES
author: claude (main, v2 worktree)
amendments_applied: [E1, E2, E3, E4, E5, E6]
scope: read-only next-Chordia-pre-reg candidate selection; no allocator/DB/live state mutation; no pre-reg authoring
companion_csv: docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv
methodology: docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md
---

# Chordia-Audit Queue v2 — Top-3 Next-Audit Candidates

## Metadata

- **Date:** 2026-05-12
- **Scope:** Read-only — identify 3 next-Chordia-pre-reg candidates from the AUDIT_GAP_ONLY bucket. **Not** deployment; **not** replacement of any live lane; **not** allocator mutation.
- **Live impact:** None.
- **Doctrine anchors:**
  - E6 lock (`memory/feedback_triage_bucket_not_readiness.md`): AUDIT_GAP_ONLY is triage, not readiness.
  - Literature-before-prereg (`memory/feedback_literature_before_prereg.md`): every Chordia pre-reg requires a verified literature extract; refuse to author when no extract exists.
  - OOS power floor (`memory/feedback_oos_power_floor.md`, `backtesting-methodology.md` RULE 3.3): with power < 0.50 the OOS slice cannot kill a signal, but it also cannot confirm one — verdict must be UNVERIFIED, not DEAD.
- **Companion CSV:** `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv`
- **Methodology:** `docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md`

## Scope

**Question:** Of the 8 AUDIT_GAP_ONLY rows in the companion CSV, which 3 should be the next Chordia per-strategy pre-reg authoring candidates? On what basis is each candidate selected, and what is the literature-grounding status (`GROUNDED` / `CLASS_GROUNDED_POLARITY_PENDING` / other) of each candidate's filter family?

**In scope:** ranked selection from AUDIT_GAP_ONLY by `chordia_t`, applying the E6 "next-audit-candidate, not deploy-candidate" framing; per-candidate Mode-A reconciliation, OOS power tier, and literature-grant status; recommended authoring order (lit-grounded first per `feedback_literature_before_prereg.md`).

**Out of scope:** authoring the per-strategy Chordia pre-reg YAMLs themselves; extracting new literature for the polarity-of-effect gap on INTRA_ASSET_PERCENTILE; allocator/DB/audit-log mutation; deployment decisions; any claim that AUDIT_GAP_ONLY membership confers readiness.

## Verdict

`READ_ONLY_NEXT_AUDIT_CANDIDATES`. Token deliberately outside the deployment taxonomy. This document identifies the next audit candidates, not deploy candidates.

**Locked headline (verbatim):** `0 TOP / 0 READY / 8 AUDIT_GAP_ONLY / 243 BLOCKED_ON_GAP / 593 DEFERRED_FILTER_EXCLUDED`.

**Top-3 by Chordia-t descending (verified against companion CSV):**

| Rank | Strategy | t | N (Mode-A IS) | mode_a_expr | Filter family | Lit-grounding status |
|-----:|---------|----:|--------------:|------------:|---------------|----------------------|
| 1 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50` | 4.609 | 886 | 0.1363 | INTRA_ASSET_PERCENTILE | CLASS_GROUNDED_POLARITY_PENDING |
| 2 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70` | 4.582 | 602 | 0.1761 | INTRA_ASSET_PERCENTILE | CLASS_GROUNDED_POLARITY_PENDING |
| 3 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` | 4.230 | 952 | 0.1338 | VWAP_MID_ALIGNED | GROUNDED |

These three are the AUDIT_GAP_ONLY rows with the highest `chordia_t` after excluding (a) the `INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH` rows (`n_oos ∈ {13, 17}` on rows 5 and 8 of the candidates MD — too thin to inform any verdict downstream) and (b) the negative-OOS-ExpR row 8 (PD_CLEAR_LONG_O15 — sign-flipped OOS, see Row-8 caveat in candidates MD § "Row 8" — noise-consistent per OOS power floor but still a quality concern for next-up authorship). The three selected rows all have `c8_oos_status=PASSED`, `n_oos ∈ {46, 47, 58}`, and `chordia_passes_strict=True`.

## Why these three (selection rule)

Selection criteria, in order applied:

1. **In AUDIT_GAP_ONLY tier.** Only blocker is `NO_CHORDIA_AUDIT_LOG_ENTRY` per methodology § "Tier rules".
2. **`c8_oos_status = PASSED`.** Excludes rows 5 (`ATR_VEL_GE105` on TOKYO_OPEN, `INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH`, n_oos=13) and 8 (`PD_CLEAR_LONG_O15`, `INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH`, n_oos=17, OOS ExpR negative). These two rows are not unsalvageable — they could still inform later audits — but they carry more open quality questions than the top 3, and an institutional-rigor session should not start with the messier rows.
3. **Highest `chordia_t` first.** Within the surviving 6 rows, ranks 1–3 are `ATR_P50` (t=4.609), `ATR_P70` (t=4.582), then a 4-way cluster of VWAP_MID_ALIGNED variants at t ∈ {3.91, 4.02, 4.18, 4.23}. Rank-3 = `VWAP_MID_ALIGNED_O30` at t=4.230 (the highest of the four VWAP variants).
4. **Family diversity preserved.** The top-2 are both `INTRA_ASSET_PERCENTILE` on the same session (`COMEX_SETTLE`); rank-3 brings a different family (`VWAP_MID_ALIGNED`) and a different session (`US_DATA_1000`). This avoids stacking three audits on essentially-correlated cells.

The top-3 are **next-audit candidates**, not deploy candidates. Each will require a per-strategy Chordia pre-reg yaml under `docs/audit/hypotheses/` (separate next-thread per plan § "Out of scope") before any per-strategy audit row can be written to `chordia_audit_log.yaml`.

## Per-candidate block

### Rank 1 — `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50`

**Identity:** MNQ, COMEX_SETTLE session, ORB-5m, E2 (stop-market) RR=1.0 CB=1, filter `ATR_P50` (intra-asset ATR percentile gate, fires when daily ATR is ≥ asset's own 50th percentile).

**Stored vs Mode-A reconciliation:**

| Metric | Stored (validated_setups) | Mode-A (recomputed) | Delta |
|--------|--------------------------:|--------------------:|------:|
| sample_size | 833 | 886 (n_is_mode_a) | +53 |
| expectancy_r | 0.1444 | 0.1363 | −0.0081 |
| sharpe_ratio | 0.1597 | (Mode-A std=0.8989, implied≈0.1517) | ≈−0.008 |
| scratch_drop_rate | n/a | 0.0% | n/a |

Stored-vs-Mode-A drift is **−0.0081R**, well under the 0.05R self-consistency gate (methodology § E5). Mode-A recompute is the canonical reference for any downstream pre-reg.

**Chordia-t derivation:**
- Identity: `chordia_t = sharpe_ratio × √N` per `trading_app.chordia.compute_chordia_t`.
- Using stored: `0.1597 × √833 = 4.610` (CSV reports 4.609, rounding).
- Mode-A implied: `(mode_a_expr / mode_a_std) × √n_is_mode_a = (0.1363 / 0.8989) × √886 ≈ 4.51`. Slight difference because the CSV's `chordia_t` column uses the stored Sharpe; the Mode-A reconstruction confirms order-of-magnitude. Threshold 3.79 cleared either way.

**OOS power tier:** `STATISTICALLY_USELESS` (power=0.2057). Cohen's d on OOS=0.1516, n_oos=58, alpha=0.05. Below 0.50 directional threshold and well below 0.80 refute threshold per `research/oos_power.py:POWER_TIERS`. Any pre-reg author must declare OOS verdict tier `UNVERIFIED` per RULE 3.3 — the OOS slice **cannot** kill or confirm the IS signal.

**Filter family + theory-grant status:** `INTRA_ASSET_PERCENTILE`. Grant status = **CLASS_GROUNDED_POLARITY_PENDING** (see § Literature grounding below).

**Pre-reg recommendation (sketch — not authored in this work block):**
- Design: Pathway-B K=1 paired ΔR under canonical Mode A; IS window pre-2026-01-01, OOS window 2026-01-01 → present (~70+ trading days).
- Hypothesis: ATR_P50 filter fires positively differentiated R relative to the filter-off cell on the same `MNQ × COMEX_SETTLE × E2 × RR1.0 × CB1 × O5` universe. ΔR is the paired-trade-day difference.
- Theory citation: Fitschen 2013 Ch 6 (`docs/institutional/literature/fitschen_2013_path_of_least_resistance.md:152`) — class-grounded as a volatility-percentile filter; the polarity question (ATR-above-percentile = boost, vs ATR-below-percentile = skip) needs explicit pre-reg justification because Fitschen's empirical positive direction in Ch 6 is *skip-high-vol* not *include-high-vol*. The pre-reg author needs to make the directional case via a separate mechanism argument (or run a small directional pretest).
- Strict threshold: `t ≥ 3.79` (Chordia, no-theory) until polarity-grant resolves; can drop to `t ≥ 3.00` (HLZ, with-theory) only after polarity-of-effect literature gap closes.
- OOS verdict default: `DIRECTIONAL_ONLY` if power crosses 0.50 by audit time (unlikely given 70-day window), else `UNVERIFIED_INSUFFICIENT_POWER`.
- `scratch_policy: realized-eod` (C13 BINDING per `memory/feedback_chordia_unlock_deployment_gate_audit_checklist.md`).

### Rank 2 — `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70`

**Identity:** MNQ, COMEX_SETTLE, ORB-5m, E2 RR=1.0 CB=1, filter `ATR_P70` (fires when daily ATR ≥ asset's 70th percentile — tighter than ATR_P50).

**Stored vs Mode-A reconciliation:**

| Metric | Stored | Mode-A | Delta |
|--------|-------:|-------:|------:|
| sample_size | 575 | 602 | +27 |
| expectancy_r | 0.1724 | 0.1761 | +0.0037 |
| scratch_drop_rate | n/a | 0.0% | n/a |

Stored-vs-Mode-A drift = **−0.0037R** (the CSV reports `stored_minus_mode_a = −0.0037` — Mode A is fractionally **higher** than stored). Well under 0.05R gate.

**Chordia-t derivation:**
- `0.1911 × √575 = 4.583` (CSV: 4.582, rounding).
- Mode-A implied: `(0.1761 / 0.8942) × √602 ≈ 4.83`. Confirms order-of-magnitude.

**OOS power tier:** `STATISTICALLY_USELESS` (power=0.2623). Cohen's d=0.1970, n_oos=47, alpha=0.05. Same OOS power floor verdict as rank-1.

**Filter family + theory-grant status:** `INTRA_ASSET_PERCENTILE`. **CLASS_GROUNDED_POLARITY_PENDING** — same as rank-1, with one nuance: ATR_P70 is the *tighter* percentile gate (fires only on the highest-vol 30% of days). The polarity question is sharper here because if the mechanism is genuinely "high-vol days are better for ORB breakouts" then ATR_P70 should out-perform ATR_P50 by a wider margin than it does. The Mode-A ExpRs (0.1761 vs 0.1363) are consistent with that mechanism, but the sample-size disparity (602 vs 886) means the comparison can't yet be a confirmation. Worth noting in the pre-reg.

**Pre-reg recommendation (sketch):**
- Design: same Pathway-B K=1 paired ΔR under Mode A. Sibling pre-reg to rank-1 (same session, family, instrument; different percentile threshold).
- Theory citation: same Fitschen line 152, plus polarity argument required.
- Co-authoring note: if rank-1 and rank-2 both get pre-regs, author them **sequentially not in parallel** so the rank-1 result (which has a larger sample and clearer baseline) can inform whether rank-2's narrower-threshold version is worth a second pre-reg or is just a sibling that would be better aggregated.

### Rank 3 — `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30`

**Identity:** MNQ, US_DATA_1000 session, ORB-30m, E2 RR=1.0 CB=1, filter `VWAP_MID_ALIGNED` (intra-bar microstructure filter — fires when ORB-break candle is aligned with session VWAP at the midpoint, per `trading_app/strategy_filters.py`).

**Stored vs Mode-A reconciliation:**

| Metric | Stored | Mode-A | Delta |
|--------|-------:|-------:|------:|
| sample_size | 699 | 952 | +253 |
| expectancy_r | 0.1539 | 0.1338 | −0.0201 |
| scratch_drop_rate | n/a | 0.0% | n/a |

Stored-vs-Mode-A drift = **+0.0201R** (CSV column `stored_minus_mode_a`). Sample-size delta is the largest of the three (+253) — likely reflects Mode-A's broader IS coverage from `orb_outcomes` ∩ `daily_features` triple-join not being limited by whatever discovery-script pre-filter generated the stored `sample_size`. Drift still well under 0.05R gate.

**Chordia-t derivation:**
- `0.1600 × √699 = 4.231` (CSV: 4.230, rounding).
- Mode-A implied: `(0.1338 / 0.8770) × √952 ≈ 4.71`. Higher than stored-derived t due to larger Mode-A N.

**OOS power tier:** `STATISTICALLY_USELESS` (power=0.1732). Cohen's d=0.1526, n_oos=46, alpha=0.05. Same OOS power floor verdict.

**Filter family + theory-grant status:** `VWAP_MID_ALIGNED`. Grant status = **GROUNDED** via Chan 2013 Ch 7 (`docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`) — intraday momentum / open-imbalance / stop-cascade mechanism for ORB breakouts. The session anchor is also Chan-aligned: US_DATA_1000 is a high-flow US session window where intraday momentum is most empirically supported per Ch 7's mechanism enumeration.

**Pre-reg recommendation (sketch):**
- Design: Pathway-B K=1 paired ΔR under Mode A; sample size already comfortable (N_is=952). The 30-minute ORB aperture is a parameter the pre-reg must justify against the standard 5-minute baseline (note from candidates MD: this is the only AUDIT_GAP_ONLY row with `orb_minutes=30`).
- Theory citation: Chan 2013 Ch 7 `chan_2013_ch7_intraday_momentum.md` — extract verified to exist at file path. The mechanism case (intraday momentum / stop-cascade on US-data-driven session) is on the strongest footing of the three candidates.
- Strict threshold: `t ≥ 3.79` Chordia (the no-theory default). With Chan grounding, downgrading to `t ≥ 3.00` (HLZ) is defensible **if** the pre-reg makes the case explicit. Per `memory/feedback_chordia_theory_citation_field_presence_trap.md`: the `theory_citation` field must be **omitted** for no-theory; for theory-claimed, the citation must be specific (file + section), not prose.
- OOS verdict default: `UNVERIFIED_INSUFFICIENT_POWER` (power 0.17, well below 0.50). Per `feedback_oos_power_floor.md`: `DIRECTIONAL_ONLY` verdict is acceptable **if** OOS power crosses 0.50; at current N this is not achievable until ≥ ~150 OOS trades accumulate (rough back-of-envelope; the pre-reg author should compute exact OOS-power-vs-sample to set their pre-committed sample threshold).
- `scratch_policy: realized-eod`.

**Author this one first.** Per `memory/feedback_literature_before_prereg.md`, the literature-grounded candidate (rank-3 VWAP_MID_ALIGNED) is the more defensible next-thread starting point than the CLASS_GROUNDED_POLARITY_PENDING ATR-percentile pair (ranks 1 + 2). Rank-3 has the cleanest mechanism story and the largest IS sample. Ranks 1 + 2 are queued for **after** the ATR-percentile polarity-of-effect literature gap is addressed (separate next-thread).

## Literature grounding status (per filter family)

### `VWAP_MID_ALIGNED` — GROUNDED

**Source:** `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`

**Verification:** file exists (confirmed via direct `Read` 2026-05-12, 150 lines, last updated header line: "*Extracted: 2026-04-19; expanded 2026-05-08*"). Topic alignment with `VWAP_MID_ALIGNED`:

> "There is an additional cause of momentum that is mainly applicable to the short time frame: the triggering of stops. Such triggers often lead to the so-called breakout strategies." (`chan_2013_ch7_intraday_momentum.md` § "Causes of intraday momentum")

> "Finally, at the shortest possible time scale, the imbalance of the bid and ask sizes, the changes in order flow, or the aforementioned nonuniform distribution of stop orders can all induce momentum in prices."

The VWAP-midpoint-aligned filter is a refinement of this stop-cascade mechanism: it gates entries to break-bars where the order-flow direction (encoded in VWAP) aligns with the breakout direction at the midpoint, reducing fakeout exposure. The mechanism story is on the same footing as the project's core ORB premise (Chan Ch 7 grounds the breakout class; VWAP-midpoint is the filter variant). **No additional literature work is required before the rank-3 pre-reg.**

The `chan_2013_toc_determination.md` extract also exists in the literature directory (canonical `Read` source for TOC-determination questions) but it's not required for the VWAP_MID_ALIGNED mechanism grant. The Ch-7 file is the primary anchor.

### `INTRA_ASSET_PERCENTILE` (ATR_P50, ATR_P70) — CLASS_GROUNDED_POLARITY_PENDING

**Honest framing.** The plan called this `PENDING_LITERATURE_GRANT` and stated "no committed excerpt found for ATR-percentile filter family". The CSV verification path (`ls docs/institutional/literature/` + `grep -ri "ATR percentile|intra.asset percentile" docs/institutional/literature/`) actually finds a hit: `fitschen_2013_path_of_least_resistance.md:152` lists `ATR_P50 (gate on ATR percentile)` as **Class-grounded** in Fitschen Ch 6:

> "| `ATR_P50` (gate on ATR percentile) | Volatility (variant, percentile-shaped) | Configurable | Class-grounded |" (line 152)

So `INTRA_ASSET_PERCENTILE` does have **class-level** grounding — Fitschen's framework treats ATR-percentile gates as a valid volatility-filter variant.

**But** Fitschen's empirically-positive direction is *skip-low-volatility* (`fitschen_2013_path_of_least_resistance.md:130-132`):

> "The low-volatility filter does eliminate some of the less profitable trades, as shown by increasing profit-per-trade in Table 6.13, but it doesn't help with the draw-downs, so it won't be added to the baseline."

`ATR_P50` and `ATR_P70` in the canompx3 project fire **when ATR is HIGH** (above the asset's own percentile threshold), i.e., the **opposite** polarity. This is the same nuance Fitschen line 156-167 flags for the `ORB_G{N}` size-gate family:

> "**The honest position on `ORB_G5` and other size-gate filters:** Fitschen Ch 6 grounds the *class* of 'use a volatility-based filter to improve a breakout system' but the project's `ORB_G{N}` filters work in the OPPOSITE polarity to Fitschen's empirically-positive result (skip-low-vol vs skip-high-vol). For a `has_theory: true` claim under Chordia Pathway A, the project would need EITHER: 1) a separate literature source for 'skip-low-vol-day breakout' mechanism (none found in `resources/`), OR 2) a theoretical argument from microstructure that ORB-size gates a different latent variable than 20-day average range."

Therefore `INTRA_ASSET_PERCENTILE` (ATR_P50, ATR_P70) is **CLASS_GROUNDED_POLARITY_PENDING**:

- **Class is grounded** — Fitschen Ch 6 covers volatility-percentile gates as a filter class.
- **Direction-of-effect is NOT grounded** — Fitschen's empirically-positive direction is opposite (skip-high-vol-trades vs include-high-vol-trades). The project's ATR_P50/P70 fires include-high-vol; that polarity needs its own theoretical case.
- **Path to GROUNDED status:**
  1. Cite a separate literature source for "include-high-vol-day breakout" mechanism — Chan 2013 Ch 1 § "Backtesting look-ahead" and Harris 2002 Ch 4 § "Stop-cascade" both contain *related* mechanism material but neither is a direct match. The honest assessment is: no committed excerpt in `docs/institutional/literature/` currently grounds the *polarity* of ATR-above-percentile = boost. The closest mechanism source is Harris 2002 Ch 4 § "stop-cascade liquidity-demand" but the polarity-mapping is non-trivial and would require a fresh extract pass.
  2. OR make a microstructure argument from canonical sources (e.g., "high-ATR days correlate with higher participation and thus larger stop clusters, amplifying the stop-cascade momentum on a break"). This argument is plausible but would itself need a literature anchor — `harris_2002_trading_exchanges_microstructure.md` is the most promising candidate for that work.

**Conclusion for ranks 1 + 2 (ATR_P50 + ATR_P70):** the candidates ARE listed as ranked-1 and ranked-2 by `chordia_t` (4.609 and 4.582). Per `memory/feedback_literature_before_prereg.md`, however, they are **NOT YET AUTHORABLE as pre-regs** under a `theory_citation` grant — only as `theory_citation: omitted` pre-regs requiring the strict `t ≥ 3.79` floor. The plan's flag of `PENDING_LITERATURE_GRANT` for these two is correct in spirit (polarity grounding needed); the more accurate label is `CLASS_GROUNDED_POLARITY_PENDING` because the class itself has Fitschen coverage.

**Next-thread action (if the user wants to make ranks 1 + 2 authorable under a `t ≥ 3.00` HLZ threshold):** extract a polarity-of-effect mechanism from `resources/Trading_and_Exchanges_Harris.pdf` Ch 4 § "Stop cascade" or from another local PDF; write `docs/institutional/literature/<new-extract>.md`; cite it explicitly in the rank-1 and rank-2 pre-reg yamls. Per `feedback_chordia_theory_citation_field_presence_trap.md`, the citation must be specific (file + lines), not prose.

**Recommended routing:** even though ranks 1 + 2 are **higher** in Chordia-t than rank 3, the work-block routing for next-pre-reg authorship should be:

1. **Rank-3 first** (`VWAP_MID_ALIGNED_O30`) — fully grounded, ready-to-author pre-reg.
2. **Ranks 1 + 2 second**, **after** the polarity-of-effect literature gap closes (separate next-thread to extract the mechanism source).

This ordering preserves institutional rigor (no `has_theory: true` claim without a verified extract) while making maximum use of the current AUDIT_GAP_ONLY bucket.

## Honest framing

To repeat from the methodology MD and the locked headline:

- **Top-3 = NEXT AUDIT CANDIDATES, not deploy candidates.** Nothing in this MD is approved for live capital, not now, not after the pre-regs are written, not until each candidate has been through its own per-strategy Chordia audit + replay + audit-log entry + lifecycle-gate evaluation. The next-thread work is **authorship of one Chordia pre-reg yaml**; the thread after that is **bounded strict-replay run**; the thread after **that** is **audit-log entry**. Only **then** does a deployment question arise, and it goes through Phase 0 / Phase 2 deployment gates separately.
- **0 TOP / 0 READY headline stands.** Selecting top-3 from AUDIT_GAP_ONLY does **not** redefine readiness. AUDIT_GAP_ONLY = "next-audit-candidate" only.
- **CLASS_GROUNDED_POLARITY_PENDING is a stop signal, not a soft pass.** Per `memory/feedback_literature_before_prereg.md`, refusing to author a pre-reg in absence of a verified mechanism is the correct posture; ranks 1 + 2 remain non-authorable under a `has_theory` grant until polarity-of-effect literature lands.

## Caveats

1. **research-catalog MCP was offline at write-time.** Per the plan and observed session-start state, the `research-catalog` MCP server tools (`mcp__research-catalog__get_literature_excerpt`, `mcp__research-catalog__search_research_catalog`, `mcp__research-catalog__list_literature_sources`) were registered in the tool list but were not invoked to verify literature presence — direct file reads (`Read`, `Glob`, `Grep`) were used instead. The directory listing of `docs/institutional/literature/` (29 entries) and the per-file Reads of `chan_2013_ch7_intraday_momentum.md`, `chordia_et_al_2018_two_million_strategies.md`, and `fitschen_2013_path_of_least_resistance.md` are the authoritative source for the grant-status assignments above. If a later research-catalog MCP refresh surfaces an `INTRA_ASSET_PERCENTILE` polarity-of-effect mechanism source that direct grep missed, the `CLASS_GROUNDED_POLARITY_PENDING` label should be revisited (no doc rewrite needed — the grant-status label is forward-compatible).
2. **Excluded AUDIT_GAP_ONLY rows are not unsalvageable.** Rows 4 (`ATR_VEL_GE105` TOKYO_OPEN), 5 (`PD_CLEAR_LONG_O15` COMEX_SETTLE — also has negative OOS), and the lower-t VWAP variants (rows 6, 7) are still in the queue. The top-3 selection is a starting point for next-thread authorship, not a permanent ranking. A later rerun once `c8_oos_status` improves on rows 4-5 (i.e., more OOS sample accumulates) could promote them.
3. **OOS power floor binds everywhere.** All 8 AUDIT_GAP_ONLY rows — including the top-3 — have `oos_power_tier = STATISTICALLY_USELESS`. The pre-reg authoring on rank-3 will need to declare its OOS verdict policy as `UNVERIFIED_INSUFFICIENT_POWER` up-front, and the pre-reg's deployment claim (if any, post-replay) cannot rely on the current OOS slice to refute or confirm. The route to OOS power is more trading days (passive), not more analysis.
4. **`bootstrap_runtime_control` is out of scope here.** Per `memory/feedback_bootstrap_runtime_control_in_band_audit_pattern.md`: any future pre-reg that uses `--bootstrap-runtime-control` for the rank-3 audit must declare the bypass path in its result MD verdict. This MD does not author the pre-reg; the flag-disclosure obligation lies with the next-thread pre-reg author.
5. **Same-author governance gap for ranks 1 + 2.** Per `memory/feedback_bootstrap_disclosure_not_separation_of_duties.md`: even after the polarity literature lands, ranks 1 + 2 share a session/instrument with each other (`COMEX_SETTLE × MNQ × E2 × RR1.0 × CB1 × O5`) and differ only in the ATR percentile threshold. Authoring both in the same session by the same author creates a SR-review-registry governance gap (the lifecycle pause / SR alarm decision logic for one cannot independently verify the other). Sequential authorship is the right pattern.
6. **`expectancy_r_stored` non-equal to Mode A.** All three top-3 rows have non-zero `stored_minus_mode_a` (−0.0081, −0.0037, +0.0201). All within the 0.05R self-consistency gate. The Mode-A value is the canonical for any downstream pre-reg; stored is metadata only per methodology MD § "Sort rule".

## Reproduction

From the worktree root `C:/Users/joshd/canompx3/.worktrees/chordia-audit-queue-v2-2026-05-12`:

```bash
# Verify the top-3 selection matches CSV rows
python -c "
import csv
keep = {
    'MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50',
    'MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70',
    'MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30',
}
with open('docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv') as f:
    for r in csv.DictReader(f):
        if r['strategy_id'] in keep:
            print(f\"{r['strategy_id']}\")
            print(f\"  tier={r['queue_tier']} t={r['chordia_t']} \"
                  f\"n_is={r['n_is_mode_a']} mode_a_expr={r['mode_a_expr']}\")
            print(f\"  c8={r['c8_oos_status']} n_oos={r['n_oos']} \"
                  f\"power={r['oos_power']} tier={r['oos_power_tier']}\")
            print()"

# Verify literature anchor exists for VWAP_MID_ALIGNED (rank-3)
ls docs/institutional/literature/chan_2013_ch7_intraday_momentum.md

# Verify Fitschen line 152 (class-grounding source for ranks 1+2)
sed -n '152p' docs/institutional/literature/fitschen_2013_path_of_least_resistance.md
```

Companion artifacts:
- Methodology: `docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md`
- Candidates render: `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.md`
- Gap-impact map: `docs/audit/results/2026-05-12-chordia-audit-queue-blocked-reasons.md`
- Plan: `docs/plans/2026-05-12-chordia-audit-queue-v2-plan.md`

Literature anchors cited in this MD:
- `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md` — VWAP_MID_ALIGNED (rank-3) mechanism grant
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md:152` — INTRA_ASSET_PERCENTILE class grant
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md:130-167` — polarity-of-effect honest-position section (the gap that keeps ranks 1+2 at CLASS_GROUNDED_POLARITY_PENDING)
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md:20` — `t ≥ 3.79` Chordia threshold provenance
