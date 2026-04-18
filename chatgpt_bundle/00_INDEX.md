# ChatGPT Project Knowledge — Index (v3)

**Project:** Multi-instrument futures data pipeline + ORB breakout trading research (MGC, MNQ, MES). Primary operator: solo quant researcher running institutional-grade discovery with pre-registered criteria.

**Your role, ChatGPT:** Be a statistically rigorous research partner. Cite the bundled file when making a claim. Never invent numbers. Never recommend re-implementing canonical logic. Push back when the user proposes something tested-dead or parameter-tunes against OOS.

**Bundle structure:** two folders on disk.

- **`chatgpt_bundle/` = 20 files** (default — upload these). Self-contained: T0-T8 summary, institutional rigor, integrity guardian, CTE trap, prop firm constraints are all inlined into the meta files here. Works for **every ChatGPT tier** (Plus 20-cap, Pro, Team).
- **`chatgpt_bundle_optional/` = 16 files** (optional — only add if you want deeper literature / niche rule coverage). These include: full T0-T8 protocol, 4 extra literature papers (Lopez-Bailey false strategy, Carver sizing, Chan 2008 regime, Pepelyshev monitoring), prop-firm full rules, pipeline architecture doc, pre-reg template, etc.

**TL;DR upload recipe:**
- **Default (any tier):** drag `chatgpt_bundle/*.md` into ChatGPT Projects → done.
- **Pro + want deeper coverage:** also drag `chatgpt_bundle_optional/*.md`. Max 36 files total.
- **Verify your ChatGPT tier's actual file cap at upload time** — shifts over 2025-2026.

**Compiled:** 2026-04-18. Total combined size ~600 KB across 36 files.

---

## Read-order (fresh session)

1. **`00_INDEX.md`** (this file) — map + glossary + **provenance honesty section**
2. **`01_OPERATING_RULES.md`** — how to think / what to refuse
3. **`02_USER_PROFILE.md`** — user's working style + escalation triggers
4. **`07_PLAYBOOKS.md`** — answer-skeletons for common question types
5. **`04_DECISION_LOG.md`** — why we chose what we chose
6. **`06_RD_GRAVEYARD.md`** — consolidated NO-GO registry (beyond STRATEGY_BLUEPRINT)
7. **`CANONICAL_VALUES.md`** — instruments / costs / sessions / holdout / profiles
8. **`TRADING_RULES.md`** — what we trade, sessions, entry models, risk
9. **`RESEARCH_RULES.md`** — statistical standards, stats thresholds
10. **`STRATEGY_BLUEPRINT.md`** — research routing + **NO-GO registry** (tested-dead hypotheses)
11. **`pre_registered_criteria.md`** — the 12 locked criteria
12. **`mechanism_priors.md`** — what drives the ORB edge + signal-to-role mapping
13. **`backtesting-methodology.md`** — 13 mandatory rules
14. **`edge-finding-playbook.md`** — 12 commandments + niche ladder
15. Everything else (rules, architecture, literature) — by topic

---

## File map — all 36 files (20 `[CORE]` for Plus, +16 `[EXTRA]` for Pro)

### Meta + ChatGPT-native (7, ALL CORE)
| Tag | File | Size | Purpose |
|-----|------|------|---------|
| `[CORE]` | `00_INDEX.md` | ~14KB | This file — map, glossary, provenance |
| `[CORE]` | `01_OPERATING_RULES.md` | ~9KB | How ChatGPT should think/refuse + inline T0-T8 summary |
| `[CORE]` | `02_USER_PROFILE.md` | ~6KB | User's style, triggers, typo map |
| `[CORE]` | `04_DECISION_LOG.md` | ~11KB | Why we chose what we chose |
| `[CORE]` | `06_RD_GRAVEYARD.md` | ~11KB | Consolidated NO-GO |
| `[CORE]` | `07_PLAYBOOKS.md` | ~10KB | Answer-skeletons for common Q types |
| `[CORE]` | `CANONICAL_VALUES.md` | ~9KB | Cost/session/instrument/holdout/profile tables |

(Numbers 03, 05 are gaps in the numbering — room to add ACTIVE_HYPOTHESES / PROJECT_VISION later if wanted.)

### Canonical rules (10)
| Tag | File | Source in repo |
|-----|------|----------------|
| `[CORE]` | `TRADING_RULES.md` | `<root>/TRADING_RULES.md` |
| `[CORE]` | `RESEARCH_RULES.md` | `<root>/RESEARCH_RULES.md` |
| `[CORE]` | `STRATEGY_BLUEPRINT.md` | `docs/STRATEGY_BLUEPRINT.md` — has NO-GO registry |
| `[CORE]` | `pre_registered_criteria.md` | `docs/institutional/pre_registered_criteria.md` |
| `[CORE]` | `mechanism_priors.md` | `docs/institutional/mechanism_priors.md` |
| `[CORE]` | `backtesting-methodology.md` | `.claude/rules/backtesting-methodology.md` |
| `[EXTRA]` | `edge-finding-playbook.md` | `docs/institutional/edge-finding-playbook.md` |
| `[EXTRA]` | `research-truth-protocol.md` | `.claude/rules/research-truth-protocol.md` — summarized in `01_OPERATING_RULES.md` |
| `[EXTRA]` | `quant-audit-protocol.md` | `.claude/rules/quant-audit-protocol.md` — T0-T8 summarized in `01_OPERATING_RULES.md` |
| `[EXTRA]` | `prop-firm-official-rules.md` | `resources/prop-firm-official-rules.md` — key constraints in `CANONICAL_VALUES.md` §5 |

### Supplementary rules / frameworks (5, ALL EXTRA)
| Tag | File | Source |
|-----|------|--------|
| `[EXTRA]` | `institutional-rigor.md` | `.claude/rules/` — 8 non-negotiables (summary in `01_OPERATING_RULES.md`) |
| `[EXTRA]` | `integrity-guardian.md` | `.claude/rules/` — 7 behavioral rules (summary in `01_OPERATING_RULES.md`) |
| `[EXTRA]` | `daily-features-joins.md` | `.claude/rules/` — CTE triple-join trap (summary in `07_PLAYBOOKS.md` Playbook 6) |
| `[EXTRA]` | `finite_data_framework.md` | `docs/institutional/` — finite-data constraints |
| `[EXTRA]` | `regime-and-rr-handling-framework.md` | `docs/institutional/` — regime + RR handling |

### Project context (3, ALL EXTRA)
| Tag | File | Source |
|-----|------|--------|
| `[EXTRA]` | `ARCHITECTURE.md` | `docs/ARCHITECTURE.md` — canonical vs derived layers |
| `[EXTRA]` | `hypothesis_registry_template.md` | `docs/institutional/` — pre-reg template |
| `[EXTRA]` | `institutional_README.md` | `docs/institutional/README.md` — institutional framework overview |

### Literature (11 verbatim extracts, one paper per file)
| Tag | File | Paper | What it answers |
|-----|------|-------|-----------------|
| `[CORE]` | `LIT_bailey_2013_pseudo_math_minBTL.md` | Bailey et al 2013 | MinBTL bound, data-snooping in finance |
| `[CORE]` | `LIT_bailey_lopez_2014_deflated_sharpe.md` | Bailey-Lopez de Prado 2014 | Deflated Sharpe Ratio (DSR) |
| `[CORE]` | `LIT_harvey_liu_2015_backtesting_haircut.md` | Harvey-Liu 2015 | BHY haircut, multiple-testing in finance |
| `[CORE]` | `LIT_chordia_2018_two_million_t379.md` | Chordia et al 2018 | t ≥ 3.79 threshold, no prior theory |
| `[CORE]` | `LIT_lopez_2020_ml_asset_managers.md` | Lopez de Prado 2020 | CPCV, theory-first ML, backtest overfitting |
| `[CORE]` | `LIT_chan_2013_ch1_backtesting_invariant.md` | Chan 2013 ch 1 (pp 1-10) | Look-ahead invariant (p4), data-snooping, survivorship |
| `[CORE]` | `LIT_fitschen_2013_orb_premise.md` | Fitschen 2013 ch 3 | Intraday trend-follow (ORB premise) |
| `[EXTRA]` | `LIT_lopez_bailey_2018_false_strategy.md` | Lopez de Prado-Bailey 2018 | False Strategy Theorem, E[max_SR] |
| `[EXTRA]` | `LIT_carver_2015_vol_target_sizing.md` | Carver 2015 | Volatility targeting, Kelly-linked sizing |
| `[EXTRA]` | `LIT_chan_2008_ch7_regime_switching.md` | Chan 2008 ch 7 | Regime switching |
| `[EXTRA]` | `LIT_pepelyshev_2015_cusum_monitoring.md` | Pepelyshev-Polunchenko 2015 | CUSUM / Shiryaev-Roberts monitoring |

### `[CORE]` 20 — strict Plus-tier set

Meta (7) + rules (6) + lit (7) = 20.

Upload these to Plus. They're self-contained: key content of every `[EXTRA]` is inlined or summarized in one of the CORE meta files:
- T0-T8 audit framework → summary in `01_OPERATING_RULES.md` (§11a added for Plus parity)
- Institutional rigor / integrity guardian → summary in `01_OPERATING_RULES.md` §7 + §8
- Research truth protocol → summary in `01_OPERATING_RULES.md` §5 + §12
- Daily features CTE trap → summary in `07_PLAYBOOKS.md` Playbook 6
- Prop firm constraints → summary in `CANONICAL_VALUES.md` §5 + `04_DECISION_LOG.md` §14
- False Strategy Theorem (LIT) → the core result is restated inside `LIT_bailey_2013_pseudo_math_minBTL.md` and `LIT_bailey_lopez_2014_deflated_sharpe.md` (these papers cite each other)

### Path-reference convention

Bundled files cite **repo paths** (e.g. `docs/institutional/pre_registered_criteria.md`, `.claude/rules/backtesting-methodology.md`). These refer to where the content lives IN THE USER'S REPO — they are breadcrumbs for the user to re-verify, not file paths inside the bundle. The bundle is flat: the same file exists as `pre_registered_criteria.md` (no folder) in this Project's uploaded files. When ChatGPT encounters a repo-path citation, look up the equivalent flat filename in this INDEX's file map.

---

## Topic → file map

| Topic | Go to |
|-------|-------|
| "Is this hypothesis worth testing?" | `07_PLAYBOOKS.md` Playbook 1 + `06_RD_GRAVEYARD.md` + `STRATEGY_BLUEPRINT.md` + `pre_registered_criteria.md` |
| "Review this backtest for bias" | `07_PLAYBOOKS.md` Playbook 2 + `backtesting-methodology.md` + `quant-audit-protocol.md` |
| "Multiple testing / BH-FDR" | `LIT_bailey_2013_pseudo_math_minBTL.md` + `LIT_harvey_liu_2015_backtesting_haircut.md` + `LIT_chordia_2018_two_million_t379.md` |
| "Is my Sharpe real?" | `LIT_bailey_lopez_2014_deflated_sharpe.md` + `LIT_lopez_bailey_2018_false_strategy.md` |
| "Why does ORB work?" | `LIT_fitschen_2013_orb_premise.md` + `mechanism_priors.md` |
| "Look-ahead bias invariant" | `LIT_chan_2013_ch1_backtesting_invariant.md` (book p4) |
| "How should I size positions?" | `LIT_carver_2015_vol_target_sizing.md` + `mechanism_priors.md` R8 role |
| "Regime switching / monitoring" | `LIT_chan_2008_ch7_regime_switching.md` + `LIT_pepelyshev_2015_cusum_monitoring.md` |
| "What's the cost for MGC?" | `CANONICAL_VALUES.md` §2 |
| "When does session X start in Brisbane?" | `CANONICAL_VALUES.md` §3 |
| "Can I train on 2026 data?" | `CANONICAL_VALUES.md` §4 + `04_DECISION_LOG.md` §1 (answer: NO, sacred) |
| "Is this strategy dead?" | `STRATEGY_BLUEPRINT.md` NO-GO registry + `06_RD_GRAVEYARD.md` |
| "CTE join looks wrong" | `daily-features-joins.md` |
| "How do I pre-register a hypothesis?" | `hypothesis_registry_template.md` |
| "What's the T0-T8 audit framework?" | `quant-audit-protocol.md` |
| "Why Mode A holdout?" | `04_DECISION_LOG.md` §1 + `pre_registered_criteria.md` Amendment 2.7 |
| "Prop firm question" | `prop-firm-official-rules.md` + `CANONICAL_VALUES.md` §5 + `04_DECISION_LOG.md` §14 |
| "Finite data / short history" | `finite_data_framework.md` |
| "Regime + RR handling" | `regime-and-rr-handling-framework.md` |

---

## Glossary — project-specific terms

| Term | Meaning |
|------|---------|
| **ORB** | Opening Range Breakout — strategy class based on range in first N minutes of a session |
| **Aperture** | ORB window length: **5, 15, or 30** minutes |
| **RR** | Risk-Reward ratio (target ÷ stop): canonical set **1.0, 1.5, 2.0** |
| **ExpR** | Expected R per trade. **Primary sort key** (never Sharpe alone). |
| **E2** | Stop-market entry model (canonical). No alternative currently deployed. |
| **E_RETEST** | Limit-on-retest entry (Phase C stub, not yet built). |
| **CB0/1/2/3** | Confirm bars before entry. CB1 most common. |
| **Session** | Named time anchor for ORB start (12 active — see `CANONICAL_VALUES.md` §3). |
| **Lane** | `instrument × session × aperture × RR × entry_model` — unit of deployment. |
| **Strategy / setup** | Lane + filter stack (e.g., G5 compression + G4 cost gate). |
| **G-filters (G1-G9)** | Canonical filter families — see `TRADING_RULES.md`. |
| **Pathway A / B** | Validation paths. Under Mode A, Pathway B = holdout-clean discovery. |
| **Mode A / Mode B** | Holdout policy modes. **Mode A operative** since Amendment 2.7 (2026-04-08): 2026-01-01 sacred. |
| **BH-FDR** | Benjamini-Hochberg False Discovery Rate. Multi-framing per `backtesting-methodology.md` Rule 4. |
| **MinBTL** | Minimum Backtest Length (Bailey 2013): `2·ln(N_trials) / E[max_N]²`. |
| **DSR** | Deflated Sharpe Ratio (Bailey-LdP 2014). |
| **WFE** | Walk-Forward Efficiency. WFE > 0.95 flags LEAKAGE_SUSPECT. |
| **t-stat** | Chordia strict ≥ 3.79 (no prior); general ≥ 3.0 (with economic prior). |
| **Δ_IS, Δ_OOS** | In/out-of-sample performance delta. `dir_match` required. |
| **Tautology (T0)** | `|corr| > 0.70` with deployed filter → killed. |
| **ARITHMETIC_ONLY** | `|wr_spread| < 3%` AND `|Δ_IS| > 0.10` → cost-screen, not edge. |
| **NO-GO** | Tested + confirmed dead. See `STRATEGY_BLUEPRINT.md` + `06_RD_GRAVEYARD.md`. |
| **Seven Sins** | Quant bias families: look-ahead / snooping / overfitting / survivorship / storytelling / outliers / cost illusion. |
| **Canonical source** | Single-source-of-truth file. Never re-encode. See `CANONICAL_VALUES.md` §6. |
| **Trading day** | 09:00 Brisbane → next 09:00 Brisbane (UTC+10, no DST). |
| **XFA / LFA** | TopStep Express / Live Funded Account. Mutually exclusive per firm rule. |
| **R1-R8 roles** | Signal-to-implementation roles in `mechanism_priors.md`: R1 FILTER → R8 PORTFOLIO allocator. |
| **T0-T8** | Audit protocol in `quant-audit-protocol.md` (T0 tautology, T3 leakage, T4 sensitivity, T7 per-year, T8 cross-instrument, etc). |

---

## ⚠ PROVENANCE & HONESTY SECTION

**Why this section exists:** the user's correction rule `CLAUDE.md` § "Local Academic / Project-Source Grounding Rule" is explicit — don't characterize what you haven't verified. This section documents what was verified vs trusted vs flagged during bundle assembly.

### Tier 1 — Verified by direct read during bundle assembly (2026-04-18)
- `pipeline/asset_configs.py` — read in full; `CANONICAL_VALUES.md` §1 sourced from here
- `pipeline/cost_model.py` — read in full; `CANONICAL_VALUES.md` §2 sourced from here
- `pipeline/dst.py` — read in full; `CANONICAL_VALUES.md` §3 sourced from here
- `trading_app/holdout_policy.py` — read in full; `CANONICAL_VALUES.md` §4 sourced from here
- `trading_app/prop_profiles.py` — read lines 340-440 partially; `CANONICAL_VALUES.md` §5 is partial (header rows only; ask user for complete current state)
- `resources/Algorithmic_Trading_Chan.pdf` pages 1-10 — read directly; `LIT_chan_2013_ch1_backtesting_invariant.md` quotes verbatim from these pages. Pages 11+ of ch1 and all later chapters NOT read.

### Tier 2 — Copied from repo trusting prior project work (not re-verified)
All rule files and 10 pre-existing literature extracts were copied from the repo without a fresh top-to-bottom read during bundle assembly. The user's prior work is canonical for its intended use; content is accurate to whatever was there in the source files at copy time:
- `TRADING_RULES.md`, `RESEARCH_RULES.md`, `STRATEGY_BLUEPRINT.md`, `pre_registered_criteria.md`, `mechanism_priors.md`, `edge-finding-playbook.md`, `backtesting-methodology.md`, `research-truth-protocol.md`, `quant-audit-protocol.md`, `prop-firm-official-rules.md`, `ARCHITECTURE.md`, `institutional-rigor.md`, `integrity-guardian.md`, `daily-features-joins.md`, `finite_data_framework.md`, `regime-and-rr-handling-framework.md`, `hypothesis_registry_template.md`, `institutional_README.md`
- All 10 pre-existing `LIT_*.md` extract files (Bailey 2013, Bailey-Lopez 2014, Lopez-Bailey 2018, Harvey-Liu, Chordia, Lopez 2020, Carver, Fitschen, Chan 2008 ch7, Pepelyshev)

### Tier 3 — Constructed from memory-index surfaced in session reminders
Generated files synthesize content from the user's `memory/MEMORY.md` index that was surfaced as system-reminder context. Specific numbers cited should be verified against primary files before research reuse:
- `06_RD_GRAVEYARD.md` — entries compiled from memory-index summary lines. Stats like "CPCV AUC 0.50", "corr=0.069", "rel_vol Sharpe 0.05-0.17" come from memory, not primary audit files.
- `04_DECISION_LOG.md` — reasoning derived from the code files I read + memory-index summaries. Reasoning for user preferences is inferred from patterns, not quoted verbatim from user statements.

### Tier 4 — Constructed from training memory with explicit labels
- **ChatGPT Projects file-count limit** ("20 Plus / 40 Pro"): from training memory — **verify in current ChatGPT settings before acting on this**. The limit may have changed since my cutoff.
- Any reference to Chan / Aronson / Carver content OUTSIDE the `LIT_*` files in this bundle: must be labeled "from training memory — not verified against local PDF."

### Files explicitly NOT in bundle + why
- `TRADING_PLAN.md` (from repo root) — **volatile**, 2026-03-01 snapshot, portfolio has moved on. Paste inline when needed.
- `HANDOFF.md`, `REPO_MAP.md`, any memory files — volatile, change fast. Paste inline.
- `docs/specs/*.md` — feature specs, relevant only when building a specific feature.
- Raw PDFs in `resources/` — would waste ChatGPT's quota on front matter / bibliography.
- Current strategy book, fitness verdicts, live P&L — volatile; paste output from `/trade-book` via Claude.

### Gaps honestly acknowledged (potential follow-up work)
The following primary sources exist in `resources/` but are **NOT yet extracted**. If a question relies on them, ask the user to run a focused PDF-extraction session:
- **Aronson, *Evidence-Based Technical Analysis* ch1 + ch6** — ch6 "Data-Mining Bias" is referenced by `quant-audit-protocol.md` but the chapter content is not in this bundle.
- **Chan 2013 ch1 pages 11+** and **ch7 "Intraday Momentum Strategies"** — ch7 most adjacent to our ORB work.
- **Chan 2008 *Quantitative Trading*** — different book from Chan 2013; only ch7 is extracted so far.
- **Benjamini-Hochberg 1995** — original FDR paper. We use BH-FDR everywhere but no extract.
- **Carver *Systematic Trading* ch4 + ch6 + ch8** — sizing extract (ch9-10) is present; trading-subsystem and forecast-combination chapters are not.
- **Man Group 2015 overfitting paper** — referenced in general context; not extracted.
- **Pardo *Design & Testing of Trading Strategies*** — referenced but not extracted.
- **Building Reliable Trading Systems** — resource exists, not extracted.

### Self-audit of bias in generated files
- I introduced a fabricated quote ("the holdout must actually be untouched") in `04_DECISION_LOG.md` §1 on the first draft; it has been removed in this bundle and replaced with citations to the actual docstring language in `holdout_policy.py`.
- `06_RD_GRAVEYARD.md` now carries a provenance note that memory-index summaries should be re-verified against primary audit files before being cited in new research docs.
- `04_DECISION_LOG.md` §14 (prop firm scaling) cites memory file `topstep_scaling_corrected_apr15.md` as the primary source; the numbers there should be re-checked against the canonical `memory/prop_firm_complete_comparison_apr1.md` before operational use.

---

## What's NOT in this bundle (paste inline)

- Current strategy book / live lanes / fitness verdicts → `/trade-book` via Claude
- Recent audit results (T0-T8 outputs) → paste the specific result file
- `gold.db` data / raw query output → query `gold-db` MCP via Claude
- Current git state / HANDOFF.md → changes hourly
- Specific code files outside CANONICAL_VALUES → paste the relevant function

---

## How to handle ChatGPT failure modes

1. **Making up a number** — STOP. Ask the user to paste. `CANONICAL_VALUES.md` has stable constants; everything else is volatile.
2. **Recommending a kill-listed strategy** — check `STRATEGY_BLUEPRINT.md` NO-GO + `06_RD_GRAVEYARD.md` first. If match → tell the user + cite the postmortem.
3. **Parameter-tuning against OOS** — refuse. Cite `04_DECISION_LOG.md` §1 (Mode A) + `pre_registered_criteria.md`.
4. **Citing a paper from training memory** — only if the `LIT_*` file doesn't cover it. Label: "from training memory — not verified against local PDF."
5. **Recommending a new module when a canonical one exists** — don't. Cite `CANONICAL_VALUES.md` §6.
6. **Offering "ship it anyway"** — don't. See `01_OPERATING_RULES.md` §7 and `institutional-rigor.md`.
7. **Claiming you read a paper you didn't** — look it up in this INDEX's provenance section before stating. If a paper is in Tier 4 or the "Gaps" list, say "not in bundle — training memory only."
