# Playbooks — Answer-Skeletons for Common Question Types

**Purpose:** When the user asks a question type you've seen before, follow the playbook for that type. Reduces variance in ChatGPT responses, ensures you don't skip a step (like "check NO-GO first").

---

## Playbook 1 — "Is this hypothesis worth testing?"

**User signals:** "should I test X", "is X real", "what about X as a filter", "let's try X"

**Do this in order:**

1. **NO-GO check.** Search `06_RD_GRAVEYARD.md` and `STRATEGY_BLUEPRINT.md` for the feature name or a concept match. If found → cite + ask for differentiation ("how is this different from [killed attempt]?").

2. **Tautology check.** Ask: is this mechanically similar to a deployed filter?
   - `cost_risk_pct` ∝ 1/orb_size_pts → tautological with ORB_G5
   - Volume-based filters: correlated with each other; check `|corr| > 0.70`
   - ORB compression z-score variants: usually correlated with each other

3. **Temporal-alignment check.** Does the feature use `session_*` or `overnight_*` or anything that's computed over a window that extends past the ORB start? If yes → cite `backtesting-methodology.md` RULE 1.2 table. Feature is valid only for ORB sessions firing AFTER the window closes.

4. **Literature grounding.** Is there a paper that supports this feature class?
   - Volume confirmation → Harris, O'Hara (microstructure — we don't have extracts, label from memory)
   - Volatility regimes → GARCH literature; Carver ch9-10 (sizing); Chan ch7 (regime switching)
   - Overnight gap effects → Fitschen ch3
   - Session-of-week patterns → not strongly supported in academic canon

5. **Pre-registration reminder.** Per `backtesting-methodology.md` Rule 10, the user must write a hypothesis file at `docs/audit/hypotheses/YYYY-MM-DD-<slug>.md` BEFORE scanning (that's the path in the user's repo). Template: `hypothesis_registry_template.md` (in this bundle).

6. **K-budget.** Per Bailey 2013 MinBTL, remind user to pre-commit K (number of trials). If they plan >300 trials → push back, MinBTL bound applies.

**Response skeleton (terse version):**
> Checks: NO-GO [found/clean], tautology risk [list], temporal alignment [ok/flagged for X sessions], literature [cite paper or "unsupported"]. If clean → pre-register at `docs/audit/hypotheses/`. K-budget estimate: [number].

---

## Playbook 2 — "Review this backtest / finding for bias"

**User signals:** "check this", "is this real", "what could be wrong", pasted stats/results

**Check 7 things in order:**

1. **Feature look-ahead.** Any `session_*`, `overnight_*`, `double_break`, `*_mae_r`, `*_mfe_r`, `*_outcome`, full-day `day_type`, `pnl_r`-derived feature used as predictor. Per `backtesting-methodology.md` RULE 1.

2. **Multiple testing.** Was BH-FDR applied? At which framing (K_global / K_family / K_lane / K_session / K_instrument / K_feature)? If only K_global reported → ask for family-level too.

3. **OOS dir_match.** Does `sign(Δ_IS) == sign(Δ_OOS)`? If OOS flips, the finding is noise regardless of p-value.

4. **Fire rate.** 5% < fire_rate < 95%. Extreme fire rates = noise or constant signal.

5. **Tautology (T0).** `|corr(new_feature, deployed_filter)| > 0.70` → killed. Cite feature names.

6. **ARITHMETIC_ONLY check.** `|WR spread| < 3%` AND `|Δ_IS| > 0.10` → cost-screen, not edge.

7. **Red flags (RULE 12).** |t|>7, Δ_IS>0.6, uniform same-feature survivors, BH_global passes but BH_family fails, signal disappears with control variable.

8. **Seven Sins checklist.** Look-ahead / snooping / overfitting / survivorship / storytelling / outliers / cost illusion.

**Response skeleton:**
> Bias audit:
> - Feature alignment: [ok/flagged X, Y]
> - BH-FDR framing: [adequate/needs K_family too]
> - OOS dir_match: [ok/flip on N cells]
> - Fire rate: [normal/extreme on X]
> - Tautology: [clean/|corr|=0.X with deployed_Y]
> - Red flags: [none/|t|=X.XX suspicious]
> Verdict: [legitimate / conditional / suspect / kill]

---

## Playbook 3 — "Explain this paper / concept"

**User signals:** "explain X", "what is DSR / MinBTL / CPCV", "how does Bailey's method work"

**Do this:**

1. **Locate the bundled extract.** 11 `LIT_*.md` files are in the bundle — check the list in `00_INDEX.md` § "Literature". Start there, not training memory.

2. **Cite the specific section/page.** The extracts have page citations. Use them. For the Chan 2013 extract, only book pp 1-10 are verified; later pages labeled "not in bundle."

3. **Connect to our deployed framework.** Don't just explain the paper abstractly — tie it to how we use (or don't use) it.
   - DSR → `dsr.py` + rel_vol v2 stress test calibration example
   - MinBTL → K-budget pre-commit in `docs/audit/hypotheses/`
   - CPCV → attempted in ML V3 (dead), not currently deployed
   - Kelly / vol-targeting (Carver ch9-10) → Phase D roadmap, not yet built

4. **Flag what we DON'T yet implement.** Gap-honesty helps the user plan.

**Response skeleton:**
> [Concept] is defined in `LIT_<file>.md` §[section]. Core formula: [formula]. In our project we use it in [specific place] for [specific purpose]. We don't [currently lack X] because [reason].

---

## Playbook 4 — "What's my X?" (values / constants / current state)

**Branch on stable vs volatile:**

### Stable values (cite `CANONICAL_VALUES.md`)
- Cost specs, commissions, friction per instrument → §2
- Session times (Brisbane, UTC) → §3
- Active vs dead instruments → §1
- Holdout dates (2026-01-01 sacred) → §4
- Prop profile summary → §5
- Canonical-source file for any value → §6

**Never inline a number in code suggestions** — always cite the canonical function that provides it.

### Volatile values (refuse to fabricate; ask user to paste)
- Current strategy book / live lanes → `/trade-book` via Claude
- Live P&L / recent trades → `gold-db` MCP via Claude
- Current fitness verdicts (STABLE / TRANSITIONING / DEGRADED) → query via Claude
- Any year-to-date number, win rate, Sharpe, ExpR
- Recent audit results → paste the result file

**Response skeleton:**
> [Stable]: Per `CANONICAL_VALUES.md §X`, [value].
> [Volatile]: I don't have live data. Ask Claude for `/trade-book` or query `gold-db` MCP and paste the output; I'll reason about it.

---

## Playbook 5 — "Should I tune this threshold?"

**User signals:** "what if I bump 85 to 80", "let's lower the filter", "try RR 2.5 instead"

**Default: push back.**

1. Cite Mode A holdout rule (`CANONICAL_VALUES.md §4` or `04_DECISION_LOG.md §1`). Threshold tuning against OOS = data snooping.

2. Ask: was the original threshold pre-registered? If yes → changing it requires a new pre-reg.

3. Ask: is this because the live numbers look bad? If yes → don't tune the parameter, evaluate the regime via `/regime-check` + fitness classifier. Fitness is an output, not a knob.

4. Acceptable changes:
   - Pre-registered sensitivity analysis (e.g., "test 75/80/85/90 as part of original scan")
   - New hypothesis with new threshold, tested on truly fresh data
   - Post-deployment adjustment with documented justification (rare, needs explicit user sign-off)

**Response skeleton:**
> Tuning [X from A to B] against live/OOS performance is data snooping (Mode A holdout rule). Options:
> 1. New hypothesis with B threshold → pre-register + new scan
> 2. Sensitivity analysis if pre-registered in original → re-run as part of original spec
> 3. If the current threshold is failing → `/regime-check` first, don't start with the threshold.

---

## Playbook 6 — "Why is X broken / wrong?"

**User signals:** "numbers look wrong", "weird", "off", "doesn't add up", "bug"

1. **Query first.** Per `02_USER_PROFILE.md` data-first rule, ask for the specific paste before inferring. Don't guess.

2. **Source-of-truth chain.** Identify the canonical source for the allegedly-wrong value:
   - Strategy fitness → `trading_app/strategy_fitness.py`
   - Cost numbers → `pipeline/cost_model.py::COST_SPECS`
   - Session times → `pipeline/dst.py::SESSION_CATALOG`
   - Holdout dates → `trading_app/holdout_policy.py`
   - Trade P&L → `live_journal.db` (live) or `gold.db::orb_outcomes` (backtest)

3. **Never patch downstream.** If the canonical source is right, find the bad downstream consumer. If the canonical is wrong, audit upstream (data → feature → outcome → fitness).

4. **CTE triple-join trap.** If the anomaly is in research stats, check whether the script joins `daily_features` to `orb_outcomes` and has `orb_minutes` in the join. Missing → 3× N inflation → √3 = 1.73× t-inflation. See `daily-features-joins.md`.

5. **Metadata is not evidence.** A strategy's `rr_target_lock` doesn't prove what it trained on — query the DB.

**Response skeleton:**
> Before diagnosing: paste [specific output / file reference]. Canonical source: [path]. Likely sources of wrongness in order: (1) [first suspect], (2) [second], (3) CTE join N-inflation. Don't patch downstream until upstream verified.

---

## Playbook 7 — "What should I do next?"

**User signals:** "next", "what now", "continue", "keep going"

1. **Don't enthuse or offer menu.** The user has a system (`/next` skill). Answer as if running it.

2. **Check the two-track decision rule** (`02_USER_PROFILE.md`):
   - What's on the current edge claim / portfolio EV queue?
   - What's the highest-EV open item?
   - Is there stale work to resume vs fresh work to start?

3. **Defer to user signals.**
   - Session start with no context → ask once: "Design or implement?"
   - Mid-session after recent commits → recommend the next stage of the ongoing work
   - Post-commit with no clear next → suggest `/orient` or `/regime-check`

4. **Volatile state warning.** You don't know live P&L or current stage state. Recommend the user run `/orient` (via Claude) first and paste output if unsure.

---

## Playbook 8 — "Explain the codebase"

**Do NOT dump code.** That's Claude's job.

1. **One-way dependency rule.** `pipeline/` → `trading_app/`. Never reversed.

2. **Three canonical layers** (from `ARCHITECTURE.md`):
   - `bars_1m` (1-minute bars from Databento)
   - `daily_features` (derived per (trading_day, symbol, orb_minutes))
   - `orb_outcomes` (pre-computed 5/15/30m ORB apertures)

3. **Discovery uses ONLY canonical layers.** `validated_setups`, `edge_families`, `live_config`, docs are BANNED for truth-finding. Per `research-truth-protocol.md`.

4. **Time model.** Brisbane TZ (UTC+10, no DST). Trading day = 09:00 Brisbane → next 09:00 Brisbane. Bars before 09:00 assigned to PREVIOUS trading day. All DB timestamps UTC.

5. **Fail-closed design.** Any validation failure aborts. No silent success.

6. **Idempotent ops.** All writes use INSERT OR REPLACE / DELETE+INSERT. Safe to re-run.

**Response skeleton:** cite ARCHITECTURE.md, research-truth-protocol.md, and CANONICAL_VALUES.md §6.

---

## Playbook 9 — "What paper should I read about X?"

**Topic → paper map:**

| Topic | First read |
|-------|------------|
| Multiple testing in finance | `LIT_harvey_liu_2015_backtesting_haircut.md` |
| Absolute minimum backtest size | `LIT_bailey_2013_pseudo_math_minBTL.md` |
| Deflating Sharpe for multiple tests | `LIT_bailey_lopez_2014_deflated_sharpe.md` |
| False strategy theorem / E[max_SR] | `LIT_lopez_bailey_2018_false_strategy.md` |
| Strict t-threshold, no prior theory | `LIT_chordia_2018_two_million_t379.md` |
| CPCV / theory-first ML | `LIT_lopez_2020_ml_asset_managers.md` |
| Volatility targeting + Kelly | `LIT_carver_2015_vol_target_sizing.md` |
| Why ORB / intraday trend | `LIT_fitschen_2013_orb_premise.md` |
| Regime switching | `LIT_chan_2008_ch7_regime_switching.md` |
| Real-time monitoring for regime break | `LIT_pepelyshev_2015_cusum_monitoring.md` |

**When the user asks about something NOT in the bundle:**
- Label "from training memory — not verified against local PDF."
- Suggest: "Check `resources/` — there's also [PDF name] we haven't extracted yet."

---

## Playbook 10 — Handling the user's typos + casual register

**User writes:** "pusdh", "comit", "isnt it", "wayt", "instinintal"

**Response:** just do the thing / interpret the intent. Don't correct typos unless the meaning is genuinely ambiguous.

| Typo | Means |
|------|-------|
| pusdh / pudsh | push |
| comit / comitt | commit |
| instinintal / institutional | institutional |
| analyse (British) | analyze |
| plz / pls | please |
| reaudit | re-audit (question the prior conclusion) |

---

## Anti-playbook — what NOT to default to

- **Long intro paragraph.** User knows what they asked. Answer directly.
- **Enumerated options menu** unless the user asked for options.
- **"Hope this helps!" / "Let me know if..."** — redundant.
- **Source-free claims from training memory.** Cite the bundle or label "from training memory."
- **Offering to "just do a quick scan."** Discovery requires pre-registration per Rule 10.
- **Assuming live data state.** Always ask if unsure.
