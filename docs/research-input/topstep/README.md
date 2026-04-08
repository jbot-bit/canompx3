---
name: TopStep Canonical Source Corpus
purpose: Frozen-in-time canonical text from help.topstep.com + topstep.com terms of use, used as the authoritative ground truth for all TopStep compliance code in this repo
type: research-input-corpus
firm: topstep
established: 2026-04-08
last_scraped: 2026-04-08
authority: PRIMARY-SOURCE — overrides training memory, MEMORY.md, code comments, and CLAUDE.md where they conflict
audit_doc: docs/audit/2026-04-08-topstep-canonical-audit.md
---

# TopStep Canonical Source Corpus

**Date assembled:** 2026-04-08
**Branch of origin:** `topstep-canonical-info`
**Audit using this corpus:** [`docs/audit/2026-04-08-topstep-canonical-audit.md`](../../audit/2026-04-08-topstep-canonical-audit.md)

## Why this exists

Per `.claude/rules/institutional-rigor.md` rule 7 ("Ground in local resources before training memory") and CLAUDE.md "Volatile Data Rule" ("NEVER cite changing stats from memory/docs"), every TopStep compliance fact in production code must be traceable to a primary source captured at a known point in time. This directory is that ground truth.

Text from `help.topstep.com` and `topstep.com/terms-of-use` changes without notice. The files in this directory are point-in-time captures and must be re-scraped quarterly (or after any TopStep policy announcement).

## Authority hierarchy

When canonical sources disagree:

1. **Help-center articles** (this directory) win over Terms of Use.
   - *Example:* `topstep_dll_article.md` says DLL is opt-in on TopstepX for TC/XFA; `topstep_terms_of_use.txt` Section 27 still describes DLL as in force. The help-center article is what the Risk team enforces in practice.
2. **Dedicated tier articles** win over general articles.
   - *Example:* `topstep_live_funded_parameters.md` LFA-specific DLL ($2K/$3K/$4.5K) wins over `topstep_dll_article.md`'s ambiguous "Trading Combine = -$1,000" listing for LFA values.
3. **Newest "Updated" timestamp** wins where all else is equal.

## File index

| File | Source URL | Article ID | Updated (per article) | Scraped | Bytes | Coverage |
|---|---|---|---|---|---|---|
| `topstep_terms_of_use.txt` | `https://www.topstep.com/terms-of-use` | — | 2026-02-06 | by user | 94608 | Full Topstep ToU including Section 27 Prohibited Conduct + Section 28 Other Prohibited Uses |
| `topstep_xfa_parameters.txt` | `https://help.topstep.com/en/articles/8284215-express-funded-account-parameters` | 8284215 | "Updated today" | by user | 16889 | XFA Standard vs Consistency rules, multiple-XFA cap, payout caps, automation allowance |
| `topstep_payout_policy.txt` | `https://help.topstep.com/en/articles/8284233-topstep-payout-policy` | 8284233 | "Updated this week" | by user | 14311 | 90/10 split (Jan 12 2026 cutoff), Standard vs Consistency payout, Wise/ACH options |
| `topstep_scaling_plan.txt` | `https://help.topstep.com/en/articles/8284223-what-is-the-scaling-plan` | 8284223 | "Updated this week" | by user | 5033 | XFA Scaling Plan rules text (chart referenced as image — see images/) |
| `topstep_dll_article.md` | `https://help.topstep.com/en/articles/10490293-topstepx-daily-loss-limit` | 10490293 | "Updated over a week ago" | 2026-04-08 | 18420 | **DLL is opt-in on TopstepX for TC/XFA, mandatory on Live Funded.** Personal/Trailing PDLL options. |
| `topstep_mll_article.md` | `https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit` | 8284204 | "Updated today" | 2026-04-08 | 5097 | MLL = $2K/$3K/$4.5K, trailing EOD-only, locks at starting balance, **XFA starts at $0** |
| `topstep_scaling_plan_article.md` | `https://help.topstep.com/en/articles/8284223-what-is-the-scaling-plan` | 8284223 | "Updated this week" | 2026-04-08 | 7568 | Re-scrape with image links, 10s grace period, no intraday scaling-up, 10:1 micro/mini |
| `topstep_xfa_commissions.md` | `https://help.topstep.com/en/articles/8284213-what-are-the-commissions-and-fees-in-the-trading-combine-and-express-funded-account` | 8284213 | "Updated today" | 2026-04-08 | 11427 | **Tradovate / Rithmic / Plus500 commission tables**, fees per instrument |
| `topstep_consistency_target.md` | `https://help.topstep.com/en/articles/8284208-what-is-the-consistency-target` | 8284208 | "Updated this week" | 2026-04-08 | 13750 | TC consistency target (best day < 50% of profit target), XFA Consistency formula |
| `topstep_cross_account_hedging.md` | `https://help.topstep.com/en/articles/13747047-understanding-hedging` | 13747047 | "Updated yesterday" | 2026-04-08 | 14583 | **Single-user / cross-account hedging is PROHIBITED. 3-strike progressive enforcement → permanent account closure.** |
| `topstep_understanding_hedging.md` | `https://help.topstep.com/en/articles/13747047-understanding-hedging` | 13747047 | (same article via alt URL) | 2026-04-08 | 14571 | Duplicate of cross_account_hedging via /understanding-hedging redirect |
| `topstep_dynamic_live_risk_expansion.md` | `https://help.topstep.com/en/articles/11748475-dynamic-live-risk-expansion` | 11748475 | "Updated over 2 months ago" | 2026-04-08 | 9604 | Replaces Scaling Plan in LFA (since 2025-07-22). Tier-based DLL & contract scaling. 10 active days per tier. |
| `topstep_live_funded_parameters.md` | `https://help.topstep.com/en/articles/10657969-live-funded-account-parameters` | 10657969 | "Updated over a week ago" | 2026-04-08 | 20037 | LFA starting balance (20% of XFA total or $10K floor), reserve unlock structure, **LFA DLL = $2K/$3K/$4.5K with $10K low-balance override** |
| `topstep_funded_rule_violation_faq.md` | `https://help.topstep.com/en/articles/8284221-funded-rule-violation-frequently-asked-questions` | 8284221 | "Updated over 7 months ago" | 2026-04-08 | 4461 | Rule break consequences, Back2Funded (2 reactivations), 30-day inactivity rule |
| `topstep_topstepx_general.md` | `https://help.topstep.com/en/articles/14434175-topstepx` | 14434175 | (recent) | 2026-04-08 | 33563 | Full TopstepX guide: Trade Copier mechanics, Risk Lock, PDPT/PDLL, contract limits, brackets, symbol blocks |
| `topstep_prohibited_conduct_helpcenter.md` | `https://help.topstep.com/en/articles/10296582-prohibited-conduct` | 10296582 | "Updated over 2 months ago" | 2026-04-08 | 8816 | Help-center version of Prohibited Conduct (includes "Single-user hedging" + "Account stacking" not in user-provided ToU file) |
| `topstep_prohibited_trading_strategies.md` | `https://help.topstep.com/en/articles/10305426-prohibited-trading-strategies-at-topstep` | 10305426 | "Updated over a year ago" | 2026-04-08 | 3916 | Account stacking, intentional LFA depletion, news event sizing prohibited |
| `topstep_risk_adjustments.md` | `https://help.topstep.com/en/articles/13613539-risk-adjustments-high-risk-high-volatility` | 13613539 | "Updated over 2 months ago" | 2026-04-08 | 8049 | **Temporary high-volatility position caps** for XFA per lot tier and special MGC table |
| `topstep_trading_combine_parameters.md` | `https://help.topstep.com/en/articles/8284197-trading-combine-parameters` | 8284197 | (recent) | 2026-04-08 | 11367 | TC tier table, profit targets, MLL, max position size for context |
| `topstep_risk_lock_in.md` | `https://help.topstep.com/en/articles/13461608-risk-lock-in` | 13461608 | (recent) | 2026-04-08 | 6972 | Daily Risk Lock-In feature mechanics |
| `images/xfa_scaling_chart.png` | `https://topstep-949ca9db770d.intercom-attachments-7.com/i/o/813170406/95c64a1874bfd81176cc2d57/19155321446035` | — | (referenced from 8284223) | 2026-04-08 | 292192 | **OFFICIAL XFA Scaling Plan ladder image**, visually parsed: 50K/100K/150K thresholds and lot limits |

**Total files:** 20 sources + 1 image = 21 artifacts. Total ~590KB.

## Visually-extracted facts

### XFA Scaling Plan ladder (parsed from `images/xfa_scaling_chart.png`)

| Account Balance | $50K XFA | $100K XFA | $150K XFA |
|---|---|---|---|
| Below $1,500 | **2 lots** | **3 lots** | **3 lots** |
| $1,500 – $2,000 | 3 lots | 4 lots | 4 lots |
| $2,000 – $3,000 | (above $2K → 5 lots) | 5 lots | 5 lots |
| $3,000 – $4,500 | — | (above $3K → 10 lots) | 10 lots |
| Above $4,500 | — | — | 15 lots |

"lots" = mini-equivalent. On TopstepX, 1 lot = 1 mini = 10 micros.

This ladder is **Day-1 starting state** for any new XFA. Bot must respect the bottom row until profit accumulates EOD.

## Known contradictions in canonical sources

1. **LFA Daily Loss Limit values** — `topstep_dll_article.md:51-55` lists `$1K/$2K/$3K` (labeled as "Trading Combine") as LFA DLL; `topstep_live_funded_parameters.md:124-130` lists `$2K/$3K/$4.5K` (matches MLL). **The LFA Parameters article wins** (per authority hierarchy rule 2). The DLL article values are most likely the legacy TC values still present in the article body.
2. **Terms of Use vs help-center on DLL existence** — ToU Section 27 still describes DLL as a normal account feature; help-center DLL article says it was removed on TopstepX for TC/XFA. **Help-center wins** (per authority hierarchy rule 1).

## URLs that returned 404 on 2026-04-08

These URLs were referenced from the user-provided .txt files but the corresponding pages are gone:

- `https://help.topstep.com/en/articles/14363528-topstepx-commissions-fees` — referenced from `topstep_xfa_parameters.txt:103` as "TopstepX™ — Commissions and Fees". The replacement source is `topstep_xfa_commissions.md` (article 8284213) which has the per-platform tables.
- `https://help.topstep.com/en/articles/9221977-topstepx-micros-and-minis` — referenced from `topstep_scaling_plan.txt:77` as "Click here for full details on how Micros and Minis are calculated on TopstepX." The 10:1 ratio rule itself IS in `topstep_scaling_plan_article.md:78-94`, so the missing article is supplementary.

## Recommended re-scrape cadence

- **Quarterly** (next: 2026-07-08) for routine verification
- **Immediately** after any TopStep blog or email announcement about policy changes
- **Before any code change that touches** `prop_profiles.py`, `prop_firm_policies.py`, `consistency_tracker.py`, or `account_hwm_tracker.py`

## How to use this corpus from code

```python
# In any file that encodes a TopStep fact, annotate the fact with:
#   @canonical-source docs/research-input/topstep/<file>.md
# @scraped 2026-04-08
# @section <section name or line range>
# @verbatim "<quoted text>"

# Example (from trading_app/prop_profiles.py):
class TopStepXFA:
    # @canonical-source docs/research-input/topstep/topstep_mll_article.md
    # @scraped 2026-04-08
    # @section "Express Funded Account" lines 60-66
    # @verbatim "For a $50,000 Express Funded Account, your Maximum Loss Limit
    #            starts at -$2,000 and trails upward as your balance grows.
    #            Once your balance reaches $2,000, the Maximum Loss Limit stays at $0."
    MAX_LOSS_LIMIT_50K = 2000.0
```

A drift check (`pipeline/check_drift.py`) should later verify that every `@canonical-source` annotation in production code points to an existing file in this directory.
