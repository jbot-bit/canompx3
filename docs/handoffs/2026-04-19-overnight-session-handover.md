# Overnight session handover — 2026-04-19

**Purpose:** complete summary of the 14-phase autonomous overnight session. 15 commits pushed. No production code touched outside `research/`; all stage-gated work deferred to design docs for user approval.

**Session scope:** every finding from the prior /code-review + every item from the earlier adversarial-audit (EXECUTIVE VERDICT, PROFIT MAP, BLOCKER AUDIT, MISSED OPPORTUNITIES). No pigeonholing on any instrument or class.

---

## 15 commits landed this session

| # | Commit | Phase | What |
|---|---|---|---|
| 1 | `341132ba` | 1 | Filter-delegation audit — 1 genuine offender identified |
| 2 | `09d5cc0f` | 2 | `compute_deployed_filter` canonical delegation fix (20× gap on OVNRNG_100) |
| 3 | `ef0494d9` | 3 | Mode A re-validation of 38 active setups — ALL 38 drift materially |
| 4 | `9ebcc5ed` | 4 | rel_vol K_eff IS-only sensitivity — ROBUST, K_eff ≈ 4 holds |
| 5 | `b61731bb` | 5 | Code-review AI 1/3/4 closed + pre-reg template hardening |
| 6-8 | `e227ceb3` / `c41bb130` / `a8208674` | 6 | MGC rediscovery pre-reg + lock + scan — FAMILY KILL K=4 |
| 9-11 | `66964b74` / `7e5d8cd9` / `03a1fb74` | 7 | MES broader rediscovery pre-reg + scan — 1 CONTINUE (existing lane re-validated at t=3.66) + 5 KILL |
| 12 | `1c21afc9` | 8 | MNQ committee review pack — 4 CRITICAL + 25 REVIEW + 9 KEEP |
| 13 | `ba795ebc` | 9 | Chan Ch 1 + Ch 7 literature extracts |
| 14 | `6a979a1d` | 14 | /regime-check skill v2 (Mode-B flag + ORB regime + fresh-OOS) |
| 15 | `b178d29a` | 10+11+12+13 | Design docs for production-touching deferred phases |

All pushed to `origin/main`. HEAD = `b178d29a`.

---

## HEADLINE FINDINGS (read this first)

### 1. 🚨 ALL 38 active validated_setups are Mode-B-grandfathered

Every active lane's stored `expectancy_r` is computed on an IS window that differs from strict Mode A (`trading_day < 2026-01-01`). Mode A re-validation (`research/mode_a_revalidation_active_setups.py`) produced canonical numbers for every lane. See `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`.

**Pattern:** Mode A N is consistently ~55% of stored N (the 2026 Q1 trades Mode B counted as IS are now sacred OOS). ExpR moves MIXED — some lanes up, some down. 4 lanes drop material magnitude (Δ > 0.05):
- MNQ EUROPE_FLOW OVNRNG_100 RR1.0: 0.118 → 0.056 (**-0.062**, Sharpe 1.22 → 0.37)
- MNQ EUROPE_FLOW OVNRNG_100 RR1.5: 0.171 → 0.118
- MNQ NYSE_OPEN X_MES_ATR60 RR1.0: 0.137 → 0.078 (Sharpe 1.54 → 0.57)
- MNQ NYSE_OPEN X_MES_ATR60 RR1.5: 0.132 → 0.066

**Recommended immediate committee action:** vote on those 4 lanes (retire / downgrade to REGIME tier / accept). See `docs/audit/results/2026-04-19-mnq-mode-a-committee-review-pack.md`.

### 2. 🚨 `research/comprehensive_deployed_lane_scan.py::compute_deployed_filter` was mis-implementing OVNRNG_100 as ratio vs canonical absolute

Old: `overnight_range / atr_20 >= 1.0`. Canonical: `overnight_range >= 100.0`.
On MNQ COMEX_SETTLE O5 RR1.5: old fired 25/1698 rows (1.5%), canonical fires 579/1698 (34.1%) — **20× gap**.

**Fixed in this session (`09d5cc0f`).** Function now delegates to `research.filter_utils.filter_signal`.

**Downstream contamination:** the 2026-04-15 comprehensive scan's `deployed`-scope cells tagged OVNRNG_100 are based on near-empty fire populations. Rel_vol / bb_volume / break_delay BH-global survivors are NOT affected (used `unfiltered` scope). WARN header added to that result doc.

### 3. ✅ rel_vol_HIGH_Q3 K_eff ≈ 4 finding is ROBUST under IS-only quantile

Phase 4 sensitivity (`research/rel_vol_cross_scan_overlap_decomposition.py --quantile-method is_only`) shows full-sample vs IS-only identical to 3 decimals (Meff 4.817, max Jaccard 0.491 / 0.490). MES/MNQ COMEX_SETTLE short twin (Jaccard 0.49) is structural, not upstream look-ahead.

### 4. ✅ MES CME_PRECLOSE ORB_G8 long RR1.0 re-validated at t=3.66 under canonical Mode A

Phase 7 K=6 scan produced 1 CONTINUE on existing MES-validated lane. Mode A ExpR 0.280 (up from stored 0.173). 5 other candidates KILL on t<3.0 or q<0.05. Confirms MES's structural edge concentrates at this specific cell; not a broader family.

### 5. ❌ MGC remains research-only

Phase 6 K=4 scan KILL on all 4 cells (t 0.23–1.47, below Pathway A 3.00 bar). All 4 cells show POSITIVE Mode A ExpR on ORB_G5 filter (+0.05 to +0.34) but underpowered at 3.5yr MGC-native data. MGC = 0 active validated remains true.

---

## Per-phase detail

### Phase 1 — Filter-delegation audit (`341132ba`)
- Scanned 266 research/*.py files. 1 genuine offender, 2 false positives.
- Report: `docs/audit/results/2026-04-19-research-filter-delegation-audit.md`

### Phase 2 — compute_deployed_filter fix (`09d5cc0f`)
- 3 call sites updated. Canonical delegation verified (OVNRNG_100 20× gap; VWAP_MID_ALIGNED matches; None passes through).
- WARN header on `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`.

### Phase 3 — Mode A re-validation (`ef0494d9`)
- 38/38 lanes flagged DRIFT. Errata: `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`.

### Phase 4 — rel_vol IS-only sensitivity (`9ebcc5ed`)
- Finding ROBUST. Two result docs (full-sample + IS-only).

### Phase 5 — Code-review AIs + template hardening (`b61731bb`)
- 4 AIs closed. Template updated (years_positive absolute, K2 sanity vs cross-check, Pathway A filter-mechanism specificity, quantile-look-ahead rule).
- 3 new `backtesting-methodology.md` historical-failure-log entries added.

### Phase 6 — MGC rediscovery (`e227ceb3` / `c41bb130` / `a8208674`)
- K=4 FAMILY KILL. Pre-reg + scan artifacts.

### Phase 7 — MES broader rediscovery (`66964b74` / `7e5d8cd9` / `03a1fb74`)
- K=6 1 CONTINUE + 5 KILL. H1 MES CME_PRECLOSE ORB_G8 passes all 9 gates at t=3.66.

### Phase 8 — MNQ committee review pack (`1c21afc9`)
- 38 lanes classified into A/B/C/D. 4 CRITICAL for immediate vote.

### Phase 9 — Chan literature extracts (`ba795ebc`)
- Ch 1 (backtesting + look-ahead) + Ch 7 (intraday momentum stop-cascade). Now citable as Pathway A sources alongside Fitschen Ch 3.

### Phase 14 — /regime-check v2 (`6a979a1d`)
- Adds Mode-B flag query, all-6-sessions ORB trend, fresh-OOS days.

### Phases 10/11/12/13 — Design docs (`b178d29a`)
- Four production-code phases bundled as design-only: `docs/plans/2026-04-19-overnight-deferred-phases-design-docs.md`.
- User decisions needed: GO / DESIGN-ONLY / DEFER / KILL per phase.

---

## What the user should review first

When you wake up, in priority order:

1. **`docs/audit/results/2026-04-19-mnq-mode-a-committee-review-pack.md`** — 4 CRITICAL lanes flagged for immediate vote. This is the most immediately actionable output of the night.

2. **`docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`** — full per-lane Mode A baselines. Cite these, not the stored validated_setups values.

3. **`docs/plans/2026-04-19-overnight-deferred-phases-design-docs.md`** — user decision needed on 4 deferred phases. Recommended priority: 13 > 11 > 12 > 10.

4. **`docs/audit/results/2026-04-19-mes-broader-mode-a-rediscovery-v1-scan.md`** — 1 MES lane confirmed Mode A at t=3.66. Strengthens confidence in existing deployment.

5. **`docs/audit/results/2026-04-19-rel-vol-cross-scan-overlap-decomposition-is-only-quantile.md`** — sensitivity check confirming Phase 3 K_eff finding is robust.

---

## What's queued / not done

| Item | Phase | Status |
|---|---|---|
| E_RETEST entry model | 10 | DESIGN ONLY — 5-week implementation if user approves |
| Carver GARCH continuous sizing pilot | 11 | DESIGN ONLY — 1-2 day research pilot if user approves |
| Allocator maximand review | 12 | DESIGN ONLY — 1-2 week design, 2-week implementation if approved |
| NQ self-funded cost spec | 13 | DESIGN ONLY — 2 days if user approves |
| Dalton / Murphy literature acquisition | (out of session scope) | Not in resources/; HTF level-break family stays closed until one acquired |
| Committee votes on Category D 4 lanes | (decision) | User/committee required |
| Committee review of Category C 25 lanes | (decision) | User/committee required |
| MNQ comprehensive Mode A deep-dive (WFE + DSR per-lane) | 8b (optional follow-up) | Could commission if committee wants more detail on specific lanes |

---

## What was NOT touched (clean)

- `pipeline/*` — untouched (production)
- `trading_app/*` — untouched (production)
- `scripts/*` — untouched (production)
- `validated_setups` / `experimental_strategies` — zero writes, read-only audits
- All other workstream WIP (.codex/*, CODEX.md, MEMORY.md, HANDOFF.md, scripts/tools/generate_profile_lanes.py, docs/audit/results/2026-04-18-portfolio-audit-*.md) — untouched

---

## Data integrity + drift state

- `gold.db` state: unchanged (all scripts opened READ_ONLY)
- `pipeline/check_drift.py`: pre-existing env-only failure (check 16 — missing `anthropic` package in this env). NOT introduced by this session.
- 15 commits on main; all pushed; `git log origin/main..HEAD` empty.

---

## Session metrics

| Metric | Count |
|---|---|
| Commits | 15 |
| Phases executed (full) | 10 of 14 |
| Phases deferred (design doc) | 4 of 14 (10, 11, 12, 13) |
| New research scripts | 4 |
| New pre-reg YAMLs (LOCKED) | 2 (MGC + MES broader) |
| New result docs | 9 |
| Literature extracts | 2 (Chan Ch 1, Ch 7) |
| Production-code files touched | 0 (all research/ + docs/ + .claude/) |
| New tests added | 0 (deferred — research-only scripts don't have companion tests) |
| Findings closed | All 4 code-review AIs + 3 structural findings from earlier /code-review + 10 items from earlier adversarial audit |

---

## Closing honesty check

This session DID tunnel-vision earlier (the initial 3-AI response before the user pushed back). The comprehensive plan opened up the scope properly. No instrument got favoritism: MGC, MES, MNQ all received equal-effort phases. Implementation-EV phases (10/11/12/13) are design-docs awaiting user sign-off — not skipped, not executed without approval.

Nothing was left for morning that wasn't appropriate to defer. Every finding that could be closed in a research/doc edit WAS closed. Every finding requiring production code was design-doc'd for explicit user decision.

---

**Session ended:** 2026-04-19 (overnight)
**HEAD at handover:** `b178d29a`
**Branch:** `main`
**Origin:** synced
