# HTF Path A session handover — 2026-04-19

**Purpose:** pick-up note for a new terminal session. Written after HTF Path A arc closed with FAMILY KILL on both prev-week and prev-month families, MES EUROPE_FLOW shadow locked and running.

**Main HEAD at handover:** `6bc4bbd8`.

---

## What landed this session (main branch)

| Commit | What | Outcome |
|---|---|---|
| `b7ee8037` | `pipeline/build_daily_features.py` + `init_db.py`: 12 canonical HTF fields (`prev_{week,month}_{high,low,open,close,range,mid}`) | Feature surface built, 14,729 rows backfilled across 8 symbols |
| `668d2680` | `check_htf_levels_integrity` (drift #59): 12-field cross-verify with DuckDB `DATE_TRUNC`; phantom + stale-miss detection | Pressure-tested with 4 injected divergences, all caught with correct class labels |
| `93b06712` | HTF Path A prev-week v1: K=24 family scan + pre-reg + fire-rate precheck + recorder + result doc | **FAMILY KILL FK1** — zero cells pass take-direction |
| `a2ee1a41` → `a1332f12` | MES EUROPE_FLOW long skip-rule observational shadow (audit-improved, v3 hardening, LOCKED) | Recorder live; fresh-OOS window 2026-04-18+; ledger empty at draft |
| `daaed1be` | Merge of the HTF branch into main (non-ff) | Seven-commit branch integrated cleanly; no conflicts with parallel VWAP/codex work |
| `3eac853f` | Prev-month v1 draft + fire-rate precheck (all 12 cells 9.6%-39.2%, in-band) | Ready for lock+run |
| `6bc4bbd8` | Prev-month v1 LOCK + run + result doc | **FAMILY KILL FK1** — same structural verdict as prev-week |

HTF research class: **near-closed**. See closure criteria in `docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md` § Closure recommendation.

---

## Adversarial audit finding NOT YET DOCUMENTED in the repo

**Finding:** the prev-month v1 result doc describes the MES EUROPE_FLOW wrong-sign pattern as "replicating" prev-week v1. That framing is **overstated**.

| Lane | prev_week fires | prev_month fires | Both | Overlap % |
|---|---:|---:|---:|---:|
| MES EUROPE_FLOW long | 215 | 344 | 148 | 43% of pm-fires are also pw-fires |
| MES TOKYO_OPEN long | 181 | 309 | 121 | 39% of pm-fires are also pw-fires |

Splitting the prev-month MES EUROPE_FLOW long RR2.0 test by overlap reveals:

| Subset | N | mean | t | raw p |
|---|---:|---:|---:|---:|
| OVERLAP (pm AND pw) | 146 | -0.353 | **-4.018** | 0.0001 |
| NON-OVERLAP (pm-only, NOT pw) | 195 | -0.119 | -1.384 | 0.168 |
| Combined (what the scan ran) | 341 | -0.219 | -3.53 | 0.0005 |

**Implication:** on MES EUROPE_FLOW long, the "replication" is driven by the OVERLAP subset — those are mostly the same days prev-week v1 already flagged. The prev-month-specific contribution (non-overlap) is not significant.

**On MES TOKYO_OPEN long RR2.0**, the opposite pattern holds:

| Subset | N | mean | t | raw p |
|---|---:|---:|---:|---:|
| OVERLAP | 121 | -0.193 | -1.863 | 0.065 |
| NON-OVERLAP | 188 | -0.275 | **-3.367** | **0.0009** |
| Combined | 309 | -0.243 | -3.79 | 0.0002 |

This is a **genuinely new independent observation** — prev-week v1 missed this lane. Not a replication. Not covered by the existing MES EUROPE_FLOW shadow.

**Action for next session (recommended, not executed):**
1. Append an adversarial-audit addendum to `docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md` correcting the replication framing and documenting both the overlap finding and the MES TOKYO_OPEN long non-overlap finding as a standalone observation (NOT a scope extension of the existing shadow).
2. Do NOT expand the existing shadow to MES TOKYO_OPEN. A separate shadow pre-reg (single-lane, peek-penalized, fresh-OOS-only) would be the honest vehicle — OR file it as a research note and move on. The MES EUROPE_FLOW shadow's evidence base is NOT strengthened by prev-month v1.

---

## What NOT to do (per institutional discipline after K=48 null)

- No more HTF-level-break variants (prev_2_weeks, prev_quarter, etc.) without a genuinely new mechanism.
- No soft reopen of prev-week v1 or prev-month v1.
- No scope expansion of `htf-mes-europe-flow-long-skip-rule-shadow` to any other lane, RR, or direction.
- Closure criteria for any future HTF-level-break pre-reg are in the prev-month v1 result doc:
  1. Structurally new mechanism (not liquidity-sweep, not calendar-institutional-rebalancing).
  2. At least one Pathway-A-qualifying literature extract under `docs/institutional/literature/` (Dalton, Murphy, or Chan 2013 Ch 4).
  3. Explicit prior-elevation argument lifting the base rate above 30% genuine.

---

## State across worktrees at handover

```
C:/Users/joshd/canompx3          main                                6bc4bbd8
C:/Users/joshd/canompx3-f5       research/f5-below-pdl-stage1        0bac6083   (KILL, closed)
C:/Users/joshd/canompx3-htf      research/htf-path-a-design          a1332f12   (merged — branch deletable)
C:/Users/joshd/canompx3-phase-d  phase-d-volume-pilot-d0             0b80df5f   (D-1 shadow locked)
```

Uncommitted in main worktree at handover (**NOT mine — other workstream's WIP, leave alone**):
```
 M .codex/COMMANDS.md
 M .codex/STARTUP.md
 M .codex/WORKFLOWS.md
 M .codex/config.toml
 M CODEX.md
 M MEMORY.md
?? docs/prompts/open_architecture_triage_prompt.md
```

---

## Critical staleness to fix before any research resumes

**`orb_outcomes` is 1 day behind `daily_features`.**

| Table | Max MNQ / MES / MGC trading_day |
|---|---|
| bars_1m | 2026-04-17 |
| daily_features | 2026-04-17 |
| **orb_outcomes** | **2026-04-16** |

Per `.claude/rules/quant-audit-protocol.md` Pre-Flight: HALT threshold is >2 days. Currently 1 day behind — not HALT, but one more day tips it over. Trigger a focused `orb_outcomes` rebuild for 2026-04-17 on MNQ/MES/MGC next session before any research-pass runs.

Dead instruments' staleness is expected and not blocking:
- GC: through 2026-04-05 (proxy, less-critical)
- M2K, M6E, MBT, SIL: months behind (dead for ORB)

---

## Active pre-regs in the broader portfolio (reference index)

| File | Status | Notes |
|---|---|---|
| `2026-04-18-htf-path-a-prev-week-v1.yaml` | LOCKED (KILL) | K=24 family, take-direction null |
| `2026-04-18-htf-path-a-prev-month-v1.yaml` | LOCKED (KILL) | K=24 family, take-direction null |
| `2026-04-18-htf-mes-europe-flow-long-skip-rule-shadow.yaml` | LOCKED | Observational shadow, recorder live, fresh OOS empty |
| `2026-04-18-h1-mes-london-metals-signal-only-shadow.yaml` | LOCKED | Other workstream (H1 audit-derived shadow) |
| `2026-04-18-phase-d-d0-rel-vol-sizing-mnq-comex-settle.yaml` | KILL 2026-04-18 | Sharpe uplift +7.3% < 10% threshold; doc-only correction applied |
| `2026-04-18-phase-d-d1-*` (referenced in branch) | LOCKED on phase-d worktree | D-1 signal-only shadow |
| `2026-04-18-mf-futures-mnq-supported-surface.yaml` | active | mf_futures parallel research direction (non-ORB) |
| `2026-04-18-vwap-comprehensive-family-scan.yaml` | DOCTRINE-CLOSED | K1+K2 fired, closed |
| `2026-04-17-garch-a4b-binding-budget.yaml` | — | Garch allocator budget work |
| `2026-04-17-garch-a4c-routing-selectivity.yaml` | — | Garch allocator routing |

---

## Recommended next-session EV triage (highest first)

1. **Pipeline lag fix (15-30 min).** Trigger `orb_outcomes` rebuild for 2026-04-17 on MNQ/MES/MGC so downstream research sees current data.
2. **Live book fitness audit (30 min).** Invoke `/regime-check` skill. 61 active validated_setups; top ExpR lanes include:
   - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` (ExpR 0.215, N=513)
   - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` (ExpR 0.210, N=701)
   - `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` (ExpR 0.196, N=194)
3. **Append prev-month v1 adversarial-audit correction** to the result doc (see §"adversarial audit finding" above).
4. **`mf_futures` module review (30 min).** NEW non-ORB research kernel at `mf_futures/` (kernel.py, carry.py, expiry.py, research.py, snapshot.py, models.py, contracts.py, config.py). Pre-reg `2026-04-18-mf-futures-mnq-supported-surface.yaml` active. Understand direction before it drifts.
5. **Literature extraction (long-lead, 2-4 hrs/paper).** Dalton *Markets in Profile*, Murphy *Technical Analysis*, Chan 2013 Ch 4 — all absent from `docs/institutional/literature/`. Any ONE of these unlocks Pathway A (t ≥ 3.00 with-theory) for ALL future level-based research.

**Do not do:** more HTF-level-break variants. The arc is closed.

---

## Shadow recorder maintenance

`research/shadow_htf_mes_europe_flow_long_skip.py` is idempotent and read-only. Invoke daily (or weekly) to append fresh fires to:

```
docs/audit/shadow_ledgers/htf-mes-europe-flow-long-skip-rule-ledger.md
```

Review conditions (from the YAML):
- `N_fire_days >= 30` (unlikely before 2028-06-30 per `confirmatory_feasibility` block), OR
- calendar cap 2028-06-30, whichever first.

Expected verdict at cap: RULE 3.2 directional-only (not confirmatory) given the structural feasibility analysis. Shadow is honest observational work; do not expect it to unlock deployment without additional data beyond 2028.

---

## One-line session summary

HTF canonical feature surface built and verified (drift #59 green); prev-week v1 + prev-month v1 both FAMILY-KILLED at K=24 on the take-direction; MES EUROPE_FLOW long residual-anomaly shadow LOCKED and recording; adversarial audit caught that the "prev-month replicates prev-week" framing was overstated (43% overlap on MES EUROPE_FLOW; MES TOKYO_OPEN non-overlap is a NEW out-of-scope observation); HTF level-break filter class near-closed pending new mechanism or Dalton/Murphy/Chan Ch 4 literature.
