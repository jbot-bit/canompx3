# PROMPT: Manual Trading Playbook Audit & Rebuild

**Objective:** Audit the current manual trading playbook against actual deployed lanes, validated edge families, and prop firm constraints. Rebuild it as a clean execution document — not a research doc. Then define the automation strategy selection criteria.

**Mode:** AUDIT + DESIGN. Read everything first. Present findings and proposed playbook structure. Do NOT edit production code. Do NOT change trading logic.

---

## PHASE 1: Ground Truth — What Are We Actually Trading?

### 1A: Query Current State
Run these against gold.db (MCP tools preferred, raw SQL if needed):

1. Read `trading_app/prop_profiles.py` — extract ALL account profiles, their status (ACTIVE/INACTIVE), instruments, lanes, and constraints.
2. Read `docs/plans/manual-trading-playbook.md` — extract all sessions, filters, and parameters listed.
3. Compare: are the playbook sessions identical to prop_profiles.py lanes? Report every mismatch (session in playbook but not in prop_profiles, or vice versa).
4. Query validated_setups for the EXACT strategies currently deployed:
```sql
-- For each lane in prop_profiles.py, find its validated_setups row
SELECT strategy_id, instrument, orb_label, entry_model, orb_minutes,
       filter_type, rr_target, confirm_bars, direction, sample_size,
       win_rate, expectancy_r, sharpe_ann, max_drawdown_r,
       family_hash, wfe, fdr_significant, p_value
FROM validated_setups
WHERE instrument = 'MNQ'
  AND entry_model = 'E2'
  -- Run once per deployed lane with matching params
ORDER BY orb_label, rr_target
```
5. For each deployed lane, also pull the edge family stats:
```sql
SELECT ef.family_hash, ef.robustness_status, ef.trade_tier,
       ef.member_count, ef.median_expectancy_r, ef.avg_sharpe_ann,
       ef.cv_expectancy, ef.pbo, ef.min_member_trades
FROM edge_families ef
WHERE ef.family_hash IN (
  -- hashes from the deployed strategies above
)
```

### 1B: Fitness Check on Deployed Lanes
Use MCP `get_strategy_fitness()` for each deployed strategy_id. Report:
- Current fitness status (FIT/WATCH/DECAY/STALE)
- Rolling 18-month ExpR
- Recent 30-trade Sharpe
- Any lane currently in DECAY or WATCH

**Critical finding from HANDOFF.md (2026-03-30):** NYSE_OPEN is flagged MONITOR/DECAY — 2026 forward is -0.26R regardless of filter. This lane may need to come OUT of the playbook. Verify with fresh data.

### 1C: What's Validated But NOT Deployed?
```sql
-- All ROBUST/WHITELISTED family heads NOT matching any deployed lane
SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.entry_model,
       vs.orb_minutes, vs.filter_type, vs.rr_target, vs.direction,
       vs.sample_size, vs.expectancy_r, vs.sharpe_ann,
       ef.robustness_status, ef.trade_tier, ef.member_count, ef.pbo
FROM validated_setups vs
JOIN edge_families ef ON vs.family_hash = ef.family_hash
WHERE vs.is_family_head = TRUE
  AND ef.robustness_status IN ('ROBUST', 'WHITELISTED')
  AND ef.trade_tier = 'CORE'
ORDER BY vs.expectancy_r DESC
```
Compare this list against deployed lanes. Identify the TOP 5 highest-ExpR validated CORE families that are NOT currently being traded anywhere.

---

## PHASE 2: Playbook Audit — Does the Doc Match Reality?

### Check each of these:
- [ ] **Session times:** Are Brisbane times in the playbook correct? Cross-reference `pipeline.dst.SESSION_CATALOG` for current event-based times (these change with DST).
- [ ] **Filter gates:** Does the playbook specify the exact filter_type per lane? (e.g., ORB_G8 vs ORB_G6 vs ORB_G4). If it says "G8" but prop_profiles.py says "VOL_RV12_N20", that's a mismatch.
- [ ] **RR targets:** Playbook vs prop_profiles vs validated_setups — all three must agree.
- [ ] **Direction constraints:** TRADING_RULES.md says TOKYO_OPEN is LONG-ONLY. Is that enforced in the playbook? In prop_profiles?
- [ ] **Kill criteria:** Does each lane have a defined exit rule? (e.g., "remove if 3 consecutive losing months" or "remove if forward ExpR < 0.10"). If not, flag as MISSING.
- [ ] **DD budget:** Total historical max DD across all lanes vs prop firm DD limit. Report the gap. If historical DD > 80% of firm limit, flag as HIGH RISK.
- [ ] **Pre-session checklist:** Is there a concrete go/no-go checklist per session? (ORB formed? Filter passes? Account funded? No conflicting positions?)

### Report format for each lane:
```
LANE: [instrument] [session] [entry_model] [RR] [filter]
STATUS: ACTIVE / INACTIVE / CONDITIONAL
PLAYBOOK: Listed / Missing / Wrong params
PROP_PROFILES: Listed / Missing / Wrong params
VALIDATED_SETUPS: strategy_id=X, N=Y, ExpR=Z, Sharpe=W
FITNESS: FIT / WATCH / DECAY / STALE (as of [date])
KILL CRITERION: [defined / missing]
NOTES: [any discrepancy or concern]
```

---

## PHASE 3: Strategy Selection Framework — Manual vs Automated

### 3A: Decision Matrix

For each validated CORE/REGIME family head, score on these dimensions:

| Dimension | Manual Score | Auto Score | Notes |
|-----------|-------------|------------|-------|
| **Session time** | +2 if waking hours (Brisbane 07:00-22:00), -2 if sleep hours | Irrelevant (bot runs 24/7) | Manual trader can't trade 02:00 sessions |
| **Execution simplicity** | +1 if E2 (stop entry), -1 if E1 (limit at edge) | Both fine | E1 needs precise limit placement |
| **RR target** | +1 if RR1.0-1.5, -1 if RR3.0+ (needs patience) | Both fine | High RR = long hold = psychological pressure |
| **Filter complexity** | +1 if single filter, -1 if composite | Both fine | Manual trader must check filter pre-trade |
| **Prop firm** | Apex = manual only, Tradeify/TopStep = auto allowed | Match to firm | Hard constraint |
| **DD contribution** | Score based on remaining DD budget | Same | Binding constraint for prop accounts |
| **Correlation** | -2 if same session as existing lane | -1 (less impact at scale) | Avoid doubling down on same risk event |

### 3B: Manual Playbook Sizing Rules

Based on literature (Carver, Davey) and prop firm constraints:

1. **Manual ceiling:** 6-8 non-overlapping session lanes maximum. Beyond this, execution quality degrades.
2. **DD budget rule:** Total historical max DD across all manual lanes must be < 70% of prop firm DD limit (leaving 30% buffer for regime change).
3. **Session spread:** No more than 2 lanes in the same 2-hour window (can't execute both cleanly).
4. **Kill discipline:** Every lane gets a pre-defined kill criterion BEFORE going live. No exceptions.
5. **Review cadence:** Monthly fitness check (use `get_strategy_fitness`). Remove any DECAY lane immediately. WATCH lanes get 60 days before removal.

### 3C: Automation Strategy Selection

For Tradeify/TopStep automated accounts:

1. **All CORE family heads with WFE > 0.50 are candidates** — automation removes the cognitive load ceiling.
2. **Prioritize by:** ExpR × sqrt(sample_size) (balances edge quality with statistical confidence).
3. **Correlation screen:** Run pairwise trade-day overlap between candidate strategies. If two strategies share > 60% of trade days, keep only the higher-ExpR one.
4. **DD stacking:** Monte Carlo the combined DD of the automated portfolio (10,000 sims). If P95 DD exceeds prop firm limit, reduce lanes.
5. **Per-account limit:** Tradeify microscalp rule (50% of trades > 10 seconds) constrains which strategies work. Any ORB strategy with sub-10-second holds (unlikely for your system, but verify) is disqualified.

---

## PHASE 4: Rebuild the Playbook

### Deliverable: Updated `docs/plans/manual-trading-playbook.md`

Structure:

```markdown
# Manual Trading Playbook v5
Last updated: [date]
Source of truth: This doc for manual execution. prop_profiles.py for account config.

## Active Lanes

### Lane 1: [INSTRUMENT] [SESSION] [ENTRY] [RR] [FILTER]
- **Account:** [which prop firm account]
- **Session time:** [Brisbane time, noting DST shift dates]
- **Entry:** [exact entry rule — e.g., "Stop buy 1 tick above ORB high on CB1 confirmation"]
- **Stop:** [exact stop rule — e.g., "0.75x ORB range below entry"]
- **Target:** [exact target rule — e.g., "RR1.0 = 1x risk above entry"]
- **Filter gate:** [exact filter — e.g., "ORB size >= 6 points (G6)"]
- **Direction:** [LONG only / SHORT only / BOTH]
- **Stats (as of [date]):**
  - N trades: [X]  |  Win rate: [X%]  |  ExpR: [X.XXR]
  - Sharpe (ann): [X.XX]  |  Max DD: [X.XXR]  |  WFE: [X.XX]
  - Family: [hash] ([ROBUST/WHITELISTED], [N] members, PBO=[X.XX])
  - Fitness: [FIT/WATCH/DECAY] (rolling 18mo ExpR: [X.XXR])
- **Kill criterion:** [specific, measurable — e.g., "Remove if rolling 18mo ExpR < 0 for 2 consecutive months"]
- **Notes:** [any session-specific quirks, DST behavior, etc.]

[Repeat for each lane]

## Watchlist (Validated, Not Yet Deployed)
[Top 5 CORE families not in active lanes, with stats, ranked by suitability for manual trading]

## Automation Candidates
[Strategies scored for Tradeify/TopStep deployment, with selection rationale]

## Pre-Session Checklist
- [ ] Account funded and within DD limits
- [ ] ORB fully formed (all bars in aperture window present)
- [ ] Filter gate passes (check daily_features or pre-trade calculator)
- [ ] Stop and target pre-calculated
- [ ] No conflicting open position in same instrument
- [ ] Platform connected, order entry tested

## Kill Rules (Global)
- 3 losing months in a row on any lane → remove lane, investigate
- Forward ExpR < 0.10 over 50+ trades → remove lane
- Prop firm rule violation → full stop, all lanes
- DD within 20% of firm limit → reduce to 1 lane until recovered

## Monthly Review Protocol
1. Run `get_strategy_fitness` for all deployed strategy_ids
2. Check each lane against kill criteria
3. Review watchlist — any WATCH→FIT promotions?
4. Update stats in this doc
5. Log review in HANDOFF.md
```

---

## PHASE 5: Verify & Cross-Check

1. **Every stat in the rebuilt playbook must come from a query run during this session.** No stats from memory, docs, or HANDOFF.md. Fresh queries only.
2. **Cross-check prop_profiles.py** — every active lane in the playbook must exist in prop_profiles. Every active lane in prop_profiles must exist in the playbook.
3. **Cross-check TRADING_RULES.md** — any session-specific constraints (LONG-ONLY, DST splits, etc.) must be reflected in the playbook lane notes.
4. **DD budget math:** Sum historical max DD across all active lanes. Compare to prop firm DD limit. Report the ratio. If > 70%, flag which lane to cut.
5. **Session time conflicts:** Map all active lanes on a Brisbane-time timeline. Flag any two lanes within 30 minutes of each other.

---

## OUTPUT

Two deliverables:

1. **Audit findings report** — what's wrong with the current playbook, what's mismatched, what's missing. Written to `docs/audits/playbook_audit_YYYY-MM-DD.md`.
2. **Rebuilt playbook** — clean, execution-focused, with fresh stats from queries. DO NOT overwrite the existing playbook until the user confirms. Write draft to `docs/plans/manual-trading-playbook-v5-DRAFT.md`.

**Important notes:**
- Recent project changes may have shifted lane configurations, fitness statuses, or validated counts. Trust the DB and code over any doc or memory file.
- The system uses event-based session times (not fixed times) — session windows shift with DST. Get current times from `pipeline.dst.SESSION_CATALOG`.
- NYSE_OPEN was flagged DECAY in HANDOFF.md as of 2026-03-30. Verify whether it should remain in the playbook or move to watchlist.
- `live_config.py` is DEAD CODE — do not reference it. Authority is `prop_profiles.py`.
