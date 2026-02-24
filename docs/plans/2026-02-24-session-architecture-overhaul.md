# Session Architecture Overhaul — Event-Based Naming

_2026-02-24. Full rebuild of ORB session definitions._

## Problem

Volume analysis of 1-minute bars revealed three issues:

1. **Fixed sessions are wrong in summer.** Sessions `0900`, `1800`, `0030`, `2300` are pinned to Brisbane clock time. When US/UK DST shifts, these sessions compute the ORB from the WRONG 5 minutes — typically 1 hour after the actual market event. The ORB for "NYSE open" at `0030` (Brisbane 00:30) actually captures 1 hour AFTER the open in summer (NYSE opens at Brisbane 23:30 in EDT).

2. **Three major volume events have no ORB session at all.**
   - `COMEX_SETTLE` (1:30 PM ET) — #1 gold volume event (100% of peak). Never tested.
   - `NYSE_CLOSE` (4:00 PM ET) — #1 equity volume event (100% of peak, 2-3x NYSE open). Never tested.
   - `US_DATA_1000` (10:00 AM ET) — #2 gold volume event (89-95% of peak). Partially captured by `US_POST_EQUITY` for equities but not for gold.

3. **Time-based names are confusing.** `0900` means nothing without knowing it's Brisbane time and that it maps to CME Globex reopen. Event names (`CME_REOPEN`) are self-documenting.

## Evidence (Volume Analysis)

Volume spikes from `bars_1m` (median volume, peak per 30-min window, split by US DST):

### MGC (Gold)
| Rank | Event | Winter Brisbane | Summer Brisbane | % of Peak |
|------|-------|----------------|-----------------|-----------|
| 1 | COMEX Settlement 1:30 PM ET | 04:29 | 03:29 | 100% |
| 2 | US 10AM Data | 01:00 | 00:00 | 89-95% |
| 3 | US 8:30 Data (NFP/CPI) | 23:30 | 22:30 | 54-62% |
| 4 | NYSE Open 9:30 AM | 00:30 | 23:30 | 54-56% |
| 5 | Pre-data warmup 8:00 AM | 23:00 | 22:00 | 42-61% |

### MES/MNQ/M2K (Equity Indices)
| Rank | Event | Winter Brisbane | Summer Brisbane | % of Peak |
|------|-------|----------------|-----------------|-----------|
| 1 | NYSE Close 4:00 PM ET | 07:00 | 06:00 | 100% |
| 2 | NYSE Open 9:30 AM | 00:30 | 23:30 | 30-90% |
| 3 | US 10AM Data / Post-Open | 01:00 | 00:00 | 15-54% |
| 4 | CME Pre-Close 2:45 PM CT | 06:45 | 05:45 | 8-25% |

### Volume Jump Analysis (minute-level)
Sharpest volume jumps confirm exact event times:
- COMEX settle: +278% at 13:29 EST (1 min before 1:30 PM settlement)
- NYSE close: +486% at 15:59 EST (1 min before 4:00 PM close)
- NYSE open: +362% at 09:30 EST (exact)
- US 10AM data: +178% at 10:00 EST (exact)
- US 8:30 data: +135% at 08:30 EST (exact)
- CME reopen: +773% at 18:00 EST winter / +4250% at 17:59 EST summer (exact)

## Design

### Final Session List (11 events, all dynamic)

Every session resolves to Brisbane time per-day using the DST-aware resolver pattern already established in `pipeline/dst.py`.

| Session Name | Market Event | Reference Time | Timezone | Instruments |
|---|---|---|---|---|
| `CME_REOPEN` | CME Globex electronic reopen | 5:00 PM CT | US/Chicago | All |
| `TOKYO_OPEN` | Tokyo Stock Exchange open | 9:00 AM JST | Asia/Tokyo | MGC, MES, MNQ |
| `SINGAPORE_OPEN` | SGX/HKEX open | 9:00 AM SGT | Asia/Singapore | MGC, MES, MNQ |
| `LONDON_METALS` | London metals AM session | 8:00 AM London | Europe/London | MGC, MES, MNQ, M2K |
| `US_DATA_830` | US economic data release | 8:30 AM ET | US/Eastern | MGC, MES |
| `NYSE_OPEN` | NYSE cash open | 9:30 AM ET | US/Eastern | All |
| `US_DATA_1000` | ISM/CC release + post-open flow | 10:00 AM ET | US/Eastern | All |
| `COMEX_SETTLE` | COMEX gold settlement | 1:30 PM ET | US/Eastern | MGC |
| `CME_PRECLOSE` | CME equity pre-settlement | 2:45 PM CT | US/Chicago | MES, MNQ, M2K |
| `NYSE_CLOSE` | NYSE closing bell | 4:00 PM ET | US/Eastern | MES, MNQ, M2K |

Notes:
- `TOKYO_OPEN` and `SINGAPORE_OPEN` resolve to fixed Brisbane times (Japan and Singapore have no DST) but are implemented as dynamic for consistency.
- `1130` (HK/SG equity open) is folded into `SINGAPORE_OPEN`.
- `US_DATA_1000` replaces both old `US_POST_EQUITY` (equities) and adds gold coverage.
- ORB duration: 5 minutes for all sessions except `TOKYO_OPEN` which uses 15 minutes (existing behavior from `1000`).

### Session Name Migration Map

| Old Name | New Name | Change Type |
|---|---|---|
| `CME_OPEN` | `CME_REOPEN` | Rename (already dynamic) |
| `0900` | `CME_REOPEN` | Replace fixed with dynamic equivalent |
| `1000` | `TOKYO_OPEN` | Rename + make dynamic (same time, no DST) |
| `1100` | `SINGAPORE_OPEN` | Rename + make dynamic (same time, no DST) |
| `1130` | `SINGAPORE_OPEN` | Merge into SINGAPORE_OPEN |
| `LONDON_OPEN` | `LONDON_METALS` | Rename (already dynamic) |
| `1800` | `LONDON_METALS` | Replace fixed with dynamic equivalent |
| `US_DATA_OPEN` | `US_DATA_830` | Rename (already dynamic) |
| `2300` | `US_DATA_830` | Replace fixed with dynamic equivalent |
| `US_EQUITY_OPEN` | `NYSE_OPEN` | Rename (already dynamic) |
| `0030` | `NYSE_OPEN` | Replace fixed with dynamic equivalent |
| `US_POST_EQUITY` | `US_DATA_1000` | Rename (already dynamic) |
| `CME_CLOSE` | `CME_PRECLOSE` | Rename (already dynamic) |
| — | `COMEX_SETTLE` | **NEW** |
| — | `NYSE_CLOSE` | **NEW** |
| — | `US_DATA_1000` (for MGC) | **NEW** (was equity-only as US_POST_EQUITY) |

### Strategy ID Migration

All strategy IDs embed the session name. Migration example:
```
MGC_0900_E1_RR2.5_CB2_ORB_G4  →  MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G4
MES_0030_E0_RR1.0_CB1_ORB_G5  →  MES_NYSE_OPEN_E0_RR1.0_CB1_ORB_G5
MGC_1000_E1_RR2.0_CB1_ORB_G4  →  MGC_TOKYO_OPEN_E1_RR2.0_CB1_ORB_G4
```

Tables affected: `validated_setups`, `experimental_strategies`, `edge_families`, `orb_outcomes`.

### Fixed → Dynamic Merge Rules

When a fixed session has a dynamic equivalent (e.g., `0900` and `CME_OPEN`):
- **Winter data:** Both sessions compute identical ORBs (same Brisbane time). Merge without data loss.
- **Summer data:** The dynamic session has the CORRECT ORB (aligned to market event). The fixed session's summer ORB is wrong (1 hour late). Use dynamic version.
- **Strategy performance:** Re-run discovery on the merged (dynamic) data. Summer performance should improve because we're now using the correct ORB.

### Implementation Approach: Parallel Run Then Cutover

**Phase 1 — Add new sessions (no breaking changes)**
- Add 3 new resolver functions: `comex_settle_brisbane()`, `nyse_close_brisbane()`, `us_data_1000_brisbane()`
- Add to SESSION_CATALOG
- Add to `build_daily_features.py` (new ORB columns alongside existing)
- Add to `outcome_builder.py` (new session outcomes)
- Run discovery + validation on new sessions only

**Phase 2 — Validate dynamic vs fixed**
- For each pair (0900 vs CME_OPEN, 1800 vs LONDON_OPEN, 0030 vs US_EQUITY_OPEN):
  - Compare summer-only performance
  - If dynamic outperforms fixed in summer: confirms the timing fix matters
  - If equal: the 1-hour offset doesn't affect edge (still rename for clarity)

**Phase 3 — Rename + cutover**
- Create migration script that updates all tables:
  - `orb_outcomes.orb_label`: old → new
  - `validated_setups.orb_label` + `strategy_id`: old → new
  - `experimental_strategies`: old → new
  - `edge_families`: old → new
- Update `daily_features` column names
- Update all config references
- Update PROP_PLAYS.md, TRADING_RULES.md

**Phase 4 — Clean up**
- Remove fixed session entries from SESSION_CATALOG
- Remove old column names from daily_features
- Remove migration map (or keep as historical reference)
- Update REPO_MAP.md

### Risk Mitigation

- **Backup gold.db before ANY migration step**
- **Parallel columns** during Phase 1-2 means nothing breaks
- **Phase 2 validation** proves or disproves the improvement before committing
- **Migration script is idempotent** (re-runnable without data loss)
- **No live trading affected** — paper_trader is not yet wired to these sessions

### Success Criteria

1. All 11 sessions have ORB outcomes in `orb_outcomes`
2. Dynamic sessions produce correct Brisbane times for both DST regimes (verified by volume alignment)
3. Discovery + validation run on all sessions
4. COMEX_SETTLE and NYSE_CLOSE results reported (edge or no-edge)
5. All strategy IDs use event-based names
6. No fixed session entries remain in SESSION_CATALOG
7. PROP_PLAYS.md updated with event-named sessions
8. All tests pass, drift checks pass
