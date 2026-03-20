# Strategy Awareness — Shared Context for All Skills

Loaded into every session. Provides baseline awareness for any skill touching strategy, research, or trading decisions.

## Before Any Strategy/Research Work

1. **Check `docs/STRATEGY_BLUEPRINT.md`** — route to correct section, check NO-GO registry (§5), check "What We Might Be Wrong About" (§10)
2. **Query canonical sources, never cite from memory** — numbers go stale after every rebuild
   - Instruments → `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`
   - Sessions → `from pipeline.dst import SESSION_CATALOG`
   - Cost models → `from pipeline.cost_model import COST_SPECS`
   - Entry models/filters → `from trading_app.config import ENTRY_MODELS, ALL_FILTERS, SKIP_ENTRY_MODELS`
   - Live portfolio → `from trading_app.live_config import LIVE_PORTFOLIO`
   - DB path → `from pipeline.paths import GOLD_DB_PATH`
3. **Check NO-GO registry before proposing anything** — if it's dead, say so immediately. Don't waste time rediscovering dead paths.
4. **Follow the test sequence (Blueprint §3)** — Mechanism → Baseline → Significance → OOS → Adversarial → Replay → Paper Trade
5. **Variable coverage rule** — before declaring ANY dimension dead, test ≥3 values (RR: 1.0/1.5/2.0, aperture: O5/O15/O30, entry: E1/E2, sessions: ALL)

## Key Facts (verified 2026-03-21, re-verify if stale)

- **MNQ E2 is the ONLY positive unfiltered baseline.** MGC/MES need size filters (G4+/G5+).
- **E0 is PURGED. E3 is in SKIP_ENTRY_MODELS.** Only E1 and E2 are active.
- **ML works at RR2.0 O30 per-aperture** (bootstrap verified p≤0.02). ML on portfolio-level negative baselines is DEAD (p=0.35).
- **filter_type must EXACTLY match a key in ALL_FILTERS.** Unknown strings = silent trade drops.
- **2026 holdout is SACRED.** 3 pre-registered strategies only. No "quick checks."
- **System is not proven right — just not-yet-wrong.** Every finding is provisional.

## Hard Lessons (from `hard_lessons.md`)

- Query `build_live_portfolio()` for "what do I trade" — NOT validated_setups
- Verify schema before SQL — column names are NOT what you think
- Grep ALL callers before changing function signatures
- DST ≠ all sessions — Asia sessions never shift
- Never compute session times manually — run `generate_trade_sheet.py`
