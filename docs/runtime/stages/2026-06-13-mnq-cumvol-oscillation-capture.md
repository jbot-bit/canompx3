# Stage: Capture raw GatewayQuote to diagnose MNQ cumulative-volume oscillation

task: The live MNQ ProjectX feed oscillates GatewayQuote.volume between the true cumulative session total (~1.28M) and a `1` sentinel on consecutive quotes. _cum_to_delta misreads the `1` as a session reset (delta<0 branch), re-baselines, then the next real reading produces a giant delta that bar_aggregator rejects as > _MAX_TICK_VOLUME. Result: hundreds of paired warnings/sec, emitted 0-volume ticks, and understated live bar volume (capital risk: volume-filtered Tokyo ORB_VOL_4K lanes read corrupted live volume). Before patching the delta logic, add temporary rate-limited logging of the FULL raw quote dict on reset, so we learn what the `1` actually is (lastSize vs a real volume field) and write the correct fix from evidence, not inference.
mode: IMPLEMENTATION

## Scope Lock
- trading_app/live/projectx/data_feed.py

## Blast Radius
- trading_app/live/projectx/data_feed.py: DIAGNOSTIC-ONLY change. _cum_to_delta sets a _last_reset_fired flag (under existing _cum_volume_lock) when it re-baselines on a negative delta. A shared _maybe_log_raw_quote_on_reset helper, called from BOTH _on_quote (async) and _on_quote_sync (foreign signalr thread), logs the full raw quote dict rate-limited to ~1/10s (visible at INFO — the live bot runs INFO per run_live_session.py:46). ZERO behavior change to delta math, the _last_cum_volume baseline, the bar_aggregator cap, or the order/bracket path. Fully reversible (revert one commit).
- Reads: live GatewayQuote dicts (in-memory). Writes: log only. No DB, no schema, no order path.
- Both call sites edited to avoid the canonical inline-copy parity trap (feedback_canonical_inline_copy_parity_bug_class.md) — a single shared helper, not two divergent inline blocks.

## DONE CRITERIA
- _cum_to_delta flags reset; shared helper logs raw quote on reset, rate-limited, visible at INFO.
- Both call sites route through the one helper (no parity drift).
- check_drift.py passes.
- Targeted test: reset path sets the flag; non-reset (normal climb) and first-quote baseline do NOT.
- Operator relaunches START_BOT during MNQ hours; pastes 2-3 raw-quote samples; THEN a follow-up stage writes the real _cum_to_delta fix from evidence.

## STATUS: IMPLEMENTATION in progress
