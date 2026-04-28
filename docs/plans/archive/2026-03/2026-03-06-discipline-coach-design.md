---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Discipline Coach — Design Document

**Date**: 2026-03-06
**Status**: Approved
**Scope**: MVP — Post-trade debrief, cooling period, pre-session priming

## Problem Statement

The gap between knowing the trading plan and executing it. Losses from
discipline failures (ignoring signals, adding narrative, chasing losses) are
more expensive than strategy failures. Traditional journaling doesn't work
because it requires effortful reflection in a low-motivation window.

The app has a structural advantage: it already knows the plan (validated
strategies), generates real-time signals, and can detect deviation automatically.
The Discipline Coach uses this to provide frictionless in-moment capture,
behavioral nudges, and pattern surfacing — all within the Streamlit UI layer.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Platform | Streamlit UI layer | Reads orchestrator signals, no execution logic changes |
| Storage | JSONL append-only files | Simple, no schema migration, matches existing pattern |
| Cooling enforcement | Configurable (hard/soft) | User chose this; calm-self picks hard, setting persists |
| Voice memo | Deferred to V2 | Keep MVP focused on tap-based capture |
| Execution modes | Both signal-only and auto | Feature works in both; signal-only has UI-only enforcement |

## Data Model

### trade_debriefs.jsonl

One record per trade exit, written by Streamlit UI after user interaction.

```json
{
  "ts": "2026-03-06T23:15:00Z",
  "trading_day": "2026-03-06",
  "instrument": "MGC",
  "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
  "signal_price": 3248.30,
  "signal_ts": "2026-03-06T23:00:05Z",
  "actual_entry_price": 3252.10,
  "actual_entry_ts": "2026-03-06T23:18:30Z",
  "exit_price": 3245.50,
  "pnl_r": -1.2,
  "pnl_dollars": -370.0,
  "adherence": "overrode",
  "deviation_trigger": "narrative",
  "emotional_temp": 0.85,
  "deviation_cost_r": 3.0,
  "deviation_cost_dollars": 495.0,
  "notes": null,
  "letter_to_future_self": "The narrative is always wrong. The signal is the plan."
}
```

**Enums:**
- `adherence`: `followed | modified | overrode | off_plan`
- `deviation_trigger`: `chart_pattern | narrative | felt_reversal | chasing_loss | fomo_late | sized_up | other` (null when followed)

### discipline_state.jsonl

Events from behavioral monitoring: cooling activations, commitments, session scores.

```json
{
  "ts": "2026-03-06T23:15:05Z",
  "event": "cooling_triggered",
  "trading_day": "2026-03-06",
  "tilt_score": 65,
  "consecutive_losses": 2,
  "session_pnl_r": -1.2,
  "cooldown_seconds": 90
}
```

Event types: `cooling_triggered`, `cooling_overridden`, `commitment`, `session_process_score`

## Component 1: Post-Trade Debrief Card

### Trigger

When a `SIGNAL_EXIT` or `ORDER_EXIT` record appears in `live_signals.jsonl`
that has no matching debrief in `trade_debriefs.jsonl`.

### UX Flow

**Layer 1 — Auto-populated (zero effort):**
Trade result, signal comparison, entry delta, deviation cost. All computed
from `live_signals.jsonl` entries (matching ENTRY/EXIT pairs) and
`orb_outcomes` counterfactual lookup.

**Layer 2 — Adherence classification (one tap):**
Four radio buttons: Followed / Modified / Overrode / Off-plan.
App pre-selects a guess based on signal-vs-execution comparison (price delta,
timing delta).

**Layer 3 — Deviation trigger (one tap, conditional):**
Only shown when adherence != followed. Six pre-built options matching
common failure modes. "Other" with optional free text.

**Layer 4 — Emotional temperature (one drag):**
Streamlit slider, 0.0 (calm) to 1.0 (hot). Default 0.5.
Behavioral benefit: affect labeling (Lieberman et al., 2007) — naming the
emotional state dampens amygdala activation.

**Layer 5 — Letter to future self (optional, conditional):**
Only prompted when emotional_temp > 0.7 AND adherence != followed.
Free text: "What do you want to remind yourself next time?"
Stored and surfaced in pre-session priming.

**Total interaction: 3-4 taps + optional slider, under 15 seconds.**

### Implementation

- `ui/discipline.py`: `render_pending_debriefs()` — called from `copilot.py`
  after `_render_signal_log()`.
- Uses `st.form()` with unique key per trade to prevent double-submission.
- Reads `live_signals.jsonl` for exit events, checks `trade_debriefs.jsonl`
  for existing debrief records (keyed by strategy_id + exit timestamp).

## Component 2: Cooling Period

### Trigger

After any trade exit where `pnl_r < 0`.

### Behavior

| Mode | Duration | Override |
|------|----------|---------|
| Hard | 90s, non-dismissable | Entry button grayed out, no signal card |
| Soft | 90s, dismissable after 15s | Warning banner, signal card dimmed, override button |

### Content During Cooling

- Session P&L (running total from `live_signals.jsonl`)
- Trades today vs plan count
- Next planned action ("wait for signal")
- Rotating quote from curated trading wisdom set
- Countdown progress bar

### Implementation

- State tracked in `st.session_state["cooling_until"]` (UTC timestamp).
- `discipline.check_cooling()` called before rendering signal cards.
- When cooling active: hard mode replaces signal log with cooling screen
  via `st.empty()`; soft mode renders overlay banner.
- Mode stored in `st.session_state["cooling_mode"]`, default "hard".
- Settings toggle in sidebar.
- On expiry, logs `cooling_completed` to `discipline_state.jsonl`.
- On override (soft mode only), logs `cooling_overridden` with remaining seconds.

### Limitations

UI-only enforcement. User can trade directly on Tradovate/TradingView to
bypass. Documented and accepted — this is a self-discipline aid, not a
compliance control. Institutional desks use the same pattern (override-with-documentation).

## Component 3: Pre-Session Priming

### Trigger

When `AppState == APPROACHING` or `ALERT` (session starting in <15 minutes).

### Content

**Pattern stats (computed from trade_debriefs.jsonl):**
- Adherence rate: X of Y signals followed (last N sessions for this session type)
- Average R by adherence type: followed vs modified/overrode
- Deviation cost: cumulative dollars lost to not following signals (this month)

**Today's plan (from portfolio strategies):**
- Session, entry model, ORB filter, RR target
- Position size
- Action rule: "Execute within 60s of signal"

**Commitment button:**
"I commit to following the plan" — logs `commitment` event to
`discipline_state.jsonl`. Used for accountability tracking.

**Letter from past self (conditional):**
If any debrief for this session type has a `letter_to_future_self`, display
the most recent one with its date and context.

### Implementation

- `ui/discipline.py`: `render_pre_session_priming()` — called from
  `copilot.py` in `_render_approaching()` / `_render_alert()`.
- Pattern stats computed by `ui/discipline_data.py` from JSONL aggregation.
- Plan details from `build_live_portfolio()` (already loaded in copilot).

## Module Structure

```
ui/
  discipline.py          # UI components: debrief card, cooling, priming
  discipline_data.py     # JSONL I/O, pattern computation, process score

data/                    # Created at runtime
  trade_debriefs.jsonl   # Append-only debrief records
  discipline_state.jsonl # Cooling, commitment, process score events
```

### Integration Points

| Existing file | Change |
|---------------|--------|
| `ui/copilot.py` | Import `discipline.py`, call `render_pending_debriefs()` after signal log, call `render_pre_session_priming()` before briefing cards, wrap signal cards with `check_cooling()` |
| No orchestrator changes | Behavioral layer reads `live_signals.jsonl` (already written), writes to own files |
| No execution engine changes | Pure UI-layer addition |

## V2 Features (Deferred)

- Voice memo capture (10s recording + LLM transcription/tagging)
- Weekly pattern report (email or in-app Sunday summary)
- Process Score dashboard (0-100 daily score, trend line)
- Behavioral dashboard (deviation triggers ranked, session heatmap)
- Accountability partner mode (daily snapshot to trusted contact)
- Pattern-triggered "Letter to Future Me" surfacing
- Tilt score calibration from user's own data

## Behavioral Science References

- **Affect labeling**: Lieberman et al. (2007) — naming emotions dampens amygdala
- **Implementation intentions**: Gollwitzer (1999) — "I will do X when Y" increases follow-through 2-3x
- **Emotional refractory period**: Ekman (2003) — initial emotion spike lasts 60-90s
- **Self-Determination Theory**: Deci & Ryan — autonomy-supportive nudges > hard blocks
- **Prospect theory**: Kahneman & Tversky — loss domain increases risk-seeking behavior
- **Self-compassion**: Neff — self-criticism after mistakes increases future mistakes
- **Temporal self-comparison**: comparing to own best > comparing to others
