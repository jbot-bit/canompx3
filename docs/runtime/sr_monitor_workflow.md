# SR-Monitor Operator Workflow

Origin: 2026-05-11. Filling the silent-failure gap where an operator (Claude) misread `sr_state.json` as current state and ran `--apply-pauses` against 3 deployed MNQ lanes that had already recovered.

## When to run which command

| Goal | Command | State mutation |
|---|---|---|
| Diagnostic: see current SR state | `python -m trading_app.sr_monitor` | None — read-only |
| Diagnostic with custom pause window | `python -m trading_app.sr_monitor --pause-days 14` | None — `--apply-pauses` not set |
| **Apply pauses on ALARM lanes** | `python -m trading_app.sr_monitor --apply-pauses` | **Writes `data/state/lane_overrides_<profile>.json`** |

The diagnostic forms (no `--apply-pauses`) only write `sr_state.json`. They are safe to run repeatedly.

## Reading sr_state.json correctly

`sr_state.json` `payload.results[]` carries TWO SR values per lane. Read both:

- `sr_stat` — the SR statistic at the FIRST-CROSSING (peak / trigger) trade. Stops updating after the alarm fires. **This is NOT current health.** Do not infer "the lane is currently in alarm" from this value. Locked by `test_run_monitor_reports_sr_at_alarm_not_at_stream_end`.

- `current_sr_stat` — the SR statistic after walking the FULL monitored stream (post-alarm path). **Use this for "is the lane currently in alarm territory?" questions.**

Two recovery diagnostics:

- `trades_since_alarm` — count of trades AFTER the trigger trade. `None` if no alarm fired.
- `recent_10_mean_r` — mean R of the last 10 trades. Signed.

If `sr_stat >> threshold` but `current_sr_stat << threshold` AND `trades_since_alarm >= 10` AND `recent_10_mean_r > 0`, the lane has recovered. Drift check `check_sr_pauses_have_recent_evidence` (Check #132) surfaces this conjunction as an advisory.

## Overturning a false-positive pause

The canonical override surface is `trading_app/sr_review_registry.py` — a code-backed dict of `SrAlarmReview` entries. Each entry carries an outcome (`watch` or `pause`), a reviewer date, a summary, and a recheck trigger.

`trading_app/lifecycle_state.py:read_lifecycle_state` consults the registry at line 236:

```python
if pause_info is not None:
    blocked = True
    ...
elif sr_status == "ALARM" and sr_review is not None and sr_review.outcome == "watch":
    blocked = False
```

**Critical:** `pause_info is not None` short-circuits BEFORE the registry check. An active `lane_overrides_*.json` entry will shadow the registry's `watch` decision.

**To overturn a false-positive pause:**

1. **Register the watch outcome** — edit `trading_app/sr_review_registry.py`, adding an `SrAlarmReview` entry with `outcome="watch"`. Include a summary citing the canonical figures (WFE, OOS/IS ratio) verified against `validated_setups`. Include a `recheck_trigger` (e.g., "Re-check after N>=100 monitored trades. Retire if SR remains ALARM and WFE < 0.50 or OOS/IS ratio < 0.40").

2. **Remove the shadowing pause entry** — delete the strategy_id key from `data/state/lane_overrides_<profile>.json`. The registry's `watch` outcome will then take effect.

3. **Verify** —
   ```bash
   python -c "from trading_app.lifecycle_state import read_lifecycle_state; \
              import json; \
              print(json.dumps(read_lifecycle_state('<profile>'), indent=2, default=str))"
   ```
   The strategy should appear with `blocked: false`, `sr_review_outcome: watch`, `sr_status: ALARM`.

## Why the separation of concerns is intentional

`scripts/tools/refresh_control_state.py:65` hardcodes `apply_pauses=False`. This is intentional: the refresh task should never autonomously pause lanes. SR-monitor pauses are state mutations that affect deployed-lane risk and require operator awareness.

The intentional gap is between (a) "refresh SR diagnostics" and (b) "act on SR alarms." Closing it with cron would push pause decisions into the background. The right place to act is the operator-side workflow documented here.

## Operator checklist for a fresh SR ALARM

When you see an SR alarm:

1. **Run the diagnostic, don't apply pauses yet.**
   ```bash
   python -m trading_app.sr_monitor
   ```

2. **Read all four fields per alarmed lane:** `sr_stat`, `current_sr_stat`, `trades_since_alarm`, `recent_10_mean_r`.

3. **Check whether the lane is in the registry.**
   ```python
   from trading_app.sr_review_registry import get_sr_alarm_review
   review = get_sr_alarm_review("<profile_id>", "<strategy_id>")
   ```
   If `review.outcome == "watch"`, the alarm has already been adjudicated — don't pause again.

4. **Decision tree:**

   - `current_sr_stat < threshold * 0.5` AND `trades_since_alarm >= 10` AND `recent_10_mean_r > 0` → **recovered.** Do not pause. If not already in registry as `watch`, add an entry.
   - `current_sr_stat >= threshold` (still above the alarm line) → **active alarm.** Verify against canonical figures (WFE, OOS/IS). Decide watch vs pause based on review floors.
   - Recent performance ambiguous (mixed R, partial recovery) → **conservative pause** with explicit recheck trigger. Always register a registry entry so the next operator has context.

5. **If you decide to pause:** use `python -m trading_app.sr_monitor --apply-pauses` (or `python -m trading_app.lane_ctl pause`). Always register the corresponding registry entry with `outcome="pause"` and a summary explaining the canonical-figure basis.

6. **Non-independence caveat:** recovery evidence from `canonical_forward` is the same data stream the SR test uses to assess persistence. A `watch` decision over `pause` on same-stream evidence is NON-INDEPENDENT and should be documented as such in the registry `summary`.

## Related canonical sources

- `trading_app/sr_review_registry.py` — the override registry
- `trading_app/sr_monitor.py` — diagnostic + pause writer (CLI)
- `trading_app/live/sr_monitor.py` — `ShiryaevRobertsMonitor` implementation
- `trading_app/lifecycle_state.py` — `read_lifecycle_state` decision logic
- `pipeline/check_drift.py` — `check_sr_pauses_have_recent_evidence` (Check #132)
- `docs/institutional/literature/` — Pepelyshev-Polunchenko 2015 (SR theory)
