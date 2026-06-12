# START_BOT trading-ready flow fixes — Stage file + CHECKPOINT

task: Make START_BOT → dashboard live-launch flow clean/honest/trustworthy (V1–V5). Standalone DEFERRED (north star).
mode: IMPLEMENTATION
owner: claude (opus) — session 2026-06-12

## Scope Lock
- trading_app/live/bot_dashboard.html   (V1 chart vendor, V2 GO-LIVE message)
- trading_app/live/bot_dashboard.py     (V3 dead fallback, V5 clean-shutdown clear)
- trading_app/live/vendor/              (NEW — vendored lightweight-charts)
- tests/test_trading_app/               (companion tests)

## Blast Radius
- bot_dashboard.html — front-end only; chart `<script src>` swap (CDN→local) + GO-LIVE message split. No backend behavior change.
- bot_dashboard.py — V3 touches LIVE-PATH capital routing (`_live_pilot_cli_args` :796, `action_start` :2972/:3001). V5 touches `_lifespan` shutdown (:158-195). CAPITAL → adversarial-audit gate on V3.
- vendor/ static serve — new route or StaticFiles mount; reads local file, no DB/capital.
- Reads: gold.db read-only (bars-recent already works). Writes: none new.

## CHECKPOINT — Stage 0 COMPLETE (empirically grounded, 2026-06-12 ~15:45 Brisbane)

### Repro environment
- Dashboard launched headless on :8137 (`BOT_DASHBOARD_PORT=8137`), connected real TopStepX broker, Playwright-driven. Background launcher pid via task bjsbb10br. STILL RUNNING — kill before done.

### V1 chart — ROOT-CAUSED + PROVEN
- `<script src="https://unpkg.com/lightweight-charts@5.2.0/...">` at bot_dashboard.html:19-23. SRI-pinned (sha384, 196203 bytes, comment :13).
- unpkg reachable RIGHT NOW (HTTP 200, 196203 bytes exact). Chart RENDERS PERFECTLY → screenshot `stage0-chart-with-cdn-reachable.png` (candles + Brisbane axis + price scale + $-677).
- `/api/bars-recent` returns 162 valid OHLCV bars. lib loaded, 7 canvases, 0 console errors.
- VERDICT: code is correct; ONLY failure mode = external CDN unreachable (transient/firewall/offline) → fallback `_showFallback` at :6707-6709 fires → empty chart. Operator's empty-chart episodes = unpkg unreachable at that moment.
- FIX (Stage 1): vendor the 196203-byte file locally (byte-verify vs SRI), serve it, drop CDN. Removes single point of failure on capital-critical cockpit. KEEP fallback notice as defense.

### V2 GO-LIVE message — CONFIRMED in the wild
- Screenshot top-bar: `HOLD TO GO LIVE / BLOCKED: NO TRADEABL…` while EXPRESS #53179846 IS selected.
- Root: bot_dashboard.html:4214-4215 — `!selectedAccountIsTradeable()` → "No tradeable broker account selected — pick one" even when one IS picked (it's just canTrade=false now).
- `selectedAccountIsTradeable()` at :5843 = `selectedAccountId != null && tradeable.some(id match)`.
- FIX (Stage 2): split none-selected vs selected-but-restricted; name account + reason. Behavior unchanged.

### V3 dead fallback — GROUNDED (capital)
- `LIVE_PILOT_ACCOUNT_ID = 23055112` (bot_dashboard.py:85).
- `_live_pilot_cli_args` :796 `account_id if not None else LIVE_PILOT_ACCOUNT_ID` — reachable for signal/demo (no guard), used as zero-arg convenience.
- `action_start` :2972 has LOUD `assert account_id is not None` on `mode=="live"` → so `:3001` ternary `else LIVE_PILOT_ACCOUNT_ID` is DEAD on live path (assert already guarantees not-None).
- FIX (Stage 3): make live path raise loud (not assert — asserts strip under -O); kill the dead `:3001` live ternary; selector = single live-arming source. Preflight #13 still hard-fails on ambiguity. ADVERSARIAL-AUDIT GATE.

### V4 legacy file — NOT a trivial delete (FLAG to operator)
- bot_dashboard_legacy.html (3715 lines, 173258 bytes) served by NOTHING (bot_dashboard.py:3484 loads only bot_dashboard.html).
- BUT guarded by test `test_legacy_html_snapshot_exists` (test_bot_dashboard_signals_recent.py:100-104) asserting it EXISTS + >100KB as a Stage-1 ROLLBACK snapshot.
- DECISION NEEDED: delete file + its test (lose rollback guarantee) OR keep. Surface to operator. Do NOT silently delete.

### V5 clean-shutdown clear — GROUNDED
- Boot cleanup exists (bot_dashboard.py:137-145) + in-request reconcile (:1649-1653), BOTH only clear if heartbeat age > SESSION_DEFINITELY_DEAD_AFTER_S.
- `_lifespan` shutdown (:158-195) cancels SSE + terminates children but NEVER calls clear_state() → clean close leaves bot_state latched → stale flash on quick re-open.
- FIX (Stage 4): clear own state on clean shutdown, guarded so a deliberately-surviving live child is not clobbered. Boot reconcile stays backstop.

### Refresh open-question — RESOLVED = PHANTOM (drop, per plan)
- 420s instrumented window: ZERO fetch() polls, but EventSource EXISTS. Refresh is SSE, not polling — fetch-counter watched wrong channel.
- Real channel: `/api/events/stream` (EventSource) — probed live = HTTP 200 text/event-stream. CONNECTED, 0 errors.
- SSE code (bot_dashboard.html:7114-7292) is ALREADY robustly defensive: onerror→CLOSED→exp-backoff reconnect (cap 30s, :7179-7186); separate watchdog for MDN "backgrounded socket silently stalls w/o onerror" (:7114,:7286-7292); tab-visibility resync (:7245); subscriber cap 4→HTTP 429 (:6424).
- VERDICT: NO refresh bug reproduced. The "Auto-refresh 5s" label is a display string; real path is SSE. Operator's "maybe idk" most-plausible real edge = subscriber cap 4 (5th tab gets 429, never streams). NOT fixing a phantom — flag the 429-cap as a one-line operator note only.

## Baseline (pre-change)
- tests/test_trading_app/test_bot_dashboard_signals_recent.py → 8/8 pass.
- Drift baseline: NOT yet captured (capture before first commit).
- HEAD at session start: 42855935 (main).

## Streamline side-list (user asked — "useless crap")
- bot_dashboard_legacy.html (V4) — dead-served, rollback-only. Candidate for removal post-operator-OK.
- (more to be appended as found — do NOT act without flagging)

## Sequence
S1 chart vendor → S2 message → S4 self-clean (low-risk) → S3 capital (audit-gated, last) → S5 flag legacy.
Verify after each. Drift + targeted tests. Final preflight 15/15 with selection, raises with none.

## CHECKPOINT — Stage 1 COMPLETE (vendor LightweightCharts, 2026-06-12)
- **Committed** (see git log for SHA): vendored byte-identical SRI-pinned build
  (196203 bytes, sha384 `q1KYLSKHgBnW5tWYGGR8+6YV4/iPy31dILoF2I1OD7XiVUvHEp/TaxIQVmB0j3R2` VERIFIED).
- `trading_app/live/vendor/lightweight-charts-5.2.0.standalone.production.js` — NEW.
- `bot_dashboard.py` — `/vendor/lightweight-charts.js` FileResponse route (immutable cache,
  404+log on missing), `FileResponse` import added. `VENDOR_DIR`/`LWC_VENDOR_FILE` consts at :3883.
- `bot_dashboard.html:19` — `<script src="/vendor/lightweight-charts.js" defer>`; unpkg removed
  (only provenance comments retain the word "unpkg"). `_showFallback` defense-in-depth KEPT.
- Tests: `test_bot_dashboard_routes.py` +3 (byte+sha384 pin, immutable cache, HTML-uses-local).
  18/18 routes + 8/8 signals-recent pass. Drift 188 pass.
- Playwright smoke (:8137, real browser): `window.LightweightCharts.createChart` from /vendor (200),
  ZERO unpkg requests, 7 canvases (matches Stage 0 CDN-reachable baseline). Screenshot
  `stage1-chart-local-vendor.png`.
- NEXT = Stage 2 (GO-LIVE message split, bot_dashboard.html:4214-4215 — front-end only, low risk).
