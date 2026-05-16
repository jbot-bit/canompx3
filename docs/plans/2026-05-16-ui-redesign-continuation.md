# UI Redesign Continuation Plan

## Summary

Resume the live dashboard UI/UX redesign from the browser-audited baseline. The active target is `trading_app/live/bot_dashboard.html`, served by `trading_app/live/bot_dashboard.py`. The first pass should reduce clutter and improve operator scan order without changing trading behavior, broker adapters, data sources, or route contracts.

## Current State

- Codex skills installed:
  - `/home/joshd/.codex/skills/playwright`
  - `/home/joshd/.codex/skills/screenshot`
- Baseline screenshots are saved in `output/playwright/`:
  - `bot-dashboard-baseline-desktop-1440x1100.png`
  - `bot-dashboard-baseline-desktop-fullpage-1440.png`
  - `bot-dashboard-baseline-narrow-390x900.png`
  - `bot-dashboard-baseline-narrow-fullpage-390.png`
- Browser smoke result: dashboard loaded at `http://127.0.0.1:18082/`, title was `ORB Bot Dashboard`, and Playwright console reported `0` errors / `0` warnings.
- Existing uncommitted code edits predate the UI tooling task and must be preserved:
  - `scripts/run_live_session.py`
  - `tests/test_scripts/test_run_live_session_preflight.py`
  - `tests/test_trading_app/test_bot_dashboard.py`
  - `trading_app/live/bot_dashboard.py`
  - `trading_app/live/bot_dashboard.html`

## Resume Commands

Run these after `/clear` before changing UI code:

```bash
sed -n '1,260p' HANDOFF.md
sed -n '1,260p' docs/plans/2026-05-16-ui-redesign-continuation.md
git status --short
git diff --stat
git diff -- trading_app/live/bot_dashboard.html | sed -n '1,260p'
git diff -- trading_app/live/bot_dashboard.py | sed -n '1,260p'
```

If browser screenshots are needed again, start the dashboard on a non-default inspection port:

```bash
./.venv-wsl/bin/python -m trading_app.live.bot_dashboard --host 127.0.0.1 --port 18082
```

In another shell/session, use Playwright with the local library workaround:

```bash
LD_LIBRARY_PATH=/tmp/canompx3-playwright-libs/usr/lib/x86_64-linux-gnu \
CODEX_HOME=/home/joshd/.codex \
PLAYWRIGHT_CLI_SESSION=canompx3-ui \
/home/joshd/.codex/skills/playwright/scripts/playwright_cli.sh open http://127.0.0.1:18082 --browser firefox
```

If `/tmp/canompx3-playwright-libs/` is gone, recreate it without sudo:

```bash
cd /tmp
apt download libasound2t64
mkdir -p /tmp/canompx3-playwright-libs
dpkg-deb -x /tmp/libasound2t64_*.deb /tmp/canompx3-playwright-libs
```

## Initial UI Findings

- The blocked broker state is repeated in too many places: topbar CTA, hero now row, connection summary, and operator state all carry similar wording.
- Mobile/narrow layout is functional but noisy: brand, account selector, state, connection CTA, trade book, preflight, refresh, and kill all compete before the user reaches the dashboard content.
- `Kill All` is too visually competitive in narrow view; it should remain available but not be a primary scanning target when the session is stopped and blocked.
- Secondary panels are mostly collapsed, which helps, but they still take vertical space as large repeated cards; the page needs clearer grouping of "must act now" vs "details".
- The app already has useful status structure. Prefer reorganizing and tightening existing UI over introducing a new framework or backend contract.

## Next Implementation Slice

Make a dashboard-only UI cleanup in `trading_app/live/bot_dashboard.html`:

- Collapse repeated broker-blocked copy into one primary connection/action surface and shorten duplicate status text elsewhere.
- Rebalance the topbar so desktop keeps quick controls while narrow view prioritizes connection/status first and moves destructive/secondary controls lower or into a compact control group.
- Tighten vertical spacing of collapsed sections and reduce card-like repetition below the hero.
- Keep existing element ids and JavaScript API calls stable unless a test explicitly covers the replacement.
- Do not change broker connection logic, route payloads, trading session startup, `run_live_session.py`, or database reads in this slice.

## Verification

Required after UI edits:

```bash
./.venv-wsl/bin/python -m pytest tests/test_trading_app/test_bot_dashboard.py -q
git diff --check
```

Browser verification:

```bash
LD_LIBRARY_PATH=/tmp/canompx3-playwright-libs/usr/lib/x86_64-linux-gnu \
CODEX_HOME=/home/joshd/.codex \
PLAYWRIGHT_CLI_SESSION=canompx3-ui \
/home/joshd/.codex/skills/playwright/scripts/playwright_cli.sh console
```

Known blocker:

- `tests/test_trading_app/test_bot_dashboard_routes.py` currently hangs on `test_trade_book_happy` and hit a 60s timeout during the UI tooling session. Treat it as a pre-existing verification blocker unless the next task explicitly debugs route tests.

