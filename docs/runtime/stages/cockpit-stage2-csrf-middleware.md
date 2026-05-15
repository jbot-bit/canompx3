---
task: CSRF protection on bot dashboard mutating routes — currently any local-network HTML page can trigger /api/action/kill, /api/action/start, /api/broker/* via cross-origin form POST
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.py
  - tests/test_trading_app/test_bot_dashboard_csrf.py
---

## Blast Radius

- `trading_app/live/bot_dashboard.py` — add Starlette `BaseHTTPMiddleware` subclass (`OriginAllowlistMiddleware`) near top of file with the other helpers; register on FastAPI `app` via `app.add_middleware(...)`. Allowlist construction reads canonical `PORT` (line 38) + optional `DASHBOARD_ALLOWED_ORIGINS` comma-list env. Mutating-method gate inside middleware; no per-route decorator changes required. Net additions: one class (~40 lines) + one `add_middleware` line.
- `tests/test_trading_app/test_bot_dashboard_csrf.py` — NEW pytest file. Allow/block matrix: (1) GET with no Origin passes, (2) POST same-origin (`http://localhost:8080`) passes, (3) POST cross-origin (`http://evil.example`) blocked 403, (4) POST no-Origin no-Referer blocked 403 in prod-like env, (5) POST no-Origin allowed when `PYTEST_CURRENT_TEST` is set (test-runner fallback), (6) Referer-only same-origin POST passes (Origin absent, Referer present and matching).
- Reads: request headers + two env vars (`BOT_DASHBOARD_PORT`, `DASHBOARD_ALLOWED_ORIGINS`). No DB, no filesystem, no network. Writes: none (middleware short-circuits with `Response(status_code=403)` on reject). Routes: zero added.
- Existing 36+ cockpit tests use `TestClient` which omits Origin by default. The `PYTEST_CURRENT_TEST` env-var fallback (pytest sets this automatically per-test) keeps every existing test green without per-test fixture changes. Verified: `grep -rn "PYTEST_CURRENT_TEST" trading_app/ pipeline/` returns zero hits — this introduces a new but standard pytest-aware pattern in production code.
- Cross-fix interaction with Stage 1: zero. Stage 1 changed HoldToKill counting logic in the HTML + a Python helper that reads state. Stage 2 gates the HTTP layer. The two fixes compose — Stage 1 already merged-ish (commits on `feat/cockpit-v4-stage1-killswitch-fix`, not yet PR'd); Stage 2 branches from `origin/main` independently.

## Why

`trading_app/live/bot_dashboard.py` exposes 8 mutating endpoints (verified via `grep "@app.post"`):

- `/api/action/kill` (line 1584) — writes STOP_FILE, terminates session subprocess. **Capital-class fail-open.**
- `/api/action/preflight` (line 1611) — runs preflight pipeline.
- `/api/action/refresh` (line 2123) — kicks off data refresh.
- `/api/action/start` (line 2212) — launches session subprocess. **Capital-class fail-open.**
- `/api/broker/add` (line 1781) — adds broker credentials.
- `/api/broker/remove` (line 1808) — removes broker.
- `/api/broker/toggle` (line 1820) — toggles broker enable.
- `/api/broker/test` (line 1840) — round-trips broker.

None of these check `Origin`, `Referer`, `X-CSRF-Token`, `SameSite` cookies, or any cross-origin gate. Verified: `grep -nE "Origin|origin|Referer|csrf|CSRF|SameSite" trading_app/live/bot_dashboard.py` returns zero hits in request-handler code.

The dashboard binds localhost-only (line 2887 runtime check + drift check `check_dashboard_localhost_only_binding`). That stops LAN attackers but **does not stop CSRF**: any HTML page the operator browses in the same browser (web ad, malicious doc, compromised local-server page) can submit `<form action="http://localhost:8080/api/action/kill" method="POST">`. Browser sends the request from the operator's loopback origin; the dashboard happily accepts it.

This is the same severity tier as Stage 1's kill-modal fail-open: an unauthenticated capital-mutating path. The fix is the standard same-origin gate.

## Fix shape

1. **Middleware class** (new, ~40 lines in `bot_dashboard.py` near the other helpers):

   ```python
   class OriginAllowlistMiddleware(BaseHTTPMiddleware):
       """Gate mutating requests to same-origin only.

       Localhost binding (line 2887) stops LAN attackers; this stops CSRF
       from other browser tabs. Mutating methods (POST/PUT/DELETE/PATCH)
       require Origin or Referer to match the dashboard's own origin.
       GET/HEAD/OPTIONS pass through.
       """

       SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

       def __init__(self, app, *, port: int, extra_origins: tuple[str, ...] = ()):
           super().__init__(app)
           self._allowed = frozenset(
               (
                   f"http://localhost:{port}",
                   f"http://127.0.0.1:{port}",
                   f"http://[::1]:{port}",
                   *extra_origins,
               )
           )

       async def dispatch(self, request, call_next):
           if request.method in self.SAFE_METHODS:
               return await call_next(request)
           origin = request.headers.get("origin")
           if origin is not None:
               if origin in self._allowed:
                   return await call_next(request)
               return Response(status_code=403, content="cross-origin POST blocked")
           referer = request.headers.get("referer")
           if referer is not None:
               # Match by origin prefix (scheme://host:port) — strip path/query.
               for allowed in self._allowed:
                   if referer.startswith(allowed + "/") or referer == allowed:
                       return await call_next(request)
               return Response(status_code=403, content="cross-origin POST blocked")
           # No Origin and no Referer. Pytest TestClient omits both — allow under pytest.
           if os.environ.get("PYTEST_CURRENT_TEST"):
               return await call_next(request)
           return Response(status_code=403, content="missing Origin/Referer on mutating request")
   ```

2. **Allowlist wiring** — single `app.add_middleware(OriginAllowlistMiddleware, port=PORT, extra_origins=_extra_origins_from_env())` call near where `app` is defined. Helper `_extra_origins_from_env()` reads `DASHBOARD_ALLOWED_ORIGINS` (comma-separated), strips whitespace, drops empties.

3. **Tests** — six TestClient cases enumerated in Blast Radius. Each issues a real HTTP request through the middleware stack against `/api/action/kill` (since that's the canonical capital-class route). The kill route writes STOP_FILE — tests must `monkeypatch` STOP_FILE to a tmp path so existing kill semantics are exercised but don't pollute. Pattern: lift the monkeypatch from `test_bot_dashboard_holdtokill.py` (Stage 1's test file).

## Canonical sources

- Port: `trading_app/live/bot_dashboard.py:38` — `PORT = int(os.environ.get("BOT_DASHBOARD_PORT", "8080"))`. Single source of truth; do NOT re-read the env var inside the middleware.
- Localhost-binding policy: `trading_app/live/bot_dashboard.py:2887-2891` (`run_dashboard` host assertion). The middleware's allowlist mirrors this set (`127.0.0.1`, `localhost`, `::1`).
- Mutating-route inventory: derived live via `grep "@app.post" trading_app/live/bot_dashboard.py` — 8 routes as of 2026-05-14. The middleware applies by HTTP method, not by route enumeration, so new POST routes are gated automatically (fail-closed default).

## Acceptance

- `python -m pytest tests/test_trading_app/test_bot_dashboard_csrf.py -v` → all 6 cases pass (exit code 0, output shown).
- All existing cockpit tests still pass: `python -m pytest tests/test_trading_app/test_bot_dashboard_*.py -v` → 36+ green.
- `python pipeline/check_drift.py` → all guardrails pass (no new drift checks added; existing ones unaffected).
- Adversarial-audit dispatched per `.claude/rules/adversarial-audit-gate.md` (capital-class trigger: `trading_app/live/` + judgment-class commit). Audit verdict recorded in this file's stage-close section before the file is deleted/archived.
- Manual sanity: run dashboard locally, browse `http://localhost:8080/`, fire HoldToKill modal — POST succeeds (same-origin). Open `https://example.com` in another tab with a tiny `<form action="http://localhost:8080/api/action/kill" method="POST"><input type="submit"></form>`, click it — POST returns 403. (Operator should perform this; not part of automated suite.)

## Stage close — PENDING

Pre-implementation gate met: scope-lock written before any code edit. Awaiting user confirmation of approach (pytest-aware allowlist) before implementation begins.
