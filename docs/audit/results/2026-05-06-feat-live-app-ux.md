# Code-Review Verdict — feat/live-app-ux

**Date:** 2026-05-06
**Branch:** `feat/live-app-ux`
**HEAD reviewed:** `d999facc` (UX commit) + post-Task-1 test additions
**Reviewer skill:** `code-review` (per `capital-review` route table — UX-only diff, normal `/code-review + /verify` path)
**Files reviewed:**
- `trading_app/live/bot_dashboard.py` (+163 lines, two new GET routes)
- `trading_app/live/bot_dashboard.html` (+361/-131; mostly whitespace, real adds: ~120 lines)
- `scripts/run_live_session.py` (+18, preflight +1 check)
- `tests/test_trading_app/test_bot_dashboard_routes.py` (new file, 6 tests)

## Scope / Question

Does the `feat/live-app-ux` Stage A diff (Trade Book panel + paused-lane badge + preflight TradeJournal health check) ship safely? Specifically: does it preserve canonical-source delegation, fail closed, avoid look-ahead/research-truth surfaces, and ship with non-vacuous companion tests? UX surface only — no order-route, kill-switch, allocator, or canonical-source change. Out of scope: Stage B (one-day signal-only run) and Stage C (LIVE flip) are operational, gated separately.

## Reproduction / Outputs

Reviewer commands run from `C:/Users/joshd/canompx3` on `feat/live-app-ux`:

- `git log origin/main..HEAD --oneline` → 3 commits + 1 fix commit (this PR's HEAD).
- `pytest tests/test_trading_app/test_bot_dashboard_routes.py -v` → 6/6 passed in 0.51s.
- `pytest tests/test_trading_app/ -k bot_dashboard -q` → 25/25 passed in ~2.3s.
- `python pipeline/check_drift.py` → 119 checks PASS, 0 skipped, 19 advisory.
- File-level greps cited inline (Section B/D/E rows).
- The two newly-added routes were exercised against in-memory DuckDB fixtures; no live-DB call performed.

---

## Grade: B+

One MEDIUM convention-drift finding. Zero seven-sin violations. Zero canonical-source violations. Zero CRITICAL/HIGH issues. Tests cover both routes' happy/empty/error paths. Preflight check additive and isolated. Production-readiness solid.

---

## Section A — Seven Sins: PASS

UX-only diff. Touches no research/discovery/strategy code paths. No look-ahead surface, no FDR exposure, no overfitting risk, no transaction-cost claim. N/A across the board. **0 sins.**

---

## Section B — Canonical Integrity: PASS

| Check | Status | Evidence |
|---|---|---|
| Instruments from `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` | N/A | UX diff doesn't enumerate instruments |
| Sessions from `pipeline.dst.SESSION_CATALOG` | N/A | UX diff doesn't resolve sessions |
| Costs from `pipeline.cost_model.COST_SPECS` | N/A | No cost arithmetic |
| DB path from `pipeline.paths.GOLD_DB_PATH` | PASS | `bot_dashboard.py:31` imports `GOLD_DB_PATH, LIVE_JOURNAL_DB_PATH` from `pipeline.paths`; new routes at `:1224` and `:1183` use them via module-level `JOURNAL_PATH = LIVE_JOURNAL_DB_PATH` and `GOLD_DB_PATH` directly. No hardcoded paths. |
| Lane state via canonical accessor | PASS | `bot_dashboard.py:1293` lazy-imports `from trading_app.lane_ctl import get_lane_override, get_paused_strategy_ids` — uses canonical accessor, never reads override JSON directly. Matches `institutional-rigor.md` § 4. |
| `configure_connection` consistency | PASS | Both new DuckDB connections call `configure_connection(con)` at `:1186` and `:1227`, mirroring the existing `/api/trades` pattern at `:1116`. |
| One-way dependency `pipeline/` ← `trading_app/` | PASS | Imports flow correctly. |

---

## Section C — Statistical Rigor: N/A

No statistical claims, no hypothesis tests, no p-values. UX surface only.

---

## Section D — Production Readiness: PASS (with one MEDIUM)

| Check | Status | Evidence |
|---|---|---|
| Fail-closed | PASS | `/api/trade-book` and `/api/lane-status` both wrap reads in try/except with structured fallback (empty arrays + `*_note` / `error` field). No silent success after failure. Matches `/api/trades` contract at `:1162`. |
| Idempotent | PASS | Read-only routes; trivially idempotent. |
| `except Exception` patterns | PASS | All four `except Exception` blocks in the new code log a note string or set the `error` field — no silent swallow. `bot_dashboard.py:1219, :1264, :1325`. |
| DB writes single-process | PASS | Both routes open `read_only=True`. No write contention. |
| Companion test exists | PASS | `tests/test_trading_app/test_bot_dashboard_routes.py` covers happy / empty / missing-DB / paused / empty-paused / bad-profile. 6/6 green. |

### MEDIUM-1 — JS innerHTML interpolation diverges from this file's escape convention

```
PREMISE:  fetchTradeBook() interpolates DB-sourced strings (strategy_id,
          exit_reason, broker, instrument, lane_name, filter_type, orb_label,
          direction, entry_model, execution_source) directly into
          tbody.innerHTML via raw `${fmt(t.field)}` template literals. fmt()
          only handles null/undefined; it does NOT escape HTML.

TRACE:    bot_dashboard.html:1835 defines `escapeHtml = v =>
          String(v ?? "").replaceAll("&", "&amp;")...`

          Established convention in this file:
          - :2091  trades blotter uses escapeHtml(meta.note)
          - :2099  trade error msg escaped
          - :2268  operator-checks innerHTML uses escapeHtml(check.name)
          - :2321  alerts innerHTML uses escapeHtml(alert.level)
          - :2553, :2574, :2583, :2710, :2832  17 other innerHTML sites use escapeHtml

          New code at :3640-3666 (live table) and :3668-3680 (paper table)
          interpolates DB-sourced fields with NO escapeHtml call.

          Catch block at :3683-3686 also raw-interpolates `${e}`:
              liveBody.innerHTML = `<tr><td colspan="12" class="muted">Fetch failed: ${e}</td></tr>`

EVIDENCE: Verified DB-sourced field values in gold.db.paper_trades:
            exit_reason: ['loss', 'scratch', 'win']
            strategy_id: 'MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20_O15' etc.
          Today these are alphanumeric + underscores (not attacker-controlled).
          But the file's own established pattern says ESCAPE on interpolation.

VERDICT:  SUPPORT (MEDIUM, not HIGH).

          Severity reasoning: data is internally pipeline-generated, not
          externally attacker-controlled (attacker needs gold.db/live_journal.db
          write access — if they have that, host is already owned). So this is
          convention/caller-discipline drift, not an active XSS vector. But the
          divergence is real, repeatable, and the file's own rule says escape.

FIX:      Wrap each `${fmt(t.field)}` and `${fmtNum(t.field, d)}` in
          `escapeHtml(fmt(t.field))` (note: fmtNum returns a string already
          via toFixed, so escapeHtml on its output is cheap and safe).
          Wrap `${e}` in `${escapeHtml(String(e))}` in the catch block.

          Estimated edit: ~30 token replacements in fetchTradeBook(); zero
          behavioral change for current data; matches the rest of the file.
```

---

## Section E — Caller Discipline: PASS

| Check | Status | Evidence |
|---|---|---|
| New route shares contract with `/api/trades` | PASS | Same try/except shape, same `configure_connection`, same fail-open-with-note pattern. Does NOT re-encode any logic from `/api/trades`. The two routes are intentionally separate (24h blotter vs lifetime book) — no shared code, but the contract is symmetric. |
| Lane-status uses canonical accessor | PASS | `from trading_app.lane_ctl import get_lane_override, get_paused_strategy_ids` — never re-encodes the override JSON parsing, never re-implements expiry logic. |
| Polling cadence wiring | PASS | `setInterval(fetchLaneStatus, 60000)` added next to the existing `setInterval` block at `:3700` — additive, no existing intervals modified. |
| Topbar button add | PASS | `btn-trade-book` inserted before existing `btn-preflight` at `:1505` — does not alter existing button event handlers; existing buttons still bind via `addEventListener` later in the file. Verified: no `btn-preflight`/`btn-refresh`/`btn-kill` queries broken. |
| Polling soak / DB handle leak | NOT FLAGGED | Per scope-rules-3 ("Don't add error handling, fallbacks, or validation for scenarios that can't happen"), no evidence of leak — `with duckdb.connect(...)` context-manages each call. |

---

## Section F — Integration & Execution: PASS

| Check | Status | Evidence |
|---|---|---|
| Routes execute end-to-end | PASS | `pytest tests/test_trading_app/test_bot_dashboard_routes.py -v` → 6/6 green in 0.51s. Tests use real DuckDB fixtures, not mocks of the DB layer. |
| Bad-profile handler | PASS | `test_lane_status_bad_profile` verifies HTTP 200 + structured `error` field, no 500. |
| Missing DB | PASS | `test_trade_book_missing_db` verifies live_note/paper_note populated. |
| Existing tests still pass | PASS | `pytest tests/test_trading_app/ -k bot_dashboard` → 25/25 green. No regression. |
| Drift checks | PASS | `pipeline/check_drift.py` → 119 checks pass, 0 skipped, 19 advisory. |
| Smoke check (browser) | PENDING (operator) | `docs/runtime/stages/feat-live-app-ux-smoke.md` written with cross-check baselines (paused=1, live_trades=2, paper_trades=580). Required before merge per the plan. |
| Preflight check 6 | PASS by inspection | `scripts/run_live_session.py:201-217` — additive, isolated; opens `TradeJournal` in `mode="preflight"`, checks `is_healthy`. The hardcoded `checks_total = 6` preserves the existing pattern (see LOW-1 below). |

---

## Section G — Blueprint Cross-Check: N/A

UX surface; no NO-GO interaction, no flagged-assumption surface, no ML revival risk.

---

## Section H — Improvements

### LOW-1 — `checks_total` hardcoded (pre-existing, preserved)

`scripts/run_live_session.py:81` keeps `checks_total = 6  # NOTE: must match number of check blocks (1-6) below`. The diff bumped 5 → 6 alongside adding check 6. This pattern was already in main; the diff preserves it rather than introducing it.

`integrity-guardian.md` § 3 says "Never hardcode check counts — compute dynamically." The proper fix is to count check blocks via list-of-callables or accumulate per block. **Not a blocker for this PR** (preservation of existing debt, not new). Recommend a follow-up cleanup ticket to refactor the preflight runner to a list-of-checks pattern.

### Improvement-1 — Apply `escapeHtml` in `fetchTradeBook` (closes MEDIUM-1)

See MEDIUM-1 fix recipe. Single-PR-scope task, ~30 token replacements.

### Improvement-2 — Add a column-count constant for the blotter colspans

`bot_dashboard.html:3641` and `:3654` both hardcode `colspan="12"` and `colspan="15"` matching the `<thead>` columns. If a future column add forgets to update colspan, the empty-state row breaks layout. Trivial; defer to follow-up unless the empty-state breakage materializes.

---

## Verdict

**B+. Ship after addressing MEDIUM-1 (escapeHtml in fetchTradeBook) and operator smoke check.**

Tests are honest (real DBs, not over-mocked), routes use canonical accessors, preflight extension is isolated, no new technical debt beyond preservation of an existing soft-count pattern. The MEDIUM finding is a real convention divergence with a 30-token fix that brings the new code in line with the rest of the file's established escape rule.

## Action items

1. **MEDIUM-1 fix** — wrap each `${fmt(t.field)}` / `${fmtNum(t.field, d)}` / `${e}` in `escapeHtml(...)` inside `fetchTradeBook()`. ~30 token replacements; behavior-identical for current data.
2. **Operator smoke check** — run `START_BOT.bat`, complete `docs/runtime/stages/feat-live-app-ux-smoke.md`, fill in observed values, sign verdict.
3. **Re-review after #1** — single pattern fix, can be self-verified by running the existing test suite (no new tests needed; the escape is HTML-output transparent for alphanumeric+underscore data).
4. **Optional follow-up ticket** — refactor preflight `checks_total` from manual constant to list-of-checks length (closes LOW-1; not a merge blocker).

After items 1+2 land: re-run `pytest tests/test_trading_app/ -k bot_dashboard -q` (regression) and merge.

## Caveats / Disconfirming Evidence / Limitations

- **TestClient ≠ browser.** The 6 route-level tests prove JSON shape and HTTP 200 contract; they cannot prove the HTML `addEventListener` wiring fires `fetchTradeBook()`, that the 60s `setInterval` polling registers, or that the badge tooltip reads the right field. Operator browser smoke (`docs/runtime/stages/feat-live-app-ux-smoke.md`) is the only check that catches those failure modes. Smoke is operator-pending at merge time; the doc ships with cross-check baselines (`paused=1`, `live_trades=2`, `paper_trades=580`) so silent drift is detectable.
- **MEDIUM-1 escapeHtml fix is defense in depth, not active mitigation.** Today, `gold.db.paper_trades.strategy_id` and `live_journal.db.live_trades.broker` are pipeline-generated alphanumeric+underscore strings (verified via gold-db query). The fix prevents future drift if a future writer ever introduces an HTML-active string into either DB; it does not patch an exploitable XSS today.
- **LOW-1 (`checks_total = 6`) deferred, not closed.** `scripts/run_live_session.py:81` keeps the hardcoded count. The proper list-of-callables refactor is filed as `preflight-checks-total-hardcode` in `docs/runtime/debt-ledger.md`. Risk if not addressed: a future reviewer adds check 7 and forgets to bump `checks_total`, making the preflight under-report. Bounded blast radius (preflight only; not on the order-route / kill-switch path).
- **`/api/lane-status` profile is hardcoded to `topstep_50k_mnq_auto`** at the route default. The `bad_profile` test verifies graceful degradation for unknown profile ids, but multi-profile operators would need to pass `?profile=…` explicitly. Not a current-state limitation (one live profile today) but worth flagging if/when XFA Express + Bulenox land.
- **No protected-path scan run.** `capital-review` was not invoked on this branch because the diff is UX-only. If a future reviewer flags the `bot_dashboard.py` route additions as touching shared infra, escalate via `/capital-review`.
- **Single-reviewer pass.** The verdict is one reviewer's read, not an adversarial-audit gate verdict (per `.claude/rules/adversarial-audit-gate.md`, the gate triggers only on CRIT/HIGH fixes in truth-layer paths — UX dashboards do not qualify). Independent re-review on PR review is the next adversarial layer.
