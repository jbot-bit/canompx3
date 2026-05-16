# ProjectX Gateway API — Phase 0 Grounding Notes (2026-05-16)

**Purpose:** Phase 0 baton doc for `C:\Users\joshd\.claude\plans\get-going-on-this-whimsical-rain.md` § 0.2.
**Canonical source:** `docs/reference/PROJECTX_API_REFERENCE.md` (644 lines, fetched 2026-03-25 from `https://gateway.docs.projectx.com`). This file is a NON-CANONICAL extract that pulls the answers Phase 1 needs from the canonical doc — do not cite from this file, cite from the canonical.

**Why not re-fetch:** Local canonical was fetched ~7 weeks ago and is comprehensive (URLs, auth, orders, positions, SignalR events, enums, pitfalls). Re-fetch deferred until Phase 1 if a referenced field is ambiguous. Per `integrity-guardian.md` § 7 (Ground in local resources before training memory) and CLAUDE.md § Local Academic / Project-Source Grounding Rule.

---

## D1 — Idempotency / duplicate-order risk (Phase 1.5)

**Canonical reference:** `docs/reference/PROJECTX_API_REFERENCE.md:150` and pitfall #4 (line 615).

**Verbatim text (line 150):** `customTag | string | optional | **Must be unique per account**`
**Verbatim pitfall #4 (line 615):** `Non-unique customTag | Tag must be unique per account across all orders, ever. Use timestamp+strategy hash`

**Implication for Phase 1.5:** ProjectX server enforces `customTag` uniqueness per account. `customTag` IS the broker-side idempotency token. Phase 1.5 work in `projectx/order_router.py` (currently no `customTag` set per agent report 2026-05-16) becomes:
- Generate `customTag = f"{strategy_id}_{trading_day}_{entry_bar_ts}"` (deterministic, deduplicating per bar) at order build site.
- Server-side rejection on duplicate is the safety net; no client-side cache needed.
- HOWEVER: tag must be unique **across all orders, ever** — so a session restart that re-fires the same bar would get a rejected order. Test path: verify rejection path is handled (OrderStatus=5, not 3 — line 390).

**Caveat:** `searchOpen` response does NOT return `customTag` (line 256), so reconciliation by tag fails post-fact. Combine with deterministic timestamp lookups.

---

## D4 — HWM `hwm_dollars=0.0` root-cause (Phase 1.4)

**Canonical reference:** `docs/reference/PROJECTX_API_REFERENCE.md:99-129` (Account search), `:312-323` (position fields), `:476-486` (GatewayUserAccount SignalR event), `:512-523` (GatewayUserPosition).

**Critical facts (verbatim from canonical):**
1. `/api/Account/search` response has `balance` (line 106) but NO `unrealizedPnL`.
2. **`docs/reference/PROJECTX_API_REFERENCE.md:320` (verbatim):** `> **CRITICAL:** There is **NO unrealizedPnL field** in the position response. Mark-to-market P&L must be computed manually: \`unrealized_pnl = (current_price - averagePrice) * size * tick_value\``
3. `GatewayUserAccount` SignalR event (lines 478-485) DOES carry `balance` push-updated. This is the live-equity surface.

**Three candidate root causes for `hwm_dollars=0.0` (to test against Monday evidence):**
- **(a)** `projectx/positions.py:query_equity()` is polling `/api/Account/search` and reading `balance`, but `balance` may not include unrealized P&L mid-session. If no fills yet, balance == starting balance (non-zero). If balance==0 in evidence: API call is failing silently or field-name mismatch.
- **(b)** Code expects `balance + unrealized_pnl` but unrealized_pnl computation upstream returns 0.0 silently because per-position MTM requires current `lastPrice` from `GatewayQuote` and that wiring may be missing.
- **(c)** Silent swallow path per `integrity-guardian.md` § 6 — exception caught with `return 0.0` somewhere.

**Phase 1.4 investigation plan:** Add temporary `log.info` at `account_hwm_tracker.update_equity()` entry (`account_hwm_tracker.py:400`) logging the raw equity dict. Re-run live. Compare against:
- `/api/Account/search` raw response → field-name confirmation.
- `GatewayUserAccount` push event → push-vs-poll divergence.
- Per Topstep rules (resources/prop-firm-official-rules.md § TopStep): firm enforces DD via account balance, no API spec for what to read — read both `balance` AND compute MTM unrealized.

---

## D2 — File logging under `--live` (Phase 1.1)

**Not API-dependent.** Pure infrastructure question. No spec extract needed.

---

## D3 — bars_recent (Phase 1.3)

**Not API-dependent.** Pure data-layer design question (BarPersister flush policy vs in-memory ring). No spec extract needed.

---

## Pitfalls re-confirmed (already in canonical, listed here for Phase 1 reviewer)

From `docs/reference/PROJECTX_API_REFERENCE.md:610-622`:
- Pitfall #2: OrderType=3 (StopLimit) vs =4 (Stop) — E2 entries must use type=4.
- Pitfall #6: PositionType Long=1 / Short=2 (NOT 0/1).
- Pitfall #7: Status=3 (Cancelled) vs =5 (Rejected) — separate handling required.
- Pitfall #9: Re-subscribe SignalR after reconnect.
- Pitfall #10: HTTP 200 ≠ success — check `response.success == true`.

---

## TopStep firm rules (Phase 0 read confirmation)

**Source:** `resources/prop-firm-official-rules.md:16-61` (TopStep section, fetched 2026-03-16).

**Relevant to Phase 1.4 (HWM grounding):**
- Automated strategies ALLOWED on Combine + Funded + Practice (line 20-26).
- TopStep does NOT publish a specific API field to read for DD compliance — firm enforces via internal account-equity stream. Implication: as long as our HWM tracker reads ProjectX's `balance` (push-updated via SignalR) + computed unrealized, we are aligned with what firm sees.
- Trade-copier rule: Live Funded Accounts CANNOT use trade copiers (line 32). Not relevant for `topstep_50k_mnq_auto` direct-API path.

---

## Read by

- [ ] Phase 1.1 implementer (D2 file logging)
- [ ] Phase 1.4 implementer (D4 HWM root-cause)
- [ ] Phase 1.5 implementer (D1 customTag, optional)

End of Phase 0 grounding notes. Update or refresh only when canonical reference is re-fetched.
