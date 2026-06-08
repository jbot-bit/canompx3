# Decision: Broker Credentials Stored Plaintext at Rest — ACCEPTED (documented threat model)

- **Date:** 2026-06-07
- **Decider:** operator (joshd)
- **Class:** capital-path security posture (Tier B design-gate)
- **Status:** ACTIVE — accepted risk, revisit on trigger below

## Decision

Broker API credentials remain stored **in plaintext at rest** on the live-bot
host. No encryption-at-rest (DPAPI / OS keyring) is added. This is a deliberate,
examined acceptance of a known posture — **not** an unexamined gap.

This was the last open item from the 2026-06-07 capital review (Phase-2 Fork #1).
Options A (OS keyring) and B (DPAPI encrypt-at-rest) were considered and declined
in favour of accepting and documenting the risk.

## Where the plaintext lives (the full surface, verified 2026-06-07)

There are **three** plaintext-at-rest / in-process surfaces, not two:

1. `data/broker_connections.json` — written by `BrokerConnectionManager.save()`
   (`trading_app/live/broker_connections.py:138`) as `json.dumps(self._connections)`.
   The `credentials` dict holds API keys / passwords / secrets verbatim
   (tradovate `password`/`sec`, projectx `api_key`).
2. `.env` — loaded at module import (`broker_connections.py:17`); `_migrate_from_env`
   (:100-132) reads `PROJECTX_API_KEY`, `TRADOVATE_USER/PASS/CID/SEC` etc. from
   `os.environ`. The **auth modules read `.env` independently** of the JSON path —
   `projectx/auth.py:83-84`, `tradovate/auth.py:117-121`, and
   `scripts/tools/fetch_broker_fills.py` all read `os.environ` directly, bypassing
   `BrokerConnectionManager` entirely.
3. `os.environ` (runtime sink) — `_create_auth` (`broker_connections.py:261-277`)
   pushes the stored credentials **into `os.environ`** before authenticating, so
   the secrets are plaintext in process memory regardless of how they are stored
   on disk.

## Rationale

- **Severity is LOW-MEDIUM.** Both files are gitignored and untracked
  (`git check-ignore` confirms; `git ls-files` empty). The host is a
  single-operator Windows box, localhost-only. The exposure is **disk-at-rest**
  (malware, backup leak, shoulder-surf) — not a network surface.
- **Partial encryption is theater.** Encrypting only `broker_connections.json`
  while `.env` holds the identical secrets in plaintext beside it — and while the
  auth modules read `.env` directly — closes none of the real exposure. A genuine
  fix would have to encrypt/migrate **all three** loaders and stop the direct
  `.env` reads in the auth modules. That is a meaningful capital-path refactor
  (4+ files including `projectx/auth.py`, `tradovate/auth.py`,
  `fetch_broker_fills.py`).
- **Encryption-at-rest cannot close the runtime sink.** Secrets sit in
  `os.environ` in-process after `_create_auth` regardless of storage. Any
  at-rest option is at-rest only — it never reaches end-to-end protection, so it
  buys less than its cost on this single-user, gitignored, localhost host.
- **A botched fix is worse than the status quo.** A decrypt path that silently
  falls back to plaintext or empty credentials on key failure would violate
  fail-closed discipline (`institutional-rigor.md` §6) on a capital path. Adding
  that risk to protect a disk-at-rest-only exposure on a single-user box is a
  poor trade.

## Scope / what this does NOT change

- No code changes. No new dependency (`keyring` / `pywin32` not added).
- The gitignore protection (both files untracked) is the load-bearing control
  and stays in place.
- Does not weaken any existing live-preflight gate.

## Revisit triggers

Re-open this decision (toward Option A or B) if **any** of the following becomes
true:

- The live bot runs on a **shared / multi-user host** (DPAPI user-scope or
  keyring becomes worth the cost; disk-at-rest exposure widens beyond one user).
- The bot runs under a **different identity** than the interactive operator
  (scheduled-task / service account) where shoulder-surf and profile isolation
  change.
- Credentials for a **real-capital self-funded** broker are stored on this host
  (raises the severity ceiling vs funded-wrapper accounts).
- The host stops being single-operator or the files stop being gitignored.

## Doctrine / provenance

- Finding source: 2026-06-07 capital-review class-of-four, Phase-2 Fork #1.
- Audit (this decision): institutional `/check` audit, 2026-06-07 — corrected the
  surface from two loaders to three (added the direct-`.env`-reader and
  `os.environ` runtime-sink findings) before the operator chose acceptance.
- Sibling decision-record format: `2026-06-01-telemetry-waiver-express-funded.md`.
