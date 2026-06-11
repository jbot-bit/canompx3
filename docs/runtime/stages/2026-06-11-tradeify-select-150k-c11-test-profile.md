---
task: Add ONE dormant tradeify_select_150k AccountProfile (C11-test only) mirroring tradeify_100k_type_b scaled to 150K; run the real survival sim; do NOT activate.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/prop_profiles.py
blast_radius:
  - "trading_app/prop_profiles.py — ACCOUNT_PROFILES gains one active=False entry. Live auto-dispatch unaffected: resolve_profile_id(active_only=True) raises on inactive (:1317), prop_portfolio.py:674 skips inactive. C11 runs it via explicit --profile (active_only=False, account_survival.py:1165) — intended, safe."
  - "Reuses existing tradeify PropFirmSpec (dd_type=eod_trailing, :493) + ('tradeify',150_000) tier (:641). NO firm-spec / tier edit."
  - "Binding drift guard: check_account_profiles_declare_is_express_funded (check_drift.py:7406). check_daily_loss_dollars_below_mll is a no-op (profile sets no daily_loss_dollars)."
  - "Reads gold.db read-only (C11 sim). C11 may persist a report to data/state unless --no-write-state."
---

## Task

Add a single dormant `AccountProfile` for the Tradeify Select 150K account, mirroring
`tradeify_100k_type_b` (`prop_profiles.py:1129-1168`) scaled to 150K. Run the real
Criterion-11 survival Monte Carlo against it. **No live activation** — decision deferred
pending a passing C11 + explicit operator GO.

Origin: operator pasted a "Qwen" proposal; audited claim-by-claim — most was confabulated
(plain-dict schema, 11 phantom fields, an "exclusivity hardening / order-fingerprint"
subsystem that is both fictional AND a detection-evasion premise → REJECTED). The legitimate
kernel: no runnable 150K Tradeify profile instance exists. This stage adds only that.

## Scope Lock

- trading_app/prop_profiles.py

## Blast Radius

- `trading_app/prop_profiles.py` — `ACCOUNT_PROFILES` gains one `active=False` entry beside
  the other Tradeify profiles (~`:1168`). Real dataclass fields only (`:91-153`).
- Live path unaffected (inactive skipped by dispatch). C11 reaches it only via explicit
  `--profile`.
- No `PropFirmSpec` / `ACCOUNT_TIERS` edit (150K tier already present `:641`).
- Reads gold.db read-only.

## Explicitly NOT doing

- No plain-dict profile. No invented fields (`max_daily_loss`, `drawdown_type`,
  `max_contracts`, `consistency_rule`, `profit_target`, `payout_split`, `platform`,
  `automation_allowed`, `copy_trading_allowed`, `exclusive_use_policy`,
  `requires_isolated_strategy_id`, `strategy_id_namespace`, `c11_survival_threshold`,
  `blocked_sessions`). No `strategy_id_namespace` / `jitter` / order-fingerprint evasion.
  No firm-spec / tier edits. No copy-router change. No live activation without GO + passing C11.
