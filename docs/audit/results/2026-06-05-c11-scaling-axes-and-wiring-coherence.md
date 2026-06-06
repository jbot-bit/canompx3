# C11 Scaling Axes + System Wiring Coherence

**Date:** 2026-06-05 · **Tool:** Claude Code · **Mode:** READ-ONLY research + wiring audit (Tier A)

## Scope / Question

Two operator pushbacks on the "account size is moot" finding
(`2026-06-05-c11-firm-tier-economics.md`):
1. **"What if it's more lanes, or full-size contracts, or multiple contracts, or
   something not obvious?"** — the size-is-moot conclusion only holds *if the book
   stays at 1 micro/lane*. Test whether any non-obvious scaling axis makes a bigger
   account actually earn more.
2. **"Ensure all these moving parts communicate, understand each other, are up to
   date — proper wiring and project management. Sometimes we discover we already
   had a tool for this."** — a coherence audit across survival-sim ↔ live-execution
   ↔ economics ↔ the concurrent peer-terminal C11 work.

Read-only: no code/DB/profile written. Source-traced from canonical modules.

## Decision / Verdict

**Scaling axes:** the position-sizing machinery to run >1 contract **already
exists and is canonical** (`compute_position_size_vol_scaled`, Carver Ch.9 vol
targeting). The book runs 1 micro/lane because of **two deliberate caps**
(`max_contracts=1` in the live engine; `contracts_per_trade_micro=1` hardcoded in
the survival sim) — not an engine limitation. So a bigger account *could* earn more
**via multiple contracts/lane** — but only if those caps are lifted, AND that path
crosses an **unguarded sim↔live consistency seam** (below) that must be closed
first. **More lanes** is the cleaner growth axis and does *not* require a bigger
account unless combined DD exceeds the belt. **Mini/full-size contracts** = a
discrete 10× risk jump, gated identically.

**Wiring:** the sizing tool is **single-source canonical** (no duplicates — good).
But there is a **latent, currently-dormant inconsistency**: the C11 survival gate
certifies safety on a **1-micro DD model**, while the live engine sizes contracts
**from equity via vol targeting**. Today this is safe (deployment is *signal-only*,
`max_contracts=1`). It becomes a **silent capital risk** the moment anyone raises
`max_contracts` or auto-executes on a bigger-equity account — the gate would
under-project real drawdown. This is the real answer to "multiple contracts": the
scaling axis is gated by an unguarded seam, recorded here as a debt item.

## Findings

### A. The sizing chain (source-traced) — the tool already exists

| Layer | File:line | What it does |
|---|---|---|
| Vol-scaled sizer (canonical, **only impl**) | `trading_app/portfolio.py` (`compute_position_size_vol_scaled`, `compute_vol_scalar`) | `equity × risk_pct ÷ risk_points`, ATR_20/median_ATR_20 Turtle vol scalar (Carver Ch.9) |
| Engine sizing | `trading_app/execution_engine.py:259` `_compute_contracts(..., max_contracts=1)` | calls the vol sizer, then **clamps to `max_contracts`** (default 1), logs `CONTRACTS CLAMPED` |
| Contract-reduction factor | `execution_engine.py:343` `reduced = int(base × suggested_contract_factor)` | a throttle that can *reduce* contracts (rejects if <1) |
| Live qty resolution | `session_orchestrator.py:2142` `_resolve_execution_order` → `prop_profiles.resolve_execution_order` | maps micro↔mini via `qty_divisor`, **fails closed** on non-divisible qty |
| Survival sim sizing | `account_survival.py:615-616` `contracts_per_trade_micro=1` (**hardcoded**) | the C11 DD Monte Carlo assumes exactly 1 micro/lane |
| Current live qty | `run_live_session.py:155`, `session_orchestrator.py:1583` `qty=1` | signal-only path emits 1 |

**"Did we already have a tool for this?" → Yes.** `compute_position_size_vol_scaled`
is a proper Carver-grounded vol-targeted sizer, single-source (grep confirms no
duplicate in `research/`/`scripts/`). Multi-contract sizing is **not** unbuilt — it
is built, canonical, and **clamped to 1** by config.

### B. The three scaling axes — which one makes a bigger account earn more?

| Axis | Return effect | DD effect | Needs bigger account? | Gate / blocker |
|---|---|---|---|---|
| **More lanes** | adds ~uncorrelated streams | sub-linear (low ρ) | only if combined DD > belt | allocator finds genuinely low-ρ validated sessions (current: 3 lanes, max ρ=0.06). Bounded by *real* uncorrelated edge, not by account size. **Cleanest axis.** |
| **Multiple micros/lane** | ~linear ↑ | ~linear ↑ | **YES** — this is the axis where MLL buys earnings | `max_contracts=1` + survival sim hardcoded 1. Lifting both crosses the **unguarded seam** (Finding C). Edge must survive scaling (cost/slippage at size). |
| **Mini (full-size) contracts** | discrete 10× | discrete 10× | YES (~10× DD headroom) | `resolve_execution_order` qty_divisor (1 mini = 10 micro); fail-closed on non-divisible. Same seam. |

So the operator's instinct is **correct**: size is moot *only at 1 micro/lane*. The
honest refinement to the prior doc: **a bigger account earns more IF the book runs
multiple contracts/lane — but that requires lifting two deliberate caps and closing
the sim↔live seam first.** It is a real path, not a non-starter; it is just not
"free," and it is gated behind a capital-safety fix, so it stays Tier B / NO-GO
until designed and audited.

**Important honesty check:** more contracts ≠ more *edge per contract*. The vol
sizer scales size with equity, but the strategy's ExpR/contract is fixed. Whether
the edge survives at 2-3× size depends on MNQ micro liquidity/slippage at the lane's
session times — **not measured here**. A scaling decision needs that measured first
(don't assume linear edge).

### C. ⚠ Wiring inconsistency — the C11 gate and the live sizer disagree on contract count

- **C11 survival gate** (`account_survival.py:616`) projects 90-day DD assuming
  **exactly 1 micro/lane** (`contracts_per_trade_micro=1`, hardcoded).
- **Live execution engine** (`execution_engine.py:287`) sizes contracts **from
  equity** via the vol sizer, clamped only by `max_contracts` (default 1).
- **No drift check** ties these together (grep of `check_drift.py` finds none).

**Today: safe.** Deployment is signal-only (operator places 1 micro by hand) and
`max_contracts=1`, so both paths agree at 1. **Tomorrow: latent capital risk.** If
anyone (a) raises `max_contracts` for a bigger account, or (b) flips to
auto-execution where the vol sizer can return 2-3 on higher equity — the C11 gate
would still certify "survives" on a 1-micro DD model while the live book runs
multiple contracts and draws down 2-3× more. The gate would **silently
under-protect**. This is precisely the "are the parts communicating?" gap. It is a
**dormant seam, not an active bug** — recorded as debt, not patched (Tier B,
capital path, adversarial-audit-gated; not in scope for a read-only stage).

### D. Cross-surface coherence (the project-management check)

| Source | Says | Consistent? |
|---|---|---|
| Peer terminal (main checkout, 9429c540) | baseline 90d DD $2,038.84; `cap_x0.75`→$1,535, `cap_x0.80`→$1,594, both clear $1,800 belt; budget = 0.90×$2,000 | ✅ matches my source-trace (§2 of economics doc): belt resolver = 0.90×MLL |
| Economics doc (this session) | size moot at 1 micro; $100k clears uncapped; cap fix is cheaper | ✅ consistent — cap fix on 50k = peer's recommendation; no bigger account needed |
| This scaling doc | multi-contract *could* earn more but crosses the unguarded seam | ✅ refines, doesn't contradict — explains *why* "stay at 50k + cap" is right *now* |
| `docs/runtime/stages/c11-80pt-cap-wiring.md` (origin/main, peer-owned) | "$2,038 at 0.75 stop / needs stop≤0.50" | ❌ **STALE/WRONG** — conflates uncapped baseline with capped result (verified lines 11,25,27,51-62,97-98). Peer flagged it; peer-owned + on origin/main ⇒ left untouched. **Debt item D-1.** |
| Repo `PAYOUT_POLICIES` (scraped 2026-04-08) | flat $5k/$6k payout caps | ❌ **STALE** vs live tier-scaled caps (economics doc §3). **Debt item D-2.** |

**Coherence verdict:** the *active truth* surfaces (peer execution at 9429c540, my
source-traces, both new docs) are mutually consistent and current. Two *committed
docs* carry stale claims (D-1 cap-wiring stage, D-2 payout canon) — both flagged,
neither autonomously editable (peer-owned / requires re-scrape). No contradiction
between the live-truth surfaces; the staleness is isolated to documentation lagging
the 2026-06-04 belt change and the post-scrape ToS drift.

## Reproduction / Files

- **Source-traced (read-only):** `trading_app/portfolio.py`
  (`compute_position_size_vol_scaled`), `trading_app/execution_engine.py:259,343`,
  `trading_app/live/session_orchestrator.py:2142,1583`,
  `trading_app/prop_profiles.py:197` (`resolve_execution_order`),
  `trading_app/account_survival.py:616`.
- **Grep evidence:** single sizing impl (`grep -rln def compute_position_size_vol_scaled`
  → `portfolio.py` only); no sim↔live contract drift check
  (`grep contracts_per_trade check_drift.py` → none).
- **Peer C11 figures:** concurrent main-checkout session at commit `9429c540`
  (cited, not re-run).
- **Companion:** `docs/audit/results/2026-06-05-c11-firm-tier-economics.md`.

## Limitations / Caveats

- **Edge-at-scale NOT measured:** the claim "multiple contracts could earn more"
  assumes the lane edge survives at 2-3× size. MNQ micro slippage/liquidity at the
  lane session times is unmeasured here — a real scaling decision must measure it
  first (do not assume linear edge per contract).
- **Seam is dormant, not proven-exploitable:** Finding C is a latent risk under the
  *current* signal-only / `max_contracts=1` config, not an active bug. It becomes
  live only if those caps change. Flagged as Tier-B debt, not fixed (read-only
  stage; capital path needs the adversarial-audit gate).
- **Debt items D-1 (stale cap-wiring stage) and D-2 (stale payout canon)** are
  peer-owned / require re-scrape — flagged for the owning work, not edited here.
- **`9b3fc530` bracket-parity adversarial-audit gate remains OPEN** — no arming
  follows from any finding here.
- Peer-terminal figures are cited as peer-provided (settled by their execution at
  9429c540), not independently re-run this session.

## Bottom Line

The operator was right to push: size is moot *only* at 1 micro/lane. The honest,
non-pigeonholed answer — **a bigger account earns more only via multiple
contracts/lane, the tool for which already exists (canonical vol sizer) but is
deliberately clamped to 1, and lifting that clamp crosses an unguarded
survival-sim↔live-sizing seam that must be closed first (Tier B).** For the book
*as it runs today* (1 micro, signal-only), stay on `topstep_50k_mnq_auto` + the cap
fix; the bigger-account / multi-contract path is a real future growth lever, not a
free one, and is correctly NO-GO until the seam is closed and `9b3fc530` is audited.
