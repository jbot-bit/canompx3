# Plan — Own-Capital Sizing: Contract-Size × Lane-Count Frontier (Carver-grounded)

## Context

**Why this change.** The own-capital max-return report
(`scripts/reports/report_max_return_own_capital.py`, branch
`feature/own-capital-return-report`) currently sizes a **micros-only** book and a
scale-ladder was added this session that multiplies the risk budget 1.0–2.0×. The
operator pushed back on three fronts, all correct:

1. *"calculate correctly, no half-ass"* — audit the math.
2. *"it's more complicated than that"* — the "ceiling = 1.0× / 16 micros" answer is
   too narrow; it treats micros-only as the only deployable instrument.
3. *"can we do minis or full-size with fewer lanes?"* — the report marks minis
   "display-only NOT DEPLOYABLE", which **dodges the real question**.
4. *"official guidelines for our specific needs?"* — size by the literature we
   already have, not freelanced rules.

**What grounding revealed (all from on-disk page-cited extracts, not memory):**

- **Carver Ch10 p159 sizing formula** puts **contract multiplier in the
  denominator**: `Position = (forecast/10) × cash_vol_target / (price × inst_vol ×
  FX × contract_multiplier)`. Micro-vs-mini-vs-full is *the same lever* as that
  denominator — the report hardcodes micros and never explores it. (Extract:
  `carver_2015_volatility_targeting_position_sizing.md`.)
- **Minis are ~1.5× cheaper per unit risk than micros — NOT 3.5×** (VERIFIED by
  computing from canonical `COST_SPECS` fields directly, NOT trusting the
  `cost_model.py:134` comment, which counts commission only). Real **total**
  friction/point (commission + spread + slippage): **MNQ $1.46/pt vs NQ
  $0.955/pt = 1.53× cheaper.** The "3.5×" in the code comment ignores spread +
  slippage (modeled in $ terms ~10× for NQ). The minis cost edge is real but
  modest, and **MGC↔GC friction/pt is IDENTICAL ($0.5740) — zero cost advantage
  to full-size gold**, only a granularity penalty. **ES is NOT in COST_SPECS at
  all** → full-size S&P cannot be modeled without inventing numbers (out of
  scope on grounding terms). So the honest contract-size axis is: MNQ→NQ (1.5×
  cheaper, real), MGC→GC (no cost edge, pure granularity), MES→ES (unmodelable).
- **BUT bigger contracts are lumpier** → the binding gate the ladder found (worst
  single day vs the $1,500 daily-loss-limit) binds *harder*, because 1 mini = 10
  micros of indivisible single-session risk. This is the real tradeoff: lower
  cost vs worse granularity/diversification.
- **Carver Ch11 p169-171 caps lane count** (verified by reading the FULL pages
  184–187 from `resources/Robert Carver - Systematic Trading.pdf` via pymupdf
  text layer — not a snippet, not the extract paraphrase): p169 *"you must also
  limit the maximum number of bets that can be placed at any time... I recommend
  that this maximum shouldn't be more than 2.5 times the average number of bets
  you expect to be holding over time."* With avg ~2–3 concurrent signals,
  Carver-compliant max ≈ 5–7 lanes — **the current 16-lane book likely violates
  Carver's own lane ceiling.**
- **METHODOLOGY BUG found by reading p170–171 in full: the report uses the WRONG
  diversification-multiplier formula.** `_diversification_multiplier` in the
  report computes `D = 1/√(w'Cw)` from the empirical correlation matrix. Reading
  the pages: that is the **asset-allocator / staunch-systems-trader** method
  (p170, "use the correlations between the returns of each instrument trading
  subsystem"). For a **semi-automatic trader — which canompx3 is documented as
  (p169)** — Carver p171 is explicit: *"the instrument diversification multiplier
  should be your maximum numbers of bets divided by the average number of bets...
  limited to a maximum of 2.5."* (footnote 119: this conservatively assumes all
  bets perfectly correlated.) So D and lane-count are **the SAME knob** for a
  semi-auto book; the report currently treats them as independent and uses the
  allocator formula. The frontier must use D = max_bets/avg_bets (capped 2.5),
  not the empirical-correlation D. This was invisible to the regex-snippet pass
  and only surfaced on a full read — the operator's "read it properly" was right.
- **Carver Ch12 p169 cross-ref (VERIFIED VERBATIM, pdfpage 219) directly answers
  the operator's question:** *"adjusting instrument weights is not an option for
  dealing with size problems. So if you have problems you should consider reducing
  the size of your portfolio by lowering the maximum number of bets, or not
  trading that instrument."* → Carver's prescribed fix for granularity/size
  problems IS fewer lanes.
- **Carver p169 also wants ≥4 blocks/instrument (VERIFIED VERBATIM, pdfpage 222):**
  *"I recommend having a maximum possible position of at least four instrument
  blocks in any instrument."* This CUTS AGAINST minis on a $30k account: micros
  let each lane hold ≥4 blocks within the DLL; a single mini = 10 micros may
  exceed the per-day risk budget before reaching 4 blocks. A real grounded tension
  the frontier must surface, not paper over.
- **Carver Ch11 p133/p170**: once the diversification multiplier hits its 2.5 cap,
  adding lanes gives **zero** Carver-compliant value — you just re-size existing
  lanes down. So "many micro lanes" past the D-cap is illusory diversification.
- **Carver Ch12 p179**: execution cost rises with trade size vs available volume
  (market impact) — a second-order penalty on minis/full-size that a naive 1
  mini = 10 micros equivalence ignores.

**Intended outcome.** Replace the micros-only ladder framing with a single
Carver-grounded **contract-size × lane-count frontier** that honestly answers:
*for $30k / $6k-DD, what book maximizes risk-first return* — across
{micro, mini} sizing and a Carver-compliant lane count — gated on the real
binding constraints (1.5×-stress DD ≤ $6k, worst-day ≤ DLL, margin/Crit-11
declared UNVERIFIED).

---

## Pre-work: math audit (BEFORE any new feature — addresses "calculate correctly")

The new gate logic must be proven correct before extending. Audit, in
`scripts/reports/report_max_return_own_capital.py`:

1. **`_worst_day` basis mismatch (SUSPECTED BUG).** Worst-day is computed on the
   **clean** post-trim series (`series`), but the DD gate uses the **1.5×-stress**
   series. The DLL is a live halt — it should be checked on the same stressed
   series the hard-stop gate uses, or at minimum the report must state which
   series each gate reads. Decide: clean worst-day (honest historical) vs stressed
   worst-day (consistent with the DD gate). Likely BOTH, labelled.
2. **DD-backstop circularity.** Confirm post-trim DD ≤ $4k is *enforced* (so it is
   never a real gate) and that pre-trim DD is the honest number — already handled,
   re-verify after edits.
3. **Diversification multiplier formula is WRONG for a semi-auto trader (CONFIRMED
   by full read of Carver p170-171).** The report's `_diversification_multiplier`
   uses the empirical `1/√(w'Cw)` allocator method; the semi-auto method is
   `D = max_bets/avg_bets` capped 2.5. This MUST be replaced as part of the
   frontier work (it is the load-bearing sizing input). Also confirm D is not
   silently re-applied per-rung in a way that double-counts when the scalar
   changes.
4. **Annualization window.** Confirm `$/yr` divides by the book's *common* span,
   and that cross-book $/yr comparisons keep the existing window-mismatch caveat.

Audit method: read the functions + a hand-traced 2-lane numeric example verified
against a one-off Python check (read-only). No silent "looks right".

---

## Approach (recommended)

Reframe the ladder as a **two-axis frontier sweep**, all read-only, micros AND
minis, Carver-compliant lane counts. Keep the binding-gate logic already fixed
this session (1.5×-stress DD ≤ $6k AND worst-day ≤ DLL; margin-micro count + the
two UNVERIFIED gates reported, never auto-capped).

### Axis 1 — Contract size (the new lever, grounded in Carver Ch10 denominator)
For each lane, allow the sizer to choose its block from {micro, mini} using the
**canonical cost model** (`pipeline.cost_model.COST_SPECS`) for the instrument's
real per-contract friction and point value — NOT a hardcoded 10× equivalence and
NOT the misleading "3.5× cheaper" comment.
- **Modelable pairs only:** MNQ↔NQ (1.53× cheaper/pt, real edge), MGC↔GC (zero
  cost edge — identical friction/pt — granularity-only). **MES↔ES is excluded:
  ES has no CostSpec; modeling it would require inventing numbers.**
- A mini block uses COST_SPECS's real point value AND real friction for the mini
  symbol, so the (modest) cost asymmetry and the lumpier granularity are both
  honestly modeled.
- Apply Carver Ch12 p179 market-impact (VERIFIED: *"the difference between the
  mid and the price you achieve will depend on how large your trades are compared
  to the available volume"*) as a **flagged, conservative** extra-slippage add-on
  for mini size, clearly labelled an assumption, not a measurement.
- Enforce Carver p169's **≥4 blocks/instrument** granularity floor as a reported
  check: flag any mini lane that cannot hold ≥4 blocks within the DLL.

### Axis 2 — Lane count = the diversification knob (Carver Ch11 p169-171)
For a semi-auto book, lane count and the diversification multiplier are ONE knob:
- Compute **avg concurrent bets** from the replay (mean over trading days of the
  count of lanes that produced a signal that day).
- **D = max_bets / avg_bets, capped 2.5** (Carver p171 semi-auto formula) —
  REPLACING the report's current empirical-`1/√(w'Cw)` allocator formula, which
  is wrong for this trader type.
- Carver-compliant **max lanes = round(2.5 × avg)**; report books at
  {Carver-max, current-16, tight-5} so the diversification-vs-concentration
  tradeoff is explicit, and surface that 16 lanes likely exceeds the compliant
  max (and that beyond the cap, extra lanes add zero risk-adjusted value — the
  D-cap means they just re-size existing lanes down).

### The frontier output
For each (contract-size policy × lane-count) cell, report the **same honest gate
panel** the ladder now produces:
`micros-equiv | pre-trim DD | post-trim DD | 1.5×-stress DD | worst-day vs DLL |
$/yr | commission drag | verdict (PASS / FAIL-stress / FAIL-DLL) | flags`.
Then name the **frontier ceiling**: the highest-$/yr cell that passes BOTH binding
gates, with margin/Crit-11 declared UNVERIFIED. Bigger-contract deployment is
ALLOWED in the *model* (it's the operator's question) but every mini/full cell
carries the market-impact + margin-UNVERIFIED + liquidity-UNVERIFIED flags, and
the recommended sketch still defaults to the most defensible passing cell.

### Honesty guardrails (non-negotiable, from doctrine + Carver Ch12 p179)
- No cell is labelled deployable on $/yr alone — the $23k–$26k artifact this
  session is the cautionary example (overconfidence, Carver Ch12 p179).
- Commission drag shown per cell (minis change it materially).
- Margin (no canonical day-margin source exists) and Crit-11 (needs a real
  profile MC) stay **UNVERIFIED**, never faked PASS.
- The `self_funded_30k` sanity guard (`max_contracts_micro=12`, DLL $1,500,
  `prop_profiles.py:610`) reported as a flag, not a cap (doctrine Rule 4).

---

## Critical files

- `scripts/reports/report_max_return_own_capital.py` — the only logic file.
  - Reuse: `carver_size` (extend with a `block_size` per lane), `_eval_rung`,
    `_book_dd`, `_worst_day`, `_slippage_series`, `_dd_backstop_trim`,
    `_diversification_multiplier`, `_common_days` — all already present.
  - Reuse canonical: `pipeline.cost_model.COST_SPECS` /
    `pipeline.cost_model.get_cost_spec` for real mini-vs-micro friction +
    point value (do NOT hardcode 10×).
  - Reuse canonical: `trading_app.account_survival._load_lane_trade_paths` +
    `_max_observed_rolling_drawdown` (already used — replay truth, no re-encode).
  - New: `avg_concurrent_bets()` helper (count days with ≥1 lane signal),
    `_frontier()` sweep over {size policy × lane count}, `_print_frontier()`.
- `docs/plans/2026-06-06-max-return-own-capital-design.md` — append a
  "Contract-size × lane-count frontier" section + the Carver citations above.
- (gitignore note) the script is shadowed by `.gitignore:143 reports/` — commit
  with `git add -f`, same as sibling reports.

NO edits to `pipeline/`, `trading_app/`, schema, `account_survival.py`, C11, or
any profile. Read-only against `gold.db`. Micros-only default path stays
byte-identical when the new axes are at their defaults.

---

## Verification

1. `python -m py_compile` + `ruff check` on the report (clean).
2. Run the report; confirm:
   - the math-audit fixes (worst-day basis labelled, DD circularity stated);
   - micros-only frontier cell reproduces the current 1.0× book ($9,599/yr,
     16 micros, worst-day −$1,279, PASS) — regression anchor;
   - mini cells show LOWER commission drag (grounded in COST_SPECS) but report
     whether the lumpier worst-day trips the DLL;
   - Carver lane-cap finding printed (avg concurrent bets → compliant max vs 16).
3. `python pipeline/check_drift.py` → expect 178/0 (report is outside drift
   surfaces; confirm no regression).
4. Stop after results. No commit until operator reviews the frontier numbers
   (capital-modeling = Tier B; operator GO before any profile/deploy step).

---

## Honesty / verification log (what was VERIFIED vs ASSUMED)

Operator instruction: "no metadata trusting, no cheating, read it properly."

- **VERIFIED by computing from canonical `COST_SPECS` fields** (not the comment):
  MNQ $1.46/pt vs NQ $0.955/pt total friction = 1.53× (the code comment's "3.5×"
  is commission-only and misleading). MGC=GC friction/pt identical. ES absent.
- **VERIFIED by reading FULL Carver PDF pages 184–187** (pymupdf text layer, no
  OCR needed — PDF has a clean text layer): the 2.5× lane cap (p169), and the
  semi-auto D = max/avg formula (p171) that the report currently gets wrong.
- **PDF ingester:** project tooling is `pymupdf 1.27.2` + `ocrmypdf 17.4.0`,
  confirmed via `scripts/tools/check_pdf_tooling.py` (PASS). Raw PDF lives in the
  MAIN checkout `resources/` only (absent in the report worktree) — re-extract
  from main, or copy, when next session needs more pages (e.g. Ch12 p177-203 for
  the cost-vs-size detail behind minis, which is NOT yet fully read — the Ch12
  extract is the scratch-EOD context, not the size-economics pages).
- **STILL ASSUMED / NOT YET READ (do next session before relying on):** Carver
  Ch12 pp196-203 "Size" economics (the actual minis-on-small-accounts math);
  Ch10 pp153-163 full position-sizing formulas with contract multiplier. The
  ≥4-blocks-per-instrument rule was seen in a snippet (pdfpage 219/222) but NOT
  yet read in full-page context — re-read before enforcing it as a gate.

## Open question for the operator (will ask via AskUserQuestion)

Scope of the contract-size axis: **{micro, mini} only**, or include **full-size**
(NQ/ES/GC full contracts)? Full-size on $30k is almost certainly margin-infeasible
(one ES ≈ $13k+ day-margin) and would be flagged UNVERIFIED-margin → likely
FAIL, but the operator asked about "minis or full size" explicitly, so confirm
whether to model full-size at all or stop at minis.
