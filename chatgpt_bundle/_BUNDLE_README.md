# ⚠ chatgpt_bundle/ — MIRROR, NOT CANONICAL · regenerate before trusting

**Status:** advisory / stale-prone. **Do not cite any file in this directory as
canonical truth.**

These files are a **hand-pasted mirror** assembled for copying into ChatGPT. They
are **not auto-regenerated** — nothing in the repo keeps them in sync with their
sources, so they drift the moment the canonical surface changes. There is
currently **no regenerator**; treat every file here as potentially stale.

## Where the real truth lives

| Bundle file (shorthand) | Canonical source — cite THIS instead |
|---|---|
| `00_INDEX.md` | `docs/INDEX.md` |
| `01_OPERATING_RULES.md` | `CLAUDE.md` + `.claude/rules/*` |
| `02_USER_PROFILE.md` | operator profile = `memory/` + global `~/.claude/CLAUDE.md` |
| `04_DECISION_LOG.md` | `docs/runtime/decision-ledger.md` + `docs/governance/decisions/` |
| `06_RD_GRAVEYARD.md` | `docs/STRATEGY_BLUEPRINT.md` § NO-GO + research-catalog `/nogo` |
| `07_PLAYBOOKS.md` | `TRADING_RULES.md` + `/trade-book` skill |
| `CANONICAL_VALUES.md` | **live code, not a doc:** `pipeline.dst`, `pipeline.cost_model`, `pipeline.asset_configs`, `trading_app.prop_profiles`, `trading_app.account_survival` |
| `STRATEGY_BLUEPRINT.md` / `TRADING_RULES.md` / `RESEARCH_RULES.md` | same-named files at repo root / `docs/` |
| `pre_registered_criteria.md` / `mechanism_priors.md` | `docs/institutional/` |
| `backtesting-methodology.md` | `.claude/rules/backtesting-methodology.md` |
| `LIT_*.md` | `docs/institutional/literature/` extracts (the citation source) |

Per `docs/governance/document_authority.md` conflict rules, **code/DB and the
canonical docs above win over this bundle every time.** If you need a fresh
bundle, re-export from the canonical surfaces — do not edit the mirrors and assume
they propagate back.
