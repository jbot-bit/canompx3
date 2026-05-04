# ChatGPT Bundle — Handoff

**Status:** Complete and ready to use. Compiled 2026-04-18.

## What exists

```
<repo-root>/
├── chatgpt_bundle/                  ← 20 files (default upload)
├── chatgpt_bundle_optional/         ← 16 files (optional Pro extras)
└── CHATGPT_PROJECT_INSTRUCTIONS.md  ← ready-to-paste instructions (LOCAL, do not upload)
```

## How to use

1. Create a ChatGPT Project (name: "canompx3 ORB Research" or similar).
2. Drag all 20 `.md` files from `chatgpt_bundle/` into the Project's file upload.
3. If on Pro and want deeper lit/rule coverage, also drag `chatgpt_bundle_optional/` files (total ≤36).
4. Open `CHATGPT_PROJECT_INSTRUCTIONS.md` at repo root. Copy Option A (1,454 chars) OR Option B (4,131 chars). Paste into ChatGPT Project → **Instructions** field. Save.
5. Test with: "what's in this project?" → should reference `00_INDEX.md`. "is `rel_vol_HIGH_Q3` real?" → should cite `06_RD_GRAVEYARD.md`. "should I tune G5 threshold?" → should refuse, cite Mode A.

## Refresh cadence

Update when the underlying canonical source changes materially:

| Bundle file | Refresh trigger |
|-------------|-----------------|
| `CANONICAL_VALUES.md` | Any edit to `pipeline/cost_model.py`, `pipeline/dst.py`, `pipeline/asset_configs.py`, `trading_app/holdout_policy.py`, `trading_app/prop_profiles.py` |
| `06_RD_GRAVEYARD.md` | Any major NO-GO decision (new kill, new PAUSED path, new DEAD entry in memory) |
| `04_DECISION_LOG.md` | Any architectural decision (new entry model, new aperture, new session, holdout mode change) |
| `02_USER_PROFILE.md` | User working-style changes (new typo pattern, new escalation rule) |
| `07_PLAYBOOKS.md` | New recurring question type emerges from ChatGPT usage |
| 17 copied rule files | When the `.claude/rules/*.md` or `docs/institutional/*.md` source changes — re-copy |
| 11 LIT files | When a new paper is extracted to `docs/institutional/literature/` — copy in |

## Known gaps (called out in `00_INDEX.md` provenance section)

Primary sources exist in `resources/` but are NOT extracted:
- Aronson ch6 (data-mining bias)
- Chan 2013 ch2-8 (we have ch1 pp 1-10 only)
- Chan 2008 *Quantitative Trading* (different book, only ch7 extracted)
- Carver ch4 / ch6 / ch8 (we have ch9-10 only)
- Benjamini-Hochberg 1995 (original FDR paper)
- Man Group 2015 overfitting
- Pardo (Trading Strategy Optimization)

To close any of these: open the PDF, extract specific pages, write a new `LIT_<name>.md` file in the same style as existing extracts (verbatim quotes + page citations + "how we use it" section). Add the file to `chatgpt_bundle_optional/` and update `00_INDEX.md` file map.

## Provenance

- 20 `chatgpt_bundle/` files: meta files (7) generated in session; rules/papers (13) copied from repo. Chan 2013 ch1 extract was directly read from PDF pp 1-10; other 6 LIT extracts trusted from prior project work.
- 16 `chatgpt_bundle_optional/` files: all copied from repo, trusted without re-verification at bundle-assembly time.
- Fabrication audit: one invented quote in `04_DECISION_LOG.md §1` was removed during a mid-session honesty check; `06_RD_GRAVEYARD.md` carries a memory-summary provenance note.
- Full provenance detail: `00_INDEX.md` §"PROVENANCE & HONESTY".

## When ChatGPT drifts

Common drift patterns and fixes documented in `CHATGPT_PROJECT_INSTRUCTIONS.md` §"When ChatGPT goes off-script". If drift recurs, edit `01_OPERATING_RULES.md` or `07_PLAYBOOKS.md` and re-upload that one file (ChatGPT picks up the new content on next chat).

## What's next (for future sessions)

- Extract Aronson ch6 when first needed — it's referenced but not verified in `quant-audit-protocol.md`.
- Extract Chan 2013 ch7 "Intraday Momentum Strategies" — closest to our ORB work but not extracted.
- If user downgrades Pro→Plus: delete `chatgpt_bundle_optional/*` from the ChatGPT Project. Nothing else needs to change.

## Memory pointer

`memory/chatgpt_bundle_location.md` records this bundle's existence for future session recall.
