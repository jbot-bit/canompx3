# ChatGPT Project Kit — Canompx3

Assembled 2026-03-22. Aligned with official OpenAI documentation (help.openai.com, updated Mar 2026).

## Official ChatGPT Projects Limits (verified from OpenAI Help Center)

| Plan | Files per project | Notes |
|------|------------------|-------|
| Free | 5 | |
| Go, Plus | **25** | |
| Pro, Business, Enterprise, Edu | **40** | |

- **Batch upload limit:** 10 files at a time (upload in two batches if >10)
- **Project instructions:** 8,000 characters (loaded every message, overrides global custom instructions)
- **Individual file max:** 512MB for docs/PDFs, 50MB for spreadsheets, 20MB for images
- **Memory:** Projects have built-in memory across all chats in the project
- **Project-only memory:** Available at project creation — isolates context to within the project only

## Setup — Step by Step

### Step 1: Create the Project
1. In ChatGPT sidebar, click **New project**
2. Name it "Canompx3 ORB Trading" (or similar)
3. Pick an icon and color
4. **IMPORTANT:** When prompted for memory setting, choose **"Project-only memory"**
   - This ensures ChatGPT only draws context from this project's files and chats
   - Prevents cross-contamination from your other ChatGPT conversations

### Step 2: Set Project Instructions
1. Click the three dots (upper right) -> **Project settings**
2. Open `PROJECT_INSTRUCTIONS.md` in a text editor
3. Copy everything BELOW the header comments (starting from "You are assisting...")
4. Paste into the **Project instructions** field
5. This is ~3,900 of the 8,000 character limit — always shapes every response

### Step 3: Upload Files (9 files, in two batches)

**Batch 1 — Project docs (3 files):**
| File | Size | What It Grounds |
|------|------|-----------------|
| `PROJECT_REFERENCE.md` | 10KB | Architecture, data flow, entry models, sessions, validation, NO-GOs, strategic direction |
| `TRADING_RULES.md` | 44KB | Complete trading rules, cost models, edge summary, calendar effects, session playbook |
| `RESEARCH_RULES.md` | 17KB | Statistical methodology, sample sizes, significance thresholds, mechanism test |

**Batch 2 — Academic papers + textbook extract (6 files):**
| File | Pages | What It Grounds |
|------|-------|-----------------|
| `benjamini-and-Hochberg-1995-fdr.pdf` | 13 | BH FDR correction — used directly in strategy validation |
| `deflated-sharpe.pdf` | 22 | Why grid-searched Sharpe is unreliable (Bailey Rule) |
| `false-strategy-lopez.pdf` | 7 | False Strategy Theorem — why 105K combos guarantee spurious winners |
| `backtesting_dukepeople_liu.pdf` | 17 | Harvey & Liu — p<0.005 discovery threshold |
| `Pseudo-mathematics-and-financial-charlatanism.pdf` | 32 | Backtest overfitting theory |
| `Carver_Fitting_and_VolTarget_Extracts.pdf` | 40 | Ch3: SR estimation, how much data to distinguish strategies. Ch9: Half-Kelly, vol targeting, position sizing. |

**Total: 9 files.** Uses 9 of 25 (Plus) or 9 of 40 (Pro) slots.

### Step 4 (Optional): Use Ad-Hoc Text Sources
OpenAI now supports pasting text directly as a project source (not just file uploads).
You can paste key findings or current state notes directly without using a file slot.

### DO NOT Upload
- `PROJECT_INSTRUCTIONS.md` — paste into Instructions field, not uploaded as a file
- `README.md` — for you, not for ChatGPT

## Features to Use Inside the Project

Once set up, these features enhance how ChatGPT works with your project:

- **Project-only memory:** ChatGPT remembers prior chats within the project and stays anchored to your trading context
- **Save responses:** When ChatGPT produces a useful analysis, save it as a project source for reuse
- **Web search:** ChatGPT can search the web for current market data while staying grounded in your project docs
- **Canvas:** Use for drafting research notes or code within the project context
- **Deep research:** Available on paid plans — can do extended research grounded in your project files

## Why These 9 Files

**Why not more?** With 25+ slots available, you COULD upload more. But ChatGPT searches
across all project files for relevance. More files = more search noise for any given question.
9 well-chosen files covering distinct topics (architecture, trading rules, research methodology,
5 distinct academic foundations, and practical sizing framework) give clean, focused retrieval.

**Room to grow:** You have 16+ unused file slots. Future additions could include:
- `Lopez_de_Prado_ML_for_Asset_Managers.pdf` when ML work resumes
- Updated `HANDOFF.md` snapshots for current state
- Research output CSVs for specific analyses
- New academic papers as methodology evolves

**What was considered and cut (with reasons):**
- Textbooks (Chan, Aronson, Pardo/Fitschen): Content overlaps with the academic papers + Carver extract
- STRATEGY_BLUEPRINT.md: Research test sequence already consolidated into PROJECT_REFERENCE.md
- CLAUDE.md: Claude-specific tooling (hooks, MCP, stage-gate) — irrelevant to ChatGPT
- AGENTS.md, ROADMAP.md, HANDOFF.md: Either tool-specific or too volatile

## Refresh Schedule
- `TRADING_RULES.md`: Replace after any strategy rebuild or NO-GO declaration
- `PROJECT_REFERENCE.md`: Replace after major architecture or validation changes
- `PROJECT_INSTRUCTIONS.md`: Re-paste if entry models, sessions, or statistical standards change
- PDFs: Never need refreshing
- Consider: Periodically save a ChatGPT analysis as a project source to build project memory
