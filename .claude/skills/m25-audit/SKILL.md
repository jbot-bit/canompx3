---
name: m25-audit
description: Run a MiniMax M2.5 second-opinion audit with automatic triage
disable-model-invocation: true
---
Run a MiniMax M2.5 second-opinion audit with automatic triage. Smart auto-detect chooses the right mode, or pass a custom prompt for institutional-grade review.

**Prerequisite:** `MINIMAX_API_KEY` must be set in `.env` or environment. Without it, scripts silently exit 0.

Usage: `/m25-audit` (auto-detect) or `/m25-audit <custom prompt for verdict mode>`

## Step 1: Detect Mode

Determine which mode to run based on `$ARGUMENTS`:

**If `$ARGUMENTS` is empty or blank** — auto-detect:
```bash
git diff --name-only HEAD
git diff --cached --name-only
```
- If output includes any `trading_app/ml/` files → use **ML mode**
- If output includes any `pipeline/` or `trading_app/` files (but not ml/) → use **Quick mode**
- If no relevant changes → tell the user "No changed pipeline/trading_app files detected. Run `/m25-audit <your review question>` for a verdict-mode audit, or specify a file path."

**If `$ARGUMENTS` looks like a file path** (ends in `.py` and the file exists on disk) → use **Single-file mode** on that file.

**If `$ARGUMENTS` is `improvements`** → use **Improvements mode** (institutional-grade improvement suggestions).

**If `$ARGUMENTS` is a custom prompt** (anything else, >20 chars) → use **Verdict mode** with that prompt.

---

## Step 2: Run the Audit

### Quick Mode
Scan changed files with auto-detected audit modes (now includes architecture context to reduce FPs):
```bash
python scripts/tools/m25_auto_audit.py --advisory
```
Capture the full output.

### ML Mode
Full ML integration audit (self-discovers all ML files + context docs, includes architecture context):
```bash
python scripts/tools/m25_ml_audit.py
```
This takes 2-5 minutes. The output is saved to `research/output/m25_ml_audit_<timestamp>.md`.
Read the saved file to get the full results.

### Improvements Mode
Institutional-grade improvement suggestions using the `improvements` audit mode:
```bash
python scripts/tools/m25_audit.py <files> --mode improvements --output research/output/m25_improvements_$(date +%Y%m%d_%H%M).md
```
For pipeline: use key pipeline files. For ML: use `python scripts/tools/m25_ml_audit.py` instead.
For trading app: use key trading app files.

### Single-File Mode
Determine the best audit mode for the file:
- Files with SQL/JOINs (`build_daily_features.py`, `build_bars_5m.py`, `init_db.py`) → `--mode joins`
- Strategy/research files (`outcome_builder.py`, `strategy_discovery.py`, `strategy_validator.py`) → `--mode bias`
- ML files (`trading_app/ml/*.py`) → `--mode bias`
- Everything else → `--mode bugs`

```bash
python scripts/tools/m25_audit.py <filepath> --mode <detected_mode> --output research/output/m25_single_<filename>_$(date +%Y%m%d_%H%M).md
```

### Verdict Mode
For institutional-grade review with a custom prompt. Use the user's prompt as the **system prompt** (the first argument to `audit()`), and send the file contents as the user message.

```bash
python -c "
import sys, glob
sys.path.insert(0, '.')
from scripts.tools.m25_audit import load_api_key, audit, read_files
from pathlib import Path
from datetime import datetime

api_key = load_api_key()

files = [
    'CLAUDE.md',
    'RESEARCH_RULES.md',
    'TRADING_RULES.md',
]

# Conditionally add ML files if prompt mentions ML/model/classifier
prompt_lower = '''$ARGUMENTS'''.lower()
if any(kw in prompt_lower for kw in ['ml', 'model', 'classifier', 'meta-label', 'feature', 'rf', 'random forest']):
    files.extend(sorted(glob.glob('trading_app/ml/*.py')))

content = read_files(files)
system_prompt = '''$ARGUMENTS'''

result = audit(content, system_prompt, api_key, include_context=True)

ts = datetime.now().strftime('%Y%m%d_%H%M')
out_dir = Path('research/output')
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f'm25_verdict_{ts}.md'
out.write_text(f'# M2.5 Verdict\n**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n---\n\n' + result, encoding='utf-8')
print(f'Saved: {out}', file=sys.stderr)
try:
    print(result)
except UnicodeEncodeError:
    print(result.encode('ascii', errors='replace').decode('ascii'))
" 2>&1
```

---

## Step 3: Auto-Triage (Claude Code — MANDATORY after every mode)

After the M2.5 audit completes, YOU (Claude Code) MUST triage every finding. Do NOT present raw M2.5 output as final. The scripts produce raw output; the triage is your responsibility.

**For each finding M2.5 reported:**

1. **Read the actual code** at the cited line numbers. M2.5 line numbers are often off by 5-20 lines — find the real location.

2. **Trace the execution path.** Follow imports, check callers, look for upstream guards (try/except, if-checks, type constraints, assertions) that may already handle the issue.

3. **Cross-reference authority docs.** Check if the finding contradicts:
   - `CLAUDE.md` rules → finding is FALSE POSITIVE (CLAUDE.md wins)
   - `.claude/rules/` contextual rules → finding may be FALSE POSITIVE
   - Project invariants (canonical sources, fail-closed, one-way deps)

4. **Classify the finding:**
   - **TRUE** — Real issue confirmed by code reading. State what needs to change.
   - **PARTIALLY TRUE** — Real concern but existing guards mitigate it. State the residual risk.
   - **FALSE POSITIVE** — M2.5 is wrong. State WHY (existing guard, wrong line, can't see cross-file context, contradicts CLAUDE.md).
   - **WORTH EXPLORING** — Not a bug but a genuine improvement suggestion. Note as research experiment.

5. **Build the triage table:**

```markdown
| # | Finding | M2.5 Severity | Claude Verdict | Reasoning | Action |
|---|---------|---------------|----------------|-----------|--------|
| 1 | ... | CRITICAL | TRUE | [why] | [what to do] |
| 2 | ... | HIGH | FALSE POSITIVE | [why wrong] | None |
| 3 | ... | MEDIUM | WORTH EXPLORING | [why useful] | Research experiment |
```

6. **Save the triage** to `research/output/m25_triage_<YYYYMMDD_HHMM>.md` with the full table and reasoning.

7. **Present the table** to the user with a summary: X findings total, Y true, Z false positives, W worth exploring.

---

## Step 4: Downstream Effects

After triage, check if any TRUE findings trigger downstream updates:

| If TRUE finding... | Then... |
|--------------------|---------|
| Affects ML conclusions or memory | Update the relevant memory file in project memory |
| Reveals a new drift pattern | Flag: "Consider adding drift check to `check_drift.py`" |
| Affects `trading_app/ml/config.py` | Note for manual review — never auto-edit config |
| Changes research conclusions | Flag: "Run `python scripts/tools/sync_pinecone.py` after addressing" |
| Contradicts CLAUDE.md | CLAUDE.md wins. Mark FALSE POSITIVE. Do NOT update CLAUDE.md based on M2.5. |

Report any downstream flags to the user.

---

## Key Rules (from `.claude/rules/m25-audit.md`)

- **Authority:** CLAUDE.md > Claude Code > M2.5 suggestions
- **Expected FP rate:** ~40-70% depending on prompt quality. Structured prompts with "what's good + what's bad" format reduce FP rate significantly.
- **Never auto-apply:** No fix without reading code first
- **Never trust line numbers:** Always verify actual location (typically off by 5-20 lines)
- **M2.5 cannot see cross-file context:** It hallucinates about guards in files it wasn't given
- **Architecture context is now auto-prepended:** All audit modes include project architecture facts that prevent common false positives (DuckDB replacement scans, BH FDR downstream, 4-gate ML system, etc.)
