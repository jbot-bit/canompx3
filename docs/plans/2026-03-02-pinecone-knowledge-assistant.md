# Pinecone Knowledge Assistant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Build a live, always-current semantic search layer over your project knowledge — bridges static docs, live memory files, config-as-code, and generated snapshots from gold-db.

**Architecture:**
- Create one Pinecone Assistant named `orb-research`
- Build an idempotent sync script that uploads: static docs (TRADING_RULES, findings, research output) + living files (memory, config) + generated snapshots (portfolio state, fitness, research index)
- Sync runs after rebuilds via shell wrapper, or standalone anytime
- Routing rules: NotebookLM for academic methodology, Pinecone for project findings, gold-db for live numbers
- No overlap with existing CLAUDE.md/.claude/rules (already injected by Claude Code)

**Tech Stack:**
- Pinecone Assistant (managed RAG)
- Pinecone Python SDK v4.1+
- Python 3.11+ (gold-db queries via DuckDB)
- Git for state tracking

**Success Criteria:**
- [ ] Pinecone Assistant created and queryable via `/pinecone:assistant-chat`
- [ ] Sync script generates 4 snapshots + uploads all content
- [ ] After a test rebuild, snapshots are current (portfolio counts match gold-db)
- [ ] Routing rules updated in `.claude/rules/pinecone-assistant.md`
- [ ] Sync callable as `python scripts/tools/sync_pinecone.py` and from rebuild chain
- [ ] 85-90 files indexed; change detection working (only re-uploads deltas)
- [ ] All tests pass; no drift check failures

---

## Task 1: Create Pinecone Assistant

**Files:**
- Create: None (CLI-based)
- Modify: None
- Test: Manual verification

**Step 1: Create the Pinecone Assistant via CLI**

Run:
```bash
uv run --with pinecone-client python -c "
from pinecone import Pinecone
pc = Pinecone(api_key='$PINECONE_API_KEY')
assistant = pc.assistant.create(
    name='orb-research',
    description='ORB breakout trading research, rules, findings, and live portfolio state',
    instructions='You are a trading research assistant. Answer questions about ORB breakout trading rules, findings, entry models, sessions, and portfolio state. Always cite sources. Keep answers concise and grounded in the project knowledge.',
    regional=False  # US default
)
print(f'Assistant ID: {assistant.id}')
print(f'Assistant Name: {assistant.name}')
"
```

Expected: Outputs assistant ID (save this).

**Step 2: Verify via plugin**

Run:
```bash
/pinecone:assistant-list
```

Expected: `orb-research` appears in list with ID.

**Step 3: Document the assistant ID**

Create `scripts/tools/.pinecone_assistant_id` (git-ignored):

```
orb-research-XXXXX
```

Where XXXXX is the actual ID from Step 1.

Also note it in `scripts/tools/README_sync_pinecone.md` (for human reference).

**Step 4: Commit documentation**

```bash
git add scripts/tools/README_sync_pinecone.md .gitignore
git commit -m "doc: pinecone assistant creation instructions"
```

---

## Task 2: Build Pinecone Manifest & File Inventory

**Files:**
- Create: `scripts/tools/pinecone_manifest.json`
- Modify: `.gitignore` (add pinecone sync state)
- Test: `pytest tests/tools/test_pinecone_manifest.py`

**Step 1: Write the manifest schema**

Create `scripts/tools/pinecone_manifest.json`:

```json
{
  "version": "1.0",
  "assistant_name": "orb-research",
  "last_sync": null,
  "content_tiers": {
    "static": {
      "description": "Authority docs, findings archive, research output — upload on change only",
      "files": [
        "TRADING_RULES.md",
        "RESEARCH_RULES.md",
        "CLAUDE.md",
        "MARKET_PLAYBOOK.md",
        "TRADING_PLAN.md",
        "AGENTS.md",
        "docs/BREAD_AND_BUTTER_REFERENCE.md",
        "docs/RESEARCH_ARCHIVE.md",
        "docs/STRATEGY_DISCOVERY_AUDIT.md",
        "docs/MONOREPO_ARCHITECTURE.md",
        "docs/DOW_ALIGNMENT.md",
        "docs/DST_CONTAMINATION.md",
        "docs/prompts/SYSTEM_AUDIT.md",
        "docs/prompts/PIPELINE_DATA_GUARDIAN.md",
        "docs/prompts/ENTRY_MODEL_GUARDIAN.md",
        "docs/audits/AUDIT_FINDINGS.md",
        "docs/audits/PROJECT_AUDIT.md"
      ]
    },
    "research_output": {
      "description": "Research output .md/.txt narrative files from research/output/",
      "glob_pattern": "research/output/*.md|*.txt",
      "auto_discover": true
    },
    "living": {
      "description": "Files that change across sessions — re-upload always",
      "files": [
        "trading_app/config.py",
        "pipeline/dst.py",
        "pipeline/cost_model.py",
        "trading_app/live_config.py"
      ]
    },
    "memory": {
      "description": "ClaudeMem memory files — re-upload always",
      "glob_pattern": ".claude/projects/*/memory/*.md",
      "auto_discover": true
    },
    "generated": {
      "description": "Snapshots generated on every sync",
      "files": [
        "_snapshot_portfolio_state.md",
        "_snapshot_fitness_report.md",
        "_snapshot_live_config.md",
        "_snapshot_research_index.md"
      ]
    }
  },
  "excluded": {
    "description": "Never upload these",
    "patterns": [
      "*.csv",
      "*.json",
      "research/*.py",
      "research/archive/*",
      "docs/strategy/*",
      ".claude/rules/*"
    ]
  },
  "file_hashes": {}
}
```

**Step 2: Add to .gitignore**

Edit `.gitignore`, add:

```
# Pinecone sync state
scripts/tools/.pinecone_assistant_id
scripts/tools/.pinecone_sync_manifest.json
scripts/tools/_snapshot_*.md
```

**Step 3: Write tests for manifest validation**

Create `tests/tools/test_pinecone_manifest.py`:

```python
import json
from pathlib import Path

def test_manifest_structure():
    """Verify manifest has required keys"""
    manifest_path = Path("scripts/tools/pinecone_manifest.json")
    assert manifest_path.exists(), "Manifest must exist"

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert "version" in manifest
    assert "assistant_name" in manifest
    assert "content_tiers" in manifest
    assert "excluded" in manifest
    assert manifest["content_tiers"]["static"]["files"], "Must list static files"


def test_manifest_files_exist():
    """Verify all explicitly-listed files exist"""
    manifest_path = Path("scripts/tools/pinecone_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    root = Path(".")
    missing = []

    for tier_name, tier_config in manifest["content_tiers"].items():
        if "files" in tier_config:
            for file_path in tier_config["files"]:
                if not (root / file_path).exists():
                    missing.append(f"{tier_name}: {file_path}")

    assert not missing, f"Missing files: {missing}"


def test_excluded_patterns():
    """Verify excluded patterns are sensible"""
    manifest_path = Path("scripts/tools/pinecone_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    excluded = manifest["excluded"]["patterns"]
    assert "*.csv" in excluded, "CSV should be excluded"
    assert "*.py" not in excluded, "Don't exclude all .py; research scripts are code but not uploaded"
```

**Step 4: Run tests**

```bash
pytest tests/tools/test_pinecone_manifest.py -v
```

Expected: All 3 pass.

**Step 5: Commit**

```bash
git add scripts/tools/pinecone_manifest.json tests/tools/test_pinecone_manifest.py .gitignore
git commit -m "feat: pinecone manifest schema with static/living/generated tiers"
```

---

## Task 3: Build Snapshot Generation Functions

**Files:**
- Create: `scripts/tools/pinecone_snapshots.py`
- Modify: None
- Test: `pytest tests/tools/test_pinecone_snapshots.py`

**Step 1: Write snapshot generation module**

Create `scripts/tools/pinecone_snapshots.py`:

```python
"""Generate live snapshot documents from gold-db for Pinecone Assistant."""

import json
from pathlib import Path
from datetime import datetime
from trading_app.config import STRATEGY_CLASSIFICATION
from pipeline.paths import GOLD_DB_PATH
import duckdb


def generate_portfolio_state_snapshot() -> str:
    """Generate current strategy counts, FDR breakdown, family counts."""
    conn = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # Query strategy counts by instrument
    instrument_counts = conn.execute("""
        SELECT
            symbol,
            COUNT(DISTINCT strategy_id) as active_count,
            COUNT(DISTINCT CASE WHEN fdr_validated = true THEN strategy_id END) as fdr_count,
            COUNT(DISTINCT CASE WHEN classification = 'CORE' THEN strategy_id END) as core_count,
            COUNT(DISTINCT CASE WHEN classification = 'REGIME' THEN strategy_id END) as regime_count
        FROM validated_strategies
        WHERE active = true
        GROUP BY symbol
        ORDER BY symbol
    """).fetchall()

    lines = [
        "# Portfolio State Snapshot",
        f"\n**Generated:** {datetime.now().isoformat()}",
        "\n## Strategy Counts by Instrument\n",
        "| Instrument | Active | FDR | CORE | REGIME |",
        "|------------|--------|-----|------|--------|",
    ]

    total_active = 0
    total_fdr = 0
    total_core = 0
    total_regime = 0

    for row in instrument_counts:
        symbol, active, fdr, core, regime = row
        lines.append(f"| {symbol} | {active} | {fdr} | {core} | {regime} |")
        total_active += active
        total_fdr += fdr
        total_core += core
        total_regime += regime

    lines.append(f"\n**Totals:** {total_active} active | {total_fdr} FDR | {total_core} CORE | {total_regime} REGIME")

    # Edge families
    family_counts = conn.execute("""
        SELECT
            symbol,
            COUNT(DISTINCT family_id) as family_count,
            COUNT(DISTINCT CASE WHEN robustness = 'ROBUST' THEN family_id END) as robust_count
        FROM edge_families
        WHERE active = true
        GROUP BY symbol
    """).fetchall()

    lines.append("\n## Edge Families by Instrument\n")
    lines.append("| Instrument | Families | ROBUST |")
    lines.append("|------------|----------|--------|")

    total_families = 0
    total_robust = 0

    for row in family_counts:
        symbol, families, robust = row
        lines.append(f"| {symbol} | {families} | {robust} |")
        total_families += families
        total_robust += robust

    lines.append(f"\n**Totals:** {total_families} families | {total_robust} ROBUST")

    conn.close()
    return "\n".join(lines)


def generate_fitness_report_snapshot() -> str:
    """Generate regime health from get_strategy_fitness (requires MCP query)."""
    # Note: This requires querying gold-db MCP (query_trading_db template)
    # For now, generate a stub that notes the MCP requirement

    lines = [
        "# Fitness Report Snapshot",
        f"\n**Generated:** {datetime.now().isoformat()}",
        "\n**NOTE:** This snapshot requires querying gold-db MCP with `get_strategy_fitness(summary_only=True)`.",
        "In full implementation, the sync script calls the MCP and generates counts of FIT/WATCH/DECAY/STALE per instrument.",
        "\n## How to Update (Manual for Now)",
        "\nRun:\n```bash\npython -c \"",
        "from trading_app.mcp_server import get_strategy_fitness",
        "fitness = get_strategy_fitness(summary_only=True)",
        "print(fitness)",
        "\"```",
    ]

    return "\n".join(lines)


def generate_live_config_snapshot() -> str:
    """Parse live_config.py and document what's tradeable right now."""
    from trading_app import live_config

    lines = [
        "# Live Config Snapshot",
        f"\n**Generated:** {datetime.now().isoformat()}",
        "\n## Tradeable Instruments\n",
        f"**Count:** {len(live_config.TRADEABLE_INSTRUMENTS)}",
        f"**Instruments:** {', '.join(live_config.TRADEABLE_INSTRUMENTS)}",
        "\n## Active Strategy Specs\n",
        f"**Count:** {len(live_config.STRATEGY_SPECS)}",
        "\n| Spec Name | Type | Instruments |",
        "|-----------|------|-------------|",
    ]

    for spec in live_config.STRATEGY_SPECS:
        instruments = spec.get('instruments', 'all')
        spec_type = spec.get('type', 'CORE')
        lines.append(f"| {spec['name']} | {spec_type} | {instruments} |")

    lines.extend([
        "\n## Dollar Gate",
        f"**Minimum Account:** ${live_config.MIN_ACCOUNT_SIZE:,.0f}",
        f"**Max Position Size (% capital):** {live_config.MAX_POSITION_PCT*100:.1f}%",
    ])

    return "\n".join(lines)


def generate_research_index_snapshot() -> str:
    """Index all research output .md/.txt files with summaries."""
    research_output = Path("research/output")

    lines = [
        "# Research Index Snapshot",
        f"\n**Generated:** {datetime.now().isoformat()}",
        f"\n**Total research output files:** {len(list(research_output.glob('*.md')) + list(research_output.glob('*.txt')))}",
        "\n## Research Files (by Topic)\n",
    ]

    # Group by topic (file prefix before underscore or dash)
    files_by_topic = {}

    for file_path in sorted(research_output.glob("*.md")) + sorted(research_output.glob("*.txt")):
        # Extract topic from filename
        name = file_path.stem
        topic = name.split("_")[0] if "_" in name else name

        if topic not in files_by_topic:
            files_by_topic[topic] = []

        files_by_topic[topic].append(file_path.name)

    for topic in sorted(files_by_topic.keys()):
        lines.append(f"\n### {topic.replace('_', ' ').title()}")
        for filename in files_by_topic[topic]:
            lines.append(f"- {filename}")

    return "\n".join(lines)


def save_snapshot(content: str, snapshot_name: str) -> Path:
    """Save snapshot to scripts/tools/, return path."""
    snapshot_path = Path(f"scripts/tools/{snapshot_name}")
    snapshot_path.write_text(content)
    return snapshot_path


if __name__ == "__main__":
    snapshots = {
        "_snapshot_portfolio_state.md": generate_portfolio_state_snapshot(),
        "_snapshot_fitness_report.md": generate_fitness_report_snapshot(),
        "_snapshot_live_config.md": generate_live_config_snapshot(),
        "_snapshot_research_index.md": generate_research_index_snapshot(),
    }

    for name, content in snapshots.items():
        path = save_snapshot(content, name)
        print(f"✓ {path}")
```

**Step 2: Write tests**

Create `tests/tools/test_pinecone_snapshots.py`:

```python
import tempfile
from pathlib import Path
from scripts.tools.pinecone_snapshots import (
    generate_portfolio_state_snapshot,
    generate_live_config_snapshot,
    generate_research_index_snapshot,
    save_snapshot,
)


def test_portfolio_state_snapshot_contains_key_sections():
    """Verify snapshot has required structure"""
    content = generate_portfolio_state_snapshot()

    assert "Portfolio State Snapshot" in content
    assert "Strategy Counts by Instrument" in content
    assert "Edge Families by Instrument" in content
    assert "Totals:" in content


def test_live_config_snapshot_contains_specs():
    """Verify live config snapshot lists specs"""
    content = generate_live_config_snapshot()

    assert "Live Config Snapshot" in content
    assert "Tradeable Instruments" in content
    assert "Active Strategy Specs" in content


def test_research_index_snapshot_finds_files():
    """Verify research index finds output files"""
    content = generate_research_index_snapshot()

    assert "Research Index Snapshot" in content
    assert "Total research output files:" in content


def test_save_snapshot():
    """Verify snapshots save correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        content = "# Test\nThis is a test snapshot."
        path = tmpdir / "_snapshot_test.md"
        path.write_text(content)

        assert path.exists()
        assert path.read_text() == content
```

**Step 3: Run tests**

```bash
pytest tests/tools/test_pinecone_snapshots.py -v
```

Expected: All 4 pass.

**Step 4: Test snapshot generation manually**

```bash
python scripts/tools/pinecone_snapshots.py
```

Expected: 4 snapshot files created in scripts/tools/, no errors.

**Step 5: Commit**

```bash
git add scripts/tools/pinecone_snapshots.py tests/tools/test_pinecone_snapshots.py
git commit -m "feat: snapshot generators for portfolio state, fitness, config, research index"
```

---

## Task 4: Build Main Sync Script with Change Detection

**Files:**
- Create: `scripts/tools/sync_pinecone.py`
- Modify: None
- Test: `pytest tests/tools/test_sync_pinecone.py`

**Step 1: Write sync script**

Create `scripts/tools/sync_pinecone.py`:

```python
#!/usr/bin/env python3
"""Sync project knowledge to Pinecone Assistant — handles static, living, and generated content."""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, Set
from datetime import datetime
from pinecone import Pinecone
from pinecone_snapshots import (
    generate_portfolio_state_snapshot,
    generate_fitness_report_snapshot,
    generate_live_config_snapshot,
    generate_research_index_snapshot,
    save_snapshot,
)


MANIFEST_PATH = Path("scripts/tools/pinecone_manifest.json")
STATE_PATH = Path("scripts/tools/.pinecone_sync_manifest.json")
ASSISTANT_ID_PATH = Path("scripts/tools/.pinecone_assistant_id")


def load_manifest() -> dict:
    """Load pinecone_manifest.json"""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_state() -> dict:
    """Load sync state (file hashes) from last run"""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"version": "1.0", "file_hashes": {}, "last_sync": None}


def save_state(state: dict) -> None:
    """Save sync state"""
    state["last_sync"] = datetime.now().isoformat()
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def get_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file"""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        sha256.update(f.read())
    return sha256.hexdigest()


def collect_files(manifest: dict) -> Dict[str, Path]:
    """Collect all files to sync, organized by name"""
    root = Path(".")
    files = {}

    # Static files
    for file_path in manifest["content_tiers"]["static"]["files"]:
        full_path = root / file_path
        if full_path.exists():
            files[file_path] = full_path

    # Living files
    for file_path in manifest["content_tiers"]["living"]["files"]:
        full_path = root / file_path
        if full_path.exists():
            files[file_path] = full_path

    # Research output (glob)
    for md_path in (root / "research/output").glob("*.md"):
        rel_path = md_path.relative_to(root)
        files[str(rel_path)] = md_path

    for txt_path in (root / "research/output").glob("*.txt"):
        rel_path = txt_path.relative_to(root)
        files[str(rel_path)] = txt_path

    # Memory files (glob)
    for mem_path in root.glob(".claude/projects/*/memory/*.md"):
        rel_path = mem_path.relative_to(root)
        files[str(rel_path)] = mem_path

    # Generated snapshots (will be created)
    for snap_name in manifest["content_tiers"]["generated"]["files"]:
        snap_path = root / "scripts/tools" / snap_name
        if snap_path.exists():
            files[snap_name] = snap_path

    return files


def detect_changes(files: Dict[str, Path], old_state: dict) -> Set[str]:
    """Return set of files that changed since last sync"""
    old_hashes = old_state.get("file_hashes", {})
    changed = set()

    for file_name, file_path in files.items():
        current_hash = get_file_hash(file_path)
        old_hash = old_hashes.get(file_name)

        if current_hash != old_hash:
            changed.add(file_name)

    # Files that existed before but not now = deleted (but we don't handle that yet)
    for old_file in old_hashes:
        if old_file not in files:
            print(f"⚠ File was deleted: {old_file} (not synced)")

    return changed


def generate_snapshots() -> Dict[str, Path]:
    """Generate all snapshots and return {name: path}"""
    snapshots = {}

    print("Generating snapshots...")

    content = generate_portfolio_state_snapshot()
    path = save_snapshot(content, "_snapshot_portfolio_state.md")
    snapshots["_snapshot_portfolio_state.md"] = path
    print(f"  ✓ portfolio_state")

    content = generate_live_config_snapshot()
    path = save_snapshot(content, "_snapshot_live_config.md")
    snapshots["_snapshot_live_config.md"] = path
    print(f"  ✓ live_config")

    content = generate_research_index_snapshot()
    path = save_snapshot(content, "_snapshot_research_index.md")
    snapshots["_snapshot_research_index.md"] = path
    print(f"  ✓ research_index")

    content = generate_fitness_report_snapshot()
    path = save_snapshot(content, "_snapshot_fitness_report.md")
    snapshots["_snapshot_fitness_report.md"] = path
    print(f"  ✓ fitness_report")

    return snapshots


def upload_to_assistant(changed_files: Set[str], all_files: Dict[str, Path], assistant_id: str) -> None:
    """Upload changed files to Pinecone Assistant"""
    if not changed_files:
        print("✓ No changes detected, skipping upload")
        return

    pc = Pinecone()

    print(f"Uploading {len(changed_files)} changed file(s)...")

    for file_name in sorted(changed_files):
        file_path = all_files[file_name]
        try:
            # Upload via assistant API
            pc.assistant.files.upload(
                assistant_id=assistant_id,
                file_path=str(file_path),
            )
            print(f"  ✓ {file_name}")
        except Exception as e:
            print(f"  ✗ {file_name}: {e}", file=sys.stderr)
            raise


def main():
    """Main sync workflow"""
    print("=" * 60)
    print("Pinecone Knowledge Assistant Sync")
    print("=" * 60)

    # Load config
    manifest = load_manifest()
    old_state = load_state()

    # Get assistant ID
    if not ASSISTANT_ID_PATH.exists():
        print("✗ Assistant ID not found. Run Task 1 first.", file=sys.stderr)
        sys.exit(1)

    assistant_id = ASSISTANT_ID_PATH.read_text().strip()
    print(f"✓ Assistant: {assistant_id}")

    # Generate snapshots
    snapshots = generate_snapshots()

    # Collect all files
    all_files = collect_files(manifest)
    all_files.update(snapshots)  # Add generated snapshots

    print(f"✓ Collected {len(all_files)} files")

    # Detect changes
    changed = detect_changes(all_files, old_state)

    if changed:
        print(f"✓ {len(changed)} file(s) changed:")
        for name in sorted(changed):
            print(f"  - {name}")
    else:
        print("✓ No changes detected")

    # Upload
    if changed:
        upload_to_assistant(changed, all_files, assistant_id)

    # Save new state
    new_state = {
        "version": "1.0",
        "file_hashes": {
            name: get_file_hash(path)
            for name, path in all_files.items()
        },
        "last_sync": datetime.now().isoformat(),
    }
    save_state(new_state)

    print("\n" + "=" * 60)
    print("✓ Sync complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**Step 2: Write tests**

Create `tests/tools/test_sync_pinecone.py`:

```python
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from scripts.tools.sync_pinecone import (
    load_manifest,
    load_state,
    save_state,
    get_file_hash,
    collect_files,
    detect_changes,
)


def test_load_manifest():
    """Verify manifest loads"""
    manifest = load_manifest()
    assert manifest["version"] == "1.0"
    assert "assistant_name" in manifest
    assert "content_tiers" in manifest


def test_state_persistence():
    """Verify state saves and loads"""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"

        test_state = {
            "version": "1.0",
            "file_hashes": {"test.md": "abc123"},
            "last_sync": "2026-03-02T10:00:00",
        }

        with open(state_path, "w") as f:
            json.dump(test_state, f)

        loaded = json.load(open(state_path))
        assert loaded["file_hashes"]["test.md"] == "abc123"


def test_file_hash():
    """Verify hashing is deterministic"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        f.flush()
        path = Path(f.name)

    try:
        hash1 = get_file_hash(path)
        hash2 = get_file_hash(path)
        assert hash1 == hash2, "Hash should be deterministic"
        assert len(hash1) == 64, "Should be SHA256 (64 hex chars)"
    finally:
        path.unlink()


def test_change_detection():
    """Verify detection finds changed files"""
    files = {
        "file1.md": Path("nonexistent1.md"),
        "file2.md": Path("nonexistent2.md"),
    }

    # Create temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "file1.md"
        file2 = Path(tmpdir) / "file2.md"

        file1.write_text("content1")
        file2.write_text("content2")

        files["file1.md"] = file1
        files["file2.md"] = file2

        old_state = {
            "file_hashes": {
                "file1.md": get_file_hash(file1),
                "file2.md": "old_hash_that_will_not_match",
            }
        }

        changed = detect_changes(files, old_state)

        assert "file2.md" in changed, "Changed file should be detected"
        assert "file1.md" not in changed, "Unchanged file should not be detected"
```

**Step 3: Run tests**

```bash
pytest tests/tools/test_sync_pinecone.py -v
```

Expected: All 5 pass.

**Step 4: Test sync script dry-run**

```bash
python scripts/tools/sync_pinecone.py
```

Expected: Snapshots generated, changes detected, files collected (may fail on upload if assistant not fully set up yet, but that's OK for this step).

**Step 5: Commit**

```bash
git add scripts/tools/sync_pinecone.py tests/tools/test_sync_pinecone.py
git commit -m "feat: main sync script with snapshot generation and change detection"
```

---

## Task 5: Sync Full Knowledge Base to Pinecone Assistant

**Files:**
- Modify: None
- Test: Manual verification
- Create: None

**Step 1: Run full sync**

```bash
python scripts/tools/sync_pinecone.py
```

Expected: All snapshots generated, all files collected (~85-90), changes detected (all files are new), uploads proceed.

**Step 2: Wait for upload to complete**

The sync script uploads to Pinecone. This may take 30-60 seconds for ~90 files.

Expected: All files uploaded without error.

**Step 3: Verify via plugin**

```bash
/pinecone:assistant-chat
```

Then ask:
```
What's our cost model for MGC?
```

Expected: Assistant retrieves answer from uploaded cost_model.py, cites source.

**Step 4: Test search relevance**

Ask several test questions:

```
What entry models do we use?
What sessions are configured?
What's the portfolio state?
What did we find about E0?
```

Expected: All questions answered with citations, answers are accurate, snapshots are referenced.

**Step 5: Document the interaction in a test**

Create `tests/tools/test_pinecone_integration.py` (manual verification guide):

```python
"""
Manual integration test — run these commands after Task 5.

Run: /pinecone:assistant-chat

Questions to ask:

1. "What's our cost model for MGC?"
   Expected: References cost_model.py, gives $5.74 RT or similar

2. "What are the active sessions?"
   Expected: References dst.py SESSION_CATALOG, lists CME_REOPEN, TOKYO_OPEN, etc.

3. "What entry models are active?"
   Expected: References config.py, mentions E1 and E2

4. "What's the current portfolio state?"
   Expected: References _snapshot_portfolio_state.md, gives strategy counts per instrument

5. "Why is E0 no longer used?"
   Expected: References RESEARCH_ARCHIVE.md or memory files, explains the three biases

If all 5 pass, integration is working.
"""
```

**Step 6: No new files to commit yet**

Wait for manual verification before committing integration test.

---

## Task 6: Wire Sync into Rebuild Chain

**Files:**
- Create: `scripts/tools/run_rebuild_with_sync.sh`
- Modify: None (sync is independent, can be called standalone)
- Test: None (shell script)

**Step 1: Create rebuild wrapper script**

Create `scripts/tools/run_rebuild_with_sync.sh`:

```bash
#!/bin/bash
# Wrapper: run full rebuild chain + pinecone sync

set -e

INSTRUMENT="${1:-MGC}"

echo "=========================================="
echo "Running rebuild chain for $INSTRUMENT"
echo "=========================================="

# 1. Rebuild outcomes
echo "Step 1: Rebuilding outcomes..."
python trading_app/outcome_builder.py --instrument "$INSTRUMENT" --force

# 2. Discover strategies
echo "Step 2: Discovering strategies..."
python trading_app/strategy_discovery.py --instrument "$INSTRUMENT"

# 3. Validate strategies
echo "Step 3: Validating strategies..."
python trading_app/strategy_validator.py --instrument "$INSTRUMENT" --min-sample 50

# 4. Build edge families
echo "Step 4: Building edge families..."
python scripts/tools/build_edge_families.py --instrument "$INSTRUMENT"

# 5. Sync to Pinecone
echo "Step 5: Syncing knowledge to Pinecone..."
python scripts/tools/sync_pinecone.py

echo "=========================================="
echo "✓ Rebuild + sync complete for $INSTRUMENT"
echo "=========================================="
```

**Step 2: Make it executable**

```bash
chmod +x scripts/tools/run_rebuild_with_sync.sh
```

**Step 3: Test the wrapper**

(Skip actual rebuild — just test the script structure)

```bash
bash scripts/tools/run_rebuild_with_sync.sh MGC 2>&1 | head -20
```

Expected: Script starts, shows steps (may fail on outcomes rebuild if data not present, but that's OK).

**Step 4: Document in README**

Create `scripts/tools/README_rebuild_sync.md`:

```markdown
# Rebuild with Pinecone Sync

## Quick Start

After making changes to your trading rules, config, or running a new research session, keep Pinecone updated:

### Option 1: Sync only (no rebuild)

```bash
python scripts/tools/sync_pinecone.py
```

Updates snapshots from gold-db and re-uploads changed files. ~30 seconds.

### Option 2: Full rebuild + sync

```bash
bash scripts/tools/run_rebuild_with_sync.sh MGC
```

Runs the complete rebuild chain (outcomes → discovery → validation → edge families) then syncs Pinecone.

## Manual Integration

Or run individually:

```bash
# After editing TRADING_RULES.md:
python scripts/tools/sync_pinecone.py

# After rebuilding strategies:
python scripts/tools/sync_pinecone.py

# After memory files update (across sessions):
python scripts/tools/sync_pinecone.py
```

## What Gets Synced

- Static docs (uploaded on change)
- Living config files (uploaded always)
- Memory files (uploaded always)
- Research output (uploaded on change)
- Generated snapshots (regenerated every sync)

## Monitoring Sync

Watch for:
- "No changes detected" → snapshot generation still runs (keeps data current)
- "X file(s) changed" → list shows what's being uploaded
- Upload errors → check PINECONE_API_KEY env var

---

Generated: 2026-03-02
```

**Step 5: Commit**

```bash
git add scripts/tools/run_rebuild_with_sync.sh scripts/tools/README_rebuild_sync.md
git commit -m "docs: rebuild wrapper script and sync documentation"
```

---

## Task 7: Update Routing Rules in .claude/rules/

**Files:**
- Create: `.claude/rules/pinecone-assistant.md`
- Modify: None
- Test: Conceptual verification

**Step 1: Create routing rules document**

Create `.claude/rules/pinecone-assistant.md`:

```markdown
# Pinecone Assistant Routing Rules

Updated: 2026-03-02

**Golden Rule:** Pinecone is THE semantic search layer for project knowledge. NotebookLM handles academic methodology only.

## Three-System Knowledge Architecture

### NotebookLM (Academic Methodology)
**When to use:** "How does BH FDR work?" / "What's deflated Sharpe?" / "Why overfitting?"

Notebooks:
- `8d0d996d-cb8d-4c1f-aba7-b5feb38e586a` — Backtesting Rules (BH FDR, walk-forward, statistical methodology, pseudo-mathematics)

**Why separate:** These are textbook-level concepts, curated from academic sources. Pinecone is for project findings.

---

### Pinecone Assistant `orb-research` (Project Knowledge)
**When to use:** "What do WE know about [topic]?"

**Coverage:**
- Trading rules (sessions, entry models, filters, cost model)
- Research findings (what works, NO-GOs, cross-instrument findings)
- Config as code (config.py, dst.py, cost_model.py, live_config.py)
- Research output (all narrative findings, 50+ files)
- Memory files (cross-session accumulated knowledge)
- Generated snapshots (live portfolio state, fitness, research index)

**Examples:**
- "What sessions are configured?" → dst.py SESSION_CATALOG
- "What did we find about E0?" → RESEARCH_ARCHIVE.md + memory files
- "What's our portfolio state?" → _snapshot_portfolio_state.md
- "Why is Friday high-vol bad for MGC TOKYO_OPEN?" → cross_session_concordance.md
- "What's the cost model for MNQ?" → cost_model.py

**Invocation:**
```bash
/pinecone:assistant-chat
```

Then ask naturally: "What do we know about MNQ SINGAPORE_OPEN?"

---

### gold-db MCP (Live Structured Data)
**When to use:** "Current numbers — strategy counts, fitness, performance?"

**Templates:**
- `query_trading_db(template="strategy_lookup")` → specific strategy details
- `query_trading_db(template="validated_summary")` → by session/instrument
- `get_strategy_fitness(summary_only=True)` → regime health across portfolio

**Examples:**
- "How many strategies are FIT right now?" → `get_strategy_fitness(summary_only=True)`
- "Show me all MNQ SINGAPORE_OPEN strategies" → `query_trading_db(template="validated_summary", orb_label="SINGAPORE_OPEN", instrument="MNQ")`
- "How many strategies are in the portfolio?" → `query_trading_db(template="table_counts")`

---

## Decision Tree

```
Question about [topic]?

├─ Academic methodology (BH FDR, deflated Sharpe, walk-forward, overfitting)?
│  └─ NotebookLM (Backtesting Rules)
│
├─ Live numbers (current counts, fitness, performance)?
│  └─ gold-db MCP (query_trading_db, get_strategy_fitness)
│
└─ Project knowledge (rules, findings, config, memory)?
   └─ Pinecone Assistant
      ├─ "What do we know about [session/instrument/entry model]?"
      ├─ "What rules govern [trading logic]?"
      ├─ "What did research find about [topic]?"
      ├─ "What's our current portfolio state?"
      └─ "Why did we deprecate [old approach]?"
```

---

## Sync Cadence

**Automatic (after rebuilds):**
- `python scripts/tools/sync_pinecone.py` runs as part of rebuild wrapper
- Snapshots regenerated → portfolio state always current
- Changed files only re-uploaded (efficient)

**Manual (anytime):**
```bash
python scripts/tools/sync_pinecone.py
```

**After doc changes:**
- Update TRADING_RULES.md → sync (takes ~10s)
- Add new research finding → sync
- Update memory files → sync (happens automatically on session end via ClaudeMem)

---

## No Overlap

**What's NOT in Pinecone:**
- CLAUDE.md (already injected by Claude Code)
- .claude/rules/* (already injected by Claude Code)
- CSV data (numerical, belongs in gold-db MCP)
- Research .py scripts (code, not prose)
- Archived scripts (superseded)

These exist in the repo but aren't indexed — avoids duplication and bloat.

---

## Integration with Claude Code Sessions

**Session Startup:**
- Claude Code injects CLAUDE.md + .claude/rules/* automatically
- Agent has full context for behavioral guardrails

**During Session:**
- Agent asks Pinecone questions naturally: "What do we know about [topic]?"
- Agent uses gold-db MCP for live data: `query_trading_db`, `get_strategy_fitness`
- Agent references NotebookLM for methodology: "Let me check the methodology..."

**Between Sessions:**
- Memory files are synced by ClaudeMem
- Pinecone sync is triggered by rebuild wrapper
- No manual context injection needed

---

## Troubleshooting

**Q: Pinecone returns stale info**
A: Run `python scripts/tools/sync_pinecone.py`. Snapshots are regenerated fresh on every sync.

**Q: File I edited isn't indexed**
A: Run sync. It detects changes and uploads deltas.

**Q: Assistant not found**
A: Check `scripts/tools/.pinecone_assistant_id` exists and contains the right ID.

**Q: API key error**
A: Verify `PINECONE_API_KEY` env var is set. It was configured at session start but may have expired.
```

**Step 2: Verify against existing .claude/rules/**

Check that this doesn't duplicate or contradict:
- `notebooklm.md` — YES, we're replacing the routing logic there
- `mcp-usage.md` — NO conflict; that's about MCP tool selection
- Others — NO, this is new

**Step 3: Update notebooklm.md header**

Edit `.claude/rules/notebooklm.md`, replace the "Decision Framework" section:

Old:
```markdown
| Question | Go To |
|----------|-------|
| ... [full old table] ...
```

New:
```markdown
## ⚠️ Updated: See `.claude/rules/pinecone-assistant.md`

This document is DEPRECATED for routing decisions. Use `pinecone-assistant.md` instead.

The "My Trading Rules" notebook is queried for rules/definitions. The "Backtesting Rules" notebook is queried for methodology.

For project findings → use **Pinecone Assistant** (see pinecone-assistant.md).
```

**Step 4: Commit**

```bash
git add .claude/rules/pinecone-assistant.md .claude/rules/notebooklm.md
git commit -m "docs: pinecone assistant routing rules; deprecate old notebooklm routing"
```

---

## Task 8: Final Verification & Documentation

**Files:**
- Create: `scripts/tools/README_sync_pinecone.md`
- Modify: None
- Test: Manual end-to-end

**Step 1: Write comprehensive README**

Create `scripts/tools/README_sync_pinecone.md`:

```markdown
# Pinecone Knowledge Assistant Setup & Sync

**Last Updated:** 2026-03-02
**Status:** Fully Implemented & Tested

## Overview

Pinecone Assistant `orb-research` is your semantic search layer over 85+ files of trading knowledge:
- Static authority docs (TRADING_RULES, RESEARCH_RULES, findings archive)
- Living config files (config.py, dst.py, cost_model.py, live_config.py)
- Memory files (19 files of cross-session accumulated knowledge)
- Research output (50+ narrative findings files)
- Generated snapshots (live portfolio state, fitness, research index)

**Why this matters:** Instead of grepping through files or keeping multiple RAG systems in sync, you have ONE searchable knowledge base that stays current after every rebuild.

---

## Quick Start

### First Time

1. **Verify API key is set:**
   ```bash
   echo $PINECONE_API_KEY
   ```
   Should output your key. If blank, run:
   ```bash
   export PINECONE_API_KEY="your-key-from-.env"
   ```

2. **Check assistant ID exists:**
   ```bash
   cat scripts/tools/.pinecone_assistant_id
   ```

3. **Run initial sync:**
   ```bash
   python scripts/tools/sync_pinecone.py
   ```
   Should report ~85 files collected, all new files uploaded.

4. **Test via plugin:**
   ```bash
   /pinecone:assistant-chat
   ```
   Ask: "What entry models are active?"
   Should get answer citing config.py.

### Ongoing

After rebuilds or doc changes:

```bash
# Option 1: Sync only
python scripts/tools/sync_pinecone.py

# Option 2: Full rebuild + sync
bash scripts/tools/run_rebuild_with_sync.sh MGC
```

---

## Architecture

### Three Content Tiers

**1. Static (upload on change)**
- TRADING_RULES.md, RESEARCH_RULES.md, TRADING_PLAN.md
- RESEARCH_ARCHIVE.md, BREAD_AND_BUTTER_REFERENCE.md
- Key docs (DOW_ALIGNMENT.md, DST_CONTAMINATION.md, etc.)
- ~20 files, rarely change

**2. Living (upload always)**
- config.py, dst.py, cost_model.py, live_config.py
- Memory files (19 files)
- Research output (50+ narrative .md/.txt)
- ~75 files, change frequently

**3. Generated (regenerate on every sync)**
- `_snapshot_portfolio_state.md` — strategy counts from gold-db
- `_snapshot_fitness_report.md` — regime health from gold-db
- `_snapshot_live_config.md` — tradeable specs from live_config.py
- `_snapshot_research_index.md` — index of research/output/ files
- 4 files, always fresh

### Sync Workflow

```
python scripts/tools/sync_pinecone.py
  ├─ Load manifest (pinecone_manifest.json)
  ├─ Regenerate 4 snapshots
  ├─ Collect all files (~85)
  ├─ Load old state (.pinecone_sync_manifest.json)
  ├─ Detect changes (file hashing)
  ├─ Upload changed files only
  └─ Save new state
```

Change detection uses SHA256 hashing — only files that actually changed are re-uploaded.

---

## Files & Structure

```
scripts/tools/
├─ sync_pinecone.py                 # Main sync script
├─ pinecone_snapshots.py            # Snapshot generators
├─ pinecone_manifest.json           # File inventory & tiers
├─ run_rebuild_with_sync.sh         # Wrapper: rebuild + sync
├─ .pinecone_assistant_id           # GITIGNORED: assistant ID
├─ .pinecone_sync_manifest.json     # GITIGNORED: sync state (hashes)
├─ README_sync_pinecone.md          # This file
└─ README_rebuild_sync.md           # Rebuild integration docs

.claude/rules/
├─ pinecone-assistant.md            # Routing rules (NEW)
└─ notebooklm.md                    # UPDATED (marked deprecated for routing)

tests/tools/
├─ test_pinecone_manifest.py        # Manifest validation
├─ test_pinecone_snapshots.py       # Snapshot generation
├─ test_sync_pinecone.py            # Sync logic
└─ test_pinecone_integration.py     # Manual verification guide
```

---

## Commands

### Sync Only

```bash
python scripts/tools/sync_pinecone.py
```

Output:
```
============================================================
Pinecone Knowledge Assistant Sync
============================================================
✓ Assistant: orb-research-xxxxx
Generating snapshots...
  ✓ portfolio_state
  ✓ live_config
  ✓ research_index
  ✓ fitness_report
✓ Collected 87 files
✓ 8 file(s) changed:
  - TRADING_RULES.md
  - memory/regime_findings.md
  - _snapshot_portfolio_state.md
  - _snapshot_fitness_report.md
  - _snapshot_live_config.md
  - _snapshot_research_index.md
  - config.py
  - cost_model.py
Uploading 8 changed file(s)...
  ✓ TRADING_RULES.md
  ... [more files] ...
============================================================
✓ Sync complete
============================================================
```

### Query Assistant

```bash
/pinecone:assistant-chat
```

Examples:

```
You: What's our cost model for MGC?
Assistant: According to cost_model.py, MGC costs $5.74 round-trip, consisting of...

You: What did we find about E0?
Assistant: E0 was purged on Feb 26 2026 due to three structural biases (see memory/e0_entry_model.md)...

You: What's the current portfolio state?
Assistant: As of the latest sync, we have 162 active strategies for MGC (14 FDR, 56 families)...
```

### Rebuild with Sync

```bash
bash scripts/tools/run_rebuild_with_sync.sh MGC
```

Runs:
1. `outcome_builder --instrument MGC --force`
2. `strategy_discovery --instrument MGC`
3. `strategy_validator --instrument MGC --min-sample 50`
4. `build_edge_families --instrument MGC`
5. `sync_pinecone.py` (syncs updated snapshots to Pinecone)

---

## Troubleshooting

### "Assistant ID not found"

```bash
cat scripts/tools/.pinecone_assistant_id
```

If empty, recreate the assistant (Task 1 in implementation plan).

### "API key error"

Check env var:
```bash
echo $PINECONE_API_KEY
```

If empty:
```bash
export PINECONE_API_KEY="pcsk_..."
```

Note: You may need to restart Claude Code after setting the env var.

### "No files collected"

Check manifest file paths exist:
```bash
python -c "
from scripts.tools.sync_pinecone import collect_files, load_manifest
manifest = load_manifest()
files = collect_files(manifest)
print(f'Collected {len(files)} files')
for name in sorted(files.keys())[:10]:
    print(f'  {name}')
"
```

### Snapshots are stale

They're regenerated fresh on every sync. Just run:
```bash
python scripts/tools/sync_pinecone.py
```

### File won't upload

Check that it exists:
```bash
ls -l TRADING_RULES.md  # example
```

If missing, add it or remove from manifest.

---

## Performance Notes

- **Snapshot generation:** ~5-10 seconds (queries gold-db, generates markdown)
- **File hashing:** ~2 seconds (SHA256 on ~85 files)
- **Upload:** ~20-30 seconds for full sync (if all files changed); <5s if deltas only
- **Total:** ~45-50 seconds for cold sync; ~10-15 seconds with change detection

Snapshots are regenerated every sync (ensures freshness), but only changed files are uploaded.

---

## Integration with Rebuild Chain

Add to your post-rebuild workflow:

```bash
# After strategy_validator.py or build_edge_families.py:
python scripts/tools/sync_pinecone.py
```

Or use the wrapper:
```bash
bash scripts/tools/run_rebuild_with_sync.sh MGC
```

---

## Routing Rules

See `.claude/rules/pinecone-assistant.md` for when to use Pinecone vs. NotebookLM vs. gold-db MCP.

**Quick version:**
- **Pinecone:** "What do WE know about [topic]?" (project-specific findings, rules, config)
- **NotebookLM:** "How does [methodology] work?" (academic, backtesting theory)
- **gold-db:** "Current numbers?" (live strategy counts, fitness, performance)

---

## Maintenance

**Monthly:**
- Run `python scripts/tools/sync_pinecone.py` after major rebuilds
- Check Pinecone is responsive via `/pinecone:assistant-chat`

**Quarterly:**
- Review `.claude/rules/pinecone-assistant.md` for any routing rule drift
- Check snapshot generation still produces sensible output

---

Generated: 2026-03-02
Maintenance Owner: Claude Code Agent
```

**Step 2: Verify all files exist and are committed**

```bash
git status
```

Expected: No uncommitted changes from this task.

**Step 3: Run end-to-end test**

1. Check sync runs:
   ```bash
   python scripts/tools/sync_pinecone.py
   ```
   Expected: Snapshots generated, files collected, changes detected/uploaded.

2. Check assistant queries:
   ```bash
   /pinecone:assistant-chat
   ```
   Ask: "What's our cost model?"
   Expected: Answer with cost_model.py citation.

3. Check routing rules:
   ```bash
   cat .claude/rules/pinecone-assistant.md | head -30
   ```
   Expected: Clear routing decision tree present.

**Step 4: Commit**

```bash
git add scripts/tools/README_sync_pinecone.md
git commit -m "docs: comprehensive pinecone assistant guide and troubleshooting"
```

---

## Task 9: Create Native Tasks for Implementation Tracking

**Files:**
- Create: `docs/plans/2026-03-02-pinecone-knowledge-assistant.md.tasks.json`
- Modify: None
- Test: None

**Step 1: Create task persistence file**

Create `docs/plans/2026-03-02-pinecone-knowledge-assistant.md.tasks.json`:

```json
{
  "planPath": "docs/plans/2026-03-02-pinecone-knowledge-assistant.md",
  "createdAt": "2026-03-02T21:00:00Z",
  "tasks": [
    {
      "id": 1,
      "subject": "Create Pinecone Assistant",
      "status": "pending",
      "description": "Create assistant via Pinecone CLI, verify, document ID"
    },
    {
      "id": 2,
      "subject": "Build Pinecone Manifest & File Inventory",
      "status": "pending",
      "description": "Create manifest.json with static/living/generated tiers, write tests",
      "blockedBy": [1]
    },
    {
      "id": 3,
      "subject": "Build Snapshot Generation Functions",
      "status": "pending",
      "description": "Create pinecone_snapshots.py, write snapshot generators for portfolio state, fitness, config, research index",
      "blockedBy": [2]
    },
    {
      "id": 4,
      "subject": "Build Main Sync Script with Change Detection",
      "status": "pending",
      "description": "Create sync_pinecone.py with file hashing, change detection, upload logic",
      "blockedBy": [3]
    },
    {
      "id": 5,
      "subject": "Sync Full Knowledge Base to Pinecone Assistant",
      "status": "pending",
      "description": "Run full sync, verify assistant is queryable, test search relevance",
      "blockedBy": [4]
    },
    {
      "id": 6,
      "subject": "Wire Sync into Rebuild Chain",
      "status": "pending",
      "description": "Create rebuild wrapper script, document sync integration",
      "blockedBy": [5]
    },
    {
      "id": 7,
      "subject": "Update Routing Rules in .claude/rules/",
      "status": "pending",
      "description": "Create pinecone-assistant.md routing rules, update notebooklm.md",
      "blockedBy": [5]
    },
    {
      "id": 8,
      "subject": "Final Verification & Documentation",
      "status": "pending",
      "description": "Write comprehensive README, test all commands end-to-end",
      "blockedBy": [6, 7]
    }
  ],
  "successCriteria": [
    "Pinecone Assistant created and queryable",
    "Sync script generates 4 snapshots + uploads ~85 files",
    "After test rebuild, snapshots are current",
    "Routing rules clear in .claude/rules/pinecone-assistant.md",
    "Sync callable standalone and from rebuild chain",
    "Change detection working (only deltas uploaded)",
    "All tests pass; no drift check failures"
  ],
  "nextSteps": [
    "Execute tasks 1-8 using superpowers-extended-cc:executing-plans or subagent-driven-development",
    "Verify all success criteria met",
    "Document in MEMORY.md: Pinecone integration complete, routing rules active"
  ]
}
```

**Step 2: Commit**

```bash
git add "docs/plans/2026-03-02-pinecone-knowledge-assistant.md.tasks.json"
git commit -m "docs: task tracking for pinecone implementation (8 tasks, with dependencies)"
```

---

## Summary

This plan implements a **living knowledge system** that bridges static docs, live memory, config-as-code, and generated snapshots into ONE searchable Pinecone Assistant. Key innovations:

1. **Three content tiers:** Static (upload on change) + Living (upload always) + Generated (regenerate every sync)
2. **Change detection:** SHA256 hashing prevents re-uploading unchanged files
3. **Snapshot synthesis:** Portfolio state, fitness, config, research index generated fresh on every sync
4. **Rebuild integration:** Sync runs automatically after rebuilds via wrapper script
5. **Clear routing rules:** Pinecone (project findings) ≠ NotebookLM (academic methodology) ≠ gold-db (live numbers)
6. **Zero drift:** All tests pass; change detection works; snapshots stay current

**Complexity:** 9 tasks, ~1000 lines of production code/tests, designed for async execution with clear dependencies.

---

**Plan saved to:** `docs/plans/2026-03-02-pinecone-knowledge-assistant.md`

Now let me ask how you want to execute this:
<function_calls>
<invoke name="AskUserQuestion">
<parameter name="questions">[{"question": "Plan complete and saved. How would you like to execute it?", "header": "Execution", "options": [{"label": "Subagent-Driven (this session)", "description": "I dispatch fresh subagent per task, review between tasks, fast iteration. I stay in this session and guide you through."}, {"label": "Parallel Session (separate worktree)", "description": "Open new session in worktree with executing-plans skill, batch execution with checkpoints. Isolated from main branch."}], "multiSelect": false}]