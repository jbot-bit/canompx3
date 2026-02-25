# Monorepo Architecture

Documentation for the canompx3 monorepo — explains service relationships, shared resources, and development conventions across all services.

## Service Overview

The canompx3 monorepo contains six independent services that share a common database and configuration layer:

| Service | Path | Tech Stack | Purpose |
|---------|------|-----------|---------|
| **Futures Pipeline** | `pipeline/`, `trading_app/` | Python, DuckDB | Core data ingestion, feature engineering, strategy discovery & validation |
| **UI** | `ui/` | Python, Streamlit | Interactive dashboard for portfolio, strategies, and market state |
| **LLM Code Scanner** | `llm-code-scanner/` | Python, Ollama, Outlines | Autonomous code quality monitoring with LLM-powered issue detection |
| **Research** | `research/` | Python | Experimental strategy analysis and research scripts |
| **CodePilot** | `CodePilot/` | TypeScript, Next.js, Electron | Desktop development tooling |
| **openclaw** | `openclaw/` | TypeScript, Node.js | API service layer (Docker-based) |

---

## Service Details

### Futures Pipeline (Core Service)

**Location:** `pipeline/`, `trading_app/`
**Tech Stack:** Python 3.13+, DuckDB, Databento
**Database:** `gold.db` (shared)

**Purpose:**
Multi-instrument futures data pipeline for ORB (Opening Range Breakout) trading strategies. Ingests Databento DBN files, builds 1-minute and 5-minute bars, computes daily trading features, and runs strategy discovery/validation across MGC, MNQ, MES, and M2K futures.

**Key Modules:**
- `pipeline/` — Data ingestion, bar aggregation, feature engineering, validation gates
- `trading_app/` — Strategy discovery, validation, execution engine, portfolio construction, risk management
- `trading_app/nested/` — Nested ORB research (15m/30m ORB + 5m entry bars)
- `trading_app/regime/` — Regime-bounded strategy analysis
- `tests/` — Tests covering pipeline and trading app

**Run Commands:**
```bash
# Initialize database
python pipeline/init_db.py

# Ingest data
python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31

# Strategy discovery + validation
python trading_app/outcome_builder.py --instrument MGC --start 2021-02-05 --end 2026-02-04
python trading_app/strategy_discovery.py --instrument MGC
python trading_app/strategy_validator.py --instrument MGC --min-sample 50

# View results
python trading_app/view_strategies.py --summary
python pipeline/dashboard.py
```

**Dependencies:**
- Consumes: Databento DBN files (in `DB/` directory)
- Writes to: `gold.db` (bars_1m, bars_5m, daily_features, orb_outcomes, experimental_strategies, validated_setups)
- Shared config: `.env`, `ruff.toml`, `pyproject.toml`, `requirements.txt`

---

### UI (Dashboard Service)

**Location:** `ui/`
**Tech Stack:** Python, Streamlit
**Port:** 8501
**Database:** `gold.db` (read-only)

**Purpose:**
Interactive web dashboard for viewing portfolio performance, strategy metrics, market state, and data quality. Provides AI-powered chat interface for querying strategy data.

**Features:**
- Portfolio view with performance metrics
- Strategy browser and filter
- Market state visualization
- Data quality monitoring
- AI chat interface (MCP server integration)

**Run Commands:**
```bash
# Launch dashboard
streamlit run ui/app.py

# Dashboard will be available at http://localhost:8501
```

**Dependencies:**
- Reads from: `gold.db` (validated_setups, orb_outcomes, daily_features)
- Shared config: `.env` (for DUCKDB_PATH, ANTHROPIC_API_KEY)

---

### LLM Code Scanner

**Location:** `llm-code-scanner/`
**Tech Stack:** Python, vLLM, Outlines, Pydantic
**Database:** None (operates on source code)

**Purpose:**
Autonomous agent for continuous code quality monitoring. Scans codebase for bugs, performance issues, security vulnerabilities, and code quality problems using LLM inference. Integrates with JIRA for automated ticket creation and email reporting.

**Features:**
- LLM-powered code analysis
- Structured output via Outlines
- JIRA integration with deduplication
- Email reporting
- Configurable scan rules

**Run Commands:**
```bash
# Run code scanner
python llm-code-scanner/agent/scanner.py

# Configuration
# Edit llm-code-scanner/config/scanner_config.yaml
```

**Dependencies:**
- Reads from: Source code files across monorepo
- Shared config: `.env` (for API keys)

---

### Research (Experimental Service)

**Location:** `research/`
**Tech Stack:** Python
**Database:** `gold.db` (read-only, occasional writes to experimental tables)

**Purpose:**
Experimental strategy analysis scripts for alternative approaches (gap fade, double break, concretum bands, ADX filters). Operates independently from production pipeline.

**Key Areas:**
- Alternative strategy prototyping
- Statistical analysis and hypothesis testing
- Archive of completed research (in `research/archive/`)

**Run Commands:**
```bash
# Example research scripts
python research/analyze_gap_fade.py
python research/analyze_double_break.py

# Shared utilities
# research/_alt_strategy_utils.py provides common functions
```

**Dependencies:**
- Reads from: `gold.db` (daily_features, bars_1m, bars_5m)
- Shared config: `.env`, `pipeline/` modules

---

### CodePilot

**Location:** `CodePilot/`
**Tech Stack:** TypeScript, Node.js
**Port:** TBD

**Purpose:**
Development tooling and automation service.

**Run Commands:**
```bash
# (Commands to be documented based on CodePilot implementation)
```

**Dependencies:**
- Consumes: openclaw API (per project_index.json)

---

### openclaw

**Location:** `openclaw/`
**Tech Stack:** TypeScript, Node.js
**Port:** TBD

**Purpose:**
API service layer providing programmatic access to project resources.

**Run Commands:**
```bash
# (Commands to be documented based on openclaw implementation)
```

**Dependencies:**
- Provides: API consumed by CodePilot

---

## Shared Resources

### Database

**Primary Database:** `gold.db` (DuckDB)
**Location:** `<project>/gold.db` (default) or override via `DUCKDB_PATH` env var
**Access Pattern:**
- **Futures Pipeline:** Read/write (primary owner)
- **UI:** Read-only
- **Research:** Read-only (occasional experimental table writes)
- **LLM Code Scanner:** None
- **CodePilot/openclaw:** TBD

**Schema Ownership:**
- `pipeline/init_db.py` creates base schema (bars_1m, bars_5m, daily_features)
- `trading_app/db_manager.py` creates trading app schema (orb_outcomes, experimental_strategies, validated_setups, edge_families)
- `trading_app/nested/` modules handle 15m/30m ORB discovery (data stored in main orb_outcomes with orb_minutes=15/30, not separate tables)
- `trading_app/regime/schema.py` creates regime schema (regime_strategies, regime_validated tables)

**CRITICAL:** Never run two write processes against the same DuckDB file simultaneously. For long-running jobs, use scratch copy workflow:
```bash
cp gold.db C:/db/gold.db
export DUCKDB_PATH=C:/db/gold.db
# ... run job ...
cp C:/db/gold.db gold.db
```

---

### Environment Variables

**Shared via:** `.env` at project root
**Used by:** All Python services

| Variable | Used By | Purpose |
|----------|---------|---------|
| `DATABENTO_API_KEY` | Futures Pipeline | Databento API access for backfills |
| `ANTHROPIC_API_KEY` | UI, LLM Code Scanner | Claude API access |
| `PROJECT_X_API_KEY` | TBD | (Purpose to be documented) |
| `TELEGRAM_BOT_TOKEN` | Futures Pipeline | Process monitoring alerts |
| `DUCKDB_PATH` | Futures Pipeline, UI, Research | Override default gold.db path |
| `SYMBOL` | Futures Pipeline | Default instrument (MGC, MNQ, MCL, MES) |
| `TZ_LOCAL` | Futures Pipeline | Local timezone (Australia/Brisbane) |

**Convention:** Services read `.env` using `python-dotenv`. Never commit `.env` to git.

---

### Configuration Files

**Shared at project root:**

| File | Scope | Purpose |
|------|-------|---------|
| `.env` | All Python services | Environment variables (git-ignored) |
| `ruff.toml` | All Python code | Linter configuration (pyflakes, pycodestyle) |
| `pyproject.toml` | All Python services | Project metadata, tool configs |
| `requirements.txt` | All Python services | Python dependencies |
| `conftest.py` | All Python tests | Shared pytest fixtures |

**Convention:** Changes to shared config files affect all services. Run full test suite before committing.

---

### Development Conventions

**Linting:**
- Tool: `ruff`
- Config: `ruff.toml`
- Rules: pyflakes + pycodestyle errors/warnings
- Line length: 120 characters
- Run: `ruff check pipeline/ trading_app/ research/ ui/ llm-code-scanner/`

**Testing:**
- Framework: `pytest`
- Config: `pyproject.toml`, `conftest.py`
- Coverage: `pytest-cov`
- Run: `python -m pytest tests/ -v` (futures pipeline tests)
- CI: GitHub Actions (`.github/workflows/ci.yml`)

**Pre-Commit Hooks:**
- Location: `.githooks/pre-commit`
- Setup: `git config core.hooksPath .githooks`
- Runs: ruff lint + drift check + fast tests + syntax validation

**Drift Detection:**
- Tool: `pipeline/check_drift.py`
- Checks: 34 static analysis rules (hardcoded symbols, import cycles, schema sync, timezone hygiene, stale session names, E0 CB guard, etc.)
- Run: `python pipeline/check_drift.py`

**Documentation:**
- Location: `docs/`
- Markdown format
- Auto-generated: `REPO_MAP.md` (via `python scripts/tools/gen_repo_map.py`)

---

## Inter-Service Dependencies

### Dependency Graph

```
Databento DBN files
  ↓
Futures Pipeline (pipeline/ → trading_app/)
  ↓
gold.db
  ↓
├─→ UI (read-only)
├─→ Research (read-only)
└─→ (CodePilot/openclaw TBD)

LLM Code Scanner → Source code (all services)

CodePilot → openclaw.api
```

### One-Way Dependency Rule

**Futures Pipeline:** `pipeline/` → `trading_app/` (never reversed)

This is enforced by `pipeline/check_drift.py` to prevent circular dependencies. `trading_app/` can import from `pipeline/`, but `pipeline/` must never import from `trading_app/`.

### API Contracts

**openclaw → CodePilot:**
- Per `project_index.json`, CodePilot consumes openclaw API
- API contract: (To be documented)

---

## Running the Full Stack

### Development Mode

**Minimal setup (Futures Pipeline only):**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env  # Edit with your API keys

# 3. Initialize database
python pipeline/init_db.py

# 4. Run pipeline
python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31
```

**Full stack (all services):**
```bash
# Terminal 1: Futures Pipeline (one-time data load)
python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31

# Terminal 2: UI Dashboard
streamlit run ui/app.py

# Terminal 3: LLM Code Scanner (optional)
python llm-code-scanner/agent/scanner.py

# Terminal 4: CodePilot (TBD)
# Terminal 5: openclaw (TBD)
```

### Production Mode

(To be documented based on deployment strategy)

---

## Service-Specific Development

### When to Use Each Service

| Use Case | Service |
|----------|---------|
| Data ingestion, bar building, feature engineering | Futures Pipeline (`pipeline/`) |
| Strategy discovery, validation, backtesting | Futures Pipeline (`trading_app/`) |
| View portfolio, browse strategies, market state | UI |
| Code quality audits, issue detection | LLM Code Scanner |
| Experimental strategy prototyping | Research |
| API access to project resources | openclaw |
| Development automation | CodePilot |

### Isolation vs Integration

**Isolated Services (can run standalone):**
- LLM Code Scanner (operates on source code only)
- Research (read-only DB access, independent experiments)

**Integrated Services (require gold.db):**
- Futures Pipeline (primary owner)
- UI (read-only consumer)

**Unknown Integration:**
- CodePilot (requires openclaw)
- openclaw (TBD)

---

## Common Workflows

### Adding a New Instrument

1. Add instrument config to `pipeline/asset_configs.py`
2. Add cost model to `pipeline/cost_model.py`
3. Run ingestion: `python pipeline/ingest_dbn.py --instrument <SYMBOL>`
4. Run pipeline: `python pipeline/run_pipeline.py --instrument <SYMBOL>`
5. Build outcomes: `python trading_app/outcome_builder.py --instrument <SYMBOL>`
6. Run discovery: `python trading_app/strategy_discovery.py --instrument <SYMBOL>`
7. Validate strategies: `python trading_app/strategy_validator.py --instrument <SYMBOL>`

### Running Tests Across Services

```bash
# Futures Pipeline tests (comprehensive)
python -m pytest tests/ -v

# Drift checks (all Python code)
python pipeline/check_drift.py

# Linting (all services)
ruff check pipeline/ trading_app/ research/ ui/ llm-code-scanner/

# Full CI simulation
ruff check . && python pipeline/check_drift.py && python -m pytest tests/ -x -q
```

### Updating Shared Configuration

**Before changing shared config files:**
1. Run full test suite: `pytest tests/ -v`
2. Run drift checks: `python pipeline/check_drift.py`
3. Test across services (futures pipeline, UI, research)
4. Update this documentation if adding new conventions

**Files requiring cross-service testing:**
- `ruff.toml` — affects all Python linting
- `requirements.txt` — affects all Python imports
- `.env` (template) — affects all service configurations
- `conftest.py` — affects all pytest runs

---

## Troubleshooting

### Database Lock Errors

**Symptom:** `database is locked` error from DuckDB

**Cause:** Two processes writing to `gold.db` simultaneously

**Solution:**
1. Identify running processes: `ps aux | grep python | grep "pipeline\|trading_app"`
2. Kill competing process or wait for completion
3. Use scratch copy workflow for long jobs (see Shared Resources → Database)

### Import Errors Across Services

**Symptom:** `ModuleNotFoundError` when running scripts

**Cause:** Python path not set, or circular imports

**Solution:**
1. Run from project root: `python -m <module>`
2. Check for circular imports: `python pipeline/check_drift.py` (checks import cycles)
3. Verify one-way dependency: `pipeline/` → `trading_app/` (never reversed)

### Environment Variable Not Found

**Symptom:** `KeyError` or `None` value for env var

**Cause:** `.env` file missing or not loaded

**Solution:**
1. Verify `.env` exists at project root
2. Check script loads dotenv: `from dotenv import load_dotenv; load_dotenv()`
3. Confirm variable is set: `echo $VARIABLE_NAME` (bash) or `os.getenv("VARIABLE_NAME")` (Python)

---

## Migration Notes

### From Single Service to Monorepo

This monorepo evolved from a single futures pipeline service. Historical commits before 2026-02-21 may not reflect the current multi-service structure.

**Key Architectural Decisions:**
- Shared `gold.db` to avoid data duplication
- Shared `.env` for cross-service API key management
- Shared linting/testing config for consistency
- One-way dependency (`pipeline/` → `trading_app/`) to prevent cycles

**Known Limitations:**
- CodePilot and openclaw documentation incomplete (symlinked from parent directory)
- No centralized service orchestration (manual terminal-per-service currently)
- No Docker/containerization (local development only)

---

## Document Maintenance

This document is manually maintained. Update when:
- Adding a new service to the monorepo
- Changing shared resource conventions (env vars, config files)
- Modifying inter-service dependencies
- Adding/removing run commands

**Last Updated:** 2026-02-21
**Authority:** See `CLAUDE.md` for conflict resolution (CLAUDE.md wins for code structure; this doc wins for service integration)
