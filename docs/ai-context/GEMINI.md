# Project Overview

This project is a sophisticated **Quantitative Trading System for Gold Futures (MGC)**, enhanced by an **LLM-powered Code Quality and Issue Detection Agent**.

The primary goal is to develop, test, and execute data-driven trading strategies, particularly focusing on Opening Range Breakout (ORB) patterns in Gold Futures (GC/MGC) markets. The system incorporates a robust data pipeline for ingestion and transformation, a comprehensive trading application with an execution engine and risk management, and a powerful LLM agent for continuous code quality assurance.

## Core Components:

1.  **Data Pipeline (`pipeline/`):**
    *   **Purpose:** Responsible for ingesting DBN (Databento) data, transforming 1-minute bars into 5-minute bars, building daily trading features, and ensuring database integrity through checks.
    *   **Technology:** Primarily Python-based, utilizing `databento` for data access and `DuckDB` as the embedded analytical database for efficient data storage and querying.
    *   **Orchestration:** `pipeline/run_pipeline.py` orchestrates the entire data flow through sequential steps (ingest, build_5m, build_features, audit).

2.  **Trading Application (`trading_app/`):**
    *   **Purpose:** Implements the core logic for defining, validating, and executing quantitative trading strategies. It processes market data bar-by-bar to detect trading opportunities, manage trade lifecycles, and apply risk controls.
    *   **Key Features:**
        *   **Execution Engine (`trading_app/execution_engine.py`):** A state machine that handles bar-by-bar processing, ORB detection, signal confirmation, and trade entry/exit logic for various entry models (E1: Market-On-Next-Bar, E2: Market-On-Confirm-Close, E3: Limit-At-ORB Retrace).
        *   **Strategy Management:** Includes modules for strategy discovery, validation, portfolio construction, and risk management (`trading_app/risk_manager.py`).
        *   **Backtesting/Simulation:** Capabilities for historical replay (`trading_app/paper_trader.py`) to test strategies against past data.

3.  **LLM Code Scanner (`llm-code-scanner/`):**
    *   **Purpose:** An autonomous agent designed to continuously monitor and improve the codebase quality. It leverages Large Language Models (LLMs) to identify potential issues and streamline the development workflow.
    *   **Functionality:** Scans code for bugs, performance issues, security vulnerabilities, and code quality problems.
    *   **Integration:** Integrates with JIRA for automated ticket creation for identified findings (with deduplication and summarization), and can send email reports.
    *   **Technology:** Uses `vLLM` (OpenAI API compatible) for LLM inference, `Outlines` for structured output, and `Pydantic` for schema validation.

## Key Technologies and Libraries:

*   **Language:** Python 3.13+
*   **Data Handling:** `databento`, `duckdb`, `pandas`, `numpy`, `pyarrow`, `zstandard`
*   **Testing:** `pytest`, `pytest-xdist`, `pytest-cov`
*   **Linting/Formatting:** `ruff`
*   **LLM Integration:** `OpenAI` client (compatible with `vLLM`), `Outlines`, `Pydantic`
*   **CI/CD:** GitHub Actions

## Building and Running:

The project uses a standard Python environment.

### Setup:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Canompx3
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Data Pipeline:

The `run_pipeline.py` script orchestrates the data ingestion and feature generation.

```bash
python pipeline/run_pipeline.py --instrument MGC --start YYYY-MM-DD --end YYYY-MM-DD
# Example dry run:
python pipeline/run_pipeline.py --instrument MGC --start 2024-01-01 --end 2024-12-31 --dry-run
```
Refer to `pipeline/run_pipeline.py` for full argument details.

### Running Tests:

Tests are managed with `pytest`. The CI configuration provides a good example of how to run them.

```bash
python -m pytest tests/ -v -n auto --dist loadscope --cov=pipeline --cov=trading_app --cov-report=term-missing
```
*Note: Some tests may require a populated `gold.db` or specific data, and `tests/test_trader_logic.py` is ignored in CI.*

### Running the LLM Code Scanner:

The LLM code scanner can be run as follows:

```bash
python llm-code-scanner/agent/scanner.py
```
Configuration for the scanner is located in `llm-code-scanner/config/scanner_config.yaml`.

## Development Conventions:

*   **Linting:** The `ruff` linter is used to enforce code style and identify potential issues. The configuration (`ruff.toml`) focuses on bug-catching rules (pyflakes, pycodestyle errors/warnings) and ignores certain stylistic rules.
    *   Line length is set to 120 characters.
    *   Specific rules are ignored for test files (`tests/*`).
    *   To run ruff: `ruff check pipeline/ trading_app/`
*   **Testing:** A strong emphasis is placed on comprehensive unit and integration testing using `pytest`. Shared fixtures in `tests/conftest.py` facilitate database and sample data setup.
*   **Data Integrity & Synchronization:** Critical importance is placed on data consistency and configuration synchronization. Custom drift checks (`pipeline/check_drift.py`) and specific test suites (`tests/test_app_sync.py`) are used to ensure that code, configurations, and database schemas remain aligned.
*   **Git Hooks:** A pre-commit hook is wired via `.githooks/` to ensure code quality checks are performed before commits.
*   **CI/CD:** GitHub Actions (`.github/workflows/ci.yml`) automatically run linting, drift checks, and tests on `push` and `pull_request` events to the `main` branch, targeting Python 3.13 on `windows-latest`.
*   **Trading Strategy Rules:**
    *   **Strategy Family Isolation:** Avoid cross-family inference; each strategy family is treated as an isolated memory container.
    *   **`daily_features` Usage:** Outcomes in `daily_features` are for RR=1.0 only; `strategy_discovery.py` should be used for higher RR backtesting.
    *   **ORB Size Regime Awareness:** Strategies must be periodically re-validated due to changing market volatility and ORB size regimes (e.g., Gold price level significantly impacts ORB dynamics).

## Project Roadmap & Status:

The project is mature, with most core development phases completed as detailed in `ROADMAP.md`. Key completed phases include:
*   Daily Features Pipeline
*   Cost Model
*   Full Trading Application (including Execution Engine, Risk Management, Paper Trading)
*   Database/Config Sync
*   Expanded Strategy Scan and Entry Model Fixes
*   Comprehensive Audit & Analysis

**Ongoing/Future Work (`TODO`):**

*   **Monitoring & Alerting (Phase 6e/8b):** Developing live strategy performance tracking, drift detection between live and backtest results, and alerting mechanisms for significant deviations (e.g., drawdown exceeding historical, win rate divergence, ORB size regime shifts).
*   **`orb_outcomes` Backfill (Phase 8c):** Extending the historical data coverage for `orb_outcomes` back to 2016-2020 to enable 10-year strategy validation.

This `GEMINI.md` serves as a comprehensive guide to the project's architecture, functionality, and development practices.