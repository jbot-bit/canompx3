# 2026-06-09 Industry Current-State Gap Audit

**Class:** result / current-state audit snapshot, not live truth.
**Commit audited:** `1565bc7d8929a581567e31df9023dcee9fa7eda1`.
**Scope:** security, monitoring/observability, data architecture, live controls, agent/AI wiring, code quality, infrastructure, testing, research strategy, and operating practice.
**Local limitation:** `/workspace/canompx3/gold.db` is absent in this WSL checkout, so DB-backed live/readiness conclusions are `NEED_REMOTE_EVIDENCE`, not inferred safe.

## Evidence gathered

### Repo-local checks

| Check | Result | Notes |
|---|---:|---|
| `git log --oneline -10` | PASS | Recent branch history inspected before edits. |
| `python scripts/tools/session_preflight.py` | WARN | `core.hooksPath` was unset at session start; fixed locally with `git config core.hooksPath .githooks`. Active stage-file warnings remain. |
| `python scripts/tools/context_resolver.py --task "comprehensive audit current state security monitoring data strategies AI ML infra testing and implement smallest diff fixes" --format markdown` | WARN | No deterministic route; used fallback authority read set. |
| `python scripts/audits/run_all.py --quick` | FAIL | Phase 1 stopped: `gold.db` missing, data integrity failed, paper trader smoke failed because DB cannot open read-only. |
| Static inspection | PASS | Inspected CI, webhook/live surfaces, pyproject, authority docs, tests, and runtime docs. |

### External baselines used

- NIST SSDF SP 800-218 v1.1 — secure software development lifecycle, vulnerability response, and recurrence prevention: https://csrc.nist.gov/pubs/sp/800/218/final
- OWASP ASVS / API security themes — authentication, logging, API verification, and service hardening: https://github.com/OWASP/ASVS
- Google SRE golden signals — latency, traffic, errors, saturation: https://sre.google/sre-book/monitoring-distributed-systems/
- OpenTelemetry logs/traces/metrics correlation: https://opentelemetry.io/docs/specs/otel/logs/
- FINRA algorithmic trading supervision and controls reference (Regulatory Notice 15-09 cited by FINRA): https://www.finra.org/rules-guidance/guidance/reports/2024-finra-annual-regulatory-oversight-report/manipulative-trading
- CFTC automated trading risk controls/system safeguards concept release: https://www.cftc.gov/PressRoom/PressReleases/6683-13
- Federal Reserve SR 11-7 model risk management: https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf
- NautilusTrader architecture/risk engine: https://nautilustrader.io/docs/nightly/concepts/architecture
- QuantConnect LEAN engine/pre-trade risk docs: https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/algorithm-engine and https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/pre-trade-risk-control
- OpenBB data pipeline/provider-standardization pattern: https://openbb.co/blog/the-openbb-platform-data-pipeline

## Comparable-app readout

- **NautilusTrader** has explicit common research/sandbox/live execution semantics, component state machines, and a `RiskEngine` with pre-trade validation, exposure monitoring, and trading states. Your project has strong fail-closed doctrine and many runtime controls, but the surfaces are still spread across scripts, dashboards, docs, and live modules rather than enforced as one event-driven kernel.
- **QuantConnect LEAN** emphasizes streaming backtest/live parity, portfolio/transaction managers, brokerage models, and pre-trade checks. Your project has serious research rigor, but parity is uneven: local DB absence blocks smoke tests, and the backtest/live bridge still relies on many bespoke scripts and docs.
- **OpenBB** standardizes provider input/output with typed query/data models. Your pipeline has canonical paths/contracts and drift checks, but provider/data lineage is still mostly repo-local and DB-file-centered rather than an explicit typed data-product interface.
- **Industry trading controls** converge on real-time monitoring, independent risk controls, kill/flatten capability, pre-trade quantity/price/capital limits, audit trails, deployment validation, and post-trade surveillance. Your project has pieces of this, but not enough machine-verifiable deployment evidence in CI/local clean checkout.

## Implemented in this patch

| Implemented fix | Why it is the smallest useful diff |
|---|---|
| Added `.github/dependabot.yml` for GitHub Actions and `uv`/Python dependency update PRs. | Converts supply-chain update hygiene from memory/manual work into scheduled review. |
| Added `.github/workflows/codeql.yml`. | Adds first-party Python SAST/code-scanning without changing app runtime. |
| Added `.github/workflows/scorecard.yml`. | Adds OpenSSF repository supply-chain posture evidence as SARIF without blocking live app paths. |
| Added this audit artifact. | Makes the audit durable, citable, and aligned across Claude/Codex/shared docs. |

## Findings ranked by severity and ROI

| # | Severity | Area | What's wrong or missing | Smallest diff, best fix | Action |
|---:|---|---|---|---|---|
| 1 | Critical | Local runtime/data | Clean WSL checkout cannot run DB-backed health, integrity, or paper-trader smoke because `gold.db` is absent. That means a new agent cannot independently prove live/readiness claims. | Add a tiny sanitized fixture DB or explicit `NEED_DB` fixture lane for smoke tests, while keeping real `gold.db` local-only. | Build a deterministic fixture DB generator and make smoke tests use it when real DB is absent. |
| 2 | Critical | Capital controls | Live-readiness conclusions are fragmented across docs, drift checks, dashboard state, preflight, and handoffs; no single signed launch manifest proves broker/account/profile/data freshness/risk limits at launch. | Emit one JSON launch manifest from the live preflight engine and require the orchestrator/dashboard to show the exact manifest hash. | Extend `trading_app/live/preflight.py` to write `runtime/live_launch_manifest.json` on successful launch. |
| 3 | Critical | Monitoring | Real-time monitoring is not yet an industry-grade control plane: logs/alerts exist, but there is no universal SLO/golden-signal dashboard for order latency, feed lag, errors, saturation, and kill-switch state. | Add a read-only `/metrics` or JSON snapshot surface with feed lag, bar lag, order latency, error counts, and kill state. | Add `trading_app/live/ops_metrics.py` fed by existing runtime state. |
| 4 | Critical | Security | Webhook authentication is a shared secret in alert JSON; body secrets leak more easily into logs/screenshots/replays than header signatures. | Support `X-Webhook-Secret` / HMAC signature headers while keeping body secret only for TradingView compatibility. | Add header auth path plus tests; prefer signed timestamped header for non-TradingView callers. |
| 5 | Critical | Supply chain | CI had dependency audit advisory only and no SAST/scorecard workflow before this patch. | Add CodeQL, OpenSSF Scorecard, and Dependabot. | Done in this patch; next step is triage first generated findings. |
| 6 | Critical | Model risk | Strategy/model inventory is rich but not one SR 11-7-style inventory with owner, purpose, validation status, limitations, monitoring metrics, and decommission criteria. | Generate a model/strategy inventory from validated shelves, profiles, result docs, and runtime gates. | Create `scripts/tools/model_inventory.py` and publish `docs/runtime/model_inventory.json`. |
| 7 | Critical | Live risk | Pre-trade controls exist, but they are not presented as one independent risk engine with explicit states like ACTIVE/REDUCING/HALTED. | Introduce a tiny `TradingState` enum and central deny/allow function used by webhook/orchestrator/order routers. | Refactor only the order-submit boundary first; do not rewrite strategy code. |
| 8 | High | Observability | Logs are structured in places but not correlated with trace/run/session IDs across data ingest, research scans, preflight, broker calls, and alerts. | Add a `run_id/session_id` context helper and include it in new JSON logs. | Start with live preflight + broker order path. |
| 9 | High | Alerting | Operator alerts are file-backed and dashboard-consumable, but no explicit alert routing/escalation contract exists for critical events. | Add `docs/runbooks/alert_response.md` and a test that every critical alert category has a runbook action. | Map critical alert categories to required actions. |
| 10 | High | CI | Pyright remains advisory with hundreds of historic errors. Type unsafety is tolerated exactly where capital-routing code needs confidence. | Make pyright hard-gate only for new/changed files or a strict package slice. | Start with `trading_app/live/` or new preflight module, not whole repo. |
| 11 | High | CI/security | `pip-audit` is advisory. This is understandable for noisy transitive debt, but it means known vulnerable packages can merge by default. | Fail only on high/critical vulnerabilities after a checked-in allowlist with expiry. | Add `pip-audit --format json` parser and `security_allowlist.yaml`. |
| 12 | High | Secrets | There is no repo-visible secret-scanning workflow/policy beyond general safety norms. | Add Gitleaks/TruffleHog CI or GitHub secret scanning policy doc. | Add a non-blocking first pass, then fail on new leaks. |
| 13 | High | Deployment | No container/build artifact defines exactly what production is. Local scripts are powerful but reproducibility depends on workstation state. | Add a minimal Dockerfile or `uv run` launch contract with pinned Python and healthcheck. | Start with paper/dashboard image, not live execution. |
| 14 | High | Data | The canonical DB is a single local DuckDB file. This is simple, but it is a disaster-recovery and provenance single point of failure. | Add snapshot manifesting: schema hash, row counts, coverage dates, source DBN hashes. | Generate manifests after pipeline writes. |
| 15 | High | Data quality | Missing DB makes integrity checks fail, but the audit output says `0 check(s) failed`, which is confusing and can be misread. | Make integrity audit distinguish `failed_checks` from `audit_execution_failed`. | Patch `scripts/tools/audit_integrity.py` output contract. |
| 16 | High | Backtest/live parity | Comparable engines use one event/time model for research and live. This project has strong doctrine but many separate scripts. | Add one documented “research-to-live parity boundary” test per deployed entry model. | Start with current live profile's active lanes. |
| 17 | High | Broker reconciliation | The audit did not prove broker positions/orders reconcile against local state on every launch/heartbeat. | Add preflight item: broker truth positions == local journal expected positions before entries are allowed. | Gate new entries, allow flatten/cancel. |
| 18 | High | Kill switch | Kill switch surfaces exist, but a regulator-grade control needs independent proof it halts all order entry surfaces. | Add a test matrix that toggles kill state and asserts webhook/orchestrator/copy-router all deny entries. | Add regression tests before more live features. |
| 19 | High | Rate limits | Webhook rate limit is process-local memory. A restart resets it, and multiple processes would bypass it. | Persist rate-limit counters in a small lock-protected runtime file or broker/account-scoped state store. | Patch webhook as a bounded follow-up. |
| 20 | High | Idempotency | Webhook dedup keys omit a TradingView alert ID/timestamp, so semantically distinct same-side actions inside the window can be blocked and replay attacks outside the window are not cryptographically prevented. | Accept optional `alert_id` and signed timestamp; include it in dedup/replay checks. | Extend request model and tests. |
| 21 | High | Authorization | There is no role-based operator control surface documented for dashboard/live commands. | Split read-only dashboard, paper controls, and live controls by explicit token/role. | Start by documenting and enforcing live-only endpoints. |
| 22 | Medium | Logging | Some exceptions are fail-open for operator alert persistence. That is OK for availability but weak for forensic completeness. | Count/log persistence failures to a secondary stderr/metrics counter. | Add metric rather than raising. |
| 23 | Medium | Testing | Coverage threshold is 35%, too low for a capital-at-risk system. | Ratchet by package/slice, not global overnight. | Set live package target and increase 2-5 points per sprint. |
| 24 | Medium | Testing | Mutation/property testing deps exist, but there is no visible scheduled mutation/property gate. | Add targeted Hypothesis properties for risk math and one nightly mutation shard. | Start with position sizing and risk limits. |
| 25 | Medium | Testing | Full tests depend on local environment and skip/timeout policy complexity. | Maintain a “clean checkout, no DB” CI profile and a “DB-backed local readiness” profile. | Document and automate both lanes. |
| 26 | Medium | Agent design | AI/agent modules exist, but no central tool-permission threat model is visible for LLM-driven research/code execution. | Add agent tool trust-boundary doc and enforce no live/broker tools from research agents. | Put in `docs/governance/agent_tool_boundaries.md`. |
| 27 | Medium | AI safety | LLM proposal generation risks strategy overfitting/tunnel vision if not forced through preregistration and multiple-testing controls. | Require LLM-generated hypotheses to write prereg YAML before scans. | Add drift check for result docs with no prereg. |
| 28 | Medium | Research | The repo has strong research doctrine, but findings/profiles can still be scattered across many docs/results. | Generate status pages from canonical DB/docs instead of manual summaries. | Extend existing context/project-pulse tooling. |
| 29 | Medium | Data lineage | DBN source hashes and vendor contract details are not surfaced in one operator-facing data-quality page. | Add a coverage/lineage report with raw source hash, ingest time, schema version, row counts. | Add `scripts/tools/data_lineage_report.py`. |
| 30 | Medium | Market data | Feed-lag/heartbeat controls appear live-specific; research data staleness and source freshness need equivalent gating. | Add freshness SLA to pipeline manifests and research run headers. | Fail research runs when data freshness is outside declared scope. |
| 31 | Medium | Infrastructure | No IaC/deployment topology shows ports, tunnels, secrets, runtime users, file permissions, and backup paths. | Add a single local deployment manifest/runbook. | Create `docs/runbooks/local_live_deployment.md`. |
| 32 | Medium | Incident response | There is no compact incident response checklist for bad fill, feed dead, stuck exit, auth leak, or DB corruption. | Add one-page incident runbook linked from dashboard alerts. | Pair with alert category mapping. |
| 33 | Medium | Compliance posture | Project is personal/proprietary, but futures automation still benefits from FINRA/CFTC-inspired auditability; there is no explicit “not compliance, but controls” matrix. | Add controls matrix mapping pre-trade, monitoring, testing, validation, audit trail. | Use this audit as seed. |
| 34 | Medium | Runtime durability | File-backed runtime state needs consistent locking, rotation, corruption handling, and backup policy across all state files. | Create a small shared atomic JSONL/state writer. | Migrate alert/journal/runtime files incrementally. |
| 35 | Medium | Dashboard | Dashboards can hide risk when data is stale if every card has custom stale handling. | Centralize stale/degraded banner logic and require all live panels to consume it. | Add dashboard contract test. |
| 36 | Medium | Broker abstraction | Broker adapters exist, but conformance tests across Tradovate/ProjectX/Rithmic should be explicit for order spec, reject handling, positions, and auth refresh. | Add adapter contract test suite with fake brokers. | Start read-only/auth/positions before live order paths. |
| 37 | Medium | Performance | No explicit latency budget exists for webhook-to-broker submit path. | Add timing around contract resolution, account resolution, submit, and response. | Emit metrics/log fields first. |
| 38 | Medium | Cost | CI and drift are large and slow; recent speed work helps but no cost/SLO budget is enforced. | Track CI duration and fail or warn on regression thresholds. | Add workflow summary timing. |
| 39 | Low | Documentation | Several docs are snapshots/plans; without generated freshness stamps, humans can over-trust stale prose. | Require class/date/commit headers on audit and plan docs. | This doc follows that pattern; add lint later. |
| 40 | Low | Coding conventions | The codebase has many scripts and historical research files; import boundaries are not always obvious. | Enforce a small import-boundary drift check for `pipeline -> trading_app` only one-way dependency. | Expand existing drift checks if not already present. |
| 41 | Low | UX | Health output reports raw failures but not always the next exact command to fix local setup. | Add remediation hints to `session_preflight.py` and health checks. | Start with missing `gold.db` and inactive hooks. |
| 42 | Low | Runbooks | Operator tasks are spread across docs and scripts. | Add an index of “daily live launch,” “emergency flatten,” “data refresh,” and “research validation.” | Use docs/runbooks index. |
| 43 | Low | Repository hygiene | Generated caches (`__pycache__`) exist in working tree inventory; ensure ignored/untracked caches are not accidentally committed. | Add/verify ignore and drift check for Python cache artifacts. | Run `git status --ignored` and add guard if needed. |
| 44 | Low | Standards tracking | External baselines change; the project has no scheduled review of OWASP/NIST/OTel/trading-control standards. | Add quarterly standards-review task in action queue. | Add durable action item. |
| 45 | Low | Screenshots/visual QA | Dashboard/live UI changes should have screenshots; this audit did not change UI. | Keep screenshot requirement for perceptible web app changes. | No action in this patch. |

## Brutal bottom line

The project is stronger than a hobby bot on research discipline, fail-closed doctrine, drift checks, and live-control awareness. It is weaker than production trading platforms on unified runtime architecture, clean-checkout reproducibility, security automation, observability standards, formal model inventory, and launch evidence. The biggest real blocker is not another strategy idea; it is making safety/readiness mechanically provable from a clean environment and at live launch time.
