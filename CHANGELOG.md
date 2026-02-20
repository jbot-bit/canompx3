## [1.0.0] - 2026-02-20

### Added
- E0 entry model with C8/C3 exit rules and DOW (day-of-week) filters
- DoubleBreakFilter feature with 1100 session activation for discovery mode
- Calendar overlay and regime research capabilities
- Multi-day drawdown analysis and retrace timing research
- Calendar-based filters, MES history tracking, and edge structure research tools
- Double-break fix with SIL pipeline and checkpoint isolation
- Parquet export layer for improved database performance
- DST (Daylight Saving Time) remediation with discovery DST split and schema migration

### Changed
- Database performance optimizations including PRAGMA tuning and batch query processing
- Complete audit and hardening of trading app modules, scripts, and infrastructure
- Enhanced code quality with REGIME and fitness test validation
- Improved documentation consistency and research output organization

### Fixed
- Removed C8 breakeven stop from outcome builder
- Session-aware filter enforcement and migration