"""Multi-instrument live session runner.

Runs one SessionOrchestrator per instrument concurrently via asyncio.gather().
Each orchestrator is fully independent — own feed, engine, ORB builder, monitor.

Usage:
    runner = MultiInstrumentRunner(["MGC", "MNQ", "MES"], ...)
    asyncio.run(runner.run())
    runner.post_session()
"""

import asyncio
import logging
from pathlib import Path

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from trading_app.portfolio import build_profile_portfolio

from .session_orchestrator import SessionOrchestrator

log = logging.getLogger(__name__)

# Stop-file path — must match data_feed modules
_STOP_FILE = Path(__file__).parent.parent.parent / "live_session.stop"


class MultiInstrumentRunner:
    """Run signal-only sessions for multiple instruments concurrently.

    Creates one SessionOrchestrator per instrument. All run in the same
    async event loop via asyncio.gather(). Stop-file is cleaned up here
    after all feeds exit (feeds no longer delete it themselves).
    """

    def __init__(
        self,
        instruments: list[str] | None = None,
        broker: str | None = None,
        demo: bool = True,
        signal_only: bool = True,
        account_id: int = 0,
        force_orphans: bool = False,
        profile_id: str | None = None,
    ):
        if instruments is None:
            if profile_id is not None:
                from trading_app.prop_profiles import ACCOUNT_PROFILES

                from trading_app.prop_profiles import effective_daily_lanes

                profile = ACCOUNT_PROFILES[profile_id]
                instruments = sorted({lane.instrument for lane in effective_daily_lanes(profile)})
            else:
                instruments = list(ACTIVE_ORB_INSTRUMENTS)

        self.instruments = instruments
        self.orchestrators: dict[str, SessionOrchestrator] = {}
        failed: list[str] = []

        for inst in instruments:
            try:
                portfolio = None
                if profile_id is not None:
                    portfolio = build_profile_portfolio(profile_id, instrument=inst)

                orch = SessionOrchestrator(
                    instrument=inst,
                    broker=broker,
                    demo=demo,
                    signal_only=signal_only,
                    account_id=account_id,
                    force_orphans=force_orphans,
                    portfolio=portfolio,
                )
                self.orchestrators[inst] = orch
                log.info(
                    "Orchestrator ready: %s (%d strategies)",
                    inst,
                    len(orch.portfolio.strategies),
                )
            except Exception as e:
                log.error("Failed to create orchestrator for %s: %s", inst, e)
                failed.append(inst)

        if not self.orchestrators:
            raise RuntimeError(f"No orchestrators created. Failed instruments: {failed}")

        if failed:
            log.warning(
                "Proceeding without: %s (%d/%d instruments active)",
                failed,
                len(self.orchestrators),
                len(instruments),
            )

        total = sum(len(o.portfolio.strategies) for o in self.orchestrators.values())
        log.info(
            "Multi-instrument runner: %d instruments, %d total strategies",
            len(self.orchestrators),
            total,
        )

    async def run(self) -> None:
        """Run all orchestrators concurrently. Returns when all finish."""
        tasks: dict[str, asyncio.Task] = {}
        for inst, orch in self.orchestrators.items():
            tasks[inst] = asyncio.create_task(
                self._run_one(inst, orch),
                name=f"orch-{inst}",
            )

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Report per-instrument results
        failures = 0
        failed_instruments: list[str] = []
        for inst, result in zip(tasks.keys(), results, strict=True):
            if isinstance(result, Exception):
                log.error("Orchestrator %s failed: %s", inst, result)
                failures += 1
                failed_instruments.append(inst)
            else:
                log.info("Orchestrator %s completed cleanly", inst)

        # Partial failure: notify operator via surviving orchestrators
        if 0 < failures < len(tasks):
            msg = (
                f"PARTIAL SESSION FAILURE: {failures}/{len(tasks)} orchestrators crashed "
                f"({', '.join(failed_instruments)}). Surviving instruments continued. "
                f"CHECK POSITIONS on failed instruments."
            )
            log.critical(msg)
            for inst, orch in self.orchestrators.items():
                if inst not in failed_instruments:
                    try:
                        orch._notify(msg)
                    except Exception:
                        pass  # best-effort notification
                    break  # one notification is enough

        # Clean up stop file after ALL feeds have exited
        _STOP_FILE.unlink(missing_ok=True)

        # Fail-closed: if ALL orchestrators crashed, surface failure to caller
        if failures == len(tasks):
            raise RuntimeError(f"All {failures} orchestrators failed — session did not complete")

    async def _run_one(self, instrument: str, orch: SessionOrchestrator) -> None:
        """Run a single orchestrator with error isolation."""
        try:
            await orch.run()
        except Exception:
            log.exception("Orchestrator %s crashed", instrument)
            raise

    def post_session(self) -> list[str]:
        """Run post-session cleanup for all instruments. Returns failed instrument list."""
        failed: list[str] = []
        for inst, orch in self.orchestrators.items():
            try:
                orch.post_session()
            except Exception:
                log.exception("Post-session failed for %s", inst)
                failed.append(inst)

        if failed:
            log.critical("Post-session FAILED for %s — CHECK POSITIONS MANUALLY", failed)

        # Final cleanup
        _STOP_FILE.unlink(missing_ok=True)
        return failed
