"""Multi-instrument live session runner.

Runs one SessionOrchestrator per instrument concurrently via asyncio.gather().
Each orchestrator is fully independent — own feed, engine, ORB builder, monitor.

Usage:
    runner = MultiInstrumentRunner(["MGC", "MNQ", "MES", "M2K"], ...)
    asyncio.run(runner.run())
    runner.post_session()
"""

import asyncio
import logging
from pathlib import Path

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

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
    ):
        if instruments is None:
            instruments = list(ACTIVE_ORB_INSTRUMENTS)

        self.instruments = instruments
        self.orchestrators: dict[str, SessionOrchestrator] = {}
        failed: list[str] = []

        for inst in instruments:
            try:
                orch = SessionOrchestrator(
                    instrument=inst,
                    broker=broker,
                    demo=demo,
                    signal_only=signal_only,
                    account_id=account_id,
                    force_orphans=force_orphans,
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
        for inst, result in zip(tasks.keys(), results, strict=True):
            if isinstance(result, Exception):
                log.error("Orchestrator %s failed: %s", inst, result)
            else:
                log.info("Orchestrator %s completed cleanly", inst)

        # Clean up stop file after ALL feeds have exited
        _STOP_FILE.unlink(missing_ok=True)

    async def _run_one(self, instrument: str, orch: SessionOrchestrator) -> None:
        """Run a single orchestrator with error isolation."""
        try:
            await orch.run()
        except Exception:
            log.exception("Orchestrator %s crashed", instrument)
            raise

    def post_session(self) -> None:
        """Run post-session cleanup for all instruments."""
        for inst, orch in self.orchestrators.items():
            try:
                orch.post_session()
            except Exception:
                log.exception("Post-session failed for %s", inst)

        # Final cleanup
        _STOP_FILE.unlink(missing_ok=True)
