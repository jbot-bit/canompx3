"""
Entry point for the TradingView → Tradovate webhook server.

Usage:
    python scripts/run_webhook_server.py            # DEMO mode (default, safe)
    python scripts/run_webhook_server.py --live     # LIVE MONEY — requires typed CONFIRM

Two-terminal workflow:
    Terminal 1: python scripts/run_webhook_server.py
    Terminal 2: cloudflared tunnel run orb-webhook
    TradingView: paste the public URL into Alert → Webhook URL field

Environment variables (.env):
    WEBHOOK_SECRET=your-secret-here   # Required for live mode
    WEBHOOK_PORT=8765                 # Optional, default 8765
    WEBHOOK_DEMO=true                 # Overridden by --live flag
    WEBHOOK_PROFILE_ID=topstep_50k_mnq_auto  # Required for --live profile binding
    WEBHOOK_LIVE_RISK_ACK=ENABLE_UNMANAGED_LIVE_WEBHOOK  # Required for --live

Security checklist before going live:
    [ ] WEBHOOK_SECRET set in .env (at least 32 random chars)
    [ ] cloudflared tunnel is a NAMED tunnel (not trycloudflare.com temporary)
    [ ] Prop firm allows automated API trading (Tradeify/TopStep YES, Apex NO)
    [ ] Rate limit tested (max 3 orders / 60s)
    [ ] Tested in DEMO mode for 5+ sessions before live
    [ ] Documented why webhook live orders are acceptable despite bypassing SessionOrchestrator risk/HWM/F-1/kill-switch controls
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="TradingView webhook server")
    parser.add_argument(
        "--live", action="store_true", help="REAL MONEY — requires typing CONFIRM and WEBHOOK_SECRET set"
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("WEBHOOK_PORT", "8765")))
    args = parser.parse_args()

    if args.live:
        # Hard gate: secret must be set
        secret = os.environ.get("WEBHOOK_SECRET", "")
        if not secret or len(secret) < 16:
            log.error("BLOCKED: WEBHOOK_SECRET must be set (≥16 chars) before going live")
            log.error("Add WEBHOOK_SECRET=<random-string> to your .env file")
            sys.exit(1)

        risk_ack = os.environ.get("WEBHOOK_LIVE_RISK_ACK", "")
        expected_ack = "ENABLE_UNMANAGED_LIVE_WEBHOOK"
        if risk_ack != expected_ack:
            log.error("BLOCKED: webhook live mode bypasses SessionOrchestrator risk/HWM/F-1/kill-switch controls")
            log.error("Set WEBHOOK_LIVE_RISK_ACK=%s only after documented operator approval", expected_ack)
            sys.exit(1)

        profile_id = os.environ.get("WEBHOOK_PROFILE_ID", "")
        if not profile_id:
            log.error("BLOCKED: WEBHOOK_PROFILE_ID is required for live webhook routing")
            sys.exit(1)

        confirm = input(
            "\n⚠  LIVE MODE — real money orders will be placed.\n"
            "   Verify cloudflared tunnel is named (not trycloudflare.com).\n"
            "   Verify WEBHOOK_PROFILE_ID is the intended live account profile.\n"
            "   Type CONFIRM to proceed: "
        ).strip()
        if confirm != "CONFIRM":
            print("Aborted.")
            sys.exit(0)

        os.environ["WEBHOOK_DEMO"] = "false"
        log.warning("=== LIVE MODE ACTIVE — REAL MONEY ORDERS ===")
    else:
        os.environ["WEBHOOK_DEMO"] = "true"
        log.info("Starting in DEMO mode (no real orders)")

    os.environ["WEBHOOK_PORT"] = str(args.port)

    import uvicorn

    from trading_app.live.webhook_server import app

    log.info("Webhook server on http://127.0.0.1:%d", args.port)
    log.info("Health check: http://localhost:%d/health", args.port)
    log.info("Trade endpoint: POST http://localhost:%d/trade", args.port)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
