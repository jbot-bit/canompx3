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

Security checklist before going live:
    [ ] WEBHOOK_SECRET set in .env (at least 32 random chars)
    [ ] cloudflared tunnel is a NAMED tunnel (not trycloudflare.com temporary)
    [ ] Apex/prop firm allows automated API trading on your account
    [ ] Rate limit tested (max 3 orders / 60s)
    [ ] Tested in DEMO mode for 5+ sessions before live
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

        confirm = input(
            "\n⚠  LIVE MODE — real money orders will be placed.\n"
            "   Verify cloudflared tunnel is named (not trycloudflare.com).\n"
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

    log.info("Webhook server on http://0.0.0.0:%d", args.port)
    log.info("Health check: http://localhost:%d/health", args.port)
    log.info("Trade endpoint: POST http://localhost:%d/trade", args.port)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
