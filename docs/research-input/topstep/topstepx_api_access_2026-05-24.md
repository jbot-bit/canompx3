# TopstepX API Access / ProjectX Gateway Grounding — 2026-05-24

Truth class: OFFICIAL_DOC_EXTRACT + LOCAL_RUNTIME_MEASUREMENT

## Official Docs Checked

- Topstep Help Center: `https://help.topstep.com/en/articles/11187768-topstepx-api-access`
- ProjectX Gateway API docs: `https://gateway.docs.projectx.com/docs/getting-started/connection-urls/`
- ProjectX Gateway API auth docs: `https://gateway.docs.projectx.com/docs/getting-started/authenticate/authenticate-api-key/`
- ProjectX Gateway API first-order/account/contract docs: `https://gateway.docs.projectx.com/docs/getting-started/placing-your-first-order/`
- ProjectX Gateway API rate limits: `https://gateway.docs.projectx.com/docs/getting-started/rate-limits/`
- ProjectX Gateway realtime overview: `https://gateway.docs.projectx.com/docs/realtime/`

## Findings

- TopstepX API access is a current paid API feature for TopstepX traders. It is set up from inside TopstepX via Settings -> API -> ProjectX Linking, then API key generation in the TopstepX platform.
- ProjectX third-party/wider partner messaging does not mean the TopstepX API path is gone. Topstep's current help article still routes TopstepX API users through ProjectX linking/dashboard.
- Current official TopstepX Gateway endpoints:
  - REST API: `https://api.topstepx.com`
  - User SignalR hub: `https://rtc.topstepx.com/hubs/user`
  - Market SignalR hub: `https://rtc.topstepx.com/hubs/market`
- API-key auth is `POST https://api.topstepx.com/api/Auth/loginKey` with JSON body fields `userName` and `apiKey`; successful response includes a JWT token.
- Active account discovery uses `POST https://api.topstepx.com/api/Account/search` with `onlyActiveAccounts: true`; the returned `id` is the API `accountId`, not necessarily the UI "Sub ID".
- Contract discovery uses `POST https://api.topstepx.com/api/Contract/available` or contract search endpoints; orders use the returned contract id such as `CON.F.US.MNQ.M26`.
- Realtime updates use SignalR over the user and market hubs. User hub covers accounts/orders/positions/trades; market hub covers quotes/trades/depth.
- Rate limits in the official docs: `POST /api/History/retrieveBars` is 50 requests per 30 seconds; other authenticated endpoints are 200 requests per 60 seconds.

## Local Runtime Measurement

On 2026-05-24, `scripts/tools/broker_handshake_check.py` against `PROJECTX_BASE_URL=https://api.topstepx.com` returned PASS:

- Auth token acquired.
- Two active accounts visible:
  - Express Funded: `EXPRESS-V2-451890-53179846`, API account id `21944866`.
  - Trading Combine: `50KTC-V2-451890-29512053`, API account id `23055112`.
- MNQ front-month contract resolved to `CON.F.US.MNQ.M26`.

## Repo Configuration Implication

- For the active `topstep_50k_mnq_auto` profile, keep:
  - `BROKER=projectx`
  - `PROJECTX_BASE_URL=https://api.topstepx.com`
  - `PROJECTX_USERNAME=<TopstepX API username>`
  - `PROJECTX_API_KEY=<TopstepX API key>`
- Use API account id `21944866` when explicitly targeting the Express Funded account. Do not confuse this with the Topstep UI Sub ID.
- Do not treat a browser/dashboard login as API readiness. The bot needs a valid TopstepX API subscription, ProjectX-linked API account, and generated API key.
