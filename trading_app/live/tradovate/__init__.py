"""Tradovate broker integration for Tradeify, MFFU, and direct Tradovate accounts.

REST API at https://live.tradovateapi.com/v1 (demo: demo.tradovateapi.com).
Auth: POST /auth/accesstokenrequest → {accessToken, mdAccessToken}.
Orders: POST /order/placeorder, /order/placeOSO, /order/cancelorder.
Positions: GET /position/list.
Contracts: GET /contract/find?name=MNQM6.

Same BrokerAuth/BrokerRouter/BrokerContracts/BrokerPositions ABCs as ProjectX.
SessionOrchestrator doesn't know which broker it's talking to.
"""
