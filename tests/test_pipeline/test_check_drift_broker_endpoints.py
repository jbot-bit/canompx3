"""Tests for ``check_no_direct_requests_to_broker_endpoints`` (Check 156).

Stage 5 of feat/live-broker-resilience locks the doctrine that every broker
HTTP call MUST route through ``trading_app/live/http_client.py::BrokerHTTPClient``
so it picks up the retry budget, idempotency keys, and the circuit-breaker
``failure_hook`` wired in Stage 4. This file formalizes the manual injection
probe executed at Stage 5 partial close (literal ``requests.get(...)`` added to
``projectx/positions.py:25`` was caught with file:line diagnostic, then
reverted) as a permanent regression. Class bug protection per
``institutional-rigor.md`` § 11 — drift checks must be mutation-proof.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.check_drift import check_no_direct_requests_to_broker_endpoints


@pytest.fixture
def fake_trading_app(tmp_path: Path) -> Path:
    """Build a temp ``trading_app`` tree with the two broker subdirs + http_client."""
    ta = tmp_path / "trading_app"
    (ta / "live").mkdir(parents=True)
    (ta / "live" / "projectx").mkdir()
    (ta / "live" / "tradovate").mkdir()
    # http_client lives at trading_app/live/http_client.py — must remain allowlisted
    (ta / "live" / "http_client.py").write_text(
        "import requests\n"
        "def post_json(url, payload):\n"
        "    return requests.post(url, json=payload)\n",
        encoding="utf-8",
    )
    # __init__.py files are skipped by the check
    (ta / "live" / "projectx" / "__init__.py").write_text("", encoding="utf-8")
    (ta / "live" / "tradovate" / "__init__.py").write_text("", encoding="utf-8")
    return ta


def test_clean_tree_passes(fake_trading_app: Path) -> None:
    """No direct requests.* calls under broker dirs → zero violations."""
    (fake_trading_app / "live" / "projectx" / "order_router.py").write_text(
        "from trading_app.live.http_client import BrokerHTTPClient\n"
        "def submit(client: BrokerHTTPClient, payload):\n"
        "    return client.post_json('/orders', payload)\n",
        encoding="utf-8",
    )
    violations = check_no_direct_requests_to_broker_endpoints(fake_trading_app)
    assert violations == [], violations


def test_injected_requests_post_in_projectx_fails(fake_trading_app: Path) -> None:
    """Mutation probe: literal ``requests.post(...)`` in projectx fires the check."""
    bad = fake_trading_app / "live" / "projectx" / "order_router.py"
    bad.write_text(
        "import requests\n"
        "def submit(payload):\n"
        "    return requests.post('https://gateway-api-demo.s2f.projectx.com/orders', json=payload)\n",
        encoding="utf-8",
    )
    violations = check_no_direct_requests_to_broker_endpoints(fake_trading_app)
    assert len(violations) == 1, violations
    assert "order_router.py:3" in violations[0]
    assert "requests.post(" in violations[0]
    assert "BrokerHTTPClient" in violations[0]  # remediation pointer present


def test_injected_requests_get_in_tradovate_fails(fake_trading_app: Path) -> None:
    """Same protection for tradovate adapter subtree."""
    bad = fake_trading_app / "live" / "tradovate" / "positions.py"
    bad.write_text(
        "import requests\n"
        "def query_open():\n"
        "    return requests.get('https://demo.tradovateapi.com/v1/position/list')\n",
        encoding="utf-8",
    )
    violations = check_no_direct_requests_to_broker_endpoints(fake_trading_app)
    assert len(violations) == 1, violations
    assert "positions.py:3" in violations[0]
    assert "requests.get(" in violations[0]


def test_http_client_is_allowlisted(fake_trading_app: Path) -> None:
    """The canonical client itself uses ``requests`` directly — must not trip the check."""
    # http_client.py already contains requests.post( from the fixture; it lives at
    # trading_app/live/http_client.py, OUTSIDE the scanned broker subdirs, so it
    # is allowlisted structurally (not by a name-match rule).
    violations = check_no_direct_requests_to_broker_endpoints(fake_trading_app)
    assert violations == [], (
        f"http_client.py at trading_app/live/http_client.py must be allowlisted "
        f"(structural — not under projectx/ or tradovate/). Got: {violations}"
    )


def test_non_broker_files_are_allowlisted(fake_trading_app: Path) -> None:
    """Files outside projectx/ and tradovate/ may use ``requests`` directly."""
    other = fake_trading_app / "live" / "bot_dashboard.py"
    other.write_text(
        "import requests\n"
        "def health():\n"
        "    return requests.get('http://localhost:8088/health')\n",
        encoding="utf-8",
    )
    violations = check_no_direct_requests_to_broker_endpoints(fake_trading_app)
    assert violations == [], violations


def test_requests_exception_import_does_not_fire(fake_trading_app: Path) -> None:
    """Importing ``requests`` for exception classes does not match the call regex."""
    ok = fake_trading_app / "live" / "projectx" / "auth.py"
    ok.write_text(
        "import requests\n"
        "from trading_app.live.http_client import BrokerHTTPClient\n"
        "def login(client: BrokerHTTPClient):\n"
        "    try:\n"
        "        return client.post_json('/login', {})\n"
        "    except requests.RequestException as exc:\n"
        "        raise RuntimeError(str(exc))\n",
        encoding="utf-8",
    )
    violations = check_no_direct_requests_to_broker_endpoints(fake_trading_app)
    assert violations == [], violations


def test_comment_lines_do_not_fire(fake_trading_app: Path) -> None:
    """Commented-out ``requests.post(...)`` lines are skipped (regex respects ``#``)."""
    ok = fake_trading_app / "live" / "projectx" / "contract_resolver.py"
    ok.write_text(
        "from trading_app.live.http_client import BrokerHTTPClient\n"
        "# legacy: requests.post(url, json=payload)  # migrated 2026-05-18\n"
        "def resolve(client: BrokerHTTPClient, instrument: str):\n"
        "    return client.post_json('/contracts/resolve', {'symbol': instrument})\n",
        encoding="utf-8",
    )
    violations = check_no_direct_requests_to_broker_endpoints(fake_trading_app)
    assert violations == [], violations


def test_missing_trading_app_dir_returns_empty(tmp_path: Path) -> None:
    """Defensive: nonexistent trading_app_dir returns ``[]`` without raising."""
    missing = tmp_path / "does_not_exist"
    violations = check_no_direct_requests_to_broker_endpoints(missing)
    assert violations == []


def test_all_six_verbs_caught(fake_trading_app: Path) -> None:
    """Regex covers get/post/request/put/delete/patch — one per file to keep diagnostics clean."""
    verbs = ["get", "post", "request", "put", "delete", "patch"]
    for verb in verbs:
        (fake_trading_app / "live" / "projectx" / f"verb_{verb}.py").write_text(
            f"import requests\nrequests.{verb}('https://example.com')\n",
            encoding="utf-8",
        )
    violations = check_no_direct_requests_to_broker_endpoints(fake_trading_app)
    assert len(violations) == len(verbs), violations
    for verb in verbs:
        assert any(f"verb_{verb}.py:2" in v and f"requests.{verb}(" in v for v in violations), (
            f"verb={verb} missing from violations: {violations}"
        )
