"""Integration tests for the FastAPI dashboard backend — no real LLM calls."""

import pytest
import httpx
from unittest.mock import patch

from promptstrike.api.server import app


# ── helper ────────────────────────────────────────────────────────────────────

def _async_client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


# ── dashboard ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dashboard_serves_html():
    """GET / should return the glassmorphism dashboard HTML."""
    async with _async_client() as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "PromptStrike" in resp.text


# ── campaigns ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_campaigns_returns_list():
    """GET /api/campaigns should always return a JSON list (possibly empty)."""
    async with _async_client() as client:
        resp = await client.get("/api/campaigns")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_get_nonexistent_campaign_returns_404():
    """GET /api/campaigns/99999 should return 404 for an unknown ID."""
    async with _async_client() as client:
        resp = await client.get("/api/campaigns/99999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Campaign not found"


# ── scan ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scan_rejects_missing_api_key():
    """POST /api/scan should return 400 when GROQ_API_KEY is not set."""
    with patch("promptstrike.api.server.settings") as mock_settings:
        mock_settings.groq_api_key = ""
        async with _async_client() as client:
            resp = await client.post("/api/scan", json={"target": "groq/test-model"})
    assert resp.status_code == 400
    assert "GROQ_API_KEY" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_scan_returns_scan_id_when_key_present():
    """POST /api/scan with a valid (mocked) key should return a scan_id immediately."""
    with patch("promptstrike.api.server.settings") as mock_settings, \
         patch("promptstrike.api.server._run_scan"):   # don't actually run the scan
        mock_settings.groq_api_key = "fake-key-for-test"
        async with _async_client() as client:
            resp = await client.post("/api/scan", json={
                "target": "groq/llama-3.3-70b-versatile",
                "algorithm": "pair",
                "goals": 1,
            })
    assert resp.status_code == 200
    body = resp.json()
    assert "scan_id" in body
    assert len(body["scan_id"]) == 8


# ── websocket ─────────────────────────────────────────────────────────────────

def test_websocket_sends_error_for_unknown_scan_id():
    """WS /ws/{unknown_id} should immediately send an error message then close."""
    import warnings
    from starlette.testclient import TestClient
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    with TestClient(app) as client:
        with client.websocket_connect("/ws/no-such-scan-abc123") as ws:
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert "no-such-scan-abc123" in msg["message"]
