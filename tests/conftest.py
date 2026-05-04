"""Shared pytest fixtures for the SDK test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure no test ever tries to talk to a real Xiaomi endpoint.

    Tests that actually want to assert on environment-driven behaviour can
    delete this in a child fixture or call ``monkeypatch.delenv`` themselves.
    """
    monkeypatch.setenv("XIAOMI_API_KEY", "test-key-deadbeef")
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.delenv("MIMO_BASE_URL", raising=False)
