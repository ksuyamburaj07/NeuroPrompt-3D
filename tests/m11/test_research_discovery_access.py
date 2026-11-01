"""Tests for development-only research discovery access."""

import pytest

from app.backend.services.research_discovery_access import (
    local_research_discovery_allowed,
)


@pytest.mark.parametrize(
    "enabled,client_host,request_host,origin,expected",
    [
        (
            "1", "127.0.0.1", "127.0.0.1:8000",
            "http://127.0.0.1:5173", True,
        ),
        (
            "1", "127.0.0.1", "localhost:8000",
            "http://localhost:5173", True,
        ),
        (
            "1", "::1", "[::1]:8000",
            None, True,
        ),
        (
            "1", "127.0.0.1", "127.0.0.1:8000",
            None, True,
        ),
        (
            None, "127.0.0.1", "127.0.0.1:8000",
            None, False,
        ),
        (
            "0", "127.0.0.1", "127.0.0.1:8000",
            None, False,
        ),
        (
            "true", "127.0.0.1", "127.0.0.1:8000",
            None, False,
        ),
        (
            "1", "192.168.1.10", "127.0.0.1:8000",
            None, False,
        ),
        (
            "1", "testclient", "127.0.0.1:8000",
            None, False,
        ),
        (
            "1", None, "127.0.0.1:8000",
            None, False,
        ),
        (
            "1", "127.0.0.1", "example.com",
            None, False,
        ),
        (
            "1", "127.0.0.1", "127.0.0.1:9000",
            None, False,
        ),
        (
            "1", "127.0.0.1", "127.0.0.1:8000",
            "https://example.com", False,
        ),
        (
            "1", "127.0.0.1", "127.0.0.1:8000",
            "null", False,
        ),
        (
            "1", "127.0.0.1", "127.0.0.1:8000",
            "http://127.0.0.1:9999", False,
        ),
    ],
)
def test_local_discovery_access(
    enabled,
    client_host,
    request_host,
    origin,
    expected,
):
    assert local_research_discovery_allowed(
        enabled=enabled,
        client_host=client_host,
        request_host=request_host,
        origin=origin,
    ) is expected
