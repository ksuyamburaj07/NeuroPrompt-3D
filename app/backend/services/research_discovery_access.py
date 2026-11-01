"""Development-only access checks for research run discovery.

This is a defense-in-depth local development guard,
NOT user authentication or production authorization.
"""

from __future__ import annotations


ALLOWED_CLIENTS = frozenset({
    "127.0.0.1",
    "::1",
})

ALLOWED_HOSTS = frozenset({
    "127.0.0.1:8000",
    "127.0.0.1:8090",
    "localhost:8000",
    "[::1]:8000",
})

ALLOWED_ORIGINS = frozenset({
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8090",
    "http://localhost:5173",
})


def local_research_discovery_allowed(
    *,
    enabled: str | None,
    client_host: str | None,
    request_host: str | None,
    origin: str | None,
) -> bool:
    """Permit explicitly enabled, expected local development requests.

    Never treat this as sufficient access control behind
    an untrusted reverse proxy or on a public deployment.
    """
    if enabled != "1":
        return False

    if client_host not in ALLOWED_CLIENTS:
        return False

    if request_host not in ALLOWED_HOSTS:
        return False

    if origin is not None and origin not in ALLOWED_ORIGINS:
        return False

    return True
