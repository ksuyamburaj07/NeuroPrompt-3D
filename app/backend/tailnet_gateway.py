"""Private Tailscale Serve gateway for the accepted M11 app.

Not a public internet entrypoint. Never use with Funnel.
"""

import hmac
import os

from starlette.responses import PlainTextResponse

from app.backend.local_unified import app as original_app

TAILNET_HOST = b"sachin-inspiron.taila912ab.ts.net"
TAILNET_ORIGIN = b"https://" + TAILNET_HOST
LOCAL_HOST = b"127.0.0.1:8090"


def single_header(scope, name):
    values = [
        value for key, value in scope.get("headers", [])
        if key == name
    ]
    return values[0] if len(values) == 1 else None


async def reject(scope, receive, send):
    response = PlainTextResponse(
        "Private Tailscale access only.",
        status_code=403,
    )
    await response(scope, receive, send)


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        await original_app(scope, receive, send)
        return

    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return

    if scope["type"] != "http":
        return

    client = scope.get("client")
    if not client or client[0] != "127.0.0.1":
        await reject(scope, receive, send)
        return

    host = single_header(scope, b"host")
    identity = single_header(scope, b"tailscale-user-login")
    origin = single_header(scope, b"origin")

    if host not in (TAILNET_HOST, TAILNET_HOST + b":443"):
        await reject(scope, receive, send)
        return

    allowed_login = os.environ.get(
        "NP3D_TAILNET_ALLOWED_LOGIN", ""
    ).encode("utf-8")

    if (
        not allowed_login
        or identity is None
        or not hmac.compare_digest(identity, allowed_login)
    ):
        await reject(scope, receive, send)
        return

    if origin is not None and origin != TAILNET_ORIGIN:
        await reject(scope, receive, send)
        return

    translated = []

    for key, value in scope["headers"]:
        if key == b"host":
            translated.append((b"host", LOCAL_HOST))
        elif key == b"origin":
            translated.append(
                (b"origin", b"http://" + LOCAL_HOST)
            )
        elif key in (
            b"x-forwarded-for",
            b"x-forwarded-host",
            b"x-forwarded-proto",
            b"forwarded",
        ):
            continue
        else:
            translated.append((key, value))

    local_scope = dict(scope)
    local_scope["headers"] = translated
    local_scope["server"] = ("127.0.0.1", 8090)
    local_scope["scheme"] = "http"

    await original_app(local_scope, receive, send)
