from __future__ import annotations

"""Small repository-native signed bearer tokens.

This is intentionally provider/infrastructure independent.  It closes the previous
"username in the URL is authorization" gap without adding a deployment dependency.
Production must set AUTH_TOKEN_SECRET to a high-entropy value; an ephemeral process key
is used only for local development and expires all sessions after restart.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass


_ENV_SECRET = os.getenv("AUTH_TOKEN_SECRET", "").encode("utf-8")
_SECRET = _ENV_SECRET or secrets.token_bytes(32)
TOKEN_TTL_SECONDS = max(300, int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "28800")))


class AuthenticationError(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    username: str
    role: str
    expires_at: int


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_token(username: str, role: str, *, now: int | None = None) -> tuple[str, int]:
    issued = int(time.time() if now is None else now)
    expires = issued + TOKEN_TTL_SECONDS
    payload = {
        "sub": str(username),
        "role": str(role),
        "iat": issued,
        "exp": expires,
        "nonce": secrets.token_urlsafe(12),
    }
    encoded = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signature = _b64encode(hmac.new(_SECRET, encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}", expires


def verify_token(token: str, *, now: int | None = None) -> Principal:
    try:
        encoded, supplied_signature = str(token or "").split(".", 1)
        expected_signature = _b64encode(
            hmac.new(_SECRET, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise AuthenticationError("invalid_token")
        payload = json.loads(_b64decode(encoded))
        username = str(payload["sub"]).strip()
        role = str(payload["role"]).strip()
        expires = int(payload["exp"])
    except AuthenticationError:
        raise
    except Exception as exc:
        raise AuthenticationError("invalid_token") from exc
    if not username or role not in {"student", "admin"}:
        raise AuthenticationError("invalid_token")
    if expires <= int(time.time() if now is None else now):
        raise AuthenticationError("expired_token")
    return Principal(username=username, role=role, expires_at=expires)


def bearer_token(authorization: str | None) -> str:
    scheme, _, token = str(authorization or "").partition(" ")
    if scheme.casefold() != "bearer" or not token.strip():
        raise AuthenticationError("missing_token")
    return token.strip()


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    sensitive = {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"}
    return {key: ("[REDACTED]" if key.casefold() in sensitive else value) for key, value in headers.items()}
