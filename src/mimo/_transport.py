"""Internal HTTP transport — shared between sync and async clients.

Responsibilities:

* Base URL / API key / headers normalization
* Single source of truth for retry + back-off policy
* Translate non-2xx responses into typed :mod:`mimo.errors` exceptions
* Parse Server-Sent Events into ``ChatCompletionChunk`` objects

Public clients in :mod:`mimo.client` and :mod:`mimo.async_client` compose
this module rather than duplicating logic.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from . import errors
from ._version import __version__
from .types import ChatCompletionChunk

DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_TIMEOUT = 60.0
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 2
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

USER_AGENT = f"mimo-sdk-py/{__version__}"


# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------


@dataclass
class TransportConfig:
    """Resolved configuration shared by sync and async clients."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    organization: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    default_headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
    ) -> TransportConfig:
        """Pull configuration from arguments, falling back to env vars."""
        key = api_key or os.environ.get("XIAOMI_API_KEY") or os.environ.get("MIMO_API_KEY")
        if not key:
            raise errors.AuthenticationError(
                "No API key provided. Pass api_key=... or set XIAOMI_API_KEY in your environment."
            )
        return cls(
            api_key=key,
            base_url=(base_url or os.environ.get("MIMO_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            organization=organization or os.environ.get("MIMO_ORGANIZATION"),
            timeout=timeout if timeout is not None else DEFAULT_TIMEOUT,
            max_retries=max_retries if max_retries is not None else DEFAULT_MAX_RETRIES,
            default_headers=dict(default_headers or {}),
        )

    def build_headers(self, extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.organization:
            headers["X-MiMo-Organization"] = self.organization
        headers.update(self.default_headers)
        if extra:
            headers.update(extra)
        return headers


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _parse_error_body(raw: bytes) -> tuple[str, Optional[str]]:
    if not raw:
        return ("(empty response body)", None)
    try:
        data = json.loads(raw)
    except Exception:
        return (raw.decode("utf-8", errors="replace")[:512], None)
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return (str(err.get("message", "API error")), err.get("code"))
    return (str(data)[:512], None)


def raise_for_response(response: httpx.Response) -> None:
    """Translate any non-2xx ``httpx.Response`` into a typed ``MiMoError``."""
    if response.is_success:
        return
    msg, code = _parse_error_body(response.content)
    request_id = response.headers.get("x-request-id") or response.headers.get("openai-request-id")
    retry_after_raw = response.headers.get("retry-after")
    retry_after: Optional[float]
    try:
        retry_after = float(retry_after_raw) if retry_after_raw is not None else None
    except ValueError:
        retry_after = None
    body: Optional[Mapping[str, Any]]
    try:
        body = response.json()
    except Exception:
        body = None
    raise errors.from_status(
        response.status_code,
        message=msg,
        code=code,
        request_id=request_id,
        body=body,
        retry_after=retry_after,
    )


# ---------------------------------------------------------------------------
# Retry helpers (policy is the same for sync + async)
# ---------------------------------------------------------------------------


def _backoff_seconds(attempt: int, retry_after: Optional[float]) -> float:
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, 30.0)
    base = 0.5 * (2 ** attempt)
    return min(base + random.random() * 0.25, 8.0)


def should_retry(exc: BaseException, attempt: int, max_retries: int) -> Optional[float]:
    """Return a sleep duration if we should retry, else ``None``."""
    if attempt >= max_retries:
        return None
    if isinstance(exc, errors.RateLimitError):
        return _backoff_seconds(attempt, exc.retry_after)
    if isinstance(exc, errors.InternalServerError):
        return _backoff_seconds(attempt, None)
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return _backoff_seconds(attempt, None)
    if isinstance(exc, httpx.TimeoutException):
        return _backoff_seconds(attempt, None)
    return None


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


def parse_sse_lines(lines: Iterator[str]) -> Iterator[ChatCompletionChunk]:
    """Synchronous SSE → ``ChatCompletionChunk`` iterator."""
    for line in lines:
        chunk = _maybe_chunk(line)
        if chunk is _STOP:
            return
        if chunk is None:
            continue
        yield chunk


async def parse_sse_lines_async(lines: AsyncIterator[str]) -> AsyncIterator[ChatCompletionChunk]:
    async for line in lines:
        chunk = _maybe_chunk(line)
        if chunk is _STOP:
            return
        if chunk is None:
            continue
        yield chunk


_STOP = object()


def _maybe_chunk(line: str) -> Any:
    """Return ``None`` (skip), ``_STOP`` (end of stream), or a parsed chunk."""
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return _STOP
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return ChatCompletionChunk.model_validate(obj)


__all__ = [
    "TransportConfig",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "RETRYABLE_STATUS",
    "USER_AGENT",
    "raise_for_response",
    "should_retry",
    "parse_sse_lines",
    "parse_sse_lines_async",
]
