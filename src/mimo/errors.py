"""Exception hierarchy for the MiMo SDK.

All MiMo-specific errors inherit from :class:`MiMoError` so callers can catch
``except MiMoError`` to handle anything the SDK throws while still being able
to discriminate on subclass when needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


class MiMoError(Exception):
    """Base class for every error raised by the MiMo SDK."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        body: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.body = dict(body) if body else None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        parts = [f"{type(self).__name__}({self.message!r}"]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.code:
            parts.append(f"code={self.code!r}")
        if self.request_id:
            parts.append(f"request_id={self.request_id!r}")
        return ", ".join(parts) + ")"


class APIConnectionError(MiMoError):
    """Network-level failure (DNS, TLS, connection reset, timeout)."""


class APITimeoutError(APIConnectionError):
    """The request did not complete before the configured timeout."""


class APIStatusError(MiMoError):
    """The API returned a non-2xx response."""


class BadRequestError(APIStatusError):
    """HTTP 400 — invalid request payload."""


class AuthenticationError(APIStatusError):
    """HTTP 401 — missing or invalid API key."""


class PermissionDeniedError(APIStatusError):
    """HTTP 403 — key valid but not authorized for the requested resource."""


class NotFoundError(APIStatusError):
    """HTTP 404 — model or endpoint not found."""


class UnprocessableEntityError(APIStatusError):
    """HTTP 422 — payload semantically invalid (e.g. wrong field type)."""


class RateLimitError(APIStatusError):
    """HTTP 429 — caller exceeded their rate or quota limit.

    ``retry_after`` is populated from the ``Retry-After`` header when present.
    """

    def __init__(self, *args: Any, retry_after: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.retry_after = retry_after


class InternalServerError(APIStatusError):
    """HTTP 5xx — upstream MiMo failure."""


_STATUS_MAP: dict[int, type[APIStatusError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}


def from_status(
    status_code: int,
    *,
    message: str,
    code: Optional[str] = None,
    request_id: Optional[str] = None,
    body: Optional[Mapping[str, Any]] = None,
    retry_after: Optional[float] = None,
) -> APIStatusError:
    """Construct the correct ``APIStatusError`` subclass for an HTTP status."""
    cls = _STATUS_MAP.get(status_code)
    if cls is None:
        cls = InternalServerError if status_code >= 500 else APIStatusError
    if cls is RateLimitError:
        return RateLimitError(
            message,
            status_code=status_code,
            code=code,
            request_id=request_id,
            body=body,
            retry_after=retry_after,
        )
    return cls(
        message,
        status_code=status_code,
        code=code,
        request_id=request_id,
        body=body,
    )


__all__ = [
    "MiMoError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "from_status",
]
