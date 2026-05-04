"""Error mapping + retry policy."""

from __future__ import annotations

import httpx
import pytest

from mimo import errors
from mimo._transport import raise_for_response, should_retry


def _make_response(status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=body,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.example.com/v1/x"),
    )


@pytest.mark.parametrize(
    "status,exc",
    [
        (400, errors.BadRequestError),
        (401, errors.AuthenticationError),
        (403, errors.PermissionDeniedError),
        (404, errors.NotFoundError),
        (422, errors.UnprocessableEntityError),
        (429, errors.RateLimitError),
        (500, errors.InternalServerError),
        (502, errors.InternalServerError),
        (599, errors.InternalServerError),
        (418, errors.APIStatusError),
    ],
)
def test_raise_for_response_maps_status(status: int, exc: type[errors.MiMoError]) -> None:
    body = b'{"error":{"message":"boom","code":"oops"}}'
    response = _make_response(status, body)
    with pytest.raises(exc) as info:
        raise_for_response(response)
    err = info.value
    assert err.status_code == status
    assert "boom" in err.message
    if isinstance(err, errors.APIStatusError):
        assert err.code == "oops"


def test_raise_for_response_2xx_is_noop() -> None:
    response = _make_response(200, b'{"ok":true}')
    raise_for_response(response)  # should not raise


def test_rate_limit_propagates_retry_after() -> None:
    response = _make_response(429, b"{}", headers={"retry-after": "12"})
    with pytest.raises(errors.RateLimitError) as info:
        raise_for_response(response)
    assert info.value.retry_after == 12.0


def test_should_retry_429_within_budget() -> None:
    exc = errors.RateLimitError("slow down", retry_after=0)
    assert should_retry(exc, attempt=0, max_retries=3) is not None


def test_should_retry_429_exhausted() -> None:
    exc = errors.RateLimitError("slow down")
    assert should_retry(exc, attempt=3, max_retries=3) is None


def test_should_retry_400_is_not_retried() -> None:
    exc = errors.BadRequestError("bad")
    assert should_retry(exc, attempt=0, max_retries=3) is None


def test_should_retry_handles_connect_error() -> None:
    exc = httpx.ConnectError("dns down")
    assert should_retry(exc, attempt=0, max_retries=2) is not None


def test_from_status_unknown_code_falls_back_to_apistatuserror() -> None:
    err = errors.from_status(418, message="teapot")
    assert isinstance(err, errors.APIStatusError)
    assert not isinstance(err, errors.RateLimitError)
    assert err.message == "teapot"
