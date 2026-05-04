"""End-to-end client behaviour against a mocked transport.

We use ``respx`` to hijack the ``httpx`` transport so the SDK can be
exercised without a real Xiaomi API key.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx

from mimo import AsyncMiMo, MiMo, errors

CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "mimo-v2-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


@respx.mock
def test_chat_completion_sync() -> None:
    route = respx.post("https://api.xiaomimimo.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    with MiMo() as client:
        resp = client.chat.completions.create(
            model="mimo-v2-flash",
            messages=[{"role": "user", "content": "Hi"}],
        )
    assert resp.text == "Hello there!"
    assert resp.usage and resp.usage.total_tokens == 7
    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "mimo-v2-flash"
    assert body["messages"][0]["content"] == "Hi"


@respx.mock
def test_chat_completion_uses_authorization_header() -> None:
    route = respx.post("https://api.xiaomimimo.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    with MiMo(api_key="sk-secret") as client:
        client.chat.completions.create(messages=[{"role": "user", "content": "x"}])
    auth = route.calls[0].request.headers.get("authorization")
    assert auth == "Bearer sk-secret"


@respx.mock
def test_streaming_yields_text() -> None:
    sse_body = (
        'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
        'data: {"id":"x","choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post("https://api.xiaomimimo.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
        )
    )
    with MiMo() as client:
        chunks = list(
            client.chat.completions.stream(
                model="mimo-v2-flash",
                messages=[{"role": "user", "content": "stream me"}],
            )
        )
    assert "".join(c.delta_text for c in chunks) == "Hello"


@respx.mock
def test_speech_decodes_audio_bytes() -> None:
    raw_audio = b"\xFF\xF3MP3DATA"
    body = {
        "id": "tts-1",
        "object": "chat.completion",
        "created": 0,
        "model": "mimo-v2.5-tts",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "audio": {
                        "data": base64.b64encode(raw_audio).decode(),
                        "format": "mp3",
                    },
                },
                "finish_reason": "stop",
            }
        ],
    }
    respx.post("https://api.xiaomimimo.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=body)
    )
    with MiMo() as client:
        result = client.speech.create("hello", voice="default_en")
    assert result.audio_bytes == raw_audio
    assert result.format == "mp3"
    assert result.voice == "default_en"


@respx.mock
def test_rate_limit_retries_then_succeeds() -> None:
    route = respx.post("https://api.xiaomimimo.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "slow"}}, headers={"retry-after": "0"}),
            httpx.Response(200, json=CHAT_RESPONSE),
        ]
    )
    with MiMo(max_retries=2) as client:
        resp = client.chat.completions.create(
            model="mimo-v2-flash",
            messages=[{"role": "user", "content": "x"}],
        )
    assert resp.text == "Hello there!"
    assert route.call_count == 2


@respx.mock
def test_authentication_error_not_retried() -> None:
    route = respx.post("https://api.xiaomimimo.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401, json={"error": {"message": "Invalid key", "code": "401"}}
        )
    )
    with MiMo(max_retries=3) as client, pytest.raises(errors.AuthenticationError):
        client.chat.completions.create(
            model="mimo-v2-flash",
            messages=[{"role": "user", "content": "x"}],
        )
    assert route.call_count == 1


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    with pytest.raises(errors.AuthenticationError):
        MiMo()


@respx.mock
@pytest.mark.asyncio
async def test_async_chat_completion() -> None:
    respx.post("https://api.xiaomimimo.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    async with AsyncMiMo() as client:
        resp = await client.chat.completions.create(
            model="mimo-v2-flash",
            messages=[{"role": "user", "content": "Hi"}],
        )
    assert resp.text == "Hello there!"


@respx.mock
@pytest.mark.asyncio
async def test_async_streaming() -> None:
    sse_body = (
        'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
        'data: {"id":"x","choices":[{"index":0,"delta":{"content":"!"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post("https://api.xiaomimimo.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body.encode(),
                                    headers={"content-type": "text/event-stream"})
    )
    async with AsyncMiMo() as client:
        chunks = []
        async for chunk in client.chat.completions.stream(
            model="mimo-v2-flash",
            messages=[{"role": "user", "content": "stream"}],
        ):
            chunks.append(chunk)
    assert "".join(c.delta_text for c in chunks) == "Hi!"
