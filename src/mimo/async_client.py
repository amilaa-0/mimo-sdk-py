"""Asynchronous high-level client.

The API mirrors :class:`mimo.MiMo` but every method is a coroutine and
streams return ``AsyncIterator``::

    from mimo import AsyncMiMo

    async def main():
        async with AsyncMiMo() as client:
            resp = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )
            print(resp.text)

            async for chunk in client.chat.completions.stream(
                messages=[{"role": "user", "content": "Tell a joke"}]
            ):
                print(chunk.delta_text, end="")
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import Any, List, Optional, Union

import httpx

from . import errors, vision
from ._request_builder import (
    MessageInput,
    ToolInput,
    build_chat_request,
    build_speech_request,
    build_vision_messages,
)
from ._transport import (
    TransportConfig,
    parse_sse_lines_async,
    raise_for_response,
    should_retry,
)
from .catalog import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VISION_MODEL,
    list_models,
)
from .client import _extract_audio_bytes  # reuse decoder
from .types import (
    AudioConfig,
    ChatCompletion,
    ChatCompletionChunk,
    ModelList,
    ResponseFormat,
    SpeechResult,
    ToolChoice,
)


class _AsyncChatCompletions:
    def __init__(self, client: AsyncMiMo) -> None:
        self._client = client

    async def create(
        self,
        *,
        messages: Sequence[MessageInput],
        model: str = DEFAULT_CHAT_MODEL,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        tools: Optional[Iterable[ToolInput]] = None,
        tool_choice: Optional[ToolChoice] = None,
        response_format: Optional[Union[ResponseFormat, Mapping[str, Any], str]] = None,
        user: Optional[str] = None,
        audio: Optional[Union[AudioConfig, Mapping[str, Any]]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        body = build_chat_request(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            stream=stream or None,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            user=user,
            audio=audio,
            extra_body=extra_body,
        )
        if stream:
            return self._client._stream_request(
                "POST", "/chat/completions", body=body, timeout=timeout, extra_headers=extra_headers
            )
        data = await self._client._request(
            "POST", "/chat/completions", body=body, timeout=timeout, extra_headers=extra_headers
        )
        return ChatCompletion.model_validate(data)

    def stream(
        self,
        *,
        messages: Sequence[MessageInput],
        model: str = DEFAULT_CHAT_MODEL,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        body = build_chat_request(
            messages=messages,
            model=model,
            stream=True,
            **kwargs,
        )
        return self._client._stream_request("POST", "/chat/completions", body=body)


class _AsyncChat:
    def __init__(self, client: AsyncMiMo) -> None:
        self.completions = _AsyncChatCompletions(client)


class _AsyncSpeech:
    def __init__(self, client: AsyncMiMo) -> None:
        self._client = client

    async def create(
        self,
        text: str,
        *,
        voice: str = "mimo_default",
        format: str = "mp3",
        style: Optional[str] = None,
        model: str = DEFAULT_TTS_MODEL,
        extra_body: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> SpeechResult:
        body = build_speech_request(
            text=text,
            model=model,
            voice=voice,
            format=format,
            style=style,
            extra_body=extra_body,
        )
        data = await self._client._request(
            "POST", "/chat/completions", body=body, timeout=timeout, extra_headers=extra_headers
        )
        completion = ChatCompletion.model_validate(data)
        audio_bytes = _extract_audio_bytes(data)
        return SpeechResult(
            audio_bytes=audio_bytes,
            format=format,
            voice=voice,
            text=text,
            raw=completion,
        )


class _AsyncVision:
    def __init__(self, client: AsyncMiMo) -> None:
        self._client = client

    async def describe(
        self,
        image: vision.ImageInput,
        prompt: str = "Describe this image in detail.",
        *,
        model: str = DEFAULT_VISION_MODEL,
        system: Optional[str] = None,
        detail: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        msgs = build_vision_messages(prompt, [image], detail=detail, system=system)
        result = await self._client.chat.completions.create(model=model, messages=msgs, **kwargs)
        assert isinstance(result, ChatCompletion)
        return result


class _AsyncModels:
    def __init__(self, client: AsyncMiMo) -> None:
        self._client = client

    async def list(self, *, timeout: Optional[float] = None) -> ModelList:
        data = await self._client._request("GET", "/models", timeout=timeout)
        return ModelList.model_validate(data)

    def list_local(self) -> List[Any]:
        return list_models()


# =============================================================================


class AsyncMiMo:
    """Asynchronous client. See :class:`mimo.MiMo` for sync usage."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._config = TransportConfig.resolve(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(self._config.timeout, connect=self._config.connect_timeout),
            follow_redirects=True,
        )
        self.chat = _AsyncChat(self)
        self.speech = _AsyncSpeech(self)
        self.vision = _AsyncVision(self)
        self.models = _AsyncModels(self)

    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncMiMo:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> Any:
        url = path if path.startswith("http") else self._config.base_url + path
        headers = self._config.build_headers(extra_headers)
        attempt = 0
        max_retries = self._config.max_retries
        while True:
            try:
                response = await self._http.request(
                    method,
                    url,
                    json=body if body is not None else None,
                    params=params,
                    headers=headers,
                    timeout=timeout if timeout is not None else self._config.timeout,
                )
                raise_for_response(response)
                if response.status_code == 204 or not response.content:
                    return None
                return response.json()
            except (errors.MiMoError, httpx.HTTPError) as exc:
                if not isinstance(exc, errors.MiMoError):
                    if isinstance(exc, httpx.TimeoutException):
                        translated: errors.MiMoError = errors.APITimeoutError(
                            f"Request timed out after {timeout or self._config.timeout}s"
                        )
                    else:
                        translated = errors.APIConnectionError(str(exc))
                    sleep = should_retry(exc, attempt, max_retries)
                    if sleep is None:
                        raise translated from exc
                else:
                    sleep = should_retry(exc, attempt, max_retries)
                    if sleep is None:
                        raise
                await asyncio.sleep(sleep)
                attempt += 1

    async def _stream_request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any],
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        url = self._config.base_url + path
        headers = self._config.build_headers({**(extra_headers or {}), "Accept": "text/event-stream"})
        attempt = 0
        max_retries = self._config.max_retries
        while True:
            try:
                async with self._http.stream(
                    method,
                    url,
                    json=dict(body),
                    headers=headers,
                    timeout=timeout if timeout is not None else self._config.timeout,
                ) as response:
                    if not response.is_success:
                        await response.aread()
                        raise_for_response(response)
                    async for chunk in parse_sse_lines_async(response.aiter_lines()):
                        yield chunk
                    return
            except (errors.MiMoError, httpx.HTTPError) as exc:
                if not isinstance(exc, errors.MiMoError):
                    if isinstance(exc, httpx.TimeoutException):
                        translated: errors.MiMoError = errors.APITimeoutError(str(exc))
                    else:
                        translated = errors.APIConnectionError(str(exc))
                    sleep = should_retry(exc, attempt, max_retries)
                    if sleep is None:
                        raise translated from exc
                else:
                    sleep = should_retry(exc, attempt, max_retries)
                    if sleep is None:
                        raise
                await asyncio.sleep(sleep)
                attempt += 1


__all__ = ["AsyncMiMo"]
