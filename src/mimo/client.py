"""Synchronous high-level client.

Usage::

    from mimo import MiMo

    client = MiMo(api_key=...)
    resp = client.chat.completions.create(
        model="mimo-v2-flash",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp.text)

    # streaming
    for chunk in client.chat.completions.stream(
        model="mimo-v2-flash",
        messages=[{"role": "user", "content": "Tell me a story"}],
    ):
        print(chunk.delta_text, end="", flush=True)

    # TTS
    audio = client.speech.create(text="Hello world", voice="default_en")
    audio.save("hello.mp3")

The SDK is OpenAI-compatible so existing OpenAI code can be migrated by
swapping ``OpenAI()`` for ``MiMo()``.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
    parse_sse_lines,
    raise_for_response,
    should_retry,
)
from .catalog import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VISION_MODEL,
    list_models,
)
from .types import (
    AudioConfig,
    ChatCompletion,
    ChatCompletionChunk,
    ModelList,
    ResponseFormat,
    SpeechResult,
    ToolChoice,
)


class _ChatCompletions:
    def __init__(self, client: MiMo) -> None:
        self._client = client

    # ------------------------------------------------------------------
    def create(
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
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Send a chat completion request.

        When ``stream=True`` returns an iterator of ``ChatCompletionChunk``.
        Otherwise returns a single ``ChatCompletion``.
        """
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
        data = self._client._request(
            "POST", "/chat/completions", body=body, timeout=timeout, extra_headers=extra_headers
        )
        return ChatCompletion.model_validate(data)

    # ------------------------------------------------------------------
    def stream(
        self,
        *,
        messages: Sequence[MessageInput],
        model: str = DEFAULT_CHAT_MODEL,
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Convenience wrapper: ``create(..., stream=True)``."""
        result = self.create(messages=messages, model=model, stream=True, **kwargs)
        assert not isinstance(result, ChatCompletion)  # narrows for type-checkers
        return result


class _Chat:
    def __init__(self, client: MiMo) -> None:
        self.completions = _ChatCompletions(client)


class _Speech:
    def __init__(self, client: MiMo) -> None:
        self._client = client

    def create(
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
        """Synthesize ``text`` into audio.

        Returns a :class:`SpeechResult` with ``audio_bytes`` ready for
        playback or :meth:`SpeechResult.save`.
        """
        body = build_speech_request(
            text=text,
            model=model,
            voice=voice,
            format=format,
            style=style,
            extra_body=extra_body,
        )
        data = self._client._request(
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


class _Vision:
    """Convenience helper around the omni model."""

    def __init__(self, client: MiMo) -> None:
        self._client = client

    def describe(
        self,
        image: vision.ImageInput,
        prompt: str = "Describe this image in detail.",
        *,
        model: str = DEFAULT_VISION_MODEL,
        system: Optional[str] = None,
        detail: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Single-shot helper: send an image + prompt, return the completion."""
        msgs = build_vision_messages(prompt, [image], detail=detail, system=system)
        result = self._client.chat.completions.create(
            model=model,
            messages=msgs,
            **kwargs,
        )
        assert isinstance(result, ChatCompletion)
        return result


class _Models:
    def __init__(self, client: MiMo) -> None:
        self._client = client

    def list(self, *, timeout: Optional[float] = None) -> ModelList:
        """Fetch the list of models the API key has access to."""
        data = self._client._request("GET", "/models", timeout=timeout)
        return ModelList.model_validate(data)

    def list_local(self) -> List[Any]:
        """Return the SDK's built-in catalog without making a request."""
        return list_models()


# =============================================================================
# Main class
# =============================================================================


class MiMo:
    """Synchronous client for the Xiaomi MiMo API.

    Example::

        client = MiMo(api_key="...")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}]
        )

    Use as a context manager to release HTTP connections eagerly::

        with MiMo() as client:
            client.chat.completions.create(...)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
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
        self._http = http_client or httpx.Client(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(self._config.timeout, connect=self._config.connect_timeout),
            follow_redirects=True,
        )
        # Namespaces
        self.chat = _Chat(self)
        self.speech = _Speech(self)
        self.vision = _Vision(self)
        self.models = _Models(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> MiMo:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal HTTP plumbing (with retry)
    # ------------------------------------------------------------------

    def _request(
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
                response = self._http.request(
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
                # Translate connection errors to our typed exception
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
                time.sleep(sleep)
                attempt += 1

    def _stream_request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any],
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> Iterator[ChatCompletionChunk]:
        url = self._config.base_url + path
        headers = self._config.build_headers({**(extra_headers or {}), "Accept": "text/event-stream"})
        attempt = 0
        max_retries = self._config.max_retries
        while True:
            try:
                with self._http.stream(
                    method,
                    url,
                    json=dict(body),
                    headers=headers,
                    timeout=timeout if timeout is not None else self._config.timeout,
                ) as response:
                    if not response.is_success:
                        # Read body so error mapping has content
                        response.read()
                        raise_for_response(response)
                    yield from parse_sse_lines(response.iter_lines())
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
                time.sleep(sleep)
                attempt += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_audio_bytes(data: Any) -> bytes:
    """Decode the audio payload returned by MiMo TTS.

    The exact response shape may evolve; we look at the most likely places:

    * ``choices[0].message.audio.data`` (OpenAI-style audio response)
    * ``choices[0].message.audio.b64`` (alternative key)
    * ``audio.data`` at the top level
    """
    if not isinstance(data, dict):
        return b""
    candidates: list[Any] = []
    audio_top = data.get("audio")
    if isinstance(audio_top, dict):
        candidates.extend([audio_top.get("data"), audio_top.get("b64"), audio_top.get("base64")])
    choices = data.get("choices") or []
    if choices and isinstance(choices, list):
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            audio = msg.get("audio")
            if isinstance(audio, dict):
                candidates.extend(
                    [audio.get("data"), audio.get("b64"), audio.get("base64"), audio.get("bytes")]
                )
    for cand in candidates:
        if isinstance(cand, (bytes, bytearray)):
            return bytes(cand)
        if isinstance(cand, str):
            try:
                return base64.b64decode(cand)
            except Exception:
                continue
    return b""


__all__ = ["MiMo"]
