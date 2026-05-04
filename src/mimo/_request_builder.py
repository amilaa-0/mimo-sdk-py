"""Pure helpers that build request bodies — no I/O, easy to unit-test."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, List, Optional, Union

from . import vision
from .catalog import DEFAULT_CHAT_MODEL, DEFAULT_TTS_MODEL
from .types import (
    AudioConfig,
    ChatCompletionRequest,
    Message,
    ResponseFormat,
    Tool,
    ToolChoice,
)

MessageInput = Union[Message, Mapping[str, Any]]
ToolInput = Union[Tool, Mapping[str, Any]]


def _coerce_message(item: MessageInput) -> Message:
    if isinstance(item, Message):
        return item
    if isinstance(item, Mapping):
        return Message.model_validate(dict(item))
    raise TypeError(f"messages must be Message or dict, got {type(item).__name__}")


def _coerce_tool(item: ToolInput) -> Tool:
    if isinstance(item, Tool):
        return item
    if isinstance(item, Mapping):
        return Tool.model_validate(dict(item))
    raise TypeError(f"tools must be Tool or dict, got {type(item).__name__}")


def build_chat_request(
    *,
    messages: Sequence[MessageInput],
    model: str = DEFAULT_CHAT_MODEL,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
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
) -> dict[str, Any]:
    """Compose the JSON body for ``POST /v1/chat/completions``.

    Returns a plain dict so the transport layer doesn't have to know the
    Pydantic schema. ``extra_body`` is merged at the top level after all
    SDK-managed fields, allowing forward-compatibility with new MiMo params.
    """
    coerced_messages = [_coerce_message(m) for m in messages]

    rf: Optional[ResponseFormat]
    if response_format is None:
        rf = None
    elif isinstance(response_format, ResponseFormat):
        rf = response_format
    elif isinstance(response_format, str):
        rf = ResponseFormat(type=response_format)  # type: ignore[arg-type]
    else:
        rf = ResponseFormat.model_validate(dict(response_format))

    audio_obj: Optional[AudioConfig]
    if audio is None:
        audio_obj = None
    elif isinstance(audio, AudioConfig):
        audio_obj = audio
    else:
        audio_obj = AudioConfig.model_validate(dict(audio))

    tools_list: Optional[List[Tool]]
    if tools is None:
        tools_list = None
    else:
        tools_list = [_coerce_tool(t) for t in tools]

    payload = ChatCompletionRequest(
        model=model,
        messages=coerced_messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
        stream=stream,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        seed=seed,
        tools=tools_list,
        tool_choice=tool_choice,
        response_format=rf,
        user=user,
        audio=audio_obj,
    ).model_dump(exclude_none=True, by_alias=True)

    payload.pop("extra_body", None)
    if extra_body:
        for k, v in extra_body.items():
            payload[k] = v
    return payload


def build_speech_request(
    *,
    text: str,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = "mimo_default",
    format: str = "mp3",
    style: Optional[str] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Compose the body for a TTS call (which is itself a chat completion)."""
    user_msg = Message(role="user", content=text)
    audio = AudioConfig(voice=voice, format=format, style=style)  # type: ignore[arg-type]
    return build_chat_request(
        messages=[user_msg],
        model=model,
        audio=audio,
        extra_body=extra_body,
    )


def build_vision_messages(
    prompt: str,
    images: Optional[Iterable[Any]] = None,
    *,
    detail: Optional[str] = None,
    system: Optional[str] = None,
) -> List[Message]:
    """Quick path for "look at this image and answer X" prompts."""
    msgs: List[Message] = []
    if system:
        msgs.append(Message(role="system", content=system))
    msgs.append(vision.build_user_message(prompt, images=images, detail=detail))
    return msgs


__all__ = [
    "MessageInput",
    "ToolInput",
    "build_chat_request",
    "build_speech_request",
    "build_vision_messages",
]
