"""Pydantic v2 models for MiMo request/response shapes.

These mirror the OpenAI Chat Completions schema served by the MiMo API
(``https://api.xiaomimimo.com/v1``). Anything missing from the upstream
response is captured via ``model_config = {"extra": "allow"}`` so new fields
don't silently break the SDK.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Common base
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Multimodal content parts
# ---------------------------------------------------------------------------


class TextPart(_Base):
    """Plain-text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageURL(_Base):
    """Either a remote URL or a ``data:image/...;base64,...`` payload."""

    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None


class ImagePart(_Base):
    """Image content block — only accepted by ``mimo-v2-omni``."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


ContentPart = Union[TextPart, ImagePart]
"""One element of a multimodal ``content`` array."""

MessageContent = Union[str, List[ContentPart]]
"""Either a plain string (text-only) or a list of typed parts."""


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------


class ToolCallFunction(_Base):
    name: str
    arguments: str


class ToolCall(_Base):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(_Base):
    # ``role`` is optional because streaming deltas often only set it on the
    # very first chunk and omit it afterwards.
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[MessageContent] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None  # MiMo reasoning models echo this


# ---------------------------------------------------------------------------
# Tool / function definitions (function calling)
# ---------------------------------------------------------------------------


class ToolFunction(_Base):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(_Base):
    type: Literal["function"] = "function"
    function: ToolFunction


ToolChoice = Union[Literal["auto", "required", "none"], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Audio (TTS) — MiMo TTS rides on /chat/completions with an ``audio`` field
# ---------------------------------------------------------------------------


class AudioConfig(_Base):
    """Configuration that turns ``/chat/completions`` into a TTS call."""

    voice: str = "mimo_default"
    format: Literal["mp3", "wav", "ogg", "flac", "pcm"] = "mp3"
    style: Optional[str] = None  # e.g. "Bright, natural, conversational tone."


# ---------------------------------------------------------------------------
# Chat completion request / response
# ---------------------------------------------------------------------------


class ResponseFormat(_Base):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(_Base):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    response_format: Optional[ResponseFormat] = None
    user: Optional[str] = None
    audio: Optional[AudioConfig] = None
    extra_body: Optional[Dict[str, Any]] = None


class Usage(_Base):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: Optional[int] = None


class Choice(_Base):
    index: int = 0
    message: Optional[Message] = None
    delta: Optional[Message] = None  # only present in streaming chunks
    finish_reason: Optional[str] = None


class ChatCompletion(_Base):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = Field(default_factory=list)
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None

    @property
    def text(self) -> str:
        """Convenience: first choice's assistant text content as a single string."""
        if not self.choices:
            return ""
        msg = self.choices[0].message or self.choices[0].delta
        if msg is None or msg.content is None:
            return ""
        if isinstance(msg.content, str):
            return msg.content
        return "".join(part.text for part in msg.content if isinstance(part, TextPart))


class ChatCompletionChunk(_Base):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[Choice] = Field(default_factory=list)
    usage: Optional[Usage] = None

    @property
    def delta_text(self) -> str:
        """The incremental text contained in this chunk (may be empty)."""
        if not self.choices:
            return ""
        delta = self.choices[0].delta
        if delta is None or delta.content is None:
            return ""
        if isinstance(delta.content, str):
            return delta.content
        return "".join(p.text for p in delta.content if isinstance(p, TextPart))


# ---------------------------------------------------------------------------
# Models listing
# ---------------------------------------------------------------------------


class ModelInfo(_Base):
    id: str
    object: str = "model"
    owned_by: Optional[str] = None
    created: Optional[int] = None


class ModelList(_Base):
    object: str = "list"
    data: List[ModelInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Speech (TTS) high-level result
# ---------------------------------------------------------------------------


class SpeechResult(_Base):
    """High-level wrapper returned by :meth:`MiMo.speech.create`.

    The MiMo TTS endpoint returns a chat completion whose first choice
    contains a base64-encoded audio payload. This object surfaces both the
    decoded ``audio_bytes`` and metadata about the request.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    audio_bytes: bytes
    format: str = "mp3"
    voice: str = "mimo_default"
    text: str = ""
    raw: Optional[ChatCompletion] = None

    def save(self, path: str) -> str:
        """Write the audio bytes to ``path`` and return the resolved path."""
        import os

        with open(path, "wb") as f:
            f.write(self.audio_bytes)
        return os.path.abspath(path)


__all__ = [
    "TextPart",
    "ImageURL",
    "ImagePart",
    "ContentPart",
    "MessageContent",
    "ToolCallFunction",
    "ToolCall",
    "Message",
    "ToolFunction",
    "Tool",
    "ToolChoice",
    "AudioConfig",
    "ResponseFormat",
    "ChatCompletionRequest",
    "Usage",
    "Choice",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ModelInfo",
    "ModelList",
    "SpeechResult",
]
