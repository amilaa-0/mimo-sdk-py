"""Pythonic SDK for the Xiaomi MiMo API.

Quick start::

    from mimo import MiMo

    client = MiMo(api_key="...")  # or set XIAOMI_API_KEY env var
    resp = client.chat.completions.create(
        model="mimo-v2-flash",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp.text)
"""

from __future__ import annotations

from . import errors, vision  # re-exported submodules
from ._version import __version__
from .async_client import AsyncMiMo
from .catalog import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_REASONING_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VISION_MODEL,
    TTS_VOICES,
    ModelSpec,
    get_model,
    list_models,
    register_model,
)
from .client import MiMo
from .errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    MiMoError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from .types import (
    AudioConfig,
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ImagePart,
    ImageURL,
    Message,
    ModelInfo,
    ModelList,
    ResponseFormat,
    SpeechResult,
    TextPart,
    Tool,
    ToolCall,
    ToolFunction,
    Usage,
)
from .vision import build_user_message, encode_image

__all__ = [
    "__version__",
    # clients
    "MiMo",
    "AsyncMiMo",
    # types
    "Message",
    "TextPart",
    "ImagePart",
    "ImageURL",
    "AudioConfig",
    "Tool",
    "ToolCall",
    "ToolFunction",
    "ResponseFormat",
    "ChatCompletion",
    "ChatCompletionChunk",
    "Choice",
    "Usage",
    "ModelInfo",
    "ModelList",
    "SpeechResult",
    # catalog
    "ModelSpec",
    "list_models",
    "get_model",
    "register_model",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_VISION_MODEL",
    "DEFAULT_REASONING_MODEL",
    "DEFAULT_TTS_MODEL",
    "TTS_VOICES",
    # vision helpers
    "build_user_message",
    "encode_image",
    # errors
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
    # submodules
    "errors",
    "vision",
]
