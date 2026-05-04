"""Built-in catalog of MiMo models with capabilities and limits.

The catalog is intentionally hard-coded: callers can override or extend it
via :func:`register_model`, but most users never need to. Pulling these
constants out of the API call code makes example/error messages much more
informative ("model X does not support image input") without an extra
network round trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata about a MiMo model."""

    id: str
    name: str
    inputs: Tuple[str, ...]  # subset of {"text", "image", "audio"}
    reasoning: bool = False
    context_window: int = 0
    max_output_tokens: int = 0
    is_tts: bool = False
    description: str = ""

    @property
    def supports_vision(self) -> bool:
        return "image" in self.inputs

    @property
    def supports_text(self) -> bool:
        return "text" in self.inputs


_BUILTIN: Dict[str, ModelSpec] = {
    "mimo-v2-flash": ModelSpec(
        id="mimo-v2-flash",
        name="Xiaomi MiMo V2 Flash",
        inputs=("text",),
        reasoning=False,
        context_window=262_144,
        max_output_tokens=8_192,
        description="Lightweight, fast text model. Best default for high-throughput chat.",
    ),
    "mimo-v2-pro": ModelSpec(
        id="mimo-v2-pro",
        name="Xiaomi MiMo V2 Pro",
        inputs=("text",),
        reasoning=True,
        context_window=1_048_576,
        max_output_tokens=32_000,
        description="Reasoning model with a 1M-token context window. Use for long documents.",
    ),
    "mimo-v2-omni": ModelSpec(
        id="mimo-v2-omni",
        name="Xiaomi MiMo V2 Omni",
        inputs=("text", "image"),
        reasoning=True,
        context_window=262_144,
        max_output_tokens=32_000,
        description="Reasoning multimodal model. Accepts text and image input.",
    ),
    "mimo-v2.5-tts": ModelSpec(
        id="mimo-v2.5-tts",
        name="Xiaomi MiMo V2.5 TTS",
        inputs=("text",),
        reasoning=False,
        is_tts=True,
        description="Text-to-speech model. Supports multiple voices and styles.",
    ),
}

# Public default for chat namespace
DEFAULT_CHAT_MODEL = "mimo-v2-flash"
DEFAULT_VISION_MODEL = "mimo-v2-omni"
DEFAULT_REASONING_MODEL = "mimo-v2-pro"
DEFAULT_TTS_MODEL = "mimo-v2.5-tts"

# Built-in voices for the TTS model
TTS_VOICES: Tuple[str, ...] = (
    "mimo_default",
    "default_zh",
    "default_en",
    "Mia",
    "Chloe",
    "Milo",
    "Dean",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_extra: Dict[str, ModelSpec] = {}


def list_models() -> List[ModelSpec]:
    """Return all known model specs (built-in + user-registered)."""
    seen: Dict[str, ModelSpec] = {}
    seen.update(_BUILTIN)
    seen.update(_extra)
    return list(seen.values())


def get_model(model_id: str) -> ModelSpec:
    """Look up a model spec by id. Raises :class:`KeyError` if unknown."""
    if model_id in _extra:
        return _extra[model_id]
    if model_id in _BUILTIN:
        return _BUILTIN[model_id]
    raise KeyError(f"Unknown MiMo model: {model_id!r}")


def register_model(spec: ModelSpec) -> None:
    """Register or override a model spec at runtime.

    Useful if MiMo ships a new model before the SDK is updated.
    """
    _extra[spec.id] = spec


__all__ = [
    "ModelSpec",
    "list_models",
    "get_model",
    "register_model",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_VISION_MODEL",
    "DEFAULT_REASONING_MODEL",
    "DEFAULT_TTS_MODEL",
    "TTS_VOICES",
]
