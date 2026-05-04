"""Pure-logic tests for the request builders."""

from __future__ import annotations

from mimo._request_builder import (
    build_chat_request,
    build_speech_request,
    build_vision_messages,
)
from mimo.types import Message


def test_build_chat_request_minimal() -> None:
    body = build_chat_request(
        messages=[{"role": "user", "content": "Hi"}],
        model="mimo-v2-flash",
    )
    assert body["model"] == "mimo-v2-flash"
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    # Optional fields excluded by default
    assert "temperature" not in body
    assert "stream" not in body
    assert "audio" not in body


def test_build_chat_request_full() -> None:
    body = build_chat_request(
        messages=[Message(role="user", content="Tell me a story")],
        model="mimo-v2-pro",
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stream=True,
        seed=42,
        stop=["END"],
        response_format="json_object",
    )
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 512
    assert body["stream"] is True
    assert body["seed"] == 42
    assert body["response_format"] == {"type": "json_object"}


def test_build_chat_request_extra_body_overrides_at_top_level() -> None:
    body = build_chat_request(
        messages=[{"role": "user", "content": "x"}],
        extra_body={"custom_param": 7, "model": "override"},
    )
    assert body["custom_param"] == 7
    # extra_body wins for forward-compat with new server fields
    assert body["model"] == "override"


def test_build_speech_request_uses_audio_field() -> None:
    body = build_speech_request(
        text="Hello world",
        voice="Mia",
        format="wav",
        style="warm",
    )
    assert body["model"] == "mimo-v2.5-tts"
    assert body["audio"] == {"voice": "Mia", "format": "wav", "style": "warm"}
    assert body["messages"] == [{"role": "user", "content": "Hello world"}]


def test_build_vision_messages_appends_image() -> None:
    msgs = build_vision_messages(
        "What is this?",
        images=[b"\x89PNG\r\n\x1a\n" + b"\x00" * 16],
        detail="low",
        system="Be concise.",
    )
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    parts = msgs[1].content
    assert isinstance(parts, list)
    assert parts[0].type == "text"
    assert parts[1].type == "image_url"
    assert parts[1].image_url.url.startswith("data:image/png;base64,")
    assert parts[1].image_url.detail == "low"
