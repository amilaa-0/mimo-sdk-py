"""Encoder edge cases for :mod:`mimo.vision`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mimo.vision import build_user_message, encode_image


def test_encode_image_passthrough_remote_url() -> None:
    url = "https://cdn.example.com/cat.jpg"
    assert encode_image(url) == url


def test_encode_image_passthrough_data_uri() -> None:
    uri = "data:image/png;base64,iVBORw0KGgoAAAANS"
    assert encode_image(uri) == uri


def test_encode_image_local_file(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    path = tmp_path / "hi.png"
    path.write_bytes(payload)
    encoded = encode_image(path)
    # ``iVBORw0KGgo`` is the canonical base64 of the 8-byte PNG magic, so we
    # *expect* that prefix when round-tripping a real PNG.
    assert encoded.startswith("data:image/png;base64,iVBORw0KGgo")
    assert len(encoded) > len("data:image/png;base64,")


@pytest.mark.parametrize(
    "magic,expected",
    [
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"GIF89a", "image/gif"),
        (b"BM", "image/bmp"),
    ],
)
def test_sniff_mime_via_bytes(magic: bytes, expected: str) -> None:
    encoded = encode_image(magic + b"\x00" * 16)
    assert encoded.startswith(f"data:{expected};base64,")


def test_build_user_message_text_only() -> None:
    msg = build_user_message("hi", images=None)
    assert msg.role == "user"
    assert msg.content == "hi"


def test_build_user_message_multimodal() -> None:
    msg = build_user_message(
        "what is this?",
        images=["https://example.com/x.png", b"\x89PNG\r\n\x1a\n"],
    )
    assert msg.role == "user"
    parts = msg.content
    assert isinstance(parts, list)
    assert parts[0].type == "text"
    assert parts[1].type == "image_url"
    assert parts[1].image_url.url == "https://example.com/x.png"
    assert parts[2].image_url.url.startswith("data:image/png;base64,")
