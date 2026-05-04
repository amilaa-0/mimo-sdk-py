"""Helpers for building multimodal (image) chat messages.

Most callers won't need to touch this module directly — :class:`MiMo.chat`
and the high-level helpers accept ``image=...`` arguments that flow through
here. Exposing it publicly is still useful for advanced cases where you want
to construct your own ``Message`` array.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Iterable
from pathlib import Path
from typing import List, Union

from .types import ImagePart, ImageURL, Message, TextPart

ImageInput = Union[str, Path, bytes]
"""Accepted shapes:

* ``str``  — either a remote URL ("http(s)://...") or a local filesystem path.
* ``Path`` — local filesystem path.
* ``bytes`` — raw image bytes (mime sniffed via magic numbers / fallback to png).
"""


def _sniff_mime(data: bytes) -> str:
    """Best-effort MIME sniffing using common image magic numbers."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return "image/png"


def encode_image(source: ImageInput) -> str:
    """Return a string accepted by ``ImageURL.url``.

    * Remote URLs are returned unchanged.
    * Local files / raw bytes are base64-encoded into a ``data:`` URI.
    """
    if isinstance(source, (bytes, bytearray)):
        b64 = base64.b64encode(bytes(source)).decode("ascii")
        return f"data:{_sniff_mime(bytes(source))};base64,{b64}"

    if isinstance(source, Path) or (
        isinstance(source, str) and not source.startswith(("http://", "https://", "data:"))
    ):
        path = Path(source)
        mime = mimetypes.guess_type(str(path))[0]
        data = path.read_bytes()
        if not mime:
            mime = _sniff_mime(data)
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # already a remote/data URL
    return str(source)


def build_user_message(
    text: str,
    images: Union[ImageInput, Iterable[ImageInput], None] = None,
    *,
    detail: Union[str, None] = None,
) -> Message:
    """Construct a multimodal ``user`` message in one call.

    Example::

        msg = build_user_message("What is in this picture?", images="cat.jpg")
        client.chat.completions.create(model="mimo-v2-omni", messages=[msg])
    """
    if images is None:
        return Message(role="user", content=text)

    if isinstance(images, (str, Path, bytes, bytearray)):
        image_list: List[ImageInput] = [images]  # type: ignore[list-item]
    else:
        image_list = list(images)

    parts: List[Union[TextPart, ImagePart]] = []
    if text:
        parts.append(TextPart(text=text))
    for img in image_list:
        parts.append(
            ImagePart(image_url=ImageURL(url=encode_image(img), detail=detail))  # type: ignore[arg-type]
        )

    return Message(role="user", content=parts)


__all__ = ["ImageInput", "encode_image", "build_user_message"]
