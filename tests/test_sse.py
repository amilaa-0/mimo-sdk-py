"""SSE chunk parsing."""

from __future__ import annotations

import pytest

from mimo._transport import parse_sse_lines, parse_sse_lines_async
from mimo.types import ChatCompletionChunk

SAMPLE_LINES = [
    'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
    "",
    'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hel"}}]}',
    'data: {"id":"x","choices":[{"index":0,"delta":{"content":"lo"}}]}',
    "data: [DONE]",
    "",
]


def test_parse_sse_lines_collects_chunks() -> None:
    chunks = list(parse_sse_lines(iter(SAMPLE_LINES)))
    assert len(chunks) == 3
    assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
    text = "".join(c.delta_text for c in chunks)
    assert text == "Hello"


def test_parse_sse_lines_skips_keepalive_and_comments() -> None:
    lines = [
        ":keepalive",
        "",
        "event: ping",
        'data: {"id":"y","choices":[{"index":0,"delta":{"content":"yo"}}]}',
        "data: [DONE]",
    ]
    chunks = list(parse_sse_lines(iter(lines)))
    assert len(chunks) == 1
    assert chunks[0].delta_text == "yo"


@pytest.mark.asyncio
async def test_parse_sse_lines_async() -> None:
    async def gen():
        for line in SAMPLE_LINES:
            yield line

    text = ""
    async for chunk in parse_sse_lines_async(gen()):
        text += chunk.delta_text
    assert text == "Hello"
