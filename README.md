# mimo-sdk-py

> Pythonic SDK for the **Xiaomi MiMo** API. Chat, vision, TTS, streaming, async, agentic — fully typed, OpenAI-compatible.

[![CI](https://github.com/amilaa-0/mimo-sdk-py/actions/workflows/ci.yml/badge.svg)](https://github.com/amilaa-0/mimo-sdk-py/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/PyPI-mimo--sdk-blue)](https://pypi.org/project/mimo-sdk/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Type-checked](https://img.shields.io/badge/typed-mypy_strict-blueviolet)](#)

`mimo-sdk-py` is an unofficial, MIT-licensed Python client for the
[Xiaomi MiMo API](https://platform.xiaomimimo.com/). It wraps every public
endpoint with a typed, batteries-included surface that should feel familiar
to anyone who's used the OpenAI SDK — and works as a drop-in replacement
when you want to swap providers.

## Why this exists

Xiaomi's MiMo platform exposes powerful models — `mimo-v2-pro` ships with
a **1 million token context window**, `mimo-v2-omni` is a reasoning-capable
multimodal model, and `mimo-v2.5-tts` produces high-quality speech in
seven voices. The platform speaks the OpenAI Chat Completions dialect, but
new SDK surface area like the multimodal TTS payload deserves a small,
well-typed wrapper that:

- Hides the bytes/base64 plumbing for audio.
- Keeps every model's capability metadata locally so you don't have to
  remember which one supports vision.
- Has a single retry/back-off policy that respects `Retry-After` headers.
- Exposes both **sync** (`MiMo`) and **async** (`AsyncMiMo`) clients with
  a streaming API on each.
- Ships a small CLI for one-off chat / TTS / vision calls.

## Install

```bash
pip install mimo-sdk          # core SDK
pip install 'mimo-sdk[cli]'   # adds the `mimo` command-line tool
```

> Requires Python **3.9+**. The SDK ships with type stubs and is checked
> with `mypy --strict`.

## Authentication

Get an API key at [platform.xiaomimimo.com](https://platform.xiaomimimo.com/)
and either pass it explicitly or expose it via env var:

```bash
export XIAOMI_API_KEY="sk-..."
# alias also supported:
export MIMO_API_KEY="sk-..."
```

You can override the base URL (e.g. for a proxy) via `MIMO_BASE_URL` or
the `base_url=` kwarg.

## 60-second tour

```python
from mimo import MiMo

with MiMo() as client:
    # Plain chat
    resp = client.chat.completions.create(
        model="mimo-v2-flash",
        messages=[{"role": "user", "content": "Hello, MiMo!"}],
    )
    print(resp.text)

    # Streaming
    for chunk in client.chat.completions.stream(
        model="mimo-v2-flash",
        messages=[{"role": "user", "content": "Tell me a haiku"}],
    ):
        print(chunk.delta_text, end="", flush=True)

    # Vision (omni model)
    answer = client.vision.describe("chart.png", "What pattern is this?")
    print(answer.text)

    # Text-to-speech
    audio = client.speech.create(
        "Hello world!",
        voice="default_en",
        style="Bright, friendly",
    )
    audio.save("hello.mp3")
```

### Async

```python
import asyncio
from mimo import AsyncMiMo

async def main() -> None:
    async with AsyncMiMo() as client:
        async for chunk in client.chat.completions.stream(
            model="mimo-v2-flash",
            messages=[{"role": "user", "content": "Stream please"}],
        ):
            print(chunk.delta_text, end="")

asyncio.run(main())
```

### CLI

```bash
mimo chat "What is the capital of Japan?"
mimo chat "Tell me a story" --stream --model mimo-v2-pro
mimo speak "Hello, world" -o hello.mp3 --voice default_en
mimo vision photo.jpg "What's in this picture?"
mimo models                 # built-in catalog with capabilities
mimo models --remote        # query the API
```

## Models at a glance

| Model              | Inputs        | Reasoning | Context | Max output | Use case                          |
| ------------------ | ------------- | --------- | ------- | ---------- | --------------------------------- |
| `mimo-v2-flash`    | text          | no        | 262 K   | 8 192      | Fast general-purpose chat         |
| `mimo-v2-pro`      | text          | **yes**   | **1 M** | 32 000     | Long-document reasoning           |
| `mimo-v2-omni`     | text + image  | yes       | 262 K   | 32 000     | Multimodal reasoning              |
| `mimo-v2.5-tts`    | text → audio  | n/a       | n/a     | n/a        | Speech synthesis (7 voices)       |

The catalog lives in `mimo.catalog` and you can register new models at
runtime if MiMo ships one before the SDK is updated:

```python
from mimo import register_model, ModelSpec

register_model(ModelSpec(
    id="mimo-v3-experimental",
    name="MiMo V3 (preview)",
    inputs=("text", "image"),
    reasoning=True,
    context_window=2_097_152,
    max_output_tokens=64_000,
))
```

## Multimodal in one call

```python
from mimo import MiMo, build_user_message

with MiMo() as client:
    resp = client.chat.completions.create(
        model="mimo-v2-omni",
        messages=[
            build_user_message(
                "What does this candlestick chart suggest?",
                images=["btc-15m.png"],
                detail="high",
            ),
        ],
    )
    print(resp.text)
```

`encode_image()` accepts:

- a local path (`"./photo.jpg"` or `pathlib.Path(...)`),
- a remote URL (`"https://..."`),
- raw `bytes`,

and produces the right `data:` URI automatically.

## Function calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

resp = client.chat.completions.create(
    model="mimo-v2-pro",
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",
)
print(resp.choices[0].message.tool_calls)
```

## Errors & retries

The SDK translates HTTP errors into a tiny exception hierarchy so you
can catch exactly what you care about:

```python
from mimo import MiMo, errors

try:
    client.chat.completions.create(...)
except errors.RateLimitError as e:
    sleep(e.retry_after or 5)
except errors.AuthenticationError:
    print("Bad API key")
except errors.MiMoError as e:
    print(f"Other API error: {e}")
```

`RateLimitError`, `InternalServerError`, and transient network errors are
retried automatically with exponential back-off (configurable via
`MiMo(max_retries=...)`). The `Retry-After` header is honored when the
server provides it.

## Comparison vs. `openai` SDK

The MiMo API is OpenAI-compatible at the chat-completion level, so you
*can* point the official OpenAI client at it:

```python
from openai import OpenAI
client = OpenAI(api_key=..., base_url="https://api.xiaomimimo.com/v1")
```

`mimo-sdk-py` chooses to ship its own client for three reasons:

1. **Audio-as-chat is a first-class citizen.** TTS runs through
   `/chat/completions` with an `audio` field; the OpenAI SDK has no
   reason to know about that. We expose `client.speech.create(...)`
   that returns decoded bytes plus the underlying chat completion.
2. **Local capability catalog.** `mimo.catalog.get_model(...)` answers
   "does this model support vision?" without a network call.
3. **Smaller dependency footprint.** `mimo-sdk-py` only depends on
   `httpx` and `pydantic` — no `tiktoken`, no `distro`, no `anyio` glue.

You can still use both side-by-side; the wire format is the same.

## Development

```bash
git clone https://github.com/amilaa-0/mimo-sdk-py
cd mimo-sdk-py
uv venv .venv --python python3.10
uv pip install --python .venv/bin/python -e ".[cli,dev]"
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy src
```

## Examples

The [`examples/`](examples) folder has runnable scripts for:

- `01_chat_basic.py` — simplest possible call
- `02_chat_streaming.py` — token-by-token output
- `03_long_context_pro.py` — feeding 500K+ tokens into `mimo-v2-pro`
- `04_vision_omni.py` — multimodal Q&A
- `05_tts_speech.py` — TTS to MP3
- `06_function_calling.py` — tool / function calling round-trip
- `07_async_batch.py` — fan-out concurrent requests
- `08_agentic_loop.py` — full agentic ReAct-style loop

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

This is an unofficial, community-maintained SDK and is not affiliated
with or endorsed by Xiaomi. "Xiaomi", "MiMo", and related marks are the
property of their respective owners.
