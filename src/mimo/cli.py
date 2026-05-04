"""``mimo`` command-line tool.

Installed as ``mimo`` via ``[project.scripts]``. Optional dependency:
``pip install mimo-sdk[cli]`` pulls in ``typer`` and ``rich``.

Examples::

    mimo chat "What is the capital of Japan?"
    mimo chat "Tell me a story" --stream --model mimo-v2-pro
    mimo speak "Hello, world" -o hello.mp3 --voice default_en
    mimo vision photo.jpg "What's in this image?"
    mimo models
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "The mimo CLI requires extra deps. Install them with:\n"
        "    pip install 'mimo-sdk[cli]'"
    ) from exc

from . import MiMo, __version__
from .catalog import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VISION_MODEL,
    list_models,
)

app = typer.Typer(
    name="mimo",
    help="Command-line interface for the Xiaomi MiMo API.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mimo-sdk-py [bold]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Print version and exit."
    ),
) -> None:
    """Xiaomi MiMo SDK command-line tool."""


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@app.command("chat")
def chat_cmd(
    prompt: str = typer.Argument(..., help="The user prompt to send."),
    model: str = typer.Option(DEFAULT_CHAT_MODEL, "--model", "-m", help="Model id."),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt."),
    stream: bool = typer.Option(False, "--stream", help="Stream tokens as they arrive."),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
) -> None:
    """One-off chat completion."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    with MiMo() as client:
        if stream:
            it = client.chat.completions.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for chunk in it:
                sys.stdout.write(chunk.delta_text)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            console.print(resp.text)


# ---------------------------------------------------------------------------
# speak (TTS)
# ---------------------------------------------------------------------------


@app.command("speak")
def speak_cmd(
    text: str = typer.Argument(..., help="Text to synthesize."),
    output: Path = typer.Option(Path("out.mp3"), "--output", "-o", help="Output audio file."),
    voice: str = typer.Option("mimo_default", "--voice", "-v", help="Voice id."),
    fmt: str = typer.Option("mp3", "--format", "-f", help="Audio format."),
    style: Optional[str] = typer.Option(None, "--style", help="Voice style hint."),
    model: str = typer.Option(DEFAULT_TTS_MODEL, "--model", "-m"),
) -> None:
    """Synthesize speech from text and save to a file."""
    with MiMo() as client:
        audio = client.speech.create(
            text=text, model=model, voice=voice, format=fmt, style=style
        )
    saved = audio.save(str(output))
    console.print(f"Saved [bold green]{len(audio.audio_bytes)}[/] bytes to [bold]{saved}[/]")


# ---------------------------------------------------------------------------
# vision
# ---------------------------------------------------------------------------


@app.command("vision")
def vision_cmd(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Path to local image."),
    prompt: str = typer.Argument("Describe this image in detail.", help="Question about the image."),
    model: str = typer.Option(DEFAULT_VISION_MODEL, "--model", "-m"),
    detail: Optional[str] = typer.Option(None, "--detail", help="auto / low / high"),
) -> None:
    """Send an image + question to the omni model."""
    with MiMo() as client:
        resp = client.vision.describe(image=image, prompt=prompt, model=model, detail=detail)
    console.print(resp.text)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@app.command("models")
def models_cmd(
    remote: bool = typer.Option(False, "--remote", help="Query the API instead of the built-in catalog."),
) -> None:
    """List known MiMo models."""
    if remote:
        with MiMo() as client:
            data = client.models.list()
        for m in data.data:
            console.print(f"[cyan]{m.id}[/]  owner={m.owned_by or '-'}")
        return

    for spec in list_models():
        flags = []
        if spec.reasoning:
            flags.append("reasoning")
        if spec.supports_vision:
            flags.append("vision")
        if spec.is_tts:
            flags.append("tts")
        flag_str = ", ".join(flags) or "text"
        console.print(
            f"[bold cyan]{spec.id}[/]  ({flag_str})  ctx={spec.context_window or '-'}  "
            f"out={spec.max_output_tokens or '-'}"
        )
        console.print(f"  [dim]{spec.description}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
