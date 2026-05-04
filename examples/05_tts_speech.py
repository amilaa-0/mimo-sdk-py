"""Synthesize speech with the MiMo TTS model.

Two modes:

1. Plain — pass a string and get an MP3 back.
2. Styled — provide a free-form ``style`` hint to colour the delivery.

The MiMo TTS endpoint rides on ``/v1/chat/completions`` with an ``audio``
field; the SDK abstracts that away so callers just see decoded bytes.
"""

from __future__ import annotations

from mimo import TTS_VOICES, MiMo


def main() -> None:
    print("Available voices:", ", ".join(TTS_VOICES))

    with MiMo() as client:
        # 1. Plain English voice
        result = client.speech.create(
            "The MiMo Python SDK is now generating speech for you.",
            voice="default_en",
        )
        path = result.save("out_plain.mp3")
        print(f"Saved {len(result.audio_bytes):,} bytes -> {path}")

        # 2. Styled, with a Mandarin voice
        styled = client.speech.create(
            "你好世界，这是来自 MiMo SDK 的语音合成示例。",
            voice="default_zh",
            style="Warm, slightly enthusiastic, conversational tone.",
        )
        path = styled.save("out_styled.mp3")
        print(f"Saved {len(styled.audio_bytes):,} bytes -> {path}")


if __name__ == "__main__":
    main()
