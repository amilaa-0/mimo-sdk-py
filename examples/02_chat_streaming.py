"""Stream tokens to stdout as they arrive."""

from __future__ import annotations

import sys

from mimo import MiMo


def main() -> None:
    with MiMo() as client:
        stream = client.chat.completions.stream(
            model="mimo-v2-flash",
            messages=[
                {"role": "user", "content": "Write a 4-line haiku about the moon."},
            ],
            temperature=0.8,
        )
        for chunk in stream:
            sys.stdout.write(chunk.delta_text)
            sys.stdout.flush()
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
