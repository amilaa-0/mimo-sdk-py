"""Feed a long document into the 1M-context ``mimo-v2-pro`` reasoning model.

This example stuffs the contents of a local file (or this script itself by
default) into a single user message and asks for a summary. With a 1 048 576
token context, ``mimo-v2-pro`` happily ingests entire codebases or books.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mimo import MiMo


def main(path: str | None = None) -> None:
    target = Path(path) if path else Path(__file__)
    text = target.read_text(encoding="utf-8")

    print(f"Loading {target} ({len(text):,} characters)…", file=sys.stderr)

    with MiMo() as client:
        resp = client.chat.completions.create(
            model="mimo-v2-pro",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior code reviewer. Reply with a numbered "
                        "list of the 5 most important observations about the file."
                    ),
                },
                {
                    "role": "user",
                    "content": f"<file path=\"{target}\">\n{text}\n</file>\n\n"
                               "Please review this file.",
                },
            ],
            temperature=0.3,
        )
    print(resp.text)
    if resp.usage:
        print(f"\n[tokens] prompt={resp.usage.prompt_tokens:,} "
              f"completion={resp.usage.completion_tokens:,}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
