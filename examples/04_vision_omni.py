"""Multimodal: ask ``mimo-v2-omni`` what's in an image.

Usage::

    python examples/04_vision_omni.py path/to/image.png "What is in this picture?"

The image argument can be a local file, a remote URL, or even a ``data:``
URI — :func:`mimo.encode_image` figures out the right wire format.
"""

from __future__ import annotations

import sys

from mimo import MiMo


def main(image: str, prompt: str) -> None:
    with MiMo() as client:
        # Two equivalent ways to do this:

        # 1. High-level helper
        resp = client.vision.describe(
            image=image,
            prompt=prompt,
            detail="high",
        )

        # 2. Manual message construction (when you want full control)
        # msg = build_user_message(prompt, images=[image], detail="high")
        # resp = client.chat.completions.create(
        #     model="mimo-v2-omni",
        #     messages=[msg],
        # )

    print(resp.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: 04_vision_omni.py <image> [prompt]", file=sys.stderr)
        sys.exit(1)
    img = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."
    main(img, question)
