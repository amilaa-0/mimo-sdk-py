"""Simplest possible chat completion.

Run::

    XIAOMI_API_KEY=sk-... python examples/01_chat_basic.py
"""

from __future__ import annotations

from mimo import MiMo


def main() -> None:
    with MiMo() as client:
        resp = client.chat.completions.create(
            model="mimo-v2-flash",
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is the capital of Indonesia?"},
            ],
        )
        print(resp.text)
        if resp.usage:
            print(f"\n[tokens] prompt={resp.usage.prompt_tokens} "
                  f"completion={resp.usage.completion_tokens}")


if __name__ == "__main__":
    main()
