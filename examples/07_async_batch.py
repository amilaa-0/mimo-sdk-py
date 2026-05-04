"""Fan out concurrent requests with :class:`AsyncMiMo`.

Demonstrates how to share a single ``AsyncMiMo`` instance (and therefore a
single underlying ``httpx.AsyncClient`` connection pool) across many
``asyncio.gather`` tasks. This is the recommended pattern for any kind of
batched / parallel inference.
"""

from __future__ import annotations

import asyncio
import time
from typing import List

from mimo import AsyncMiMo

QUESTIONS: List[str] = [
    "What is the capital of Japan?",
    "Translate 'good morning' to Indonesian.",
    "Name three side effects of caffeine.",
    "Give a one-sentence summary of relativity.",
    "Recommend a Linux file manager.",
    "What does HTTP/2 'multiplexing' mean?",
    "Convert 100 USD to JPY (approximate).",
    "Suggest a weekend hike near Tokyo.",
]


async def ask(client: AsyncMiMo, question: str) -> tuple[str, str]:
    resp = await client.chat.completions.create(
        model="mimo-v2-flash",
        messages=[{"role": "user", "content": question}],
        max_tokens=120,
        temperature=0.4,
    )
    return question, resp.text.strip()


async def main() -> None:
    async with AsyncMiMo() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(ask(client, q) for q in QUESTIONS))
        elapsed = time.perf_counter() - t0

    for question, answer in results:
        print(f"Q: {question}\nA: {answer}\n")
    print(f"Completed {len(results)} requests in {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
