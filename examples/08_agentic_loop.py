"""A minimal ReAct-style agent built on top of ``mimo-sdk-py``.

The model is given a tiny set of tools (``calculator``, ``read_file``, and
``finish``) and a budget of agent turns. Each turn the SDK emits the
model's tool call, we execute it, and we append the result back into
``messages`` so the model can decide what to do next.

This pattern is the spine of every "agentic" workflow — coding agents,
research agents, autonomous trading bots — and shows that the SDK is
production-ready for such loops, not just one-shot chats.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from mimo import MiMo

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic math expression like '2 + 2 * pi'. "
                           "Allowed names: pi, e, sqrt, sin, cos, log, exp.",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Return the first 4 KB of a local text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Stop reasoning and return the final answer to the user.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def run_tool(name: str, args: dict[str, Any]) -> str:
    if name == "calculator":
        env = {
            "pi": math.pi, "e": math.e,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "log": math.log, "exp": math.exp,
        }
        try:
            value = eval(args["expr"], {"__builtins__": {}}, env)  # noqa: S307 - sandboxed
            return json.dumps({"value": value})
        except Exception as exc:  # pragma: no cover - tool error path
            return json.dumps({"error": str(exc)})

    if name == "read_file":
        try:
            path = Path(args["path"])
            return json.dumps({"path": str(path), "content": path.read_text()[:4096]})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    if name == "finish":
        return json.dumps({"answer": args.get("answer", "")})

    return json.dumps({"error": f"unknown tool {name}"})


def agent(question: str, *, max_turns: int = 6) -> str:
    """Run a small agent loop until the model calls ``finish`` or budget runs out."""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a small reasoning agent. You can call calculator, "
                "read_file, or finish. Always call finish when you are done."
            ),
        },
        {"role": "user", "content": question},
    ]

    with MiMo() as client:
        for turn in range(max_turns):
            resp = client.chat.completions.create(
                model="mimo-v2-pro",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            calls = msg.tool_calls or []

            if not calls:
                return msg.content if isinstance(msg.content, str) else resp.text

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [c.model_dump() for c in calls],
            })

            for call in calls:
                args = json.loads(call.function.arguments or "{}")
                print(f"[turn {turn+1}] tool={call.function.name} args={args}")
                result = run_tool(call.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": result,
                })
                if call.function.name == "finish":
                    return json.loads(result).get("answer", "")

    return "(agent ran out of turns)"


if __name__ == "__main__":
    answer = agent("What is sqrt(57121) plus the value of pi rounded to 4 decimals?")
    print("\n=== AGENT ANSWER ===")
    print(answer)
