"""Tool / function calling round-trip with ``mimo-v2-pro``.

The model is exposed two tools: ``get_weather`` and ``get_time``. The user
asks a question that requires calling one of them, the SDK serializes the
tool definitions, the model emits a ``tool_call``, we execute the tool
locally, then send the result back to the model so it can finish the answer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from mimo import MiMo

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Return the current time in UTC.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Stand-in implementations of the tools."""
    if name == "get_weather":
        city = arguments.get("city", "?")
        unit = arguments.get("unit", "celsius")
        # In real life this would hit a weather API.
        return json.dumps({"city": city, "temp": 27, "unit": unit, "condition": "sunny"})
    if name == "get_time":
        return json.dumps({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return json.dumps({"error": f"unknown tool {name}"})


def main() -> None:
    with MiMo() as client:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What's the weather in Jakarta and what time is it?"},
        ]

        # First turn — model decides which tool(s) to call.
        first = client.chat.completions.create(
            model="mimo-v2-pro",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = first.choices[0].message
        tool_calls = msg.tool_calls or []
        print(f"Model issued {len(tool_calls)} tool call(s)")

        # Append assistant turn (with the tool calls) to history.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        })

        # Execute each tool and send results back.
        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = execute_tool(tc.function.name, args)
            print(f"  -> {tc.function.name}({args}) = {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            })

        # Second turn — model produces final answer.
        final = client.chat.completions.create(
            model="mimo-v2-pro",
            messages=messages,
        )
        print("\n=== Final answer ===")
        print(final.text)


if __name__ == "__main__":
    main()
