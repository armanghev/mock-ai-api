from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from app.common import (
    anthropic_sse,
    chunk_text,
    estimate_tokens,
    extract_prompt_text,
    generate_id,
    stream_with_delay,
)
from app.scenarios import Scenario


def build_message(scenario: Scenario, messages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_text = extract_prompt_text(messages)
    input_tokens = estimate_tokens(prompt_text)

    if scenario.name in {"mock-tool-use", "mock-tool-stream"}:
        return {
            "id": generate_id("msg_"),
            "type": "message",
            "role": "assistant",
            "model": scenario.base_model,
            "content": [
                {
                    "type": "tool_use",
                    "id": generate_id("toolu_"),
                    "name": "Write",
                    "input": {"path": "tmp/mock-file.txt", "contents": "mock output"},
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 14},
        }

    response_text = _assistant_text(scenario, prompt_text)
    return {
        "id": generate_id("msg_"),
        "type": "message",
        "role": "assistant",
        "model": scenario.base_model,
        "content": [{"type": "text", "text": response_text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": estimate_tokens(response_text),
        },
    }


def count_tokens(messages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_text = extract_prompt_text(messages)
    return {"input_tokens": estimate_tokens(prompt_text)}


async def stream_message(
    scenario: Scenario, messages: list[dict[str, Any]]
) -> AsyncIterator[str]:
    prompt_text = extract_prompt_text(messages)
    input_tokens = estimate_tokens(prompt_text)

    if scenario.name == "mock-stream-error":
        yield anthropic_sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": "Mock streaming failure"},
            },
        )
        return

    if scenario.name in {"mock-tool-stream", "mock-tool-use"}:
        async for event in stream_with_delay(
            _tool_stream_events(scenario, input_tokens),
            scenario_name=scenario.name,
        ):
            yield event
        return

    response_text = _assistant_text(
        scenario, prompt_text, prefix="Mock Anthropic stream for: "
    )
    output_tokens = estimate_tokens(response_text)
    message_id = generate_id("msg_")
    events = _text_stream_events(
        scenario=scenario,
        message_id=message_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        response_text=response_text,
    )
    async for event in stream_with_delay(events, scenario_name=scenario.name):
        yield event


def _assistant_text(
    scenario: Scenario,
    prompt_text: str,
    *,
    prefix: str = "Mock Anthropic response for: ",
) -> str:
    if scenario.name in {"mock-text", "mock-anthropic-stream"}:
        return "Hello. It's nice to meet you."
    return f"{prefix}{prompt_text or 'empty prompt'}"


def _text_stream_events(
    *,
    scenario: Scenario,
    message_id: str,
    input_tokens: int,
    output_tokens: int,
    response_text: str,
) -> list[str]:
    events: list[str] = []
    events.append(anthropic_sse("ping", {"type": "ping"}))
    events.append(
        anthropic_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": scenario.base_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        )
    )
    events.append(
        anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
    )
    for part in chunk_text(response_text):
        events.append(
            anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": part},
                },
            )
        )
    events.append(
        anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    events.append(
        anthropic_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
    )
    events.append(anthropic_sse("message_stop", {"type": "message_stop"}))
    return events


def _tool_stream_events(scenario: Scenario, input_tokens: int) -> list[str]:
    message_id = generate_id("msg_")
    tool_id = generate_id("toolu_")
    tool_input = {"path": "tmp/mock-file.txt", "contents": "mock output"}
    serialized = json.dumps(tool_input)
    events: list[str] = []
    events.append(anthropic_sse("ping", {"type": "ping"}))
    events.append(
        anthropic_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": scenario.base_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        )
    )
    events.append(
        anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Write",
                    "input": {},
                },
            },
        )
    )
    for fragment in _json_fragments(serialized):
        events.append(
            anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": fragment},
                },
            )
        )
    events.append(
        anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    events.append(
        anthropic_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 14},
            },
        )
    )
    events.append(anthropic_sse("message_stop", {"type": "message_stop"}))
    return events


def _json_fragments(serialized: str) -> list[str]:
    return [serialized[index : index + 8] for index in range(0, len(serialized), 8)]
