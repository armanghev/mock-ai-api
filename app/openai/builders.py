from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from app.common import (
    chunk_text,
    estimate_tokens,
    extract_prompt_text,
    generate_id,
    openai_sse,
    stream_with_delay,
    unix_timestamp,
)
from app.scenarios import Scenario


def build_chat_completion(scenario: Scenario, messages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_text = extract_prompt_text(messages)
    prompt_tokens = estimate_tokens(prompt_text)
    completion_id = generate_id("chatcmpl_")
    created = unix_timestamp()

    if scenario.name in {"mock-tool-call", "mock-bad-tool-json", "mock-tool-stream"}:
        arguments = json.dumps({"path": "tmp/mock-file.txt", "contents": "mock output"})
        if scenario.name == "mock-bad-tool-json":
            arguments = '{"path":"tmp/mock-file.txt","contents":"unterminated"'
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": scenario.base_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": generate_id("call_"),
                                "type": "function",
                                "function": {"name": "Write", "arguments": arguments},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 18,
                "total_tokens": prompt_tokens + 18,
            },
        }

    response_text = assistant_text(scenario, prompt_text)
    completion_tokens = estimate_tokens(response_text)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": scenario.base_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_legacy_completion(scenario: Scenario, prompt: str) -> dict[str, Any]:
    prompt_tokens = estimate_tokens(prompt)
    text = assistant_text(scenario, prompt, prefix="Mock completion for: ")
    completion_tokens = estimate_tokens(text)
    return {
        "id": generate_id("cmpl_"),
        "object": "text_completion",
        "created": unix_timestamp(),
        "model": scenario.base_model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_embeddings(model: str, input_value: Any) -> dict[str, Any]:
    from app.common import deterministic_embedding, extract_input_text

    if isinstance(input_value, list):
        inputs = [str(item) for item in input_value]
    else:
        inputs = [str(input_value)]
    data = [
        {"object": "embedding", "index": index, "embedding": deterministic_embedding(text)}
        for index, text in enumerate(inputs)
    ]
    token_count = max(1, sum(estimate_tokens(text) for text in inputs))
    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": token_count, "total_tokens": token_count},
    }


def build_moderation(model: str, input_value: str) -> dict[str, Any]:
    flagged = any(word in input_value.lower() for word in ("hurt", "kill", "attack", "hate"))
    return {
        "id": generate_id("modr-"),
        "model": model,
        "results": [
            {
                "flagged": flagged,
                "categories": {
                    "violence": flagged,
                    "self-harm": False,
                    "hate": False,
                    "sexual": False,
                    "harassment": False,
                },
                "category_scores": {
                    "violence": 0.98 if flagged else 0.01,
                    "self-harm": 0.01,
                    "hate": 0.0,
                    "sexual": 0.0,
                    "harassment": 0.0,
                },
            }
        ],
    }


def build_image_generation(model: str, prompt: str) -> dict[str, Any]:
    return {
        "created": unix_timestamp(),
        "data": [{"b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}],
        "model": model,
        "prompt": prompt,
    }


async def stream_chat_completion(
    scenario: Scenario,
    messages: list[dict[str, Any]],
) -> AsyncIterator[str]:
    prompt_text = extract_prompt_text(messages)
    prompt_tokens = estimate_tokens(prompt_text)
    completion_id = generate_id("chatcmpl_")
    created = unix_timestamp()
    events: list[str] = []

    if scenario.name == "mock-tool-stream":
        events.extend(
            _tool_stream_events(
                completion_id=completion_id,
                created=created,
                model=scenario.base_model,
                prompt_tokens=prompt_tokens,
            )
        )
    else:
        response_text = assistant_text(scenario, prompt_text, prefix="Mock stream for: ")
        completion_tokens = estimate_tokens(response_text)
        events.append(
            openai_sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": scenario.base_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )
        )
        for part in chunk_text(response_text):
            events.append(
                openai_sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": scenario.base_model,
                        "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
                    }
                )
            )
        events.append(
            openai_sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": scenario.base_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
        )

    events.append("data: [DONE]\n\n")
    async for event in stream_with_delay(events, scenario_name=scenario.name):
        yield event


async def stream_legacy_completion(
    scenario: Scenario,
    prompt: str,
) -> AsyncIterator[str]:
    prompt_tokens = estimate_tokens(prompt)
    text = assistant_text(scenario, prompt, prefix="Mock completion stream for: ")
    completion_tokens = estimate_tokens(text)
    completion_id = generate_id("cmpl_")
    created = unix_timestamp()
    events: list[str] = []

    for part in chunk_text(text):
        events.append(
            openai_sse(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": scenario.base_model,
                    "choices": [{"text": part, "index": 0, "logprobs": None, "finish_reason": None}],
                }
            )
        )
    events.append(
        openai_sse(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": scenario.base_model,
                "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )
    )
    events.append("data: [DONE]\n\n")
    async for event in stream_with_delay(events, scenario_name=scenario.name):
        yield event


async def stream_response(
    scenario: Scenario,
    input_value: Any,
    output_text: str,
) -> AsyncIterator[str]:
    from app.common import extract_input_text

    prompt_text = extract_input_text(input_value)
    input_tokens = estimate_tokens(prompt_text)
    output_tokens = estimate_tokens(output_text)
    response_id = generate_id("resp_")
    message_id = generate_id("msg_")
    created = unix_timestamp()
    events: list[str] = []

    events.append(
        openai_sse(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created,
                    "status": "in_progress",
                    "model": scenario.base_model,
                    "output": [],
                },
            }
        )
    )
    events.append(
        openai_sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            }
        )
    )
    events.append(
        openai_sse(
            {
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            }
        )
    )
    for part in chunk_text(output_text):
        events.append(
            openai_sse(
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": part,
                }
            )
        )
    events.append(
        openai_sse(
            {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": output_text,
            }
        )
    )
    events.append(
        openai_sse(
            {
                "type": "response.content_part.done",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": output_text},
            }
        )
    )
    events.append(
        openai_sse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": output_text}],
                },
            }
        )
    )
    events.append(
        openai_sse(
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created,
                    "status": "completed",
                    "model": scenario.base_model,
                    "output": [
                        {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": output_text}],
                        }
                    ],
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                },
            }
        )
    )
    async for event in stream_with_delay(events, scenario_name=scenario.name):
        yield event


def assistant_text(scenario: Scenario, prompt_text: str, *, prefix: str = "Mock response for: ") -> str:
    if scenario.name == "mock-text":
        return f"Hello. It's nice to meet you."
    return f"{prefix}{prompt_text or 'empty prompt'}"


def _tool_stream_events(
    *,
    completion_id: str,
    created: int,
    model: str,
    prompt_tokens: int,
) -> list[str]:
    call_id = generate_id("call_")
    events: list[str] = []
    events.append(
        openai_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
    )
    events.append(
        openai_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": "Write", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
    )
    for fragment in ['{"path":', ' "tmp/mock-file.txt",', ' "contents":', ' "mock output"}']:
        events.append(
            openai_sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
    events.append(
        openai_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 18,
                    "total_tokens": prompt_tokens + 18,
                },
            }
        )
    )
    return events
