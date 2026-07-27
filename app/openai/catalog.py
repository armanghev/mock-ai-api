from __future__ import annotations

from typing import Any

from app.common import generate_id, unix_timestamp
from app.scenarios import (
    Scenario,
    error_type_for_status,
    find_scenario,
    scenarios_for_provider,
)

OPENAI_MODELS: list[dict[str, Any]] = [
    {"id": "gpt-4.1", "object": "model", "created": 1686935002, "owned_by": "openai"},
    {
        "id": "gpt-4o-mini",
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    },
    {
        "id": "gpt-3.5-turbo-instruct",
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    },
    {
        "id": "text-embedding-3-small",
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    },
    {
        "id": "omni-moderation-latest",
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    },
    {
        "id": "gpt-image-1",
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    },
    {
        "id": "gpt-4o-mini-tts",
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    },
    {
        "id": "gpt-4o-transcribe",
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    },
    {"id": "whisper-1", "object": "model", "created": 1686935002, "owned_by": "openai"},
]

OPENAI_MODELS.extend(
    {
        "id": scenario.example_model,
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    }
    for scenario in scenarios_for_provider("openai")
)


def list_models() -> dict[str, Any]:
    return {"object": "list", "data": OPENAI_MODELS}


def get_model(model_id: str) -> dict[str, Any] | None:
    for model in OPENAI_MODELS:
        if model["id"] == model_id:
            return model
    return None


def make_error(status_code: int, scenario_name: str) -> dict[str, Any]:
    return {
        "status_code": status_code,
        "body": {
            "error": {
                "message": f"Mock scenario {scenario_name} triggered",
                "type": error_type_for_status(status_code),
                "code": f"mock_{status_code}",
            }
        },
    }


def maybe_raise_error(scenario: Scenario) -> dict[str, Any] | None:
    definition = find_scenario(scenario.name, "openai")
    if definition is None or definition.status_code is None:
        return None
    return make_error(definition.status_code, scenario.name)


class ResponseStore:
    def __init__(self) -> None:
        self._responses: dict[str, dict[str, Any]] = {}

    def save(self, response: dict[str, Any]) -> dict[str, Any]:
        self._responses[response["id"]] = response
        return response

    def get(self, response_id: str) -> dict[str, Any] | None:
        return self._responses.get(response_id)

    def cancel(self, response_id: str) -> dict[str, Any] | None:
        response = self._responses.get(response_id)
        if response is None:
            return None
        response = {**response, "status": "cancelled"}
        self._responses[response_id] = response
        return response


response_store = ResponseStore()


def build_response_record(
    *,
    model: str,
    input_value: Any,
    output_text: str,
    scenario: Scenario,
) -> dict[str, Any]:
    from app.common import estimate_tokens, extract_input_text

    prompt_text = extract_input_text(input_value)
    input_tokens = estimate_tokens(prompt_text)
    output_tokens = estimate_tokens(output_text)
    response_id = generate_id("resp_")
    return {
        "id": response_id,
        "object": "response",
        "created_at": unix_timestamp(),
        "status": "completed",
        "model": scenario.base_model,
        "output": [
            {
                "id": generate_id("msg_"),
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
        "_input": input_value,
    }


def build_input_items(response: dict[str, Any]) -> dict[str, Any]:
    input_value = response.get("_input", [])
    items: list[dict[str, Any]] = []
    if isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, dict) and item.get("role"):
                content = item.get("content", [])
                if isinstance(content, str):
                    normalized = [{"type": "input_text", "text": content}]
                elif isinstance(content, list):
                    normalized = content
                else:
                    normalized = [{"type": "input_text", "text": str(content)}]
                items.append(
                    {"type": "message", "role": item["role"], "content": normalized}
                )
    elif isinstance(input_value, str) and input_value:
        items.append(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_value}],
            }
        )
    return {"object": "list", "data": items}
