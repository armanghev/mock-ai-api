from __future__ import annotations

from typing import Any

from app.common import generate_id, iso_timestamp
from app.scenarios import Scenario, error_type_for_status

ANTHROPIC_MODELS: list[dict[str, Any]] = [
    {
        "id": "claude-opus-4-6",
        "display_name": "Claude Opus 4.6",
        "created_at": "2025-01-01T00:00:00Z",
        "type": "model",
    },
    {
        "id": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
        "created_at": "2025-05-14T00:00:00Z",
        "type": "model",
    },
    {
        "id": "claude-sonnet-4-20250514:mock-anthropic-stream",
        "display_name": "Claude Sonnet 4 (mock stream)",
        "created_at": "2025-05-14T00:00:00Z",
        "type": "model",
    },
    {
        "id": "claude-sonnet-4-20250514:mock-tool-use",
        "display_name": "Claude Sonnet 4 (mock tool use)",
        "created_at": "2025-05-14T00:00:00Z",
        "type": "model",
    },
    {
        "id": "claude-sonnet-4-20250514:mock-tool-stream",
        "display_name": "Claude Sonnet 4 (mock tool stream)",
        "created_at": "2025-05-14T00:00:00Z",
        "type": "model",
    },
    {
        "id": "claude-sonnet-4-20250514:mock-429",
        "display_name": "Claude Sonnet 4 (mock 429)",
        "created_at": "2025-05-14T00:00:00Z",
        "type": "model",
    },
]


def list_models() -> dict[str, Any]:
    data = ANTHROPIC_MODELS
    return {
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "has_more": False,
        "last_id": data[-1]["id"] if data else None,
    }


def get_model(model_id: str) -> dict[str, Any] | None:
    for model in ANTHROPIC_MODELS:
        if model["id"] == model_id:
            return model
    return None


def maybe_raise_error(scenario: Scenario) -> dict[str, Any] | None:
    status_map = {
        "mock-400": 400,
        "mock-401": 401,
        "mock-429": 429,
        "mock-500": 500,
        "mock-503": 503,
    }
    status_code = status_map.get(scenario.name)
    if status_code is None:
        return None
    return {
        "status_code": status_code,
        "body": {
            "type": "error",
            "error": {
                "type": error_type_for_status(status_code),
                "message": f"Mock scenario {scenario.name} triggered",
            },
        },
    }


class BatchStore:
    def __init__(self) -> None:
        self._batches: dict[str, dict[str, Any]] = {}

    def create(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        batch_id = generate_id("msgbatch_")
        batch = {
            "id": batch_id,
            "type": "message_batch",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": len(requests),
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "created_at": iso_timestamp(),
            "expires_at": iso_timestamp(),
            "results_url": f"/v1/messages/batches/{batch_id}/results",
            "_requests": requests,
        }
        self._batches[batch_id] = batch
        return batch

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._batches.values())

    def get(self, batch_id: str) -> dict[str, Any] | None:
        return self._batches.get(batch_id)

    def cancel(self, batch_id: str) -> dict[str, Any] | None:
        batch = self._batches.get(batch_id)
        if batch is None:
            return None
        batch = {**batch, "processing_status": "canceling"}
        self._batches[batch_id] = batch
        return batch

    def delete(self, batch_id: str) -> bool:
        return self._batches.pop(batch_id, None) is not None

    def results(self, batch_id: str) -> list[dict[str, Any]] | None:
        batch = self._batches.get(batch_id)
        if batch is None:
            return None
        lines: list[dict[str, Any]] = []
        for request_item in batch.get("_requests", []):
            custom_id = request_item.get("custom_id", "job-1")
            params = request_item.get("params", {})
            model = params.get("model", "claude-sonnet-4-20250514")
            messages = params.get("messages", [])
            from app.common import estimate_tokens, extract_prompt_text

            prompt_text = extract_prompt_text(messages)
            response_text = "A unified model API gateway."
            lines.append(
                {
                    "custom_id": custom_id,
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "id": generate_id("msg_"),
                            "type": "message",
                            "role": "assistant",
                            "model": model.split(":")[0] if ":" in model else model,
                            "content": [{"type": "text", "text": response_text}],
                            "stop_reason": "end_turn",
                            "usage": {
                                "input_tokens": estimate_tokens(prompt_text),
                                "output_tokens": estimate_tokens(response_text),
                            },
                        },
                    },
                }
            )
        ended_batch = {
            **batch,
            "processing_status": "ended",
            "ended_at": iso_timestamp(),
            "request_counts": {
                "processing": 0,
                "succeeded": len(lines),
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
        }
        self._batches[batch_id] = ended_batch
        return lines


class FileStore:
    def __init__(self) -> None:
        self._files: dict[str, dict[str, Any]] = {}

    def create(self, filename: str, mime_type: str, size_bytes: int, content: bytes) -> dict[str, Any]:
        file_id = generate_id("file_")
        record = {
            "id": file_id,
            "type": "file",
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "created_at": iso_timestamp(),
            "downloadable": True,
            "_content": content,
        }
        self._files[file_id] = record
        return record

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._files.values())

    def get(self, file_id: str) -> dict[str, Any] | None:
        return self._files.get(file_id)

    def delete(self, file_id: str) -> dict[str, Any] | None:
        record = self._files.pop(file_id, None)
        if record is None:
            return None
        return {"id": file_id, "type": "file_deleted"}


batch_store = BatchStore()
file_store = FileStore()


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}
