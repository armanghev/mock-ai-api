from __future__ import annotations

import json
import time


def test_openai_health(openai_client) -> None:
    response = openai_client.get("/health")
    assert response.status_code == 200
    assert response.json()["provider"] == "openai"


def test_anthropic_health(anthropic_client) -> None:
    response = anthropic_client.get("/health")
    assert response.status_code == 200
    assert response.json()["provider"] == "anthropic"


def test_openai_chat_completion_returns_text_response(openai_client) -> None:
    response = openai_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1:mock-text",
            "messages": [{"role": "user", "content": "Hello mock"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-4.1"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hello. It's nice to meet you."
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0


def test_openai_chat_completion_returns_tool_call_with_bad_json_when_requested(
    openai_client,
) -> None:
    response = openai_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1:mock-bad-tool-json",
            "messages": [{"role": "user", "content": "Call the Write tool"}],
        },
    )

    assert response.status_code == 200
    tool_call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "Write"
    assert tool_call["function"]["arguments"].endswith('"unterminated"')
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_chat_completion_streams_text_and_finishes_with_usage_chunk(
    openai_client,
) -> None:
    started = time.perf_counter()
    with openai_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1:mock-text",
            "stream": True,
            "messages": [{"role": "user", "content": "Stream please"}],
        },
    ) as response:
        chunks = [line for line in response.iter_lines() if line]

    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    data_lines = [line for line in chunks if line.startswith("data: ")]
    assert data_lines[-1] == "data: [DONE]"

    payloads = [json.loads(line.removeprefix("data: ")) for line in data_lines[:-1]]
    assert payloads[0]["object"] == "chat.completion.chunk"
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(
        payload["choices"][0]["delta"].get("content")
        for payload in payloads
        if payload["choices"]
    )
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["usage"]["prompt_tokens"] > 0
    assert elapsed >= 0.08


def test_openai_chat_completion_streams_tool_call_deltas(openai_client) -> None:
    with openai_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1:mock-tool-stream",
            "stream": True,
            "messages": [{"role": "user", "content": "Use a tool"}],
        },
    ) as response:
        chunks = [line for line in response.iter_lines() if line.startswith("data: {")]

    assert response.status_code == 200
    payloads = [json.loads(line.removeprefix("data: ")) for line in chunks]
    tool_delta_payloads = [
        payload
        for payload in payloads
        if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_delta_payloads
    assert (
        tool_delta_payloads[0]["choices"][0]["delta"]["tool_calls"][0]["function"][
            "name"
        ]
        == "Write"
    )
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_chat_completion_can_return_rate_limit_error(openai_client) -> None:
    response = openai_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1:mock-429",
            "messages": [{"role": "user", "content": "fail"}],
        },
    )

    assert response.status_code == 429
    body = response.json()
    assert body["error"]["type"] == "rate_limit_error"


def test_openai_legacy_completion(openai_client) -> None:
    response = openai_client.post(
        "/v1/completions",
        json={
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Write a tagline for an ice cream shop.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_openai_models_list_and_retrieve(openai_client) -> None:
    list_response = openai_client.get("/v1/models")
    assert list_response.status_code == 200
    assert list_response.json()["object"] == "list"
    assert any(model["id"] == "gpt-4.1" for model in list_response.json()["data"])

    retrieve_response = openai_client.get("/v1/models/gpt-4.1")
    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["id"] == "gpt-4.1"


def test_openai_moderations(openai_client) -> None:
    response = openai_client.post(
        "/v1/moderations",
        json={
            "model": "omni-moderation-latest",
            "input": "I would like to hurt someone.",
        },
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["flagged"] is True


def test_anthropic_message_returns_tool_use_response(anthropic_client) -> None:
    response = anthropic_client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514:mock-tool-use",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Write a file"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["model"] == "claude-sonnet-4-20250514"
    assert body["content"][0]["type"] == "tool_use"
    assert body["stop_reason"] == "tool_use"
    assert body["usage"]["input_tokens"] > 0


def test_anthropic_message_streams_events_in_expected_order(anthropic_client) -> None:
    started = time.perf_counter()
    with anthropic_client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514:mock-anthropic-stream",
            "stream": True,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Stream please"}],
        },
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    event_lines = [line for line in lines if line.startswith("event: ")]
    assert event_lines[0] == "event: ping"
    assert "event: message_start" in event_lines
    assert "event: content_block_start" in event_lines
    assert "event: content_block_delta" in event_lines
    assert event_lines[-1] == "event: message_stop"
    assert elapsed >= 0.08


def test_anthropic_tool_stream_emits_input_json_deltas(anthropic_client) -> None:
    with anthropic_client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514:mock-tool-stream",
            "stream": True,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Use a tool"}],
        },
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert "event: content_block_delta" in lines
    delta_payloads = []
    for index, line in enumerate(lines):
        if line == "event: content_block_delta" and index + 1 < len(lines):
            delta_payloads.append(json.loads(lines[index + 1].removeprefix("data: ")))
    assert any(
        payload["delta"]["type"] == "input_json_delta" for payload in delta_payloads
    )


def test_anthropic_stream_can_emit_error_event(anthropic_client) -> None:
    with anthropic_client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514:mock-stream-error",
            "stream": True,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Break"}],
        },
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert "event: error" in lines
    error_payload_line = next(line for line in lines if line.startswith("data: "))
    error_payload = json.loads(error_payload_line.removeprefix("data: "))
    assert error_payload["type"] == "error"
    assert error_payload["error"]["type"] == "api_error"


def test_openai_embeddings_returns_realistic_embedding(openai_client) -> None:
    response = openai_client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello embeddings"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"][0]["embedding"]) == 1536


def test_openai_responses_create_and_retrieve(openai_client) -> None:
    create_response = openai_client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Write a one-line summary of ModelPort.",
                        }
                    ],
                }
            ],
        },
    )

    assert create_response.status_code == 200
    body = create_response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["type"] == "output_text"

    retrieve_response = openai_client.get(f"/v1/responses/{body['id']}")
    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["id"] == body["id"]

    input_items_response = openai_client.get(f"/v1/responses/{body['id']}/input_items")
    assert input_items_response.status_code == 200
    assert input_items_response.json()["data"][0]["role"] == "user"


def test_anthropic_count_tokens_returns_input_tokens(anthropic_client) -> None:
    response = anthropic_client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hello tokens"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


def test_anthropic_models_and_batch_flow(anthropic_client) -> None:
    models_response = anthropic_client.get("/v1/models")
    assert models_response.status_code == 200
    assert models_response.json()["data"][0]["type"] == "model"

    batch_response = anthropic_client.post(
        "/v1/messages/batches",
        json={
            "requests": [
                {
                    "custom_id": "job-1",
                    "params": {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 128,
                        "messages": [
                            {"role": "user", "content": "Summarize this in five words."}
                        ],
                    },
                }
            ]
        },
    )
    assert batch_response.status_code == 200
    batch_id = batch_response.json()["id"]

    results_response = anthropic_client.get(f"/v1/messages/batches/{batch_id}/results")
    assert results_response.status_code == 200
    assert "custom_id" in results_response.text


def test_anthropic_file_upload_and_download(anthropic_client) -> None:
    upload_response = anthropic_client.post(
        "/v1/files",
        files={"file": ("document.pdf", b"%PDF-1.4 mock", "application/pdf")},
    )
    assert upload_response.status_code == 200
    file_id = upload_response.json()["id"]

    download_response = anthropic_client.get(f"/v1/files/{file_id}/content")
    assert download_response.status_code == 200
    assert download_response.content.startswith(b"%PDF")

    delete_response = anthropic_client.delete(f"/v1/files/{file_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["type"] == "file_deleted"
