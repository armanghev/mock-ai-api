from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/chat/completions", {"messages": []}),
        (
            "/v1/chat/completions",
            {"model": "gpt-4.1", "messages": "not a list"},
        ),
    ],
)
def test_openai_validation_errors_have_provider_shape(
    openai_client, path, payload
) -> None:
    response = openai_client.post(path, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "Traceback" not in response.text


def test_openai_malformed_json_has_provider_shape(openai_client) -> None:
    response = openai_client.post(
        "/v1/chat/completions",
        content='{"model":',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "Traceback" not in response.text


def test_openai_unknown_resource_has_provider_shape(openai_client) -> None:
    response = openai_client.get("/v1/not-a-resource")

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("path", ["/v1/responses", "/v1/embeddings"])
def test_openai_json_object_routes_reject_array_bodies(openai_client, path) -> None:
    response = openai_client.post(path, json=[])

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "Traceback" not in response.text


@pytest.mark.parametrize(
    ("scenario", "status", "error_type"),
    [
        ("mock-400", 400, "invalid_request_error"),
        ("mock-401", 401, "authentication_error"),
        ("mock-429", 429, "rate_limit_error"),
        ("mock-500", 500, "api_error"),
        ("mock-503", 503, "api_error"),
    ],
)
def test_openai_scenario_errors_have_provider_shape(
    openai_client, scenario, status, error_type
) -> None:
    response = openai_client.post(
        "/v1/chat/completions",
        json={
            "model": f"gpt-4.1:{scenario}",
            "messages": [{"role": "user", "content": "Fail"}],
        },
    )

    assert response.status_code == status
    assert response.json()["error"]["type"] == error_type
    assert "Traceback" not in response.text


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/messages", {"max_tokens": 10, "messages": []}),
        (
            "/v1/messages",
            {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 10,
                "messages": "not a list",
            },
        ),
    ],
)
def test_anthropic_validation_errors_have_provider_shape(
    anthropic_client, path, payload
) -> None:
    response = anthropic_client.post(path, json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "Traceback" not in response.text


def test_anthropic_malformed_json_has_provider_shape(anthropic_client) -> None:
    response = anthropic_client.post(
        "/v1/messages",
        content='{"model":',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "Traceback" not in response.text


def test_anthropic_unknown_resource_has_provider_shape(anthropic_client) -> None:
    response = anthropic_client.get("/v1/not-a-resource")

    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "not_found_error"


@pytest.mark.parametrize("path", ["/v1/messages/count_tokens", "/v1/messages/batches"])
def test_anthropic_json_object_routes_reject_array_bodies(
    anthropic_client, path
) -> None:
    response = anthropic_client.post(path, json=[])

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "Traceback" not in response.text


@pytest.mark.parametrize(
    ("scenario", "status", "error_type"),
    [
        ("mock-400", 400, "invalid_request_error"),
        ("mock-401", 401, "authentication_error"),
        ("mock-429", 429, "rate_limit_error"),
        ("mock-500", 500, "api_error"),
        ("mock-503", 503, "api_error"),
    ],
)
def test_anthropic_scenario_errors_have_provider_shape(
    anthropic_client, scenario, status, error_type
) -> None:
    response = anthropic_client.post(
        "/v1/messages",
        json={
            "model": f"claude-sonnet-4-20250514:{scenario}",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Fail"}],
        },
    )

    assert response.status_code == status
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == error_type
    assert "Traceback" not in response.text
