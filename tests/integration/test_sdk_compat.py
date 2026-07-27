from __future__ import annotations

import json

import pytest

from anthropic import Anthropic, RateLimitError as AnthropicRateLimitError
from openai import OpenAI, RateLimitError as OpenAIRateLimitError


def test_openai_sdk_chat_completion(openai_base_url: str) -> None:
    client = OpenAI(api_key="mock-key", base_url=openai_base_url)

    completion = client.chat.completions.create(
        model="gpt-4.1:mock-text",
        messages=[{"role": "user", "content": "Hello mock"}],
    )

    assert completion.choices[0].message.content == "Hello. It's nice to meet you."


def test_openai_sdk_streams_complete_chat_sequence(openai_base_url: str) -> None:
    client = OpenAI(api_key="mock-key", base_url=openai_base_url)

    stream = client.chat.completions.create(
        model="gpt-4.1:mock-text",
        messages=[{"role": "user", "content": "Stream please"}],
        stream=True,
    )
    chunks = list(stream)

    assert chunks[0].choices[0].delta.role == "assistant"
    assert (
        "".join(
            chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
        )
        == "Hello. It's nice to meet you."
    )
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_openai_sdk_tool_call(openai_base_url: str) -> None:
    client = OpenAI(api_key="mock-key", base_url=openai_base_url)

    completion = client.chat.completions.create(
        model="gpt-4.1:mock-tool-call",
        messages=[{"role": "user", "content": "Write a file"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert completion.choices[0].finish_reason == "tool_calls"
    tool_call = completion.choices[0].message.tool_calls[0]
    assert tool_call.id.startswith("call_")
    assert tool_call.type == "function"
    assert tool_call.function.name == "Write"
    assert json.loads(tool_call.function.arguments) == {
        "path": "tmp/mock-file.txt",
        "contents": "mock output",
    }


def test_openai_sdk_surfaces_rate_limit(openai_base_url: str) -> None:
    client = OpenAI(api_key="mock-key", base_url=openai_base_url, max_retries=0)

    with pytest.raises(OpenAIRateLimitError):
        client.chat.completions.create(
            model="gpt-4.1:mock-429",
            messages=[{"role": "user", "content": "Fail"}],
        )


def test_openai_sdk_responses_create_and_retrieve(openai_base_url: str) -> None:
    client = OpenAI(api_key="mock-key", base_url=openai_base_url)

    created = client.responses.create(model="gpt-4.1:mock-text", input="Hello")
    retrieved = client.responses.retrieve(created.id)

    assert retrieved.id == created.id
    assert retrieved.output_text == "Hello. It's nice to meet you."


def test_openai_sdk_embeddings(openai_base_url: str) -> None:
    client = OpenAI(api_key="mock-key", base_url=openai_base_url)

    response = client.embeddings.create(
        model="text-embedding-3-small", input="Hello mock"
    )

    assert len(response.data) == 1
    assert response.data[0].embedding


def test_anthropic_sdk_message(anthropic_base_url: str) -> None:
    client = Anthropic(api_key="mock-key", base_url=anthropic_base_url)

    message = client.messages.create(
        model="claude-sonnet-4-20250514:mock-text",
        max_tokens=128,
        messages=[{"role": "user", "content": "Hello mock"}],
    )

    assert message.content[0].text == "Hello. It's nice to meet you."


def test_anthropic_sdk_streams_events_in_order(anthropic_base_url: str) -> None:
    client = Anthropic(api_key="mock-key", base_url=anthropic_base_url)

    with client.messages.stream(
        model="claude-sonnet-4-20250514:mock-anthropic-stream",
        max_tokens=128,
        messages=[{"role": "user", "content": "Stream please"}],
    ) as stream:
        event_types = [event.type for event in stream]

    assert event_types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "text",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


def test_anthropic_sdk_tool_use(anthropic_base_url: str) -> None:
    client = Anthropic(api_key="mock-key", base_url=anthropic_base_url)

    message = client.messages.create(
        model="claude-sonnet-4-20250514:mock-tool-use",
        max_tokens=128,
        messages=[{"role": "user", "content": "Write a file"}],
        tools=[
            {
                "name": "Write",
                "description": "Write a file",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    assert message.stop_reason == "tool_use"
    assert message.content[0].name == "Write"


def test_anthropic_sdk_surfaces_rate_limit(anthropic_base_url: str) -> None:
    client = Anthropic(api_key="mock-key", base_url=anthropic_base_url, max_retries=0)

    with pytest.raises(AnthropicRateLimitError):
        client.messages.create(
            model="claude-sonnet-4-20250514:mock-429",
            max_tokens=128,
            messages=[{"role": "user", "content": "Fail"}],
        )


def test_anthropic_sdk_counts_tokens(anthropic_base_url: str) -> None:
    client = Anthropic(api_key="mock-key", base_url=anthropic_base_url)

    count = client.messages.count_tokens(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Count these words"}],
    )

    assert count.input_tokens > 0


def test_anthropic_sdk_uploads_and_downloads_file(anthropic_base_url: str) -> None:
    client = Anthropic(api_key="mock-key", base_url=anthropic_base_url)
    contents = b"%PDF-1.4 mock"

    uploaded = client.beta.files.upload(
        file=("document.pdf", contents, "application/pdf")
    )
    downloaded = client.beta.files.download(uploaded.id)

    assert uploaded.id.startswith("file_")
    assert uploaded.filename == "document.pdf"
    assert uploaded.mime_type == "application/pdf"
    assert uploaded.size_bytes == len(contents)
    assert uploaded.downloadable is True
    assert downloaded.read() == contents
