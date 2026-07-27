from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

DEFAULT_STREAM_DELAY_MS = 40
SLOW_STREAM_DELAY_MS = 80
SLOW_STREAM_INITIAL_DELAY_MS = 200


def generate_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


def unix_timestamp() -> int:
    return int(time.time())


def iso_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_prompt_text(messages: list[dict[str, Any]]) -> str:
    text_parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        text_parts.append(item["text"])
                    elif item.get("type") == "input_text" and isinstance(
                        item.get("text"), str
                    ):
                        text_parts.append(item["text"])
    return "\n".join(part for part in text_parts if part)


def extract_input_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") in {"input_text", "output_text"} and isinstance(
                    item.get("text"), str
                ):
                    parts.append(item["text"])
                elif item.get("role") and item.get("content"):
                    parts.append(extract_input_text(item["content"]))
        return "\n".join(parts)
    return str(value) if value is not None else ""


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 2)


def deterministic_embedding(text: str, dimensions: int = 1536) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    index = 0
    while len(values) < dimensions:
        chunk = digest[index % len(digest) : (index % len(digest)) + 4]
        if len(chunk) < 4:
            chunk = (chunk + digest)[:4]
        raw = int.from_bytes(chunk, "big", signed=False)
        values.append(round((raw / 0xFFFFFFFF) * 2 - 1, 6))
        index += 4
    return values


def chunk_text(text: str, chunk_size: int = 8) -> list[str]:
    words = text.split()
    if not words:
        return [text] if text else []
    chunks: list[str] = []
    for index in range(0, len(words), chunk_size):
        piece = " ".join(words[index : index + chunk_size])
        if index + chunk_size < len(words):
            piece += " "
        chunks.append(piece)
    return chunks or [text]


async def stream_with_delay(
    events: list[str],
    *,
    scenario_name: str,
) -> AsyncIterator[str]:
    if scenario_name == "mock-delay":
        await asyncio.sleep(SLOW_STREAM_INITIAL_DELAY_MS / 1000)
        delay_ms = SLOW_STREAM_DELAY_MS
    else:
        delay_ms = DEFAULT_STREAM_DELAY_MS

    for index, event in enumerate(events):
        if index > 0:
            await asyncio.sleep(delay_ms / 1000)
        yield event


def openai_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def anthropic_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
