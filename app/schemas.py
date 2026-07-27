from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None


class OpenAILegacyCompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None


class AnthropicMessageRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int | None = None
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
