from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DEFAULT_SCENARIO = "mock-text"
PROVIDERS = ("openai", "anthropic")
Provider = Literal["openai", "anthropic"]
ResponseMode = Literal["text", "tool call", "stream", "delay", "error"]


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    description: str
    provider: Provider
    endpoint: str
    response_mode: ResponseMode
    example_model: str
    status_code: int | None = None


SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        "mock-text",
        "Returns a fixed greeting.",
        "openai",
        "/v1/chat/completions",
        "text",
        "gpt-4.1:mock-text",
    ),
    ScenarioDefinition(
        "mock-tool-call",
        "Returns one function tool call.",
        "openai",
        "/v1/chat/completions",
        "tool call",
        "gpt-4.1:mock-tool-call",
    ),
    ScenarioDefinition(
        "mock-tool-stream",
        "Streams function tool-call deltas.",
        "openai",
        "/v1/chat/completions",
        "stream",
        "gpt-4.1:mock-tool-stream",
    ),
    ScenarioDefinition(
        "mock-bad-tool-json",
        "Returns a tool call with intentionally malformed JSON arguments.",
        "openai",
        "/v1/chat/completions",
        "tool call",
        "gpt-4.1:mock-bad-tool-json",
    ),
    ScenarioDefinition(
        "mock-delay",
        "Adds deterministic delay between streaming events.",
        "openai",
        "/v1/chat/completions",
        "delay",
        "gpt-4.1:mock-delay",
    ),
    ScenarioDefinition(
        "mock-400",
        "Returns an invalid-request error.",
        "openai",
        "/v1/chat/completions",
        "error",
        "gpt-4.1:mock-400",
        400,
    ),
    ScenarioDefinition(
        "mock-401",
        "Returns an authentication error.",
        "openai",
        "/v1/chat/completions",
        "error",
        "gpt-4.1:mock-401",
        401,
    ),
    ScenarioDefinition(
        "mock-429",
        "Returns a rate-limit error.",
        "openai",
        "/v1/chat/completions",
        "error",
        "gpt-4.1:mock-429",
        429,
    ),
    ScenarioDefinition(
        "mock-500",
        "Returns an API error.",
        "openai",
        "/v1/chat/completions",
        "error",
        "gpt-4.1:mock-500",
        500,
    ),
    ScenarioDefinition(
        "mock-503",
        "Returns a service-unavailable error.",
        "openai",
        "/v1/chat/completions",
        "error",
        "gpt-4.1:mock-503",
        503,
    ),
    ScenarioDefinition(
        "mock-text",
        "Returns a fixed greeting.",
        "anthropic",
        "/v1/messages",
        "text",
        "claude-sonnet-4-20250514:mock-text",
    ),
    ScenarioDefinition(
        "mock-anthropic-stream",
        "Streams text in Anthropic event order.",
        "anthropic",
        "/v1/messages",
        "stream",
        "claude-sonnet-4-20250514:mock-anthropic-stream",
    ),
    ScenarioDefinition(
        "mock-tool-use",
        "Returns one tool-use content block.",
        "anthropic",
        "/v1/messages",
        "tool call",
        "claude-sonnet-4-20250514:mock-tool-use",
    ),
    ScenarioDefinition(
        "mock-tool-stream",
        "Streams tool-use JSON deltas.",
        "anthropic",
        "/v1/messages",
        "stream",
        "claude-sonnet-4-20250514:mock-tool-stream",
    ),
    ScenarioDefinition(
        "mock-delay",
        "Adds deterministic delay between streaming events.",
        "anthropic",
        "/v1/messages",
        "delay",
        "claude-sonnet-4-20250514:mock-delay",
    ),
    ScenarioDefinition(
        "mock-stream-error",
        "Emits an API error event after a streaming request starts.",
        "anthropic",
        "/v1/messages",
        "error",
        "claude-sonnet-4-20250514:mock-stream-error",
    ),
    ScenarioDefinition(
        "mock-400",
        "Returns an invalid-request error.",
        "anthropic",
        "/v1/messages",
        "error",
        "claude-sonnet-4-20250514:mock-400",
        400,
    ),
    ScenarioDefinition(
        "mock-401",
        "Returns an authentication error.",
        "anthropic",
        "/v1/messages",
        "error",
        "claude-sonnet-4-20250514:mock-401",
        401,
    ),
    ScenarioDefinition(
        "mock-429",
        "Returns a rate-limit error.",
        "anthropic",
        "/v1/messages",
        "error",
        "claude-sonnet-4-20250514:mock-429",
        429,
    ),
    ScenarioDefinition(
        "mock-500",
        "Returns an API error.",
        "anthropic",
        "/v1/messages",
        "error",
        "claude-sonnet-4-20250514:mock-500",
        500,
    ),
    ScenarioDefinition(
        "mock-503",
        "Returns a service-unavailable error.",
        "anthropic",
        "/v1/messages",
        "error",
        "claude-sonnet-4-20250514:mock-503",
        503,
    ),
)


def validate_scenario_registry() -> None:
    seen_models: set[str] = set()
    for scenario in SCENARIOS:
        if not scenario.name.startswith("mock-"):
            raise ValueError(f"Scenario names must start with mock-: {scenario.name}")
        if scenario.provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {scenario.provider}")
        if not scenario.endpoint.startswith("/v1/"):
            raise ValueError(
                f"Scenario endpoint must be versioned: {scenario.endpoint}"
            )
        if not scenario.example_model.endswith(scenario.name):
            raise ValueError(f"Example model must select {scenario.name}")
        if scenario.example_model in seen_models:
            raise ValueError(f"Duplicate example model: {scenario.example_model}")
        seen_models.add(scenario.example_model)
        if scenario.status_code is not None and scenario.response_mode != "error":
            raise ValueError("Only error scenarios may define a status code")


def scenarios_for_provider(provider: Provider) -> tuple[ScenarioDefinition, ...]:
    return tuple(scenario for scenario in SCENARIOS if scenario.provider == provider)


def find_scenario(name: str, provider: Provider) -> ScenarioDefinition | None:
    return next(
        (
            scenario
            for scenario in SCENARIOS
            if scenario.name == name and scenario.provider == provider
        ),
        None,
    )


@dataclass(frozen=True)
class Scenario:
    raw_model: str
    base_model: str
    name: str


def parse_scenario(model: str) -> Scenario:
    if ":mock-" not in model:
        return Scenario(raw_model=model, base_model=model, name=DEFAULT_SCENARIO)

    base_model, scenario_name = model.split(":", 1)
    return Scenario(
        raw_model=model, base_model=base_model, name=scenario_name or DEFAULT_SCENARIO
    )


def error_type_for_status(
    status_code: int,
) -> Literal[
    "invalid_request_error", "authentication_error", "rate_limit_error", "api_error"
]:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 429:
        return "rate_limit_error"
    return "api_error"


validate_scenario_registry()
