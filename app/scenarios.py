from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DEFAULT_SCENARIO = "mock-text"


@dataclass(frozen=True)
class Scenario:
    raw_model: str
    base_model: str
    name: str


def parse_scenario(model: str) -> Scenario:
    if ":mock-" not in model:
        return Scenario(raw_model=model, base_model=model, name=DEFAULT_SCENARIO)

    base_model, scenario_name = model.split(":", 1)
    return Scenario(raw_model=model, base_model=base_model, name=scenario_name or DEFAULT_SCENARIO)


def error_type_for_status(status_code: int) -> Literal["invalid_request_error", "authentication_error", "rate_limit_error", "api_error"]:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 429:
        return "rate_limit_error"
    return "api_error"
