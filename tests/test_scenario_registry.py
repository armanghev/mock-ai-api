from __future__ import annotations

from pathlib import Path

from app.anthropic.catalog import ANTHROPIC_MODELS
from app.openai.catalog import OPENAI_MODELS
from app.scenarios import PROVIDERS, SCENARIOS, validate_scenario_registry


def test_scenario_registry_is_valid() -> None:
    validate_scenario_registry()

    assert {scenario.provider for scenario in SCENARIOS} == set(PROVIDERS)
    assert all(scenario.description for scenario in SCENARIOS)
    assert all(scenario.endpoint.startswith("/v1/") for scenario in SCENARIOS)
    assert all(scenario.example_model.endswith(scenario.name) for scenario in SCENARIOS)


def test_model_catalogs_advertise_exactly_their_registered_scenarios() -> None:
    catalogs = {
        "openai": OPENAI_MODELS,
        "anthropic": ANTHROPIC_MODELS,
    }

    for provider, models in catalogs.items():
        advertised = {model["id"] for model in models if ":mock-" in model["id"]}
        registered = {
            scenario.example_model
            for scenario in SCENARIOS
            if scenario.provider == provider
        }
        assert advertised == registered


def test_readme_documents_every_registered_scenario() -> None:
    readme = Path("README.md").read_text()

    for scenario in SCENARIOS:
        assert (
            f"| `{scenario.name}` | {scenario.provider} | `{scenario.endpoint}` |"
            in readme
        )
