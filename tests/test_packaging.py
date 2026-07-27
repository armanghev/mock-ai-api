from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_defines_installable_cli_and_dependencies() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    metadata = project["project"]
    assert metadata["requires-python"] == ">=3.11"
    assert {
        "fastapi==0.115.14",
        "httpx>=0.22.0",
        "python-multipart==0.0.32",
    } <= set(metadata["dependencies"])
    assert metadata["scripts"] == {"mock-ai-api": "server:main"}
    dev_dependencies = metadata["optional-dependencies"]["dev"]
    assert {"pytest", "pytest-asyncio", "ruff"} <= {
        dependency.split("==", maxsplit=1)[0].split(">=", maxsplit=1)[0]
        for dependency in dev_dependencies
    }
    assert "ruff==0.16.0" in dev_dependencies
    assert project["tool"]["setuptools"]["py-modules"] == ["server"]
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["app*"]


def test_requirements_file_matches_project_runtime_dependencies() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    requirements = {
        line
        for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    }

    assert requirements == set(project["project"]["dependencies"])
