from __future__ import annotations

import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

MOCK_API_ROOT = Path(__file__).resolve().parents[1]
if str(MOCK_API_ROOT) not in sys.path:
    sys.path.insert(0, str(MOCK_API_ROOT))

from app.anthropic_app import DEFAULT_PORT as ANTHROPIC_PORT, create_app as create_anthropic_app
from app.openai_app import DEFAULT_PORT as OPENAI_PORT, create_app as create_openai_app


@pytest.fixture()
def openai_client() -> TestClient:
    with TestClient(create_openai_app()) as test_client:
        yield test_client


@pytest.fixture()
def anthropic_client() -> TestClient:
    with TestClient(create_anthropic_app()) as test_client:
        yield test_client


@pytest.fixture()
def client(openai_client: TestClient) -> TestClient:
    return openai_client


def test_default_ports_are_separate() -> None:
    assert OPENAI_PORT == 8011
    assert ANTHROPIC_PORT == 8012
    assert OPENAI_PORT != ANTHROPIC_PORT
