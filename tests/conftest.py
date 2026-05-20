from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.services as services
from app.api.v1 import route
from app.main import app


@pytest.fixture(autouse=True)
def reset_runtime_state() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    services.get_nlp_resources.cache_clear()
    yield
    app.dependency_overrides.clear()
    services.get_nlp_resources.cache_clear()


@pytest.fixture
def api_client() -> Generator[TestClient, None, None]:
    with TestClient(app) as client:
        yield client


@pytest.fixture
def override_embedding_service() -> Callable[[Any], Any]:
    def _override(service: Any) -> Any:
        app.dependency_overrides[route.get_embedding_service] = lambda: service
        return service

    return _override
