from __future__ import annotations

from http import HTTPStatus

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.v1 import route
from app.main import app
from app.providers.fasttext import FastTextModelLoadError
from app.services import (
    EmbeddingInferenceError,
    EmbeddingInputError,
    EmbeddingNotFoundError,
)


class StubService:
    def __init__(
        self, embeddings: np.ndarray | None = None, error: Exception | None = None
    ) -> None:
        self._embeddings = embeddings
        self._error = error
        self.texts: list[str] | None = None

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        self.texts = texts
        if self._error is not None:
            raise self._error
        if self._embeddings is None:
            raise AssertionError("Embeddings must be provided for success path tests.")
        return self._embeddings


def _override_service(service: StubService) -> None:
    app.dependency_overrides[route.get_embedding_service] = lambda: service


def test_root_health_returns_ok() -> None:
    app.dependency_overrides.clear()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"status": "ok"}


def test_get_embedding_success_returns_vector() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(embeddings=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/embedding/hello")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"word": "hello", "embedding": [1.0, 2.0, 3.0]}
    app.dependency_overrides.clear()


def test_get_embedding_invalid_input_returns_400() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(error=EmbeddingInputError("Text entries must not be blank."))
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/embedding/blank")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json() == {"detail": "Text entries must not be blank."}
    app.dependency_overrides.clear()


def test_get_embedding_oov_returns_404() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(
            error=EmbeddingNotFoundError("Word is out-of-vocabulary: 'unknown'")
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/embedding/unknown")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json() == {"detail": "Word is out-of-vocabulary: 'unknown'"}
    app.dependency_overrides.clear()


def test_get_similarity_success_returns_cosine_similarity() -> None:
    app.dependency_overrides.clear()
    service = StubService(
        embeddings=np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    _override_service(service)

    with TestClient(app) as client:
        response = client.get("/api/v1/similarity/king/queen")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["similarity"] == pytest.approx(0.7071067811865475)
    assert service.texts == ["king", "queen"]
    app.dependency_overrides.clear()


def test_get_similarity_invalid_input_returns_400() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(error=EmbeddingInputError("Text entries must not be blank."))
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similarity/king/blank")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json() == {"detail": "Text entries must not be blank."}
    app.dependency_overrides.clear()


def test_get_similarity_oov_returns_404() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(
            error=EmbeddingNotFoundError("Word is out-of-vocabulary: 'unknown'")
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similarity/king/unknown")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json() == {"detail": "Word is out-of-vocabulary: 'unknown'"}
    app.dependency_overrides.clear()


def test_get_similarity_zero_norm_vector_returns_422() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(
            embeddings=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similarity/zero/queen")

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json() == {
        "detail": "Cannot compute similarity for zero-norm vector.",
    }
    app.dependency_overrides.clear()


def test_model_loading_failure_in_dependency_returns_500(
    monkeypatch,
) -> None:
    app.dependency_overrides.clear()

    def broken_provider():
        raise FastTextModelLoadError("FastText model file not found")

    monkeypatch.setattr(route, "get_embedding_provider", broken_provider)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/embedding/hello")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    app.dependency_overrides.clear()
