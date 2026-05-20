from __future__ import annotations

from http import HTTPStatus

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.v1 import route
from app.main import app
from app.nlp.exceptions import EmbeddingModelLoadError
from app.services import (
    EmbeddingInferenceError,
    EmbeddingInputError,
    EmbeddingNotFoundError,
    EmbeddingRankError,
    NthSimilarWordResult,
    SimilarityRankResult,
)


class StubService:
    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        error: Exception | None = None,
        rank_result: SimilarityRankResult | None = None,
        rank_error: Exception | None = None,
        nth_result: NthSimilarWordResult | None = None,
        nth_error: Exception | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._error = error
        self._rank_result = rank_result
        self._rank_error = rank_error
        self._nth_result = nth_result
        self._nth_error = nth_error
        self.texts: list[str] | None = None
        self.rank_words: tuple[str, str] | None = None
        self.nth_args: tuple[str, int] | None = None

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        self.texts = texts
        if self._error is not None:
            raise self._error
        if self._embeddings is None:
            raise AssertionError("Embeddings must be provided for success path tests.")
        return self._embeddings

    def calculate_similarity_rank(
        self,
        base_word: str,
        compared_word: str,
    ) -> SimilarityRankResult:
        self.rank_words = (base_word, compared_word)
        if self._rank_error is not None:
            raise self._rank_error
        if self._rank_result is None:
            raise AssertionError("Rank result must be provided for success path tests.")
        return self._rank_result

    def find_nth_similar_word(
        self,
        base_word: str,
        rank: int,
    ) -> NthSimilarWordResult:
        self.nth_args = (base_word, rank)
        if self._nth_error is not None:
            raise self._nth_error
        if self._nth_result is None:
            raise AssertionError("Nth similar word result must be provided.")
        return self._nth_result


def _override_service(service: StubService) -> None:
    app.dependency_overrides[route.get_embedding_service] = lambda: service


def _pop_elapsed_time_ms(payload: dict[str, object]) -> float:
    elapsed_time_ms = payload.pop("elapsed_time_ms")
    assert isinstance(elapsed_time_ms, float)
    assert elapsed_time_ms >= 0
    return elapsed_time_ms


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


def test_get_word_detail_success_returns_embedding() -> None:
    app.dependency_overrides.clear()
    service = StubService(embeddings=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    _override_service(service)

    with TestClient(app) as client:
        response = client.get("/api/v1/word/hello")

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    _pop_elapsed_time_ms(payload)
    assert payload == {"word": "hello", "embedding": [1.0, 2.0, 3.0]}
    assert service.texts == ["hello"]
    app.dependency_overrides.clear()


def test_get_word_detail_invalid_input_returns_400() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(error=EmbeddingInputError("Text entries must not be blank."))
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/word/blank")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json() == {"detail": "Text entries must not be blank."}
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


def test_get_similarity_rank_success_returns_rank() -> None:
    app.dependency_overrides.clear()
    service = StubService(
        rank_result=SimilarityRankResult(
            base_word="king",
            compared_word="queen",
            rank=12,
            similarity=0.71,
            vocabulary_size=1000,
        )
    )
    _override_service(service)

    with TestClient(app) as client:
        response = client.get("/api/v1/similarity-rank/king/queen")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "base_word": "king",
        "compared_word": "queen",
        "rank": 12,
        "similarity": 0.71,
        "vocabulary_size": 1000,
    }
    assert service.rank_words == ("king", "queen")
    app.dependency_overrides.clear()


def test_get_similarity_rank_invalid_input_returns_400() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(rank_error=EmbeddingInputError("Text entries must not be blank."))
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similarity-rank/king/blank")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json() == {"detail": "Text entries must not be blank."}
    app.dependency_overrides.clear()


def test_get_similarity_rank_oov_returns_404() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(
            rank_error=EmbeddingNotFoundError("Word is out-of-vocabulary: 'unknown'")
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similarity-rank/king/unknown")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json() == {"detail": "Word is out-of-vocabulary: 'unknown'"}
    app.dependency_overrides.clear()


def test_get_similarity_rank_provider_error_returns_502() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(rank_error=EmbeddingInferenceError("Provider failed."))
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similarity-rank/king/queen")

    assert response.status_code == HTTPStatus.BAD_GATEWAY
    assert response.json() == {"detail": "Provider failed."}
    app.dependency_overrides.clear()


def test_get_nth_similar_word_success_returns_word() -> None:
    app.dependency_overrides.clear()
    service = StubService(
        nth_result=NthSimilarWordResult(
            base_word="apple",
            rank=3,
            word="fruit",
            similarity=0.82,
            vocabulary_size=1000,
        )
    )
    _override_service(service)

    with TestClient(app) as client:
        response = client.get("/api/v1/similar-words/apple/rank/3")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "base_word": "apple",
        "rank": 3,
        "word": "fruit",
        "similarity": 0.82,
        "vocabulary_size": 1000,
    }
    assert service.nth_args == ("apple", 3)
    app.dependency_overrides.clear()


def test_get_nth_similar_word_invalid_input_returns_400() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(nth_error=EmbeddingInputError("Text entries must not be blank."))
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similar-words/apple/rank/1")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json() == {"detail": "Text entries must not be blank."}
    app.dependency_overrides.clear()


def test_get_nth_similar_word_oov_returns_404() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(
            nth_error=EmbeddingNotFoundError("Word is out-of-vocabulary: 'unknown'")
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similar-words/unknown/rank/1")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json() == {"detail": "Word is out-of-vocabulary: 'unknown'"}
    app.dependency_overrides.clear()


def test_get_nth_similar_word_invalid_rank_returns_422() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(
            nth_error=EmbeddingRankError("Rank must be greater than or equal to 1.")
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similar-words/apple/rank/0")

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json() == {"detail": "Rank must be greater than or equal to 1."}
    app.dependency_overrides.clear()


def test_get_nth_similar_word_provider_error_returns_502() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(nth_error=EmbeddingInferenceError("Provider failed."))
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/similar-words/apple/rank/1")

    assert response.status_code == HTTPStatus.BAD_GATEWAY
    assert response.json() == {"detail": "Provider failed."}
    app.dependency_overrides.clear()


def test_get_word_similarity_by_word_returns_similarity_rank() -> None:
    app.dependency_overrides.clear()
    service = StubService(
        rank_result=SimilarityRankResult(
            base_word="king",
            compared_word="queen",
            rank=12,
            similarity=0.71,
            vocabulary_size=1000,
        )
    )
    _override_service(service)

    with TestClient(app) as client:
        response = client.get("/api/v1/word/king/similarity?by_word=queen")

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    _pop_elapsed_time_ms(payload)
    assert payload == {
        "base_word": "king",
        "compared_word": "queen",
        "rank": 12,
        "similarity": 0.71,
        "vocabulary_size": 1000,
    }
    assert service.rank_words == ("king", "queen")
    app.dependency_overrides.clear()


def test_get_word_similarity_by_rank_returns_nth_similar_word() -> None:
    app.dependency_overrides.clear()
    service = StubService(
        nth_result=NthSimilarWordResult(
            base_word="apple",
            rank=3,
            word="fruit",
            similarity=0.82,
            vocabulary_size=1000,
        )
    )
    _override_service(service)

    with TestClient(app) as client:
        response = client.get("/api/v1/word/apple/similarity?by_rank=3")

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    _pop_elapsed_time_ms(payload)
    assert payload == {
        "base_word": "apple",
        "compared_word": "fruit",
        "rank": 3,
        "similarity": 0.82,
        "vocabulary_size": 1000,
    }
    assert service.nth_args == ("apple", 3)
    app.dependency_overrides.clear()


def test_get_word_similarity_requires_exactly_one_query_parameter() -> None:
    app.dependency_overrides.clear()
    _override_service(StubService())

    with TestClient(app) as client:
        missing_response = client.get("/api/v1/word/apple/similarity")
        duplicated_response = client.get(
            "/api/v1/word/apple/similarity?by_word=fruit&by_rank=1"
        )

    assert missing_response.status_code == HTTPStatus.BAD_REQUEST
    assert missing_response.json() == {
        "detail": "Exactly one of by_word or by_rank must be provided."
    }
    assert duplicated_response.status_code == HTTPStatus.BAD_REQUEST
    assert duplicated_response.json() == {
        "detail": "Exactly one of by_word or by_rank must be provided."
    }
    app.dependency_overrides.clear()


def test_get_word_similarity_by_rank_invalid_rank_returns_422() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(
            nth_error=EmbeddingRankError("Rank must be greater than or equal to 1.")
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/word/apple/similarity?by_rank=0")

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json() == {"detail": "Rank must be greater than or equal to 1."}
    app.dependency_overrides.clear()


def test_get_word_similarity_by_word_oov_returns_404() -> None:
    app.dependency_overrides.clear()
    _override_service(
        StubService(
            rank_error=EmbeddingNotFoundError("Word is out-of-vocabulary: 'unknown'")
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/word/apple/similarity?by_word=unknown")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json() == {"detail": "Word is out-of-vocabulary: 'unknown'"}
    app.dependency_overrides.clear()


def test_deprecated_endpoints_are_marked_in_openapi() -> None:
    schema = app.openapi()

    assert schema["paths"]["/api/v1/embedding/{word}"]["get"]["deprecated"] is True
    assert (
        schema["paths"]["/api/v1/similarity/{word1}/{word2}"]["get"]["deprecated"]
        is True
    )
    assert (
        schema["paths"]["/api/v1/similarity-rank/{base_word}/{compared_word}"]["get"][
            "deprecated"
        ]
        is True
    )
    assert (
        schema["paths"]["/api/v1/similar-words/{base_word}/rank/{rank}"]["get"][
            "deprecated"
        ]
        is True
    )


def test_model_loading_failure_in_dependency_returns_500(
    monkeypatch,
) -> None:
    app.dependency_overrides.clear()

    def broken_provider():
        raise EmbeddingModelLoadError("FastText model file not found")

    monkeypatch.setattr(route, "get_embedding_provider", broken_provider)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/embedding/hello")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    app.dependency_overrides.clear()
