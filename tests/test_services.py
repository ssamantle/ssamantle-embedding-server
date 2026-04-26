from __future__ import annotations

import numpy as np
import pytest

import app.services as services
from app.providers.fasttext import FastTextModelLoadError, FastTextOOVError
from app.services import (
    EmbeddingInferenceError,
    EmbeddingInputError,
    EmbeddingNotFoundError,
    EmbeddingService,
)


class EchoProvider:
    def __init__(self) -> None:
        self.last_inputs: list[str] | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        self.last_inputs = texts
        return np.array([[1.0, 2.0]], dtype=np.float64)


class OOVProvider:
    def embed(self, texts: list[str]) -> np.ndarray:
        raise FastTextOOVError(f"Word is out-of-vocabulary: '{texts[0]}'")


class RankProvider:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.rank_inputs: tuple[str, str] | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 2.0]], dtype=np.float32)

    def similarity_rank(
        self, base_word: str, compared_word: str
    ) -> tuple[int, float, int]:
        self.rank_inputs = (base_word, compared_word)
        if self.error is not None:
            raise self.error
        return 3, 0.5, 10


def test_generate_embeddings_normalizes_input_and_returns_float32() -> None:
    provider = EchoProvider()
    service = EmbeddingService(provider=provider)

    embeddings = service.generate_embeddings(["  token  "])

    assert provider.last_inputs == ["token"]
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (1, 2)


def test_generate_embeddings_with_blank_text_raises_input_error() -> None:
    service = EmbeddingService(provider=EchoProvider())

    with pytest.raises(EmbeddingInputError, match="must not be blank"):
        service.generate_embeddings(["   "])


def test_generate_embeddings_oov_raises_not_found_error() -> None:
    service = EmbeddingService(provider=OOVProvider())

    with pytest.raises(EmbeddingNotFoundError, match="out-of-vocabulary"):
        service.generate_embeddings(["unknown"])


def test_calculate_similarity_rank_normalizes_input_and_returns_result() -> None:
    provider = RankProvider()
    service = EmbeddingService(provider=provider)

    result = service.calculate_similarity_rank("  base  ", "  compared  ")

    assert provider.rank_inputs == ("base", "compared")
    assert result.base_word == "base"
    assert result.compared_word == "compared"
    assert result.rank == 3
    assert result.similarity == 0.5
    assert result.vocabulary_size == 10


def test_calculate_similarity_rank_oov_raises_not_found_error() -> None:
    provider = RankProvider(
        error=FastTextOOVError("Word is out-of-vocabulary: 'unknown'")
    )
    service = EmbeddingService(provider=provider)

    with pytest.raises(EmbeddingNotFoundError, match="out-of-vocabulary"):
        service.calculate_similarity_rank("base", "unknown")


def test_calculate_similarity_rank_unsupported_provider_raises_inference_error() -> (
    None
):
    service = EmbeddingService(provider=EchoProvider())

    with pytest.raises(EmbeddingInferenceError, match="does not support"):
        service.calculate_similarity_rank("base", "compared")


def test_get_embedding_provider_propagates_model_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenFastTextProvider:
        def __init__(self) -> None:
            raise FastTextModelLoadError("FastText model file not found")

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fasttext")
    monkeypatch.setattr(services, "FastTextProvider", BrokenFastTextProvider)
    services.get_embedding_provider.cache_clear()

    with pytest.raises(FastTextModelLoadError, match="model file not found"):
        services.get_embedding_provider()

    services.get_embedding_provider.cache_clear()
