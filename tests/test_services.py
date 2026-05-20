from __future__ import annotations

import numpy as np
import pytest

import app.services as services
from app.nlp.exceptions import (
    EmbeddingModelLoadError,
    EmbeddingOOVError,
    KiwiInitializationError,
)
from app.nlp.exceptions import EmbeddingRankError as ProviderRankError
from app.services import (
    EmbeddingInferenceError,
    EmbeddingInputError,
    EmbeddingNotFoundError,
    EmbeddingRankError,
    EmbeddingService,
)
from tests.fakes import (
    EchoProvider,
    KiwiAwareProvider,
    OOVProvider,
    RankProvider,
    StubKiwiAnalyzer,
    build_nlp_resources,
)


def test_generate_embeddings_normalizes_input_and_returns_float32() -> None:
    provider = EchoProvider()
    service = EmbeddingService(nlp_resources=build_nlp_resources(provider))

    embeddings = service.generate_embeddings(["  token  "])

    assert provider.last_inputs == ["token"]
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (1, 2)


def test_generate_embeddings_with_blank_text_raises_input_error() -> None:
    service = EmbeddingService(nlp_resources=build_nlp_resources(EchoProvider()))

    with pytest.raises(EmbeddingInputError, match="must not be blank"):
        service.generate_embeddings(["   "])


def test_generate_embeddings_oov_raises_not_found_error() -> None:
    service = EmbeddingService(nlp_resources=build_nlp_resources(OOVProvider()))

    with pytest.raises(EmbeddingNotFoundError, match="out-of-vocabulary"):
        service.generate_embeddings(["unknown"])


def test_generate_embeddings_with_kiwi_enabled_preserves_direct_hit() -> None:
    provider = KiwiAwareProvider()
    kiwi_analyzer = StubKiwiAnalyzer({"known": ["ignored"]})
    service = EmbeddingService(
        nlp_resources=build_nlp_resources(provider, kiwi_analyzer=kiwi_analyzer)
    )

    embeddings = service.generate_embeddings(["known"])

    assert kiwi_analyzer.calls == []
    assert provider.embed_calls == [["known"]]
    assert np.allclose(embeddings, np.array([[1.0, 2.0]], dtype=np.float32))


def test_generate_embeddings_with_kiwi_fallback_averages_morpheme_vectors() -> None:
    provider = KiwiAwareProvider()
    kiwi_analyzer = StubKiwiAnalyzer({"compoundword": ["stem", "tail"]})
    service = EmbeddingService(
        nlp_resources=build_nlp_resources(provider, kiwi_analyzer=kiwi_analyzer)
    )

    embeddings = service.generate_embeddings(["compoundword"])

    assert kiwi_analyzer.calls == ["compoundword"]
    assert provider.embed_calls == [["compoundword"], ["stem"], ["tail"]]
    assert np.allclose(embeddings, np.array([[0.5, 0.5]], dtype=np.float32))


def test_generate_embeddings_with_kiwi_fallback_skips_oov_morphemes() -> None:
    provider = KiwiAwareProvider()
    kiwi_analyzer = StubKiwiAnalyzer({"partialword": ["head", "missing", "tail"]})
    service = EmbeddingService(
        nlp_resources=build_nlp_resources(provider, kiwi_analyzer=kiwi_analyzer)
    )

    embeddings = service.generate_embeddings(["partialword"])

    assert kiwi_analyzer.calls == ["partialword"]
    assert provider.embed_calls == [["partialword"], ["head"], ["missing"], ["tail"]]
    assert np.allclose(embeddings, np.array([[1.0, 0.5]], dtype=np.float32))


def test_generate_embeddings_with_kiwi_fallback_raises_when_all_morphemes_are_oov() -> (
    None
):
    provider = KiwiAwareProvider()
    kiwi_analyzer = StubKiwiAnalyzer({"unknownword": ["missing", "ghost"]})
    service = EmbeddingService(
        nlp_resources=build_nlp_resources(provider, kiwi_analyzer=kiwi_analyzer)
    )

    with pytest.raises(EmbeddingNotFoundError, match="out-of-vocabulary"):
        service.generate_embeddings(["unknownword"])


def test_calculate_similarity_rank_normalizes_input_and_returns_result() -> None:
    provider = RankProvider()
    service = EmbeddingService(nlp_resources=build_nlp_resources(provider))

    result = service.calculate_similarity_rank("  base  ", "  compared  ")

    assert provider.rank_inputs == ("base", "compared")
    assert result.base_word == "base"
    assert result.compared_word == "compared"
    assert result.rank == 3
    assert result.similarity == 0.5
    assert result.vocabulary_size == 10


def test_calculate_similarity_rank_oov_raises_not_found_error() -> None:
    provider = RankProvider(
        error=EmbeddingOOVError("Word is out-of-vocabulary: 'unknown'")
    )
    service = EmbeddingService(nlp_resources=build_nlp_resources(provider))

    with pytest.raises(EmbeddingNotFoundError, match="out-of-vocabulary"):
        service.calculate_similarity_rank("base", "unknown")


def test_calculate_similarity_rank_unsupported_provider_raises_inference_error() -> (
    None
):
    service = EmbeddingService(nlp_resources=build_nlp_resources(EchoProvider()))

    with pytest.raises(EmbeddingInferenceError, match="does not support"):
        service.calculate_similarity_rank("base", "compared")


def test_find_nth_similar_word_normalizes_input_and_returns_result() -> None:
    provider = RankProvider()
    service = EmbeddingService(nlp_resources=build_nlp_resources(provider))

    result = service.find_nth_similar_word("  base  ", 2)

    assert provider.nth_inputs == ("base", 2)
    assert result.base_word == "base"
    assert result.rank == 2
    assert result.word == "similar"
    assert result.similarity == 0.75
    assert result.vocabulary_size == 10


def test_find_nth_similar_word_invalid_rank_raises_rank_error() -> None:
    service = EmbeddingService(nlp_resources=build_nlp_resources(RankProvider()))

    with pytest.raises(EmbeddingRankError, match="greater than or equal to 1"):
        service.find_nth_similar_word("base", 0)


def test_find_nth_similar_word_oov_raises_not_found_error() -> None:
    provider = RankProvider(
        error=EmbeddingOOVError("Word is out-of-vocabulary: 'base'")
    )
    service = EmbeddingService(nlp_resources=build_nlp_resources(provider))

    with pytest.raises(EmbeddingNotFoundError, match="out-of-vocabulary"):
        service.find_nth_similar_word("base", 1)


def test_find_nth_similar_word_out_of_range_rank_raises_rank_error() -> None:
    provider = RankProvider(error=ProviderRankError("Rank exceeds vocabulary size."))
    service = EmbeddingService(nlp_resources=build_nlp_resources(provider))

    with pytest.raises(EmbeddingRankError, match="Rank exceeds"):
        service.find_nth_similar_word("base", 11)


def test_find_nth_similar_word_unsupported_provider_raises_inference_error() -> None:
    service = EmbeddingService(nlp_resources=build_nlp_resources(EchoProvider()))

    with pytest.raises(EmbeddingInferenceError, match="does not support"):
        service.find_nth_similar_word("base", 1)


def test_get_nlp_resources_propagates_model_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenFastTextProvider:
        def __init__(self) -> None:
            raise EmbeddingModelLoadError("FastText model file not found")

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fasttext")
    monkeypatch.setattr(services, "FastTextProvider", BrokenFastTextProvider)

    with pytest.raises(EmbeddingModelLoadError, match="model file not found"):
        services.get_nlp_resources()


def test_get_nlp_resources_adds_kiwi_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_model = EchoProvider()
    kiwi_analyzer = object()

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fasttext")
    monkeypatch.setenv("KIWI_ENABLED", "true")
    monkeypatch.setattr(services, "FastTextProvider", lambda: embedding_model)
    monkeypatch.setattr(services, "KiwiAnalyzer", lambda: kiwi_analyzer)

    resources = services.get_nlp_resources()

    assert resources.embedding_model is embedding_model
    assert resources.kiwi_analyzer is kiwi_analyzer


def test_get_nlp_resources_propagates_kiwi_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fasttext")
    monkeypatch.setenv("KIWI_ENABLED", "true")
    monkeypatch.setattr(services, "FastTextProvider", EchoProvider)

    def broken_kiwi_analyzer() -> None:
        raise KiwiInitializationError("kiwipiepy is required")

    monkeypatch.setattr(services, "KiwiAnalyzer", broken_kiwi_analyzer)

    with pytest.raises(KiwiInitializationError, match="kiwipiepy is required"):
        services.get_nlp_resources()
