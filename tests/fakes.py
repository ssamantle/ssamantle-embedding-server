from __future__ import annotations

import numpy as np

from app.nlp.base import EmbeddingProvider
from app.nlp.exceptions import EmbeddingOOVError
from app.services import NlpResources, NthSimilarWordResult, SimilarityRankResult


class EchoProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.last_inputs: list[str] | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        self.last_inputs = texts
        return np.array([[1.0, 2.0]], dtype=np.float64)


class OOVProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> np.ndarray:
        raise EmbeddingOOVError(f"Word is out-of-vocabulary: '{texts[0]}'")


class RankProvider(EmbeddingProvider):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.rank_inputs: tuple[str, str] | None = None
        self.nth_inputs: tuple[str, int] | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 2.0]], dtype=np.float32)

    def similarity_rank(
        self,
        base_word: str,
        compared_word: str,
    ) -> tuple[int, float, int]:
        self.rank_inputs = (base_word, compared_word)
        if self.error is not None:
            raise self.error
        return 3, 0.5, 10

    def nth_similar_word(
        self,
        base_word: str,
        rank: int,
    ) -> tuple[str, float, int]:
        self.nth_inputs = (base_word, rank)
        if self.error is not None:
            raise self.error
        return "similar", 0.75, 10


class KiwiAwareProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []
        self.vectors = {
            "known": np.array([1.0, 2.0], dtype=np.float32),
            "stem": np.array([1.0, 0.0], dtype=np.float32),
            "tail": np.array([0.0, 1.0], dtype=np.float32),
            "head": np.array([2.0, 0.0], dtype=np.float32),
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embed_calls.append(texts)
        token = texts[0]
        if token not in self.vectors:
            raise EmbeddingOOVError(f"Word is out-of-vocabulary: '{token}'")
        return np.array([self.vectors[token]], dtype=np.float32)


class StubKiwiAnalyzer:
    def __init__(self, token_map: dict[str, list[str]]) -> None:
        self.token_map = token_map
        self.calls: list[str] = []

    def tokenize_forms(self, text: str) -> list[str]:
        self.calls.append(text)
        return self.token_map[text]


class StubService:
    def __init__(
        self,
        *,
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


def build_nlp_resources(
    embedding_model: EmbeddingProvider,
    *,
    kiwi_analyzer: StubKiwiAnalyzer | None = None,
) -> NlpResources:
    return NlpResources(
        embedding_model=embedding_model,
        kiwi_analyzer=kiwi_analyzer,
    )
