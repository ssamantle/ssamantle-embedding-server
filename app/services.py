from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np

from app.core.settings import settings
from app.dto import EmbeddingResponseDTO
from app.providers.base import EmbeddingOOVError as ProviderOOVError
from app.providers.base import EmbeddingProvider, EmbeddingProviderError
from app.providers.base import EmbeddingRankError as ProviderRankError
from app.providers.fasttext import FastTextProvider
from app.providers.word2vec import Word2VecProvider, Word2VecProviderNotImplementedError
from app.utils import InputNormalizationError, normalize_texts

logger = logging.getLogger(__name__)


class UnknownEmbeddingProviderError(ValueError):
    """Raised when an unsupported embedding provider is configured."""


class EmbeddingServiceError(Exception):
    """Base exception for service-layer failures."""


class EmbeddingInputError(EmbeddingServiceError, ValueError):
    """Raised when embedding input payload is invalid."""


class EmbeddingNotFoundError(EmbeddingServiceError):
    """Raised when a requested embedding cannot be found."""


class EmbeddingInferenceError(EmbeddingServiceError):
    """Raised when provider inference fails."""


class EmbeddingRankError(EmbeddingServiceError, ValueError):
    """Raised when a requested rank is invalid for the embedding vocabulary."""


@dataclass(frozen=True)
class SimilarityRankResult:
    base_word: str
    compared_word: str
    rank: int
    similarity: float
    vocabulary_size: int


@dataclass(frozen=True)
class NthSimilarWordResult:
    base_word: str
    rank: int
    word: str
    similarity: float
    vocabulary_size: int


def _resolve_provider_name() -> str:
    configured = getattr(settings, "embedding_provider", None)
    provider_name = configured or os.getenv("EMBEDDING_PROVIDER", "fasttext")
    return str(provider_name).strip().lower()


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    provider_name = _resolve_provider_name()
    logger.info("Resolving embedding provider provider=%s", provider_name)

    if provider_name == "fasttext":
        return FastTextProvider()
    if provider_name == "word2vec":
        return Word2VecProvider()

    raise UnknownEmbeddingProviderError(
        "Unsupported embedding provider: "
        f"'{provider_name}'. Supported values are: fasttext, word2vec."
    )


class EmbeddingService:
    """Service layer for embedding request normalization and inference."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Normalize input and return embeddings as ndarray for internal processing."""
        normalized_texts = self._normalize_or_raise(texts)
        started_at = perf_counter()
        logger.info("Generating embeddings input_count=%s", len(normalized_texts))

        try:
            embeddings = self._provider.embed(normalized_texts)
        except ProviderOOVError as exc:
            logger.warning(
                "Embedding lookup failed because a token is out of vocabulary"
            )
            raise EmbeddingNotFoundError(str(exc)) from exc
        except (EmbeddingProviderError, Word2VecProviderNotImplementedError) as exc:
            logger.exception("Embedding provider failed during inference")
            raise EmbeddingInferenceError(str(exc)) from exc
        except Exception as exc:  # defensive mapping for unforeseen provider errors
            logger.exception("Unexpected embedding provider error")
            raise EmbeddingInferenceError("Failed to generate embeddings.") from exc

        if not isinstance(embeddings, np.ndarray):
            raise EmbeddingInferenceError("Provider must return numpy.ndarray.")
        if embeddings.ndim != 2:
            raise EmbeddingInferenceError(
                "Provider must return a 2D embedding matrix with shape (N, D)."
            )
        if embeddings.shape[0] != len(normalized_texts):
            raise EmbeddingInferenceError(
                "Embedding row count does not match input text count."
            )

        result = embeddings.astype(np.float32, copy=False)
        elapsed = perf_counter() - started_at
        logger.info(
            "Generated embeddings in %.4fs input_count=%s shape=%s",
            elapsed,
            len(normalized_texts),
            result.shape,
        )
        return result

    def calculate_similarity_rank(
        self,
        base_word: str,
        compared_word: str,
    ) -> SimilarityRankResult:
        normalized_words = self._normalize_or_raise([base_word, compared_word])
        normalized_base_word = normalized_words[0]
        normalized_compared_word = normalized_words[1]

        ranker = getattr(self._provider, "similarity_rank", None)
        if ranker is None:
            raise EmbeddingInferenceError(
                "Provider does not support similarity rank calculation."
            )

        started_at = perf_counter()
        logger.info("Calculating similarity rank")
        try:
            rank, similarity, vocabulary_size = ranker(
                normalized_base_word,
                normalized_compared_word,
            )
        except ProviderOOVError as exc:
            logger.warning(
                "Similarity rank failed because a token is out of vocabulary"
            )
            raise EmbeddingNotFoundError(str(exc)) from exc
        except (EmbeddingProviderError, Word2VecProviderNotImplementedError) as exc:
            logger.exception(
                "Embedding provider failed during similarity rank calculation"
            )
            raise EmbeddingInferenceError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected similarity rank provider error")
            raise EmbeddingInferenceError(
                "Failed to calculate similarity rank."
            ) from exc

        elapsed = perf_counter() - started_at
        logger.info(
            "Calculated similarity rank in %.4fs rank=%s vocabulary_size=%s",
            elapsed,
            rank,
            vocabulary_size,
        )
        return SimilarityRankResult(
            base_word=normalized_base_word,
            compared_word=normalized_compared_word,
            rank=int(rank),
            similarity=float(similarity),
            vocabulary_size=int(vocabulary_size),
        )

    def find_nth_similar_word(
        self,
        base_word: str,
        rank: int,
    ) -> NthSimilarWordResult:
        if rank < 1:
            raise EmbeddingRankError("Rank must be greater than or equal to 1.")

        normalized_base_word = self._normalize_or_raise([base_word])[0]

        finder = getattr(self._provider, "nth_similar_word", None)
        if finder is None:
            raise EmbeddingInferenceError(
                "Provider does not support nth similar word lookup."
            )

        started_at = perf_counter()
        logger.info("Finding nth similar word rank=%s", rank)
        try:
            word, similarity, vocabulary_size = finder(normalized_base_word, rank)
        except ProviderOOVError as exc:
            logger.warning(
                "Nth similar word lookup failed because base word is out of vocabulary"
            )
            raise EmbeddingNotFoundError(str(exc)) from exc
        except ProviderRankError as exc:
            logger.warning("Nth similar word lookup failed because rank is invalid")
            raise EmbeddingRankError(str(exc)) from exc
        except (EmbeddingProviderError, Word2VecProviderNotImplementedError) as exc:
            logger.exception("Embedding provider failed during nth similar word lookup")
            raise EmbeddingInferenceError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected nth similar word provider error")
            raise EmbeddingInferenceError("Failed to find nth similar word.") from exc

        elapsed = perf_counter() - started_at
        logger.info(
            "Found nth similar word in %.4fs rank=%s vocabulary_size=%s",
            elapsed,
            rank,
            vocabulary_size,
        )
        return NthSimilarWordResult(
            base_word=normalized_base_word,
            rank=rank,
            word=str(word),
            similarity=float(similarity),
            vocabulary_size=int(vocabulary_size),
        )

    def assemble_response(self, embeddings: np.ndarray) -> EmbeddingResponseDTO:
        """Build API response DTO and convert to list only at serialization boundary."""
        if not isinstance(embeddings, np.ndarray):
            raise EmbeddingInferenceError("Embeddings must be numpy.ndarray.")
        if embeddings.ndim != 2:
            raise EmbeddingInferenceError(
                "Embeddings must be a 2D matrix with shape (N, D)."
            )
        return EmbeddingResponseDTO(embeddings=embeddings.tolist())

    def embed(self, texts: list[str]) -> EmbeddingResponseDTO:
        """End-to-end service flow: normalize, infer as ndarray, then assemble response."""
        embeddings = self.generate_embeddings(texts)
        return self.assemble_response(embeddings)

    @staticmethod
    def _normalize_or_raise(texts: list[str]) -> list[str]:
        try:
            return normalize_texts(texts)
        except InputNormalizationError as exc:
            raise EmbeddingInputError(str(exc)) from exc
