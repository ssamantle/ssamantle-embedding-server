from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np

from app.core.settings import settings
from app.dto import EmbeddingResponseDTO
from app.nlp.base import EmbeddingProvider
from app.nlp.exceptions import EmbeddingOOVError as ProviderOOVError
from app.nlp.exceptions import EmbeddingProviderError, KiwiError
from app.nlp.exceptions import EmbeddingRankError as ProviderRankError
from app.nlp.fasttext import FastTextProvider
from app.nlp.kiwi import KiwiAnalyzer
from app.nlp.word2vec import (
    Word2VecProvider,
    Word2VecProviderNotImplementedError,
)
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


@dataclass(frozen=True)
class NlpResources:
    embedding_model: EmbeddingProvider
    kiwi_analyzer: KiwiAnalyzer | None = None


def _resolve_provider_name() -> str:
    configured = getattr(settings, "embedding_provider", None)
    provider_name = configured or os.getenv("EMBEDDING_PROVIDER", "fasttext")
    return str(provider_name).strip().lower()


def _resolve_kiwi_enabled() -> bool:
    configured = os.getenv("KIWI_ENABLED")
    if configured is None:
        return bool(getattr(settings, "kiwi_enabled", False))

    normalized = configured.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_nlp_resources() -> NlpResources:
    provider_name = _resolve_provider_name()
    kiwi_enabled = _resolve_kiwi_enabled()
    logger.info(
        "Resolving NLP resources embedding_model=%s kiwi_enabled=%s",
        provider_name,
        kiwi_enabled,
    )

    if provider_name == "fasttext":
        embedding_model = FastTextProvider()
        kiwi_analyzer = KiwiAnalyzer() if kiwi_enabled else None
        return NlpResources(
            embedding_model=embedding_model,
            kiwi_analyzer=kiwi_analyzer,
        )
    if provider_name == "word2vec":
        return NlpResources(embedding_model=Word2VecProvider())

    raise UnknownEmbeddingProviderError(
        "Unsupported embedding provider: "
        f"'{provider_name}'. Supported values are: fasttext, word2vec."
    )


class EmbeddingService:
    """Service layer for embedding request normalization and inference."""

    def __init__(self, *, nlp_resources: NlpResources) -> None:
        self._nlp_resources = nlp_resources

    @property
    def _embedding_model(self) -> EmbeddingProvider:
        return self._nlp_resources.embedding_model

    @property
    def _kiwi_analyzer(self) -> KiwiAnalyzer | None:
        return self._nlp_resources.kiwi_analyzer

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Normalize input and return embeddings as ndarray for internal processing."""
        normalized_texts = self._normalize_or_raise(texts)
        started_at = perf_counter()
        logger.info("Generating embeddings input_count=%s", len(normalized_texts))

        try:
            embeddings = self._generate_embeddings_with_fallback(normalized_texts)
        except EmbeddingInferenceError:
            raise
        except ProviderOOVError as exc:
            logger.warning(
                "Embedding lookup failed because a token is out of vocabulary"
            )
            raise EmbeddingNotFoundError(str(exc)) from exc
        except (
            EmbeddingProviderError,
            KiwiError,
            Word2VecProviderNotImplementedError,
        ) as exc:
            logger.exception("Embedding provider failed during inference")
            raise EmbeddingInferenceError(str(exc)) from exc
        except Exception as exc:  # defensive mapping for unforeseen provider errors
            logger.exception("Unexpected embedding provider error")
            raise EmbeddingInferenceError("Failed to generate embeddings.") from exc

        self._validate_embedding_matrix(
            embeddings,
            expected_rows=len(normalized_texts),
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

        ranker = getattr(self._embedding_model, "similarity_rank", None)
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

        finder = getattr(self._embedding_model, "nth_similar_word", None)
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

    def _generate_embeddings_with_fallback(self, texts: list[str]) -> np.ndarray:
        if self._kiwi_analyzer is None:
            return self._embedding_model.embed(texts)

        vectors = [self._embed_single_text(text) for text in texts]
        return np.vstack(vectors).astype(np.float32, copy=False)

    def _embed_single_text(self, text: str) -> np.ndarray:
        try:
            return self._embed_single_text_direct(text)
        except ProviderOOVError as exc:
            if self._kiwi_analyzer is None:
                raise

            return self._embed_single_text_with_kiwi_fallback(text, exc)

    def _embed_single_text_direct(self, text: str) -> np.ndarray:
        embeddings = self._embedding_model.embed([text])
        self._validate_embedding_matrix(embeddings, expected_rows=1)
        return np.asarray(embeddings[0], dtype=np.float32)

    def _embed_single_text_with_kiwi_fallback(
        self,
        text: str,
        original_error: ProviderOOVError,
    ) -> np.ndarray:
        if self._kiwi_analyzer is None:
            raise original_error

        morphemes = [
            token
            for token in self._kiwi_analyzer.tokenize_forms(text)
            if isinstance(token, str) and token.strip()
        ]
        if not morphemes:
            raise original_error

        logger.info(
            "Falling back to Kiwi tokenization for embedding text=%s morpheme_count=%s",
            text,
            len(morphemes),
        )

        vectors: list[np.ndarray] = []
        skipped_tokens: list[str] = []
        for morpheme in morphemes:
            try:
                vectors.append(self._embed_single_text_direct(morpheme))
            except ProviderOOVError:
                skipped_tokens.append(morpheme)

        if not vectors:
            logger.warning(
                "Kiwi fallback produced no in-vocabulary morphemes text=%s morphemes=%s",
                text,
                morphemes,
            )
            raise original_error

        if skipped_tokens:
            logger.info(
                "Kiwi fallback skipped out-of-vocabulary morphemes text=%s skipped=%s",
                text,
                skipped_tokens,
            )

        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        return np.asarray(matrix.mean(axis=0, dtype=np.float32), dtype=np.float32)

    @staticmethod
    def _normalize_or_raise(texts: list[str]) -> list[str]:
        try:
            return normalize_texts(texts)
        except InputNormalizationError as exc:
            raise EmbeddingInputError(str(exc)) from exc

    @staticmethod
    def _validate_embedding_matrix(
        embeddings: np.ndarray,
        *,
        expected_rows: int,
    ) -> None:
        if not isinstance(embeddings, np.ndarray):
            raise EmbeddingInferenceError("Provider must return numpy.ndarray.")
        if embeddings.ndim != 2:
            raise EmbeddingInferenceError(
                "Provider must return a 2D embedding matrix with shape (N, D)."
            )
        if embeddings.shape[0] != expected_rows:
            raise EmbeddingInferenceError(
                "Embedding row count does not match input text count."
            )
