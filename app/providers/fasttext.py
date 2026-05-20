from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors

from app.core.settings import settings
from app.nlp.exceptions import (
    EmbeddingModelLoadError,
    EmbeddingOOVError,
    EmbeddingProviderError,
    EmbeddingRankError,
)
from app.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class FastTextProviderError(EmbeddingProviderError):
    """Base exception for FastText provider failures."""


class FastTextModelLoadError(FastTextProviderError, EmbeddingModelLoadError):
    """Raised when FastText model loading fails."""


class FastTextOOVError(FastTextProviderError, EmbeddingOOVError):
    """Raised when a token is out-of-vocabulary."""


class FastTextRankError(FastTextProviderError, EmbeddingRankError):
    """Raised when a similarity rank is outside the vocabulary range."""


class FastTextProvider(EmbeddingProvider):
    """FastText embedding provider backed by gensim KeyedVectors."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        configured_path = (
            model_path
            if model_path is not None
            else getattr(settings, "fasttext_model_path", "data/models/cc.ko.300.bin")
        )
        self.model_path = Path(configured_path).expanduser().resolve()
        logger.info(
            "Initializing FastText provider with model_path=%s",
            self.model_path,
        )
        self._model = self._load_model(self.model_path)

    def _load_model(self, model_path: Path) -> KeyedVectors:
        if not model_path.exists() or not model_path.is_file():
            logger.error("FastText model file not found: %s", model_path)
            raise FastTextModelLoadError(f"FastText model file not found: {model_path}")

        errors: list[str] = []
        started_at = perf_counter()
        for loader in self._candidate_loaders(model_path):
            loader_started_at = perf_counter()
            logger.info(
                "Loading FastText model with loader=%s path=%s",
                loader.__name__,
                model_path,
            )
            try:
                model = loader(model_path)
                if not isinstance(model, KeyedVectors):
                    raise TypeError(f"Unsupported model type: {type(model)!r}")
                elapsed = perf_counter() - loader_started_at
                total_elapsed = perf_counter() - started_at
                logger.info(
                    "Loaded FastText model with loader=%s in %.2fs total=%.2fs "
                    "vector_size=%s vocab_size=%s",
                    loader.__name__,
                    elapsed,
                    total_elapsed,
                    model.vector_size,
                    len(model),
                )
                return model
            except Exception as exc:
                elapsed = perf_counter() - loader_started_at
                logger.warning(
                    "Failed to load FastText model with loader=%s in %.2fs: %s",
                    loader.__name__,
                    elapsed,
                    exc,
                )
                errors.append(f"{loader.__name__}: {exc}")

        detail = "; ".join(errors)
        logger.error(
            "Failed to load FastText model from %s. Tried loaders: %s",
            model_path,
            detail,
        )
        raise FastTextModelLoadError(
            f"Failed to load FastText model from {model_path}. Tried loaders: {detail}"
        )

    def _candidate_loaders(self, model_path: Path):
        suffixes = {part.lower() for part in model_path.suffixes}

        if ".kv" in suffixes:
            return [self._load_gensim_native]
        if ".bin" in suffixes:
            return [self._load_facebook_bin, self._load_word2vec_text]
        if {".vec", ".txt", ".gz"} & suffixes:
            return [self._load_word2vec_text, self._load_facebook_bin]
        return [
            self._load_gensim_native,
            self._load_facebook_bin,
            self._load_word2vec_text,
        ]

    @staticmethod
    def _load_gensim_native(model_path: Path) -> KeyedVectors:
        return KeyedVectors.load(str(model_path), mmap="r")

    @staticmethod
    def _load_facebook_bin(model_path: Path) -> KeyedVectors:
        return load_facebook_vectors(str(model_path))

    @staticmethod
    def _load_word2vec_text(model_path: Path) -> KeyedVectors:
        return KeyedVectors.load_word2vec_format(
            str(model_path),
            binary=False,
            unicode_errors="ignore",
        )

    def _require_in_vocab(self, word: str) -> None:
        token = word.strip()
        if not token:
            raise FastTextOOVError("Word is empty after trimming.")
        if not self._model.has_index_for(token):
            raise FastTextOOVError(f"Word is out-of-vocabulary: '{token}'")

    def get_word_vector(self, word: str) -> np.ndarray:
        token = word.strip()
        self._require_in_vocab(token)
        vector = self._model.get_vector(token)
        return np.asarray(vector, dtype=np.float32)

    def similarity(self, word1: str, word2: str) -> float:
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denominator == 0.0:
            raise FastTextProviderError(
                "Cannot compute similarity for zero-norm vector."
            )
        score = float(np.dot(vec1, vec2) / denominator)
        return score

    def similarity_rank(
        self,
        base_word: str,
        compared_word: str,
    ) -> tuple[int, float, int]:
        base_token = base_word.strip()
        compared_token = compared_word.strip()
        self._require_in_vocab(base_token)
        self._require_in_vocab(compared_token)

        started_at = perf_counter()
        rank = int(self._model.rank(base_token, compared_token))
        similarity = float(self._model.similarity(base_token, compared_token))
        vocabulary_size = len(self._model)
        elapsed = perf_counter() - started_at
        logger.info(
            "Calculated FastText similarity rank in %.4fs rank=%s vocabulary_size=%s",
            elapsed,
            rank,
            vocabulary_size,
        )
        return rank, similarity, vocabulary_size

    def nth_similar_word(self, base_word: str, rank: int) -> tuple[str, float, int]:
        base_token = base_word.strip()
        self._require_in_vocab(base_token)

        vocabulary_size = len(self._model)
        if rank > vocabulary_size:
            raise FastTextRankError(
                f"Rank must be less than or equal to vocabulary size: {vocabulary_size}"
            )

        started_at = perf_counter()
        word, similarity = self._model.most_similar(base_token, topn=rank)[-1]
        elapsed = perf_counter() - started_at
        logger.info(
            "Found FastText nth similar word in %.4fs rank=%s vocabulary_size=%s",
            elapsed,
            rank,
            vocabulary_size,
        )
        return str(word), float(similarity), vocabulary_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._model.vector_size), dtype=np.float32)
        started_at = perf_counter()
        vectors = [self.get_word_vector(text) for text in texts]
        embeddings = np.vstack(vectors).astype(np.float32, copy=False)
        elapsed = perf_counter() - started_at
        logger.info(
            "Generated FastText embeddings in %.4fs count=%s dimension=%s",
            elapsed,
            len(texts),
            embeddings.shape[1],
        )
        return embeddings
