from __future__ import annotations


class EmbeddingException(Exception):
    """Base exception for embedding provider failures."""


class EmbeddingProviderError(EmbeddingException):
    """Raised when an embedding provider fails."""


class EmbeddingModelLoadError(EmbeddingProviderError):
    """Raised when an embedding model cannot be loaded."""


class EmbeddingOOVError(EmbeddingProviderError):
    """Raised when a token is out-of-vocabulary."""


class EmbeddingRankError(EmbeddingProviderError, ValueError):
    """Raised when a requested rank is outside the vocabulary range."""


__all__ = [
    "EmbeddingException",
    "EmbeddingProviderError",
    "EmbeddingModelLoadError",
    "EmbeddingOOVError",
    "EmbeddingRankError",
]
