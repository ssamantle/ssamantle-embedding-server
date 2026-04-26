from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


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


class EmbeddingProvider(ABC):
    """Abstract contract for embedding engine providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> "np.ndarray":
        """Return sentence embeddings as an array with shape (N, D)."""
