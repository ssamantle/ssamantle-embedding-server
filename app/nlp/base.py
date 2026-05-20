from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class EmbeddingProvider(ABC):
    """Abstract contract for embedding engine providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> "np.ndarray":
        """Return sentence embeddings as an array with shape (N, D)."""


__all__ = ["EmbeddingProvider"]
