from __future__ import annotations

import numpy as np

from app.nlp.base import EmbeddingProvider


class Word2VecProviderNotImplementedError(NotImplementedError):
    """Raised when Word2Vec provider is selected but not implemented yet."""


class Word2VecProvider(EmbeddingProvider):
    """Word2Vec provider skeleton.

    This class intentionally raises a clear exception until the actual
    Word2Vec model loading/inference implementation is introduced.
    """

    def __init__(self) -> None:
        self._reason = (
            "Word2Vec provider is not implemented yet. "
            "Set EMBEDDING_PROVIDER=fasttext to use the supported engine."
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        raise Word2VecProviderNotImplementedError(self._reason)


__all__ = [
    "Word2VecProvider",
    "Word2VecProviderNotImplementedError",
]
