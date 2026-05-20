from __future__ import annotations

"""Compatibility re-exports for Word2Vec provider imports."""

from app.nlp.word2vec import (
    Word2VecProvider,
    Word2VecProviderNotImplementedError,
)

__all__ = [
    "Word2VecProvider",
    "Word2VecProviderNotImplementedError",
]
