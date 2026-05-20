from __future__ import annotations

"""Compatibility re-exports for FastText provider imports."""

from app.nlp.fasttext import (
    FastTextModelLoadError,
    FastTextOOVError,
    FastTextProvider,
    FastTextProviderError,
    FastTextRankError,
)

__all__ = [
    "FastTextProvider",
    "FastTextProviderError",
    "FastTextModelLoadError",
    "FastTextOOVError",
    "FastTextRankError",
]
