from __future__ import annotations


class InputNormalizationError(ValueError):
    """Raised when input texts cannot be normalized for embedding."""


def normalize_texts(texts: list[str]) -> list[str]:
    """Return trimmed non-empty texts for provider inference."""
    if not isinstance(texts, list):
        raise InputNormalizationError("'texts' must be a list of strings.")

    if not texts:
        raise InputNormalizationError("'texts' must contain at least one item.")

    normalized: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            raise InputNormalizationError("All items in 'texts' must be strings.")

        token = text.strip()
        if not token:
            raise InputNormalizationError("Text entries must not be blank.")

        normalized.append(token)

    return normalized
