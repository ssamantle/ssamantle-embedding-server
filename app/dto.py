from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EmbeddingRequestDTO(BaseModel):
    """Embedding generation request payload."""

    texts: List[str] = Field(..., min_length=1)


class EmbeddingResponseDTO(BaseModel):
    """Embedding generation response payload.

    Note:
        Embeddings are converted to list only at API response serialization time.
    """

    embeddings: List[List[float]]
