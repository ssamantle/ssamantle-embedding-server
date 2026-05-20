from __future__ import annotations

from app.nlp.base import EmbeddingProvider
from app.nlp.exceptions import (
    EmbeddingException,
    EmbeddingModelLoadError,
    EmbeddingOOVError,
    EmbeddingProviderError,
    EmbeddingRankError,
)
from app.nlp.fasttext import FastTextProvider
from app.nlp.word2vec import Word2VecProvider
from app.providers.base import EmbeddingProvider as ProviderEmbeddingProvider
from app.providers.exceptions import (
    EmbeddingException as ProviderEmbeddingException,
)
from app.providers.exceptions import (
    EmbeddingModelLoadError as ProviderEmbeddingModelLoadError,
)
from app.providers.exceptions import EmbeddingOOVError as ProviderEmbeddingOOVError
from app.providers.exceptions import (
    EmbeddingProviderError as ProviderEmbeddingProviderError,
)
from app.providers.exceptions import EmbeddingRankError as ProviderEmbeddingRankError
from app.providers.fasttext import FastTextProvider as ProviderFastTextProvider
from app.providers.word2vec import Word2VecProvider as ProviderWord2VecProvider


def test_nlp_package_reexports_existing_provider_symbols() -> None:
    assert EmbeddingProvider is ProviderEmbeddingProvider
    assert EmbeddingException is ProviderEmbeddingException
    assert EmbeddingProviderError is ProviderEmbeddingProviderError
    assert EmbeddingModelLoadError is ProviderEmbeddingModelLoadError
    assert EmbeddingOOVError is ProviderEmbeddingOOVError
    assert EmbeddingRankError is ProviderEmbeddingRankError
    assert FastTextProvider is ProviderFastTextProvider
    assert Word2VecProvider is ProviderWord2VecProvider
