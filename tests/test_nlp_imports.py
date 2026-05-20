from __future__ import annotations

from app.nlp.base import EmbeddingProvider
from app.nlp.exceptions import (
    EmbeddingException,
    EmbeddingModelLoadError,
    EmbeddingOOVError,
    EmbeddingProviderError,
    EmbeddingRankError,
)
from app.nlp.fasttext import (
    FastTextModelLoadError,
    FastTextOOVError,
    FastTextProvider,
    FastTextProviderError,
    FastTextRankError,
)
from app.nlp.word2vec import (
    Word2VecProvider,
    Word2VecProviderNotImplementedError,
)
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
from app.providers.fasttext import (
    FastTextModelLoadError as ProviderFastTextModelLoadError,
)
from app.providers.fasttext import FastTextOOVError as ProviderFastTextOOVError
from app.providers.fasttext import FastTextProvider as ProviderFastTextProvider
from app.providers.fasttext import (
    FastTextProviderError as ProviderFastTextProviderError,
)
from app.providers.fasttext import FastTextRankError as ProviderFastTextRankError
from app.providers.word2vec import Word2VecProvider as ProviderWord2VecProvider
from app.providers.word2vec import (
    Word2VecProviderNotImplementedError as ProviderWord2VecProviderNotImplementedError,
)


def test_nlp_package_reexports_existing_provider_symbols() -> None:
    assert EmbeddingProvider is ProviderEmbeddingProvider
    assert EmbeddingException is ProviderEmbeddingException
    assert EmbeddingProviderError is ProviderEmbeddingProviderError
    assert EmbeddingModelLoadError is ProviderEmbeddingModelLoadError
    assert EmbeddingOOVError is ProviderEmbeddingOOVError
    assert EmbeddingRankError is ProviderEmbeddingRankError
    assert FastTextProvider is ProviderFastTextProvider
    assert FastTextProviderError is ProviderFastTextProviderError
    assert FastTextModelLoadError is ProviderFastTextModelLoadError
    assert FastTextOOVError is ProviderFastTextOOVError
    assert FastTextRankError is ProviderFastTextRankError
    assert Word2VecProvider is ProviderWord2VecProvider
    assert (
        Word2VecProviderNotImplementedError
        is ProviderWord2VecProviderNotImplementedError
    )
