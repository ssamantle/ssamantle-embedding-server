from __future__ import annotations

from http import HTTPStatus
from time import perf_counter
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from app.providers.base import EmbeddingModelLoadError
from app.services import (
    EmbeddingInferenceError,
    EmbeddingInputError,
    EmbeddingNotFoundError,
    EmbeddingRankError,
    EmbeddingService,
    UnknownEmbeddingProviderError,
    get_embedding_provider,
)

router = APIRouter()


class SimilarityResponse(BaseModel):
    similarity: float


class SimilarityRankResponse(BaseModel):
    base_word: str
    compared_word: str
    rank: int
    similarity: float
    vocabulary_size: int


class WordDetailResponse(BaseModel):
    word: str
    embedding: list[float]
    elapsed_time_ms: float = Field(
        description="요청 처리에 걸린 시간입니다. 단위는 밀리초입니다.",
        ge=0,
    )


class WordSimilarityResponse(SimilarityRankResponse):
    elapsed_time_ms: float = Field(
        description="요청 처리에 걸린 시간입니다. 단위는 밀리초입니다.",
        ge=0,
    )


class NthSimilarWordResponse(BaseModel):
    base_word: str
    rank: int
    word: str
    similarity: float
    vocabulary_size: int


def _elapsed_time_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000


def get_embedding_service() -> EmbeddingService:
    try:
        provider = get_embedding_provider()
    except EmbeddingModelLoadError as exc:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except UnknownEmbeddingProviderError as exc:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    return EmbeddingService(provider=provider)


@router.get(
    "/embedding/{word}",
    deprecated=True,
    tags=["embeddings"],
    summary="단어 임베딩 조회",
    description="단일 단어의 벡터 임베딩을 생성해 반환합니다.",
    response_description="요청한 단어와 임베딩 벡터입니다.",
    responses={
        HTTPStatus.BAD_REQUEST: {"description": "요청한 단어가 유효하지 않습니다."},
        HTTPStatus.NOT_FOUND: {
            "description": "요청한 단어가 임베딩 모델 어휘에 없습니다."
        },
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "지원하지 않는 임베딩 제공자가 설정되었습니다."
        },
        HTTPStatus.BAD_GATEWAY: {
            "description": "임베딩 제공자가 벡터 생성 중 실패했습니다."
        },
        HTTPStatus.SERVICE_UNAVAILABLE: {
            "description": "임베딩 모델 파일을 로드할 수 없어 서비스를 사용할 수 없습니다."
        },
    },
)
def get_embedding(
    word: Annotated[
        str,
        Path(description="임베딩을 조회할 단어입니다."),
    ],
    service: EmbeddingService = Depends(get_embedding_service),
) -> dict[str, object]:
    try:
        embeddings = service.generate_embeddings([word])
    except EmbeddingInputError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)
        ) from exc
    except EmbeddingNotFoundError as exc:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except EmbeddingInferenceError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY, detail=str(exc)
        ) from exc

    return {
        "word": word,
        "embedding": embeddings[0].tolist(),
    }


@router.get(
    "/similarity/{word1}/{word2}",
    response_model=SimilarityResponse,
    deprecated=True,
    tags=["embeddings"],
    summary="단어 유사도 계산",
    description="두 단어 임베딩 간의 코사인 유사도를 계산합니다.",
    response_description="두 단어의 코사인 유사도 점수입니다.",
    responses={
        HTTPStatus.BAD_REQUEST: {
            "description": "요청한 단어 중 하나 이상이 유효하지 않습니다."
        },
        HTTPStatus.NOT_FOUND: {
            "description": "요청한 단어 중 하나 이상이 임베딩 모델 어휘에 없습니다."
        },
        HTTPStatus.UNPROCESSABLE_ENTITY: {
            "description": "영벡터에 대해서는 유사도를 계산할 수 없습니다."
        },
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "지원하지 않는 임베딩 제공자가 설정되었습니다."
        },
        HTTPStatus.BAD_GATEWAY: {
            "description": "임베딩 제공자가 벡터 생성 중 실패했습니다."
        },
        HTTPStatus.SERVICE_UNAVAILABLE: {
            "description": "임베딩 모델 파일을 로드할 수 없어 서비스를 사용할 수 없습니다."
        },
    },
)
def calculate_similarity(
    word1: Annotated[
        str,
        Path(description="비교할 첫 번째 단어입니다."),
    ],
    word2: Annotated[
        str,
        Path(description="비교할 두 번째 단어입니다."),
    ],
    service: EmbeddingService = Depends(get_embedding_service),
) -> SimilarityResponse:
    return _calculate_similarity(word1=word1, word2=word2, service=service)


@router.get(
    "/similarity-rank/{base_word}/{compared_word}",
    response_model=SimilarityRankResponse,
    deprecated=True,
    tags=["embeddings"],
    summary="단어 유사도 순위 계산",
    description=(
        "기준 단어에 대해 비교 단어가 전체 vocabulary 중 몇 번째로 유사한지 계산합니다."
    ),
    response_description="비교 단어의 유사도 순위와 유사도 점수입니다.",
    responses={
        HTTPStatus.BAD_REQUEST: {
            "description": "요청한 단어 중 하나 이상이 유효하지 않습니다."
        },
        HTTPStatus.NOT_FOUND: {
            "description": "요청한 단어 중 하나 이상이 임베딩 모델 어휘에 없습니다."
        },
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "지원하지 않는 임베딩 제공자가 설정되었습니다."
        },
        HTTPStatus.BAD_GATEWAY: {
            "description": "임베딩 제공자가 순위 계산 중 실패했습니다."
        },
        HTTPStatus.SERVICE_UNAVAILABLE: {
            "description": "임베딩 모델 파일을 로드할 수 없어 서비스를 사용할 수 없습니다."
        },
    },
)
def calculate_similarity_rank(
    base_word: Annotated[
        str,
        Path(description="유사도 순위를 계산할 기준 단어입니다."),
    ],
    compared_word: Annotated[
        str,
        Path(description="기준 단어에 대해 순위를 확인할 비교 단어입니다."),
    ],
    service: EmbeddingService = Depends(get_embedding_service),
) -> SimilarityRankResponse:
    try:
        result = service.calculate_similarity_rank(
            base_word=base_word,
            compared_word=compared_word,
        )
    except EmbeddingInputError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)
        ) from exc
    except EmbeddingNotFoundError as exc:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except EmbeddingInferenceError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY, detail=str(exc)
        ) from exc

    return SimilarityRankResponse(
        base_word=result.base_word,
        compared_word=result.compared_word,
        rank=result.rank,
        similarity=result.similarity,
        vocabulary_size=result.vocabulary_size,
    )


@router.get(
    "/similar-words/{base_word}/rank/{rank}",
    response_model=NthSimilarWordResponse,
    deprecated=True,
    tags=["embeddings"],
    summary="N번째 유사 단어 조회",
    description="기준 단어에 대해 전체 vocabulary에서 N번째로 가까운 단어를 조회합니다.",
    response_description="N번째 유사 단어와 유사도 점수입니다.",
    responses={
        HTTPStatus.BAD_REQUEST: {
            "description": "요청한 기준 단어가 유효하지 않습니다."
        },
        HTTPStatus.NOT_FOUND: {
            "description": "기준 단어가 임베딩 모델 어휘에 없습니다."
        },
        HTTPStatus.UNPROCESSABLE_ENTITY: {
            "description": "순위가 1보다 작거나 vocabulary 크기를 초과했습니다."
        },
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "지원하지 않는 임베딩 제공자가 설정되었습니다."
        },
        HTTPStatus.BAD_GATEWAY: {
            "description": "임베딩 제공자가 유사 단어 조회 중 실패했습니다."
        },
        HTTPStatus.SERVICE_UNAVAILABLE: {
            "description": "임베딩 모델 파일을 로드할 수 없어 서비스를 사용할 수 없습니다."
        },
    },
)
def get_nth_similar_word(
    base_word: Annotated[
        str,
        Path(description="유사 단어를 조회할 기준 단어입니다."),
    ],
    rank: Annotated[
        int,
        Path(description="조회할 유사도 순위입니다. 1부터 시작합니다."),
    ],
    service: EmbeddingService = Depends(get_embedding_service),
) -> NthSimilarWordResponse:
    try:
        result = service.find_nth_similar_word(base_word=base_word, rank=rank)
    except EmbeddingInputError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except EmbeddingNotFoundError as exc:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except EmbeddingRankError as exc:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except EmbeddingInferenceError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return NthSimilarWordResponse(
        base_word=result.base_word,
        rank=result.rank,
        word=result.word,
        similarity=result.similarity,
        vocabulary_size=result.vocabulary_size,
    )


@router.get(
    "/word/{word}",
    response_model=WordDetailResponse,
    tags=["words"],
    summary="단어 상세 조회",
    description="단어의 상세 정보를 조회합니다. 현재는 임베딩 벡터를 반환합니다.",
    response_description="단어와 임베딩 벡터입니다.",
    responses={
        HTTPStatus.BAD_REQUEST: {"description": "요청한 단어가 유효하지 않습니다."},
        HTTPStatus.NOT_FOUND: {
            "description": "요청한 단어가 임베딩 모델 어휘에 없습니다."
        },
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "지원하지 않는 임베딩 제공자가 설정되었습니다."
        },
        HTTPStatus.BAD_GATEWAY: {
            "description": "임베딩 제공자가 벡터 생성 중 실패했습니다."
        },
        HTTPStatus.SERVICE_UNAVAILABLE: {
            "description": "임베딩 모델 파일을 로드할 수 없어 서비스를 사용할 수 없습니다."
        },
    },
)
def get_word_detail(
    word: Annotated[
        str,
        Path(description="상세 정보를 조회할 단어입니다."),
    ],
    service: EmbeddingService = Depends(get_embedding_service),
) -> WordDetailResponse:
    started_at = perf_counter()
    try:
        embeddings = service.generate_embeddings([word])
    except EmbeddingInputError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)
        ) from exc
    except EmbeddingNotFoundError as exc:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except EmbeddingInferenceError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY, detail=str(exc)
        ) from exc

    return WordDetailResponse(
        word=word,
        embedding=embeddings[0].tolist(),
        elapsed_time_ms=_elapsed_time_ms(started_at),
    )


@router.get(
    "/word/{word}/similarity",
    response_model=WordSimilarityResponse,
    tags=["words"],
    summary="단어 유사도 조회",
    description=(
        "`by_word`를 지정하면 두 단어의 유사도와 순위를 계산하고, "
        "`by_rank`를 지정하면 해당 순위의 유사 단어를 조회합니다."
    ),
    response_description="비교 단어의 유사도 순위와 유사도 점수입니다.",
    responses={
        HTTPStatus.BAD_REQUEST: {
            "description": "요청 단어가 유효하지 않거나 query parameter 조합이 잘못되었습니다."
        },
        HTTPStatus.NOT_FOUND: {
            "description": "요청한 단어 중 하나 이상이 임베딩 모델 어휘에 없습니다."
        },
        HTTPStatus.UNPROCESSABLE_ENTITY: {
            "description": "순위가 1보다 작거나 vocabulary 크기를 초과했습니다."
        },
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "지원하지 않는 임베딩 제공자가 설정되었습니다."
        },
        HTTPStatus.BAD_GATEWAY: {
            "description": "임베딩 제공자가 유사도 조회 중 실패했습니다."
        },
        HTTPStatus.SERVICE_UNAVAILABLE: {
            "description": "임베딩 모델 파일을 로드할 수 없어 서비스를 사용할 수 없습니다."
        },
    },
)
def get_word_similarity(
    word: Annotated[
        str,
        Path(description="유사도를 조회할 기준 단어입니다."),
    ],
    by_word: Annotated[
        str | None,
        Query(description="기준 단어와 직접 비교할 단어입니다."),
    ] = None,
    by_rank: Annotated[
        int | None,
        Query(description="조회할 유사도 순위입니다. 1부터 시작합니다."),
    ] = None,
    service: EmbeddingService = Depends(get_embedding_service),
) -> WordSimilarityResponse:
    started_at = perf_counter()
    if (by_word is None and by_rank is None) or (
        by_word is not None and by_rank is not None
    ):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Exactly one of by_word or by_rank must be provided.",
        )

    try:
        if by_word is not None:
            result = service.calculate_similarity_rank(
                base_word=word,
                compared_word=by_word,
            )
            return WordSimilarityResponse(
                base_word=result.base_word,
                compared_word=result.compared_word,
                rank=result.rank,
                similarity=result.similarity,
                vocabulary_size=result.vocabulary_size,
                elapsed_time_ms=_elapsed_time_ms(started_at),
            )

        if by_rank is None:
            raise AssertionError("by_rank must be provided when by_word is absent.")
        result = service.find_nth_similar_word(base_word=word, rank=by_rank)
        return WordSimilarityResponse(
            base_word=result.base_word,
            compared_word=result.word,
            rank=result.rank,
            similarity=result.similarity,
            vocabulary_size=result.vocabulary_size,
            elapsed_time_ms=_elapsed_time_ms(started_at),
        )
    except EmbeddingInputError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except EmbeddingNotFoundError as exc:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except EmbeddingRankError as exc:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except EmbeddingInferenceError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=str(exc),
        ) from exc


def _calculate_similarity(
    word1: str,
    word2: str,
    service: EmbeddingService,
) -> SimilarityResponse:
    try:
        embeddings = service.generate_embeddings([word1, word2])
    except EmbeddingInputError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)
        ) from exc
    except EmbeddingNotFoundError as exc:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except EmbeddingInferenceError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY, detail=str(exc)
        ) from exc

    vec1 = embeddings[0]
    vec2 = embeddings[1]
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if float(denominator) == 0.0:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="Cannot compute similarity for zero-norm vector.",
        )

    similarity = float(np.dot(vec1, vec2) / denominator)
    return SimilarityResponse(similarity=similarity)
