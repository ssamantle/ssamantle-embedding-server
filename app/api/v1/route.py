from __future__ import annotations

from http import HTTPStatus
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel

from app.providers.fasttext import FastTextModelLoadError
from app.services import (
    EmbeddingInferenceError,
    EmbeddingInputError,
    EmbeddingNotFoundError,
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


def get_embedding_service() -> EmbeddingService:
    try:
        provider = get_embedding_provider()
    except FastTextModelLoadError as exc:
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
