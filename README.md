# 싸맨틀 임베딩 API 서버

FastAPI 기반의 한국어 단어 임베딩 API 서버입니다. fastText 한국어 벡터 모델을 기반으로 단어 임베딩 벡터와 두 단어 간 코사인 유사도를 제공합니다.

## 주요 기능

- 한국어 단어 임베딩 벡터 조회
- 두 단어 임베딩 간 코사인 유사도 계산
- Docker 기반 실행 환경
- OpenAPI 문서 제공

## API

서버 실행 후 OpenAPI 문서는 다음 주소에서 확인할 수 있습니다.

Swagger UI:
```
http://localhost:8080/docs
```

ReDoc:
```
http://localhost:8080/redoc
```

주요 엔드포인트:

```http
GET /health
GET /api/v1/word/{word}
GET /api/v1/word/{word}/similarity?by_word={by_word}
GET /api/v1/word/{word}/similarity?by_rank={by_rank}
```

예시:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/word/사과
curl "http://localhost:8080/api/v1/word/사과/similarity?by_word=배"
curl "http://localhost:8080/api/v1/word/사과/similarity?by_rank=10"
```

이전 엔드포인트인 `/api/v1/embedding/{word}`, `/api/v1/similarity/{word1}/{word2}`, `/api/v1/similarity-rank/{base_word}/{compared_word}`, `/api/v1/similar-words/{base_word}/rank/{rank}`는 deprecated 상태입니다.

## 실행

Docker Compose로 빌드하고 실행합니다.

```bash
docker compose up --build
```

서버는 기본적으로 `8080` 포트에서 실행됩니다.

```text
http://localhost:8080
```

## 임베딩 모델

Docker 빌드 과정에서 fastText 한국어 벡터(`cc.ko.300.vec.gz`)를 내려받고, gensim native format인 `cc.ko.300.kv`로 변환합니다. 런타임에서는 `FASTTEXT_MODEL_PATH`가 가리키는 `.kv` 파일을 `mmap="r"` 방식으로 로드합니다.

기본 모델 경로:

```text
/app/data/models/cc.ko.300.kv
```

모델 경로를 바꾸려면 다음 환경변수를 사용할 수 있습니다.

```text
FASTTEXT_MODEL_PATH
EMBEDDING_MODEL_PATH
```

## 개발

기여를 위한 개발 환경 설정과 커밋 전 검사 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

로컬 테스트는 `uv`를 사용합니다.

```bash
uv run pytest
```
