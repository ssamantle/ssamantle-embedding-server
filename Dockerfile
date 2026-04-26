# 1단계: 외부 리소스 다운로드 (Network Stage)
FROM alpine:latest AS downloader

WORKDIR /download

RUN apk add --no-cache wget git

# 임베딩 벡터 다운로드
RUN wget -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz

# FastText 소스 코드 클론
RUN git clone https://github.com/facebookresearch/fastText.git


# 2단계: 빌드 및 컴파일 (Builder Stage)
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# 다운로더에서 받은 파일들 가져오기
COPY --from=downloader /download/cc.ko.300.vec.gz .
COPY --from=downloader /download/fastText ./fastText

# FastText 설치
RUN uv pip install --system ./fastText

# Python 의존성 설치
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# 벡터 모델 변환
RUN python - <<'PY'
from time import perf_counter

from gensim.models import KeyedVectors

started_at = perf_counter()
print("Converting FastText vector model to gensim native format", flush=True)
vectors = KeyedVectors.load_word2vec_format(
    "/app/cc.ko.300.vec.gz",
    binary=False,
    unicode_errors="ignore",
)
vectors.save("/app/cc.ko.300.kv")
elapsed = perf_counter() - started_at
print(f"Converted FastText vector model in {elapsed:.2f}s", flush=True)
PY


# 3단계: 최종 실행 환경 (Runner Stage)
FROM python:3.12-slim

# .pyc 파일 생성 방지
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENV=production

ENV FASTTEXT_MODEL_PATH=/app/data/models/cc.ko.300.kv

EXPOSE 8000

# 보안을 위해 비루트 사용자 생성 및 전환
RUN useradd -m appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder --chown=appuser:appuser /app/cc.ko.300.kv* /app/data/models/
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser app ./app

USER appuser

# Gunicorn + UvicornWorker 권장이지만 지금은 uvicorn을 사용하도록 함.
# worker 수는 보통 (2 x CPU 코어 수) + 1 로 설정합니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
