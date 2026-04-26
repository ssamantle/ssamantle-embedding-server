from fastapi import FastAPI

from app.api.v1.route import router as v1_router
from app.core.logger import configure_logger
from app.core.settings import settings

configure_logger()

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
)
app.include_router(v1_router, prefix=settings.api_v1_prefix)


@app.get(
    "/health",
    tags=["health"],
    summary="서비스 상태 확인",
    description="컨테이너와 로드 밸런서 헬스체크를 위한 간단한 상태 응답을 반환합니다.",
    response_description="서비스 상태입니다.",
)
def health_check() -> dict[str, str]:
    return {"status": "ok"}
