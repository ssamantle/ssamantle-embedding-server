# Contributing

이 문서는 `ssamantle-embedding-server` 개발 시 지켜야 할 기본 규칙을 정리합니다.

## 개발 전 확인

- 변경 범위를 작게 유지합니다.
- 사용자에게 보이는 동작이 바뀌면 테스트와 문서를 함께 갱신합니다.
- API 응답 형식, 상태 코드, 환경변수, Docker 실행 방식이 바뀌면 `README.md`도 확인합니다.

## 개발 환경 설정

개발 의존성을 설치합니다.

```bash
uv sync --dev
```

커밋 전 검사를 자동으로 실행하려면 pre-commit hook을 설치합니다.

```bash
uv run pre-commit install
```

설치 후 `git commit`을 실행하면 configured hook이 자동으로 실행됩니다. 현재 hook은 `uv run pytest`를 실행하며, 테스트 또는 포맷 검사가 실패하면 커밋이 중단됩니다.

hook을 수동으로 전체 파일에 대해 실행하려면 다음 명령을 사용합니다.

```bash
uv run pre-commit run --all-files
```

## 버전 관리

릴리스에 포함될 변경사항을 만들 때는 `pyproject.toml`의 `[project].version` 갱신 여부를 반드시 검토합니다.

버전은 Semantic Versioning 형식을 따릅니다.

```text
MAJOR.MINOR.PATCH
```

- `MAJOR`: 기존 API 또는 사용 방식과 호환되지 않는 변경
- `MINOR`: 하위 호환되는 기능 추가
- `PATCH`: 하위 호환되는 버그 수정, 문서 수정, 내부 개선

예시:

- `0.1.0` -> `0.2.0`: 새 API 추가
- `0.1.0` -> `0.1.1`: 기존 API 버그 수정
- `0.1.0` -> `1.0.0`: 안정 버전으로 공개하거나 호환성 깨지는 변경 포함

버전을 올리는 변경이라면 관련 변경사항을 `CHANGELOG.md`에 기록하는 것을 권장합니다.

## CHANGELOG 작성 권고

사용자에게 영향을 주는 변경사항은 `CHANGELOG.md`에 기록합니다. 아직 파일이 없다면 루트에 새로 만듭니다.

권장 형식:

```md
# Changelog

## [0.2.0] - 2026-04-26

### Added

- 새로 추가된 기능을 적습니다.

### Changed

- 기존 동작의 변경사항을 적습니다.

### Fixed

- 수정한 버그를 적습니다.

### Removed

- 제거된 기능이나 설정을 적습니다.
```

항목 분류:

- `Added`: 새 기능, 새 API, 새 설정
- `Changed`: 기존 동작 변경, 응답 형식 변경, 기본값 변경
- `Fixed`: 버그 수정
- `Removed`: 제거된 기능, 제거된 설정
- `Deprecated`: 앞으로 제거될 기능
- `Security`: 보안 관련 수정

기록할 때는 구현 세부사항보다 사용자나 운영자가 알아야 할 영향을 우선합니다.

## 테스트

변경 후 가능한 경우 테스트를 실행합니다.

```bash
uv run pytest
```

테스트를 실행하지 못했다면 PR 또는 작업 기록에 이유를 남깁니다.
