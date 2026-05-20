import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


def _load_project_metadata() -> tuple[str, str, str]:
    default_name = "App name not found"
    default_version = "0.0.0"
    default_description = "Description not found."
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"

    try:
        with pyproject_path.open("rb") as file:
            pyproject = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError):
        return default_name, default_version, default_description

    project = pyproject.get("project", {})
    app_name = str(project.get("name", default_name))
    app_version = str(project.get("version", default_version))
    app_description = str(project.get("description", default_description))
    return app_name, app_version, app_description


PROJECT_NAME, PROJECT_VERSION, PROJECT_DESCRIPTION = _load_project_metadata()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_fasttext_model_path() -> str:
    configured = os.getenv("FASTTEXT_MODEL_PATH") or os.getenv("EMBEDDING_MODEL_PATH")
    if configured:
        model_path = Path(configured).expanduser()
    else:
        model_path = Path("data/models/cc.ko.300.bin")

    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()

    return str(model_path)


def _resolve_optional_path(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    path = Path(stripped).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    return str(path)


def _resolve_env_bool(name: str, default: bool) -> bool:
    configured = os.getenv(name)
    if configured is None:
        return default

    normalized = configured.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_env_int(name: str, default: int) -> int:
    configured = os.getenv(name)
    if configured is None:
        return default

    try:
        return int(configured.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = PROJECT_NAME
    app_version: str = PROJECT_VERSION
    app_description: str = PROJECT_DESCRIPTION
    api_v1_prefix: str = "/api/v1"
    fasttext_model_path: str = _resolve_fasttext_model_path()
    kiwi_model_path: str | None = _resolve_optional_path(os.getenv("KIWI_MODEL_PATH"))
    kiwi_user_dictionary_path: str | None = _resolve_optional_path(
        os.getenv("KIWI_USER_DICTIONARY_PATH")
    )
    kiwi_num_workers: int = _resolve_env_int("KIWI_NUM_WORKERS", -1)
    kiwi_load_default_dict: bool = _resolve_env_bool("KIWI_LOAD_DEFAULT_DICT", True)
    kiwi_integrate_allomorph: bool = _resolve_env_bool("KIWI_INTEGRATE_ALLOMORPH", True)
    kiwi_model_type: str | None = os.getenv("KIWI_MODEL_TYPE")
    kiwi_enabled_dialects: str = os.getenv("KIWI_ENABLED_DIALECTS", "standard")


settings: Settings = Settings()
