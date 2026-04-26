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


@dataclass(frozen=True)
class Settings:
    app_name: str = PROJECT_NAME
    app_version: str = PROJECT_VERSION
    app_description: str = PROJECT_DESCRIPTION
    api_v1_prefix: str = "/api/v1"
    fasttext_model_path: str = _resolve_fasttext_model_path()


settings: Settings = Settings()
