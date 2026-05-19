from dataclasses import dataclass
from pathlib import Path
import os
import tomllib

BASE_DIR = Path(__file__).resolve().parents[3]


with (BASE_DIR / "pyproject.toml").open("rb") as f:
    PYPROJECT = tomllib.load(f)


@dataclass(frozen=True)
class Settings:
    app_name: str = PYPROJECT["project"]["name"]
    app_version: str = PYPROJECT["project"]["version"]
    app_description: str = PYPROJECT["project"]["description"]

    fasttext_model_path: Path = Path(os.environ["FASTTEXT_MODEL_PATH"]).resolve()
