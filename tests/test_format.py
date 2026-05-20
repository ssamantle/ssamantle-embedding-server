from __future__ import annotations

import subprocess
from pathlib import Path


def _ruff_command() -> list[str]:
    local_ruff = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "ruff.exe"
    if local_ruff.exists():
        return [str(local_ruff)]

    return ["ruff"]


def test_code_is_formatted() -> None:
    result = subprocess.run(
        [*_ruff_command(), "format", "--check", "app", "tests"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_imports_are_sorted() -> None:
    result = subprocess.run(
        [*_ruff_command(), "check", "--select", "I", "app", "tests"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
