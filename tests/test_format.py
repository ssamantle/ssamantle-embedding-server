from __future__ import annotations

import subprocess


def test_code_is_formatted() -> None:
    result = subprocess.run(
        ["ruff", "format", "--check", "app", "tests"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_imports_are_sorted() -> None:
    result = subprocess.run(
        ["ruff", "check", "--select", "I", "app", "tests"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
