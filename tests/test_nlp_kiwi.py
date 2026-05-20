from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.nlp.exceptions import (
    KiwiAnalysisError,
    KiwiConfigurationError,
    KiwiInitializationError,
)
from app.nlp.kiwi import KiwiAnalyzer


class FakeToken:
    def __init__(self, form: str, tag: str = "NNG") -> None:
        self.form = form
        self.tag = tag


class FakeKiwi:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.loaded_user_dictionary: str | None = None
        self.raise_on_tokenize = False
        self.raise_on_analyze = False

    def load_user_dictionary(self, path: str) -> int:
        self.loaded_user_dictionary = path
        return 3

    def tokenize(self, text: str, **kwargs):
        if self.raise_on_tokenize:
            raise RuntimeError("tokenize failed")
        return [FakeToken("안녕"), FakeToken("하세요")]

    def analyze(self, text: str, top_n: int = 1, **kwargs):
        if self.raise_on_analyze:
            raise RuntimeError("analyze failed")
        return [([FakeToken("테스트"), FakeToken("입니다", "EF")], -1.0)]


def test_kiwi_analyzer_initializes_with_arguments_and_loads_user_dictionary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dictionary = tmp_path / "user.dic"
    user_dictionary.write_text("테스트\tNNP\n", encoding="utf-8")

    monkeypatch.setattr(
        "app.nlp.kiwi.importlib.import_module",
        lambda name: SimpleNamespace(Kiwi=FakeKiwi),
    )

    analyzer = KiwiAnalyzer(
        model_path="models/kiwi",
        user_dictionary_path=str(user_dictionary),
        num_workers=2,
        load_default_dict=False,
        integrate_allomorph=False,
        model_type="cong",
        enabled_dialects="standard",
    )

    assert analyzer.raw.init_kwargs == {
        "num_workers": 2,
        "model_path": "models/kiwi",
        "load_default_dict": False,
        "integrate_allomorph": False,
        "model_type": "cong",
        "enabled_dialects": "standard",
    }
    assert analyzer.raw.loaded_user_dictionary == str(user_dictionary)
    assert analyzer.loaded_user_entries == 3


def test_kiwi_analyzer_raises_initialization_error_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_import_error(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("app.nlp.kiwi.importlib.import_module", _raise_import_error)

    with pytest.raises(KiwiInitializationError, match="kiwipiepy"):
        KiwiAnalyzer()


def test_kiwi_analyzer_rejects_missing_user_dictionary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "app.nlp.kiwi.importlib.import_module",
        lambda name: SimpleNamespace(Kiwi=FakeKiwi),
    )

    missing_path = tmp_path / "missing.dic"
    with pytest.raises(KiwiConfigurationError, match="user dictionary file not found"):
        KiwiAnalyzer(user_dictionary_path=str(missing_path))


def test_kiwi_analyzer_tokenize_forms_returns_surface_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.nlp.kiwi.importlib.import_module",
        lambda name: SimpleNamespace(Kiwi=FakeKiwi),
    )
    analyzer = KiwiAnalyzer()

    assert analyzer.tokenize_forms("안녕하세요") == ["안녕", "하세요"]


def test_kiwi_analyzer_wraps_tokenize_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.nlp.kiwi.importlib.import_module",
        lambda name: SimpleNamespace(Kiwi=FakeKiwi),
    )
    analyzer = KiwiAnalyzer()
    analyzer.raw.raise_on_tokenize = True

    with pytest.raises(KiwiAnalysisError, match="tokenizing"):
        analyzer.tokenize("안녕하세요")


def test_kiwi_analyzer_wraps_analyze_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.nlp.kiwi.importlib.import_module",
        lambda name: SimpleNamespace(Kiwi=FakeKiwi),
    )
    analyzer = KiwiAnalyzer()
    analyzer.raw.raise_on_analyze = True

    with pytest.raises(KiwiAnalysisError, match="analyzing"):
        analyzer.analyze("테스트입니다.")


@pytest.mark.parametrize("invalid_text", ["", "   ", None])
def test_kiwi_analyzer_rejects_invalid_text_inputs(
    monkeypatch: pytest.MonkeyPatch,
    invalid_text,
) -> None:
    monkeypatch.setattr(
        "app.nlp.kiwi.importlib.import_module",
        lambda name: SimpleNamespace(Kiwi=FakeKiwi),
    )
    analyzer = KiwiAnalyzer()

    with pytest.raises(KiwiConfigurationError, match="input text"):
        analyzer.tokenize(invalid_text)  # type: ignore[arg-type]
