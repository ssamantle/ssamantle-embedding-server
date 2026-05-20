from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from app.core.settings import settings
from app.nlp.exceptions import (
    KiwiAnalysisError,
    KiwiConfigurationError,
    KiwiInitializationError,
)


class KiwiAnalyzer:
    """Thin wrapper around kiwipiepy's Kiwi analyzer.

    The wrapper keeps the external dependency lazy so the rest of the
    application can import this module even when `kiwipiepy` is not installed.
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        user_dictionary_path: str | None = None,
        num_workers: int | None = None,
        load_default_dict: bool | None = None,
        integrate_allomorph: bool | None = None,
        model_type: str | None = None,
        enabled_dialects: str | None = None,
    ) -> None:
        self.model_path = model_path or settings.kiwi_model_path
        self.user_dictionary_path = (
            user_dictionary_path or settings.kiwi_user_dictionary_path
        )
        self.num_workers = (
            settings.kiwi_num_workers if num_workers is None else num_workers
        )
        self.load_default_dict = (
            settings.kiwi_load_default_dict
            if load_default_dict is None
            else load_default_dict
        )
        self.integrate_allomorph = (
            settings.kiwi_integrate_allomorph
            if integrate_allomorph is None
            else integrate_allomorph
        )
        self.model_type = model_type or settings.kiwi_model_type
        self.enabled_dialects = enabled_dialects or settings.kiwi_enabled_dialects

        self._kiwi = self._initialize_kiwi()
        self.loaded_user_entries = self._load_user_dictionary_if_configured()

    @property
    def raw(self) -> Any:
        """Expose the underlying Kiwi instance for advanced use cases."""
        return self._kiwi

    def tokenize(self, text: str, **kwargs: Any) -> list[Any]:
        self._require_text(text)
        try:
            return list(self._kiwi.tokenize(text, **kwargs))
        except Exception as exc:
            raise KiwiAnalysisError("Kiwi failed while tokenizing text.") from exc

    def tokenize_forms(self, text: str, **kwargs: Any) -> list[str]:
        return [str(token.form) for token in self.tokenize(text, **kwargs)]

    def analyze(self, text: str, top_n: int = 1, **kwargs: Any) -> list[Any]:
        self._require_text(text)
        try:
            return list(self._kiwi.analyze(text, top_n=top_n, **kwargs))
        except Exception as exc:
            raise KiwiAnalysisError("Kiwi failed while analyzing text.") from exc

    def _initialize_kiwi(self) -> Any:
        try:
            kiwi_module = importlib.import_module("kiwipiepy")
            kiwi_cls = getattr(kiwi_module, "Kiwi")
        except Exception as exc:
            raise KiwiInitializationError(
                "kiwipiepy is required to use KiwiAnalyzer. "
                "Install the 'kiwipiepy' package first."
            ) from exc

        try:
            return kiwi_cls(
                num_workers=self.num_workers,
                model_path=self.model_path,
                load_default_dict=self.load_default_dict,
                integrate_allomorph=self.integrate_allomorph,
                model_type=self.model_type,
                enabled_dialects=self.enabled_dialects,
            )
        except Exception as exc:
            raise KiwiInitializationError(
                "Failed to initialize kiwipiepy.Kiwi."
            ) from exc

    def _load_user_dictionary_if_configured(self) -> int:
        if self.user_dictionary_path is None:
            return 0

        dictionary_path = Path(self.user_dictionary_path)
        if not dictionary_path.exists() or not dictionary_path.is_file():
            raise KiwiConfigurationError(
                f"Kiwi user dictionary file not found: {dictionary_path}"
            )

        try:
            loaded_entries = self._kiwi.load_user_dictionary(str(dictionary_path))
        except Exception as exc:
            raise KiwiInitializationError(
                f"Failed to load Kiwi user dictionary: {dictionary_path}"
            ) from exc

        return int(loaded_entries)

    @staticmethod
    def _require_text(text: str) -> None:
        if not isinstance(text, str):
            raise KiwiConfigurationError("Kiwi input text must be a string.")
        if not text.strip():
            raise KiwiConfigurationError("Kiwi input text must not be blank.")


__all__ = ["KiwiAnalyzer"]
