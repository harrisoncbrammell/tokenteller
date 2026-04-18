from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ...core.types import TokenizationResult


class BaseModelDriver(ABC):
    def __init__(self, name: str):
         self.name = name
         self._tokenization_cache: dict[tuple[str, bool], TokenizationResult] = {}

    @abstractmethod
    def encode(self, text: str) -> TokenizationResult:
        """Turn one input string into a shared tokenization result."""
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        # decode tokens to text if supported
        raise NotImplementedError("This model driver does not implement decode().")

    def tokenize(self, text: str, *, with_offsets: bool = False) -> TokenizationResult:
        cache_key = (text, with_offsets)
        cached = self._tokenization_cache.get(cache_key)
        if cached is not None:
            return cached

        tokenization = self.encode(text)
        self._tokenization_cache[cache_key] = tokenization
        return tokenization

    def batch_encode(self, texts: Sequence[str]) -> list[TokenizationResult]:
        # special batch encode api if model has one
        return [self.tokenize(text) for text in texts]

    def token_count(self, text: str) -> int:
        return self.tokenize(text).token_count


BaseTokenizerDriver = BaseModelDriver

__all__ = ["BaseModelDriver", "BaseTokenizerDriver"]
