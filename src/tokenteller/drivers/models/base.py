from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ...core.types import TokenizationResult


class BaseModelDriver(ABC):
    def __init__(self, name: str):
         self.name = name

    @abstractmethod
    def encode(self, text: str) -> TokenizationResult:
        """Turn one input string into a shared tokenization result."""
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        # decode tokens to text if supported
        raise NotImplementedError("This model driver does not implement decode().")

    def batch_encode(self, texts: Sequence[str]) -> list[TokenizationResult]:
        # special batch encode api if model has one
        return [self.encode(text) for text in texts]

    def token_count(self, text: str) -> int:
        return self.encode(text).token_count


BaseTokenizerDriver = BaseModelDriver

__all__ = ["BaseModelDriver", "BaseTokenizerDriver"]
