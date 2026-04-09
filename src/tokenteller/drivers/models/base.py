from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ...core.types import TokenizationResult


class BaseModelDriver(ABC):
    """Base class for every model/tokenizer driver in the project."""

    def __init__(self, name: str):
        # Every driver gets a short stable name for summaries and lookups.
        self.name = name

    @abstractmethod
    def encode(self, text: str) -> TokenizationResult:
        """Turn one input string into a shared tokenization result."""
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        """
        Optional helper for drivers that can map ids back to text.

        The current project does not require decode(), so driver writers can
        ignore it unless their model library makes it easy to provide.
        """
        raise NotImplementedError("This model driver does not implement decode().")

    def batch_encode(self, texts: Sequence[str]) -> list[TokenizationResult]:
        """Default batch helper built from repeated encode() calls."""
        return [self.encode(text) for text in texts]

    def token_count(self, text: str) -> int:
        """Convenience helper used by reports and tests."""
        return self.encode(text).token_count

    def fragmentation_stats(self, text: str) -> dict[str, Any]:
        """
        Estimate how much the tokenizer splits words into smaller pieces.

        If offsets exist, this is based on exact spans. If not, we still return a
        rough estimate so the rest of the project can continue to run.
        """
        # Start with the tokenizer's own output.
        tokenization = self.encode(text)
        # Split the raw text into whitespace-delimited words.
        words = [{"text": match.group(0), "span": match.span()} for match in re.finditer(r"\S+", text)]
        if not words:
            return {
                "word_count": 0,
                "token_count": tokenization.token_count,
                "pieces_per_word": 0.0,
                "max_pieces_per_word": 0,
                "word_fragments": [],
            }

        word_fragments: list[dict[str, Any]] = []
        if tokenization.offsets:
            # Offsets let us count exactly which tokens overlap each word.
            for word in words:
                start, end = word["span"]
                token_indexes = [
                    index
                    for index, offset in enumerate(tokenization.offsets)
                    if offset and offset[1] > start and offset[0] < end
                ]
                word_fragments.append(
                    {
                        "word": word["text"],
                        "span": (start, end),
                        "pieces": len(token_indexes),
                        "tokens": [tokenization.tokens[index] for index in token_indexes],
                    }
                )
        else:
            # Without offsets we fall back to a rough average estimate.
            average_pieces = max(1, round(tokenization.token_count / len(words)))
            for word in words:
                word_fragments.append(
                    {
                        "word": word["text"],
                        "span": word["span"],
                        "pieces": average_pieces,
                        "tokens": [],
                    }
                )

        # Summarize the per-word fragment counts into a compact stats object.
        piece_counts = [fragment["pieces"] for fragment in word_fragments]
        return {
            "word_count": len(words),
            "token_count": tokenization.token_count,
            "pieces_per_word": sum(piece_counts) / len(piece_counts),
            "max_pieces_per_word": max(piece_counts),
            "word_fragments": word_fragments,
        }


# "Tokenizer" and "model" mean the same thing in this project.
BaseTokenizerDriver = BaseModelDriver

__all__ = ["BaseModelDriver", "BaseTokenizerDriver"]
