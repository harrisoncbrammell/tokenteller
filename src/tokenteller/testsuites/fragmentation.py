from __future__ import annotations

import re
from typing import Any

from ..core.types import DatasetQuery, TestCaseResult
from ..drivers.datasets.base import BaseDatasetDriver
from ..core.utils import render_table
from .base import BaseTestDriver


class FragmentationTest(BaseTestDriver):
    """Measure how much a tokenizer splits words into smaller pieces."""

    def __init__(
        self,
        model,
        dataset: BaseDatasetDriver,
        query: DatasetQuery | None = None,
        label: str | None = None,
    ):
        super().__init__(model=model, label=label)
        self.dataset = dataset
        self.query = query or DatasetQuery()

    def name(self) -> str:
        return "fragmentation"

    def run(self) -> None:
        records = list(self.dataset.iter_records(self.query))
        if not records:
            self.warnings.append("No dataset records matched the test query.")
            return

        for record in records:
            tokenization = self.model.encode(record.text)
            stats = _fragmentation_stats(record.text, tokenization)
            self.results.append(
                TestCaseResult(
                    record_id=record.id,
                    tokenizer_name=self.model.name,
                    test_name=self.name(),
                    metrics={
                        "token_count": stats["token_count"],
                        "word_count": stats["word_count"],
                        "pieces_per_word": stats["pieces_per_word"],
                        "max_pieces_per_word": stats["max_pieces_per_word"],
                    },
                    artifacts={"word_fragments": stats["word_fragments"]},
                )
            )

        self.summary = [
            {
                "test": self.label,
                "type": self.name(),
                "model": self.model.name,
                "tokenizer": self.model.name,
                "status": "completed",
                "token_count": sum(result.metrics["token_count"] for result in self.results) / len(self.results),
                "word_count": sum(result.metrics["word_count"] for result in self.results) / len(self.results),
                "pieces_per_word": sum(result.metrics["pieces_per_word"] for result in self.results) / len(self.results),
                "max_pieces_per_word": max(result.metrics["max_pieces_per_word"] for result in self.results),
            }
        ]

    def compare(self, *others: "FragmentationTest") -> str:
        tests = (self, *others)
        for test in tests:
            if not isinstance(test, FragmentationTest):
                raise TypeError("compare() requires FragmentationTest objects.")
            if test.status != "completed" or not test.summary:
                raise ValueError("All fragmentation tests must be completed before comparison.")

        rows = [
            {
                "test": test.label,
                "model": test.model.name,
                "pieces_per_word": test.summary[0]["pieces_per_word"],
                "max_pieces_per_word": test.summary[0]["max_pieces_per_word"],
            }
            for test in tests
        ]
        return render_table(rows)


def _fragmentation_stats(text: str, tokenization) -> dict[str, Any]:
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

    piece_counts = [fragment["pieces"] for fragment in word_fragments]
    return {
        "word_count": len(words),
        "token_count": tokenization.token_count,
        "pieces_per_word": sum(piece_counts) / len(piece_counts),
        "max_pieces_per_word": max(piece_counts),
        "word_fragments": word_fragments,
    }
