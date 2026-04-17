from __future__ import annotations

import re

from ..core.types import DatasetQuery, TestCaseResult
from ..drivers.datasets.base import BaseDatasetDriver
from .base import BaseTestDriver


class MeanTokensPerSentenceTest(BaseTestDriver):
    """Compute mean tokens per sentence from total tokens / total sentences."""

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
        return "mean_tokens_per_sentence"

    def run(self) -> None:
        records = list(self.dataset.iter_records(self.query))
        if not records:
            self.warnings.append("No dataset records matched the test query.")
            return

        for record in records:
            tokenization = self.model.encode(record.text)
            sentence_count = _sentence_count(record.text)
            mean_tokens = None if sentence_count == 0 else tokenization.token_count / sentence_count
            self.results.append(
                TestCaseResult(
                    record_id=record.id,
                    tokenizer_name=self.model.name,
                    test_name=self.name(),
                    metrics={
                        "token_count": tokenization.token_count,
                        "sentence_count": sentence_count,
                        "mean_tokens_per_sentence": mean_tokens,
                    },
                    artifacts={"text": record.text, "tokens": tokenization.tokens},
                )
            )

        valid = [result.metrics["mean_tokens_per_sentence"] for result in self.results if result.metrics["mean_tokens_per_sentence"] is not None]
        self.summary = [
            {
                "test": self.label,
                "type": self.name(),
                "model": self.model.name,
                "tokenizer": self.model.name,
                "status": "completed",
                "mean_tokens_per_sentence": sum(valid) / len(valid) if valid else None,
            }
        ]


def _sentence_count(text: str) -> int:
    parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    return len(parts)
