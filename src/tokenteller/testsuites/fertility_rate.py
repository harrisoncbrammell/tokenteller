from __future__ import annotations

import re

from ..core.types import DatasetQuery
from ..drivers.datasets.base import BaseDatasetDriver
from .base import BaseTestDriver


class FertilityRateTest(BaseTestDriver):
    """Compute fertility rate T / W from token count and word count."""

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
        return "fertility_rate"

    def run(self) -> None:
        records = list(self.dataset.iter_records(self.query))
        if not records:
            self.warnings.append("No dataset records matched the test query.")
            return

        for record in records:
            tokenization = self.model.tokenize(record.text)
            word_count = len(re.findall(r"\S+", record.text))
            fertility_rate = None if word_count == 0 else tokenization.token_count / word_count
            self.results.append(
                self.make_result(
                    record,
                    metrics={
                        "token_count": tokenization.token_count,
                        "word_count": word_count,
                        "fertility_rate": fertility_rate,
                    },
                    tokenization=tokenization,
                )
            )

        valid = [result.metrics["fertility_rate"] for result in self.results if result.metrics["fertility_rate"] is not None]
        self.summary = [
            {
                "test": self.label,
                "type": self.name(),
                "model": self.model.name,
                "tokenizer": self.model.name,
                "status": "completed",
                "fertility_rate": sum(valid) / len(valid) if valid else None,
            }
        ]
