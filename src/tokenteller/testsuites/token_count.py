from __future__ import annotations

from ..core.types import DatasetQuery
from ..drivers.datasets.base import BaseDatasetDriver
from .base import BaseTestDriver


class TokenCountTest(BaseTestDriver):
    """Count tokens and keep the token split for each record."""

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
        return "token_count"

    def run(self) -> None:
        records = list(self.dataset.iter_records(self.query))
        if not records:
            self.warnings.append("No dataset records matched the test query.")
            return

        for record in records:
            tokenization = self.model.tokenize(record.text)
            self.results.append(self.make_result(record, metrics={"token_count": tokenization.token_count}, tokenization=tokenization))

        self.summary = [
            {
                "test": self.label,
                "type": self.name(),
                "model": self.model.name,
                "tokenizer": self.model.name,
                "status": "completed",
                "token_count": sum(result.metrics["token_count"] for result in self.results) / len(self.results),
            }
        ]
