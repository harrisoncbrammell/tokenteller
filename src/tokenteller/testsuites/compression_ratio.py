from __future__ import annotations

from ..core.types import DatasetQuery
from ..drivers.datasets.base import BaseDatasetDriver
from .base import BaseTestDriver


class CompressionRatioTest(BaseTestDriver):
    """Compute compression ratio T / C from token count and character count."""

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
        return "compression_ratio"

    def run(self) -> None:
        records = list(self.dataset.iter_records(self.query))
        if not records:
            self.warnings.append("No dataset records matched the test query.")
            return

        for record in records:
            tokenization = self.model.tokenize(record.text)
            char_count = len(record.text)
            compression_ratio = None if char_count == 0 else tokenization.token_count / char_count
            self.results.append(
                self.make_result(
                    record,
                    metrics={
                        "token_count": tokenization.token_count,
                        "char_count": char_count,
                        "compression_ratio": compression_ratio,
                    },
                    tokenization=tokenization,
                )
            )

        valid = [result.metrics["compression_ratio"] for result in self.results if result.metrics["compression_ratio"] is not None]
        self.summary = [
            {
                "test": self.label,
                "type": self.name(),
                "model": self.model.name,
                "tokenizer": self.model.name,
                "status": "completed",
                "compression_ratio": sum(valid) / len(valid) if valid else None,
            }
        ]
