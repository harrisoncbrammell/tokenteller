from __future__ import annotations

from ..core.types import DatasetQuery
from ..drivers.datasets.base import BaseDatasetDriver
from .base import BaseTestDriver


class CostEstimateTest(BaseTestDriver):
    """Estimate token cost using a fixed per-1k-token price."""

    def __init__(
        self,
        model,
        dataset: BaseDatasetDriver,
        cost_per_1k_tokens: float,
        query: DatasetQuery | None = None,
        label: str | None = None,
    ):
        super().__init__(model=model, label=label)
        self.dataset = dataset
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.query = query or DatasetQuery()

    def name(self) -> str:
        return "cost"

    def run(self) -> None:
        records = list(self.dataset.iter_records(self.query))
        if not records:
            self.warnings.append("No dataset records matched the test query.")
            return

        for record in records:
            tokenization = self.model.tokenize(record.text)
            estimated_cost = tokenization.token_count / 1000.0 * self.cost_per_1k_tokens
            self.results.append(
                self.make_result(
                    record,
                    metrics={
                        "token_count": tokenization.token_count,
                        "estimated_cost": estimated_cost,
                    },
                    tokenization=tokenization,
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
                "cost_per_1k_tokens": self.cost_per_1k_tokens,
                "estimated_cost": sum(result.metrics["estimated_cost"] for result in self.results),
            }
        ]
