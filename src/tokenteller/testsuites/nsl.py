from __future__ import annotations

from ..core.types import DatasetQuery
from ..drivers.datasets.base import BaseDatasetDriver
from ..drivers.models.base import BaseModelDriver
from .base import BaseTestDriver


class NSLTest(BaseTestDriver):
    """Compare token count against a baseline tokenizer on the same records."""

    def __init__(
        self,
        model,
        baseline_model: BaseModelDriver,
        dataset: BaseDatasetDriver,
        query: DatasetQuery | None = None,
        label: str | None = None,
    ):
        super().__init__(model=model, label=label)
        self.baseline_model = baseline_model
        self.dataset = dataset
        self.query = query or DatasetQuery()

    def name(self) -> str:
        return "nsl"

    def run(self) -> None:
        records = list(self.dataset.iter_records(self.query))
        if not records:
            self.warnings.append("No dataset records matched the test query.")
            return

        for record in records:
            tokenization = self.model.tokenize(record.text)
            baseline_tokenization = self.baseline_model.tokenize(record.text)
            baseline_count = baseline_tokenization.token_count
            nsl = None if baseline_count == 0 else tokenization.token_count / baseline_count

            self.results.append(
                self.make_result(
                    record,
                    metrics={
                        "token_count": tokenization.token_count,
                        "baseline_token_count": baseline_count,
                        "nsl": nsl,
                    },
                    tokenization=tokenization,
                    output_metadata={
                        "baseline_model": self.baseline_model.name,
                        "baseline_tokenizer_metadata": dict(getattr(baseline_tokenization, "raw", {})),
                    },
                )
            )

        valid_nsl = [result.metrics["nsl"] for result in self.results if result.metrics["nsl"] is not None]
        self.summary = [
            {
                "test": self.label,
                "type": self.name(),
                "model": self.model.name,
                "tokenizer": self.model.name,
                "status": "completed",
                "token_count": sum(result.metrics["token_count"] for result in self.results) / len(self.results),
                "baseline": self.baseline_model.name,
                "nsl": sum(valid_nsl) / len(valid_nsl) if valid_nsl else None,
            }
        ]
