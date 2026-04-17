from __future__ import annotations

from ..core.types import DatasetQuery, TestCaseResult
from ..drivers.datasets.base import BaseDatasetDriver
from .base import BaseTestDriver


class OOVRateTest(BaseTestDriver):
    """Compute OOV rate U / T using model-specific or user-supplied unk markers."""

    def __init__(
        self,
        model,
        dataset: BaseDatasetDriver,
        query: DatasetQuery | None = None,
        label: str | None = None,
        unknown_token_ids: set[int] | None = None,
        unknown_tokens: set[str] | None = None,
    ):
        super().__init__(model=model, label=label)
        self.dataset = dataset
        self.query = query or DatasetQuery()
        self.unknown_token_ids = set(unknown_token_ids or [])
        self.unknown_tokens = set(unknown_tokens or [])

    def name(self) -> str:
        return "oov_rate"

    def run(self) -> None:
        records = list(self.dataset.iter_records(self.query))
        if not records:
            self.warnings.append("No dataset records matched the test query.")
            return

        unknown_token_ids = set(self.unknown_token_ids)
        unknown_tokens = set(self.unknown_tokens)
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            unk_token_id = getattr(tokenizer, "unk_token_id", None)
            unk_token = getattr(tokenizer, "unk_token", None)
            if unk_token_id is not None:
                unknown_token_ids.add(unk_token_id)
            if unk_token:
                unknown_tokens.add(unk_token)

        for record in records:
            tokenization = self.model.encode(record.text)
            oov_count = sum(
                1
                for token_id, token in zip(tokenization.token_ids, tokenization.tokens)
                if token_id in unknown_token_ids or token in unknown_tokens
            )
            oov_rate = None if tokenization.token_count == 0 else oov_count / tokenization.token_count
            self.results.append(
                TestCaseResult(
                    record_id=record.id,
                    tokenizer_name=self.model.name,
                    test_name=self.name(),
                    metrics={
                        "token_count": tokenization.token_count,
                        "oov_count": oov_count,
                        "oov_rate": oov_rate,
                    },
                    artifacts={"text": record.text, "tokens": tokenization.tokens},
                )
            )

        valid = [result.metrics["oov_rate"] for result in self.results if result.metrics["oov_rate"] is not None]
        self.summary = [
            {
                "test": self.label,
                "type": self.name(),
                "model": self.model.name,
                "tokenizer": self.model.name,
                "status": "completed",
                "oov_rate": sum(valid) / len(valid) if valid else None,
            }
        ]
