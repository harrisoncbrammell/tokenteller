from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import DatasetRecord, TestCaseResult
from ..drivers.models.base import BaseModelDriver

# TODO: Make markdown output part of test driver or maybe part of experiment driver?
class BaseTestDriver(ABC):
    """Small base class for one complete test."""

    def __init__(self, model: BaseModelDriver, label: str | None = None):
        self.model = model
        self.label = label or self.name()
        self.status = "not_run"
        self.results: list[TestCaseResult] = []
        self.summary: list[dict[str, object]] = []
        self.warnings: list[str] = []

    @abstractmethod
    def name(self) -> str:
        """Return a short stable name for this test type."""
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        """Run the test and save results on the object."""
        raise NotImplementedError

    def make_result(
        self,
        record: DatasetRecord,
        *,
        metrics: dict[str, object],
        tokenization: object | None = None,
        output_metadata: dict[str, object] | None = None,
    ) -> TestCaseResult:
        combined_output_metadata = dict(output_metadata or {})
        if tokenization is not None:
            raw = getattr(tokenization, "raw", None)
            if isinstance(raw, dict):
                combined_output_metadata.update(raw)
            offsets = getattr(tokenization, "offsets", None)
            if offsets is not None:
                combined_output_metadata["has_offsets"] = True
                combined_output_metadata["offset_count"] = len(offsets)

        return TestCaseResult(
            record_id=record.id,
            tokenizer_name=self.model.name,
            test_name=self.name(),
            metrics=metrics,
            input_metadata={
                "categories": dict(record.categories),
                "metadata": dict(record.metadata),
            },
            output_metadata=combined_output_metadata,
        )

    def compare(self, *others: "BaseTestDriver") -> str:
        """Compare completed runs of the same test type."""
        raise NotImplementedError(f"{self.name()} does not support comparisons.")


__all__ = ["BaseTestDriver"]
