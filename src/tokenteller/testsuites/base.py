from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from statistics import mean

from ..core.types import DatasetRecord, TestCaseResult, TestContext
from ..core.utils import render_table
from ..drivers.models.base import BaseModelDriver


class BaseTestDriver(ABC):
    """Base class for one metric or analysis."""

    def __init__(self, label: str | None = None):
        # Tests keep their own run state so printing one object is useful.
        self.label = label or self.name()
        self.model_name: str | None = None
        self.status = "not_run"
        self.results: list[TestCaseResult] = []
        self.summary: list[dict[str, object]] = []
        self.warnings: list[str] = []

    @abstractmethod
    def name(self) -> str:
        """Return a short stable name for this test type."""
        raise NotImplementedError

    @abstractmethod
    def run_case(
        self,
        tokenizer: BaseModelDriver,
        record: DatasetRecord,
        context: TestContext,
    ) -> TestCaseResult:
        """Run the test on one tokenizer / one record pair."""
        raise NotImplementedError

    def get_records(self) -> Sequence[DatasetRecord]:
        """Build or fetch the dataset records used by this test."""
        raise NotImplementedError

    def run_batch(
        self,
        tokenizer: BaseModelDriver,
        records: Sequence[DatasetRecord],
        context: TestContext,
    ) -> list[TestCaseResult]:
        """Run the test on each record in order."""
        return [self.run_case(tokenizer, record, context) for record in records]

    def run(
        self,
        tokenizer: BaseModelDriver,
        context: TestContext,
    ) -> list[TestCaseResult]:
        """Fetch records for this test and run the cases."""
        records = list(self.get_records())
        if not records:
            context.warnings.append("No dataset records matched the test query.")
        return self.run_batch(tokenizer, records, context)

    def store_results(
        self,
        *,
        results: list[TestCaseResult],
        warnings: list[str] | None = None,
    ) -> None:
        """Save the outcome of a completed run on the test object itself."""
        self.status = "completed"
        self.results = results
        self.warnings = list(warnings or [])
        self.summary = [self._build_summary_row()]

    def summary_rows(self) -> list[dict[str, object]]:
        """Return either the saved summary row or a simple status row."""
        if self.summary:
            return self.summary

        row: dict[str, object] = {
            "test": self.label,
            "type": self.name(),
            "model": self.model_name or "",
            "tokenizer": self.model_name or "",
            "status": self.status,
        }
        if self.status == "not_run":
            row["message"] = "not run yet"
        return [row]

    def summary_table(self) -> str:
        """Render the current summary rows as a small text table."""
        return render_table(self.summary_rows())

    def compare(self, other: "BaseTestDriver") -> str:
        """
        Compare two completed tests of the same type.

        Tests can override this if they need a more specialized comparison.
        """
        if self.__class__ is not other.__class__:
            raise TypeError("compare() requires two tests of the same type.")
        if self.status != "completed":
            return f"{self.label} has not been run yet."
        if other.status != "completed":
            return f"{other.label} has not been run yet."

        left = self.summary_rows()[0]
        right = other.summary_rows()[0]
        rows = []
        # Only compare numeric summary fields by default.
        for key, value in left.items():
            other_value = right.get(key)
            if isinstance(value, (int, float)) and isinstance(other_value, (int, float)):
                rows.append(
                    {
                        "metric": key,
                        "first": value,
                        "second": other_value,
                        "difference": other_value - value,
                    }
                )
        if not rows:
            return "No comparable numeric metrics."
        return render_table(rows)

    def _build_summary_row(self) -> dict[str, object]:
        """Build one compact summary row from the stored per-record results."""
        row: dict[str, object] = {
            "test": self.label,
            "type": self.name(),
            "model": self.model_name or "",
            "tokenizer": self.model_name or "",
            "status": self.status,
        }
        metrics: dict[str, list[float]] = {}
        # Average any numeric metrics across the test's record-level outputs.
        for result in self.results:
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    metrics.setdefault(key, []).append(float(value))
        for key, values in metrics.items():
            row[key] = mean(values) if values else None
        return row

    def __str__(self) -> str:
        """Make printing a test object useful during interactive work."""
        text = self.summary_table()
        if self.warnings:
            text += "\nWarnings: " + "; ".join(self.warnings)
        return text


__all__ = ["BaseTestDriver"]
