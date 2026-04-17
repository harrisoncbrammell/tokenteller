from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import TestCaseResult, TestContext
from ..drivers.models.base import BaseModelDriver


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
    def run(self, context: TestContext) -> None:
        """Run the test and save results on the object."""
        raise NotImplementedError

    def compare(self, *others: "BaseTestDriver") -> str:
        """Compare completed runs of the same test type."""
        raise NotImplementedError(f"{self.name()} does not support comparisons.")


__all__ = ["BaseTestDriver"]
