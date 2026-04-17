from __future__ import annotations

from .base import BaseTestDriver


class CostEstimateTest(BaseTestDriver):
    """Placeholder for a project-specific cost test."""

    def name(self) -> str:
        return "cost"

    def run(self) -> None:
        raise NotImplementedError("Rewrite this test for the simpler base class.")
