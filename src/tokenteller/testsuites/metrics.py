from __future__ import annotations

from ..core.types import TestContext
from .base import BaseTestDriver


class TokenCountTest(BaseTestDriver):
    """Placeholder for a project-specific token count test."""

    def name(self) -> str:
        return "token_count"

    def run(self, context: TestContext) -> None:
        raise NotImplementedError("Rewrite this test for the simpler base class.")


class FragmentationTest(BaseTestDriver):
    """Placeholder for a project-specific fragmentation test."""

    def name(self) -> str:
        return "fragmentation"

    def run(self, context: TestContext) -> None:
        raise NotImplementedError("Rewrite this test for the simpler base class.")


class NSLTest(BaseTestDriver):
    """Placeholder for a project-specific NSL test."""

    def name(self) -> str:
        return "nsl"

    def run(self, context: TestContext) -> None:
        raise NotImplementedError("Rewrite this test for the simpler base class.")


class CostEstimateTest(BaseTestDriver):
    """Placeholder for a project-specific cost test."""

    def name(self) -> str:
        return "cost"

    def run(self, context: TestContext) -> None:
        raise NotImplementedError("Rewrite this test for the simpler base class.")
