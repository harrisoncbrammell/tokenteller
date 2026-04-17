from __future__ import annotations

from .base import BaseTestDriver


class NSLTest(BaseTestDriver):
    """Placeholder for a project-specific NSL test."""

    def name(self) -> str:
        return "nsl"

    def run(self) -> None:
        raise NotImplementedError("Rewrite this test for the simpler base class.")
