from __future__ import annotations

from .base import BaseTestDriver


class TokenCountTest(BaseTestDriver):
    """Placeholder for a project-specific token count test."""

    def name(self) -> str:
        return "token_count"

    def run(self) -> None:
        raise NotImplementedError("Rewrite this test for the simpler base class.")
