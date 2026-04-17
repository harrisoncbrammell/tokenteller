from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .utils import render_table


@dataclass(slots=True)
class DatasetRecord:
    """One text example plus any categories or extra metadata."""

    id: str
    text: str
    categories: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetQuery:
    """Simple filter and sampling settings for dataset drivers."""

    filters: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    sample_strategy: str = "head"
    seed: int | None = None


@dataclass(slots=True)
class TokenizationResult:
    """Shared tokenization output returned by every model driver."""

    token_ids: list[int]
    tokens: list[str]
    token_count: int
    offsets: list[tuple[int, int]] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TestCaseResult:
    """Result of running one test on one record with one model."""

    record_id: str
    tokenizer_name: str
    test_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TestRunReport:
    """Top-level experiment output with summary rows and raw results."""

    summary: list[dict[str, Any]]
    results: list[TestCaseResult]
    warnings: list[str]

    def summary_table(self) -> str:
        """Render the experiment summary rows as a plain text table."""
        return render_table(self.summary)
