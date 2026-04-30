from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .utils import render_table


@dataclass(slots=True)
class DatasetRecord:
    id: str
    text: str
    categories: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetQuery:
    filters: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    sample_strategy: str = "random"
    seed: int | None = None


@dataclass(slots=True)
class TokenizationResult:
    token_ids: list[int]
    tokens: list[str]
    token_count: int
    offsets: list[tuple[int, int]] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TestCaseResult:
    record_id: str
    tokenizer_name: str
    test_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    input_metadata: dict[str, Any] = field(default_factory=dict)
    output_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TestRunReport:
    summary: list[dict[str, Any]]
    results: list[TestCaseResult]
    warnings: list[str]

    def summary_table(self) -> str:
        return render_table(self.summary)
