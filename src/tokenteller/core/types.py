from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from .utils import render_table, serialize_value


class SimpleObject:
    """Small helper mixin for turning results into dicts and text tables."""

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses and helpers into plain dictionaries."""
        return serialize_value(self)

    def summary_rows(self) -> list[dict[str, Any]]:
        """Default one-row summary used by small result objects."""
        return [self.to_dict()]

    def summary_table(self) -> str:
        """Render the summary rows as a plain text table."""
        return render_table(self.summary_rows())


@dataclass(slots=True)
class DatasetRecord(SimpleObject):
    """One text example plus any categories or extra metadata."""

    id: str
    text: str
    categories: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetQuery(SimpleObject):
    """Simple filter and sampling settings for dataset drivers."""

    filters: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    sample_strategy: str = "head"
    seed: int | None = None


@dataclass(slots=True)
class TokenizationResult(SimpleObject):
    """Shared tokenization output returned by every model driver."""

    token_ids: list[int]
    tokens: list[str]
    token_count: int
    offsets: list[tuple[int, int]] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunConfig(SimpleObject):
    """Small group of experiment-wide settings."""

    max_workers: int | None = None
    baseline_tokenizer: str | None = None
    cost_per_1k_tokens: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TestCaseResult(SimpleObject):
    """Result of running one test on one record with one model."""

    record_id: str
    tokenizer_name: str
    test_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TestRunReport(SimpleObject):
    """Top-level experiment output with summary rows and raw results."""

    summary: list[dict[str, Any]]
    results: list[TestCaseResult]
    warnings: list[str]

    def summary_rows(self) -> list[dict[str, Any]]:
        """Use the precomputed summary rows directly."""
        return self.summary


@dataclass
class TestContext:
    """Shared per-test-run state used while processing many records."""

    run_config: RunConfig
    baseline_tokenizer_name: str | None = None
    baseline_token_counts: dict[str, int] = field(default_factory=dict)
    tokenization_cache: dict[tuple[str, str], TokenizationResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _cache_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def get_tokenization(self, tokenizer: Any, record: DatasetRecord) -> TokenizationResult:
        """Cache tokenization results so multiple tests do not recompute them."""
        key = (tokenizer.name, record.id)
        cached = self.tokenization_cache.get(key)
        if cached is not None:
            return cached

        # Lock only around the cache write path so threads can share results safely.
        with self._cache_lock:
            cached = self.tokenization_cache.get(key)
            if cached is not None:
                return cached
            result = tokenizer.encode(record.text)
            self.tokenization_cache[key] = result
            return result

    def add_warning(self, warning: str) -> None:
        """Add a warning once without duplicating it."""
        if warning not in self.warnings:
            self.warnings.append(warning)
