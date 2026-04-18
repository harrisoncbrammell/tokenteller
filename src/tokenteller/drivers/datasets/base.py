from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ...core.types import DatasetQuery, DatasetRecord


class BaseDatasetDriver(ABC):
    def __init__(self, name: str):
        # Every dataset gets a short stable name for experiment setup.
        self.name = name
        self._records_cache: dict[tuple[object, ...], list[DatasetRecord]] = {}

    @abstractmethod
    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        """Yield dataset records that match the query."""
        raise NotImplementedError

    def _query_cache_key(self, query: DatasetQuery) -> tuple[object, ...]:
        return (
            tuple(sorted((key, repr(value)) for key, value in query.filters.items())),
            query.limit,
            query.sample_strategy,
            query.seed,
        )

    def _get_cached_records(self, query: DatasetQuery) -> list[DatasetRecord] | None:
        return self._records_cache.get(self._query_cache_key(query))

    def _store_cached_records(self, query: DatasetQuery, records: list[DatasetRecord]) -> list[DatasetRecord]:
        self._records_cache[self._query_cache_key(query)] = records
        return records

    def count(self, query: DatasetQuery) -> int | None:
        return len(list(self.iter_records(query)))


__all__ = ["BaseDatasetDriver"]
