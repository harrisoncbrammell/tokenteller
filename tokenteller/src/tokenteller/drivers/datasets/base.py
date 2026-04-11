from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ...core.types import DatasetQuery, DatasetRecord


class BaseDatasetDriver(ABC):
    """Base class for any source of text records."""

    def __init__(self, name: str):
        # Every dataset gets a short stable name for experiment setup.
        self.name = name

    @abstractmethod
    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        """Yield dataset records that match the query."""
        raise NotImplementedError

    def count(self, query: DatasetQuery) -> int | None:
        """
        Default count implementation for simple datasets.

        Drivers can override this if counting can be done more efficiently than
        fully iterating through the records.
        """
        return len(list(self.iter_records(query)))


__all__ = ["BaseDatasetDriver"]
