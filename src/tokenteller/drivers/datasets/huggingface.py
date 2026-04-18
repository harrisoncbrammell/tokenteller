from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from tokenteller.core.types import DatasetQuery, DatasetRecord

from .base import BaseDatasetDriver


class HuggingFaceDatasetDriver(BaseDatasetDriver):
    def __init__(
        self,
        dataset_id: str,
        text_field: str,
        subset: str | None = None,
        split: str = "train",
        name: str | None = None,
        streaming: bool = True,
    ):
        super().__init__(name=name or dataset_id)
        self.dataset_id = dataset_id
        self.text_field = text_field
        self.subset = subset
        self.split = split
        self.streaming = streaming

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceDatasetDriver requires the 'datasets' package."
            ) from exc

        self.dataset = load_dataset(
            dataset_id,
            subset,
            split=split,
            streaming=streaming,
        )

    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        cached_records = self._get_cached_records(query)
        if cached_records is not None:
            for record in cached_records:
                yield record
            return

        filtered_records = self._iter_filtered_records(query)
        strategy = query.sample_strategy
        limit = query.limit
        records: list[DatasetRecord]

        if strategy == "head":
            if limit is None:
                records = list(filtered_records)
                self._store_cached_records(query, records)
                for record in records:
                    yield record
                return

            records = []
            for record in filtered_records:
                if len(records) >= limit:
                    break
                records.append(record)
            self._store_cached_records(query, records)
            for record in records:
                yield record
            return

        if strategy == "tail":
            if limit is None:
                records = list(filtered_records)
            else:
                records = list(deque(filtered_records, maxlen=limit))

            records = list(reversed(records))
            self._store_cached_records(query, records)
            for record in records:
                yield record
            return

        if strategy == "random":
            rng = random.Random(query.seed)

            if limit is None:
                records = list(filtered_records)
                rng.shuffle(records)
                self._store_cached_records(query, records)
                for record in records:
                    yield record
                return

            # Reservoir sample to keep memory bounded for streaming datasets.
            reservoir: list[DatasetRecord] = []
            for seen, record in enumerate(filtered_records, start=1):
                if seen <= limit:
                    reservoir.append(record)
                    continue

                replace_at = rng.randrange(seen)
                if replace_at < limit:
                    reservoir[replace_at] = record

            rng.shuffle(reservoir)
            self._store_cached_records(query, reservoir)
            for record in reservoir:
                yield record
            return

        raise ValueError(
            "Unsupported sample strategy. Expected one of: head, tail, random."
        )

    def _iter_filtered_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        for index, row in enumerate(self.dataset, start=1):
            record = self._row_to_record(row, index=index)
            if record is not None and self._matches(record, query.filters):
                yield record

    def _row_to_record(self, row: object, *, index: int) -> DatasetRecord | None:
        if not isinstance(row, Mapping):
            row = dict(row)

        text = str(row.get(self.text_field, "")).strip()
        if not text:
            return None

        record_id = str(row.get("id", f"{self.name}-{index}"))
        metadata = {key: value for key, value in row.items() if key != self.text_field}

        return DatasetRecord(
            id=record_id,
            text=text,
            categories={"source": "huggingface", "split": self.split},
            metadata=metadata,
        )

    def _matches(self, record: DatasetRecord, filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            actual = record.categories.get(key, record.metadata.get(key))
            if actual != expected:
                return False
        return True
