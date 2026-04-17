from __future__ import annotations

import random
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
        records = (
            self._row_to_record(row, index=index)
            for index, row in enumerate(self.dataset, start=1)
        )
        records = [record for record in records if record is not None and self._matches(record, query.filters)]

        if query.sample_strategy == "random":
            random.Random(query.seed).shuffle(records)
        elif query.sample_strategy == "tail":
            records = records[::-1]

        if query.limit is not None:
            records = records[: query.limit]

        for record in records:
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
