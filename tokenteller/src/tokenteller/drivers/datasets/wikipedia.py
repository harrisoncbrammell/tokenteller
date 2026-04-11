import json
import random
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from tokenteller.core.types import DatasetQuery, DatasetRecord

from .base import BaseDatasetDriver

class WikipediaCorpusDatasetDriver(BaseDatasetDriver):
    """
    Copy this file when adding a real dataset driver.

    Suggested naming:
    - ``wikipedia_driver.py``
    - ``opensubtitles_driver.py``
    - ``my_custom_dataset_driver.py``
    """

    def __init__(self, data_path: str, name: str = "wikipedia_corpus"):
        super().__init__(name=name)

        self.data_path = Path(data_path)
        self.records = self._load_records()

    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        # TODO: apply filtering and sampling rules based on query
        records = list(self.records)

        for key, expected in query.filters.items():
            records = [
                record
                for record in records
                if record.categories.get(key, record.metadata.get(key)) == expected
            ]

        if query.sample_strategy == "random":
            records = records[:]
            random.Random(query.seed).shuffle(records)
        elif query.sample_strategy == "tail":
            records = records[::-1]

        # Trim the list if the query asked for a limit.
        if query.limit is not None:
            records = records[: query.limit]

        # Yield one record at a time in the final order.
        for record in records:
            yield record

    def _load_records(self) -> list[DatasetRecord]:
        # Build one DatasetRecord for each JSON row in the export file.
        records: list[DatasetRecord] = []
        with self.data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                # Use the standard Common Crawl-style text field.
                text = str(row.get("text", "")).strip()
                if not text:
                    continue

                # Pull a few useful grouping fields out of the row.
                url = str(row.get("url", ""))
                domain = urlparse(url).netloc if url else ""
                language = str(row.get("language", "")).strip()

                # Put easy-to-filter fields in categories.
                categories = {
                    "language": language,
                    "domain": domain,
                    "source": "wikipedia_corpus",
                }

                # Keep the rest of the row in metadata.
                metadata = dict(row)
                metadata.pop("text", None)

                records.append(
                    DatasetRecord(
                        id=str(row.get("id", f"{self.name}-{line_number}")),
                        text=text,
                        categories=categories,
                        metadata=metadata,
                    )
                )

        # Return the finished in-memory list.
        return records
