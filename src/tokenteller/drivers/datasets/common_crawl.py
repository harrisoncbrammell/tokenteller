from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from urllib.parse import urlparse

from tokenteller.core.types import DatasetQuery, DatasetRecord

from .base import BaseDatasetDriver


class CommonCrawlDatasetDriver(BaseDatasetDriver):
    """
    Small Common Crawl driver backed by cdx_toolkit.

    This driver keeps the same DatasetQuery shape used by every other dataset
    driver. It looks for a few plain filter keys and translates them into a CDX
    query:

    - ``url``: exact url or wildcarded Common Crawl query
    - ``url_pattern``: explicit CDX query string such as ``example.com/*``
    - ``domain``: domain query converted to ``*.example.com``
    - ``status``: exact HTTP status filter
    - ``mime``: exact mime filter

    Other filter keys are applied after records are built, so callers can keep
    using the same ``DatasetQuery(filters=...)`` pattern across drivers.

    Example:

    ```python
    from tokenteller.core.types import DatasetQuery
    from tokenteller.drivers.datasets import CommonCrawlDatasetDriver

    dataset = CommonCrawlDatasetDriver()

    query = DatasetQuery(filters={"domain": "example.org"}, limit=10)
    query = DatasetQuery(filters={"url": "https://commoncrawl.org/"}, limit=5)
    query = DatasetQuery(
        filters={"url_pattern": "*.example.org/*", "status": "200", "mime": "text/html"},
        limit=20,
    )
    ```
    """

    def __init__(self, name: str = "common_crawl", fetcher: object | None = None):
        super().__init__(name=name)
        self.fetcher = fetcher or self._build_fetcher()

    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        cached_records = self._get_cached_records(query)
        if cached_records is not None:
            for record in cached_records:
                yield record
            return

        filters = query.filters
        if "url_pattern" in filters:
            target = str(filters["url_pattern"])
        elif "url" in filters:
            target = str(filters["url"])
        elif "domain" in filters:
            domain = str(filters["domain"]).strip()
            if domain.startswith("*.") or domain.endswith("/*"):
                target = domain
            else:
                target = f"*.{domain.lstrip('.')}"
        else:
            raise ValueError(
                "CommonCrawlDatasetDriver requires one of filters['url'], "
                "filters['url_pattern'], or filters['domain']."
            )

        cdx_filters: list[str] = []
        if "status" in filters:
            cdx_filters.append(f"=status:{filters['status']}")
        if "mime" in filters:
            cdx_filters.append(f"=mime:{filters['mime']}")

        records: list[DatasetRecord] = []
        for index, capture in enumerate(
            self.fetcher.iter(target, limit=query.limit, filter=cdx_filters or None),
            start=1,
        ):
            record = self._capture_to_record(capture, index=index)
            if record is None:
                continue

            matches = True
            for key, expected in filters.items():
                if key in {"url", "url_pattern"}:
                    continue
                actual = record.categories.get(key, record.metadata.get(key))
                if actual != expected:
                    matches = False
                    break
            if matches:
                records.append(record)

        if query.sample_strategy == "random":
            records = records[:]
            random.Random(query.seed).shuffle(records)
        elif query.sample_strategy == "tail":
            records = records[::-1]

        if query.limit is not None:
            records = records[: query.limit]

        self._store_cached_records(query, records)
        for record in records:
            yield record

    def _build_fetcher(self) -> object:
        try:
            import cdx_toolkit
        except ImportError as exc:
            raise ImportError(
                "CommonCrawlDatasetDriver requires the 'cdx-toolkit' package."
            ) from exc

        return cdx_toolkit.CDXFetcher(source="cc")

    def _capture_to_record(self, capture: object, *, index: int) -> DatasetRecord | None:
        if isinstance(capture, Mapping):
            row = dict(capture)
        elif hasattr(capture, "items") and callable(capture.items):
            row = dict(capture.items())
        else:
            raise TypeError("Common Crawl captures must behave like dictionaries.")

        content = getattr(capture, "content", row.pop("content", None))
        if content is None:
            text = ""
        elif isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace").strip()
        else:
            text = str(content).strip()
        if not text:
            return None

        url = str(row.get("url", ""))
        parsed = urlparse(url)
        domain = parsed.netloc
        status = str(row.get("status", ""))
        mime = str(row.get("mime", row.get("mime-detected", "")))
        record_id = str(row.get("digest") or row.get("urlkey") or f"{self.name}-{index}")

        return DatasetRecord(
            id=record_id,
            text=text,
            categories={
                "source": "common_crawl",
                "domain": domain,
                "status": status,
                "mime": mime,
            },
            metadata=row,
        )
