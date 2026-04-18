from __future__ import annotations

from collections.abc import Iterable
import os
import sys

if __package__ in {None, ""}:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == script_dir:
        sys.path.pop(0)
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    sys.path.insert(0, src_root)
    from tokenteller.core.types import DatasetQuery, DatasetRecord
    from tokenteller.drivers.datasets.base import BaseDatasetDriver
else:
    from ...core.types import DatasetQuery, DatasetRecord
    from .base import BaseDatasetDriver


class DatasetDriverTemplate(BaseDatasetDriver):
    """
    Copy this file when adding a real dataset driver.

    Suggested naming:
    - ``wikipedia_driver.py``
    - ``opensubtitles_driver.py``
    - ``my_custom_dataset_driver.py``
    """

    def __init__(self, name: str = "my_dataset"):
        super().__init__(name=name)
        # TODO: load dataset state here

    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        # TODO: apply filtering and sampling rules based on query
        yield DatasetRecord(
            id="example-1",
            text="Replace this with a real dataset record.",
            categories={"language": "en", "domain": "example"},
            metadata={},
        )
