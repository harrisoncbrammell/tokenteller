from __future__ import annotations

import re
from collections.abc import Iterable

from tokenteller.core.types import DatasetQuery, DatasetRecord, TokenizationResult
from tokenteller.drivers.datasets.base import BaseDatasetDriver
from tokenteller.drivers.models.base import BaseModelDriver


class FakeTokenizerDriver(BaseModelDriver):
    def __init__(self, name: str, mode: str = "word"):
        super().__init__(name=name)
        self.mode = mode
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str) -> TokenizationResult:
        if self.mode == "char":
            tokens = []
            offsets = []
            for index, char in enumerate(text):
                if char.isspace():
                    continue
                tokens.append(char)
                offsets.append((index, index + 1))
        elif self.mode == "hybrid":
            tokens = []
            offsets = []
            for match in re.finditer(r"\S+", text):
                word = match.group(0)
                start, _ = match.span()
                if len(word) <= 4:
                    pieces = [word]
                else:
                    pieces = [word[:3], word[3:]]
                cursor = start
                for piece in pieces:
                    tokens.append(piece)
                    offsets.append((cursor, cursor + len(piece)))
                    cursor += len(piece)
        else:
            tokens = []
            offsets = []
            for match in re.finditer(r"\S+", text):
                tokens.append(match.group(0))
                offsets.append(match.span())

        token_ids = [self._get_id(token) for token in tokens]
        return TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            token_count=len(token_ids),
            offsets=offsets,
            raw={"mode": self.mode},
        )

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self._id_to_token[token_id] for token_id in token_ids]
        if self.mode == "char":
            return "".join(tokens)
        return " ".join(tokens)

    def info(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": f"fake-{self.mode}",
            "vocab_size": max(1, len(self._token_to_id)),
            "supports_offsets": True,
        }

    def _get_id(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]


class FakeDatasetDriver(BaseDatasetDriver):
    def __init__(self, name: str, records: list[DatasetRecord]):
        super().__init__(name=name)
        self.records = records

    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        records = self.records
        for key, expected in query.filters.items():
            records = [
                record
                for record in records
                if record.categories.get(key, record.metadata.get(key)) == expected
            ]

        if query.sample_strategy == "random":
            import random

            records = records[:]
            random.Random(query.seed).shuffle(records)
        elif query.sample_strategy == "tail":
            records = records[::-1]

        if query.limit is None:
            return records
        return records[: query.limit]
