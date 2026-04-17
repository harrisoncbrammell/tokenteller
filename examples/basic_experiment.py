from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord, TestCaseResult, TokenizationResult
from tokenteller.drivers.datasets.base import BaseDatasetDriver
from tokenteller.drivers.models.base import BaseModelDriver
from tokenteller.testsuites.base import BaseTestDriver


class DemoTokenizer(BaseModelDriver):
    def __init__(self, name: str, mode: str = "word"):
        super().__init__(name=name)
        self.mode = mode
        self._token_to_id: dict[str, int] = {}

    def encode(self, text: str) -> TokenizationResult:
        if self.mode == "char":
            tokens = []
            offsets = []
            for index, char in enumerate(text):
                if char.isspace():
                    continue
                tokens.append(char)
                offsets.append((index, index + 1))
        else:
            tokens = []
            offsets = []
            for match in re.finditer(r"\S+", text):
                tokens.append(match.group(0))
                offsets.append(match.span())

        token_ids = [self._token_id(token) for token in tokens]
        return TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            token_count=len(token_ids),
            offsets=offsets,
            raw={"mode": self.mode},
        )

    def _token_id(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self._token_to_id) + 1
        return self._token_to_id[token]


class DemoDataset(BaseDatasetDriver):
    def __init__(self):
        super().__init__(name="demo")
        self.records = [
            DatasetRecord(id="1", text="hello world", categories={"language": "en", "domain": "chat"}),
            DatasetRecord(
                id="2",
                text="tokenization behaves differently",
                categories={"language": "en", "domain": "docs"},
            ),
            DatasetRecord(id="3", text="hello again", categories={"language": "en", "domain": "chat"}),
        ]

    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        records = self.records
        for key, expected in query.filters.items():
            records = [record for record in records if record.categories.get(key) == expected]
        if query.sample_strategy == "tail":
            records = list(reversed(records))
        if query.limit is None:
            return records
        return records[: query.limit]


class EnglishChatTokenCountTest(BaseTestDriver):
    def __init__(self, model: BaseModelDriver, label: str | None = None):
        super().__init__(model=model, label=label)
        self.dataset = DemoDataset()
        self.query = DatasetQuery(filters={"language": "en", "domain": "chat"}, limit=10)

    def name(self) -> str:
        return "token_count"

    def run(self) -> None:
        records = list(self.dataset.iter_records(self.query))
        if not records:
            self.warnings.append("No dataset records matched the test query.")
            return

        for record in records:
            tokenization = self.model.encode(record.text)
            self.results.append(
                TestCaseResult(
                    record_id=record.id,
                    tokenizer_name=self.model.name,
                    test_name=self.name(),
                    metrics={"token_count": tokenization.token_count},
                    artifacts={},
                )
            )

        average = sum(result.metrics["token_count"] for result in self.results) / len(self.results)
        self.summary = [
            {
                "test": self.label,
                "type": self.name(),
                "model": self.model.name,
                "tokenizer": self.model.name,
                "status": "completed",
                "token_count": average,
            }
        ]


def main() -> None:
    experiment = Experiment()
    experiment.add_test(EnglishChatTokenCountTest(DemoTokenizer("word-demo", mode="word"), label="english chat words"))
    experiment.add_test(EnglishChatTokenCountTest(DemoTokenizer("char-demo", mode="char"), label="english chat chars"))

    report = experiment.run()
    print(report.summary_table())
    if report.warnings:
        print()
        print("Warnings:")
        for warning in report.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
