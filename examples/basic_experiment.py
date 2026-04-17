from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord, RunConfig, TestCaseResult, TestContext, TokenizationResult
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
    def __init__(self, label: str | None = None):
        super().__init__(label=label)
        self.dataset = DemoDataset()
        self.query = DatasetQuery(filters={"language": "en", "domain": "chat"}, limit=10)

    def name(self) -> str:
        return "token_count"

    def get_records(self) -> list[DatasetRecord]:
        return list(self.dataset.iter_records(self.query))

    def run_case(
        self,
        tokenizer: BaseModelDriver,
        record: DatasetRecord,
        context: TestContext,
    ) -> TestCaseResult:
        tokenization = context.get_tokenization(tokenizer, record)
        return TestCaseResult(
            record_id=record.id,
            tokenizer_name=tokenizer.name,
            test_name=self.name(),
            metrics={"token_count": tokenization.token_count},
            artifacts={},
        )


def main() -> None:
    experiment = Experiment(
        run_config=RunConfig(
            baseline_tokenizer="word-demo",
            cost_per_1k_tokens={"word-demo": 0.20, "char-demo": 0.45},
        )
    )
    experiment.add_model(DemoTokenizer("word-demo", mode="word"))
    experiment.add_model(DemoTokenizer("char-demo", mode="char"))
    experiment.add_test(EnglishChatTokenCountTest(label="english chat words"), model="word-demo")
    experiment.add_test(EnglishChatTokenCountTest(label="english chat chars"), model="char-demo")

    report = experiment.run()
    print(report.summary_table())
    if report.warnings:
        print()
        print("Warnings:")
        for warning in report.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
