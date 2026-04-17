from __future__ import annotations

from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord, TestCaseResult
from tokenteller.testsuites.base import BaseTestDriver
from tokenteller.testsuites.fragmentation import FragmentationTest

from .fakes import FakeDatasetDriver, FakeTokenizerDriver


class EnglishTokenCountTest(BaseTestDriver):
    def __init__(self, model, label: str | None = None):
        super().__init__(model=model, label=label)
        self.dataset = FakeDatasetDriver(
            name="custom",
            records=[
                DatasetRecord(id="1", text="hello world", categories={"language": "en", "domain": "chat"}),
                DatasetRecord(id="2", text="namaste duniya", categories={"language": "hi", "domain": "chat"}),
                DatasetRecord(id="3", text="goodbye friend", categories={"language": "en", "domain": "docs"}),
            ],
        )
        self.query = DatasetQuery(filters={"language": "en"}, limit=2)

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


def test_experiment_end_to_end_returns_summary():
    experiment = Experiment()
    experiment.add_test(EnglishTokenCountTest(FakeTokenizerDriver("tt-word", mode="word"), label="word count"))
    experiment.add_test(EnglishTokenCountTest(FakeTokenizerDriver("tt-char", mode="char"), label="char count"))

    report = experiment.run()
    summary_rows = {(row["tokenizer"], row["test"]): row for row in report.summary}

    assert ("tt-word", "word count") in summary_rows
    assert ("tt-char", "char count") in summary_rows
    assert summary_rows[("tt-word", "word count")]["token_count"] == 2.0
    assert summary_rows[("tt-char", "char count")]["token_count"] > 2.0
    assert "tokenizer" in report.summary_table()


def test_test_object_tracks_status_and_output():
    test = EnglishTokenCountTest(FakeTokenizerDriver("word"), label="english words")

    assert test.status == "not_run"
    assert test.results == []

    Experiment().add_test(test).run()

    assert test.status == "completed"
    assert len(test.results) == 2
    assert test.summary[0]["token_count"] == 2.0


def test_compare_raises_by_default():
    first = EnglishTokenCountTest(FakeTokenizerDriver("word"), label="first")
    second = EnglishTokenCountTest(FakeTokenizerDriver("hybrid", mode="hybrid"), label="second")

    Experiment().add_test(first).add_test(second).run()

    try:
        first.compare(second)
    except NotImplementedError as error:
        assert "does not support comparisons" in str(error)
    else:
        raise AssertionError("compare() should raise by default.")


def test_fragmentation_test_calculates_word_piece_stats():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="fragmentation matters", categories={"language": "en"})],
    )

    test = FragmentationTest(
        FakeTokenizerDriver("hybrid", mode="hybrid"),
        dataset=dataset,
        query=DatasetQuery(limit=1),
        label="fragmentation demo",
    )

    report = Experiment().add_test(test).run()

    assert report.summary[0]["pieces_per_word"] > 1.0
    assert report.summary[0]["max_pieces_per_word"] == 2
    assert test.results[0].artifacts["word_fragments"][0]["tokens"] == ["fra", "gmentation"]
