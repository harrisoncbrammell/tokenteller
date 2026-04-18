from __future__ import annotations

from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord
from tokenteller.testsuites import (
    CompressionRatioTest,
    CostEstimateTest,
    FertilityRateTest,
    MeanTokensPerSentenceTest,
    NSLTest,
    OOVRateTest,
    TokenCountTest,
)
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
            tokenization = self.model.tokenize(record.text)
            self.results.append(self.make_result(record, metrics={"token_count": tokenization.token_count}, tokenization=tokenization))

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
    assert test.results[0].output_metadata["fragment_count"] == 2
    assert test.results[0].output_metadata["has_offsets"] is True


def test_token_count_test_keeps_lightweight_metadata():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="hello world", categories={"language": "en"})],
    )

    test = TokenCountTest(
        FakeTokenizerDriver("word", mode="word"),
        dataset=dataset,
        query=DatasetQuery(limit=1),
        label="token splits",
    )

    report = Experiment().add_test(test).run()

    assert report.summary[0]["token_count"] == 2.0
    assert test.results[0].input_metadata["categories"] == {"language": "en"}
    assert test.results[0].output_metadata["has_offsets"] is True
    assert test.results[0].output_metadata["offset_count"] == 2


def test_cost_estimate_test_computes_total_cost():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="hello world", categories={"language": "en"})],
    )

    test = CostEstimateTest(
        FakeTokenizerDriver("word", mode="word"),
        dataset=dataset,
        cost_per_1k_tokens=0.5,
        query=DatasetQuery(limit=1),
        label="cost demo",
    )

    report = Experiment().add_test(test).run()

    assert report.summary[0]["estimated_cost"] == 0.001
    assert test.results[0].metrics["estimated_cost"] == 0.001


def test_nsl_test_compares_against_baseline_model():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="fragmentation matters", categories={"language": "en"})],
    )

    baseline = FakeTokenizerDriver("word", mode="word")
    test = NSLTest(
        FakeTokenizerDriver("hybrid", mode="hybrid"),
        baseline_model=baseline,
        dataset=dataset,
        query=DatasetQuery(limit=1),
        label="nsl demo",
    )

    report = Experiment().add_test(test).run()

    assert report.summary[0]["baseline"] == "word"
    assert report.summary[0]["nsl"] > 1.0


def test_fragmentation_compare_renders_table():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="fragmentation matters", categories={"language": "en"})],
    )

    first = FragmentationTest(
        FakeTokenizerDriver("word", mode="word"),
        dataset=dataset,
        query=DatasetQuery(limit=1),
        label="word frag",
    )
    second = FragmentationTest(
        FakeTokenizerDriver("hybrid", mode="hybrid"),
        dataset=dataset,
        query=DatasetQuery(limit=1),
        label="hybrid frag",
    )

    Experiment().add_test(first).add_test(second).run()
    comparison = first.compare(second)

    assert "pieces_per_word" in comparison
    assert "hybrid frag" in comparison


def test_compression_ratio_test_uses_token_count_over_char_count():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="hello", categories={"language": "en"})],
    )

    test = CompressionRatioTest(
        FakeTokenizerDriver("char", mode="char"),
        dataset=dataset,
        query=DatasetQuery(limit=1),
        label="compression",
    )

    report = Experiment().add_test(test).run()

    assert report.summary[0]["compression_ratio"] == 1.0


def test_oov_rate_test_defaults_to_zero_without_unknown_markers():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="hello world", categories={"language": "en"})],
    )

    test = OOVRateTest(
        FakeTokenizerDriver("word", mode="word"),
        dataset=dataset,
        query=DatasetQuery(limit=1),
        label="oov",
    )

    report = Experiment().add_test(test).run()

    assert report.summary[0]["oov_rate"] == 0.0


def test_fertility_rate_test_computes_tokens_per_word():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="fragmentation matters", categories={"language": "en"})],
    )

    test = FertilityRateTest(
        FakeTokenizerDriver("hybrid", mode="hybrid"),
        dataset=dataset,
        query=DatasetQuery(limit=1),
        label="fertility",
    )

    report = Experiment().add_test(test).run()

    assert report.summary[0]["fertility_rate"] == 2.0


def test_mean_tokens_per_sentence_test_computes_sentence_average():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="Hello world. Goodbye friend.", categories={"language": "en"})],
    )

    test = MeanTokensPerSentenceTest(
        FakeTokenizerDriver("word", mode="word"),
        dataset=dataset,
        query=DatasetQuery(limit=1),
        label="sentence mean",
    )

    report = Experiment().add_test(test).run()

    assert report.summary[0]["mean_tokens_per_sentence"] == 2.0
