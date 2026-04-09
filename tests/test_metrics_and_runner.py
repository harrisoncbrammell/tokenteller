from __future__ import annotations

from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord, RunConfig
from tokenteller.testsuites.metrics import CostEstimateTest, FragmentationTest, NSLTest, TokenCountTest

from .fakes import FakeDatasetDriver, FakeTokenizerDriver


def test_experiment_end_to_end_returns_summary():
    dataset = FakeDatasetDriver(
        name="custom",
        records=[
            DatasetRecord(id="1", text="hello world", categories={"language": "en", "domain": "chat"}),
            DatasetRecord(id="2", text="नमस्ते दुनिया", categories={"language": "hi", "domain": "chat"}),
        ],
    )

    experiment = Experiment(
        run_config=RunConfig(
            baseline_tokenizer="tt-word",
            cost_per_1k_tokens={"tt-word": 0.25, "tt-char": 0.5},
            max_workers=4,
        )
    )
    experiment.add_model(FakeTokenizerDriver("tt-word", mode="word"))
    experiment.add_model(FakeTokenizerDriver("tt-char", mode="char"))
    experiment.add_dataset(dataset)

    report = experiment.add_tests(
        [TokenCountTest(), FragmentationTest(), NSLTest(), CostEstimateTest()],
        model="tt-word",
        dataset="custom",
    ).add_tests(
        [TokenCountTest(), FragmentationTest(), NSLTest(), CostEstimateTest()],
        model="tt-char",
        dataset="custom",
    ).run()

    summary_rows = {(row["tokenizer"], row["test"]): row for row in report.summary}

    assert ("tt-word", "token_count") in summary_rows
    assert ("tt-char", "nsl") in summary_rows
    assert summary_rows[("tt-word", "nsl")]["nsl"] == 1.0
    assert summary_rows[("tt-char", "cost")]["estimated_cost"] > 0
    assert "tokenizer" in report.summary_table()

def test_nsl_metric_tracks_baseline_ratio():
    dataset = FakeDatasetDriver(
        name="single",
        records=[
            DatasetRecord(id="1", text="fragmentation matters", categories={"language": "en"}),
        ],
    )

    experiment = Experiment(run_config=RunConfig(baseline_tokenizer="tt-word"))
    experiment.add_model(FakeTokenizerDriver("tt-word", mode="word"))
    experiment.add_model(FakeTokenizerDriver("tt-hybrid", mode="hybrid"))
    experiment.add_dataset(dataset)

    report = experiment.add_test(NSLTest(), model="tt-word", dataset="single").add_test(
        NSLTest(),
        model="tt-hybrid",
        dataset="single",
    ).run()

    summary_rows = {(row["tokenizer"], row["test"]): row for row in report.summary}
    assert summary_rows[("tt-word", "nsl")]["nsl"] == 1.0
    assert summary_rows[("tt-hybrid", "nsl")]["nsl"] > 1.0


def test_experiment_object_collects_tests_and_runs_them():
    dataset = FakeDatasetDriver(
        name="single",
        records=[DatasetRecord(id="1", text="hello world", categories={"language": "en"})],
    )
    experiment = Experiment(run_config=RunConfig(baseline_tokenizer="word", max_workers=2))
    experiment.add_model(FakeTokenizerDriver("word"))
    experiment.add_model(FakeTokenizerDriver("hybrid", mode="hybrid"))
    experiment.add_dataset(dataset)

    experiment.add_test(TokenCountTest(), model="word", dataset="single", query=DatasetQuery(limit=1))
    experiment.add_test(NSLTest(), model="hybrid", dataset="single", query=DatasetQuery(limit=1))
    report = experiment.run()

    summary_rows = {(row["model"], row["type"]): row for row in report.summary}
    assert ("word", "token_count") in summary_rows
    assert ("hybrid", "nsl") in summary_rows


def test_test_object_tracks_status_and_compare_output():
    dataset = FakeDatasetDriver(
        name="demo",
        records=[DatasetRecord(id="1", text="hello world", categories={"language": "en"})],
    )
    word_test = TokenCountTest(label="english-word")
    hybrid_test = TokenCountTest(label="english-hybrid")

    assert "not run yet" in str(word_test)

    experiment = Experiment(run_config=RunConfig(max_workers=2))
    experiment.add_model(FakeTokenizerDriver("word"))
    experiment.add_model(FakeTokenizerDriver("hybrid", mode="hybrid"))
    experiment.add_dataset(dataset)
    experiment.add_test(word_test, model="word", dataset="demo")
    experiment.add_test(hybrid_test, model="hybrid", dataset="demo")
    experiment.run()

    assert word_test.status == "completed"
    comparison = word_test.compare(hybrid_test)
    assert "difference" in comparison
