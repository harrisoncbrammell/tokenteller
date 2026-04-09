from __future__ import annotations

from tokenteller.core.types import DatasetQuery, DatasetRecord, RunConfig, TestContext
from tokenteller.testsuites.base import BaseTestDriver
from tokenteller.testsuites.metrics import TokenCountTest

from .fakes import FakeDatasetDriver, FakeTokenizerDriver


def test_tokenizer_helper_methods_work():
    # Build a simple fake tokenizer for the helper-method checks.
    tokenizer = FakeTokenizerDriver("fake-word", mode="word")

    # The shared model-driver helpers should delegate to encode() correctly.
    batch = tokenizer.batch_encode(["one two", "three"])

    assert [result.token_count for result in batch] == [2, 1]
    assert tokenizer.token_count("one two") == 2
    stats = tokenizer.fragmentation_stats("hello world")
    assert stats["word_count"] == 2
    assert stats["pieces_per_word"] == 1.0


def test_dataset_driver_supports_filters_and_deterministic_sampling():
    # Build a small mixed dataset for filter and sampling checks.
    dataset = FakeDatasetDriver(
        name="demo",
        records=[
            DatasetRecord(id="1", text="hello", categories={"language": "en", "domain": "wiki"}),
            DatasetRecord(id="2", text="नमस्ते", categories={"language": "hi", "domain": "wiki"}),
            DatasetRecord(id="3", text="bye", categories={"language": "en", "domain": "chat"}),
        ],
    )

    # Filtering should keep only the matching Hindi row.
    hindi = list(dataset.iter_records(DatasetQuery(filters={"language": "hi"})))
    # Random sampling should repeat with the same seed.
    first_random = [record.id for record in dataset.iter_records(DatasetQuery(sample_strategy="random", seed=7))]
    second_random = [record.id for record in dataset.iter_records(DatasetQuery(sample_strategy="random", seed=7))]

    assert [record.id for record in hindi] == ["2"]
    assert first_random == second_random


def test_direct_driver_instantiation_is_enough():
    # The library should work with direct driver objects and no registry.
    tokenizer = FakeTokenizerDriver("direct-word", mode="word")

    assert tokenizer.name == "direct-word"
    assert tokenizer.token_count("a b c") == 3


def test_base_test_driver_parallel_batch_preserves_input_order():
    # This fake test adds delays so futures finish out of order.
    class OrderedEchoTest(BaseTestDriver):
        def name(self) -> str:
            return "ordered_echo"

        def run_case(self, tokenizer, record, context):
            # Delay two records to force out-of-order completion.
            import time

            if record.id == "1":
                time.sleep(0.03)
            if record.id == "2":
                time.sleep(0.01)
            return TokenCountTest().run_case(tokenizer, record, context)

    # The final result order should still match the input order.
    records = [
        DatasetRecord(id="1", text="alpha beta"),
        DatasetRecord(id="2", text="gamma"),
        DatasetRecord(id="3", text="delta epsilon zeta"),
    ]
    context = TestContext(
        run_config=RunConfig(max_workers=3),
    )

    results = OrderedEchoTest().run_batch(FakeTokenizerDriver("parallel-fake"), records, context)

    assert [result.record_id for result in results] == ["1", "2", "3"]
