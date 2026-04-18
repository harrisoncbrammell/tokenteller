from __future__ import annotations

from pathlib import Path

from tokenteller.core.types import DatasetQuery, DatasetRecord
from tokenteller.drivers.datasets import CommonCrawlDatasetDriver
from tokenteller.drivers.models import SentencePieceModelDriver
from tokenteller.testsuites.base import BaseTestDriver

from .fakes import FakeDatasetDriver, FakeTokenizerDriver


TEST_DATA_DIR = Path(__file__).resolve().parent / "_generated"


def test_tokenizer_helper_methods_work():
    # Build a simple fake tokenizer for the helper-method checks.
    tokenizer = FakeTokenizerDriver("fake-word", mode="word")

    # The shared model-driver helpers should delegate to encode() correctly.
    batch = tokenizer.batch_encode(["one two", "three"])

    assert [result.token_count for result in batch] == [2, 1]
    assert tokenizer.token_count("one two") == 2


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


def test_common_crawl_driver_builds_records_from_index_captures():
    # Use a fake fetcher so the driver contract stays testable without network access.
    class FakeCapture(dict):
        def __init__(self, content: bytes, **fields):
            super().__init__(fields)
            self.content = content

    class FakeCDXFetcher:
        def iter(self, target, limit=None, filter=None):
            assert target == "*.example.org"
            assert filter == ["=status:200"]
            captures = [
                FakeCapture(
                    b"hello world",
                    url="https://example.org/a",
                    status="200",
                    mime="text/plain",
                    digest="digest-1",
                ),
                FakeCapture(
                    b"hola mundo",
                    url="https://example.org/b",
                    status="200",
                    mime="text/plain",
                    digest="digest-2",
                ),
            ]
            if limit is None:
                return captures
            return captures[:limit]

    driver = CommonCrawlDatasetDriver(fetcher=FakeCDXFetcher())
    records = list(
        driver.iter_records(
            DatasetQuery(
                filters={"domain": "example.org", "status": "200", "mime": "text/plain"},
                limit=1,
            )
        )
    )

    # The driver should produce normal DatasetRecord objects from CDX captures.
    assert [record.id for record in records] == ["digest-1"]
    assert records[0].categories["domain"] == "example.org"
    assert records[0].categories["source"] == "common_crawl"


def test_sentencepiece_driver_imports_and_is_optional():
    # This test keeps the example driver lightweight when sentencepiece is unavailable.
    try:
        import sentencepiece as spm
    except ImportError:
        return

    # Train a tiny temporary SentencePiece model so the example driver can load it.
    TEST_DATA_DIR.mkdir(exist_ok=True)
    temp_path = TEST_DATA_DIR
    corpus_path = temp_path / "sentencepiece_corpus.txt"
    model_prefix = temp_path / "sentencepiece_test"
    model_path = Path(str(model_prefix) + ".model")
    vocab_path = Path(str(model_prefix) + ".vocab")
    corpus_path.write_text("hello world\nhello token\nsentence piece example\n", encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=20,
        model_type="unigram",
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )

    driver = SentencePieceModelDriver(str(model_path), name="spm")
    encoded = driver.encode("hello world")

    # The example driver should return a normal TokenizationResult.
    assert encoded.token_count > 0
    assert driver.decode(encoded.token_ids)


def test_base_test_driver_owns_model_and_results():
    class EchoTest(BaseTestDriver):
        def name(self) -> str:
            return "echo"

        def run(self) -> None:
            record = DatasetRecord(id="1", text="alpha beta")
            tokenization = self.model.tokenize(record.text)
            self.results = [self.make_result(record, metrics={"token_count": tokenization.token_count}, tokenization=tokenization)]

    test = EchoTest(FakeTokenizerDriver("echo-model"))
    test.run()

    assert test.model.name == "echo-model"
    assert test.results[0].metrics["token_count"] == 2
