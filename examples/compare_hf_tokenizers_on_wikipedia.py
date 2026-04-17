from __future__ import annotations

import random
import re
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord
from tokenteller.drivers.datasets.base import BaseDatasetDriver
from tokenteller.drivers.datasets import HuggingFaceDatasetDriver
from tokenteller.drivers.models import HuggingFaceTokenizerDriver
from tokenteller.testsuites import (
    CompressionRatioTest,
    CostEstimateTest,
    FertilityRateTest,
    FragmentationTest,
    MeanTokensPerSentenceTest,
    NSLTest,
    OOVRateTest,
    TokenCountTest,
)


class SingleSentenceDataset(BaseDatasetDriver):
    def __init__(self, sentence: str):
        super().__init__(name="single_sentence")
        self.record = DatasetRecord(
            id="selected-sentence",
            text=sentence,
            categories={"source": "wikipedia", "unit": "sentence"},
            metadata={},
        )

    def iter_records(self, query: DatasetQuery) -> Iterable[DatasetRecord]:
        return [self.record]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def choose_random_sentence(seed: int = 7) -> str:
    wikipedia = HuggingFaceDatasetDriver(
        dataset_id="wikimedia/wikipedia",
        subset="20231101.en",
        text_field="text",
        split="train",
        name="wikipedia",
        streaming=True,
    )

    # Pull first 1000 streamed records, then pick one deterministically.
    pool_query = DatasetQuery(limit=1000, sample_strategy="head")
    articles = list(wikipedia.iter_records(pool_query))
    if not articles:
        raise ValueError("No Wikipedia records were returned.")

    article = random.Random(seed).choice(articles)
    sentences = split_sentences(article.text)
    if not sentences:
        raise ValueError("Could not find a sentence in the sampled Wikipedia article.")
    return random.Random(seed).choice(sentences)


def main() -> None:
    sentence = choose_random_sentence(seed=7)
    dataset = SingleSentenceDataset(sentence)

    gpt2 = HuggingFaceTokenizerDriver("gpt2", name="gpt2")
    llama = HuggingFaceTokenizerDriver("openlm-research/open_llama_3b", name="llama-sp")
    t5 = HuggingFaceTokenizerDriver("t5-small", name="t5")
    wordpiece = HuggingFaceTokenizerDriver("bert-base-uncased", name="wordpiece")

    models = [gpt2, llama, t5, wordpiece]
    baseline_model = gpt2
    cost_per_1k = {
        "gpt2": 0.0020,
        "llama-sp": 0.0030,
        "t5": 0.0025,
        "wordpiece": 0.0015,
    }

    experiment = Experiment()
    token_count_tests: list[TokenCountTest] = []
    fragmentation_tests: list[FragmentationTest] = []

    for model in models:
        token_count = TokenCountTest(model, dataset=dataset, label=f"{model.name} token count")
        fragmentation = FragmentationTest(model, dataset=dataset, label=f"{model.name} fragmentation")

        token_count_tests.append(token_count)
        fragmentation_tests.append(fragmentation)

        experiment.add_test(token_count)
        experiment.add_test(CompressionRatioTest(model, dataset=dataset, label=f"{model.name} compression"))
        experiment.add_test(OOVRateTest(model, dataset=dataset, label=f"{model.name} oov"))
        experiment.add_test(FertilityRateTest(model, dataset=dataset, label=f"{model.name} fertility"))
        experiment.add_test(MeanTokensPerSentenceTest(model, dataset=dataset, label=f"{model.name} sentence mean"))
        experiment.add_test(fragmentation)
        experiment.add_test(
            CostEstimateTest(
                model,
                dataset=dataset,
                cost_per_1k_tokens=cost_per_1k[model.name],
                label=f"{model.name} cost",
            )
        )
        experiment.add_test(
            NSLTest(
                model,
                baseline_model=baseline_model,
                dataset=dataset,
                label=f"{model.name} nsl",
            )
        )

    report = experiment.run()

    print("Selected Wikipedia sentence:")
    print(sentence)
    print()
    print(report.summary_table())
    print()
    print("Token splits:")
    for test in token_count_tests:
        tokens = test.results[0].artifacts["tokens"]
        print(f"{test.model.name}: {tokens}")
    print()
    print("Fragmentation comparison:")
    print(fragmentation_tests[0].compare(*fragmentation_tests[1:]))


if __name__ == "__main__":
    main()
