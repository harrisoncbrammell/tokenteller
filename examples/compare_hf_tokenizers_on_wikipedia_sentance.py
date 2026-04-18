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

OUTPUT_STEM = "wikipedia_single_sentence_tokenizer_comparison"
RESULTS_DIR = Path(__file__).with_name("results")


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


def choose_random_sentence() -> str:
    wikipedia = HuggingFaceDatasetDriver(
        dataset_id="wikimedia/wikipedia",
        subset="20231101.en",
        text_field="text",
        split="train",
        name="wikipedia",
        streaming=True,
    )

    # Pull first 1000 streamed records, then pick one at random.
    pool_query = DatasetQuery(limit=1000, sample_strategy="head")
    articles = list(wikipedia.iter_records(pool_query))
    if not articles:
        raise ValueError("No Wikipedia records were returned.")

    article = random.choice(articles)
    sentences = split_sentences(article.text)
    if not sentences:
        raise ValueError("Could not find a sentence in the sampled Wikipedia article.")
    return random.choice(sentences)


def escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


METRIC_LABELS = {
    "tokenizer": "Tokenizer",
    "token_count": "Token count",
    "compression_ratio": "Compression ratio",
    "oov_rate": "OOV rate",
    "fertility_rate": "Fertility rate",
    "mean_tokens_per_sentence": "Mean tokens per sentence",
    "pieces_per_word": "Pieces per word",
    "max_pieces_per_word": "Max pieces per word",
    "estimated_cost": "Estimated cost",
    "nsl": "Normalized sequence length",
}


METRIC_DESCRIPTIONS = {
    "token_count": "Average number of tokens produced for the selected input.",
    "compression_ratio": "Characters per token, which gives a rough sense of how compactly the tokenizer encodes the text.",
    "oov_rate": "Share of tokens treated as unknown or out-of-vocabulary by the tokenizer.",
    "fertility_rate": "Average number of tokens generated per word in the input.",
    "mean_tokens_per_sentence": "Average token count per sentence. For this report it reflects the sampled sentence input.",
    "pieces_per_word": "Average number of subword pieces used to represent each word.",
    "max_pieces_per_word": "Largest number of pieces used for any single word in the input.",
    "estimated_cost": "Estimated tokenization cost for this input using the example price per 1,000 tokens.",
    "nsl": "Tokenizer length relative to the baseline tokenizer. `1.0` matches the baseline length; larger values mean more tokens than baseline.",
}


def display_label(column: str) -> str:
    return METRIC_LABELS.get(column, column.replace("_", " ").title())


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No rows_"

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    header = "| " + " | ".join(display_label(column) for column in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(escape_markdown_cell(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def pivot_summary_by_tokenizer(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_columns: list[str] = []
    rows_by_tokenizer: dict[str, dict[str, object]] = {}
    ignored_columns = {"test", "type", "model", "tokenizer", "status", "baseline"}

    for summary_row in summary_rows:
        tokenizer = str(summary_row.get("tokenizer", summary_row.get("model", "unknown")))
        tokenizer_row = rows_by_tokenizer.setdefault(tokenizer, {"tokenizer": tokenizer})

        for key, value in summary_row.items():
            if key in ignored_columns:
                continue
            tokenizer_row[key] = value
            if key not in metric_columns:
                metric_columns.append(key)

    return [
        {"tokenizer": row["tokenizer"], **{column: row.get(column, "") for column in metric_columns}}
        for row in rows_by_tokenizer.values()
    ]


def build_markdown_summary(
    sentence: str,
    report_summary: list[dict[str, object]],
) -> str:
    pivoted_summary = pivot_summary_by_tokenizer(report_summary)
    metric_columns = [column for column in pivoted_summary[0] if column != "tokenizer"] if pivoted_summary else []

    lines = [
        "# Hugging Face tokenizer comparison on Wikipedia",
        "",
        "## Selected sentence",
        "",
        f"> {sentence}",
        "",
        "## Test summary",
        "",
        markdown_table(pivoted_summary),
        "",
        "## Metric guide",
        "",
    ]

    for column in metric_columns:
        lines.append(
            f"- **{display_label(column)}**: {METRIC_DESCRIPTIONS.get(column, 'Metric reported by the test summary.')}"
        )

    return "\n".join(lines)


def main() -> None:
    sentence = choose_random_sentence()
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

    for model in models:
        token_count = TokenCountTest(model, dataset=dataset, label=f"{model.name} token count")
        fragmentation = FragmentationTest(model, dataset=dataset, label=f"{model.name} fragmentation")

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
    markdown_summary = build_markdown_summary(
        sentence=sentence,
        report_summary=report.summary,
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"{OUTPUT_STEM}_report.md"
    output_path.write_text(markdown_summary, encoding="utf-8")

    print("Selected Wikipedia sentence:")
    print(sentence)
    print()
    print(f"Markdown summary written to: {output_path}")


if __name__ == "__main__":
    main()
