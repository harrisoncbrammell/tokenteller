from __future__ import annotations

import csv
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenteller import Experiment
from tokenteller.core.types import DatasetQuery, DatasetRecord, TestRunReport
from tokenteller.drivers.datasets import HuggingFaceDatasetDriver
from tokenteller.drivers.datasets.base import BaseDatasetDriver
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


DATASET_ID = "wikimedia/wikipedia"
DATASET_SUBSET = "20231101.en"
SAMPLE_SIZE = 100
POOL_SIZE = 10000
SAMPLE_SEED = 123124
OUTPUT_STEM = "wikipedia_tokenizer_comparison"
RESULTS_DIR = Path(__file__).with_name("results")


class SampledDataset(BaseDatasetDriver):
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
    "token_count": "Average number of tokens produced per sampled article.",
    "compression_ratio": "Average token-to-character ratio, which gives a rough sense of how compactly the tokenizer encodes the text.",
    "oov_rate": "Average share of tokens treated as unknown or out-of-vocabulary by the tokenizer.",
    "fertility_rate": "Average number of tokens generated per word.",
    "mean_tokens_per_sentence": "Average token count per sentence across the sampled articles.",
    "pieces_per_word": "Average number of subword pieces used to represent each word.",
    "max_pieces_per_word": "Largest number of pieces used for any single word seen in the sampled articles.",
    "estimated_cost": "Estimated tokenization cost across the sampled articles using the example price per 1,000 tokens.",
    "nsl": "Tokenizer length relative to the baseline tokenizer. `1.0` matches the baseline length; larger values mean more tokens than baseline.",
}


def escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


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
    ignored_columns = {"test", "type", "model", "tokenizer", "status", "baseline", "cost_per_1k_tokens"}

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
    *,
    sample_size: int,
    pool_size: int,
    dataset_name: str,
    subset: str,
    report_summary: list[dict[str, object]],
) -> str:
    pivoted_summary = pivot_summary_by_tokenizer(report_summary)
    metric_columns = [column for column in pivoted_summary[0] if column != "tokenizer"] if pivoted_summary else []

    lines = [
        "# Hugging Face tokenizer comparison on random Wikipedia articles",
        "",
        "## Sample",
        "",
        f"- **Dataset**: `{dataset_name}`",
        f"- **Subset**: `{subset}`",
        "- **Record type**: one Wikipedia article from the dataset.",
        "- **Sampling method**: bounded random sample.",
        f"- **Pool size**: first {pool_size} streamed articles",
        f"- **Sample size**: {sample_size} random articles",
        f"- **How records were chosen**: we first streamed the first {pool_size} articles from the dataset, then used a fixed random seed (`{SAMPLE_SEED}`) to select {sample_size} articles from that pooled set.",
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


def print_progress(prefix: str, current: int, total: int) -> None:
    width = 28
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if current >= total else ""
    print(f"\r{prefix} [{bar}] {current}/{total}", end=end, flush=True)


def collect_pool_with_progress(dataset: BaseDatasetDriver, query: DatasetQuery) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    total = query.limit or 0
    print_progress("Sampling pool", 0, total)
    for index, record in enumerate(dataset.iter_records(query), start=1):
        records.append(record)
        print_progress("Sampling pool", index, total)
    return records


def run_experiment_with_progress(experiment: Experiment) -> TestRunReport:
    if not experiment.tests:
        raise ValueError("Experiment.run() requires at least one test.")

    summary: list[dict[str, object]] = []
    results = []
    warnings: list[str] = []
    total = len(experiment.tests)

    print_progress("Running tests", 0, total)
    for index, test in enumerate(experiment.tests, start=1):
        test.status = "not_run"
        test.results = []
        test.summary = []
        test.warnings = []

        test.run()
        test.status = "completed"

        summary.extend(test.summary or [experiment._default_summary_row(test)])
        results.extend(test.results)
        warnings.extend(test.warnings)
        print_progress("Running tests", index, total)

    return TestRunReport(summary=summary, results=results, warnings=warnings)


def write_results_csv(report: TestRunReport, output_path: Path) -> None:
    metric_columns = sorted({key for result in report.results for key in result.metrics})
    fieldnames = [
        "record_id",
        "tokenizer_name",
        "test_name",
        *metric_columns,
        "input_categories_json",
        "input_metadata_json",
        "output_metadata_json",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in report.results:
            row = {
                "record_id": result.record_id,
                "tokenizer_name": result.tokenizer_name,
                "test_name": result.test_name,
                "input_categories_json": json.dumps(result.input_metadata.get("categories", {}), ensure_ascii=False, sort_keys=True),
                "input_metadata_json": json.dumps(result.input_metadata.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                "output_metadata_json": json.dumps(result.output_metadata, ensure_ascii=False, sort_keys=True),
            }
            for column in metric_columns:
                row[column] = result.metrics.get(column, "")
            writer.writerow(row)


def main() -> None:
    source_dataset = HuggingFaceDatasetDriver(
        dataset_id=DATASET_ID,
        subset=DATASET_SUBSET,
        text_field="text",
        split="train",
        name="wikipedia",
        streaming=True,
    )
    pool_query = DatasetQuery(limit=POOL_SIZE, sample_strategy="head")

    print(f"Building a pool from the first {POOL_SIZE} Wikipedia articles...")
    pooled_articles = collect_pool_with_progress(source_dataset, pool_query)
    if not pooled_articles:
        raise ValueError("No Wikipedia records were returned.")
    if len(pooled_articles) < SAMPLE_SIZE:
        raise ValueError(f"Expected at least {SAMPLE_SIZE} pooled articles but found {len(pooled_articles)}.")

    sampled_articles = random.Random(SAMPLE_SEED).sample(pooled_articles, SAMPLE_SIZE)
    dataset = SampledDataset(name="sampled_wikipedia", records=sampled_articles)
    query = DatasetQuery(limit=SAMPLE_SIZE, sample_strategy="head")
    print(f"Selected {len(sampled_articles)} random articles from the pooled set.")

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
        experiment.add_test(TokenCountTest(model, dataset=dataset, query=query, label=f"{model.name} token count"))
        experiment.add_test(
            CompressionRatioTest(model, dataset=dataset, query=query, label=f"{model.name} compression")
        )
        experiment.add_test(OOVRateTest(model, dataset=dataset, query=query, label=f"{model.name} oov"))
        experiment.add_test(FertilityRateTest(model, dataset=dataset, query=query, label=f"{model.name} fertility"))
        experiment.add_test(
            MeanTokensPerSentenceTest(model, dataset=dataset, query=query, label=f"{model.name} sentence mean")
        )
        experiment.add_test(FragmentationTest(model, dataset=dataset, query=query, label=f"{model.name} fragmentation"))
        experiment.add_test(
            CostEstimateTest(
                model,
                dataset=dataset,
                query=query,
                cost_per_1k_tokens=cost_per_1k[model.name],
                label=f"{model.name} cost",
            )
        )
        experiment.add_test(
            NSLTest(
                model,
                baseline_model=baseline_model,
                dataset=dataset,
                query=query,
                label=f"{model.name} nsl",
            )
        )

    report = run_experiment_with_progress(experiment)
    markdown_summary = build_markdown_summary(
        sample_size=SAMPLE_SIZE,
        pool_size=POOL_SIZE,
        dataset_name=DATASET_ID,
        subset=DATASET_SUBSET,
        report_summary=report.summary,
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    markdown_path = RESULTS_DIR / f"{OUTPUT_STEM}_report.md"
    csv_path = RESULTS_DIR / f"{OUTPUT_STEM}_record_level_results.csv"
    markdown_path.write_text(markdown_summary, encoding="utf-8")
    write_results_csv(report, csv_path)

    print("Compared Hugging Face tokenizers on random Wikipedia articles.")
    print(f"Dataset: {DATASET_ID} / {DATASET_SUBSET}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print()
    print(f"Markdown summary written to: {markdown_path}")
    print(f"CSV results written to: {csv_path}")


if __name__ == "__main__":
    main()
