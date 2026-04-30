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
from tokenteller.drivers.tests import (
    CompressionRatioTest,
    CostEstimateTest,
    FertilityRateTest,
    FragmentationTest,
    MeanTokensPerSentenceTest,
    NSLTest,
    OOVRateTest,
    TokenCountTest,
)


RESULTS_DIR = Path(__file__).with_name("results")
MODEL_SPECS = [
    ("gpt2", "gpt2"),
    ("openlm-research/open_llama_3b", "llama-sp"),
    ("t5-small", "t5"),
    ("bert-base-uncased", "wordpiece"),
]
COST_PER_1K = {
    "gpt2": 0.0020,
    "llama-sp": 0.0030,
    "t5": 0.0025,
    "wordpiece": 0.0015,
}
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
            records = records[:]
            random.Random(query.seed).shuffle(records)
        elif query.sample_strategy == "tail":
            records = records[::-1]

        if query.limit is None:
            return records
        return records[: query.limit]


def run_random_sample_experiment(
    *,
    dataset_id: str,
    subset: str,
    text_field: str,
    source_name: str,
    sample_name: str,
    sample_size: int,
    pool_size: int,
    sample_seed: int,
    output_stem: str,
    summary_title: str,
    record_description: str,
    metric_descriptions: dict[str, str],
    pool_message: str,
    selected_message: str,
    done_message: str,
) -> None:
    source_dataset = HuggingFaceDatasetDriver(
        dataset_id=dataset_id,
        subset=subset,
        text_field=text_field,
        split="train",
        name=source_name,
        streaming=True,
    )
    pool_query = DatasetQuery(limit=pool_size, sample_strategy="head")

    print(pool_message)
    pooled_records = list(source_dataset.iter_records(pool_query))
    if not pooled_records:
        raise ValueError(f"No {source_name} records were returned.")
    if len(pooled_records) < sample_size:
        raise ValueError(
            f"Expected at least {sample_size} pooled records but found {len(pooled_records)}."
        )

    sampled_records = random.Random(sample_seed).sample(pooled_records, sample_size)
    dataset = SampledDataset(name=sample_name, records=sampled_records)
    query = DatasetQuery(limit=sample_size, sample_strategy="head")
    print(selected_message.format(count=len(sampled_records)))

    models = [HuggingFaceTokenizerDriver(model_id, name=name) for model_id, name in MODEL_SPECS]
    baseline_model = models[0]

    experiment = Experiment()
    for model in models:
        experiment.add_test(TokenCountTest(model, dataset=dataset, query=query, label=f"{model.name} token count"))
        experiment.add_test(CompressionRatioTest(model, dataset=dataset, query=query, label=f"{model.name} compression"))
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
                cost_per_1k_tokens=COST_PER_1K[model.name],
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

    report = experiment.run()
    RESULTS_DIR.mkdir(exist_ok=True)
    markdown_path = RESULTS_DIR / f"{output_stem}_report.md"
    csv_path = RESULTS_DIR / f"{output_stem}_record_level_results.csv"

    markdown_path.write_text(
        build_markdown_summary(
            summary_title=summary_title,
            dataset_id=dataset_id,
            subset=subset,
            text_field=text_field,
            record_description=record_description,
            sample_size=sample_size,
            pool_size=pool_size,
            sample_seed=sample_seed,
            report=report,
            metric_descriptions=metric_descriptions,
        ),
        encoding="utf-8",
    )
    write_results_csv(report, csv_path)

    print(done_message)
    print(f"Dataset: {dataset_id} / {subset}")
    print(f"Text field: {text_field}")
    print(f"Sample size: {sample_size}")
    print()
    print(f"Markdown summary written to: {markdown_path}")
    print(f"CSV results written to: {csv_path}")


def build_markdown_summary(
    *,
    summary_title: str,
    dataset_id: str,
    subset: str,
    text_field: str,
    record_description: str,
    sample_size: int,
    pool_size: int,
    sample_seed: int,
    report: TestRunReport,
    metric_descriptions: dict[str, str],
) -> str:
    rows = summary_rows_by_tokenizer(report.summary)
    metric_columns = [column for column in rows[0] if column != "tokenizer"] if rows else []

    lines = [
        summary_title,
        "",
        "## Sample",
        "",
        f"- **Dataset**: `{dataset_id}`",
        f"- **Subset**: `{subset}`",
        f"- **Text field**: `{text_field}`",
        f"- **Record type**: {record_description}",
        "- **Sampling method**: bounded random sample.",
        f"- **Pool size**: first {pool_size} streamed records",
        f"- **Sample size**: {sample_size} random records",
        f"- **How records were chosen**: we first streamed the first {pool_size} rows from the dataset, then used a fixed random seed (`{sample_seed}`) to select {sample_size} records from that pooled set.",
        "",
        "## Test summary",
        "",
        markdown_table(rows),
        "",
        "## Metric guide",
        "",
    ]

    for column in metric_columns:
        description = metric_descriptions.get(column, "extra value from the test summary")
        lines.append(f"- **{display_label(column)}**: {description}")

    return "\n".join(lines)


def summary_rows_by_tokenizer(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_tokenizer: dict[str, dict[str, object]] = {}
    metric_columns: list[str] = []
    ignored_columns = {"test", "type", "model", "tokenizer", "status", "baseline", "cost_per_1k_tokens"}

    for summary_row in summary_rows:
        tokenizer = str(summary_row.get("tokenizer", summary_row.get("model", "unknown")))
        row = rows_by_tokenizer.setdefault(tokenizer, {"tokenizer": tokenizer})

        for key, value in summary_row.items():
            if key in ignored_columns:
                continue
            row[key] = value
            if key not in metric_columns:
                metric_columns.append(key)

    return [
        {"tokenizer": row["tokenizer"], **{column: row.get(column, "") for column in metric_columns}}
        for row in rows_by_tokenizer.values()
    ]


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


def display_label(column: str) -> str:
    return METRIC_LABELS.get(column, column.replace("_", " ").title())


def escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


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
                "input_categories_json": json.dumps(
                    result.input_metadata.get("categories", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "input_metadata_json": json.dumps(
                    result.input_metadata.get("metadata", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "output_metadata_json": json.dumps(
                    result.output_metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for column in metric_columns:
                row[column] = result.metrics.get(column, "")
            writer.writerow(row)
