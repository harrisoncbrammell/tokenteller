from __future__ import annotations

from collections.abc import Sequence

from .types import DatasetQuery, RunConfig, TestContext, TestRunReport
from ..drivers.datasets.base import BaseDatasetDriver
from ..drivers.models.base import BaseModelDriver
from ..testsuites.base import BaseTestDriver


class Experiment:
    """Small orchestration object that holds models, datasets, and tests."""

    def __init__(
        self,
        *,
        tokenizers: Sequence[BaseModelDriver] | None = None,
        dataset: BaseDatasetDriver | None = None,
        query: DatasetQuery | None = None,
        run_config: RunConfig | None = None,
    ):
        # Models and datasets are stored by short user-facing names.
        self.models: dict[str, BaseModelDriver] = {}
        self.datasets: dict[str, BaseDatasetDriver] = {}
        # The default query is reused when a test does not provide its own.
        self.default_query = query or DatasetQuery()
        self.run_config = run_config or RunConfig()
        self.tests: list[BaseTestDriver] = []

        # These convenience constructor arguments are optional.
        for tokenizer in tokenizers or []:
            self.add_model(tokenizer)
        if dataset is not None:
            self.add_dataset(dataset)

    def add_model(self, model: BaseModelDriver, *, name: str | None = None) -> "Experiment":
        """Add a fully constructed model driver object."""
        self.models[name or model.name] = model
        return self

    def add_dataset(self, dataset: BaseDatasetDriver, *, name: str | None = None) -> "Experiment":
        """Add a fully constructed dataset driver object."""
        self.datasets[name or dataset.name] = dataset
        return self

    def add_test(
        self,
        test: BaseTestDriver,
        *,
        model: str,
        dataset: str,
        query: DatasetQuery | None = None,
    ) -> "Experiment":
        """Bind one test object to one model and one dataset."""
        if model not in self.models:
            raise KeyError(f"Unknown model '{model}'. Add it before adding tests.")
        if dataset not in self.datasets:
            raise KeyError(f"Unknown dataset '{dataset}'. Add it before adding tests.")

        test.bind(model_name=model, dataset_name=dataset, query=query or self.default_query)
        self.tests.append(test)
        return self

    def add_tests(
        self,
        tests: Sequence[BaseTestDriver],
        *,
        model: str,
        dataset: str,
        query: DatasetQuery | None = None,
    ) -> "Experiment":
        """Bind many test objects to the same model and dataset."""
        for test in tests:
            self.add_test(test, model=model, dataset=dataset, query=query)
        return self

    def run(self) -> TestRunReport:
        """Run all configured tests and collect their saved results."""
        if not self.tests:
            raise ValueError("Experiment.run() requires at least one test.")

        # Reuse the same dataset slice when many tests point at the same data.
        record_cache: dict[tuple[str, tuple[tuple[str, str], ...], int | None, str, int | None], list[object]] = {}
        summary: list[dict[str, object]] = []
        results = []
        warnings: list[str] = []
        for test in self.tests:
            # Resolve the bound model, dataset, and query for this test object.
            model = self.models[test.model_name]
            dataset = self.datasets[test.dataset_name]
            query = test.query or self.default_query
            cache_key = self._query_key(dataset.name, query)
            if cache_key not in record_cache:
                record_cache[cache_key] = list(dataset.iter_records(query))
            records = record_cache[cache_key]

            # Each test gets its own context, but the test object stores the final result.
            context = TestContext(
                run_config=self.run_config,
                baseline_tokenizer_name=self.run_config.baseline_tokenizer or model.name,
            )

            # NSL-style tests need one baseline token count per record.
            baseline_model = self.models.get(context.baseline_tokenizer_name, model)
            if context.baseline_tokenizer_name not in self.models and self.run_config.baseline_tokenizer:
                context.add_warning(
                    f"Baseline model '{self.run_config.baseline_tokenizer}' was not added, so '{model.name}' was used."
                )
            context.baseline_tokenizer_name = baseline_model.name
            for record in records:
                context.baseline_token_counts[record.id] = context.get_tokenization(
                    baseline_model, record
                ).token_count

            try:
                # The base test class handles the per-record parallel work.
                test_results = test.run_batch(model, records, context)
                if not records:
                    context.add_warning("No dataset records matched the provided query.")
                test.store_results(results=test_results, warnings=context.warnings)
            except Exception as exc:
                # Store failures on the test object so they still show up in summaries.
                test.store_error(str(exc))

            # Collect the saved state from the test object into the experiment report.
            summary.extend(test.summary_rows())
            results.extend(test.results)
            warnings.extend(test.warnings)
            if test.error:
                warnings.append(f"{test.label} failed: {test.error}")

        return TestRunReport(summary=summary, results=results, warnings=self._dedupe(warnings))

    def _query_key(
        self,
        dataset_name: str,
        query: DatasetQuery,
    ) -> tuple[str, tuple[tuple[str, str], ...], int | None, str, int | None]:
        """Build a stable cache key for one dataset/query combination."""
        return (
            dataset_name,
            tuple(sorted((str(key), repr(value)) for key, value in query.filters.items())),
            query.limit,
            query.sample_strategy,
            query.seed,
        )

    def _dedupe(self, values: Sequence[str]) -> list[str]:
        """Keep warnings in insertion order while removing duplicates."""
        unique: list[str] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique


# Keep the old name around so earlier code still works.
ExperimentRunner = Experiment
