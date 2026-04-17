from __future__ import annotations

from .types import RunConfig, TestContext, TestRunReport
from ..drivers.models.base import BaseModelDriver
from ..testsuites.base import BaseTestDriver


class Experiment:
    """Small orchestration object that holds models, datasets, and tests."""

    def __init__(
        self,
        *,
        run_config: RunConfig | None = None,
    ):
        self.models: dict[str, BaseModelDriver] = {}
        self.run_config = run_config or RunConfig()
        self.tests: list[BaseTestDriver] = []

    def add_model(self, model: BaseModelDriver, *, name: str | None = None) -> "Experiment":
        """Add a fully constructed model driver object."""
        self.models[name or model.name] = model
        return self

    def add_test(
        self,
        test: BaseTestDriver,
        *,
        model: str,
    ) -> "Experiment":
        """Attach one test object to one model."""
        if model not in self.models:
            raise KeyError(f"Unknown model '{model}'. Add it before adding tests.")

        test.model_name = model
        test.status = "not_run"
        test.results = []
        test.summary = []
        test.warnings = []
        self.tests.append(test)
        return self

    def run(self) -> TestRunReport:
        """Run all configured tests and collect their saved results."""
        if not self.tests:
            raise ValueError("Experiment.run() requires at least one test.")

        summary: list[dict[str, object]] = []
        results = []
        warnings: list[str] = []
        for test in self.tests:
            model = self.models[test.model_name]
            context = TestContext(
                run_config=self.run_config,
                models=self.models,
            )

            test_results = test.run(model, context)
            test.store_results(results=test_results, warnings=context.warnings)
            summary.extend(test.summary_rows())
            results.extend(test.results)
            warnings.extend(test.warnings)
        return TestRunReport(summary=summary, results=results, warnings=warnings)
