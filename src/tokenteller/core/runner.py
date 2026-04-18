from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == script_dir:
        sys.path.pop(0)
    src_root = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, src_root)
    from tokenteller.core.types import TestRunReport
    from tokenteller.testsuites.base import BaseTestDriver
else:
    from .types import TestRunReport
    from ..testsuites.base import BaseTestDriver


class Experiment:
    def __init__(self):
        self.tests: list[BaseTestDriver] = []

    def add_test(self, test: BaseTestDriver) -> "Experiment":
        self.tests.append(test)
        return self

    def run(self) -> TestRunReport:
        """Run all configured tests and collect their saved output."""
        if not self.tests:
            raise ValueError("Experiment.run() requires at least one test.")

        summary: list[dict[str, object]] = []
        results = []
        warnings: list[str] = []
        for test in self.tests:
            test.status = "not_run"
            test.results = []
            test.summary = []
            test.warnings = []

            test.run()
            test.status = "completed"

            summary.extend(test.summary or [self._default_summary_row(test)])
            results.extend(test.results)
            warnings.extend(test.warnings)
        return TestRunReport(summary=summary, results=results, warnings=warnings)

    def _default_summary_row(self, test: BaseTestDriver) -> dict[str, object]:
        return {
            "test": test.label,
            "type": test.name(),
            "model": test.model.name,
            "tokenizer": test.model.name,
            "status": test.status,
        }
