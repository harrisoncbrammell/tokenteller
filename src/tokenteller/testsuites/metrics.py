from __future__ import annotations

from ..core.types import DatasetRecord, TestCaseResult, TestContext
from ..drivers.models.base import BaseModelDriver
from .base import BaseTestDriver


class TokenCountTest(BaseTestDriver):
    """Count how many tokens a tokenizer creates for a piece of text."""

    def name(self) -> str:
        # Return the stable test name used in summaries.
        return "token_count"

    def run_case(
        self,
        tokenizer: BaseModelDriver,
        record: DatasetRecord,
        context: TestContext,
    ) -> TestCaseResult:
        # This is the simplest metric in the project.
        tokenization = context.get_tokenization(tokenizer, record)
        return TestCaseResult(
            record_id=record.id,
            tokenizer_name=tokenizer.name,
            test_name=self.name(),
            metrics={"token_count": tokenization.token_count},
            artifacts={"tokens": tokenization.tokens, "token_ids": tokenization.token_ids},
        )


class FragmentationTest(BaseTestDriver):
    """Measure how much a tokenizer splits words into smaller pieces."""

    def name(self) -> str:
        # Return the stable test name used in summaries.
        return "fragmentation"

    def run_case(
        self,
        tokenizer: BaseModelDriver,
        record: DatasetRecord,
        context: TestContext,
    ) -> TestCaseResult:
        # Reuse the helper from the tokenizer base class so this stays short.
        stats = tokenizer.fragmentation_stats(record.text)
        metrics = {
            "token_count": stats["token_count"],
            "word_count": stats["word_count"],
            "pieces_per_word": stats["pieces_per_word"],
            "max_pieces_per_word": stats["max_pieces_per_word"],
        }
        return TestCaseResult(
            record_id=record.id,
            tokenizer_name=tokenizer.name,
            test_name=self.name(),
            metrics=metrics,
            artifacts={"word_fragments": stats["word_fragments"]},
        )


class NSLTest(BaseTestDriver):
    """Compare token count to the baseline tokenizer for the same record."""

    def name(self) -> str:
        # Return the stable test name used in summaries.
        return "nsl"

    def run_case(
        self,
        tokenizer: BaseModelDriver,
        record: DatasetRecord,
        context: TestContext,
    ) -> TestCaseResult:
        # NSL asks: how long is this tokenized sequence relative to the baseline?
        tokenization = context.get_tokenization(tokenizer, record)
        baseline_name = context.run_config.baseline_tokenizer or tokenizer.name
        baseline_model = context.models.get(baseline_name)
        if baseline_model is None:
            raise KeyError(f"Baseline model '{baseline_name}' was not added.")

        baseline_count = context.get_tokenization(baseline_model, record).token_count
        if baseline_count == 0:
            context.warnings.append(f"Baseline token count for record '{record.id}' is zero.")
            nsl = None
        else:
            # Compare this token count to the baseline token count.
            nsl = tokenization.token_count / baseline_count
        return TestCaseResult(
            record_id=record.id,
            tokenizer_name=tokenizer.name,
            test_name=self.name(),
            metrics={
                "token_count": tokenization.token_count,
                "baseline_token_count": baseline_count,
                "nsl": nsl,
            },
            artifacts={},
        )


class CostEstimateTest(BaseTestDriver):
    """Very simple estimated cost based on a user-supplied cost per 1k tokens."""

    def name(self) -> str:
        # Return the stable test name used in summaries.
        return "cost"

    def run_case(
        self,
        tokenizer: BaseModelDriver,
        record: DatasetRecord,
        context: TestContext,
    ) -> TestCaseResult:
        # This is intentionally simple and meant for comparison, not billing.
        tokenization = context.get_tokenization(tokenizer, record)
        # Look up the configured cost rate for this model.
        cost_per_1k = context.run_config.cost_per_1k_tokens.get(tokenizer.name, 0.0)
        # Turn a token count into a rough estimated cost.
        estimated_cost = tokenization.token_count / 1000.0 * cost_per_1k
        return TestCaseResult(
            record_id=record.id,
            tokenizer_name=tokenizer.name,
            test_name=self.name(),
            metrics={
                "token_count": tokenization.token_count,
                "cost_per_1k_tokens": cost_per_1k,
                "estimated_cost": estimated_cost,
            },
            artifacts={},
        )
