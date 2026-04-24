from tokenteller.core.types import DatasetRecord, TestCaseResult, TestContext
from tokenteller.drivers.models.base import BaseModelDriver

from .base import BaseTestDriver


class TestDriverTemplate(BaseTestDriver):
    """
    Copy this file when adding a real test driver.

    Suggested naming:
    - ``token_count_test.py``
    - ``language_split_test.py``
    - ``compression_test.py``
    """

    def __init__(self, label: str | None = None):
        super().__init__(label=label)
        # TODO: store any extra setup for this test here

    def name(self) -> str:
        # Return a short stable name for this test type.
        return "my_test"

    def run_case(
        self,
        tokenizer: BaseModelDriver,
        record: DatasetRecord,
        context: TestContext,
    ) -> TestCaseResult:
        # TODO: compute a metric using tokenizer and record
        tokenization = context.get_tokenization(tokenizer, record)

        return TestCaseResult(
            record_id=record.id,
            tokenizer_name=tokenizer.name,
            test_name=self.name(),
            metrics={"token_count": tokenization.token_count},
            artifacts={},
        )

    def compare(self, other: "TestDriverTemplate") -> str:
        # Optional: override this if a custom comparison view makes sense.
        return super().compare(other)
