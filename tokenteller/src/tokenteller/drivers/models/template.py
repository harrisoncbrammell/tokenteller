
from tokenteller.core.types import TokenizationResult
from tokenteller.drivers.models.base import BaseModelDriver


class ModelDriverTemplate(BaseModelDriver):
    """
    Copy this file when adding a real model driver.

    Suggested naming:
    - ``gpt2_driver.py``
    - ``bert_driver.py``
    - ``llama_driver.py``
    """

    def __init__(self, name: str = "my_model"):
        # Save the short model name used in experiment summaries.
        super().__init__(name=name)
        # TODO: load the real tokenizer object here

    def encode(self, text: str) -> TokenizationResult:
        # TODO: turn the input string into token ids and token strings
        token_ids = []
        tokens = []

        return TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            token_count=len(token_ids),
            offsets=None,
            raw={},
        )

    def decode(self, token_ids: list[int]) -> str:
        # TODO: convert token ids back into text if your library supports it
        return ""
