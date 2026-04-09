
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
        super().__init__(name=name)
        # TODO: load or store the tokenizer/model object here

    def encode(self, text: str) -> TokenizationResult:
        # TODO: tokenize the input string
        token_ids = []
        tokens = []
        offsets = []

        return TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            token_count=len(token_ids),
            offsets=offsets,
            raw={},
        )

    def decode(self, token_ids: list[int]) -> str:
        # TODO: convert token ids back into text
        return ""

    def info(self) -> dict[str, object]:
        # Optional: return extra information about the model/tokenizer
        return {"name": self.name}
