from tokenteller.core.types import TokenizationResult

from .base import BaseModelDriver


class HuggingFaceTokenizerDriver(BaseModelDriver):
    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        return_offset_mapping: bool = True,
    ):
        super().__init__(name=name or model_id)
        self.model_id = model_id
        self.return_offset_mapping = return_offset_mapping

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceTokenizerDriver requires the 'transformers' package."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def encode(self, text: str) -> TokenizationResult:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=self.return_offset_mapping,
        )
        token_ids = list(encoded["input_ids"])
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        offsets = encoded.get("offset_mapping") if self.return_offset_mapping else None

        return TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            token_count=len(token_ids),
            offsets=[tuple(offset) for offset in offsets] if offsets is not None else None,
            raw={},
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
