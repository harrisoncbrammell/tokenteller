from tokenteller.core.types import TokenizationResult

from .base import BaseModelDriver


class HuggingFaceTokenizerDriver(BaseModelDriver):
    _tokenizer_cache: dict[str, object] = {}

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        return_offset_mapping: bool = False,
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

        if model_id not in self._tokenizer_cache:
            self._tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer = self._tokenizer_cache[model_id]

    def encode(self, text: str) -> TokenizationResult:
        return self.tokenize(text, with_offsets=self.return_offset_mapping)

    def tokenize(self, text: str, *, with_offsets: bool = False) -> TokenizationResult:
        cache_key = (text, with_offsets)
        cached = self._tokenization_cache.get(cache_key)
        if cached is not None:
            return cached

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=with_offsets,
        )
        token_ids = list(encoded["input_ids"])
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        offsets = encoded.get("offset_mapping") if with_offsets else None

        tokenization = TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            token_count=len(token_ids),
            offsets=[tuple(offset) for offset in offsets] if offsets is not None else None,
            raw={},
        )
        self._tokenization_cache[cache_key] = tokenization
        return tokenization

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
