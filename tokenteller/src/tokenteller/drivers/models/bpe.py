from tokenteller.core.types import TokenizationResult

from .base import BaseModelDriver


class BPEModelDriver(BaseModelDriver):
    """Byte Pair Encoder model driver using the Hugging Face tokenizers library."""

    def __init__(self, tokenizer_path: str, name: str = "bpe"):
        # Save the short driver name used by the experiment.
        super().__init__(name=name)
        # Save the tokenizer path so users can see what file was loaded.
        self.tokenizer_path = tokenizer_path

        # Import tokenizers only when someone actually uses this driver.
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "BPEModelDriver requires the 'tokenizers' package."
            ) from exc

        # Load the existing .json tokenizer file.
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

    def encode(self, text: str) -> TokenizationResult:
        # Use the tokenizer to encode the text.
        encoding = self.tokenizer.encode(text)

        # Extract token ids and token strings from the encoding.
        token_ids = encoding.ids
        tokens = encoding.tokens

        # Extract byte offsets (character spans) from the encoding.
        # The tokenizers library provides (start, end) tuples for each token.
        offsets = encoding.offsets

        # Return the shared Tokenteller result object.
        return TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            token_count=len(token_ids),
            offsets=offsets,
            raw={"tokenizer_path": self.tokenizer_path},
        )

    def decode(self, token_ids: list[int]) -> str:
        # Convert token ids back into text.
        return self.tokenizer.decode(token_ids)
