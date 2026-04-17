from tokenteller.core.types import TokenizationResult

from .base import BaseModelDriver


class SentencePieceModelDriver(BaseModelDriver):

    def __init__(self, model_path: str, name: str = "sentencepiece"):
        super().__init__(name=name)
        self.model_path = model_path

        # import SentencePiece only when someone actually uses this driver
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError("SentencePieceModelDriver requires the 'sentencepiece' package.") from exc

        # load existing .model file into one sentencepiece processor object
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    def encode(self, text: str) -> TokenizationResult:
        # Ask SentencePiece for the token ids.
        token_ids = self.sp.encode(text, out_type=int)
        # Ask SentencePiece for readable token strings too.
        tokens = self.sp.encode(text, out_type=str)

        # Return the shared Tokenteller result object.
        return TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            token_count=len(token_ids),
            offsets=None,
            raw={"model_path": self.model_path},
        )

    def decode(self, token_ids: list[int]) -> str:
        # Convert token ids back into text.
        return self.sp.decode(token_ids)
