"""Tests for the Byte Pair Encoder (BPE) model driver."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


class MockEncoding:
    """Mock the Encoding object returned by tokenizers.Tokenizer.encode()."""

    def __init__(self, ids: list[int], tokens: list[str], offsets: list[tuple[int, int]]):
        self.ids = ids
        self.tokens = tokens
        self.offsets = offsets


class MockTokenizer:
    """Mock the Tokenizer class from the tokenizers library."""

    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path

    @staticmethod
    def from_file(path: str) -> MockTokenizer:
        """Create a mock tokenizer from a file path."""
        return MockTokenizer(path)

    def encode(self, text: str) -> MockEncoding:
        """Simple mock encoding: split on whitespace."""
        tokens = text.split()
        ids = list(range(len(tokens)))
        offsets = []
        pos = 0
        for token in tokens:
            start = text.find(token, pos)
            end = start + len(token)
            offsets.append((start, end))
            pos = end
        return MockEncoding(ids, tokens, offsets)

    def decode(self, token_ids: list[int]) -> str:
        """Simple mock decoding: join tokens with spaces."""
        # For this mock, we don't have the actual token list, so we just return a placeholder
        return f"decoded_{len(token_ids)}_tokens"


# Mock the tokenizers module before importing BPEModelDriver
mock_tokenizers_module = MagicMock()
mock_tokenizers_module.Tokenizer = MockTokenizer
sys.modules["tokenizers"] = mock_tokenizers_module

from tokenteller.drivers.models.bpe import BPEModelDriver


def test_bpe_model_driver_encode():
    """Test that BPEModelDriver.encode() returns a valid TokenizationResult."""
    model = BPEModelDriver(tokenizer_path="mock.json", name="test-bpe")

    result = model.encode("hello world")

    assert result.token_ids == [0, 1]
    assert result.tokens == ["hello", "world"]
    assert result.token_count == 2
    assert result.offsets is not None
    assert len(result.offsets) == 2
    assert result.raw["tokenizer_path"] == "mock.json"


def test_bpe_model_driver_decode():
    """Test that BPEModelDriver.decode() works correctly."""
    model = BPEModelDriver(tokenizer_path="mock.json", name="test-bpe")

    decoded = model.decode([0, 1])
    assert decoded == "decoded_2_tokens"


def test_bpe_model_driver_offsets_populated():
    """Test that BPEModelDriver provides byte offset information."""
    model = BPEModelDriver(tokenizer_path="mock.json", name="test-bpe")

    result = model.encode("hello world")

    # Offsets should be populated (unlike SentencePiece which returns None)
    assert result.offsets is not None
    # Each token should have a start and end offset
    for offset in result.offsets:
        assert isinstance(offset, tuple)
        assert len(offset) == 2


def test_bpe_model_driver_fragmentation_stats():
    """Test that fragmentation stats work with populated offsets."""
    model = BPEModelDriver(tokenizer_path="mock.json", name="test-bpe")

    # fragmentation_stats is inherited from BaseModelDriver
    stats = model.fragmentation_stats("hello world test")

    assert "word_count" in stats
    assert "token_count" in stats
    assert "pieces_per_word" in stats
    assert stats["word_count"] == 3
    assert stats["token_count"] == 3


def test_bpe_model_driver_batch_encode():
    """Test that BPEModelDriver.batch_encode() works correctly."""
    model = BPEModelDriver(tokenizer_path="mock.json", name="test-bpe")

    results = model.batch_encode(["hello world", "foo bar baz"])

    assert len(results) == 2
    assert results[0].token_count == 2
    assert results[1].token_count == 3


def test_bpe_model_driver_custom_name():
    """Test that BPEModelDriver accepts a custom name."""
    model = BPEModelDriver(tokenizer_path="mock.json", name="custom-bpe-name")
    assert model.name == "custom-bpe-name"
