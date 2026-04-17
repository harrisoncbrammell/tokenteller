# Re-export the built-in tests and the test template.
from .base import BaseTestDriver
from .compression_ratio import CompressionRatioTest
from .cost_estimate import CostEstimateTest
from .fertility_rate import FertilityRateTest
from .fragmentation import FragmentationTest
from .mean_tokens_per_sentence import MeanTokensPerSentenceTest
from .nsl import NSLTest
from .oov_rate import OOVRateTest
from .token_count import TokenCountTest

# Keep the testsuite namespace focused on reusable test objects.
__all__ = [
    "BaseTestDriver",
    "CompressionRatioTest",
    "CostEstimateTest",
    "FertilityRateTest",
    "FragmentationTest",
    "MeanTokensPerSentenceTest",
    "NSLTest",
    "OOVRateTest",
    "TokenCountTest",
]
