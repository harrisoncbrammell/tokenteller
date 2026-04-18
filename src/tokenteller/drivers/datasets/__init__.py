"""Dataset driver folder.

Add one Python file per dataset driver here.
"""

# Re-export the dataset base class and concrete drivers.
from .base import BaseDatasetDriver
from .common_crawl import CommonCrawlDatasetDriver
from .huggingface import HuggingFaceDatasetDriver

# Keep dataset-driver imports short for teammates.
__all__ = [
    "BaseDatasetDriver",
    "CommonCrawlDatasetDriver",
    "HuggingFaceDatasetDriver",
]
