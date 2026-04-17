"""Dataset driver folder.

Add one Python file per dataset driver here.
"""

# Re-export the dataset base class and starter template.
from .base import BaseDatasetDriver
from .common_crawl import CommonCrawlDatasetDriver
from .huggingface import HuggingFaceDatasetDriver
from .template import DatasetDriverTemplate

# Keep dataset-driver imports short for teammates.
__all__ = [
    "BaseDatasetDriver",
    "CommonCrawlDatasetDriver",
    "DatasetDriverTemplate",
    "HuggingFaceDatasetDriver",
]
