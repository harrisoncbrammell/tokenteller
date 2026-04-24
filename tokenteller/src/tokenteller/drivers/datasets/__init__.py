"""Dataset driver folder.

Add one Python file per dataset driver here.
"""

# Re-export the dataset base class and starter template.
from .base import BaseDatasetDriver
from .common_crawl import CommonCrawlDatasetDriver
from .template import DatasetDriverTemplate
from .wikipedia import WikipediaCorpusDatasetDriver

# Keep dataset-driver imports short for teammates.
__all__ = [
    "BaseDatasetDriver",
    "CommonCrawlDatasetDriver",
    "DatasetDriverTemplate",
    "WikipediaCorpusDatasetDriver",
]
