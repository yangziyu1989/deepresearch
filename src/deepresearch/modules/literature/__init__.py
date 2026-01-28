"""Literature search and novelty checking module."""

from deepresearch.modules.literature.novelty_checker import NoveltyChecker
from deepresearch.modules.literature.searcher import (
    ArxivClient,
    LiteratureSearcher,
    SemanticScholarClient,
)

__all__ = [
    "ArxivClient",
    "LiteratureSearcher",
    "NoveltyChecker",
    "SemanticScholarClient",
]
