"""Desktop Search Engine package."""

from .indexer import InvertedIndex
from .query import SearchEngine

__all__ = ["InvertedIndex", "SearchEngine"]
