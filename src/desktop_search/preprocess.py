"""Text preprocessing utilities for indexing and query processing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class TextPreprocessor:
    """Tokenizes and normalizes text with optional stopword removal/stemming."""

    remove_stopwords: bool = False
    use_stemming: bool = False

    def process(self, text: str) -> List[str]:
        tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in DEFAULT_STOPWORDS]
        if self.use_stemming:
            tokens = [self._simple_stem(token) for token in tokens]
        return [token for token in tokens if token]

    def process_terms(self, terms: Iterable[str]) -> List[str]:
        normalized = []
        for term in terms:
            lowered = term.lower()
            parts = TOKEN_PATTERN.findall(lowered)
            for token in parts:
                if self.remove_stopwords and token in DEFAULT_STOPWORDS:
                    continue
                normalized.append(self._simple_stem(token) if self.use_stemming else token)
        return normalized

    @staticmethod
    def _simple_stem(token: str) -> str:
        """Small fallback stemmer to avoid external dependencies."""

        suffixes = ["ingly", "edly", "ing", "ed", "ly", "ies", "s"]
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                if suffix == "ies":
                    return token[: -len(suffix)] + "y"
                return token[: -len(suffix)]
        return token
