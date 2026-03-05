"""Smoke tests for indexing and retrieval behaviors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.desktop_search.indexer import InvertedIndex
from src.desktop_search.query import SearchEngine


class SearchEngineSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "a.txt").write_text(
            "Punjab University has an NLP department and a search engine project.",
            encoding="utf-8",
        )
        (root / "b.txt").write_text(
            "Pakistan won a cricket match. The cricket analysis used statistics.",
            encoding="utf-8",
        )
        (root / "c.txt").write_text(
            "Desktop retrieval with tf idf and cosine similarity.", encoding="utf-8"
        )

        self.index = InvertedIndex(remove_stopwords=False, use_stemming=False)
        self.index.build(root, extensions=[".txt"], progress_every=0)
        self.engine = SearchEngine(self.index)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_keyword_query(self) -> None:
        results = self.engine.search("cosine similarity", top_k=3)
        self.assertTrue(results)
        self.assertTrue(results[0].path.endswith("c.txt"))

    def test_phrase_query(self) -> None:
        results = self.engine.search('"Punjab University"', top_k=3)
        self.assertTrue(results)
        self.assertTrue(results[0].path.endswith("a.txt"))

    def test_proximity_query(self) -> None:
        results = self.engine.search("pakistan cricket ^5", top_k=3)
        self.assertTrue(results)
        self.assertTrue(results[0].path.endswith("b.txt"))


if __name__ == "__main__":
    unittest.main()
