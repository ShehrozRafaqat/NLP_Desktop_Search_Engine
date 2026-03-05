"""Requirement-focused tests for the NLP desktop search engine assignment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.desktop_search.indexer import InvertedIndex
from src.desktop_search.query import SearchEngine


class AssignmentRequirementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        nested = self.root / "nested"
        nested.mkdir(parents=True, exist_ok=True)

        (self.root / "d1.txt").write_text(
            "Punjab University offers NLP research. Punjab University is in Lahore.",
            encoding="utf-8",
        )
        (self.root / "d2.txt").write_text(
            "Pakistan cricket team won. Pakistan and cricket fans celebrated.",
            encoding="utf-8",
        )
        (nested / "d3.txt").write_text(
            "Search engine retrieval uses tf idf and cosine similarity.",
            encoding="utf-8",
        )
        (nested / "ignore.md").write_text("not indexed", encoding="utf-8")

        self.index = InvertedIndex(remove_stopwords=False, use_stemming=False)
        self.stats = self.index.build(self.root, extensions=[".txt"], progress_every=0)
        self.engine = SearchEngine(self.index)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_recursive_discovery_and_txt_filter(self) -> None:
        self.assertEqual(self.stats.total_documents, 3)
        self.assertEqual(self.stats.indexed_files, 3)
        self.assertEqual(len(self.index.doc_id_to_path), 3)
        for path in self.index.doc_id_to_path:
            self.assertTrue(path.endswith(".txt"))

    def test_inverted_index_contains_tf_df_and_norms(self) -> None:
        self.assertIn("punjab", self.index.postings)
        self.assertIn("cricket", self.index.postings)
        self.assertGreaterEqual(self.index.df["punjab"], 1)
        self.assertGreaterEqual(self.index.df["cricket"], 1)

        for mode in ("boolean", "count", "tfidf"):
            self.assertEqual(len(self.index.doc_norms[mode]), self.index.total_docs)
            self.assertTrue(all(value >= 0 for value in self.index.doc_norms[mode]))

    def test_persistence_save_and_reload(self) -> None:
        index_dir = self.root / "idx"
        self.index.save(index_dir)
        reloaded = InvertedIndex.load(index_dir)

        self.assertEqual(reloaded.total_docs, self.index.total_docs)
        self.assertEqual(reloaded.vocabulary_size, self.index.vocabulary_size)
        self.assertEqual(reloaded.df, self.index.df)
        self.assertEqual(reloaded.doc_id_to_path, self.index.doc_id_to_path)

        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["total_documents"], 3)
        self.assertIn("index_file", manifest)

    def test_weighting_and_similarity_matrix(self) -> None:
        query = "search engine"
        for weighting in ("boolean", "count", "tfidf"):
            for similarity in ("dot", "cosine"):
                results = self.engine.search(
                    query,
                    top_k=10,
                    weighting=weighting,
                    similarity=similarity,
                )
                self.assertTrue(results)
                self.assertLessEqual(len(results), 10)
                # Document containing both terms should appear in ranking.
                self.assertTrue(any(r.path.endswith("d3.txt") for r in results))

    def test_phrase_query(self) -> None:
        results = self.engine.search('"Punjab University"', top_k=10)
        self.assertTrue(results)
        self.assertTrue(results[0].path.endswith("d1.txt"))

    def test_proximity_query(self) -> None:
        results = self.engine.search("pakistan cricket ^5", top_k=10)
        self.assertTrue(results)
        self.assertTrue(results[0].path.endswith("d2.txt"))

    def test_top_k_behavior(self) -> None:
        results = self.engine.search("pakistan cricket", top_k=1)
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
