"""Inverted index construction and persistence."""

from __future__ import annotations

import gzip
import json
import math
import os
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from .preprocess import TextPreprocessor

WEIGHTING_MODES = {"boolean", "count", "tfidf"}
SIMILARITY_MODES = {"dot", "cosine"}


@dataclass
class IndexStats:
    total_documents: int
    total_terms: int
    total_tokens: int
    indexed_files: int
    elapsed_seconds: float


class InvertedIndex:
    """Stores postings, statistics, and vector norms for retrieval."""

    def __init__(
        self,
        remove_stopwords: bool = False,
        use_stemming: bool = False,
    ) -> None:
        self.preprocessor = TextPreprocessor(
            remove_stopwords=remove_stopwords,
            use_stemming=use_stemming,
        )
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming

        self.doc_id_to_path: List[str] = []
        self.doc_lengths: List[int] = []

        # term -> {doc_id: tf}
        self.postings: Dict[str, Dict[int, int]] = {}
        # term -> {doc_id: [positions]}
        self.positions: Dict[str, Dict[int, List[int]]] = {}

        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

        self.doc_norms: Dict[str, List[float]] = {
            "boolean": [],
            "count": [],
            "tfidf": [],
        }

        self.total_tokens: int = 0
        self.built_at_unix: float = 0.0

    @property
    def total_docs(self) -> int:
        return len(self.doc_id_to_path)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @staticmethod
    def discover_files(root_dir: str | Path, extensions: Iterable[str]) -> List[str]:
        normalized_ext: Set[str] = {ext.lower() for ext in extensions}
        root = Path(root_dir)
        discovered: List[str] = []

        for current_root, _, files in os.walk(root):
            for name in files:
                suffix = Path(name).suffix.lower()
                if suffix in normalized_ext:
                    discovered.append(str(Path(current_root) / name))

        discovered.sort()
        return discovered

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as file_obj:
            return file_obj.read()

    def build(
        self,
        data_dir: str | Path,
        extensions: Iterable[str] = (".txt",),
        progress_every: int = 2000,
    ) -> IndexStats:
        started = time.time()
        files = self.discover_files(data_dir, extensions)
        indexed_files = 0

        postings_builder: DefaultDict[str, Dict[int, int]] = defaultdict(dict)
        positions_builder: DefaultDict[str, Dict[int, List[int]]] = defaultdict(dict)

        boolean_norm_squares: List[float] = []
        count_norm_squares: List[float] = []

        for idx, path in enumerate(files, start=1):
            text = self._read_text(path)
            tokens = self.preprocessor.process(text)

            doc_id = len(self.doc_id_to_path)
            self.doc_id_to_path.append(path)
            self.doc_lengths.append(len(tokens))
            self.total_tokens += len(tokens)

            if not tokens:
                boolean_norm_squares.append(0.0)
                count_norm_squares.append(0.0)
                continue

            tf_counter = Counter(tokens)
            positions_by_term: DefaultDict[str, List[int]] = defaultdict(list)
            for position, token in enumerate(tokens):
                positions_by_term[token].append(position)

            for term, tf in tf_counter.items():
                postings_builder[term][doc_id] = tf
                positions_builder[term][doc_id] = positions_by_term[term]

            boolean_norm_squares.append(float(len(tf_counter)))
            count_norm_squares.append(float(sum(tf * tf for tf in tf_counter.values())))
            indexed_files += 1

            if progress_every > 0 and idx % progress_every == 0:
                print(f"Indexed {idx}/{len(files)} files...")

        self.postings = dict(postings_builder)
        self.positions = dict(positions_builder)

        total_docs = self.total_docs
        for term, term_postings in self.postings.items():
            df = len(term_postings)
            self.df[term] = df
            self.idf[term] = math.log((total_docs + 1.0) / (df + 1.0)) + 1.0

        tfidf_norm_squares = [0.0 for _ in range(total_docs)]
        for term, term_postings in self.postings.items():
            term_idf = self.idf[term]
            for doc_id, tf in term_postings.items():
                weight = tf * term_idf
                tfidf_norm_squares[doc_id] += weight * weight

        self.doc_norms = {
            "boolean": [math.sqrt(value) for value in boolean_norm_squares],
            "count": [math.sqrt(value) for value in count_norm_squares],
            "tfidf": [math.sqrt(value) for value in tfidf_norm_squares],
        }

        elapsed = time.time() - started
        self.built_at_unix = time.time()

        return IndexStats(
            total_documents=self.total_docs,
            total_terms=self.vocabulary_size,
            total_tokens=self.total_tokens,
            indexed_files=indexed_files,
            elapsed_seconds=elapsed,
        )

    def save(self, index_dir: str | Path) -> None:
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "remove_stopwords": self.remove_stopwords,
            "use_stemming": self.use_stemming,
            "doc_id_to_path": self.doc_id_to_path,
            "doc_lengths": self.doc_lengths,
            "postings": self.postings,
            "positions": self.positions,
            "df": self.df,
            "idf": self.idf,
            "doc_norms": self.doc_norms,
            "total_tokens": self.total_tokens,
            "built_at_unix": self.built_at_unix,
        }

        with gzip.open(index_path / "index.pkl.gz", "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        manifest = {
            "total_documents": self.total_docs,
            "vocabulary_size": self.vocabulary_size,
            "total_tokens": self.total_tokens,
            "remove_stopwords": self.remove_stopwords,
            "use_stemming": self.use_stemming,
            "built_at_unix": self.built_at_unix,
            "index_file": "index.pkl.gz",
        }
        with open(index_path / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    @classmethod
    def load(cls, index_dir: str | Path) -> "InvertedIndex":
        index_path = Path(index_dir)
        with gzip.open(index_path / "index.pkl.gz", "rb") as handle:
            payload = pickle.load(handle)

        instance = cls(
            remove_stopwords=bool(payload.get("remove_stopwords", False)),
            use_stemming=bool(payload.get("use_stemming", False)),
        )

        instance.doc_id_to_path = payload["doc_id_to_path"]
        instance.doc_lengths = payload["doc_lengths"]
        instance.postings = payload["postings"]
        instance.positions = payload["positions"]
        instance.df = payload["df"]
        instance.idf = payload["idf"]
        instance.doc_norms = payload["doc_norms"]
        instance.total_tokens = payload.get("total_tokens", 0)
        instance.built_at_unix = payload.get("built_at_unix", 0.0)

        return instance

    def weight_document_term(self, term: str, doc_tf: int, weighting: str) -> float:
        if weighting == "boolean":
            return 1.0 if doc_tf > 0 else 0.0
        if weighting == "count":
            return float(doc_tf)
        if weighting == "tfidf":
            return float(doc_tf) * self.idf.get(term, self._idf_for_unseen_term())
        raise ValueError(f"Unsupported weighting mode: {weighting}")

    def weight_query_terms(self, terms: List[str], weighting: str) -> Tuple[Dict[str, float], float]:
        if not terms:
            return {}, 0.0

        term_counts = Counter(terms)
        weights: Dict[str, float] = {}

        for term, tf in term_counts.items():
            if weighting == "boolean":
                weights[term] = 1.0
            elif weighting == "count":
                weights[term] = float(tf)
            elif weighting == "tfidf":
                weights[term] = float(tf) * self.idf.get(term, self._idf_for_unseen_term())
            else:
                raise ValueError(f"Unsupported weighting mode: {weighting}")

        norm = math.sqrt(sum(value * value for value in weights.values()))
        return weights, norm

    def _idf_for_unseen_term(self) -> float:
        # df = 0 case from assignment formula.
        return math.log((self.total_docs + 1.0) / 1.0) + 1.0

    def validate_modes(self, weighting: str, similarity: str) -> None:
        if weighting not in WEIGHTING_MODES:
            raise ValueError(f"weighting must be one of {sorted(WEIGHTING_MODES)}")
        if similarity not in SIMILARITY_MODES:
            raise ValueError(f"similarity must be one of {sorted(SIMILARITY_MODES)}")

