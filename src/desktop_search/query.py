"""Query engine supporting keyword, phrase, and proximity search."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .indexer import InvertedIndex

PROXIMITY_PATTERN = re.compile(r"^\s*([A-Za-z0-9]+)\s+([A-Za-z0-9]+)\s*\^(\d+)\s*$")
PHRASE_PATTERN = re.compile(r'"([^"]+)"')


@dataclass
class SearchResult:
    rank: int
    doc_id: int
    path: str
    score: float


@dataclass
class ParsedQuery:
    raw: str
    terms: List[str]
    phrases: List[List[str]]
    proximity: Optional[Tuple[str, str, int]]


class SearchEngine:
    """High-level retrieval and ranking engine."""

    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    @classmethod
    def from_disk(cls, index_dir: str) -> "SearchEngine":
        return cls(InvertedIndex.load(index_dir))

    def parse_query(self, raw_query: str) -> ParsedQuery:
        proximity_match = PROXIMITY_PATTERN.match(raw_query)
        if proximity_match:
            t1, t2, k_value = proximity_match.groups()
            terms = self.index.preprocessor.process_terms([t1, t2])
            if len(terms) == 2:
                return ParsedQuery(
                    raw=raw_query,
                    terms=terms,
                    phrases=[],
                    proximity=(terms[0], terms[1], int(k_value)),
                )

        phrases: List[List[str]] = []
        for phrase_content in PHRASE_PATTERN.findall(raw_query):
            phrase_terms = self.index.preprocessor.process_terms([phrase_content])
            if phrase_terms:
                phrases.append(phrase_terms)

        query_without_phrases = PHRASE_PATTERN.sub(" ", raw_query)
        terms = self.index.preprocessor.process_terms([query_without_phrases])

        return ParsedQuery(
            raw=raw_query,
            terms=terms,
            phrases=phrases,
            proximity=None,
        )

    def search(
        self,
        raw_query: str,
        top_k: int = 10,
        weighting: str = "tfidf",
        similarity: str = "cosine",
    ) -> List[SearchResult]:
        self.index.validate_modes(weighting, similarity)

        parsed = self.parse_query(raw_query)
        if parsed.proximity:
            t1, t2, max_distance = parsed.proximity
            return self._search_proximity(t1, t2, max_distance, top_k, weighting, similarity)

        candidate_docs: Optional[Set[int]] = None
        for phrase_terms in parsed.phrases:
            phrase_docs = self._phrase_docs(phrase_terms)
            candidate_docs = phrase_docs if candidate_docs is None else candidate_docs & phrase_docs
            if not candidate_docs:
                return []

        merged_terms = list(parsed.terms)
        for phrase_terms in parsed.phrases:
            merged_terms.extend(phrase_terms)

        return self._rank_terms(
            merged_terms,
            top_k=top_k,
            weighting=weighting,
            similarity=similarity,
            candidate_docs=candidate_docs,
        )

    def _rank_terms(
        self,
        terms: List[str],
        top_k: int,
        weighting: str,
        similarity: str,
        candidate_docs: Optional[Set[int]] = None,
    ) -> List[SearchResult]:
        if not terms:
            return []

        query_weights, query_norm = self.index.weight_query_terms(terms, weighting)
        if not query_weights:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for term, query_weight in query_weights.items():
            term_postings = self.index.postings.get(term)
            if not term_postings:
                continue
            for doc_id, doc_tf in term_postings.items():
                if candidate_docs is not None and doc_id not in candidate_docs:
                    continue
                doc_weight = self.index.weight_document_term(term, doc_tf, weighting)
                scores[doc_id] += doc_weight * query_weight

        if similarity == "cosine":
            if query_norm == 0.0:
                return []
            for doc_id in list(scores.keys()):
                denom = self.index.doc_norms[weighting][doc_id] * query_norm
                scores[doc_id] = scores[doc_id] / denom if denom else 0.0

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            SearchResult(
                rank=index + 1,
                doc_id=doc_id,
                path=self.index.doc_id_to_path[doc_id],
                score=score,
            )
            for index, (doc_id, score) in enumerate(ranked)
            if score > 0
        ]

    def _phrase_docs(self, phrase_terms: List[str]) -> Set[int]:
        if not phrase_terms:
            return set()

        term_docs = []
        for term in phrase_terms:
            postings = self.index.positions.get(term)
            if not postings:
                return set()
            term_docs.append(set(postings.keys()))

        candidate_docs = set.intersection(*term_docs)
        matches: Set[int] = set()

        for doc_id in candidate_docs:
            if self._count_phrase_occurrences(doc_id, phrase_terms) > 0:
                matches.add(doc_id)
        return matches

    def _count_phrase_occurrences(self, doc_id: int, phrase_terms: List[str]) -> int:
        first_positions = self.index.positions[phrase_terms[0]][doc_id]
        other_sets = [set(self.index.positions[term][doc_id]) for term in phrase_terms[1:]]

        count = 0
        for pos in first_positions:
            matches = True
            for offset, position_set in enumerate(other_sets, start=1):
                if (pos + offset) not in position_set:
                    matches = False
                    break
            if matches:
                count += 1

        return count

    def _search_proximity(
        self,
        term1: str,
        term2: str,
        max_distance: int,
        top_k: int,
        weighting: str,
        similarity: str,
    ) -> List[SearchResult]:
        postings_1 = self.index.positions.get(term1)
        postings_2 = self.index.positions.get(term2)
        if not postings_1 or not postings_2:
            return []

        candidate_docs = set(postings_1.keys()) & set(postings_2.keys())
        if not candidate_docs:
            return []

        proximity_counts: Dict[int, int] = {}
        for doc_id in candidate_docs:
            count = self._count_proximity_occurrences(
                postings_1[doc_id], postings_2[doc_id], max_distance
            )
            if count > 0:
                proximity_counts[doc_id] = count

        if not proximity_counts:
            return []

        ranked = self._rank_terms(
            [term1, term2],
            top_k=max(top_k, len(proximity_counts)),
            weighting=weighting,
            similarity=similarity,
            candidate_docs=set(proximity_counts.keys()),
        )

        base_scores = {result.doc_id: result.score for result in ranked}
        final = []
        for doc_id, count in proximity_counts.items():
            score = base_scores.get(doc_id, 0.0) + float(count)
            final.append((doc_id, score))

        final_sorted = sorted(final, key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            SearchResult(
                rank=index + 1,
                doc_id=doc_id,
                path=self.index.doc_id_to_path[doc_id],
                score=score,
            )
            for index, (doc_id, score) in enumerate(final_sorted)
        ]

    @staticmethod
    def _count_proximity_occurrences(
        positions1: List[int], positions2: List[int], max_distance: int
    ) -> int:
        i = 0
        j = 0
        count = 0

        while i < len(positions1) and j < len(positions2):
            p1 = positions1[i]
            p2 = positions2[j]
            distance = abs(p1 - p2)

            if distance <= max_distance:
                count += 1
                if p1 <= p2:
                    i += 1
                else:
                    j += 1
            elif p1 < p2:
                i += 1
            else:
                j += 1

        return count

