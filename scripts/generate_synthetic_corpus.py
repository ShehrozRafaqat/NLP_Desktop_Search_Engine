#!/usr/bin/env python3
"""Generate a large synthetic .txt corpus for indexing benchmarks."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

TOPIC_WORDS = [
    "nlp",
    "language",
    "tokenization",
    "index",
    "search",
    "document",
    "retrieval",
    "vector",
    "cosine",
    "similarity",
    "frequency",
    "ranking",
    "model",
    "science",
    "dataset",
    "analysis",
    "learning",
    "engine",
    "desktop",
    "query",
    "punjab",
    "university",
    "pakistan",
    "cricket",
    "lahore",
    "karachi",
    "islamabad",
    "corpus",
    "feature",
    "python",
    "assignment",
    "evaluation",
    "precision",
    "recall",
    "idf",
    "term",
    "posting",
    "file",
    "text",
    "pipeline",
]


def build_document(doc_id: int, rng: random.Random, min_words: int, max_words: int) -> str:
    length = rng.randint(min_words, max_words)
    words = [rng.choice(TOPIC_WORDS) for _ in range(length)]

    # Ensure phrase query support appears in a subset of files.
    if doc_id % 17 == 0 and len(words) > 5:
        pos = rng.randint(0, len(words) - 2)
        words[pos] = "punjab"
        words[pos + 1] = "university"

    # Ensure proximity query examples appear in a subset of files.
    if doc_id % 23 == 0 and len(words) > 10:
        start = rng.randint(0, len(words) - 7)
        gap = rng.randint(0, 5)
        words[start] = "pakistan"
        words[start + gap + 1] = "cricket"

    # Add light punctuation for more realistic text extraction.
    sentence_breaks = {rng.randint(8, max(9, len(words) - 1)) for _ in range(3)}
    chunks: List[str] = []
    current: List[str] = []

    for idx, token in enumerate(words, start=1):
        current.append(token)
        if idx in sentence_breaks:
            chunks.append(" ".join(current).capitalize() + ".")
            current = []

    if current:
        chunks.append(" ".join(current).capitalize() + ".")

    return "\n".join(chunks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic corpus")
    parser.add_argument("--output-dir", required=True, help="Output root for .txt files")
    parser.add_argument("--num-files", type=int, default=50000, help="Number of documents")
    parser.add_argument("--min-words", type=int, default=35, help="Minimum words per file")
    parser.add_argument("--max-words", type=int, default=75, help="Maximum words per file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for doc_id in range(args.num_files):
        bucket = output / f"batch_{doc_id // 1000:03d}"
        bucket.mkdir(parents=True, exist_ok=True)
        path = bucket / f"doc_{doc_id:05d}.txt"
        path.write_text(
            build_document(doc_id, rng, args.min_words, args.max_words),
            encoding="utf-8",
        )

        if (doc_id + 1) % 5000 == 0:
            print(f"Generated {doc_id + 1}/{args.num_files} files")

    print(f"Synthetic corpus generated at: {output.resolve()}")


if __name__ == "__main__":
    main()
