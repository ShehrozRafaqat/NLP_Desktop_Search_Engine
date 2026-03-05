#!/usr/bin/env python3
"""Run assignment experiments and store a reproducible summary."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.desktop_search.query import SearchEngine


def load_manifest(index_dir: Path) -> Dict[str, object]:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval experiments")
    parser.add_argument("--index-dir", required=True, help="Directory containing saved index")
    parser.add_argument(
        "--output",
        default="artifacts/experiment_results.json",
        help="Path for experiment JSON output",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    engine = SearchEngine.from_disk(str(index_dir))
    manifest = load_manifest(index_dir)

    queries = [
        "nlp search engine",
        "\"punjab university\"",
        "pakistan cricket ^5",
    ]
    weightings = ["boolean", "count", "tfidf"]
    similarities = ["dot", "cosine"]

    runs: List[Dict[str, object]] = []

    for weighting in weightings:
        for similarity in similarities:
            for query in queries:
                started = time.perf_counter()
                results = engine.search(
                    raw_query=query,
                    top_k=args.top_k,
                    weighting=weighting,
                    similarity=similarity,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                runs.append(
                    {
                        "query": query,
                        "weighting": weighting,
                        "similarity": similarity,
                        "latency_ms": round(elapsed_ms, 4),
                        "result_count": len(results),
                        "top_result": (
                            {
                                "path": results[0].path,
                                "score": round(results[0].score, 6),
                            }
                            if results
                            else None
                        ),
                    }
                )

    latencies = [entry["latency_ms"] for entry in runs]
    summary = {
        "index_manifest": manifest,
        "dataset_summary": {
            "total_documents": engine.index.total_docs,
            "vocabulary_size": engine.index.vocabulary_size,
            "total_tokens": engine.index.total_tokens,
        },
        "aggregate": {
            "runs": len(runs),
            "latency_ms_min": round(min(latencies), 4),
            "latency_ms_max": round(max(latencies), 4),
            "latency_ms_mean": round(statistics.mean(latencies), 4),
            "latency_ms_median": round(statistics.median(latencies), 4),
        },
        "runs": runs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Experiment results written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
