"""Command-line interface for the desktop search engine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from .indexer import InvertedIndex
from .query import SearchEngine


def _add_common_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument(
        "--weighting",
        choices=["boolean", "count", "tfidf"],
        default="tfidf",
        help="Vector weighting scheme",
    )
    parser.add_argument(
        "--similarity",
        choices=["dot", "cosine"],
        default="cosine",
        help="Similarity measure",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="desktop-search",
        description="Desktop search engine using an inverted index",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build and save an inverted index")
    index_parser.add_argument("--data-dir", required=True, help="Directory containing documents")
    index_parser.add_argument(
        "--index-dir", default="index_data", help="Output directory for persisted index"
    )
    index_parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".txt"],
        help="File extensions to include (e.g. .txt)",
    )
    index_parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Enable stopword removal during indexing",
    )
    index_parser.add_argument(
        "--stem",
        action="store_true",
        help="Enable lightweight stemming during indexing",
    )
    index_parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print progress after this many files",
    )

    search_parser = subparsers.add_parser("search", help="Run a single query")
    search_parser.add_argument("--index-dir", default="index_data", help="Persisted index location")
    search_parser.add_argument("--query", required=True, help="Query text")
    _add_common_retrieval_args(search_parser)

    shell_parser = subparsers.add_parser("shell", help="Start interactive query shell")
    shell_parser.add_argument("--index-dir", default="index_data", help="Persisted index location")
    _add_common_retrieval_args(shell_parser)

    stats_parser = subparsers.add_parser("stats", help="Show index manifest and summary")
    stats_parser.add_argument("--index-dir", default="index_data", help="Persisted index location")

    return parser


def run_index(args: argparse.Namespace) -> None:
    index = InvertedIndex(
        remove_stopwords=args.remove_stopwords,
        use_stemming=args.stem,
    )
    stats = index.build(
        data_dir=args.data_dir,
        extensions=args.extensions,
        progress_every=args.progress_every,
    )
    index.save(args.index_dir)

    summary = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "index_dir": str(Path(args.index_dir).resolve()),
        "total_documents": stats.total_documents,
        "indexed_files": stats.indexed_files,
        "vocabulary_size": stats.total_terms,
        "total_tokens": stats.total_tokens,
        "elapsed_seconds": round(stats.elapsed_seconds, 3),
        "remove_stopwords": args.remove_stopwords,
        "use_stemming": args.stem,
    }

    print("Indexing complete:")
    print(json.dumps(summary, indent=2))


def _print_results(results) -> None:
    if not results:
        print("No documents matched.")
        return

    print(f"Returned {len(results)} result(s):")
    for result in results:
        print(f"{result.rank:>2}. score={result.score:.6f} path={result.path}")


def run_search(args: argparse.Namespace) -> None:
    engine = SearchEngine.from_disk(args.index_dir)
    started = time.time()
    results = engine.search(
        raw_query=args.query,
        top_k=args.top_k,
        weighting=args.weighting,
        similarity=args.similarity,
    )
    elapsed_ms = (time.time() - started) * 1000.0

    print(
        f"Query='{args.query}' weighting={args.weighting} similarity={args.similarity} "
        f"top_k={args.top_k} latency_ms={elapsed_ms:.3f}"
    )
    _print_results(results)


def run_shell(args: argparse.Namespace) -> None:
    engine = SearchEngine.from_disk(args.index_dir)
    weighting = args.weighting
    similarity = args.similarity
    top_k = args.top_k

    print("Interactive shell ready.")
    print("Commands: :weight boolean|count|tfidf, :sim dot|cosine, :k <int>, :quit")

    while True:
        try:
            raw = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye")
            break

        if not raw:
            continue
        if raw == ":quit":
            print("Bye")
            break

        if raw.startswith(":weight "):
            candidate = raw.split(maxsplit=1)[1].strip()
            if candidate in {"boolean", "count", "tfidf"}:
                weighting = candidate
                print(f"weighting={weighting}")
            else:
                print("Invalid weighting")
            continue

        if raw.startswith(":sim "):
            candidate = raw.split(maxsplit=1)[1].strip()
            if candidate in {"dot", "cosine"}:
                similarity = candidate
                print(f"similarity={similarity}")
            else:
                print("Invalid similarity")
            continue

        if raw.startswith(":k "):
            try:
                candidate = int(raw.split(maxsplit=1)[1].strip())
                if candidate <= 0:
                    raise ValueError
                top_k = candidate
                print(f"top_k={top_k}")
            except ValueError:
                print("Invalid top_k value")
            continue

        started = time.time()
        results = engine.search(
            raw_query=raw,
            top_k=top_k,
            weighting=weighting,
            similarity=similarity,
        )
        elapsed_ms = (time.time() - started) * 1000.0
        print(
            f"weighting={weighting} similarity={similarity} top_k={top_k} "
            f"latency_ms={elapsed_ms:.3f}"
        )
        _print_results(results)


def run_stats(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    manifest = index_dir / "manifest.json"

    if not manifest.exists():
        raise FileNotFoundError(f"No manifest found at {manifest}")

    with open(manifest, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    print("Index manifest:")
    print(json.dumps(data, indent=2))


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        run_index(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "shell":
        run_shell(args)
    elif args.command == "stats":
        run_stats(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
