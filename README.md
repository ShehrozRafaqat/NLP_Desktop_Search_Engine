# NLP Desktop Search Engine (Inverted Index)

This project implements the full NLP lab assignment: a desktop search engine that indexes 50,000+ local text files and supports ranked retrieval using multiple vector and similarity variants.

## Assignment Coverage

Implemented requirements:
- Recursive file discovery and text extraction for `.txt` files.
- Tokenization, lowercasing, punctuation removal.
- Optional stopword removal and optional stemming.
- Inverted index with:
  - `term -> postings`
  - postings entries: `(docID, term frequency)`
  - document frequency (`df`)
  - document vector norms (for cosine similarity)
- Persisted index saved to disk and reloadable without rebuilding.
- Vector variants:
  - Boolean / Binary
  - Count / Frequency
  - TF-IDF using:
    - `idf(t) = log((N + 1) / (df(t) + 1)) + 1`
    - `w(t,d) = tf(t,d) * idf(t)`
- Similarity measures:
  - Inner Product (Dot Product)
  - Cosine Similarity
- Top-k ranked retrieval (default `k=10`) with rank, path, and score.
- Query mode switching for weighting and similarity.
- Bonus:
  - Phrase queries (`"Punjab University"`)
  - Proximity queries (`Pakistan Cricket ^5`)

## Project Structure

- `run_search_engine.py`: main CLI entrypoint
- `src/desktop_search/preprocess.py`: tokenization/normalization
- `src/desktop_search/indexer.py`: index construction + persistence
- `src/desktop_search/query.py`: keyword/phrase/proximity retrieval
- `src/desktop_search/cli.py`: CLI commands (`index`, `search`, `shell`, `stats`)
- `scripts/generate_synthetic_corpus.py`: generate 50k+ `.txt` files
- `scripts/run_experiments.py`: run experiments and export results
- `tests/test_smoke.py`: core behavior smoke tests
- `artifacts/`: proof and experiment outputs

## Quick Start

### 1. Build index

```bash
python3 run_search_engine.py index \
  --data-dir /path/to/text/files \
  --index-dir index_data/my_index \
  --extensions .txt \
  --progress-every 2000
```

Optional preprocessing flags:
- `--remove-stopwords`
- `--stem`

### 2. Run keyword search

```bash
python3 run_search_engine.py search \
  --index-dir index_data/my_index \
  --query "nlp search engine" \
  --top-k 10 \
  --weighting tfidf \
  --similarity cosine
```

### 3. Run phrase/proximity queries

```bash
python3 run_search_engine.py search --index-dir index_data/my_index --query '"Punjab University"'
python3 run_search_engine.py search --index-dir index_data/my_index --query 'Pakistan Cricket ^5'
```

### 4. Interactive shell

```bash
python3 run_search_engine.py shell --index-dir index_data/my_index
```

Shell commands:
- `:weight boolean|count|tfidf`
- `:sim dot|cosine`
- `:k <integer>`
- `:quit`

## 50,000+ File Proof

This repository includes proof artifacts from an executed 50k-file run:
- `artifacts/indexing_proof.json`
- `artifacts/experiment_results.json`
- `artifacts/sample_query_keyword.txt`
- `artifacts/sample_query_phrase.txt`
- `artifacts/sample_query_proximity.txt`

The proof run indexed:
- `50,000` documents
- `2,750,691` total tokens
- index size: `11M` (compressed `index.pkl.gz`)

## Reproduce the 50k Run

```bash
python3 scripts/generate_synthetic_corpus.py \
  --output-dir data/synthetic_50000 \
  --num-files 50000 --min-words 35 --max-words 75 --seed 42

python3 run_search_engine.py index \
  --data-dir data/synthetic_50000 \
  --index-dir index_data/synthetic_50000_idx \
  --extensions .txt --progress-every 5000

python3 scripts/run_experiments.py \
  --index-dir index_data/synthetic_50000_idx \
  --output artifacts/experiment_results.json
```

## Testing

```bash
python3 -m unittest discover -s tests -q
```

## Notes

- This implementation targets local `.txt` corpora and is optimized for assignment workflows.
- Persistence format is compressed pickle (`index.pkl.gz`) plus `manifest.json`.
