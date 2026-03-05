# NLP Lab Assignment Report
## Desktop Search Engine using Inverted Index (50,000+ Files)

### Student Project Summary
This report documents the implementation and evaluation of a desktop search engine for local text files using an inverted index. The system was designed to satisfy all required assignment objectives:
- scalable indexing for 50,000+ files,
- Boolean/Count/TF-IDF weighting,
- Dot Product/Cosine similarity,
- top-k ranked retrieval,
- phrase and proximity query support.

The final implementation is available in this repository and includes executable scripts, reproducible experiments, and proof artifacts.

---

## 1. Problem Statement
The task is to build an NLP-based desktop search engine that can index a large local corpus and retrieve relevant documents for keyword queries. The engine must support multiple vector representations and ranking strategies while remaining reloadable from disk without re-indexing.

Key requirements addressed:
1. Recursive discovery of local files (minimum `.txt`).
2. Preprocessing pipeline (tokenization, normalization; optional stopword removal/stemming).
3. Persistent inverted index with term statistics and document norms.
4. Ranking with dot product and cosine similarity.
5. Bonus query types: phrase and proximity.

---

## 2. System Design

### 2.1 Preprocessing
Each document is processed through:
1. Lowercasing
2. Token extraction with regex (`[A-Za-z0-9]+`)
3. Punctuation removal (implicit via token pattern)
4. Optional stopword removal
5. Optional lightweight stemming

This normalization is also applied to user queries to ensure matching consistency.

### 2.2 Inverted Index Structure
The index stores:
- `doc_id_to_path`: mapping from integer `docID` to file path
- `postings`: `term -> {docID: tf}`
- `positions`: `term -> {docID: [pos1, pos2, ...]}` (for phrase/proximity)
- `df`: document frequency for each term
- `idf`: inverse document frequency
- `doc_norms`: precomputed norms for Boolean, Count, and TF-IDF vectors

Persistence:
- `index.pkl.gz` (compressed serialized index payload)
- `manifest.json` (metadata summary)

The index can be loaded directly for querying without rebuilding.

### 2.3 Vector Weighting Implementations
Implemented weighting variants:
1. **Boolean/Binary**
   - `w(t,d) = 1` if term appears, else `0`
2. **Count/Frequency**
   - `w(t,d) = tf(t,d)`
3. **TF-IDF**
   - `idf(t) = log((N + 1) / (df(t) + 1)) + 1`
   - `w(t,d) = tf(t,d) * idf(t)`

Same weighting choices are applied to query vectors.

### 2.4 Similarity Functions
1. **Dot Product**
   - `score(d,q) = sum_t w(t,d) * w(t,q)`
2. **Cosine Similarity**
   - `score(d,q) = dot(d,q) / (||d|| * ||q||)`

Document norms are precomputed to reduce query-time overhead.

### 2.5 Query Processing
Supported query modes:
1. **Keyword query** (default): ranked retrieval on query terms.
2. **Phrase query** (`"Punjab University"`): exact consecutive-token matching using positional postings.
3. **Proximity query** (`Pakistan Cricket ^5`): documents where terms occur within `k` words.

For phrase/proximity matches, ranking is still performed with selected weighting/similarity to keep behavior consistent.

---

## 3. Dataset and Experimental Setup

### 3.1 Corpus
A synthetic text corpus was generated for reproducible benchmarking:
- Total files: **50,000 `.txt` documents**
- Total tokens indexed: **2,750,691**
- Vocabulary size: **40 terms** (controlled topical vocabulary)
- Directory structure: recursively nested (`batch_*` folders)

Generation script: `scripts/generate_synthetic_corpus.py`

### 3.2 Indexing Run
Indexing command executed:
```bash
python3 run_search_engine.py index \
  --data-dir data/synthetic_50000 \
  --index-dir index_data/synthetic_50000_idx \
  --extensions .txt --progress-every 5000
```

Observed indexing statistics:
- documents indexed: **50,000**
- elapsed time: **12.505 seconds**
- compressed index size: **11M**

Proof file: `artifacts/indexing_proof.json`

### 3.3 Query Evaluation Configuration
Queries evaluated:
1. `nlp search engine` (keyword)
2. `"punjab university"` (phrase)
3. `pakistan cricket ^5` (proximity)

Mode combinations tested:
- Weighting: Boolean, Count, TF-IDF
- Similarity: Dot, Cosine
- Total runs: 18

Output file: `artifacts/experiment_results.json`

---

## 4. Results

### 4.1 Aggregate Latency
From 18 runs:
- Minimum latency: **69.9815 ms**
- Maximum latency: **435.2187 ms**
- Mean latency: **146.6123 ms**
- Median latency: **130.5733 ms**

### 4.2 Retrieval Behavior by Mode

Observations from experiment outputs:
1. **Boolean + Dot** returns many tied scores for broad queries, because document term presence is binary.
2. **Count + Dot** favors documents with repeated term occurrences.
3. **TF-IDF + Dot** emphasizes informative terms and produces stronger separation in scores.
4. **Cosine** normalizes vector length and reduces bias toward long documents.

Example top scores (selected):
- Keyword (`nlp search engine`):
  - TF-IDF + Dot top score: **27.062830**
  - TF-IDF + Cosine top score: **0.660527**
- Phrase (`"punjab university"`):
  - TF-IDF + Cosine top score: **0.681798**
- Proximity (`pakistan cricket ^5`):
  - TF-IDF + Cosine top score: **9.636515**

Sample query logs:
- `artifacts/sample_query_keyword.txt`
- `artifacts/sample_query_phrase.txt`
- `artifacts/sample_query_proximity.txt`

### 4.3 Bonus Feature Validation
- Phrase matching correctly retrieved files containing exact contiguous phrase tokens.
- Proximity matching correctly retrieved files where the two terms occur within the user-defined window.
- Both feature types integrate with the same top-k ranking interface.

---

## 5. Discussion

### 5.1 Why the Design Scales to 50k+
- Indexing uses a single pass over files.
- Posting lists are stored by term and only touched for query terms.
- Norms are precomputed once during build time.
- Persisted index avoids repeated rebuild cost.

### 5.2 Trade-offs
- Positional postings improve phrase/proximity support but increase index size.
- Pickle-based persistence is fast and simple, but less language-agnostic than a database format.
- Synthetic data demonstrates scale/performance reproducibly; real-world corpora would provide richer semantic variability.

### 5.3 Recommended Defaults
For general ranked retrieval, best default in this assignment context:
- Weighting: **TF-IDF**
- Similarity: **Cosine**

This combination gave stable and interpretable ranking quality across tested query types.

---

## 6. Conclusion
The assignment objectives were fully implemented and validated:
1. A persistent inverted-index desktop search engine was built.
2. All required vector and similarity variants were implemented.
3. Top-k ranked retrieval interface works for local corpus search.
4. Bonus phrase/proximity queries are supported with positional indexing.
5. A full 50,000-document indexing proof and experiment report are included.

This solution is ready for demo and further extension (additional file formats, larger vocabularies, evaluation against labeled relevance judgments, and GUI integration).
