# NeuroSeek

NeuroSeek is a semantic search engine for your own content. You give it text — from files, directories, or strings — and it finds the passages that mean what you're looking for, not just the ones that share your exact words. It runs entirely locally, stores everything in a single file, and requires no server. The core is ~1 100 lines of pure Python built on an HNSW graph and sentence-transformers.

```python
from neuroseek import SearchEngine

engine = SearchEngine()
engine.add("The mitochondria is the powerhouse of the cell.")
engine.add("Neural networks learn by adjusting weights through backpropagation.")
engine.add("Photosynthesis converts sunlight into chemical energy in plants.")

results = engine.search("how do cells produce energy?")
for r in results:
    print(f"{r['score']:.3f}  {r['text']}")
```

```
0.847  The mitochondria is the powerhouse of the cell.
0.412  Photosynthesis converts sunlight into chemical energy in plants.
0.201  Neural networks learn by adjusting weights through backpropagation.
```

The query "how do cells produce energy?" never appears in any document. NeuroSeek finds the right answer anyway.

---

## Installation

```bash
pip install neuroseek
```

Requires Python 3.10+. The first run downloads the embedding model (~90 MB, cached automatically by Hugging Face).

---

## Usage

### Index a file and search it

```python
from neuroseek import SearchEngine
from neuroseek.ingestion.ingestor import ingest_file
from neuroseek.ingestion.chunker import chunk_text

engine = SearchEngine()

text, metadata = ingest_file("notes.txt")
for chunk in chunk_text(text):
    engine.add(chunk, metadata=metadata)

results = engine.search("deadline for the project", top_k=3)
for r in results:
    print(f"{r['score']:.3f}  {r['text'][:80]}")
```

### Multiple namespaces

Namespaces let you keep separate corpora in one index file — for example, one namespace per project or document collection.

```python
from neuroseek import NamespaceManager

nm = NamespaceManager()
nm.add("Black holes warp spacetime.", namespace="physics")
nm.add("The Fed raised interest rates by 50 basis points.", namespace="finance")

nm.search("gravity and curvature", namespace="physics", top_k=1)
# [{'id': 0, 'text': 'Black holes warp spacetime.', 'score': 0.74, 'metadata': {}}]
```

### Save and load

```python
from neuroseek import save_namespace_manager, load_namespace_manager

save_namespace_manager(nm, "index.pkl")
nm = load_namespace_manager("index.pkl")
```

---

## CLI

The `neuroseek` command provides a full CLI over a persistent index (default: `~/.neuroseek/index.pkl`).

```
# Index a file or directory
neuroseek index notes.txt
neuroseek index ./docs --chunk-size 256 --chunk-overlap 32

# Search
neuroseek search "how do I reset a password?"
neuroseek search "database migrations" --namespace backend --top-k 10

# Manage sources
neuroseek list-sources
neuroseek delete notes.txt
neuroseek update notes.txt          # re-index after editing

# Export / import
neuroseek export backup.json
neuroseek import backup.json

# List all namespaces
neuroseek list
```

Full reference:

| Command | Options |
|---|---|
| `index <path>` | `--namespace`, `--chunk-size`, `--chunk-overlap` |
| `search "<query>"` | `--namespace`, `--top-k` |
| `delete <filename>` | `--namespace` |
| `delete --query "<q>"` | `--namespace`, `--top-k`, `--dry-run` |
| `update <path>` | `--namespace`, `--chunk-size`, `--chunk-overlap` |
| `list` | — |
| `list-sources` | `--namespace` |
| `export <output.json>` | `--namespace` |
| `import <input.json>` | `--namespace` |

Override the index path with `--index <path>` or the `NEUROSEEK_INDEX` environment variable.

---

## Python API

```python
from neuroseek import (
    SearchEngine,
    NamespaceManager,
    Embedder,
    DocumentStore,
    HNSWIndex,
    chunk_text,
    ingest_file,
    ingest_directory,
    SUPPORTED_EXTENSIONS,
    save_namespace_manager,
    load_namespace_manager,
)
from neuroseek.persistence.json_persistence import export_namespace_manager, import_from_json
```

**`SearchEngine`** — the main object. Wraps embedder + HNSW index + document store.

| Method | Description |
|---|---|
| `add(text, metadata=None)` | Embed and index one document |
| `add_batch(texts, metadata_list=None)` | Embed and index many documents |
| `search(query, top_k=5, filter=None)` | Return top-k semantically similar docs |
| `delete(id)` | Remove one document by ID |
| `delete_by_source(filename)` | Remove all docs from a source file |
| `delete_by_query(query, top_k=5)` | Search then delete matching docs |
| `update_source(filename, chunks)` | Re-index all chunks for a file |
| `list_sources()` | Set of distinct filenames in the index |

`search` results are `list[dict]` — each dict has `id` (int), `text` (str), `score` (float, cosine similarity in [0, 1]), and `metadata` (dict).

**`NamespaceManager`** — same API as `SearchEngine`, with a `namespace` argument on every call.

**`Embedder`** — wraps `sentence-transformers`. Default model: `multi-qa-MiniLM-L6-cos-v1` (384-dim).

**Supported file types**: `.txt`, `.md`, `.py`, `.json`, `.csv`

---

## Benchmarks

MS MARCO (10 000 passages, 26 queries, top-10, `multi-qa-MiniLM-L6-cos-v1`, DIM=384):

| System | Build | p50 search | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| **NeuroSeek** | 111 s | 24 ms | **0.96** | **0.58** | **0.67** |
| hnswlib | 1.8 s | 23 ms | 0.92 | 0.56 | 0.65 |
| FAISS | 2.4 s | 23 ms | 0.96 | 0.58 | 0.67 |
| ChromaDB | 4.8 s | 31 ms | 0.96 | 0.58 | 0.67 |
| BM25 | 0.9 s | 35 ms | 0.65 | 0.30 | 0.38 |

Search latency and retrieval quality match compiled C++ libraries (hnswlib, FAISS, ChromaDB). Index build time is slower: pure Python graph traversal carries overhead that compiled libraries avoid. For 10k documents, 111 s is a one-time cost paid at startup.

To reproduce: `pip install -r benchmarks/requirements.txt && python3 -m benchmarks.run --passages 10000 --queries 200 --top-k 10`

---

## How it works

NeuroSeek embeds every document chunk into a 384-dimensional vector using `multi-qa-MiniLM-L6-cos-v1`, a model trained specifically for semantic search. At query time the query string is embedded the same way, and the engine finds the nearest vectors in the HNSW graph by cosine similarity.

[HNSW](https://arxiv.org/abs/1603.09320) (Hierarchical Navigable Small World) is an approximate nearest-neighbour algorithm. It builds a layered graph where each node connects to its `M` closest neighbours. Search navigates down the layers greedily, reaching the neighbourhood of the query vector in O(log n) hops.

The implementation follows the paper closely:

- Vectors are pre-normalised at insertion time and stored in a contiguous `float32` numpy matrix. Cosine distance becomes a single dot product with no norm recomputation.
- Candidate neighbour distances are computed in one batched BLAS call (`matrix[rows] @ query`) rather than one Python loop per neighbour.
- Layer assignment uses the paper formula `floor(-ln(u) / ln(M))` which produces O(log_M N) layers — typically 3–4 for 10k vectors.
- Neighbour selection uses Algorithm 4 (heuristic selection): a candidate is kept only if it is closer to the query than to any already-selected neighbour, ensuring diverse graph connectivity and high recall.
- Deletion is O(degree × L) using a reverse-adjacency index rather than O(N × L × M) full scan.

The index and document store are serialised together into a single pickle file with a version header. Loading a file from a different version raises a clear `ValueError` rather than silently returning wrong results.

---

## Running tests

```bash
PYTHONPATH=/path/to/NeuroSeek pytest tests/ -q
```

The test suite has 1 056 tests covering the HNSW core, embedder, document store, search engine, namespace manager, persistence, JSON export/import, CLI, and ingestion pipeline. All tests use real models and real data — no mocks.

---

## License

MIT
