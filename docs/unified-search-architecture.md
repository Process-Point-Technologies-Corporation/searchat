# Unified Two-Layer Search Architecture

> **Note**: After code changes, restart server with `searchat-web` to enable unified search.
> Test with: `python scripts/test_search_modes.py`

## Overview

Searchat implements a two-layer search architecture that combines:
- **Layer 1 (Verbatim)**: Raw conversation messages with full context
- **Layer 2 (Palace)**: Compacted objects with curated summaries, room assignments, file paths

## Search Modes

### Per-Layer Modes

Each layer supports three search modes:

| Mode | Method | Strengths | Weaknesses |
|------|--------|-----------|------------|
| **Keyword** | BM25 (Okapi) | Exact matches, partial words, specific terms | No semantic understanding |
| **Semantic** | FAISS + embeddings | Conceptual similarity, synonyms | May miss exact terms |
| **Hybrid** | BM25 + FAISS weighted | Best of both | Slightly slower |

### Mode Combinations (3×3 Matrix)

| Verbatim Mode | Palace Mode | Use Case |
|---------------|-------------|----------|
| hybrid | hybrid | **Default** - comprehensive search |
| keyword | keyword | Exact term matching across both layers |
| semantic | semantic | Conceptual/meaning-based search |
| keyword | hybrid | Exact verbatim + broad palace |
| hybrid | keyword | Broad verbatim + exact palace |
| semantic | keyword | Conceptual verbatim + exact palace |
| keyword | semantic | Exact verbatim + conceptual palace |
| hybrid | semantic | Broad verbatim + conceptual palace |
| semantic | hybrid | Conceptual verbatim + broad palace |

### Current Implementation

**Default (unified hybrid-hybrid):**
- Verbatim: mode parameter (hybrid/semantic/keyword)
- Palace: always hybrid (BM25 + FAISS)
- Merge: weighted combination with intersection boost

## Architecture Decision: Why Hybrid-Hybrid Default

### Rationale

1. **Complementary strengths**: Keyword catches exact matches (error codes, file paths); semantic catches conceptual matches (synonyms, related concepts)

2. **Layer coverage**:
   - Palace may miss some conversations (not yet compacted)
   - Verbatim may miss high-level summaries
   - Running both ensures no gaps

3. **Intersection signal**: Results appearing in both layers are more likely relevant → 20% boost

4. **Latency acceptable**: Parallel execution keeps total time under 200ms

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Parallel merge (chosen) | Complete coverage, intersection boost | Higher resource usage |
| Palace-first cascade | Lower latency, less noise | May miss verbatim-only matches |
| Verbatim-only | Simpler, faster | Loses palace summaries/rooms |

## Scoring Algorithm

### Within-Layer Scoring

**BM25 (keyword):**
```
score = Σ IDF(term) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × dl/avgdl))
```

**FAISS (semantic):**
```
score = 1 / (1 + L2_distance)
```

**Hybrid combination:**
```
normalized_kw = kw_score / max(kw_scores)
normalized_sem = sem_score / max(sem_scores)
combined = (kw_weight × normalized_kw) + (sem_weight × normalized_sem)
if in_both: combined *= 1.2  # intersection boost
```

### Cross-Layer Merge

```python
# Normalize each layer's scores to 0-1
palace_norm = palace_score / max(palace_scores)
verbatim_norm = verbatim_score / max(verbatim_scores)

# Combine with weights (default 0.5/0.5)
combined = palace_weight * palace_norm + verbatim_weight * verbatim_norm

# Boost intersections (same conversation in both layers)
if has_palace and has_verbatim:
    combined *= 1.2
```

## Data Flow

```
Query
  │
  ├──────────────────────────────────────┐
  │                                      │
  ▼                                      ▼
┌─────────────────────┐    ┌─────────────────────┐
│   Verbatim Search   │    │   Palace Search     │
│                     │    │                     │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │ BM25 (keyword)  │ │    │ │ BM25 (keyword)  │ │
│ └────────┬────────┘ │    │ └────────┬────────┘ │
│          │ merge    │    │          │ merge    │
│ ┌────────▼────────┐ │    │ ┌────────▼────────┐ │
│ │ FAISS (semantic)│ │    │ │ FAISS (semantic)│ │
│ └────────┬────────┘ │    │ └────────┬────────┘ │
│          ▼          │    │          ▼          │
│   Ranked Results    │    │   Ranked Results    │
└──────────┬──────────┘    └──────────┬──────────┘
           │                          │
           └──────────┬───────────────┘
                      ▼
              ┌───────────────┐
              │  Merge by     │
              │ conversation  │
              │      ID       │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │ Intersection  │
              │    Boost      │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │    Sort by    │
              │ combined_score│
              └───────┬───────┘
                      ▼
               Unified Results
```

## Performance Characteristics

| Metric | Target | Typical |
|--------|--------|---------|
| BM25 index build | < 3s | ~1s for 14k objects |
| Palace search | < 50ms | 20-40ms |
| Verbatim search | < 100ms | 50-80ms |
| Unified (parallel) | < 200ms | 80-150ms |

## API Usage

```bash
# Default unified search (hybrid-hybrid)
curl "http://localhost:8000/api/search?q=FAISS"

# Keyword mode (verbatim keyword, palace hybrid)
curl "http://localhost:8000/api/search?q=IndexFlatL2&mode=keyword"

# Semantic mode (verbatim semantic, palace hybrid)
curl "http://localhost:8000/api/search?q=vector similarity&mode=semantic"
```

## Evaluation Metrics

Standard IR metrics for comparing modes:

| Metric | Formula | Best For |
|--------|---------|----------|
| **MRR** | mean(1/rank_first_relevant) | Single-answer queries |
| **Precision@K** | relevant_in_K / K | Result quality |
| **Recall@K** | relevant_in_K / total_relevant | Coverage |
| **nDCG@K** | DCG@K / ideal_DCG@K | Graded relevance |

## Testing & Benchmarking

### Quick Test
```bash
python scripts/test_search_modes.py
```

### Full Benchmark
```bash
python scripts/benchmark_search.py -n 5 -v -o benchmark_results.json
```

### Expected Behavior by Query Type

| Query Type | Best Mode | Why |
|------------|-----------|-----|
| Exact terms (`IndexFlatL2`) | keyword | BM25 matches exact tokens |
| File paths (`compactor.py`) | keyword | Path tokenization |
| Conceptual (`vector search`) | semantic/hybrid | Embedding similarity |
| Error messages | keyword | Exact string matching |
| Mixed (`FAISS search`) | hybrid | Combines both strengths |

### Interpreting Results

**Unified search active** when response contains:
- `palace_count` and `verbatim_count` fields
- `has_palace`, `has_verbatim`, `is_intersection` flags per result

**Quality indicators**:
- High intersection count → query terms well-indexed in both layers
- Palace-only results → good compaction coverage
- Verbatim-only results → may indicate missing compaction

## References

- [Pinecone: Evaluation Measures in IR](https://www.pinecone.io/learn/offline-evaluation/)
- [Weaviate: Retrieval Evaluation Metrics](https://weaviate.io/blog/retrieval-evaluation-metrics)
- BM25: Robertson & Zaragoza, "The Probabilistic Relevance Framework"
- FAISS: Johnson et al., "Billion-scale similarity search with GPUs"
