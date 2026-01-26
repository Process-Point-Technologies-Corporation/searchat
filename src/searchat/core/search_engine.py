from __future__ import annotations

import json
import time
import hashlib
from collections import defaultdict, OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional, Dict, Tuple, TYPE_CHECKING, cast
import duckdb
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from searchat.models import (
    SearchMode,
    SearchFilters,
    SearchResult,
    SearchResults,
)
from searchat.core.query_parser import QueryParser
from searchat.config import Config


if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SearchEngine:
    def __init__(self, search_dir: Path, config: Config | None = None):
        self.search_dir = search_dir
        self.faiss_index: Optional[faiss.Index] = None
        self.embedder: SentenceTransformer | None = None
        self.query_parser = QueryParser()

        self._init_lock = Lock()
        
        if config is None:
            config = Config.load()
        self.config = config
        
        self.conversations_dir = self.search_dir / "data" / "conversations"
        self.metadata_path = self.search_dir / "data" / "indices" / "embeddings.metadata.parquet"
        self.index_path = self.search_dir / "data" / "indices" / "embeddings.faiss"
        self.conversations_glob = str(self.conversations_dir / "*.parquet")

        # LRU cache for search results
        self.cache_size = config.performance.query_cache_size
        self.result_cache: OrderedDict[str, Tuple[SearchResults, float]] = OrderedDict()
        self.cache_ttl = 300  # 5 minutes TTL

        # Columns needed for search (exclude large 'messages' column)
        self.search_columns = [
            "conversation_id",
            "project_id",
            "file_path",
            "title",
            "created_at",
            "updated_at",
            "message_count",
            "full_text",
            "embedding_id",
            "file_hash",
            "indexed_at",
        ]
        
        # Keyword search only depends on conversation parquets.
        self._validate_keyword_files()
    
    def ensure_metadata_ready(self) -> None:
        """Ensure index metadata parquet exists and matches config."""
        with self._init_lock:
            self._ensure_metadata_ready_locked()


    def ensure_faiss_loaded(self) -> None:
        """Ensure FAISS index is loaded."""
        with self._init_lock:
            self._ensure_faiss_loaded_locked()


    def ensure_embedder_loaded(self) -> None:
        """Ensure sentence-transformers model is loaded."""
        with self._init_lock:
            self._ensure_embedder_loaded_locked()


    def ensure_semantic_ready(self) -> None:
        """Ensure semantic components (metadata, FAISS, embedder) are loaded."""
        with self._init_lock:
            self._ensure_metadata_ready_locked()
            self._ensure_faiss_loaded_locked()
            self._ensure_embedder_loaded_locked()


    def _ensure_metadata_ready_locked(self) -> None:
        self._validate_keyword_files()
        self._validate_index_metadata()

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata parquet not found at {self.metadata_path}. Run indexer first."
            )


    def _ensure_faiss_loaded_locked(self) -> None:
        self._ensure_metadata_ready_locked()

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. Run indexer first."
            )

        if self.faiss_index is None:
            self.faiss_index = faiss.read_index(str(self.index_path))


    def _ensure_embedder_loaded_locked(self) -> None:
        self._validate_index_metadata()

        if self.embedder is None:
            device = self.config.embedding.get_device()
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.config.embedding.model, device=device)


    def refresh_index(self) -> None:
        """Clear caches and mark index-backed components stale.

        This is used after append-only indexing or restore operations.
        """
        with self._init_lock:
            self.result_cache.clear()
            # Reload FAISS on next semantic search.
            self.faiss_index = None
            # Keep embedder loaded (heavy), model mismatch is already blocked by the indexer.


    def _validate_keyword_files(self) -> None:
        if not self.conversations_dir.exists() or not any(self.conversations_dir.glob("*.parquet")):
            raise FileNotFoundError(f"No conversation parquet files found in {self.conversations_dir}")

    def _connect(self):
        con = duckdb.connect(database=":memory:")
        # Keep this conservative; avoid forcing a value if config is missing.
        try:
            mem_mb = int(self.config.performance.memory_limit_mb)
            con.execute(f"PRAGMA memory_limit='{mem_mb}MB'")
        except Exception:
            pass
        return con

    def _where_from_filters(
        self,
        filters: Optional[SearchFilters],
        params: list[object],
        *,
        table_alias: str = "",
    ) -> str:
        prefix = f"{table_alias}." if table_alias else ""
        conditions = [f"{prefix}message_count > 0"]

        if filters:
            if filters.project_ids:
                placeholders = ",".join(["?"] * len(filters.project_ids))
                conditions.append(f"{prefix}project_id IN ({placeholders})")
                params.extend(filters.project_ids)

            if filters.date_from:
                conditions.append(f"{prefix}updated_at >= ?")
                params.append(filters.date_from)

            if filters.date_to:
                conditions.append(f"{prefix}updated_at <= ?")
                params.append(filters.date_to)

            if filters.min_messages > 0:
                conditions.append(f"{prefix}message_count >= ?")
                params.append(int(filters.min_messages))

        return " AND ".join(conditions)
    
    def _validate_index_metadata(self) -> None:
        metadata_path = self.search_dir / 'data/indices/index_metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Index metadata not found at {metadata_path}. "
                "Index format outdated, rebuild required. Run indexer."
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('model_name') != self.config.embedding.model:
            raise ValueError(
                f"Model mismatch: index uses '{metadata.get('model_name')}', "
                f"config specifies '{self.config.embedding.model}'. "
                "Rebuild index with correct model."
            )
        
        if metadata.get('schema_version') != 1:
            raise ValueError(
                f"Schema version mismatch: index uses version {metadata.get('schema_version')}, "
                "expected version 1. Rebuild index required."
            )
    
    def _get_cache_key(self, query: str, mode: SearchMode, filters: Optional[SearchFilters]) -> str:
        """Generate a cache key for the search query"""
        key_parts = [query, mode.value]
        if filters:
            if filters.project_ids:
                key_parts.append(f"projects:{','.join(filters.project_ids)}")
            if filters.date_from:
                key_parts.append(f"from:{filters.date_from.isoformat()}")
            if filters.date_to:
                key_parts.append(f"to:{filters.date_to.isoformat()}")
            if filters.min_messages > 0:
                key_parts.append(f"min_msgs:{filters.min_messages}")
        
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[SearchResults]:
        """Get results from cache if valid"""
        if cache_key in self.result_cache:
            result, timestamp = self.result_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                # Move to end (most recently used)
                self.result_cache.move_to_end(cache_key)
                return result
            else:
                # Expired, remove from cache
                del self.result_cache[cache_key]
        return None
    
    def _add_to_cache(self, cache_key: str, result: SearchResults) -> None:
        """Add results to cache with LRU eviction"""
        # Remove oldest if cache is full
        if len(self.result_cache) >= self.cache_size:
            self.result_cache.popitem(last=False)
        
        self.result_cache[cache_key] = (result, time.time())
    
    def search(
        self, 
        query: str, 
        mode: SearchMode = SearchMode.HYBRID,
        filters: Optional[SearchFilters] = None
    ) -> SearchResults:
        start_time = time.time()

        # Treat wildcard query as keyword-only browsing.
        if query.strip() == "*" and mode != SearchMode.KEYWORD:
            mode = SearchMode.KEYWORD
        
        # Check cache first
        cache_key = self._get_cache_key(query, mode, filters)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            # Update search time to reflect cache hit
            cached_result.search_time_ms = (time.time() - start_time) * 1000
            return cached_result
        
        try:
            if mode == SearchMode.HYBRID:
                keyword_results = self._keyword_search(query, filters)
                semantic_results = self._semantic_search(query, filters)
                results = self._merge_results(keyword_results, semantic_results)
            elif mode == SearchMode.KEYWORD:
                results = self._keyword_search(query, filters)
            else:
                results = self._semantic_search(query, filters)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            search_result = SearchResults(
                results=results,
                total_count=len(results),
                search_time_ms=elapsed_ms,
                mode_used=mode.value
            )
            
            # Add to cache
            self._add_to_cache(cache_key, search_result)
            
            return search_result
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e
    
    def _keyword_search(self, query: str, filters: Optional[SearchFilters]) -> List[SearchResult]:
        parsed = self.query_parser.parse(query)

        # Treat '*' as "no text query" for filter-only browsing.
        is_wildcard = parsed.original.strip() == "*"

        params: list[object] = [self.conversations_glob]
        where_clause = self._where_from_filters(filters, params)

        text_conditions = []

        if not is_wildcard:
            for phrase in parsed.exact_phrases:
                text_conditions.append("full_text ILIKE '%' || ? || '%'")
                params.append(phrase)

            for term in parsed.must_include:
                text_conditions.append("full_text ILIKE '%' || ? || '%'")
                params.append(term)

            for term in parsed.should_include:
                clean_term = term.strip("'\"")
                if clean_term:
                    text_conditions.append("full_text ILIKE '%' || ? || '%'")
                    params.append(clean_term)

            for term in parsed.must_exclude:
                text_conditions.append("NOT (full_text ILIKE '%' || ? || '%')")
                params.append(term)

        if text_conditions:
            where_clause = f"{where_clause} AND " + " AND ".join(text_conditions)

        # Cap candidates to keep worst-case queries bounded.
        candidate_limit = 5000
        params.append(candidate_limit)

        con = self._connect()
        try:
            sql = f"""
            SELECT
              conversation_id,
              project_id,
              title,
              created_at,
              updated_at,
              message_count,
              file_path,
              full_text
            FROM parquet_scan(?)
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ?
            """

            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()

        if not rows:
            return []

        # Wildcard query: return most-recent conversations with a neutral score.
        if is_wildcard:
            search_results = []
            for (
                conversation_id,
                project_id,
                title,
                created_at,
                updated_at,
                message_count,
                file_path,
                full_text,
            ) in rows[:100]:
                search_results.append(
                    SearchResult(
                        conversation_id=conversation_id,
                        project_id=project_id,
                        title=title,
                        created_at=created_at,
                        updated_at=updated_at,
                        message_count=message_count,
                        file_path=file_path,
                        score=0.0,
                        snippet=(full_text or "")[:200],
                    )
                )
            return search_results

        # Use BM25 for scoring (industry standard)
        docs = [r[7] or "" for r in rows]
        corpus = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(corpus)

        # Tokenize query (combine all search terms)
        all_terms = parsed.exact_phrases + parsed.must_include + parsed.should_include
        query_tokens = ' '.join(all_terms).lower().split()

        # Calculate BM25 scores
        bm25_scores = bm25.get_scores(query_tokens)

        all_terms_lower = [t.lower() for t in all_terms]

        scored = []
        for idx, row in enumerate(rows):
            title = row[2] or ""
            title_l = title.lower()
            title_boost = 2.0 if any(term in title_l for term in all_terms_lower) else 1.0
            message_boost = float(np.log1p(row[5]))
            score = float(bm25_scores[idx]) * title_boost * message_boost
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        search_results = []
        for score, row in scored[:100]:
            (
                conversation_id,
                project_id,
                title,
                created_at,
                updated_at,
                message_count,
                file_path,
                full_text,
            ) = row
            search_results.append(
                SearchResult(
                    conversation_id=conversation_id,
                    project_id=project_id,
                    title=title,
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=message_count,
                    file_path=file_path,
                    score=score,
                    snippet=self._create_snippet(full_text or "", parsed.original),
                )
            )

        return search_results
    
    def _semantic_search(self, query: str, filters: Optional[SearchFilters]) -> List[SearchResult]:
        self.ensure_faiss_loaded()
        self.ensure_embedder_loaded()
        embedder = self.embedder
        faiss_index = self.faiss_index
        if embedder is None or faiss_index is None:
            raise RuntimeError("Search engine not initialized")

        query_embedding = np.asarray(embedder.encode(query), dtype=np.float32)
        
        k = 100
        # Use the stable Python binding signature: search(x, k) -> (D, I).
        # Some FAISS index wrappers don't expose the 4-arg C++ signature.
        distances, labels = faiss_index.search(query_embedding.reshape(1, -1), k)
        
        valid_mask = labels[0] >= 0
        hits = []
        for vector_id, distance, order in zip(labels[0][valid_mask], distances[0][valid_mask], np.arange(len(labels[0]))[valid_mask]):
            hits.append((int(vector_id), float(distance), int(order)))

        if not hits:
            return []

        values_clause = ", ".join(["(?, ?, ?)"] * len(hits))
        params: list[object] = []
        for vector_id, distance, order in hits:
            params.extend([vector_id, distance, order])

        # Metadata parquet scan, then conversations parquet scan
        params.append(str(self.metadata_path))
        params.append(self.conversations_glob)

        filter_params: list[object] = []
        where_clause = self._where_from_filters(filters, filter_params, table_alias="c")
        params.extend(filter_params)

        con = self._connect()
        try:
            sql = f"""
            WITH hits(vector_id, distance, faiss_order) AS (
              VALUES {values_clause}
            )
            SELECT
              m.conversation_id,
              c.project_id,
              c.title,
              c.created_at,
              c.updated_at,
              c.message_count,
              c.file_path,
              m.chunk_text,
              m.message_start_index,
              m.message_end_index,
              hits.distance,
              hits.faiss_order
            FROM hits
            JOIN parquet_scan(?) AS m
              ON m.vector_id = hits.vector_id
            JOIN parquet_scan(?) AS c
              ON c.conversation_id = m.conversation_id
            WHERE {where_clause}
            QUALIFY row_number() OVER (PARTITION BY m.conversation_id ORDER BY hits.faiss_order) = 1
            ORDER BY hits.faiss_order
            """
            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()

        if not rows:
            return []

        search_results = []
        for (
            conversation_id,
            project_id,
            title,
            created_at,
            updated_at,
            message_count,
            file_path,
            chunk_text,
            message_start_index,
            message_end_index,
            distance,
            _faiss_order,
        ) in rows:
            score = 1.0 / (1.0 + float(distance))
            snippet_text = chunk_text or ""
            snippet = snippet_text[:300] + ("..." if len(snippet_text) > 300 else "")

            search_results.append(
                SearchResult(
                    conversation_id=conversation_id,
                    project_id=project_id,
                    title=title,
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=message_count,
                    file_path=file_path,
                    score=score,
                    snippet=snippet,
                    message_start_index=int(message_start_index),
                    message_end_index=int(message_end_index),
                )
            )

        return search_results
    
    def _merge_results(
        self, 
        keyword: List[SearchResult], 
        semantic: List[SearchResult]
    ) -> List[SearchResult]:
        scores: Dict[str, float] = defaultdict(float)
        result_map: Dict[str, SearchResult] = {}
        
        # Normalize keyword scores
        if keyword:
            max_keyword_score = max(r.score for r in keyword) if keyword else 1.0
            min_keyword_score = min(r.score for r in keyword) if keyword else 0.0
            score_range = max_keyword_score - min_keyword_score if max_keyword_score != min_keyword_score else 1.0
            
            for rank, result in enumerate(keyword, 1):
                # Normalized score (0-1) with rank-based decay
                norm_score = (result.score - min_keyword_score) / score_range
                rank_weight = 1.0 / (1.0 + 0.1 * rank)  # Gentler decay than RRF
                weighted_score = norm_score * rank_weight * 0.6  # 60% weight for keyword
                
                scores[result.conversation_id] += weighted_score
                result_map[result.conversation_id] = result
        
        # Normalize semantic scores
        if semantic:
            max_semantic_score = max(r.score for r in semantic) if semantic else 1.0
            min_semantic_score = min(r.score for r in semantic) if semantic else 0.0
            score_range = max_semantic_score - min_semantic_score if max_semantic_score != min_semantic_score else 1.0
            
            for rank, result in enumerate(semantic, 1):
                # Normalized score (0-1) with rank-based decay
                norm_score = (result.score - min_semantic_score) / score_range
                rank_weight = 1.0 / (1.0 + 0.1 * rank)
                weighted_score = norm_score * rank_weight * 0.4  # 40% weight for semantic
                
                scores[result.conversation_id] += weighted_score
                if result.conversation_id not in result_map:
                    result_map[result.conversation_id] = result
        
        # Boost scores for results appearing in both
        for conv_id in scores:
            in_keyword = any(r.conversation_id == conv_id for r in keyword)
            in_semantic = any(r.conversation_id == conv_id for r in semantic)
            if in_keyword and in_semantic:
                scores[conv_id] *= 1.2  # 20% boost for appearing in both
        
        final_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        merged = []
        for conv_id, score in final_results[:50]:
            result = result_map[conv_id]
            result.score = score
            merged.append(result)
        
        return merged
    
    def _create_snippet(self, full_text: str, query: str, length: int = 200) -> str:
        text_lower = full_text.lower()
        
        # Parse the query to get search terms
        parsed = self.query_parser.parse(query)
        
        # Try to find exact phrases first
        best_pos = -1
        if parsed.exact_phrases:
            for phrase in parsed.exact_phrases:
                pos = text_lower.find(phrase.lower())
                if pos != -1:
                    best_pos = pos
                    break
        
        # If no exact phrase found, look for any term
        if best_pos == -1:
            all_terms = parsed.must_include + parsed.should_include
            if all_terms:
                # Find the position where most terms appear close together
                for term in all_terms:
                    pos = text_lower.find(term.lower())
                    if pos != -1:
                        best_pos = pos
                        break
        
        # If still nothing found, just use the beginning
        if best_pos == -1:
            return full_text[:length] + "..."
        
        # Create snippet centered around the match
        start = max(0, best_pos - length // 2)
        end = min(len(full_text), best_pos + length)
        
        snippet = full_text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(full_text):
            snippet = snippet + "..."
        
        return snippet
