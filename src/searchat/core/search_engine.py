import json
import time
import hashlib
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import duckdb
import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

from searchat.models import (
    SearchMode,
    SearchFilters,
    SearchResult,
    SearchResults,
    ParsedQuery
)
from searchat.core.query_parser import QueryParser
from searchat.config import Config


class SearchEngine:
    def __init__(self, search_dir: Path, config: Config = None):
        self.search_dir = search_dir
        self.faiss_index: Optional[faiss.Index] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.query_parser = QueryParser()
        
        if config is None:
            config = Config.load()
        self.config = config
        
        self.conversations_dir = self.search_dir / 'data/conversations'
        self.metadata_path = self.search_dir / 'data/indices/embeddings.metadata.parquet'
        
        # LRU cache for search results
        self.cache_size = config.performance.query_cache_size
        self.result_cache: OrderedDict[str, Tuple[SearchResults, float]] = OrderedDict()
        self.cache_ttl = 300  # 5 minutes TTL
        
        self._initialize()
    
    def _initialize(self) -> None:
        self._validate_index_metadata()
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata parquet not found at {self.metadata_path}. Run indexer first.")
        
        index_path = self.search_dir / 'data/indices/embeddings.faiss'
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run indexer first.")
        
        self.faiss_index = faiss.read_index(str(index_path))

        # Initialize embedder with GPU if available
        device = self.config.embedding.get_device()
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(self.config.embedding.model, device=device)

        # Load parquet files into memory
        self.metadata_df = pq.read_table(self.metadata_path).to_pandas()

        # Store parquet file paths for predicate pushdown queries
        self.conv_parquet_files = list(self.conversations_dir.glob('*.parquet'))
        if not self.conv_parquet_files:
            raise FileNotFoundError(f"No conversation parquet files found in {self.conversations_dir}")

        # Define columns needed for search (exclude large 'messages' column)
        self.search_columns = [
            'conversation_id', 'project_id', 'file_path', 'title',
            'created_at', 'updated_at', 'message_count', 'full_text',
            'embedding_id', 'file_hash', 'indexed_at'
        ]

        # Load conversations into memory with projection (exclude messages column)
        conv_tables = [pq.read_table(f, columns=self.search_columns) for f in self.conv_parquet_files]
        self.conversations_df = pd.concat([t.to_pandas() for t in conv_tables], ignore_index=True)
        # Add conversation_id index for O(1) lookups
        self.conversations_df = self.conversations_df.set_index('conversation_id', drop=False)
    
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

        # Apply filters (including the 0-message filter) - creates new DataFrame
        df = self._apply_filters(self.conversations_df, filters)
        df = df.reset_index(drop=True)
        
        # Apply search terms
        mask = pd.Series([True] * len(df), index=df.index)
        
        if parsed.exact_phrases:
            for phrase in parsed.exact_phrases:
                mask &= df['full_text'].str.contains(phrase, case=False, regex=False, na=False)
        
        if parsed.must_include:
            for term in parsed.must_include:
                mask &= df['full_text'].str.contains(term, case=False, regex=False, na=False)
        
        # For multiple words, require ALL words to be present (AND logic by default)
        if parsed.should_include:
            for term in parsed.should_include:
                # Clean up any stray quotes that might have slipped through
                clean_term = term.strip("'\"")
                if clean_term:  # Only search for non-empty terms
                    mask &= df['full_text'].str.contains(clean_term, case=False, regex=False, na=False)
        
        if parsed.must_exclude:
            for term in parsed.must_exclude:
                mask &= ~df['full_text'].str.contains(term, case=False, regex=False, na=False)

        results = df[mask]

        if results.empty:
            return []

        # Use BM25 for scoring (industry standard, much faster than hand-rolled)
        # Tokenize documents
        corpus = [doc.lower().split() for doc in results['full_text'].tolist()]
        bm25 = BM25Okapi(corpus)

        # Tokenize query (combine all search terms)
        all_terms = parsed.exact_phrases + parsed.must_include + parsed.should_include
        query_tokens = ' '.join(all_terms).lower().split()

        # Calculate BM25 scores
        bm25_scores = bm25.get_scores(query_tokens)

        # Title boost: multiply score by 2 if any query term appears in title
        title_boost = results['title'].str.lower().apply(
            lambda title: 2.0 if any(term.lower() in title for term in all_terms) else 1.0
        ).values

        # Message count boost: log scaling
        message_boost = np.log1p(results['message_count'].values)

        # Combined score
        results['relevance_score'] = bm25_scores * title_boost * message_boost

        results = results.sort_values('relevance_score', ascending=False)

        # Vectorized result conversion
        top_results = results.head(100)
        search_results = [
            SearchResult(
                conversation_id=row['conversation_id'],
                project_id=row['project_id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                message_count=row['message_count'],
                file_path=row['file_path'],
                score=float(row['relevance_score']),
                snippet=self._create_snippet(row['full_text'], parsed.original)
            )
            for row in top_results.to_dict('records')
        ]

        return search_results
    
    def _semantic_search(self, query: str, filters: Optional[SearchFilters]) -> List[SearchResult]:
        query_embedding = self.embedder.encode(query)
        
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            k=100
        )
        
        vector_ids = indices[0].tolist()
        vector_ids_filtered = [v for v in vector_ids if v >= 0]
        
        if not vector_ids_filtered:
            return []
        
        # Filter metadata by vector IDs
        metadata_matches = self.metadata_df[self.metadata_df['vector_id'].isin(vector_ids_filtered)]

        if metadata_matches.empty:
            return []

        # Join with conversations (handle overlapping columns)
        # Select only needed columns from conversations to avoid collisions
        # Use reset_index() to access both index and columns without ambiguity
        conv_df_for_merge = self.conversations_df.reset_index(drop=True).copy()
        conv_cols = ['conversation_id', 'project_id', 'title', 'created_at', 'updated_at',
                     'message_count', 'file_path', 'full_text']
        results = metadata_matches.merge(
            conv_df_for_merge[conv_cols],
            on='conversation_id',
            how='inner',
            suffixes=('', '_conv')
        )
        
        # Apply filters (including the 0-message filter)
        results = self._apply_filters(results, filters)
        
        if results.empty:
            return []
        
        # Build results maintaining FAISS order with single merge (not N+1 loop)
        # Create DataFrame with vector_id, distance, and FAISS order
        valid_mask = indices[0] >= 0
        vector_scores = pd.DataFrame({
            'vector_id': indices[0][valid_mask],
            'distance': distances[0][valid_mask],
            'faiss_order': np.arange(len(indices[0]))[valid_mask]
        })

        # Single merge instead of N lookups
        results_with_scores = results.merge(vector_scores, on='vector_id', how='inner')

        if results_with_scores.empty:
            return []

        # Sort by FAISS order, then deduplicate by conversation
        results_with_scores = results_with_scores.sort_values('faiss_order')
        results_with_scores = results_with_scores.drop_duplicates(subset=['conversation_id'], keep='first')

        # Calculate scores and build search results
        search_results = []
        for row in results_with_scores.to_dict('records'):
            score = 1.0 / (1.0 + float(row['distance']))

            try:
                snippet = row['chunk_text'][:300] + "..." if len(row['chunk_text']) > 300 else row['chunk_text']
            except KeyError as e:
                available_cols = list(row.keys())
                raise RuntimeError(f"Missing column 'chunk_text'. Available columns: {available_cols}. Vector ID: {row['vector_id']}") from e

            # Columns from metadata keep original names, conv columns get _conv suffix
            search_results.append(SearchResult(
                conversation_id=row['conversation_id'],
                project_id=row['project_id_conv'],
                title=row['title'],
                created_at=row['created_at_conv'],
                updated_at=row['updated_at'],
                message_count=row['message_count'],
                file_path=row['file_path'],
                score=score,
                snippet=snippet,
                message_start_index=int(row['message_start_index']),
                message_end_index=int(row['message_end_index'])
                ))
        
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
    
    def _query_with_predicate_pushdown(self, filters: Optional[SearchFilters]) -> pd.DataFrame:
        """
        Query parquet files with predicate and projection pushdown using DuckDB.
        - Predicate pushdown: Filters applied at parquet layer
        - Projection pushdown: Only load needed columns
        """
        # Build WHERE clause conditions
        conditions = ["message_count > 0"]  # Always exclude 0-message conversations

        if filters:
            if filters.project_ids:
                # Use IN clause for project filtering
                project_list = ", ".join([f"'{p}'" for p in filters.project_ids])
                conditions.append(f"project_id IN ({project_list})")

            if filters.date_from:
                conditions.append(f"updated_at >= '{filters.date_from.isoformat()}'")

            if filters.date_to:
                conditions.append(f"updated_at <= '{filters.date_to.isoformat()}'")

            if filters.min_messages > 0:
                conditions.append(f"message_count >= {filters.min_messages}")

        where_clause = " AND ".join(conditions)

        # Build column list (exclude large 'messages' column)
        columns = ", ".join(self.search_columns)

        # Build SQL query with both predicate and projection pushdown
        parquet_pattern = str(self.conversations_dir / '*.parquet').replace('\\', '/')
        query = f"""
            SELECT {columns}
            FROM read_parquet('{parquet_pattern}')
            WHERE {where_clause}
        """

        # Execute query with DuckDB (only loads matching rows + needed columns)
        result_df = duckdb.query(query).to_df()

        # Set conversation_id as index for O(1) lookups
        if not result_df.empty:
            result_df = result_df.set_index('conversation_id', drop=False)

        return result_df

    def _apply_filters(self, df: pd.DataFrame, filters: SearchFilters) -> pd.DataFrame:
        """
        Apply search filters to a DataFrame.
        Uses predicate pushdown when filters present for better performance.
        """
        # If filters are present, use predicate pushdown for efficiency
        has_filters = (
            filters and (
                filters.project_ids or
                filters.date_from or
                filters.date_to or
                filters.min_messages > 0
            )
        )

        if has_filters:
            # Use DuckDB predicate pushdown - only loads matching rows
            return self._query_with_predicate_pushdown(filters)
        else:
            # No filters except message_count > 0, filter in-memory
            return df[df['message_count'] > 0]
    
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
