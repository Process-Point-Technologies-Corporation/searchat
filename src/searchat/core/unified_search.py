"""Unified search engine using DuckDB with native vector and full-text search.

This module provides search capabilities using the unified DuckDB database:
- BM25 keyword search via FTS extension
- Semantic search via VSS HNSW indexes
- Hybrid search combining both with configurable weights

Compatible API with existing SearchEngine for easy migration.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from searchat.config import Config
from searchat.core.normalize import normalize_score, percentile_normalize
from searchat.core.query_parser import QueryParser
from searchat.core.query_classifier import QueryClassifier
from searchat.core.unified_storage import UnifiedStorage
from searchat.models.domain import SearchFilters, SearchResult, SearchResults
from searchat.models.enums import SearchMode, AlgorithmType

logger = logging.getLogger(__name__)


class UnifiedSearchEngine:
    """Search engine using unified DuckDB storage with VSS and FTS."""

    def __init__(
        self,
        search_dir: Path,
        config: Optional[Config] = None,
        storage: Optional[UnifiedStorage] = None,
        embedder: Optional[SentenceTransformer] = None,
    ):
        """Initialize the unified search engine.

        Args:
            search_dir: Root directory for searchat data (~/.searchat)
            config: Configuration object (loads default if None)
            storage: Optional pre-existing UnifiedStorage (for testing)
            embedder: Optional shared SentenceTransformer instance
        """
        self.search_dir = search_dir
        self.data_dir = search_dir / "data"

        if config is None:
            config = Config.load()
        self.config = config

        if storage is not None:
            self.storage = storage
            self._external_storage = True
        else:
            self.storage = UnifiedStorage(self.data_dir)
            self._external_storage = False

        # Initialize embedder (use shared instance if provided)
        if embedder is not None:
            self.embedder = embedder
        else:
            from sentence_transformers import SentenceTransformer
            device = config.embedding.get_device()
            self.embedder = SentenceTransformer(config.embedding.model, device=device)

        # Query parser
        self.query_parser = QueryParser()

        # Query classifier for adaptive hybrid weights
        self.query_classifier = QueryClassifier()

        # LRU cache for search results
        self.cache_size = config.performance.query_cache_size
        self.result_cache: OrderedDict[str, Tuple[SearchResults, float]] = OrderedDict()
        self.cache_ttl = 300  # 5 minutes

    def search(
        self,
        query: str,
        algorithm: AlgorithmType = AlgorithmType.SEMANTIC,
        filters: Optional[SearchFilters] = None,
        limit: int = 50,
    ) -> SearchResults:
        """Search conversations with exchange-level granularity.

        Args:
            query: Search query string
            algorithm: Algorithm type (KEYWORD, SEMANTIC, HYBRID)
            filters: Optional search filters (project, date, etc.)
            limit: Maximum results to return

        Returns:
            SearchResults with exchange-level matches
        """
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(query, algorithm, filters, limit)
        cached = self._get_from_cache(cache_key)
        if cached:
            cached.search_time_ms = (time.time() - start_time) * 1000
            return cached

        try:
            if algorithm == AlgorithmType.CROSS_LAYER:
                results = self._cross_layer_search(query, filters, limit)
            elif algorithm == AlgorithmType.DISTILL:
                results = self._distill_search(query, filters, limit)
            elif algorithm == AlgorithmType.HYBRID:
                results = self._hybrid_search(query, filters, limit)
            elif algorithm == AlgorithmType.KEYWORD:
                results = self._keyword_search(query, filters, limit)
            else:
                results = self._semantic_search(query, filters, limit)

            elapsed_ms = (time.time() - start_time) * 1000

            search_result = SearchResults(
                results=results,
                total_count=len(results),
                search_time_ms=elapsed_ms,
                mode_used=algorithm.value,
            )

            self._add_to_cache(cache_key, search_result)
            return search_result

        except Exception as e:
            logger.error("Search failed: %s", e)
            raise RuntimeError(f"Search failed: {e}") from e

    def _keyword_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[SearchResult]:
        """BM25 keyword search over exchanges."""
        parsed = self.query_parser.parse(query)

        # Get project filter
        project_ids = filters.project_ids if filters else None

        # Search exchanges
        results = self.storage.search_verbatim_bm25(
            query=query,
            limit=limit * 2,  # Get more for deduplication
            project_ids=project_ids,
        )

        if not results:
            return []

        # Apply additional filters
        results = self._apply_filters(results, filters, parsed)

        # Deduplicate by conversation, keeping best exchange per conversation
        seen_convs: Dict[str, Dict] = {}
        for r in results:
            conv_id = r["conversation_id"]
            if conv_id not in seen_convs or r["score"] > seen_convs[conv_id]["score"]:
                seen_convs[conv_id] = r

        # Convert to SearchResult objects
        search_results = []
        for r in sorted(seen_convs.values(), key=lambda x: x["score"], reverse=True)[:limit]:
            search_results.append(self._to_search_result(r, query, parsed))

        return search_results

    def _semantic_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[SearchResult]:
        """Semantic search over exchange embeddings."""
        # Generate query embedding
        query_embedding = self.embedder.encode(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Get project filter
        project_ids = filters.project_ids if filters else None

        # Search exchanges
        results = self.storage.search_verbatim_semantic(
            query_embedding=query_embedding,
            limit=limit * 2,
            project_ids=project_ids,
        )

        if not results:
            return []

        # Apply filters
        parsed = self.query_parser.parse(query)
        results = self._apply_filters(results, filters, parsed)

        # Deduplicate by conversation
        seen_convs: Dict[str, Dict] = {}
        for r in results:
            conv_id = r["conversation_id"]
            if conv_id not in seen_convs or r["score"] > seen_convs[conv_id]["score"]:
                seen_convs[conv_id] = r

        # Convert to SearchResult objects
        search_results = []
        for r in sorted(seen_convs.values(), key=lambda x: x["score"], reverse=True)[:limit]:
            search_results.append(self._to_search_result(r, query, parsed))

        return search_results

    def _hybrid_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[SearchResult]:
        """Hybrid search combining BM25 keyword and semantic search."""
        parsed = self.query_parser.parse(query)
        project_ids = filters.project_ids if filters else None

        # Get keyword results
        keyword_results = self.storage.search_verbatim_bm25(
            query=query,
            limit=limit * 2,
            project_ids=project_ids,
        )

        # Get semantic results
        query_embedding = self.embedder.encode(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        semantic_results = self.storage.search_verbatim_semantic(
            query_embedding=query_embedding,
            limit=limit * 2,
            project_ids=project_ids,
        )

        # Apply filters to both
        keyword_results = self._apply_filters(keyword_results, filters, parsed)
        semantic_results = self._apply_filters(semantic_results, filters, parsed)

        # Merge results
        merged = self._merge_results(keyword_results, semantic_results, query, limit)
        return merged

    def _adaptive_hybrid_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[SearchResult]:
        """Adaptive hybrid search with query-specific weight adjustment.

        Classifies the query into keyword/semantic/balanced categories and
        adjusts BM25/semantic weights accordingly:
        - Keyword queries (file paths, identifiers): BM25=0.8, semantic=0.2
        - Semantic queries (conceptual terms): BM25=0.2, semantic=0.8
        - Balanced queries: BM25=0.5, semantic=0.5

        Logs classification decision for analysis.
        """
        parsed = self.query_parser.parse(query)
        project_ids = filters.project_ids if filters else None

        # Classify query to determine optimal weights
        classification = self.query_classifier.classify(query)

        logger.info(
            "Adaptive search: query='%s' classified as %s (confidence=%.2f, reason: %s) → weights=(bm25=%.1f, sem=%.1f)",
            query[:100],
            classification.query_type,
            classification.confidence,
            classification.reasoning,
            classification.bm25_weight,
            classification.semantic_weight,
        )

        # Get keyword results
        keyword_results = self.storage.search_verbatim_bm25(
            query=query,
            limit=limit * 2,
            project_ids=project_ids,
        )

        # Get semantic results
        query_embedding = self.embedder.encode(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        semantic_results = self.storage.search_verbatim_semantic(
            query_embedding=query_embedding,
            limit=limit * 2,
            project_ids=project_ids,
        )

        # Apply filters to both
        keyword_results = self._apply_filters(keyword_results, filters, parsed)
        semantic_results = self._apply_filters(semantic_results, filters, parsed)

        # Merge results with adaptive weights
        merged = self._merge_results_with_weights(
            keyword_results,
            semantic_results,
            query,
            limit,
            keyword_weight=classification.bm25_weight,
            semantic_weight=classification.semantic_weight,
        )
        return merged

    def _hierarchical_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[SearchResult]:
        """Hierarchical facet-based search with automatic project scoping.

        Uses hierarchical facets (full path, directory, basename) with weighted
        voting to resolve query to most likely project, then performs scoped
        hybrid search within that project.

        Logs facet resolution details and which facet level matched.
        """
        from searchat.palace.hierarchical_facets import HierarchicalFacetResolver

        parsed = self.query_parser.parse(query)

        # Attempt hierarchical facet resolution
        resolver = HierarchicalFacetResolver(self.storage, self.embedder)
        resolved_projects, metadata = resolver.resolve(
            query,
            top_k=10,
            confidence_threshold=0.6,
            max_project_count=3,
        )

        # Merge with explicit project filters if provided
        if filters and filters.project_ids:
            if resolved_projects:
                # Intersect resolved with explicit filters
                resolved_set = set(resolved_projects)
                filter_set = set(filters.project_ids)
                final_projects = list(resolved_set & filter_set)
                if not final_projects:
                    logger.warning(
                        "Hierarchical resolution (%s) conflicts with filters (%s), using filters",
                        resolved_projects,
                        filters.project_ids,
                    )
                    final_projects = filters.project_ids
            else:
                final_projects = filters.project_ids
        else:
            final_projects = resolved_projects

        # Log resolution details
        if resolved_projects:
            facet_details = metadata.get("facets", [])
            top_3_facets = facet_details[:3]
            facet_summary = ", ".join(
                f"{f['text']} (level={f['level']}, score={f['combined_score']:.2f})"
                for f in top_3_facets
            )
            logger.info(
                "Hierarchical search: '%s' → projects=%s, reason='%s', top_facets=[%s]",
                query[:100],
                resolved_projects,
                metadata.get("reason", ""),
                facet_summary,
            )

        # Perform scoped hybrid search
        project_ids = final_projects

        # Get keyword results
        keyword_results = self.storage.search_verbatim_bm25(
            query=query,
            limit=limit * 2,
            project_ids=project_ids,
        )

        # Get semantic results
        query_embedding = self.embedder.encode(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        semantic_results = self.storage.search_verbatim_semantic(
            query_embedding=query_embedding,
            limit=limit * 2,
            project_ids=project_ids,
        )

        # Apply filters to both
        keyword_results = self._apply_filters(keyword_results, filters, parsed)
        semantic_results = self._apply_filters(semantic_results, filters, parsed)

        # Merge results
        merged = self._merge_results(keyword_results, semantic_results, query, limit)

        # Annotate results with hierarchical metadata
        for result in merged:
            result.search_metadata = {
                "facet_resolution": metadata if resolved_projects else None,
                "scoped_to_projects": final_projects,
            }

        return merged

    def _cross_layer_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[SearchResult]:
        """CombMNZ fusion of BM25-FTS on verbatim + HNSW on distilled palace objects.

        Combines verbatim keyword search with palace semantic search using
        CombMNZ: score = sum(norm_scores) * count(nonzero_signals).
        """
        project_ids = filters.project_ids if filters else None

        # Get verbatim BM25 results
        verbatim_results = self.storage.search_verbatim_bm25(
            query=query, limit=limit * 2, project_ids=project_ids,
        )

        # Get palace semantic results
        query_embedding = self.embedder.encode(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        palace_results = self.storage.search_palace_semantic(
            query_embedding=query_embedding, limit=limit * 2, project_ids=project_ids,
        )

        if not verbatim_results and not palace_results:
            return []

        # Normalize verbatim scores independently
        v_scores: Dict[str, float] = {}
        v_data: Dict[str, Dict] = {}
        if verbatim_results:
            raw_scores = [float(r["score"]) for r in verbatim_results]
            divisor = percentile_normalize(raw_scores)
            for r in verbatim_results:
                eid = r["exchange_id"]
                v_scores[eid] = normalize_score(float(r["score"]), divisor)
                v_data[eid] = r

        # Normalize palace scores independently
        p_scores: Dict[str, float] = {}
        p_data: Dict[str, Dict] = {}
        if palace_results:
            raw_scores = [float(r["score"]) for r in palace_results]
            divisor = percentile_normalize(raw_scores)
            for r in palace_results:
                eid = r["exchange_id"]
                p_scores[eid] = normalize_score(float(r["score"]), divisor)
                p_data[eid] = r

        # CombMNZ fusion: score = sum(norm_scores) * count(nonzero_signals)
        all_eids = set(v_scores) | set(p_scores)
        combined: Dict[str, float] = {}
        for eid in all_eids:
            v_norm = v_scores.get(eid, 0.0)
            p_norm = p_scores.get(eid, 0.0)
            nonzero_count = (1 if v_norm > 0 else 0) + (1 if p_norm > 0 else 0)
            combined[eid] = (v_norm + p_norm) * nonzero_count

        # Enrich palace-only results with conversation metadata
        palace_only_eids = set(p_data) - set(v_data)
        if palace_only_eids:
            self._enrich_with_conversation_metadata(
                [p_data[eid] for eid in palace_only_eids]
            )

        # Apply filters
        parsed = self.query_parser.parse(query)

        # Deduplicate by conversation_id (keep best exchange per conversation)
        seen_convs: Dict[str, tuple] = {}
        for eid in sorted(combined, key=lambda e: combined[e], reverse=True):
            # Prefer verbatim data (has full metadata), fall back to palace
            data = v_data.get(eid) or p_data.get(eid)

            # Apply filters inline
            if filters:
                if filters.date_from:
                    updated_at = data.get("updated_at")
                    if updated_at and updated_at < filters.date_from:
                        continue
                if filters.date_to:
                    updated_at = data.get("updated_at")
                    if updated_at and updated_at > filters.date_to:
                        continue

            conv_id = data["conversation_id"]
            if conv_id not in seen_convs:
                seen_convs[conv_id] = (eid, combined[eid], data)

        # Build SearchResult objects
        search_results = []
        sorted_convs = sorted(seen_convs.values(), key=lambda x: x[1], reverse=True)[:limit]
        for eid, score, data in sorted_convs:
            result = self._to_search_result(data, query, parsed)
            result.score = score
            result.bm25_score = v_scores.get(eid)
            result.semantic_score = p_scores.get(eid)
            # Attach palace metadata if this exchange has palace data
            if eid in p_data:
                pd = p_data[eid]
                result.palace_summary = pd.get("exchange_core")
                result.palace_context = pd.get("specific_context")
                result.files_touched_raw = pd.get("files_touched")
                result.object_id = pd.get("object_id")
            search_results.append(result)

        return search_results

    def _distill_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[SearchResult]:
        """Pure semantic search on distilled palace objects."""
        project_ids = filters.project_ids if filters else None

        # Encode query and search palace
        query_embedding = self.embedder.encode(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        palace_results = self.storage.search_palace_semantic(
            query_embedding=query_embedding, limit=limit * 2, project_ids=project_ids,
        )

        if not palace_results:
            return []

        # Deduplicate by conversation_id BEFORE enriching (avoid fetching metadata for discarded results)
        seen_convs: Dict[str, Dict] = {}
        for r in palace_results:
            conv_id = r["conversation_id"]
            if conv_id not in seen_convs or r["score"] > seen_convs[conv_id]["score"]:
                seen_convs[conv_id] = r

        deduped = sorted(seen_convs.values(), key=lambda x: x["score"], reverse=True)[:limit]

        # Enrich only the results we'll return with conversation metadata
        self._enrich_with_conversation_metadata(deduped)

        # Apply date filters after enrichment (needs updated_at from metadata)
        filtered = []
        for r in deduped:
            if filters:
                if filters.date_from:
                    updated_at = r.get("updated_at")
                    if updated_at and updated_at < filters.date_from:
                        continue
                if filters.date_to:
                    updated_at = r.get("updated_at")
                    if updated_at and updated_at > filters.date_to:
                        continue
            filtered.append(r)

        # Build SearchResult objects
        search_results = []
        for r in filtered:
            result = self._to_search_result(r, query)
            result.semantic_score = r["score"]
            result.palace_summary = r.get("exchange_core")
            result.palace_context = r.get("specific_context")
            result.files_touched_raw = r.get("files_touched")
            result.object_id = r.get("object_id")
            search_results.append(result)

        return search_results

    def _enrich_with_conversation_metadata(self, palace_results: List[Dict]) -> None:
        """Batch-enrich palace result dicts with conversation metadata.

        Palace results from search_palace_semantic lack title, file_path,
        message_count, and updated_at. This method patches them in from
        the conversations table.
        """
        conv_ids = list({r["conversation_id"] for r in palace_results})
        if not conv_ids:
            return

        rows = self.storage._get_read_cursor().execute("""
            SELECT conversation_id, title, file_path, message_count, updated_at
            FROM conversations
            WHERE conversation_id IN (SELECT UNNEST(?::VARCHAR[]))
        """, [conv_ids]).fetchall()

        metadata_map = {
            row[0]: {
                "title": row[1],
                "file_path": row[2],
                "message_count": row[3],
                "updated_at": row[4],
            }
            for row in rows
        }

        for r in palace_results:
            meta = metadata_map.get(r["conversation_id"], {})
            r.setdefault("title", meta.get("title", "Untitled"))
            r.setdefault("file_path", meta.get("file_path", ""))
            r.setdefault("message_count", meta.get("message_count", 0))
            r.setdefault("updated_at", meta.get("updated_at"))

    def _merge_results(
        self,
        keyword_results: List[Dict],
        semantic_results: List[Dict],
        query: str,
        limit: int,
    ) -> List[SearchResult]:
        """Merge keyword and semantic results with score fusion.

        Uses configurable weights and intersection boost from config.
        """
        ranking = self.config.search.ranking
        return self._merge_results_with_weights(
            keyword_results,
            semantic_results,
            query,
            limit,
            keyword_weight=ranking.keyword_weight,
            semantic_weight=ranking.semantic_weight,
        )

    def _merge_results_with_weights(
        self,
        keyword_results: List[Dict],
        semantic_results: List[Dict],
        query: str,
        limit: int,
        keyword_weight: float,
        semantic_weight: float,
    ) -> List[SearchResult]:
        """Merge keyword and semantic results with custom weights.

        Args:
            keyword_results: BM25 search results
            semantic_results: Semantic search results
            query: Original search query
            limit: Maximum results to return
            keyword_weight: Weight for BM25 scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)

        Returns:
            Merged and deduplicated search results
        """
        ranking = self.config.search.ranking
        intersection_boost = ranking.boost_multiplier
        rank_decay = ranking.rank_decay
        title_boost = ranking.title_boost

        # Build score maps by exchange_id
        keyword_scores: Dict[str, float] = {}
        semantic_scores: Dict[str, float] = {}
        exchange_data: Dict[str, Dict] = {}

        # Normalize keyword scores
        if keyword_results:
            kw_raw_scores = [float(r["score"]) for r in keyword_results]
            kw_divisor = percentile_normalize(kw_raw_scores)

            for rank, r in enumerate(keyword_results, 1):
                exchange_id = r["exchange_id"]
                norm_score = normalize_score(float(r["score"]), kw_divisor)
                rank_weight = 1.0 / (1.0 + rank_decay * rank)
                keyword_scores[exchange_id] = norm_score * rank_weight
                exchange_data[exchange_id] = r

        # Normalize semantic scores
        if semantic_results:
            sem_raw_scores = [float(r["score"]) for r in semantic_results]
            sem_divisor = percentile_normalize(sem_raw_scores)

            for rank, r in enumerate(semantic_results, 1):
                exchange_id = r["exchange_id"]
                norm_score = normalize_score(float(r["score"]), sem_divisor)
                rank_weight = 1.0 / (1.0 + rank_decay * rank)
                semantic_scores[exchange_id] = norm_score * rank_weight
                if exchange_id not in exchange_data:
                    exchange_data[exchange_id] = r

        # Combine scores
        all_exchange_ids = set(keyword_scores.keys()) | set(semantic_scores.keys())
        combined_scores: Dict[str, Tuple[float, float, float]] = {}
        query_terms = query.lower().split()

        for exchange_id in all_exchange_ids:
            kw_score = keyword_scores.get(exchange_id, 0.0)
            sem_score = semantic_scores.get(exchange_id, 0.0)

            base_score = keyword_weight * kw_score + semantic_weight * sem_score

            # Intersection boost
            if exchange_id in keyword_scores and exchange_id in semantic_scores:
                base_score *= intersection_boost

            # Title boost
            data = exchange_data[exchange_id]
            title = data.get("title", "").lower()
            if any(term in title for term in query_terms):
                base_score *= title_boost

            combined_scores[exchange_id] = (base_score, kw_score, sem_score)

        # Deduplicate by conversation, keeping best exchange
        conv_best: Dict[str, Tuple[str, float, float, float]] = {}
        for exchange_id, (score, kw, sem) in combined_scores.items():
            data = exchange_data[exchange_id]
            conv_id = data["conversation_id"]
            if conv_id not in conv_best or score > conv_best[conv_id][1]:
                conv_best[conv_id] = (exchange_id, score, kw, sem)

        # Sort and build results
        sorted_results = sorted(
            conv_best.items(),
            key=lambda x: x[1][1],  # Sort by score
            reverse=True,
        )[:limit]

        search_results = []
        for conv_id, (exchange_id, score, kw_score, sem_score) in sorted_results:
            data = exchange_data[exchange_id]
            result = self._to_search_result(data, query)
            result.score = score
            result.bm25_score = kw_score if kw_score > 0 else None
            result.semantic_score = sem_score if sem_score > 0 else None
            search_results.append(result)

        return search_results

    def _apply_filters(
        self,
        results: List[Dict],
        filters: Optional[SearchFilters],
        parsed,
    ) -> List[Dict]:
        """Apply search filters to results."""
        if not filters and not parsed.exact_phrases and not parsed.must_exclude:
            return results

        filtered = []
        for r in results:
            # Date filters
            if filters:
                if filters.date_from:
                    updated_at = r.get("updated_at")
                    if updated_at and updated_at < filters.date_from:
                        continue
                if filters.date_to:
                    updated_at = r.get("updated_at")
                    if updated_at and updated_at > filters.date_to:
                        continue
                if filters.min_messages > 0:
                    msg_count = r.get("message_count", 0)
                    if msg_count < filters.min_messages:
                        continue

            # Exact phrase matching
            exchange_text = r.get("exchange_text", "").lower()
            if parsed.exact_phrases:
                if not all(phrase.lower() in exchange_text for phrase in parsed.exact_phrases):
                    continue

            # Must exclude
            if parsed.must_exclude:
                if any(term.lower() in exchange_text for term in parsed.must_exclude):
                    continue

            filtered.append(r)

        return filtered

    def _resolve_facets(self, query: str, top_k: int = 5) -> Optional[List[str]]:
        """Resolve query to project_ids via semantic facet matching.

        CURRENT BEHAVIOR: Winner-takes-all (top-1 facet only).
        Returns list of project_ids if distinctive facets found, None otherwise.
        """
        if not self.storage._vss_available:
            return None

        # Embed query
        query_emb = self.embedder.encode(query)
        query_emb = np.array(query_emb, dtype=np.float32)

        # Search facet_embeddings
        results = self.storage.search_facet_embeddings(
            query_emb,
            limit=top_k,
            max_project_count=3,  # Only distinctive facets
        )

        if not results:
            return None

        # Return project_ids from top match (winner-takes-all)
        project_ids = results[0]["project_ids"]
        return project_ids

    def _resolve_facets_weighted(
        self,
        query: str,
        top_k: int = 5,
        confidence_threshold: float = 0.6,
    ) -> Optional[List[str]]:
        """Resolve query to project_ids via weighted voting across top-K facets.

        IMPROVEMENT OVER _resolve_facets: Uses weighted voting instead of winner-takes-all.

        Given top-5 facet results:
        - parse_sutra.py (0.94) → sanskrit0
        - portfolio_optimizer.py (0.92) → quant
        - options_pricing.py (0.90) → quant
        - risk_metrics.py (0.88) → quant

        Winner-takes-all incorrectly resolves to sanskrit0 (1/4 votes).
        Weighted voting correctly resolves to quant (0.92 + 0.90 + 0.88 = 2.70 vs 0.94).

        Args:
            query: Search query to resolve
            top_k: Number of top facets to consider for voting
            confidence_threshold: Minimum vote share for winner (0.6 = 60%)

        Returns:
            List containing single project_id if confident resolution found, None otherwise.

        Logs:
            - INFO: Successful resolution with vote breakdown
            - INFO: Low confidence resolution (below threshold)
            - DEBUG: Facet details and vote accumulation
        """
        if not self.storage._vss_available:
            return None

        # Embed query
        query_emb = self.embedder.encode(query)
        query_emb = np.array(query_emb, dtype=np.float32)

        # Search facet_embeddings
        results = self.storage.search_facet_embeddings(
            query_emb,
            limit=top_k,
            max_project_count=3,  # Only distinctive facets
        )

        if not results:
            logger.debug("Weighted facet resolution: no distinctive facets found for query '%s'", query)
            return None

        # Accumulate weighted votes per project
        votes: Dict[str, float] = defaultdict(float)

        for i, facet in enumerate(results):
            facet_text = facet["facet_text"]
            facet_type = facet["facet_type"]
            similarity = facet["score"]  # Already converted from distance (1/(1+d))
            project_ids = facet["project_ids"]

            logger.debug(
                "Weighted facet resolution [%d/%d]: %s (%s) score=%.4f projects=%s",
                i + 1,
                len(results),
                facet_text,
                facet_type,
                similarity,
                project_ids,
            )

            # Distribute vote weight to all projects in this facet
            for pid in project_ids:
                votes[pid] += similarity

        if not votes:
            logger.debug("Weighted facet resolution: no project votes accumulated")
            return None

        # Find winner
        total_votes = sum(votes.values())
        winner_pid, winner_votes = max(votes.items(), key=lambda x: x[1])
        confidence = winner_votes / total_votes if total_votes > 0 else 0.0

        # Format vote breakdown for logging
        vote_breakdown = ", ".join(
            f"{pid}={v:.2f}" for pid, v in sorted(votes.items(), key=lambda x: x[1], reverse=True)
        )

        if confidence >= confidence_threshold:
            logger.info(
                "Weighted facet resolution: '%s' → %s (confidence=%.1f%%, votes: %s)",
                query,
                winner_pid,
                confidence * 100,
                vote_breakdown,
            )
            return [winner_pid]
        else:
            logger.info(
                "Weighted facet resolution: '%s' rejected (confidence=%.1f%% < %.1f%%, votes: %s)",
                query,
                confidence * 100,
                confidence_threshold * 100,
                vote_breakdown,
            )
            return None

    def _resolve_facets_temporal(
        self,
        query: str,
        top_k: int = 5,
        confidence_threshold: float = 0.6,
        decay_rate: float = 0.01,
    ) -> Optional[List[str]]:
        """Resolve query to project_ids using temporal decay weighted voting.

        IMPROVEMENT: Applies exponential decay to facet scores based on recency.
        Recent facets are weighted higher than old facets.

        Decay formula: weight = base_score * exp(-decay_rate * days_old)
        Default decay_rate=0.01 means:
        - Today: weight = 1.0
        - 30 days old: weight = 0.74
        - 90 days old: weight = 0.41
        - 180 days old: weight = 0.17

        Args:
            query: Search query to resolve
            top_k: Number of top facets to consider for voting
            confidence_threshold: Minimum vote share for winner (0.6 = 60%)
            decay_rate: Exponential decay coefficient (default 0.01)

        Returns:
            List containing single project_id if confident resolution found, None otherwise.

        Logs:
            - INFO: Successful resolution with vote breakdown and decay factors
            - DEBUG: Per-facet decay calculations
        """
        if not self.storage._vss_available:
            return None

        # Embed query
        query_emb = self.embedder.encode(query)
        query_emb = np.array(query_emb, dtype=np.float32)

        # Search facet_embeddings with temporal decay
        results = self.storage.search_facet_embeddings(
            query_emb,
            limit=top_k,
            max_project_count=3,
            apply_temporal_decay=True,
            decay_rate=decay_rate,
        )

        if not results:
            logger.debug("Temporal facet resolution: no distinctive facets found for query '%s'", query)
            return None

        # Accumulate weighted votes per project using temporal_score
        votes: Dict[str, float] = defaultdict(float)

        for i, facet in enumerate(results):
            facet_text = facet["facet_text"]
            facet_type = facet["facet_type"]
            base_score = facet["score"]
            temporal_score = facet["temporal_score"]
            decay_factor = facet["decay_factor"]
            last_seen = facet.get("last_seen")
            project_ids = facet["project_ids"]

            logger.debug(
                "Temporal facet resolution [%d/%d]: %s (%s) base=%.4f temporal=%.4f decay=%.4f last_seen=%s projects=%s",
                i + 1,
                len(results),
                facet_text,
                facet_type,
                base_score,
                temporal_score,
                decay_factor,
                last_seen.isoformat() if last_seen else "unknown",
                project_ids,
            )

            # Distribute temporal-weighted vote to all projects in this facet
            for pid in project_ids:
                votes[pid] += temporal_score

        if not votes:
            logger.debug("Temporal facet resolution: no project votes accumulated")
            return None

        # Find winner
        total_votes = sum(votes.values())
        winner_pid, winner_votes = max(votes.items(), key=lambda x: x[1])
        confidence = winner_votes / total_votes if total_votes > 0 else 0.0

        # Format vote breakdown for logging
        vote_breakdown = ", ".join(
            f"{pid}={v:.2f}" for pid, v in sorted(votes.items(), key=lambda x: x[1], reverse=True)
        )

        if confidence >= confidence_threshold:
            logger.info(
                "Temporal facet resolution: '%s' → %s (confidence=%.1f%%, decay_rate=%.3f, votes: %s)",
                query,
                winner_pid,
                confidence * 100,
                decay_rate,
                vote_breakdown,
            )
            return [winner_pid]
        else:
            logger.info(
                "Temporal facet resolution: '%s' rejected (confidence=%.1f%% < %.1f%%, votes: %s)",
                query,
                confidence * 100,
                confidence_threshold * 100,
                vote_breakdown,
            )
            return None

    def _temporal_decay_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[SearchResult]:
        """Hybrid search with temporal decay facet resolution.

        Uses temporal decay to resolve facets (recent facets weighted higher),
        then performs hybrid search scoped to resolved projects.

        Decay rate: 0.01 (configurable via future config option)
        """
        parsed = self.query_parser.parse(query)

        # Attempt temporal facet resolution
        resolved_projects = self._resolve_facets_temporal(
            query,
            top_k=5,
            confidence_threshold=0.6,
            decay_rate=0.01,
        )

        # Override filters with resolved projects if found
        scoped_filters = filters
        if resolved_projects:
            if filters is None:
                scoped_filters = SearchFilters(project_ids=list(resolved_projects))
            else:
                scoped_filters = replace(filters, project_ids=list(resolved_projects))
            logger.info(
                "Temporal decay search: scoping to projects %s via facet resolution",
                resolved_projects,
            )

        # Perform hybrid search with resolved project scope
        project_ids = scoped_filters.project_ids if scoped_filters else None

        # Get keyword results
        keyword_results = self.storage.search_verbatim_bm25(
            query=query,
            limit=limit * 2,
            project_ids=project_ids,
        )

        # Get semantic results
        query_embedding = self.embedder.encode(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        semantic_results = self.storage.search_verbatim_semantic(
            query_embedding=query_embedding,
            limit=limit * 2,
            project_ids=project_ids,
        )

        # Apply filters to both
        keyword_results = self._apply_filters(keyword_results, scoped_filters, parsed)
        semantic_results = self._apply_filters(semantic_results, scoped_filters, parsed)

        # Merge results with standard hybrid weights
        merged = self._merge_results(keyword_results, semantic_results, query, limit)
        return merged

    def _to_search_result(self, data: Dict, query: str, parsed=None) -> SearchResult:
        """Convert storage result dict to SearchResult."""
        exchange_text = data.get("exchange_text", "")
        if parsed is None:
            parsed = self.query_parser.parse(query)
        snippet = self._create_snippet(exchange_text, parsed)

        return SearchResult(
            conversation_id=data["conversation_id"],
            project_id=data["project_id"],
            title=data.get("title", "Untitled"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            message_count=data.get("message_count", 0),
            file_path=data.get("file_path", ""),
            score=data.get("score", 0.0),
            snippet=snippet,
            message_start_index=data.get("ply_start"),
            message_end_index=data.get("ply_end"),
            exchange_id=data.get("exchange_id"),
            exchange_text=exchange_text,
        )

    def _create_snippet(self, text: str, parsed, length: int = 200) -> str:
        """Create a query-centered snippet from text."""
        if not text:
            return ""

        text_lower = text.lower()

        # Find best position
        best_pos = -1

        # Try exact phrases first
        if parsed.exact_phrases:
            for phrase in parsed.exact_phrases:
                pos = text_lower.find(phrase.lower())
                if pos != -1:
                    best_pos = pos
                    break

        # Try other terms
        if best_pos == -1:
            all_terms = parsed.must_include + parsed.should_include
            if all_terms:
                for term in all_terms:
                    pos = text_lower.find(term.lower())
                    if pos != -1:
                        best_pos = pos
                        break

        # Fallback to beginning
        if best_pos == -1:
            return text[:length] + ("..." if len(text) > length else "")

        # Create centered snippet
        start = max(0, best_pos - length // 2)
        end = min(len(text), best_pos + length)

        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def _get_cache_key(
        self,
        query: str,
        algorithm: AlgorithmType,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> str:
        """Generate cache key for search query."""
        key_parts = [query, algorithm.value, f"limit:{limit}"]
        if filters:
            if filters.project_ids:
                key_parts.append(f"projects:{','.join(filters.project_ids)}")
            if filters.date_from:
                key_parts.append(f"from:{filters.date_from.isoformat()}")
            if filters.date_to:
                key_parts.append(f"to:{filters.date_to.isoformat()}")
            if filters.min_messages > 0:
                key_parts.append(f"min_msgs:{filters.min_messages}")

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[SearchResults]:
        """Get results from cache if valid."""
        if cache_key in self.result_cache:
            result, timestamp = self.result_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.result_cache.move_to_end(cache_key)
                return result
            del self.result_cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, result: SearchResults) -> None:
        """Add results to cache with LRU eviction."""
        if len(self.result_cache) >= self.cache_size:
            self.result_cache.popitem(last=False)
        self.result_cache[cache_key] = (result, time.time())

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return self.storage.get_stats()

    def close(self) -> None:
        """Close the storage connection."""
        if not self._external_storage:
            self.storage.close()
