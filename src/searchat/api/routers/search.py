"""Search endpoints - main search and projects list."""
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from datetime import datetime, timedelta

from fastapi import APIRouter, Query, HTTPException

from searchat.agents import detect_provider_id
from searchat.models import AlgorithmType, SearchFilters
from searchat.api.models import (
    UnifiedSearchResultResponse,
    SearchResultResponse,
    RoomResponse,
    FileTouchedResponse,
)
from searchat.api.dependencies import (
    get_unified_search_engine,
    get_palace_query,
    get_config,
    get_projects_cache,
    set_projects_cache,
)
from searchat.core.result_merger import merge_results, merge_results_with_scoping
from searchat.core.progressive_fallback import ProgressiveFallback
from searchat.core.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)

# Instantiate query classifier for adaptive mode
_query_classifier = QueryClassifier()

router = APIRouter()

# Thread pool for running sync search operations in parallel
_executor = ThreadPoolExecutor(max_workers=3)

# Global progressive fallback instance (initialized on first use)
_progressive_fallback: Optional[ProgressiveFallback] = None
_progressive_fallback_lock = threading.Lock()

# Shared mode string → AlgorithmType mapping
_MODE_MAP = {
    "distill": AlgorithmType.HYBRID,
    "hybrid": AlgorithmType.HYBRID,
    "semantic": AlgorithmType.SEMANTIC,
    "keyword": AlgorithmType.KEYWORD,
    "adaptive": AlgorithmType.ADAPTIVE,
}

# Cross-layer endpoint uses different mode names
_CROSS_LAYER_MODE_MAP = {
    "cross-layer": AlgorithmType.CROSS_LAYER,
    "verbatim": AlgorithmType.KEYWORD,
    "distill": AlgorithmType.DISTILL,
}


def _detect_source(file_path: str) -> str:
    """Detect if file path is WSL or Windows."""
    return "WSL" if "/home/" in file_path or "wsl" in file_path.lower() else "WIN"


def _parse_date_filters(
    filters: SearchFilters,
    date: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> None:
    """Apply date filtering to SearchFilters in place."""
    if date == "custom" and (date_from or date_to):
        if date_from:
            filters.date_from = datetime.fromisoformat(date_from)
        if date_to:
            filters.date_to = datetime.fromisoformat(date_to) + timedelta(days=1)
    elif date:
        now = datetime.now()
        if date == "today":
            filters.date_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
            filters.date_to = now
        elif date == "week":
            filters.date_from = now - timedelta(days=7)
            filters.date_to = now
        elif date == "month":
            filters.date_from = now - timedelta(days=30)
            filters.date_to = now


def _detect_tool(file_path: str) -> str:
    """Detect which agent provider owns the file path."""
    return detect_provider_id(file_path) or "unknown"


def _search_result_to_response(r) -> UnifiedSearchResultResponse:
    """Convert a SearchResult to UnifiedSearchResultResponse."""
    source = _detect_source(r.file_path)
    has_palace = r.palace_summary is not None
    has_verbatim = r.bm25_score is not None

    # Build files_touched from raw palace data
    files_touched = []
    if r.files_touched_raw:
        for f in r.files_touched_raw:
            if isinstance(f, dict):
                files_touched.append(
                    FileTouchedResponse(path=f.get("path", ""), action=f.get("action", ""))
                )

    return UnifiedSearchResultResponse(
        conversation_id=r.conversation_id,
        project_id=r.project_id,
        title=r.title,
        created_at=r.created_at.isoformat() if isinstance(r.created_at, datetime) else str(r.created_at) if r.created_at else "",
        updated_at=r.updated_at.isoformat() if isinstance(r.updated_at, datetime) else str(r.updated_at) if r.updated_at else "",
        message_count=r.message_count,
        file_path=r.file_path,
        combined_score=r.score,
        source=source,
        tool=_detect_tool(r.file_path),
        # Palace data
        palace_score=r.semantic_score,
        palace_summary=r.palace_summary,
        palace_context=r.palace_context,
        rooms=[],
        files_touched=files_touched,
        ply_start=r.message_start_index,
        ply_end=r.message_end_index,
        object_id=r.object_id,
        # Verbatim data
        verbatim_score=r.bm25_score,
        verbatim_snippet=r.snippet if has_verbatim else None,
        message_start_index=r.message_start_index,
        message_end_index=r.message_end_index,
        # Sub-scores: cross-layer uses verbatim BM25 + palace semantic
        verbatim_bm25_score=r.bm25_score,
        palace_semantic_score=r.semantic_score,
        # Flags
        has_palace=has_palace,
        has_verbatim=has_verbatim,
        is_intersection=has_palace and has_verbatim,
    )


@router.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    mode: str = Query("cross-layer", description="Search mode: cross-layer (default), verbatim, or distill"),
    project: Optional[str] = Query(None, description="Filter by project"),
    date: Optional[str] = Query(None, description="Date filter: today, week, month, or custom"),
    date_from: Optional[str] = Query(None, description="Custom date from (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Custom date to (YYYY-MM-DD)"),
    sort_by: str = Query("relevance", description="Sort by: relevance, date_newest, date_oldest, messages"),
    limit: int = Query(100, description="Max results to return (1-100)", ge=1, le=100)
):
    """Search using unified DuckDB engine with manuscript-optimal modes.

    Modes:
    - cross-layer (default): BM25-FTS(verbatim) + HNSW(distilled) / CombMNZ fusion
    - verbatim: BM25-FTS keyword search on full conversation text
    - distill: Semantic search on compressed palace objects
    """
    start_time = time.time()

    unified_engine = get_unified_search_engine()
    if unified_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Unified search engine not available. Index unified database first.",
        )

    try:
        # Map mode strings to algorithm types
        algorithm = _CROSS_LAYER_MODE_MAP.get(mode)
        if algorithm is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{mode}'. Use: cross-layer, verbatim, distill",
            )

        # Build filters
        filters = SearchFilters()
        if project:
            filters.project_ids = [project]
        _parse_date_filters(filters, date, date_from, date_to)

        # Run search via unified engine
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            _executor,
            lambda: unified_engine.search(q, algorithm=algorithm, filters=filters, limit=limit),
        )

        # Sort if non-default
        result_list = list(results.results)
        if sort_by == "date_newest":
            result_list.sort(key=lambda r: r.updated_at or "", reverse=True)
        elif sort_by == "date_oldest":
            result_list.sort(key=lambda r: r.updated_at or "", reverse=False)
        elif sort_by == "messages":
            result_list.sort(key=lambda r: r.message_count, reverse=True)

        # Convert to response format
        response_results = [_search_result_to_response(r) for r in result_list]

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "results": response_results,
            "total": results.total_count,
            "search_time_ms": elapsed_ms,
            "mode_used": results.mode_used,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects")
async def get_projects() -> List[str]:
    """Get list of all projects in the index."""
    projects_cache = get_projects_cache()
    if projects_cache is None:
        unified_engine = get_unified_search_engine()

        if unified_engine:
            # Get projects from DuckDB
            result = unified_engine.storage._get_read_cursor().execute(
                "SELECT DISTINCT project_id FROM exchanges ORDER BY project_id"
            ).fetchall()
            projects_cache = [row[0] for row in result]
        else:
            projects_cache = []
        set_projects_cache(projects_cache)

    return projects_cache


@router.get("/search/unified")
async def search_unified(
    q: str = Query(..., description="Search query"),
    mode: str = Query("distill", description="Search mode: distill (default), semantic, or keyword"),
    project: Optional[str] = Query(None, description="Filter by project"),
    date: Optional[str] = Query(None, description="Date filter: today, week, month, or custom"),
    date_from: Optional[str] = Query(None, description="Custom date from (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Custom date to (YYYY-MM-DD)"),
    limit: int = Query(50, description="Max results to return (1-100)", ge=1, le=100),
):
    """Search using the unified DuckDB engine with exchange-level results.

    This endpoint uses the DuckDB-based search with native VSS and FTS.
    Returns exchange-level matches with traceable results.
    """
    start_time = time.time()

    unified_engine = get_unified_search_engine()
    if unified_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Unified search engine not available. Index unified database first.",
        )

    try:
        # Convert mode string to AlgorithmType enum
        search_mode = _MODE_MAP.get(mode, AlgorithmType.HYBRID)

        # Build filters
        filters = SearchFilters()
        if project:
            filters.project_ids = [project]
        _parse_date_filters(filters, date, date_from, date_to)

        # Run search
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            _executor,
            lambda: unified_engine.search(q, algorithm=search_mode, filters=filters, limit=limit),
        )

        # Convert to response format
        response_results = []
        for r in results.results:
            source = _detect_source(r.file_path)

            response_results.append(SearchResultResponse(
                conversation_id=r.conversation_id,
                project_id=r.project_id,
                title=r.title,
                created_at=r.created_at.isoformat() if isinstance(r.created_at, datetime) else str(r.created_at),
                updated_at=r.updated_at.isoformat() if isinstance(r.updated_at, datetime) else str(r.updated_at),
                message_count=r.message_count,
                file_path=r.file_path,
                snippet=r.snippet,
                score=r.score,
                message_start_index=r.message_start_index,
                message_end_index=r.message_end_index,
                source=source,
                tool=_detect_tool(r.file_path),
                bm25_score=r.bm25_score,
                semantic_score=r.semantic_score,
                exchange_id=r.exchange_id,
                exchange_text=r.exchange_text,
                match_source="unified",
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "results": response_results,
            "total": results.total_count,
            "search_time_ms": elapsed_ms,
            "mode_used": results.mode_used,
            "engine": "unified",
        }

    except Exception as e:
        logger.error("Unified search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/fallback")
async def search_with_progressive_fallback(
    q: str = Query(..., description="Search query"),
    mode: str = Query("distill", description="Search mode: distill (default), semantic, keyword, or adaptive"),
    project: Optional[str] = Query(None, description="Filter by project"),
    date: Optional[str] = Query(None, description="Date filter: today, week, month, or custom"),
    date_from: Optional[str] = Query(None, description="Custom date from (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Custom date to (YYYY-MM-DD)"),
    sort_by: str = Query("relevance", description="Sort by: relevance, date_newest, date_oldest, messages"),
    limit: int = Query(100, description="Max results to return (1-100)", ge=1, le=100),
    min_results: int = Query(3, description="Min results before fallback triggers", ge=1, le=10),
):
    """Search with progressive fallback on empty palace results.

    Three-tier fallback strategy:
    1. Scoped search (resolved projects only)
    2. Related projects expansion (if <min_results)
    3. Unscoped search (if still <min_results)

    Returns results with fallback_tier indicator for analysis.
    """
    global _progressive_fallback
    start_time = time.time()

    try:
        unified_engine = get_unified_search_engine()
        palace_query = get_palace_query()
        config = get_config()

        if unified_engine is None:
            raise HTTPException(status_code=503, detail="Unified search engine not available")

        # Initialize progressive fallback if needed (thread-safe)
        if _progressive_fallback is None:
            with _progressive_fallback_lock:
                if _progressive_fallback is None:
                    if palace_query is None:
                        raise HTTPException(
                            status_code=503,
                            detail="Palace query engine not available",
                        )
                    _progressive_fallback = ProgressiveFallback(
                        palace_query=palace_query,
                        min_results=min_results,
                    )

        _progressive_fallback.min_results = min_results

        # Convert mode string to AlgorithmType enum
        search_mode = _MODE_MAP.get(mode, AlgorithmType.HYBRID)

        # Build filters
        filters = SearchFilters()
        if project:
            filters.project_ids = [project]
        _parse_date_filters(filters, date, date_from, date_to)

        # Run searches in parallel
        loop = asyncio.get_running_loop()

        def run_verbatim():
            return unified_engine.search(q, algorithm=search_mode, filters=filters, limit=limit)

        def run_palace_with_fallback():
            if palace_query is None:
                return [], "unscoped"

            # Attempt facet resolution if no explicit project filter
            pids = filters.project_ids if filters.project_ids else None
            if pids is None:
                resolved_pids = unified_engine._resolve_facets(q)
                if resolved_pids:
                    logger.info("Facet resolution: '%s' → %s", q, resolved_pids)
                    pids = resolved_pids

            # Determine weights based on mode
            if search_mode == AlgorithmType.KEYWORD:
                kw_weight, sem_weight = 1.0, 0.0
            elif search_mode == AlgorithmType.SEMANTIC:
                kw_weight, sem_weight = 0.0, 1.0
            else:
                kw_weight, sem_weight = 0.5, 0.5

            # Execute progressive fallback search
            results, tier = _progressive_fallback.search_with_fallback(
                query=q,
                project_ids=pids,
                limit=limit,
                keyword_weight=kw_weight,
                semantic_weight=sem_weight,
            )

            return results, tier.value

        # Execute in parallel using thread pool
        verbatim_future = loop.run_in_executor(_executor, run_verbatim)
        palace_future = loop.run_in_executor(_executor, run_palace_with_fallback)

        verbatim_results, palace_with_tier = await asyncio.gather(
            verbatim_future, palace_future
        )

        palace_results, fallback_tier = palace_with_tier

        # Merge results using config
        unified_results = merge_results(
            palace_results,
            verbatim_results.results,
            unified_engine.storage,
            config.search.ranking,
        )

        # Add fallback tier to all results
        for r in unified_results:
            r.fallback_tier = fallback_tier if r.has_palace else None

        # Sort results based on sort_by parameter
        if sort_by == "date_newest":
            unified_results.sort(key=lambda r: r.updated_at, reverse=True)
        elif sort_by == "date_oldest":
            unified_results.sort(key=lambda r: r.updated_at, reverse=False)
        elif sort_by == "messages":
            unified_results.sort(key=lambda r: r.message_count, reverse=True)

        # Convert to response format
        response_results = []
        for r in unified_results[:limit]:
            source = _detect_source(r.file_path)

            response_results.append(UnifiedSearchResultResponse(
                conversation_id=r.conversation_id,
                project_id=r.project_id,
                title=r.title,
                created_at=r.created_at.isoformat() if isinstance(r.created_at, datetime) else str(r.created_at),
                updated_at=r.updated_at.isoformat() if isinstance(r.updated_at, datetime) else str(r.updated_at),
                message_count=r.message_count,
                file_path=r.file_path,
                combined_score=r.combined_score,
                source=source,
                tool=_detect_tool(r.file_path),
                palace_score=r.palace_score,
                palace_summary=r.palace_summary,
                palace_context=r.palace_context,
                rooms=[
                    RoomResponse(
                        room_id=room.room_id,
                        room_type=room.room_type,
                        room_key=room.room_key,
                        room_label=room.room_label,
                    )
                    for room in r.rooms
                ],
                files_touched=[
                    FileTouchedResponse(path=f.path, action=f.action)
                    for f in r.files_touched
                ],
                ply_start=r.ply_start,
                ply_end=r.ply_end,
                object_id=r.object_id,
                verbatim_score=r.verbatim_score,
                verbatim_snippet=r.verbatim_snippet,
                message_start_index=r.message_start_index,
                message_end_index=r.message_end_index,
                palace_bm25_score=r.palace_bm25_score,
                palace_semantic_score=r.palace_semantic_score,
                verbatim_bm25_score=r.verbatim_bm25_score,
                verbatim_semantic_score=r.verbatim_semantic_score,
                has_palace=r.has_palace,
                has_verbatim=r.has_verbatim,
                is_intersection=r.is_intersection,
                fallback_tier=r.fallback_tier,
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        # Get fallback stats
        fallback_stats = _progressive_fallback.get_fallback_stats()

        return {
            "results": response_results,
            "total": len(unified_results),
            "search_time_ms": elapsed_ms,
            "palace_count": len(palace_results),
            "verbatim_count": verbatim_results.total_count,
            "fallback_tier": fallback_tier,
            "fallback_stats": fallback_stats,
        }

    except Exception as e:
        logger.error("Progressive fallback search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/fallback/stats")
async def get_fallback_stats():
    """Get statistics on progressive fallback usage.

    Returns tier usage counts and percentages for analysis.
    """
    global _progressive_fallback

    if _progressive_fallback is None:
        return {
            "message": "No fallback statistics available (no searches performed yet)",
            "stats": {
                "total_searches": 0,
                "scoped_count": 0,
                "scoped_pct": 0.0,
                "related_count": 0,
                "related_pct": 0.0,
                "unscoped_count": 0,
                "unscoped_pct": 0.0,
            }
        }

    return {
        "message": "Progressive fallback statistics",
        "stats": _progressive_fallback.get_fallback_stats(),
    }


@router.post("/search/fallback/stats/reset")
async def reset_fallback_stats():
    """Reset progressive fallback statistics.

    Use this to start fresh tracking after configuration changes.
    """
    global _progressive_fallback

    if _progressive_fallback is None:
        return {"message": "No fallback instance to reset"}

    _progressive_fallback.reset_stats()
    return {"message": "Fallback statistics reset successfully"}


@router.get("/search/facet_weighted")
async def search_facet_weighted(
    q: str = Query(..., description="Search query"),
    mode: str = Query("distill", description="Search mode: distill (default), semantic, keyword, or adaptive"),
    limit: int = Query(100, description="Max results to return (1-100)", ge=1, le=100),
    top_k: int = Query(5, description="Number of top facets to vote (1-20)", ge=1, le=20),
    confidence: float = Query(0.6, description="Confidence threshold (0.0-1.0)", ge=0.0, le=1.0),
):
    """Search with weighted facet voting for project resolution.

    Uses weighted voting across top-K facets instead of winner-takes-all.
    Returns search results with facet resolution metadata for evaluation.
    """
    start_time = time.time()

    try:
        unified_engine = get_unified_search_engine()
        palace_query = get_palace_query()
        config = get_config()

        if unified_engine is None:
            raise HTTPException(status_code=503, detail="Unified search engine not available")

        # Convert mode string to AlgorithmType enum
        search_mode = _MODE_MAP.get(mode, AlgorithmType.HYBRID)

        # Build filters with no explicit project
        filters = SearchFilters()

        # Resolve using BOTH methods for comparison
        winner_takes_all = unified_engine._resolve_facets(q, top_k=top_k)
        weighted_voting = unified_engine._resolve_facets_weighted(
            q, top_k=top_k, confidence_threshold=confidence
        )

        # Run searches in parallel
        loop = asyncio.get_running_loop()

        def run_verbatim():
            return unified_engine.search(q, algorithm=search_mode, filters=filters, limit=limit)

        def run_palace_winner():
            if palace_query is None or winner_takes_all is None:
                return []
            if search_mode == AlgorithmType.KEYWORD:
                return palace_query.search_hybrid(q, limit=limit, keyword_weight=1.0, semantic_weight=0.0, project_ids=winner_takes_all)
            elif search_mode == AlgorithmType.SEMANTIC:
                return palace_query.search_hybrid(q, limit=limit, keyword_weight=0.0, semantic_weight=1.0, project_ids=winner_takes_all)
            else:
                return palace_query.search_hybrid(q, limit=limit, project_ids=winner_takes_all)

        def run_palace_weighted():
            if palace_query is None or weighted_voting is None:
                return []
            if search_mode == AlgorithmType.KEYWORD:
                return palace_query.search_hybrid(q, limit=limit, keyword_weight=1.0, semantic_weight=0.0, project_ids=weighted_voting)
            elif search_mode == AlgorithmType.SEMANTIC:
                return palace_query.search_hybrid(q, limit=limit, keyword_weight=0.0, semantic_weight=1.0, project_ids=weighted_voting)
            else:
                return palace_query.search_hybrid(q, limit=limit, project_ids=weighted_voting)

        # Execute in parallel
        verbatim_future = loop.run_in_executor(_executor, run_verbatim)
        winner_future = loop.run_in_executor(_executor, run_palace_winner)
        weighted_future = loop.run_in_executor(_executor, run_palace_weighted)

        verbatim_results, palace_winner, palace_weighted = await asyncio.gather(
            verbatim_future, winner_future, weighted_future
        )

        # Merge results for both methods
        winner_merged = merge_results(
            palace_winner,
            verbatim_results.results,
            unified_engine.storage,
            config.search.ranking,
        )

        weighted_merged = merge_results(
            palace_weighted,
            verbatim_results.results,
            unified_engine.storage,
            config.search.ranking,
        )

        # Convert to response format
        def to_response(results):
            response = []
            for r in results[:limit]:
                source = _detect_source(r.file_path)
                response.append({
                    "conversation_id": r.conversation_id,
                    "project_id": r.project_id,
                    "title": r.title,
                    "combined_score": r.combined_score,
                    "palace_score": r.palace_score,
                    "verbatim_score": r.verbatim_score,
                    "source": source,
                })
            return response

        winner_response = to_response(winner_merged)
        weighted_response = to_response(weighted_merged)

        # Calculate agreement
        winner_conv_ids = {r["conversation_id"] for r in winner_response}
        weighted_conv_ids = {r["conversation_id"] for r in weighted_response}
        intersection = winner_conv_ids & weighted_conv_ids
        union = winner_conv_ids | weighted_conv_ids

        # Top-5 agreement
        winner_top5 = [r["conversation_id"] for r in winner_response[:5]]
        weighted_top5 = [r["conversation_id"] for r in weighted_response[:5]]
        top5_agreement = len(set(winner_top5) & set(weighted_top5)) / 5 if weighted_response else 0.0

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "winner_takes_all": {
                "results": winner_response,
                "resolved_project": winner_takes_all[0] if winner_takes_all else None,
                "palace_count": len(palace_winner),
            },
            "weighted_voting": {
                "results": weighted_response,
                "resolved_project": weighted_voting[0] if weighted_voting else None,
                "palace_count": len(palace_weighted),
            },
            "comparison": {
                "winner_count": len(winner_response),
                "weighted_count": len(weighted_response),
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard_similarity": len(intersection) / len(union) if union else 0.0,
                "top5_agreement": top5_agreement,
                "same_project": winner_takes_all == weighted_voting,
            },
            "parameters": {
                "top_k": top_k,
                "confidence_threshold": confidence,
                "mode": mode,
            },
            "verbatim_count": verbatim_results.total_count,
            "search_time_ms": elapsed_ms,
        }

    except Exception as e:
        logger.error("Facet weighted search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/scoped")
async def search_with_consistent_scoping(
    q: str = Query(..., description="Search query"),
    mode: str = Query("distill", description="Search mode: distill (default), semantic, keyword, or adaptive"),
    project: Optional[str] = Query(None, description="Filter by project"),
    date: Optional[str] = Query(None, description="Date filter: today, week, month, or custom"),
    date_from: Optional[str] = Query(None, description="Custom date from (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Custom date to (YYYY-MM-DD)"),
    sort_by: str = Query("relevance", description="Sort by: relevance, date_newest, date_oldest, messages"),
    limit: int = Query(100, description="Max results to return (1-100)", ge=1, le=100),
    verbatim_boost: float = Query(1.5, description="Boost multiplier for verbatim results in resolved projects (1.0-3.0)", ge=1.0, le=3.0),
    top_k: int = Query(5, description="Number of top facets for resolution (1-20)", ge=1, le=20),
):
    """Search with consistent project scoping across palace and verbatim layers.

    - Facet resolution determines relevant projects
    - Palace search is scoped to resolved projects (hard scoping)
    - Verbatim search is soft-scoped: boost resolved projects but don't exclude others
    """
    start_time = time.time()

    try:
        unified_engine = get_unified_search_engine()
        palace_query = get_palace_query()
        config = get_config()

        if unified_engine is None:
            raise HTTPException(status_code=503, detail="Unified search engine not available")

        # Convert mode string to AlgorithmType enum
        search_mode = _MODE_MAP.get(mode, AlgorithmType.HYBRID)

        # Build filters
        filters = SearchFilters()
        if project:
            filters.project_ids = [project]
        _parse_date_filters(filters, date, date_from, date_to)

        # Resolve projects via facet matching (if no explicit project filter)
        resolved_projects = None
        if not filters.project_ids:
            resolved_projects = unified_engine._resolve_facets(q, top_k=top_k)
            if resolved_projects:
                logger.info("Facet resolution: '%s' → %s (top_k=%d)", q, resolved_projects, top_k)

        # Run searches in parallel
        loop = asyncio.get_running_loop()

        def run_verbatim():
            # Verbatim search is UNSCOPED - we'll apply soft scoping in the merger
            return unified_engine.search(q, algorithm=search_mode, filters=filters, limit=limit)

        def run_palace():
            if palace_query is None:
                return []

            # Palace search is HARD SCOPED to resolved projects
            pids = filters.project_ids if filters.project_ids else resolved_projects

            if search_mode == AlgorithmType.KEYWORD:
                return palace_query.search_hybrid(q, limit=limit, keyword_weight=1.0, semantic_weight=0.0, project_ids=pids)
            elif search_mode == AlgorithmType.SEMANTIC:
                return palace_query.search_hybrid(q, limit=limit, keyword_weight=0.0, semantic_weight=1.0, project_ids=pids)
            elif search_mode == AlgorithmType.ADAPTIVE:
                classification = _query_classifier.classify(q)
                logger.info(
                    "Palace adaptive search: '%s' → %s (bm25=%.1f, sem=%.1f)",
                    q[:100],
                    classification.query_type,
                    classification.bm25_weight,
                    classification.semantic_weight,
                )
                return palace_query.search_hybrid(
                    q,
                    limit=limit,
                    keyword_weight=classification.bm25_weight,
                    semantic_weight=classification.semantic_weight,
                    project_ids=pids,
                )
            else:
                return palace_query.search_hybrid(q, limit=limit, project_ids=pids)

        # Execute in parallel
        verbatim_future = loop.run_in_executor(_executor, run_verbatim)
        palace_future = loop.run_in_executor(_executor, run_palace)

        verbatim_results, palace_results = await asyncio.gather(
            verbatim_future, palace_future
        )

        # Merge with consistent scoping
        unified_results, scoping_stats = merge_results_with_scoping(
            palace_results,
            verbatim_results.results,
            unified_engine.storage,
            config.search.ranking,
            resolved_project_ids=resolved_projects,
            verbatim_boost=verbatim_boost,
        )

        # Sort results based on sort_by parameter
        if sort_by == "date_newest":
            unified_results.sort(key=lambda r: r.updated_at, reverse=True)
        elif sort_by == "date_oldest":
            unified_results.sort(key=lambda r: r.updated_at, reverse=False)
        elif sort_by == "messages":
            unified_results.sort(key=lambda r: r.message_count, reverse=True)
        # else keep default relevance sorting (by combined_score)

        # Convert to response format
        response_results = []
        for r in unified_results[:limit]:
            source = _detect_source(r.file_path)

            response_results.append(UnifiedSearchResultResponse(
                conversation_id=r.conversation_id,
                project_id=r.project_id,
                title=r.title,
                created_at=r.created_at.isoformat() if isinstance(r.created_at, datetime) else str(r.created_at),
                updated_at=r.updated_at.isoformat() if isinstance(r.updated_at, datetime) else str(r.updated_at),
                message_count=r.message_count,
                file_path=r.file_path,
                combined_score=r.combined_score,
                source=source,
                tool=_detect_tool(r.file_path),
                palace_score=r.palace_score,
                palace_summary=r.palace_summary,
                palace_context=r.palace_context,
                rooms=[
                    RoomResponse(
                        room_id=room.room_id,
                        room_type=room.room_type,
                        room_key=room.room_key,
                        room_label=room.room_label,
                    )
                    for room in r.rooms
                ],
                files_touched=[
                    FileTouchedResponse(path=f.path, action=f.action)
                    for f in r.files_touched
                ],
                ply_start=r.ply_start,
                ply_end=r.ply_end,
                object_id=r.object_id,
                verbatim_score=r.verbatim_score,
                verbatim_snippet=r.verbatim_snippet,
                message_start_index=r.message_start_index,
                message_end_index=r.message_end_index,
                palace_bm25_score=r.palace_bm25_score,
                palace_semantic_score=r.palace_semantic_score,
                verbatim_bm25_score=r.verbatim_bm25_score,
                verbatim_semantic_score=r.verbatim_semantic_score,
                has_palace=r.has_palace,
                has_verbatim=r.has_verbatim,
                is_intersection=r.is_intersection,
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "results": response_results,
            "total": len(unified_results),
            "search_time_ms": elapsed_ms,
            "palace_count": len(palace_results),
            "verbatim_count": verbatim_results.total_count,
            "resolved_projects": resolved_projects,
            "scoping_stats": scoping_stats,
            "scoping_config": {
                "verbatim_boost": verbatim_boost,
                "top_k": top_k,
                "mode": mode,
            },
        }

    except Exception as e:
        logger.error("Scoped search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
