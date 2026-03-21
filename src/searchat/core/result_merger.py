"""
Unified search result merger.

Merges palace (Layer 2) and verbatim (Layer 1) search results
into a single ranked list by conversation_id.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

from searchat.models import SearchResult
from searchat.models.domain import PalaceSearchResult, UnifiedSearchResult
from searchat.config.settings import RankingConfig
from searchat.core.normalize import percentile_normalize, normalize_score
from searchat.core.unified_storage import UnifiedStorage

logger = logging.getLogger(__name__)


def merge_results(
    palace_results: List[PalaceSearchResult],
    verbatim_results: List[SearchResult],
    storage: UnifiedStorage,
    ranking: RankingConfig,
) -> List[UnifiedSearchResult]:
    """Merge palace and verbatim results by conversation_id.

    Args:
        palace_results: Results from palace hybrid search
        verbatim_results: Results from verbatim hybrid search
        storage: UnifiedStorage for conversation metadata lookups
        ranking: RankingConfig with weights and boost factor

    Returns:
        List of UnifiedSearchResult sorted by combined_score descending
    """
    by_conv: Dict[str, UnifiedSearchResult] = {}

    # Get scaled weights (adjusted so max intersection score = 1.0)
    palace_weight = ranking.scaled_palace_weight
    verbatim_weight = ranking.scaled_verbatim_weight
    boost_multiplier = ranking.boost_multiplier  # 0.2 -> 1.2

    # Normalize palace scores (percentile-based for stability with sparse results)
    palace_divisor = percentile_normalize([r.score for r in palace_results])

    # Batch-fetch conversation metadata for all palace results (eliminates N+1)
    palace_conv_ids = list({p.conversation_id for p in palace_results})
    conv_meta_cache = storage.get_conversations_batch(palace_conv_ids)

    # Add palace results
    for p in palace_results:
        norm_score = normalize_score(p.score, palace_divisor)
        conv_meta = _get_conversation_metadata_cached(
            p.conversation_id, conv_meta_cache, p,
        )

        by_conv[p.conversation_id] = UnifiedSearchResult(
            conversation_id=p.conversation_id,
            project_id=p.project_id,
            title=conv_meta["title"],
            created_at=conv_meta["created_at"],
            updated_at=conv_meta["updated_at"],
            message_count=conv_meta["message_count"],
            file_path=conv_meta["file_path"],
            combined_score=norm_score * palace_weight,
            palace_score=p.score,
            palace_summary=p.exchange_core,
            palace_context=p.specific_context,
            rooms=p.rooms,
            files_touched=p.files_touched,
            ply_start=p.ply_start,
            ply_end=p.ply_end,
            object_id=p.object_id,
            palace_bm25_score=p.keyword_score,
            palace_semantic_score=p.semantic_score,
        )

    # Normalize verbatim scores (percentile-based for stability with sparse results)
    verbatim_divisor = percentile_normalize([r.score for r in verbatim_results])

    # Merge verbatim results
    for v in verbatim_results:
        norm_score = normalize_score(v.score, verbatim_divisor)

        if v.conversation_id in by_conv:
            # Intersection: boost and merge
            existing = by_conv[v.conversation_id]
            existing.verbatim_score = v.score
            existing.verbatim_snippet = v.snippet
            existing.message_start_index = v.message_start_index
            existing.message_end_index = v.message_end_index
            existing.verbatim_bm25_score = v.bm25_score
            existing.verbatim_semantic_score = v.semantic_score

            # Recalculate combined score with intersection boost
            palace_norm = normalize_score(existing.palace_score, palace_divisor) if existing.palace_score else 0
            verbatim_norm = norm_score
            existing.combined_score = (
                palace_weight * palace_norm + verbatim_weight * verbatim_norm
            ) * boost_multiplier  # Max intersection score = 1.0
        else:
            # Verbatim only
            by_conv[v.conversation_id] = UnifiedSearchResult(
                conversation_id=v.conversation_id,
                project_id=v.project_id,
                title=v.title,
                created_at=v.created_at,
                updated_at=v.updated_at,
                message_count=v.message_count,
                file_path=v.file_path,
                combined_score=norm_score * verbatim_weight,
                verbatim_score=v.score,
                verbatim_snippet=v.snippet,
                message_start_index=v.message_start_index,
                message_end_index=v.message_end_index,
                verbatim_bm25_score=v.bm25_score,
                verbatim_semantic_score=v.semantic_score,
            )

    # Sort by combined_score descending
    return sorted(by_conv.values(), key=lambda x: x.combined_score, reverse=True)


def merge_results_with_scoping(
    palace_results: List[PalaceSearchResult],
    verbatim_results: List[SearchResult],
    storage: UnifiedStorage,
    ranking: RankingConfig,
    resolved_project_ids: Optional[List[str]] = None,
    verbatim_boost: float = 1.5,
) -> tuple[List[UnifiedSearchResult], Dict[str, int]]:
    """Merge palace and verbatim results with consistent project scoping.

    Args:
        palace_results: Results from palace hybrid search (already scoped)
        verbatim_results: Results from verbatim hybrid search (unscoped)
        storage: UnifiedStorage for conversation metadata lookups
        ranking: RankingConfig with weights and boost factor
        resolved_project_ids: List of project IDs resolved via facet matching
        verbatim_boost: Boost multiplier for verbatim results in resolved projects (default 1.5x)

    Returns:
        Tuple of:
            - List of UnifiedSearchResult sorted by combined_score descending
            - Dict with scoping statistics: {"boosted": N, "not_boosted": M, "excluded": 0}
    """
    by_conv: Dict[str, UnifiedSearchResult] = {}
    scoping_stats = {"boosted": 0, "not_boosted": 0, "excluded": 0}

    # Get scaled weights
    palace_weight = ranking.scaled_palace_weight
    verbatim_weight = ranking.scaled_verbatim_weight
    boost_multiplier = ranking.boost_multiplier

    # Normalize palace scores
    palace_divisor = percentile_normalize([r.score for r in palace_results])

    # Batch-fetch conversation metadata (eliminates N+1)
    palace_conv_ids = list({p.conversation_id for p in palace_results})
    conv_meta_cache = storage.get_conversations_batch(palace_conv_ids)

    # Add palace results (already scoped)
    for p in palace_results:
        norm_score = normalize_score(p.score, palace_divisor)
        conv_meta = _get_conversation_metadata_cached(
            p.conversation_id, conv_meta_cache, p,
        )

        by_conv[p.conversation_id] = UnifiedSearchResult(
            conversation_id=p.conversation_id,
            project_id=p.project_id,
            title=conv_meta["title"],
            created_at=conv_meta["created_at"],
            updated_at=conv_meta["updated_at"],
            message_count=conv_meta["message_count"],
            file_path=conv_meta["file_path"],
            combined_score=norm_score * palace_weight,
            palace_score=p.score,
            palace_summary=p.exchange_core,
            palace_context=p.specific_context,
            rooms=p.rooms,
            files_touched=p.files_touched,
            ply_start=p.ply_start,
            ply_end=p.ply_end,
            object_id=p.object_id,
            palace_bm25_score=p.keyword_score,
            palace_semantic_score=p.semantic_score,
        )

    # Normalize verbatim scores
    verbatim_divisor = percentile_normalize([r.score for r in verbatim_results])

    # Apply soft scoping to verbatim results
    resolved_set = set(resolved_project_ids) if resolved_project_ids else set()

    for v in verbatim_results:
        norm_score = normalize_score(v.score, verbatim_divisor)

        # Soft scoping: boost if in resolved projects, but don't exclude others
        is_scoped = v.project_id in resolved_set if resolved_set else False
        scoping_multiplier = verbatim_boost if is_scoped else 1.0

        # Track scoping decisions
        if is_scoped:
            scoping_stats["boosted"] += 1
        else:
            scoping_stats["not_boosted"] += 1

        if v.conversation_id in by_conv:
            # Intersection: boost and merge
            existing = by_conv[v.conversation_id]
            existing.verbatim_score = v.score
            existing.verbatim_snippet = v.snippet
            existing.message_start_index = v.message_start_index
            existing.message_end_index = v.message_end_index
            existing.verbatim_bm25_score = v.bm25_score
            existing.verbatim_semantic_score = v.semantic_score

            # Recalculate combined score with soft scoping boost
            palace_norm = normalize_score(existing.palace_score, palace_divisor) if existing.palace_score else 0
            verbatim_norm = norm_score * scoping_multiplier
            existing.combined_score = (
                palace_weight * palace_norm + verbatim_weight * verbatim_norm
            ) * boost_multiplier
        else:
            # Verbatim only - apply soft scoping boost
            by_conv[v.conversation_id] = UnifiedSearchResult(
                conversation_id=v.conversation_id,
                project_id=v.project_id,
                title=v.title,
                created_at=v.created_at,
                updated_at=v.updated_at,
                message_count=v.message_count,
                file_path=v.file_path,
                combined_score=norm_score * scoping_multiplier * verbatim_weight,
                verbatim_score=v.score,
                verbatim_snippet=v.snippet,
                message_start_index=v.message_start_index,
                message_end_index=v.message_end_index,
                verbatim_bm25_score=v.bm25_score,
                verbatim_semantic_score=v.semantic_score,
            )

    # Log scoping decisions
    if resolved_set:
        logger.info(
            "Scoping applied: resolved_projects=%s, verbatim_boost=%.1fx, "
            "boosted=%d, not_boosted=%d",
            list(resolved_set),
            verbatim_boost,
            scoping_stats["boosted"],
            scoping_stats["not_boosted"],
        )

    # Sort by combined_score descending
    results = sorted(by_conv.values(), key=lambda x: x.combined_score, reverse=True)
    return results, scoping_stats


def _get_conversation_metadata_cached(
    conversation_id: str,
    conv_cache: Dict[str, dict],
    palace_result: PalaceSearchResult,
) -> dict:
    """Get conversation metadata from pre-fetched cache or fall back to palace data."""
    conv = conv_cache.get(conversation_id)
    if conv is not None:
        return {
            "title": conv["title"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "message_count": int(conv["message_count"]),
            "file_path": conv["file_path"],
        }
    summary = palace_result.exchange_core
    return {
        "title": summary[:50] + "..." if len(summary) > 50 else summary,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "message_count": 0,
        "file_path": "",
    }
