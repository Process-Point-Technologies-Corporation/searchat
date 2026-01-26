"""Statistics endpoint - index statistics and metadata."""

from fastapi import APIRouter

import searchat.api.dependencies as deps


router = APIRouter()


@router.get("/statistics")
async def get_statistics():
    """Get search index statistics."""
    if deps.stats_cache is None:
        store = deps.get_duckdb_store()
        stats = store.get_statistics()
        deps.stats_cache = {
            "total_conversations": stats.total_conversations,
            "total_messages": stats.total_messages,
            "avg_messages": stats.avg_messages,
            "total_projects": stats.total_projects,
            "earliest_date": stats.earliest_date,
            "latest_date": stats.latest_date,
        }
    return deps.stats_cache
