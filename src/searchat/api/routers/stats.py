"""Statistics endpoint - index statistics and metadata."""
from fastapi import APIRouter, HTTPException

from searchat.api.dependencies import get_unified_search_engine


router = APIRouter()


@router.get("/statistics")
async def get_statistics():
    """Get search index statistics from DuckDB."""
    engine = get_unified_search_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Unified search engine not available")

    cursor = engine.storage._get_read_cursor()
    row = cursor.execute("""
        SELECT COUNT(*) as total_conversations,
               COALESCE(SUM(message_count), 0) as total_messages,
               COALESCE(AVG(message_count), 0) as avg_messages,
               COUNT(DISTINCT project_id) as total_projects,
               MIN(created_at) as earliest_date,
               MAX(updated_at) as latest_date
        FROM conversations
    """).fetchone()

    return {
        "total_conversations": row[0],
        "total_messages": int(row[1]),
        "avg_messages": float(row[2]),
        "total_projects": row[3],
        "earliest_date": row[4].isoformat() if row[4] else None,
        "latest_date": row[5].isoformat() if row[5] else None,
    }
