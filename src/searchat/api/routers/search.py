"""Search endpoints - main search and projects list."""
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from searchat.models import SearchMode, SearchFilters
from searchat.api.models import SearchResultResponse
import searchat.api.dependencies as deps

from searchat.api.dependencies import get_search_engine, trigger_search_engine_warmup
from searchat.api.readiness import get_readiness, warming_payload, error_payload


router = APIRouter()


@router.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    mode: str = Query("hybrid", description="Search mode: hybrid, semantic, or keyword"),
    project: Optional[str] = Query(None, description="Filter by project"),
    date: Optional[str] = Query(None, description="Date filter: today, week, month, or custom"),
    date_from: Optional[str] = Query(None, description="Custom date from (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Custom date to (YYYY-MM-DD)"),
    sort_by: str = Query("relevance", description="Sort by: relevance, date_newest, date_oldest, messages"),
    limit: int = Query(100, description="Max results to return (1-100)", ge=1, le=100)
):
    """Search conversations with filters and sorting."""
    try:
        readiness = get_readiness().snapshot()
        engine_state = readiness.components.get("search_engine")
        if engine_state == "error":
            return JSONResponse(status_code=500, content=error_payload())
        if engine_state != "ready":
            trigger_search_engine_warmup()
            return JSONResponse(status_code=503, content=warming_payload())

        search_engine = get_search_engine()

        # Convert mode string to SearchMode enum
        mode_map = {
            "hybrid": SearchMode.HYBRID,
            "semantic": SearchMode.SEMANTIC,
            "keyword": SearchMode.KEYWORD
        }
        search_mode = mode_map.get(mode, SearchMode.HYBRID)

        # Build filters
        filters = SearchFilters()
        if project:
            filters.project_ids = [project]

        # Handle date filtering
        if date == "custom" and (date_from or date_to):
            # Custom date range
            if date_from:
                filters.date_from = datetime.fromisoformat(date_from)
            if date_to:
                # Add 1 day to include the entire end date
                filters.date_to = datetime.fromisoformat(date_to) + timedelta(days=1)
        elif date:
            # Preset date ranges
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

        # Execute search
        results = search_engine.search(q, mode=search_mode, filters=filters)

        # Sort results based on sort_by parameter
        sorted_results = results.results.copy()
        if sort_by == "date_newest":
            sorted_results.sort(key=lambda r: r.updated_at, reverse=True)
        elif sort_by == "date_oldest":
            sorted_results.sort(key=lambda r: r.updated_at, reverse=False)
        elif sort_by == "messages":
            sorted_results.sort(key=lambda r: r.message_count, reverse=True)
        # else keep default relevance sorting (by score)

        # Convert results to response format
        response_results = []
        for r in sorted_results[:limit]:
            source = "WSL" if "/home/" in r.file_path or "wsl" in r.file_path.lower() else "WIN"
            response_results.append(SearchResultResponse(
                conversation_id=r.conversation_id,
                project_id=r.project_id,
                title=r.title,
                created_at=r.created_at.isoformat(),
                updated_at=r.updated_at.isoformat(),
                message_count=r.message_count,
                file_path=r.file_path,
                snippet=r.snippet,
                score=r.score,
                message_start_index=r.message_start_index,
                message_end_index=r.message_end_index,
                source=source
            ))

        return {
            "results": response_results,
            "total": results.total_count,
            "search_time_ms": results.search_time_ms
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects")
async def get_projects():
    """Get list of all projects in the index."""
    store = deps.get_duckdb_store()

    if deps.projects_cache is None:
        deps.projects_cache = store.list_projects()
    return deps.projects_cache
