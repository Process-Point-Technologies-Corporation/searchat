"""Conversation endpoints - listing, viewing, and session resume."""
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Query, HTTPException

from searchat.agents import detect_provider, detect_provider_id
from searchat.api.models import (
    SearchResultResponse,
    ConversationResponse,
    ResumeRequest,
)
from searchat.api.dependencies import get_unified_search_engine, get_platform_manager
from searchat.api.routers.search import _detect_source


def _detect_tool_from_path(file_path: str) -> str:
    """Detect agent tool from file path using string matching (no disk I/O)."""
    normalized = file_path.replace("\\", "/").lower()
    if "/.codex/" in normalized:
        return "codex"
    if "/.vibe/" in normalized:
        return "vibe"
    return "claude"


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/conversations/all")
async def get_all_conversations(
    sort_by: str = Query("length", description="Sort by: length, date_newest, date_oldest, title"),
    project: Optional[str] = Query(None, description="Filter by project"),
    date: Optional[str] = Query(None, description="Date filter: today, week, month, or custom"),
    date_from: Optional[str] = Query(None, description="Custom date from (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Custom date to (YYYY-MM-DD)"),
    limit: int = Query(50, description="Max results per page", ge=1, le=200),
    offset: int = Query(0, description="Offset for pagination", ge=0),
):
    """Get conversations with sorting, filtering, and pagination."""
    try:
        unified_engine = get_unified_search_engine()
        if unified_engine is None:
            raise HTTPException(status_code=503, detail="Unified search engine not available")

        # Build WHERE clause
        conditions = ["message_count > 0"]
        params: list = []

        if project:
            conditions.append("project_id = ?")
            params.append(project)

        if date == "custom" and (date_from or date_to):
            if date_from:
                conditions.append("updated_at >= ?")
                params.append(datetime.fromisoformat(date_from))
            if date_to:
                conditions.append("updated_at < ?")
                params.append(datetime.fromisoformat(date_to) + timedelta(days=1))
        elif date:
            now = datetime.now()
            if date == "today":
                conditions.append("updated_at >= ?")
                params.append(now.replace(hour=0, minute=0, second=0, microsecond=0))
            elif date == "week":
                conditions.append("updated_at >= ?")
                params.append(now - timedelta(days=7))
            elif date == "month":
                conditions.append("updated_at >= ?")
                params.append(now - timedelta(days=30))

        sort_map = {
            "length": "message_count DESC",
            "date_newest": "updated_at DESC",
            "date_oldest": "updated_at ASC",
            "title": "title ASC",
        }
        order_by = sort_map.get(sort_by, "message_count DESC")
        where_clause = " AND ".join(conditions)

        count_sql = f"SELECT COUNT(*) FROM conversations WHERE {where_clause}"
        data_sql = f"""
            SELECT conversation_id, project_id, title, created_at, updated_at,
                   message_count, file_path
            FROM conversations
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
        """

        # Direct execution — query takes ~3ms, no need for thread pool
        # (run_in_executor gets blocked by GIL-holding embedding threads)
        cursor = unified_engine.storage._get_read_cursor()
        total_count = cursor.execute(count_sql, params).fetchone()[0]
        rows = cursor.execute(data_sql, params + [limit, offset]).fetchall()

        response_results = [
            SearchResultResponse(
                conversation_id=r[0],
                project_id=r[1],
                title=r[2],
                created_at=r[3].isoformat() if isinstance(r[3], datetime) else str(r[3]) if r[3] else "",
                updated_at=r[4].isoformat() if isinstance(r[4], datetime) else str(r[4]) if r[4] else "",
                message_count=r[5],
                file_path=r[6],
                snippet="",
                score=0.0,
                message_start_index=None,
                message_end_index=None,
                source=_detect_source(r[6]),
                tool=_detect_tool_from_path(r[6]),
            )
            for r in rows
        ]

        return {
            "results": response_results,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count,
            "search_time_ms": 0,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str) -> ConversationResponse:
    """Get a specific conversation with all messages."""
    try:
        unified_engine = get_unified_search_engine()
        if unified_engine is None:
            raise HTTPException(status_code=503, detail="Unified search engine not available")

        # Query from DuckDB
        conv = unified_engine.storage.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found in index")
        file_path = conv['file_path']

        # Check if file exists
        if not Path(file_path).exists():
            logger.error(f"Conversation file not found: {file_path} (conversation_id: {conversation_id})")
            raise HTTPException(
                status_code=404,
                detail=f"Conversation file not found. The file may have been moved or deleted: {file_path}"
            )

        provider = detect_provider(Path(file_path))
        if provider is None:
            raise HTTPException(status_code=400, detail=f"Unknown conversation format: {file_path}")

        # Load messages from transcript file
        try:
            messages = await asyncio.get_running_loop().run_in_executor(
                None, lambda: provider.load_messages(Path(file_path))
            )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in conversation file {file_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse conversation file (invalid JSON)"
            )
        except ValueError as e:
            logger.error(f"Malformed transcript in conversation file {file_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse conversation file ({e})"
            )
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {file_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to read conversation file (encoding error)"
            )

        logger.info(f"Successfully loaded conversation {conversation_id} with {len(messages)} messages")

        return ConversationResponse(
            conversation_id=conversation_id,
            title=conv['title'],
            project_id=conv['project_id'],
            file_path=conv['file_path'],
            tool=provider.agent_id,
            message_count=len(messages),
            messages=messages
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/resume")
async def resume_session(request: ResumeRequest):
    """Resume a conversation session in its original tool (Claude Code or Vibe)."""
    try:
        unified_engine = get_unified_search_engine()
        platform_manager = get_platform_manager()

        if unified_engine is None:
            raise HTTPException(status_code=503, detail="Unified search engine not available")

        # Find conversation in DuckDB
        conv = unified_engine.storage.get_conversation(request.conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        file_path = conv['file_path']
        session_id = conv['conversation_id']

        # Extract working directory from conversation file
        cwd = None

        provider = detect_provider(Path(file_path))
        if provider is None:
            raise HTTPException(status_code=400, detail=f"Unknown conversation format: {file_path}")
        tool = provider.agent_id
        cwd = await asyncio.get_running_loop().run_in_executor(
            None, lambda: provider.extract_cwd(Path(file_path))
        )
        command = provider.build_resume_command(session_id)

        # Normalize path for current platform
        if cwd:
            cwd = platform_manager.normalize_path(cwd)

        logger.info(f"Resuming {tool} session {session_id}")
        logger.info(f"  Platform: {platform_manager.platform}")
        logger.info(f"  Original CWD: {cwd}")
        logger.info(f"  Command: {command}")

        # Open terminal with command using platform-specific implementation
        # Path translation happens automatically in open_terminal_with_command
        platform_manager.open_terminal_with_command(command, cwd)

        return {
            "success": True,
            "tool": tool,
            "cwd": cwd,
            "command": command,
            "platform": platform_manager.platform
        }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        # Command not found (claude, codex, or vibe not installed)
        logger.error(f"Command not found: {e}")
        tool_name = locals().get('tool', 'claude/codex/vibe')
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute command. Make sure {tool_name} is installed and in PATH."
        )
    except Exception as e:
        logger.error(f"Failed to resume session {request.conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
