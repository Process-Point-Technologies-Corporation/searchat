"""Conversation endpoints - listing, viewing, and session resume."""
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Query, HTTPException

from searchat.api.models import (
    SearchResultResponse,
    ConversationMessage,
    ConversationResponse,
    ResumeRequest,
)
import searchat.api.dependencies as deps

from searchat.api.dependencies import get_platform_manager


router = APIRouter()
logger = logging.getLogger(__name__)


async def read_file_async(file_path: str, encoding: str = 'utf-8') -> str:
    """Read file asynchronously to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: Path(file_path).read_text(encoding=encoding))


@router.get("/conversations/all")
async def get_all_conversations(
    sort_by: str = Query("length", description="Sort by: length, date_newest, date_oldest, title"),
    project: Optional[str] = Query(None, description="Filter by project"),
    date: Optional[str] = Query(None, description="Date filter: today, week, month, or custom"),
    date_from: Optional[str] = Query(None, description="Custom date from (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Custom date to (YYYY-MM-DD)")
):
    """Get all conversations with sorting and filtering."""
    try:
        store = deps.get_duckdb_store()

        # Handle date filtering
        date_from_dt = None
        date_to_dt = None
        if date == "custom" and (date_from or date_to):
            # Custom date range
            if date_from:
                date_from_dt = datetime.fromisoformat(date_from)
            if date_to:
                # Add 1 day to include the entire end date
                date_to_dt = datetime.fromisoformat(date_to) + timedelta(days=1)
        elif date:
            # Preset date ranges
            now = datetime.now()
            if date == "today":
                date_from_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
                date_to_dt = now
            elif date == "week":
                date_from_dt = now - timedelta(days=7)
                date_to_dt = now
            elif date == "month":
                date_from_dt = now - timedelta(days=30)
                date_to_dt = now

        rows = store.list_conversations(
            sort_by=sort_by,
            project_id=project,
            date_from=date_from_dt,
            date_to=date_to_dt,
        )

        response_results = []
        for row in rows:
            file_path = row["file_path"]
            full_text = row.get("full_text") or ""
            response_results.append(
                SearchResultResponse(
                    conversation_id=row["conversation_id"],
                    project_id=row["project_id"],
                    title=row["title"],
                    created_at=row["created_at"].isoformat(),
                    updated_at=row["updated_at"].isoformat(),
                    message_count=row["message_count"],
                    file_path=file_path,
                    snippet=full_text[:200] + ("..." if len(full_text) > 200 else ""),
                    score=0.0,
                    message_start_index=None,
                    message_end_index=None,
                    source="WSL" if "/home/" in file_path or "wsl" in file_path.lower() else "WIN",
                )
            )

        return {
            "results": response_results,
            "total": len(response_results),
            "search_time_ms": 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all messages."""
    try:
        store = deps.get_duckdb_store()
        conv = store.get_conversation_meta(conversation_id)
        if conv is None:
            logger.warning(f"Conversation not found in index: {conversation_id}")
            raise HTTPException(status_code=404, detail="Conversation not found in index")
        file_path = conv["file_path"]

        # Check if file exists
        if not Path(file_path).exists():
            logger.error(f"Conversation file not found: {file_path} (conversation_id: {conversation_id})")
            raise HTTPException(
                status_code=404,
                detail=f"Conversation file not found. The file may have been moved or deleted: {file_path}"
            )

        # Load messages from JSONL file (async to avoid blocking)
        try:
            content = await read_file_async(file_path)
            lines = [json.loads(line) for line in content.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in conversation file {file_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse conversation file (invalid JSON)"
            )
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {file_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to read conversation file (encoding error)"
            )

        messages = []
        for entry in lines:
            if entry.get('type') in ('user', 'assistant'):
                raw_content = entry.get('message', {}).get('content', '')
                if isinstance(raw_content, str):
                    content = raw_content
                elif isinstance(raw_content, list):
                    content = '\n\n'.join(
                        block.get('text', '')
                        for block in raw_content
                        if block.get('type') == 'text'
                    )
                else:
                    content = ''

                if content:
                    messages.append(ConversationMessage(
                        role=entry.get('type'),
                        content=content,
                        timestamp=entry.get('timestamp', '')
                    ))

        logger.info(f"Successfully loaded conversation {conversation_id} with {len(messages)} messages")

        return ConversationResponse(
            conversation_id=conversation_id,
            title=conv["title"],
            project_id=conv["project_id"],
            file_path=conv["file_path"],
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
        store = deps.get_duckdb_store()
        platform_manager = get_platform_manager()

        conv = store.get_conversation_meta(request.conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        file_path = conv["file_path"]
        session_id = conv["conversation_id"]

        # Extract working directory from conversation file
        cwd = None

        if file_path.endswith('.jsonl'):
            # Claude Code - read lines until we find one with cwd (async)
            tool = 'claude'
            content = await read_file_async(file_path)
            for line in content.splitlines():
                if line.strip():
                    entry = json.loads(line)
                    if 'cwd' in entry:
                        cwd = entry['cwd']
                        break
            command = f'claude --resume {session_id}'
        elif file_path.endswith('.json'):
            # Vibe - read JSON metadata (async)
            tool = 'vibe'
            content = await read_file_async(file_path)
            data = json.loads(content)
            cwd = data.get('metadata', {}).get('environment', {}).get('working_directory', None)
            command = f'vibe --resume {session_id}'
        else:
            raise HTTPException(status_code=400, detail=f"Unknown conversation format: {file_path}")

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
        # Command not found (claude or vibe not installed)
        logger.error(f"Command not found: {e}")
        tool_name = locals().get('tool', 'claude/vibe')
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute command. Make sure {tool_name} is installed and in PATH."
        )
    except Exception as e:
        logger.error(f"Failed to resume session {request.conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
