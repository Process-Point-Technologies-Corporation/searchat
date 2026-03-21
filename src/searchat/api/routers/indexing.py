"""Indexing endpoints - manual reindex and index missing conversations."""
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from searchat.agents import iter_providers
from searchat.core.conversation_filter import exclude_automated_conversations
from searchat.api.dependencies import (
    get_config,
    get_unified_indexer,
    get_distiller,
    get_watcher,
    reset_projects_cache,
    indexing_state,
    indexing_lock,
)


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/reindex")
async def reindex():
    """Rebuild the search index - DISABLED FOR DATA SAFETY."""
    # SAFETY GUARD: Block all reindexing to protect irreplaceable conversation data
    raise HTTPException(
        status_code=403,
        detail="BLOCKED: Reindexing disabled to protect irreplaceable conversation data. "
               "Source JSONLs are missing - rebuilding would cause data loss."
    )


@router.post("/index_missing")
async def index_missing():
    """Index conversations that aren't already indexed (append-only, safe).

    Indexes directly from JSONL/JSON source files into unified DuckDB.
    """
    global indexing_state

    unified_indexer = get_unified_indexer()
    if unified_indexer is None:
        raise HTTPException(
            status_code=503,
            detail="Unified indexer not available. DuckDB not initialized.",
        )

    if not indexing_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail=f"Indexing already in progress: {indexing_state.get('operation', 'unknown')}",
        )

    try:
        config = get_config()

        # Use watcher's cached file list if available (instant),
        # otherwise fall back to rglob (1-5s).
        watcher = get_watcher()
        cached = watcher.get_known_files() if watcher and watcher.is_running else None

        if cached is not None:
            all_files = cached
        else:
            all_files = []
            for provider in iter_providers():
                for root_dir in provider.discover_dirs(config):
                    try:
                        pattern = "*.json" if provider.agent_id == "vibe" else "*.jsonl"
                        files = list(root_dir.glob(pattern)) if provider.agent_id == "vibe" else list(root_dir.rglob(pattern))
                        all_files.extend([str(f) for f in files])
                    except Exception as e:
                        logger.warning("Error scanning %s: %s", root_dir, e)

        total_scanned = len(all_files)

        # Detect new and changed files (replaces conversation_id pre-dedup)
        new_files, changed_files = unified_indexer.detect_changed_files(all_files)
        already_unchanged = total_scanned - len(new_files) - len(changed_files)

        # Pre-scan filter: only filter NEW files (changed were already filtered on first index)
        excluded_dir = config.paths.excluded_conversations_dir
        pre_filter_count = len(new_files)
        new_files = exclude_automated_conversations(new_files, excluded_dir, config)
        excluded_automated = pre_filter_count - len(new_files)

        # Mark indexing in progress
        indexing_state["in_progress"] = True
        indexing_state["operation"] = "manual_index"
        indexing_state["started_at"] = datetime.now().isoformat()
        indexing_state["files_total"] = len(new_files) + len(changed_files)
        indexing_state["files_processed"] = 0

        # Index new + re-index changed
        stats = unified_indexer.index_from_source_files(
            new_files, changed_file_paths=changed_files,
        )

        # Clear projects cache
        reset_projects_cache()

        # Defer FTS rebuild to background (debounced, serialized against indexing)
        if stats["exchanges_created"] > 0:
            from searchat.api.app import _schedule_fts_rebuild
            _schedule_fts_rebuild(unified_indexer.storage, reason="manual_index")

        new_count = stats["new_conversations"]
        updated_count = stats["updated_conversations"]
        error_count = stats["skipped_errors"]

        if error_count > 0:
            message = f"Added {new_count}, updated {updated_count} conversations, {error_count} failed"
            logger.warning(
                "Indexing complete: %d added, %d updated, %d failed",
                new_count, updated_count, error_count,
            )
        else:
            message = f"Added {new_count}, updated {updated_count} conversations"
            logger.info("Indexing complete: %d added, %d updated", new_count, updated_count)

        return {
            "success": True,
            "new_conversations": new_count,
            "updated_conversations": updated_count,
            "changed_detected": len(changed_files),
            "failed_conversations": error_count,
            "total_files": total_scanned,
            "already_indexed": already_unchanged,
            "excluded_automated": excluded_automated,
            "time_seconds": round(stats["time_seconds"], 2),
            "message": message,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error indexing missing conversations: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        indexing_state["in_progress"] = False
        indexing_state["operation"] = None
        indexing_lock.release()


@router.post("/distill")
async def distill_pending():
    """Distill all pending conversations using Haiku LLM."""
    distiller = get_distiller()
    if distiller is None:
        raise HTTPException(
            status_code=503,
            detail="Distiller not available. Palace not initialized."
        )

    # Clear LLM error skips before distilling — these are transient failures
    # from previous provider issues that should be retried.
    cleared = distiller.storage.clear_llm_error_skips()
    if cleared:
        logger.info("Cleared %d LLM-error skipped conversations for retry", cleared)

    stats = distiller.distill_all_pending()
    return {
        "success": True,
        "conversations_processed": stats.conversations_processed,
        "objects_created": stats.objects_created,
        "rooms_created": stats.rooms_created,
        "time_seconds": round(stats.distillation_time_seconds, 2),
        "retried_from_skip": cleared,
    }


@router.post("/distill/{conversation_id}")
async def distill_conversation(conversation_id: str):
    """Distill a single conversation by ID."""
    distiller = get_distiller()
    if distiller is None:
        raise HTTPException(
            status_code=503,
            detail="Distiller not available. Palace not initialized."
        )

    try:
        objects = distiller.distill_conversation(conversation_id)
        return {
            "success": True,
            "conversation_id": conversation_id,
            "objects_created": len(objects),
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
