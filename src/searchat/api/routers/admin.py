"""Admin endpoints - server shutdown and watcher status."""
import os
import signal
import logging

from fastapi import APIRouter, BackgroundTasks

from searchat.api.dependencies import get_watcher, watcher_stats, indexing_state


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/watcher/status")
async def get_watcher_status():
    """Get live file watcher status."""
    watcher = get_watcher()

    return {
        "running": watcher.is_running if watcher else False,
        "watched_directories": [str(d) for d in watcher.get_watched_directories()] if watcher else [],
        "indexed_since_start": watcher_stats["indexed_count"],
        "last_update": watcher_stats["last_update"],
    }


@router.get("/indexing/status")
async def get_indexing_status():
    """Get current indexing status for UI polling."""
    return {
        "in_progress": indexing_state["in_progress"],
        "operation": indexing_state["operation"],
        "started_at": indexing_state["started_at"],
        "files_total": indexing_state["files_total"],
        "files_processed": indexing_state["files_processed"],
    }


@router.post("/shutdown")
async def shutdown_server(background_tasks: BackgroundTasks, force: bool = False):
    """Shutdown the server. DuckDB is transactional so no data corruption risk."""
    logger.info("Server shutdown requested via API")

    def shutdown():
        """Shutdown function to run in background."""
        import time
        time.sleep(0.5)  # Give time for response to be sent

        # Stop watcher if running
        watcher = get_watcher()
        if watcher and watcher.is_running:
            logger.info("Stopping file watcher...")
            watcher.stop()

        logger.info("Shutting down server...")
        # Use os._exit() for immediate exit on all platforms (Windows compatible)
        os._exit(0)

    background_tasks.add_task(shutdown)

    return {
        "success": True,
        "message": "Server shutting down"
    }
