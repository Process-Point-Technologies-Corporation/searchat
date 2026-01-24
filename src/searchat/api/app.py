"""FastAPI application initialization and configuration."""
import os
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from searchat.core import ConversationWatcher
from searchat.core.logging_config import setup_logging, get_logger
from searchat.core.progress import LoggingProgressAdapter
from searchat.api.dependencies import (
    initialize_services,
    get_config,
    get_indexer,
    get_search_engine,
    get_watcher,
    set_watcher,
    projects_cache,
    watcher_stats,
    indexing_state,
)
from searchat.api.routers import (
    search_router,
    conversations_router,
    stats_router,
    indexing_router,
    backup_router,
    admin_router,
)
from searchat.config.constants import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PORT_SCAN_RANGE,
    ENV_PORT,
    ENV_HOST,
    ERROR_INVALID_PORT,
    ERROR_PORT_IN_USE,
)


# Cache HTML at module load for faster responses
_HTML_PATH = Path(__file__).parent.parent / "web" / "index.html"
_CACHED_HTML = _HTML_PATH.read_text(encoding='utf-8')


# Create FastAPI app
app = FastAPI(
    title="Searchat API",
    description="Semantic search for AI coding agent conversations",
    version="0.2.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "web" / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Register routers
app.include_router(search_router, prefix="/api", tags=["search"])
app.include_router(conversations_router, prefix="/api", tags=["conversations"])
app.include_router(stats_router, prefix="/api", tags=["statistics"])
app.include_router(indexing_router, prefix="/api", tags=["indexing"])
app.include_router(backup_router, prefix="/api/backup", tags=["backup"])
app.include_router(admin_router, prefix="/api", tags=["admin"])


def on_new_conversations(file_paths: List[str]) -> None:
    """Callback when watcher detects new conversation files."""
    global projects_cache, watcher_stats, indexing_state

    logger = get_logger(__name__)
    logger.info(f"Auto-indexing {len(file_paths)} new conversations")

    try:
        indexer = get_indexer()
        search_engine = get_search_engine()

        # Mark indexing in progress
        indexing_state["in_progress"] = True
        indexing_state["operation"] = "watcher"
        indexing_state["started_at"] = datetime.now().isoformat()
        indexing_state["files_total"] = len(file_paths)
        indexing_state["files_processed"] = 0

        # Use logging-based progress for background task
        progress = LoggingProgressAdapter()

        stats = indexer.index_append_only(file_paths, progress)

        if stats.new_conversations > 0:
            # Reload search engine to pick up new data
            search_engine._initialize()
            projects_cache = None  # Clear cache

            watcher_stats["indexed_count"] += stats.new_conversations
            watcher_stats["last_update"] = datetime.now().isoformat()

            logger.info(
                f"Indexed {stats.new_conversations} new conversations "
                f"in {stats.update_time_seconds:.2f}s"
            )
    except Exception as e:
        logger.error(f"Failed to index new conversations: {e}")
    finally:
        # Mark indexing complete
        indexing_state["in_progress"] = False
        indexing_state["operation"] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services and start the file watcher on server startup."""
    # IMPORTANT: Initialize services FIRST (loads config)
    initialize_services()

    # THEN get config and setup logging
    config = get_config()
    setup_logging(config.logging)
    logger = get_logger(__name__)

    # Start file watcher
    indexer = get_indexer()

    watcher = ConversationWatcher(
        config=config,
        on_update=on_new_conversations,
        batch_delay_seconds=5.0,
        debounce_seconds=2.0,
    )

    # Initialize with already-indexed files
    indexed_paths = indexer.get_indexed_file_paths()
    watcher.set_indexed_files(indexed_paths)

    watcher.start()
    set_watcher(watcher)

    logger.info(f"Live watcher started, monitoring {len(watcher.get_watched_directories())} directories")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the file watcher on server shutdown."""
    watcher = get_watcher()
    if watcher:
        watcher.stop()
        set_watcher(None)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    return HTMLResponse(_CACHED_HTML)


@app.get("/conversation/{conversation_id}", response_class=HTMLResponse)
async def serve_conversation_page(conversation_id: str):
    """Serve HTML page for viewing a specific conversation."""
    # For now, serve the same main page (it handles conversation viewing via client-side routing)
    return HTMLResponse(_CACHED_HTML)


def main():
    """Run the server with configurable host and port."""
    import uvicorn
    import socket

    # Get host from environment or use default
    host = os.getenv(ENV_HOST, DEFAULT_HOST)

    # Get port from environment or scan for available port
    env_port = os.getenv(ENV_PORT)
    if env_port:
        try:
            port = int(env_port)
            if not (1 <= port <= 65535):
                print(ERROR_INVALID_PORT.format(port=port))
                return
        except ValueError:
            print(ERROR_INVALID_PORT.format(port=env_port))
            return
    else:
        # Scan for available port in range
        port, max_port = PORT_SCAN_RANGE

        while port <= max_port:
            try:
                # Test if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((host, port))
                # Port is available
                break
            except OSError:
                port += 1

        if port > max_port:
            print(ERROR_PORT_IN_USE.format(
                start=PORT_SCAN_RANGE[0],
                end=PORT_SCAN_RANGE[1],
                port=port
            ))
            return

    print(f"Starting Searchat server...")
    print(f"  URL: http://localhost:{port}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print()
    print("Press Ctrl+C to stop")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
