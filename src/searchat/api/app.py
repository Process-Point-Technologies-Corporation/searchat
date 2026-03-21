"""FastAPI application initialization and configuration."""
import asyncio
import logging
import logging.handlers
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from searchat.config import PathResolver
from searchat.core import ConversationWatcher
from searchat.core.conversation_filter import exclude_automated_conversations
from searchat.api.dependencies import (
    initialize_services,
    get_config,
    get_unified_indexer,
    get_unified_search_engine,
    get_palace_query,
    get_watcher,
    set_watcher,
    get_distiller,
    reset_projects_cache,
    watcher_stats,
    indexing_state,
    indexing_lock,
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

_CONVERSATION_HTML_PATH = Path(__file__).parent.parent / "web" / "conversation.html"
_CACHED_CONVERSATION_HTML = _CONVERSATION_HTML_PATH.read_text(encoding='utf-8')


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

_distillation_scheduler_lock = threading.Lock()
_distillation_requested = False
_distillation_thread: threading.Thread | None = None


def on_new_conversations(file_paths: List[str]) -> bool:
    """Callback when watcher detects new or modified conversation files.

    Returns True if the batch was consumed (success or non-recoverable error).
    Returns False on lock contention — watcher retries next cycle.
    """
    global watcher_stats, indexing_state

    logger = logging.getLogger(__name__)
    logger.info("Watcher detected %d files", len(file_paths))

    if not indexing_lock.acquire(blocking=False):
        logger.info("Deferring %d watcher files — indexing lock held", len(file_paths))
        return False

    try:
        unified_indexer = get_unified_indexer()
        if unified_indexer is None:
            return True

        new_files, changed_files = unified_indexer.detect_changed_files(file_paths)

        # Filter only new files (changed were already filtered on first index)
        config = get_config()
        new_files = exclude_automated_conversations(
            new_files, config.paths.excluded_conversations_dir, config,
        )

        # Policy gate: suppress changed files when reindex_on_modification is off
        if not config.indexing.reindex_on_modification:
            changed_files = []

        if not new_files and not changed_files:
            return True

        # Mark indexing in progress
        indexing_state["in_progress"] = True
        indexing_state["operation"] = "watcher"
        indexing_state["started_at"] = datetime.now().isoformat()
        indexing_state["files_total"] = len(new_files) + len(changed_files)
        indexing_state["files_processed"] = 0

        stats = unified_indexer.index_from_source_files(
            new_files, changed_file_paths=changed_files,
        )

        new_count = stats["new_conversations"]
        updated_count = stats["updated_conversations"]
        logger.info(
            "Watcher indexing: %d new, %d updated, %d exchanges",
            new_count, updated_count, stats["exchanges_created"],
        )
        invalid_count = stats.get("invalid_transcript_count", 0)
        if invalid_count:
            examples = stats.get("invalid_transcript_examples", [])
            sample = ", ".join(Path(item["file_path"]).name for item in examples)
            suffix = f" Examples: {sample}" if sample else ""
            logger.info(
                "Watcher indexing skipped %d invalid transcript files.%s",
                invalid_count,
                suffix,
            )
        skipped_known_invalid = stats.get("skipped_known_invalid", 0)
        if skipped_known_invalid:
            logger.info(
                "Watcher indexing skipped %d unchanged invalid transcript files.",
                skipped_known_invalid,
            )

        if stats["exchanges_created"] > 0:
            _schedule_fts_rebuild(unified_indexer.storage, reason="watcher")

        watcher_stats["indexed_count"] += new_count + updated_count
        watcher_stats["last_update"] = datetime.now().isoformat()
        reset_projects_cache()
    except Exception as e:
        logger.exception("Failed to index new conversations: %s", e)
        return False
    finally:
        indexing_state["in_progress"] = False
        indexing_state["operation"] = None
        indexing_lock.release()

    # Trigger distillation for new/updated conversations after indexing succeeds.
    distiller = get_distiller()
    if distiller:
        try:
            _schedule_background_distillation(distiller, reason="watcher")
        except Exception as e:
            logger.warning("Failed to schedule distillation after watcher indexing: %s", e)

    return True


def _get_cached_watcher_files(watcher) -> List[str] | None:
    """Return persisted watcher file cache when the watcher exposes a real collection."""
    if watcher is None or not hasattr(watcher, "get_known_files"):
        return None

    cached_files = watcher.get_known_files()
    if isinstance(cached_files, (list, tuple, set)):
        return list(cached_files)
    return None


async def _background_warmup():
    """Run warmup searches in background without blocking server startup."""
    logger = logging.getLogger(__name__)
    try:
        from searchat.models import AlgorithmType

        loop = asyncio.get_running_loop()
        unified_engine = get_unified_search_engine()
        palace_query = get_palace_query()
        config = get_config()
        warmup_mode = (config.performance.startup_warmup_mode or "keyword").strip().lower()

        if warmup_mode == "none":
            logger.info("Background warmup skipped (startup_warmup_mode=none)")
            return

        if unified_engine:
            algorithm = (
                AlgorithmType.SEMANTIC
                if warmup_mode == "semantic"
                else AlgorithmType.KEYWORD
            )
            await loop.run_in_executor(
                None, lambda: unified_engine.search("warmup", algorithm=algorithm, limit=1)
            )
        if palace_query and warmup_mode == "semantic":
            await loop.run_in_executor(
                None, lambda: palace_query.search_hybrid("warmup", limit=1)
            )

        logger.info("Search engine warmed up (background, mode=%s)", warmup_mode)
    except Exception as e:
        logger.warning(f"Background warmup failed (non-fatal): {e}")


async def _background_scan_and_start_watcher():
    """Scan files, detect changes, index, then start the watcher observer.

    Runs scan_all_files() BEFORE the observer starts so directory walking
    does not trigger watchdog events. Change detection and indexing complete
    before the observer begins, eliminating the race between the watcher
    callback and this startup path.
    """
    logger = logging.getLogger(__name__)
    loop = asyncio.get_running_loop()
    unified_indexer = get_unified_indexer()
    watcher = get_watcher()
    config = get_config()

    if unified_indexer is None or watcher is None:
        return

    def _scan_detect_index():
        cached_files = _get_cached_watcher_files(watcher)
        if cached_files is not None:
            logger.info("Startup cache bootstrap: %d cached files", len(cached_files))
            return _index_detected_files(unified_indexer, config, cached_files, operation="startup")

        all_files = watcher.scan_all_files()
        return _index_detected_files(unified_indexer, config, all_files, operation="startup")

    should_rebuild_fts = False

    try:
        stats = await loop.run_in_executor(None, _scan_detect_index)
        if stats is not None:
            new_c = stats["new_conversations"]
            upd_c = stats["updated_conversations"]
            logger.info(
                "Startup catch-up: %d new, %d updated, %d exchanges",
                new_c, upd_c, stats["exchanges_created"],
            )
            invalid_count = stats.get("invalid_transcript_count", 0)
            if invalid_count:
                examples = stats.get("invalid_transcript_examples", [])
                sample = ", ".join(Path(item["file_path"]).name for item in examples)
                suffix = f" Examples: {sample}" if sample else ""
                logger.info(
                    "Startup catch-up skipped %d invalid transcript files.%s",
                    invalid_count,
                    suffix,
                )
            skipped_known_invalid = stats.get("skipped_known_invalid", 0)
            if skipped_known_invalid:
                logger.info(
                    "Startup catch-up skipped %d unchanged invalid transcript files.",
                    skipped_known_invalid,
                )
            should_rebuild_fts = stats["exchanges_created"] > 0
        else:
            logger.info("Startup catch-up: no changed files detected")
    except Exception as e:
        logger.exception("Background startup catch-up failed: %s", e)

    # 3. Start observer even if catch-up failed, so live updates still work.
    try:
        watcher.start()
        logger.info(
            "Live watcher started, monitoring %d directories",
            len(watcher.get_watched_directories()),
        )
    except Exception as e:
        set_watcher(None)
        logger.exception("Live watcher failed to start: %s", e)
        return

    if should_rebuild_fts:
        _schedule_fts_rebuild(unified_indexer.storage, reason="startup")

    if _get_cached_watcher_files(watcher) is not None:
        asyncio.create_task(
            _background_reconcile_watcher_cache(watcher, unified_indexer, config)
        )


async def _background_reconcile_watcher_cache(watcher, unified_indexer, config) -> None:
    """Run a full scan after startup cache bootstrap to catch offline file changes."""
    logger = logging.getLogger(__name__)
    loop = asyncio.get_running_loop()

    def _reconcile():
        all_files = watcher.scan_all_files()
        return all_files, _index_detected_files(
            unified_indexer, config, all_files, operation="reconcile",
        )

    try:
        all_files, stats = await loop.run_in_executor(None, _reconcile)
        if stats is None:
            logger.info("Startup reconciliation: no additional changes detected")
            return

        logger.info(
            "Startup reconciliation: %d new, %d updated, %d exchanges",
            stats["new_conversations"],
            stats["updated_conversations"],
            stats["exchanges_created"],
        )
        invalid_count = stats.get("invalid_transcript_count", 0)
        if invalid_count:
            logger.info(
                "Startup reconciliation skipped %d invalid transcript files.",
                invalid_count,
            )
        skipped_known_invalid = stats.get("skipped_known_invalid", 0)
        if skipped_known_invalid:
            logger.info(
                "Startup reconciliation skipped %d unchanged invalid transcript files.",
                skipped_known_invalid,
            )
        if stats["exchanges_created"] > 0:
            _schedule_fts_rebuild(unified_indexer.storage, reason="reconcile")
    except Exception as e:
        logger.exception("Startup reconciliation scan failed: %s", e)


def _index_detected_files(unified_indexer, config, all_files: List[str], operation: str):
    """Detect changed files and index them under the shared indexing lock."""
    logger = logging.getLogger(__name__)
    if not indexing_lock.acquire(blocking=False):
        logger.info("%s scan skipped — indexing already in progress", operation.capitalize())
        return None

    try:
        indexing_state["in_progress"] = True
        indexing_state["operation"] = operation
        indexing_state["started_at"] = datetime.now().isoformat()

        # Force a fresh DB cursor so detect_changed_files sees the latest
        # committed data (not a stale snapshot from a previous thread run).
        unified_indexer.storage._close_thread_cursor()

        new_files, changed_files = unified_indexer.detect_changed_files(all_files)
        if not new_files and not changed_files:
            return None
        new_files_filtered = exclude_automated_conversations(
            new_files, config.paths.excluded_conversations_dir, config,
        )

        indexing_state["files_total"] = len(new_files_filtered) + len(changed_files)
        indexing_state["files_processed"] = 0

        return unified_indexer.index_from_source_files(
            new_files_filtered, changed_file_paths=changed_files,
        )
    finally:
        indexing_state["in_progress"] = False
        indexing_state["operation"] = None
        indexing_lock.release()


def _run_fts_rebuild(storage, reason: str) -> None:
    """Rebuild FTS in the background, serialized against indexer transactions."""
    logger = logging.getLogger(__name__)
    if not indexing_lock.acquire(timeout=60):
        logger.warning("FTS rebuild skipped — indexing lock held for >60s (%s)", reason)
        return
    try:
        storage.create_fts_index()
    except Exception as e:
        logger.exception("Background FTS rebuild failed after %s indexing: %s", reason, e)
    finally:
        # Close this thread's DuckDB cursor so no implicit transaction lingers
        # and conflicts with the next writer on a different thread.
        storage._close_thread_cursor()
        indexing_lock.release()


_fts_rebuild_timer: threading.Timer | None = None
_fts_timer_lock = threading.Lock()


def _schedule_fts_rebuild(storage, reason: str) -> None:
    """Schedule a debounced background FTS rebuild (30s coalescing window)."""
    global _fts_rebuild_timer
    with _fts_timer_lock:
        if _fts_rebuild_timer is not None:
            _fts_rebuild_timer.cancel()
        _fts_rebuild_timer = threading.Timer(
            30.0, _run_fts_rebuild, args=(storage, reason),
        )
        _fts_rebuild_timer.daemon = True
        _fts_rebuild_timer.start()


def _run_distillation_loop(distiller) -> None:
    """Drain coalesced background distillation requests in a single daemon thread."""
    global _distillation_requested, _distillation_thread
    logger = logging.getLogger(__name__)

    while True:
        with _distillation_scheduler_lock:
            _distillation_requested = False

        try:
            stats = distiller.distill_all_pending()
            logger.info(
                "Background distillation complete: %d conversations, %d objects in %.2fs",
                stats.conversations_processed,
                stats.objects_created,
                stats.distillation_time_seconds,
            )
        except Exception as e:
            logger.warning("Background distillation failed (non-fatal): %s", e)

        with _distillation_scheduler_lock:
            if not _distillation_requested:
                _distillation_thread = None
                return


def _schedule_background_distillation(distiller, reason: str) -> bool:
    """Queue background distillation without blocking watcher or startup indexing."""
    global _distillation_requested, _distillation_thread
    logger = logging.getLogger(__name__)

    with _distillation_scheduler_lock:
        _distillation_requested = True
        if _distillation_thread is not None and _distillation_thread.is_alive():
            logger.info(
                "Background distillation already running; coalescing %s request",
                reason,
            )
            return False

        _distillation_thread = threading.Thread(
            target=_run_distillation_loop,
            args=(distiller,),
            daemon=True,
            name="SearchatDistill",
        )
        _distillation_thread.start()

    logger.info("Scheduled background distillation after %s", reason)
    return True


def _setup_logging(log_dir: Path) -> None:
    """Configure logging to both console and file with gzip rotation."""
    import gzip
    import shutil

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "searchat.log"

    def gzip_rotator(source, dest):
        with open(source, "rb") as f_in:
            with gzip.open(f"{dest}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)

    def gzip_namer(name):
        return name + ".gz"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,  # 5MB
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.rotator = gzip_rotator
    file_handler.namer = gzip_namer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            file_handler,
        ],
    )


@app.on_event("startup")
async def startup_event():
    """Initialize services and start the file watcher on server startup."""
    # Initialize services in thread pool to keep event loop responsive
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, initialize_services)

    # Setup logging to ~/.searchat/logs/
    from searchat.api.dependencies import get_search_dir
    _setup_logging(get_search_dir() / "logs")
    logger = logging.getLogger(__name__)
    logger.info("Searchat server starting...")

    # Warmup: run searches in background to not block server startup
    asyncio.create_task(_background_warmup())

    # Catch-up: clear transient LLM error skips and distill pending conversations
    distiller = get_distiller()
    if distiller:
        cleared = distiller.storage.clear_llm_error_skips()
        if cleared:
            logger.info("Cleared %d LLM-error skipped conversations for retry", cleared)
        pending = distiller.list_pending_conversations()
        if pending:
            logger.info(f"Starting background catch-up: {len(pending)} conversations...")
            _schedule_background_distillation(distiller, reason="startup catch-up")

    # Create watcher (but don't start observer yet — background task does that
    # after scan + change detection to avoid event flood and race condition)
    config = get_config()

    watcher = ConversationWatcher(
        config=config,
        on_update=on_new_conversations,
        batch_delay_seconds=5.0,
        debounce_seconds=2.0,
    )

    set_watcher(watcher)

    # Background: scan files, detect changes, index, THEN start watcher observer
    asyncio.create_task(_background_scan_and_start_watcher())


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
    return HTMLResponse(_CACHED_CONVERSATION_HTML)


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
