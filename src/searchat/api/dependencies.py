"""Shared dependencies for FastAPI routes.

This module must stay lightweight: avoid importing heavy ML/search modules at
import time. Heavy resources are initialized lazily and/or in background warmup.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from threading import Lock
from typing import Optional
from pathlib import Path

from searchat.services import BackupManager, PlatformManager
from searchat.config import Config, PathResolver
from searchat.api.readiness import get_readiness


logger = logging.getLogger(__name__)


# Global singletons (initialized on startup)
_config: Optional[Config] = None
_search_dir: Optional[Path] = None
_search_engine = None
_indexer = None
_backup_manager: Optional[BackupManager] = None
_platform_manager: Optional[PlatformManager] = None
_watcher = None
_duckdb_store = None

_service_lock = Lock()
_warmup_task: asyncio.Task[None] | None = None


# Shared state
projects_cache = None
stats_cache = None
watcher_stats = {"indexed_count": 0, "last_update": None}
indexing_state = {
    "in_progress": False,
    "operation": None,  # "manual_index" or "watcher"
    "started_at": None,
    "files_total": 0,
    "files_processed": 0
}


def initialize_services():
    """Initialize all services on app startup."""
    global _config, _search_dir, _backup_manager, _platform_manager, _duckdb_store

    readiness = get_readiness()
    readiness.set_component("services", "loading")
    try:
        _config = Config.load()
        _search_dir = PathResolver.get_shared_search_dir(_config)
        _backup_manager = BackupManager(_search_dir)
        _platform_manager = PlatformManager()

        from searchat.api.duckdb_store import DuckDBStore

        _duckdb_store = DuckDBStore(_search_dir, memory_limit_mb=_config.performance.memory_limit_mb)
        readiness.set_component("services", "ready")
    except Exception as e:
        readiness.set_component("services", "error", error=str(e))
        raise


def start_background_warmup() -> None:
    """Kick off background warmup (non-blocking, idempotent)."""
    global _warmup_task

    if _config is None or _search_dir is None:
        return

    readiness = get_readiness()
    readiness.mark_warmup_started()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Not in an event loop; cannot schedule async task.
        return

    if _warmup_task is not None and not _warmup_task.done():
        return

    _warmup_task = loop.create_task(_warmup_all())


async def _warmup_all() -> None:
    """Warm up heavy components in the background."""
    await asyncio.to_thread(_warmup_duckdb_parquet)
    # Warm search engine object first (cheap)
    await asyncio.to_thread(_ensure_search_engine)
    # Then warm semantic components (FAISS, metadata, embedder)
    await asyncio.to_thread(_warmup_semantic_components)


def _warmup_duckdb_parquet() -> None:
    readiness = get_readiness()
    started = time.perf_counter()
    try:
        readiness.set_component("duckdb", "loading")
        readiness.set_component("parquet", "loading")

        store = get_duckdb_store()
        store.validate_parquet_scan()

        readiness.set_component("duckdb", "ready")
        readiness.set_component("parquet", "ready")
    except Exception as e:
        msg = str(e)
        readiness.set_component("duckdb", "error", error=msg)
        readiness.set_component("parquet", "error", error=msg)
    finally:
        if os.getenv("SEARCHAT_PROFILE_WARMUP") == "1":
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info("Warmup: duckdb/parquet %.1fms", elapsed_ms)


def _warmup_semantic_components() -> None:
    readiness = get_readiness()
    started = time.perf_counter()
    try:
        engine = _ensure_search_engine()

        readiness.set_component("metadata", "loading")
        engine.ensure_metadata_ready()
        readiness.set_component("metadata", "ready")

        readiness.set_component("faiss", "loading")
        engine.ensure_faiss_loaded()
        readiness.set_component("faiss", "ready")

        readiness.set_component("embedder", "loading")
        engine.ensure_embedder_loaded()
        readiness.set_component("embedder", "ready")
    except Exception as e:
        msg = str(e)
        snap = readiness.snapshot()
        if snap.components.get("metadata") != "ready":
            readiness.set_component("metadata", "error", error=msg)
        if snap.components.get("faiss") != "ready":
            readiness.set_component("faiss", "error", error=msg)
        if snap.components.get("embedder") != "ready":
            readiness.set_component("embedder", "error", error=msg)
    finally:
        if os.getenv("SEARCHAT_PROFILE_WARMUP") == "1":
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info("Warmup: semantic components %.1fms", elapsed_ms)


def get_or_create_search_engine():
    """Get search engine, creating it if needed (cheap)."""
    return _ensure_search_engine()


def invalidate_search_index() -> None:
    """Clear caches and mark semantic components stale after indexing."""
    global projects_cache, stats_cache

    projects_cache = None
    stats_cache = None

    engine = _search_engine
    if engine is not None:
        engine.refresh_index()

    readiness = get_readiness()
    # Mark semantic components as not ready; warmup will rebuild.
    readiness.set_component("metadata", "idle")
    readiness.set_component("faiss", "idle")
    readiness.set_component("embedder", "idle")

    start_background_warmup()


def get_config() -> Config:
    """Get configuration singleton."""
    if _config is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _config


def get_search_dir() -> Path:
    """Get search directory path."""
    if _search_dir is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _search_dir


def get_duckdb_store():
    """Get DuckDBStore singleton."""
    if _duckdb_store is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _duckdb_store


def get_search_engine():
    """Get search engine singleton."""
    if _config is None or _search_dir is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    engine = _search_engine
    if engine is None:
        raise RuntimeError("Search engine not ready")
    return engine


def get_indexer():
    """Get indexer singleton."""
    if _config is None or _search_dir is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    idx = _indexer
    if idx is None:
        # Indexer is created lazily on first use.
        idx = _ensure_indexer()
    return idx


def get_backup_manager() -> BackupManager:
    """Get backup manager singleton."""
    if _backup_manager is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _backup_manager


def get_platform_manager() -> PlatformManager:
    """Get platform manager singleton."""
    if _platform_manager is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _platform_manager


def get_watcher():
    """Get watcher singleton (may be None if not started)."""
    return _watcher


def set_watcher(watcher):
    """Set watcher singleton."""
    global _watcher
    _watcher = watcher


def _ensure_search_engine():
    """Create and initialize search engine (blocking)."""
    global _search_engine
    readiness = get_readiness()

    if _config is None or _search_dir is None:
        raise RuntimeError("Services not initialized")

    with _service_lock:
        if _search_engine is not None:
            return _search_engine

        readiness.set_component("search_engine", "loading")
        try:
            from searchat.core.search_engine import SearchEngine

            _search_engine = SearchEngine(_search_dir, _config)
            readiness.set_component("search_engine", "ready")
        except Exception as e:
            readiness.set_component("search_engine", "error", error=str(e))
            raise
        return _search_engine


def _ensure_indexer():
    """Create indexer lazily (blocking)."""
    global _indexer
    readiness = get_readiness()

    if _config is None or _search_dir is None:
        raise RuntimeError("Services not initialized")

    with _service_lock:
        if _indexer is not None:
            return _indexer

        readiness.set_component("indexer", "loading")
        try:
            from searchat.core.indexer import ConversationIndexer

            _indexer = ConversationIndexer(_search_dir, _config)
            readiness.set_component("indexer", "ready")
        except Exception as e:
            readiness.set_component("indexer", "error", error=str(e))
            raise
        return _indexer


def trigger_search_engine_warmup() -> None:
    """Ensure warmup is scheduled and search engine initialization is triggered."""
    start_background_warmup()
