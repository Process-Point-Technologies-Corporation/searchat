"""
Shared dependencies for FastAPI routes.
Singleton pattern for heavy resources (search engine, indexer, embedder).
"""
from __future__ import annotations

import logging
import threading
from typing import Optional
from pathlib import Path

from searchat.core import ConversationWatcher
from searchat.core.unified_indexer import UnifiedIndexer
from searchat.core.unified_search import UnifiedSearchEngine
from searchat.palace.query import PalaceQuery
from searchat.palace.distiller import Distiller
from searchat.palace.llm import CLIDistillationLLM
from searchat.services import BackupManager, PlatformManager
from searchat.config import Config, PathResolver

logger = logging.getLogger(__name__)

# Global singletons (initialized on startup)
_config: Optional[Config] = None
_search_dir: Optional[Path] = None
_unified_search_engine: Optional[UnifiedSearchEngine] = None
_palace_query: Optional[PalaceQuery] = None
_backup_manager: Optional[BackupManager] = None
_platform_manager: Optional[PlatformManager] = None
_watcher: Optional[ConversationWatcher] = None
_distiller: Optional[Distiller] = None
_unified_indexer: Optional[UnifiedIndexer] = None


# Shared state
projects_cache = None
watcher_stats = {"indexed_count": 0, "last_update": None}
indexing_state = {
    "in_progress": False,
    "operation": None,  # "manual_index", "watcher", or "startup"
    "started_at": None,
    "files_total": 0,
    "files_processed": 0
}
indexing_lock = threading.Lock()


def get_projects_cache():
    """Get cached projects list if available."""
    return projects_cache


def set_projects_cache(projects):
    """Replace cached projects list."""
    global projects_cache
    projects_cache = projects


def reset_projects_cache():
    """Clear cached projects list."""
    global projects_cache
    projects_cache = None


class _LazyEmbedder:
    """Proxy that defers SentenceTransformer import + model load to first use."""
    __slots__ = ("_model_name", "_device", "_model", "_lock")

    def __init__(self, model_name: str, device: str):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def encode(self, *args, **kwargs):
        return self._load().encode(*args, **kwargs)

    def get_sentence_embedding_dimension(self):
        return self._load().get_sentence_embedding_dimension()


def initialize_services():
    """Initialize all services on app startup."""
    global _config, _search_dir, _unified_search_engine, _unified_indexer, _palace_query, _backup_manager, _platform_manager, _distiller

    _config = Config.load()
    _search_dir = PathResolver.get_shared_search_dir(_config)

    # Shared embedder — defers torch + model load to first .encode() call.
    # encode() is thread-safe (read-only weights after load).
    device = _config.embedding.get_device()
    embedder = _LazyEmbedder(_config.embedding.model, device)

    # Initialize unified search engine (DuckDB)
    unified_db_path = _search_dir / "data" / "searchat.duckdb"
    if unified_db_path.exists():
        _unified_search_engine = UnifiedSearchEngine(_search_dir, _config, embedder=embedder)
        _unified_indexer = UnifiedIndexer(_search_dir, _config, storage=_unified_search_engine.storage, embedder=embedder)
        logger.info("Unified search engine initialized")
    else:
        logger.warning(
            "Unified database not found at %s. "
            "Run unified indexer first.",
            unified_db_path,
        )

    # Initialize palace query and distiller (optional - may not exist for fresh installs)
    palace_db_path = _search_dir / "data" / "palace.duckdb"
    if palace_db_path.exists():
        from searchat.palace.storage import PalaceStorage
        palace_storage = PalaceStorage(_search_dir / "data")
        _palace_query = PalaceQuery(_search_dir / "data", _config, embedder=embedder, palace_storage=palace_storage)

        # Initialize distiller with the configured subscription-backed CLI (shared palace_storage + embedder)
        llm = CLIDistillationLLM(
            provider=_config.distillation.provider,
            model=_config.distillation.cli_model,
            prompt_template=_config.distillation.prompt,
        )
        # Shared storage — cursor-per-thread in UnifiedStorage handles isolation.
        # Each thread gets its own DuckDB cursor via threading.local(), so the
        # distiller's background thread won't interfere with watcher or API threads.
        unified_storage = _unified_search_engine.storage if _unified_search_engine is not None else None
        _distiller = Distiller(
            _search_dir, _config, llm=llm, unified_storage=unified_storage,
            embedder=embedder, palace_storage=palace_storage,
            indexing_lock=indexing_lock,
        )

    _backup_manager = BackupManager(_search_dir)
    _platform_manager = PlatformManager()


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


def get_unified_search_engine() -> Optional[UnifiedSearchEngine]:
    """Get unified search engine singleton (may be None if not initialized)."""
    return _unified_search_engine


def get_palace_query() -> Optional[PalaceQuery]:
    """Get palace query singleton (may be None if palace not initialized)."""
    return _palace_query


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


def get_watcher() -> Optional[ConversationWatcher]:
    """Get watcher singleton (may be None if not started)."""
    return _watcher


def set_watcher(watcher: Optional[ConversationWatcher]):
    """Set watcher singleton."""
    global _watcher
    _watcher = watcher


def get_unified_indexer() -> Optional[UnifiedIndexer]:
    """Get unified indexer singleton (may be None if not initialized)."""
    return _unified_indexer


def get_distiller() -> Optional[Distiller]:
    """Get distiller singleton (may be None if palace not initialized)."""
    return _distiller
