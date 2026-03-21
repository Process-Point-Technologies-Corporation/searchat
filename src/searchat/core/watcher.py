"""
Live file watcher for AI coding agent conversation directories.

Monitors for new/modified conversation files and triggers append-only indexing.

Supported agents:
- Claude Code: ~/.claude/projects/**/*.jsonl
- Codex: ~/.codex/sessions/**/*.jsonl
- Mistral Vibe: ~/.vibe/logs/session/*.json
"""

import logging
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
from queue import Queue, Empty

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

from searchat.agents import iter_providers
from searchat.config import PathResolver, Config

logger = logging.getLogger(__name__)


class ConversationEventHandler(FileSystemEventHandler):
    """Handles file system events for conversation files (JSONL and JSON)."""

    # Supported file extensions
    SUPPORTED_EXTENSIONS = ('.jsonl', '.json')

    def __init__(self, pending_queue: Queue, debounce_seconds: float = 2.0):
        super().__init__()
        self.pending_queue = pending_queue
        self.debounce_seconds = debounce_seconds
        self._last_event_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    # Directories to exclude from watching
    EXCLUDED_DIRS = ('plugins', 'hooks', 'commands', 'todos', 'settings',
                     'usage-sessions', 'telemetry', 'sessions', '_corrupt_jsonls',
                     'plans')

    # Root-level files to exclude (config/settings files)
    EXCLUDED_FILES = ('.credentials.json', 'stats-cache.json', 'settings.json',
                      'settings.local.json', 'settings.backup.json', 'history.jsonl',
                      'sessions-index.json', 'mcp-needs-auth-cache.json')

    def _should_process(self, path: str) -> bool:
        """Check if file should be processed (debounce rapid events)."""
        # Check for supported extensions
        if not any(path.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
            return False

        # Exclude specific root-level config files
        path_obj = Path(path)
        if path_obj.name in self.EXCLUDED_FILES:
            return False

        # Exclude subagent metadata files
        if path_obj.name.endswith('.meta.json'):
            return False

        # Exclude non-conversation directories
        path_lower = path.lower().replace('\\', '/')
        for excluded in self.EXCLUDED_DIRS:
            if f'/.claude/{excluded}/' in path_lower or f'\\.claude\\{excluded}\\' in path.lower():
                return False

        current_time = time.time()
        with self._lock:
            last_time = self._last_event_times.get(path, 0)
            if current_time - last_time < self.debounce_seconds:
                return False
            self._last_event_times[path] = current_time
            # Prune stale entries periodically to prevent unbounded growth
            if len(self._last_event_times) > 5000:
                cutoff = current_time - self.debounce_seconds * 10
                self._last_event_times = {
                    k: v for k, v in self._last_event_times.items() if v > cutoff
                }
            return True

    def on_created(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.info(f"New conversation detected: {event.src_path}")
            self.pending_queue.put(('created', event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.info(f"Conversation modified: {event.src_path}")
            self.pending_queue.put(('modified', event.src_path))


class ConversationWatcher:
    """
    Watches AI coding agent conversation directories for changes.

    Supported agents:
    - Claude Code: ~/.claude/projects/
    - Codex: ~/.codex/sessions/
    - Mistral Vibe: ~/.vibe/logs/session/

    Triggers append-only indexing when new or modified files are detected.
    Does NOT trigger on file deletions (preserves orphaned data).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        on_update: Optional[Callable[[List[str]], bool]] = None,
        batch_delay_seconds: float = 5.0,
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize the conversation watcher.

        Args:
            config: Configuration object
            on_update: Callback when files need indexing. Receives list of paths.
                Returns True on success (or non-recoverable error), False on
                lock contention (watcher retries next cycle).
            batch_delay_seconds: Wait time before processing batched updates
            debounce_seconds: Minimum time between events for same file
        """
        if config is None:
            config = Config.load()
        self.config = config

        self.path_resolver = PathResolver()
        self.provider_dirs = {
            provider.agent_id: provider.discover_dirs(config)
            for provider in iter_providers()
        }

        # Combine all watched directories
        self.watched_dirs = [
            path for paths in self.provider_dirs.values() for path in paths
        ]

        self.on_update = on_update
        self.batch_delay_seconds = batch_delay_seconds
        self.debounce_seconds = debounce_seconds
        self._cache_path = (
            PathResolver.get_shared_search_dir(config) / "data" / "watcher_file_cache.json"
        )

        self._pending_queue: Queue = Queue()
        self._observer: Optional[Observer] = None
        self._processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # File cache: populated by scan_all_files(), maintained by watcher events
        self._known_files: Set[str] = set()
        self._scan_complete = False
        self._files_dirty = False
        self._load_cached_file_list()

    def start(self) -> None:
        """Start watching directories for changes."""
        if self._running:
            logger.warning("Watcher already running")
            return

        self._stop_event.clear()
        self._running = True

        # Start file system observer
        self._observer = Observer()
        handler = ConversationEventHandler(
            self._pending_queue,
            debounce_seconds=self.debounce_seconds
        )

        for watch_dir in self.watched_dirs:
            if PathResolver.safe_exists(watch_dir):
                logger.info(f"Watching directory: {watch_dir}")
                self._observer.schedule(handler, str(watch_dir), recursive=True)

        self._observer.start()

        # Start batch processor thread
        self._processor_thread = threading.Thread(
            target=self._process_pending_updates,
            daemon=True,
            name="ConversationWatcherProcessor"
        )
        self._processor_thread.start()

        logger.info("Conversation watcher started")

    def stop(self) -> None:
        """Stop watching directories."""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
            self._processor_thread = None

        self._persist_known_files()
        logger.info("Conversation watcher stopped")

    def _process_pending_updates(self) -> None:
        """Background thread that batches and processes pending updates.

        Drains the event queue into a pending set. When batch_delay_seconds
        elapse since the last event, hands the set to _process_batch. If
        the callback returns False (lock contention), the set is retained
        and retried after another batch_delay_seconds back-off.
        """
        pending_paths: Set[str] = set()
        last_event_time = 0.0

        while not self._stop_event.is_set():
            try:
                _event_type, file_path = self._pending_queue.get(timeout=1.0)
                pending_paths.add(file_path)
                last_event_time = time.time()
            except Empty:
                pass

            if pending_paths and time.time() - last_event_time >= self.batch_delay_seconds:
                if self._process_batch(pending_paths):
                    pending_paths = set()
                else:
                    last_event_time = time.time()  # back off before retry

        # Process any remaining files on shutdown
        if pending_paths:
            self._process_batch(pending_paths)

    def _process_batch(self, file_paths: Set[str]) -> bool:
        """Process a batch of pending file updates.

        Returns True if the batch was consumed (success or non-recoverable error).
        Returns False if the callback deferred (lock contention — retry later).
        """
        if not file_paths:
            return True

        if self._scan_complete:
            before = len(self._known_files)
            self._known_files.update(file_paths)
            if len(self._known_files) > before:
                self._files_dirty = True
                self._persist_known_files()

        logger.info("Processing batch of %d files", len(file_paths))
        if self.on_update:
            return self.on_update(list(file_paths))
        return True

    @property
    def is_running(self) -> bool:
        """Check if watcher is currently running."""
        return self._running

    def get_watched_directories(self) -> List[Path]:
        """Get list of directories being watched."""
        return self.watched_dirs.copy()

    # =========================================================================
    # File cache (os.scandir-based fast scanning)
    # =========================================================================

    def scan_all_files(self) -> List[str]:
        """Fast full scan of all watched directories using os.scandir.

        On Windows, os.scandir reads file metadata from directory entries
        without extra stat syscalls — 2-5x faster than Path.rglob for
        flat-ish directory trees like ~/.claude/projects/*/*.jsonl.

        Caches results. Subsequent watcher events maintain the cache.
        """
        files: List[str] = []

        for agent_id, agent_dirs in self.provider_dirs.items():
            for agent_dir in agent_dirs:
                if not PathResolver.safe_exists(agent_dir):
                    continue
                if agent_id == "vibe":
                    files.extend(self._scandir_flat(agent_dir, ".json"))
                else:
                    files.extend(self._scandir_recursive(agent_dir, ".jsonl"))

        self._known_files = set(files)
        self._scan_complete = True
        self._persist_known_files()
        logger.info("File scan complete: %d files cached", len(files))
        return files

    def get_known_files(self) -> Optional[List[str]]:
        """Return cached file paths from last scan.

        Returns None if scan_all_files() has not been called yet,
        signalling the caller to fall back to rglob.
        """
        if not self._scan_complete:
            return None
        return list(self._known_files)

    @staticmethod
    def _is_excluded_path(path: str) -> bool:
        """Check if a cached path should be excluded under current filter rules."""
        name = Path(path).name
        if name in ConversationEventHandler.EXCLUDED_FILES:
            return True
        if name.endswith('.meta.json'):
            return True
        path_lower = path.lower().replace('\\', '/')
        for excluded in ConversationEventHandler.EXCLUDED_DIRS:
            if f'/.claude/{excluded}/' in path_lower:
                return True
        return False

    def _load_cached_file_list(self) -> None:
        """Restore known files from the persisted watcher cache if present."""
        try:
            if not self._cache_path.exists():
                return
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            files = data.get("files", [])
            if not isinstance(files, list):
                return
            raw_count = len(files)
            self._known_files = {
                str(path) for path in files
                if not self._is_excluded_path(str(path))
            }
            self._scan_complete = True
            pruned = raw_count - len(self._known_files)
            msg = f"Restored watcher file cache: {len(self._known_files)} files"
            if pruned:
                msg += f" ({pruned} excluded by current filters)"
            logger.info(msg)
        except Exception as e:
            logger.debug("Failed to load watcher cache from %s: %s", self._cache_path, e)

    def _persist_known_files(self) -> None:
        """Persist known files so the next startup can bootstrap from cache."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": datetime.utcnow().isoformat(),
                "files": sorted(self._known_files),
            }
            self._cache_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            self._files_dirty = False
        except Exception as e:
            logger.debug("Failed to persist watcher cache to %s: %s", self._cache_path, e)

    @staticmethod
    def _scandir_recursive(directory: Path, extension: str) -> List[str]:
        """Walk directory tree collecting files with given extension."""
        results: List[str] = []
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        if (
                            entry.name.endswith(extension)
                            and entry.name not in ConversationEventHandler.EXCLUDED_FILES
                        ):
                            results.append(entry.path)
                    elif entry.is_dir(follow_symlinks=False):
                        if entry.name not in ConversationEventHandler.EXCLUDED_DIRS:
                            results.extend(
                                ConversationWatcher._scandir_recursive(
                                    Path(entry.path), extension
                                )
                            )
        except (PermissionError, OSError):
            pass
        return results

    @staticmethod
    def _scandir_flat(directory: Path, extension: str) -> List[str]:
        """Scan single directory (non-recursive) for files with given extension."""
        results: List[str] = []
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False) and entry.name.endswith(extension):
                        results.append(entry.path)
        except (PermissionError, OSError):
            pass
        return results
