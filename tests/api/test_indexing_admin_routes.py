"""Unit tests for indexing and admin API routes."""
import asyncio
import threading

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from fastapi.testclient import TestClient

from searchat.api.app import app
from searchat.api.app import (
    _background_scan_and_start_watcher,
    _background_warmup,
    _schedule_background_distillation,
    on_new_conversations,
)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_config():
    """Mock Config."""
    mock = Mock()
    return mock


@pytest.fixture
def mock_unified_indexer():
    """Mock UnifiedIndexer."""
    mock = Mock()
    mock.storage.get_all_conversation_ids.return_value = []
    mock.detect_changed_files.return_value = (
        ["new1.jsonl", "new2.jsonl", "new3.jsonl"],  # new_files
        ["changed1.jsonl"],  # changed_files
    )
    mock.index_from_source_files.return_value = {
        "new_conversations": 5,
        "updated_conversations": 1,
        "exchanges_created": 20,
        "embeddings_created": 20,
        "skipped_already_indexed": 2,
        "skipped_errors": 0,
        "total_files": 4,
        "time_seconds": 1.5,
    }
    return mock


@pytest.fixture
def mock_watcher():
    """Mock ConversationWatcher."""
    mock = Mock()
    mock.is_running = True
    mock.get_watched_directories.return_value = [Path("/watched/dir1"), Path("/watched/dir2")]
    mock.stop = Mock()
    return mock


# ============================================================================
# INDEXING ENDPOINT TESTS
# ============================================================================

@pytest.mark.unit
class TestReindexEndpoint:
    """Tests for POST /api/reindex endpoint."""

    def test_reindex_blocked_for_safety(self, client):
        """Test that reindex is blocked for data safety."""
        response = client.post("/api/reindex")

        assert response.status_code == 403
        assert "BLOCKED" in response.json()["detail"]
        assert "data loss" in response.json()["detail"].lower()


@pytest.mark.unit
class TestIndexMissingEndpoint:
    """Tests for POST /api/index_missing endpoint."""

    def test_index_missing_success(self, client, mock_config, mock_unified_indexer, tmp_path):
        """Test indexing missing conversations successfully."""
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()
        (claude_dir / "conv1.jsonl").write_text('{"type": "user"}')
        (claude_dir / "conv2.jsonl").write_text('{"type": "user"}')
        (claude_dir / "conv3.jsonl").write_text('{"type": "user"}')

        vibe_dir = tmp_path / "vibe"
        vibe_dir.mkdir()
        (vibe_dir / "session1.json").write_text('{}')

        codex_dir = tmp_path / ".codex" / "sessions"
        codex_dir.mkdir(parents=True)
        (codex_dir / "rollout1.jsonl").write_text('{"type":"session_meta","payload":{"id":"codex-1","cwd":"D:\\\\projects\\\\searchat"}}')

        passthrough = lambda files, *a, **kw: files
        with patch('searchat.api.routers.indexing.exclude_automated_conversations', side_effect=passthrough):
            with patch('searchat.api.routers.indexing.get_config', return_value=mock_config):
                with patch('searchat.api.routers.indexing.get_unified_indexer', return_value=mock_unified_indexer):
                    with patch('searchat.api.routers.indexing.iter_providers', return_value=[
                        Mock(agent_id="claude", discover_dirs=Mock(return_value=[claude_dir])),
                        Mock(agent_id="vibe", discover_dirs=Mock(return_value=[vibe_dir])),
                        Mock(agent_id="codex", discover_dirs=Mock(return_value=[codex_dir])),
                    ]):
                        with patch('searchat.api.dependencies.projects_cache', None):
                            with patch('searchat.api.routers.indexing.indexing_lock', threading.Lock()):
                                with patch('searchat.api.routers.indexing.indexing_state', {"in_progress": False, "operation": None}):
                                    response = client.post("/api/index_missing")

                                    assert response.status_code == 200
                                    data = response.json()

                                    assert data["success"] is True
                                    assert data["new_conversations"] == 5
                                    assert data["updated_conversations"] == 1
                                    assert data["total_files"] == 5
                                    assert "time_seconds" in data

                                    mock_unified_indexer.detect_changed_files.assert_called_once()
                                    mock_unified_indexer.index_from_source_files.assert_called_once()

    def test_index_missing_all_indexed(self, client, mock_config, mock_unified_indexer, tmp_path):
        """Test when all conversations are already indexed."""
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()
        (claude_dir / "conv1.jsonl").write_text('{"type": "user"}')

        mock_unified_indexer.detect_changed_files.return_value = ([], [])
        mock_unified_indexer.index_from_source_files.return_value = {
            "new_conversations": 0,
            "updated_conversations": 0,
            "exchanges_created": 0,
            "embeddings_created": 0,
            "skipped_already_indexed": 1,
            "skipped_errors": 0,
            "total_files": 1,
            "time_seconds": 0.1,
        }

        passthrough = lambda files, *a, **kw: files
        with patch('searchat.api.routers.indexing.exclude_automated_conversations', side_effect=passthrough):
            with patch('searchat.api.routers.indexing.get_config', return_value=mock_config):
                with patch('searchat.api.routers.indexing.get_unified_indexer', return_value=mock_unified_indexer):
                    with patch('searchat.api.routers.indexing.iter_providers', return_value=[
                        Mock(agent_id="claude", discover_dirs=Mock(return_value=[claude_dir])),
                    ]):
                        with patch('searchat.api.routers.indexing.indexing_lock', threading.Lock()):
                            with patch('searchat.api.routers.indexing.indexing_state', {"in_progress": False, "operation": None}):
                                response = client.post("/api/index_missing")

                                assert response.status_code == 200
                                data = response.json()

                                assert data["success"] is True
                                assert data["new_conversations"] == 0

    def test_index_missing_sets_indexing_state(self, client, mock_config, mock_unified_indexer, tmp_path):
        """Test that indexing state is properly managed."""
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()
        (claude_dir / "conv1.jsonl").write_text('{"type": "user"}')

        indexing_state_local = {"in_progress": False, "operation": None}

        passthrough = lambda files, *a, **kw: files
        with patch('searchat.api.routers.indexing.exclude_automated_conversations', side_effect=passthrough):
            with patch('searchat.api.routers.indexing.get_config', return_value=mock_config):
                with patch('searchat.api.routers.indexing.get_unified_indexer', return_value=mock_unified_indexer):
                    with patch('searchat.api.routers.indexing.iter_providers', return_value=[
                        Mock(agent_id="claude", discover_dirs=Mock(return_value=[claude_dir])),
                    ]):
                        with patch('searchat.api.dependencies.projects_cache', None):
                            with patch('searchat.api.routers.indexing.indexing_lock', threading.Lock()):
                                with patch('searchat.api.routers.indexing.indexing_state', indexing_state_local):
                                    response = client.post("/api/index_missing")

                                    assert indexing_state_local["in_progress"] is False
                                    assert indexing_state_local["operation"] is None

    def test_index_missing_error_handling(self, client, mock_config, mock_unified_indexer, tmp_path):
        """Test error handling when indexing fails."""
        mock_unified_indexer.index_from_source_files.side_effect = Exception("Indexing error")

        passthrough = lambda files, *a, **kw: files
        with patch('searchat.api.routers.indexing.exclude_automated_conversations', side_effect=passthrough):
            with patch('searchat.api.routers.indexing.get_config', return_value=mock_config):
                with patch('searchat.api.routers.indexing.get_unified_indexer', return_value=mock_unified_indexer):
                    with patch('searchat.api.routers.indexing.iter_providers', return_value=[
                        Mock(agent_id="claude", discover_dirs=Mock(return_value=[tmp_path])),
                    ]):
                        with patch('searchat.api.routers.indexing.indexing_lock', threading.Lock()):
                            with patch('searchat.api.routers.indexing.indexing_state', {"in_progress": False, "operation": None}):
                                response = client.post("/api/index_missing")

                                assert response.status_code == 500
                                assert "Indexing error" in response.json()["detail"]

    def test_index_missing_no_unified_indexer(self, client):
        """Test 503 when unified indexer is not available."""
        with patch('searchat.api.routers.indexing.get_unified_indexer', return_value=None):
            response = client.post("/api/index_missing")

            assert response.status_code == 503
            assert "not available" in response.json()["detail"].lower()

    def test_index_missing_blocked_by_concurrent_indexing(self, client, mock_unified_indexer):
        """Test 409 when another indexing operation holds the lock."""
        held_lock = threading.Lock()
        held_lock.acquire()  # Pre-acquire to simulate concurrent indexing

        with patch('searchat.api.routers.indexing.get_unified_indexer', return_value=mock_unified_indexer):
            with patch('searchat.api.routers.indexing.indexing_lock', held_lock):
                with patch('searchat.api.routers.indexing.indexing_state', {"in_progress": True, "operation": "startup"}):
                    response = client.post("/api/index_missing")

                    assert response.status_code == 409
                    assert "already in progress" in response.json()["detail"].lower()


# ============================================================================
# ADMIN ENDPOINT TESTS
# ============================================================================

@pytest.mark.unit
class TestWatcherStatusEndpoint:
    """Tests for GET /api/watcher/status endpoint."""

    def test_get_watcher_status_running(self, client, mock_watcher):
        """Test getting watcher status when running."""
        watcher_stats = {"indexed_count": 5, "last_update": "2025-01-20T10:00:00"}

        with patch('searchat.api.routers.admin.get_watcher', return_value=mock_watcher):
            with patch('searchat.api.routers.admin.watcher_stats', watcher_stats):
                response = client.get("/api/watcher/status")

                assert response.status_code == 200
                data = response.json()

                assert data["running"] is True
                assert len(data["watched_directories"]) == 2
                assert data["indexed_since_start"] == 5
                assert data["last_update"] == "2025-01-20T10:00:00"

    def test_get_watcher_status_not_running(self, client):
        """Test getting watcher status when not running."""
        watcher_stats = {"indexed_count": 0, "last_update": None}

        with patch('searchat.api.routers.admin.get_watcher', return_value=None):
            with patch('searchat.api.routers.admin.watcher_stats', watcher_stats):
                response = client.get("/api/watcher/status")

                assert response.status_code == 200
                data = response.json()

                assert data["running"] is False
                assert data["watched_directories"] == []


@pytest.mark.unit
class TestShutdownEndpoint:
    """Tests for POST /api/shutdown endpoint.

    The background task calls os._exit(0), which kills the pytest process.
    Patch it so the test runner survives.
    """

    def test_shutdown_success(self, client):
        """Test unconditional shutdown returns success."""
        with patch('os._exit'):
            response = client.post("/api/shutdown")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "shutting down" in data["message"].lower()

    def test_shutdown_with_force_param(self, client):
        """Test shutdown with force param still succeeds (param is accepted but no-op)."""
        with patch('os._exit'):
            response = client.post("/api/shutdown?force=true")
            assert response.status_code == 200
            assert response.json()["success"] is True


@pytest.mark.unit
class TestWatcherCallbacks:
    """Tests for watcher callback and background startup behavior."""

    def test_on_new_conversations_retries_when_indexing_fails(self):
        """Watcher batches should not be consumed when indexing fails."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_config.indexing.reindex_on_modification = True

        mock_indexer = Mock()
        mock_indexer.detect_changed_files.return_value = (["new.jsonl"], [])
        mock_indexer.index_from_source_files.side_effect = RuntimeError("db write failed")

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_config", return_value=mock_config):
                with patch("searchat.api.app.exclude_automated_conversations", side_effect=lambda files, *_: files):
                    with patch("searchat.api.app.indexing_lock", threading.Lock()):
                        with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                            assert on_new_conversations(["new.jsonl"]) is False

    def test_on_new_conversations_schedules_background_distillation(self):
        """Watcher indexing should schedule background distillation instead of blocking."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_config.indexing.reindex_on_modification = True

        mock_indexer = Mock()
        mock_indexer.detect_changed_files.return_value = (["new.jsonl"], [])
        mock_indexer.index_from_source_files.return_value = {
            "new_conversations": 1,
            "updated_conversations": 0,
            "exchanges_created": 0,
        }

        mock_distiller = Mock()

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_config", return_value=mock_config):
                with patch("searchat.api.app.get_distiller", return_value=mock_distiller):
                    with patch("searchat.api.app._schedule_background_distillation") as schedule_distillation:
                        with patch("searchat.api.app.exclude_automated_conversations", side_effect=lambda files, *_: files):
                            with patch("searchat.api.app.reset_projects_cache"):
                                with patch("searchat.api.app.indexing_lock", threading.Lock()):
                                    with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                                        assert on_new_conversations(["new.jsonl"]) is True

        schedule_distillation.assert_called_once_with(mock_distiller, reason="watcher")

    def test_on_new_conversations_does_not_retry_when_distillation_scheduling_fails(self):
        """Scheduling failure should not requeue an already indexed batch."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_config.indexing.reindex_on_modification = True

        mock_indexer = Mock()
        mock_indexer.detect_changed_files.return_value = (["new.jsonl"], [])
        mock_indexer.index_from_source_files.return_value = {
            "new_conversations": 1,
            "updated_conversations": 0,
            "exchanges_created": 0,
        }

        mock_distiller = Mock()

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_config", return_value=mock_config):
                with patch("searchat.api.app.get_distiller", return_value=mock_distiller):
                    with patch("searchat.api.app._schedule_background_distillation", side_effect=RuntimeError("scheduler down")):
                        with patch("searchat.api.app.exclude_automated_conversations", side_effect=lambda files, *_: files):
                            with patch("searchat.api.app.reset_projects_cache"):
                                with patch("searchat.api.app.indexing_lock", threading.Lock()):
                                    with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                                        assert on_new_conversations(["new.jsonl"]) is True

    def test_on_new_conversations_logs_invalid_transcript_summary(self, caplog):
        """Watcher batches should summarize invalid transcripts at info level."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_config.indexing.reindex_on_modification = True

        mock_indexer = Mock()
        mock_indexer.detect_changed_files.return_value = (["bad.jsonl"], [])
        mock_indexer.index_from_source_files.return_value = {
            "new_conversations": 0,
            "updated_conversations": 0,
            "exchanges_created": 0,
            "invalid_transcript_count": 2,
            "invalid_transcript_examples": [
                {"file_path": "C:/tmp/one.jsonl", "reason": "bad line"},
                {"file_path": "C:/tmp/two.jsonl", "reason": "empty"},
            ],
        }

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_config", return_value=mock_config):
                with patch("searchat.api.app.get_distiller", return_value=None):
                    with patch("searchat.api.app.exclude_automated_conversations", side_effect=lambda files, *_: files):
                        with patch("searchat.api.app.reset_projects_cache"):
                            with patch("searchat.api.app.indexing_lock", threading.Lock()):
                                with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                                    with caplog.at_level("INFO"):
                                        assert on_new_conversations(["bad.jsonl"]) is True

        assert "Watcher indexing skipped 2 invalid transcript files." in caplog.text
        assert "one.jsonl, two.jsonl" in caplog.text

    def test_on_new_conversations_schedules_fts_rebuild(self):
        """Watcher indexing should not block on inline FTS rebuilds."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_config.indexing.reindex_on_modification = True

        mock_indexer = Mock()
        mock_indexer.detect_changed_files.return_value = (["new.jsonl"], [])
        mock_indexer.index_from_source_files.return_value = {
            "new_conversations": 1,
            "updated_conversations": 0,
            "exchanges_created": 2,
        }

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_config", return_value=mock_config):
                with patch("searchat.api.app.get_distiller", return_value=None):
                    with patch("searchat.api.app.exclude_automated_conversations", side_effect=lambda files, *_: files):
                        with patch("searchat.api.app.reset_projects_cache"):
                            with patch("searchat.api.app._schedule_fts_rebuild") as schedule_fts:
                                with patch("searchat.api.app.indexing_lock", threading.Lock()):
                                    with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                                        assert on_new_conversations(["new.jsonl"]) is True

        schedule_fts.assert_called_once_with(mock_indexer.storage, reason="watcher")

    def test_background_startup_starts_watcher_after_catchup_failure(self):
        """Live watcher should still start when startup catch-up fails."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_indexer = Mock()
        mock_indexer.detect_changed_files.side_effect = RuntimeError("scan failed")
        mock_watcher = Mock()
        mock_watcher.get_watched_directories.return_value = [Path("/watched")]

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_watcher", return_value=mock_watcher):
                with patch("searchat.api.app.get_config", return_value=mock_config):
                    with patch("searchat.api.app.indexing_lock", threading.Lock()):
                        with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                            asyncio.run(_background_scan_and_start_watcher())

        mock_watcher.start.assert_called_once()

    def test_background_startup_clears_watcher_when_start_fails(self):
        """Watcher singleton should be cleared if the observer cannot start."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_indexer = Mock()
        mock_indexer.detect_changed_files.return_value = ([], [])
        mock_watcher = Mock()
        mock_watcher.start.side_effect = RuntimeError("observer init failed")

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_watcher", return_value=mock_watcher):
                with patch("searchat.api.app.get_config", return_value=mock_config):
                    with patch("searchat.api.app.set_watcher") as set_watcher:
                        with patch("searchat.api.app.indexing_lock", threading.Lock()):
                            with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                                asyncio.run(_background_scan_and_start_watcher())

        set_watcher.assert_called_once_with(None)

    def test_background_startup_logs_invalid_transcript_summary(self, caplog):
        """Startup catch-up should log one info summary for invalid transcripts."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_indexer = Mock()
        mock_indexer.detect_changed_files.return_value = (["bad.jsonl"], [])
        mock_indexer.index_from_source_files.return_value = {
            "new_conversations": 1,
            "updated_conversations": 0,
            "exchanges_created": 3,
            "invalid_transcript_count": 2,
            "invalid_transcript_examples": [
                {"file_path": "C:/tmp/alpha.jsonl", "reason": "bad line"},
                {"file_path": "C:/tmp/beta.jsonl", "reason": "not objects"},
            ],
        }
        mock_watcher = Mock()
        mock_watcher.scan_all_files.return_value = ["bad.jsonl"]
        mock_watcher.get_watched_directories.return_value = [Path("/watched")]

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_watcher", return_value=mock_watcher):
                with patch("searchat.api.app.get_config", return_value=mock_config):
                    with patch("searchat.api.app.exclude_automated_conversations", side_effect=lambda files, *_: files):
                        with patch("searchat.api.app.indexing_lock", threading.Lock()):
                            with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                                with caplog.at_level("INFO"):
                                    asyncio.run(_background_scan_and_start_watcher())

        assert "Startup catch-up skipped 2 invalid transcript files." in caplog.text
        assert "alpha.jsonl, beta.jsonl" in caplog.text

    def test_background_startup_starts_watcher_before_scheduling_fts(self):
        """Live watcher should come up before the asynchronous FTS rebuild is queued."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_indexer = Mock()
        mock_indexer.detect_changed_files.return_value = (["new.jsonl"], [])
        mock_indexer.index_from_source_files.return_value = {
            "new_conversations": 1,
            "updated_conversations": 0,
            "exchanges_created": 3,
        }
        mock_watcher = Mock()
        mock_watcher.scan_all_files.return_value = ["new.jsonl"]
        mock_watcher.get_watched_directories.return_value = [Path("/watched")]
        call_order = []

        def record_start():
            call_order.append("watcher.start")

        def record_fts(_storage, reason):
            call_order.append(f"fts.{reason}")

        mock_watcher.start.side_effect = record_start

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_watcher", return_value=mock_watcher):
                with patch("searchat.api.app.get_config", return_value=mock_config):
                    with patch("searchat.api.app.exclude_automated_conversations", side_effect=lambda files, *_: files):
                        with patch("searchat.api.app._schedule_fts_rebuild", side_effect=record_fts):
                            with patch("searchat.api.app.indexing_lock", threading.Lock()):
                                with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                                    asyncio.run(_background_scan_and_start_watcher())

        assert call_order == ["watcher.start", "fts.startup"]

    def test_background_startup_uses_cached_files_and_schedules_reconciliation(self):
        """Persisted watcher cache should bootstrap startup and trigger background reconciliation."""
        mock_config = Mock()
        mock_config.paths.excluded_conversations_dir = ""
        mock_indexer = Mock()
        mock_indexer.detect_changed_files.return_value = ([], [])
        mock_watcher = Mock()
        mock_watcher.get_known_files.return_value = ["cached.jsonl"]
        mock_watcher.get_watched_directories.return_value = [Path("/watched")]
        created_coroutines = []

        def capture_task(coro):
            created_coroutines.append(coro)
            coro.close()
            return Mock()

        with patch("searchat.api.app.get_unified_indexer", return_value=mock_indexer):
            with patch("searchat.api.app.get_watcher", return_value=mock_watcher):
                with patch("searchat.api.app.get_config", return_value=mock_config):
                    with patch("searchat.api.app.indexing_lock", threading.Lock()):
                        with patch("searchat.api.app.indexing_state", {"in_progress": False, "operation": None}):
                            with patch("searchat.api.app.asyncio.create_task", side_effect=capture_task) as create_task:
                                asyncio.run(_background_scan_and_start_watcher())

        mock_watcher.scan_all_files.assert_not_called()
        mock_indexer.detect_changed_files.assert_called_once_with(["cached.jsonl"])
        create_task.assert_called_once()
        assert len(created_coroutines) == 1

    def test_schedule_background_distillation_coalesces_running_worker(self):
        """Repeated watcher/startup requests should reuse the same background worker."""
        distiller = Mock()
        running_thread = Mock()
        running_thread.is_alive.return_value = True

        with patch("searchat.api.app._distillation_thread", running_thread):
            with patch("searchat.api.app._distillation_requested", False):
                scheduled = _schedule_background_distillation(distiller, reason="watcher")

        assert scheduled is False


@pytest.mark.unit
class TestWarmupBehavior:
    """Tests for startup warmup policy."""

    def test_background_warmup_defaults_to_keyword_mode(self):
        mock_config = Mock()
        mock_config.performance.startup_warmup_mode = "keyword"
        mock_unified = Mock()

        with patch("searchat.api.app.get_config", return_value=mock_config):
            with patch("searchat.api.app.get_unified_search_engine", return_value=mock_unified):
                with patch("searchat.api.app.get_palace_query", return_value=Mock()) as get_palace:
                    asyncio.run(_background_warmup())

        assert mock_unified.search.call_args[1]["algorithm"].value == "keyword"
        get_palace.return_value.search_hybrid.assert_not_called()

    def test_background_warmup_skips_when_disabled(self):
        mock_config = Mock()
        mock_config.performance.startup_warmup_mode = "none"
        mock_unified = Mock()

        with patch("searchat.api.app.get_config", return_value=mock_config):
            with patch("searchat.api.app.get_unified_search_engine", return_value=mock_unified):
                asyncio.run(_background_warmup())

        mock_unified.search.assert_not_called()
