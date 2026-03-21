"""Unit tests for search API routes."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from contextlib import contextmanager

from fastapi.testclient import TestClient

from searchat.models import SearchResult, SearchResults, AlgorithmType
from searchat.api.app import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_unified_engine():
    """Mock UnifiedSearchEngine for testing."""
    mock = Mock()

    now = datetime.now()
    sample_result = SearchResult(
        conversation_id="test-conv-123",
        project_id="test-project",
        title="Test Conversation",
        created_at=now - timedelta(days=5),
        updated_at=now - timedelta(days=1),
        message_count=10,
        file_path="/home/user/.claude/test-conv-123.jsonl",
        snippet="This is a test conversation about Python",
        score=0.95,
        message_start_index=0,
        message_end_index=5,
    )

    mock.search.return_value = SearchResults(
        results=[sample_result],
        total_count=1,
        search_time_ms=15.5,
        mode_used="cross_layer",
    )

    # For projects endpoint
    mock_cursor = Mock()
    mock_cursor.execute.return_value.fetchall.return_value = [
        ("project-a",), ("project-b",), ("project-c",),
    ]
    mock.storage._get_cursor.return_value = mock_cursor
    mock.storage._get_read_cursor.return_value = mock_cursor

    return mock


@contextmanager
def mock_unified_dependencies(mock_engine):
    """Context manager that patches unified search engine."""
    with patch('searchat.api.routers.search.get_unified_search_engine', return_value=mock_engine):
        yield


@pytest.mark.unit
class TestSearchEndpoint:
    """Tests for /api/search endpoint."""

    def test_basic_search(self, client, mock_unified_engine):
        """Test basic search with query."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test")

            assert response.status_code == 200
            data = response.json()

            assert "results" in data
            assert "total" in data
            assert "search_time_ms" in data
            assert data["total"] == 1
            assert len(data["results"]) == 1

            result = data["results"][0]
            assert result["conversation_id"] == "test-conv-123"
            assert result["project_id"] == "test-project"
            assert result["title"] == "Test Conversation"
            assert result["source"] == "WSL"
            assert result["combined_score"] > 0

    def test_default_mode_is_cross_layer(self, client, mock_unified_engine):
        """Test that default search mode is cross-layer."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test")

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            assert call_args[1]['algorithm'] == AlgorithmType.CROSS_LAYER

    def test_mode_verbatim(self, client, mock_unified_engine):
        """Test verbatim mode maps to KEYWORD algorithm."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&mode=verbatim")

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            assert call_args[1]['algorithm'] == AlgorithmType.KEYWORD

    def test_mode_distill(self, client, mock_unified_engine):
        """Test distill mode maps to DISTILL algorithm."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&mode=distill")

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            assert call_args[1]['algorithm'] == AlgorithmType.DISTILL

    def test_mode_cross_layer(self, client, mock_unified_engine):
        """Test cross-layer mode maps to CROSS_LAYER algorithm."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&mode=cross-layer")

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            assert call_args[1]['algorithm'] == AlgorithmType.CROSS_LAYER

    def test_invalid_mode_returns_400(self, client, mock_unified_engine):
        """Test that invalid mode returns 400."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&mode=invalid")

            assert response.status_code == 400
            assert "Invalid mode" in response.json()["detail"]

    def test_search_with_project_filter(self, client, mock_unified_engine):
        """Test search with project filter."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&project=test-project")

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            filters = call_args[1]['filters']
            assert filters.project_ids == ["test-project"]

    def test_search_with_date_filter_today(self, client, mock_unified_engine):
        """Test search with 'today' date filter."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&date=today")

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            filters = call_args[1]['filters']
            assert filters.date_from is not None
            assert filters.date_to is not None
            assert filters.date_from.hour == 0

    def test_search_with_date_filter_week(self, client, mock_unified_engine):
        """Test search with 'week' date filter."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&date=week")

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            filters = call_args[1]['filters']
            assert filters.date_from is not None
            days_diff = (filters.date_to - filters.date_from).days
            assert 6 <= days_diff <= 8

    def test_search_with_date_filter_month(self, client, mock_unified_engine):
        """Test search with 'month' date filter."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&date=month")

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            filters = call_args[1]['filters']
            assert filters.date_from is not None
            days_diff = (filters.date_to - filters.date_from).days
            assert 29 <= days_diff <= 31

    def test_search_with_custom_date_range(self, client, mock_unified_engine):
        """Test search with custom date range."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get(
                "/api/search?q=test&date=custom&date_from=2025-01-01&date_to=2025-01-31"
            )

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            filters = call_args[1]['filters']
            assert filters.date_from == datetime(2025, 1, 1)
            assert filters.date_to == datetime(2025, 2, 1)

    def test_search_sort_by_date_newest(self, client, mock_unified_engine):
        """Test search sorted by newest date."""
        now = datetime.now()
        results = [
            SearchResult(
                conversation_id=f"conv-{i}",
                project_id="test",
                title=f"Conv {i}",
                created_at=now - timedelta(days=i + 5),
                updated_at=now - timedelta(days=i),
                message_count=10,
                file_path=f"/test/conv-{i}.jsonl",
                snippet="Test",
                score=0.9,
            )
            for i in range(3)
        ]

        mock_unified_engine.search.return_value = SearchResults(
            results=results, total_count=3, search_time_ms=20.0, mode_used="cross_layer",
        )

        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&sort_by=date_newest")

            assert response.status_code == 200
            data = response.json()
            assert data["results"][0]["conversation_id"] == "conv-0"
            assert data["results"][2]["conversation_id"] == "conv-2"

    def test_search_sort_by_messages(self, client, mock_unified_engine):
        """Test search sorted by message count."""
        now = datetime.now()
        results = [
            SearchResult(
                conversation_id=f"conv-{i}",
                project_id="test",
                title=f"Conv {i}",
                created_at=now,
                updated_at=now,
                message_count=(i + 1) * 5,
                file_path=f"/test/conv-{i}.jsonl",
                snippet="Test",
                score=0.9,
            )
            for i in range(3)
        ]

        mock_unified_engine.search.return_value = SearchResults(
            results=results, total_count=3, search_time_ms=20.0, mode_used="cross_layer",
        )

        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&sort_by=messages")

            assert response.status_code == 200
            data = response.json()
            assert data["results"][0]["message_count"] == 15
            assert data["results"][2]["message_count"] == 5

    def test_search_with_limit(self, client, mock_unified_engine):
        """Test search with result limit."""
        now = datetime.now()
        results = [
            SearchResult(
                conversation_id=f"conv-{i}",
                project_id="test",
                title=f"Conv {i}",
                created_at=now,
                updated_at=now,
                message_count=10,
                file_path=f"/test/conv-{i}.jsonl",
                snippet="Test",
                score=0.9,
            )
            for i in range(10)
        ]

        mock_unified_engine.search.return_value = SearchResults(
            results=results, total_count=10, search_time_ms=20.0, mode_used="cross_layer",
        )

        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&limit=3")

            assert response.status_code == 200
            data = response.json()
            # Limit is passed to unified_engine.search, so results come back limited
            call_args = mock_unified_engine.search.call_args
            assert call_args[1]['limit'] == 3

    def test_search_source_detection_wsl(self, client, mock_unified_engine):
        """Test that WSL paths are detected correctly."""
        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test")

            assert response.status_code == 200
            data = response.json()
            assert data["results"][0]["source"] == "WSL"

    def test_search_source_detection_windows(self, client, mock_unified_engine):
        """Test that Windows paths are detected correctly."""
        now = datetime.now()
        mock_unified_engine.search.return_value = SearchResults(
            results=[
                SearchResult(
                    conversation_id="conv-1",
                    project_id="test",
                    title="Conv 1",
                    created_at=now,
                    updated_at=now,
                    message_count=10,
                    file_path="C:\\Users\\Test\\.claude\\conv-1.jsonl",
                    snippet="Test",
                    score=0.9,
                )
            ],
            total_count=1, search_time_ms=10.0, mode_used="keyword",
        )

        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test&mode=verbatim")

            assert response.status_code == 200
            data = response.json()
            assert data["results"][0]["source"] == "WIN"

    def test_search_palace_fields_in_response(self, client, mock_unified_engine):
        """Test that palace fields are included in cross-layer response."""
        now = datetime.now()
        result = SearchResult(
            conversation_id="conv-palace",
            project_id="test",
            title="Palace Test",
            created_at=now,
            updated_at=now,
            message_count=5,
            file_path="/test/conv.jsonl",
            snippet="Test snippet",
            score=0.85,
            bm25_score=0.7,
            semantic_score=0.9,
            palace_summary="Implemented connection pooling",
            palace_context="Database optimization context",
            files_touched_raw=[{"path": "db.py", "action": "modified"}],
            object_id="obj-123",
        )

        mock_unified_engine.search.return_value = SearchResults(
            results=[result], total_count=1, search_time_ms=15.0, mode_used="cross_layer",
        )

        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test")

            assert response.status_code == 200
            data = response.json()
            r = data["results"][0]
            assert r["palace_summary"] == "Implemented connection pooling"
            assert r["palace_context"] == "Database optimization context"
            assert r["has_palace"] is True
            assert r["has_verbatim"] is True
            assert r["is_intersection"] is True
            assert len(r["files_touched"]) == 1
            assert r["files_touched"][0]["path"] == "db.py"
            assert r["object_id"] == "obj-123"

    def test_search_no_unified_engine_returns_503(self, client):
        """Test that missing unified engine returns 503."""
        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=None):
            response = client.get("/api/search?q=test")

            assert response.status_code == 503

    def test_search_error_handling(self, client, mock_unified_engine):
        """Test that search errors are handled properly."""
        mock_unified_engine.search.side_effect = Exception("Search failed")

        with mock_unified_dependencies(mock_unified_engine):
            response = client.get("/api/search?q=test")

            assert response.status_code == 500
            assert "Search failed" in response.json()["detail"]


@pytest.mark.unit
class TestProjectsEndpoint:
    """Tests for /api/projects endpoint."""

    def test_get_projects_from_unified(self, client, mock_unified_engine):
        """Test getting projects from unified engine."""
        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=mock_unified_engine):
            with patch('searchat.api.dependencies.projects_cache', None):
                response = client.get("/api/projects")

                assert response.status_code == 200
                projects = response.json()
                assert isinstance(projects, list)
                assert len(projects) == 3

    def test_get_projects_uses_cache(self, client):
        """Test that projects endpoint uses cache."""
        with patch('searchat.api.dependencies.projects_cache', ["cached-project"]):
            response = client.get("/api/projects")

            assert response.status_code == 200
            projects = response.json()
            assert projects == ["cached-project"]

    def test_search_unified_uses_algorithm_keyword(self, client, mock_unified_engine):
        """Test unified search endpoint calls engine with algorithm kwarg."""
        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=mock_unified_engine):
            response = client.get("/api/search/unified?q=test&mode=semantic")

            assert response.status_code == 200
            call_args = mock_unified_engine.search.call_args
            assert call_args[1]["algorithm"] == AlgorithmType.SEMANTIC
