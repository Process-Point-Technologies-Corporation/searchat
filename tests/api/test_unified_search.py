"""Tests for unified search endpoints and merge_results function."""
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from searchat.models import SearchResult, SearchResults, AlgorithmType
from searchat.models.domain import (
    PalaceSearchResult,
    Room,
    FileTouched,
)
from searchat.config.settings import RankingConfig
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
        conversation_id="conv-123",
        project_id="test-project",
        title="Test Conversation",
        created_at=now - timedelta(days=5),
        updated_at=now - timedelta(days=1),
        message_count=10,
        file_path="/home/user/.claude/conv-123.jsonl",
        snippet="This is a test snippet",
        score=0.85,
        message_start_index=0,
        message_end_index=5,
        bm25_score=0.7,
        semantic_score=0.9,
        palace_summary="Implemented search",
        palace_context="FAISS index",
    )

    mock.search.return_value = SearchResults(
        results=[sample_result],
        total_count=1,
        search_time_ms=15.5,
        mode_used="cross_layer"
    )

    # Mock storage for get_conversation
    mock.storage.get_conversation.return_value = {
        "conversation_id": "conv-123",
        "project_id": "test-project",
        "title": "Test Conversation",
        "created_at": now - timedelta(days=5),
        "updated_at": now - timedelta(days=1),
        "message_count": 10,
        "file_path": "/home/user/.claude/conv-123.jsonl",
    }

    return mock


@pytest.mark.unit
class TestMainSearchEndpoint:
    """Tests for /api/search endpoint (cross-layer, verbatim, distill)."""

    def test_search_returns_results(self, client, mock_unified_engine):
        """Search returns results from unified engine."""
        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=mock_unified_engine):
            response = client.get("/api/search?q=FAISS")

            assert response.status_code == 200
            data = response.json()

            assert "results" in data
            assert "total" in data
            assert "search_time_ms" in data
            assert data["total"] == 1

    def test_search_modes(self, client, mock_unified_engine):
        """Search supports cross-layer, verbatim, distill modes."""
        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=mock_unified_engine):
            for mode in ["cross-layer", "verbatim", "distill"]:
                response = client.get(f"/api/search?q=test&mode={mode}")
                assert response.status_code == 200

    def test_search_invalid_mode(self, client, mock_unified_engine):
        """Search rejects invalid mode."""
        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=mock_unified_engine):
            response = client.get("/api/search?q=test&mode=invalid")
            assert response.status_code == 400

    def test_search_no_engine(self, client):
        """Search returns 503 when unified engine not available."""
        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=None):
            response = client.get("/api/search?q=test")
            assert response.status_code == 503

    def test_search_source_detection(self, client, mock_unified_engine):
        """Source (WSL/WIN) is detected correctly."""
        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=mock_unified_engine):
            response = client.get("/api/search?q=test")

            assert response.status_code == 200
            data = response.json()
            # conv-123 has /home/ path -> WSL
            assert data["results"][0]["source"] == "WSL"

    def test_search_sorting(self, client):
        """Search respects sort_by parameter."""
        mock = Mock()
        now = datetime.now()
        results = [
            SearchResult(
                conversation_id=f"conv-{i}",
                project_id="test",
                title=f"Conv {i}",
                created_at=now - timedelta(days=i+5),
                updated_at=now - timedelta(days=i),
                message_count=10 * (i + 1),
                file_path=f"/test/conv-{i}.jsonl",
                snippet="Test",
                score=0.9 - i * 0.1,
            )
            for i in range(3)
        ]

        mock.search.return_value = SearchResults(
            results=results,
            total_count=3,
            search_time_ms=20.0,
            mode_used="cross_layer"
        )

        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=mock):
            response = client.get("/api/search?q=test&sort_by=date_newest")
            data = response.json()
            assert data["results"][0]["conversation_id"] == "conv-0"

            response = client.get("/api/search?q=test&sort_by=date_oldest")
            data = response.json()
            assert data["results"][0]["conversation_id"] == "conv-2"

            response = client.get("/api/search?q=test&sort_by=messages")
            data = response.json()
            assert data["results"][0]["message_count"] == 30

    def test_search_error_handling(self, client, mock_unified_engine):
        """Search errors are handled properly."""
        mock_unified_engine.search.side_effect = Exception("Search failed")

        with patch('searchat.api.routers.search.get_unified_search_engine', return_value=mock_unified_engine):
            response = client.get("/api/search?q=test")
            assert response.status_code == 500
            assert "Search failed" in response.json()["detail"]


@pytest.mark.unit
class TestMergeResults:
    """Tests for the merge_results function."""

    def test_merge_normalizes_scores(self):
        """Scores are normalized to 0-1 range before combining."""
        from searchat.core.result_merger import merge_results

        now = datetime.now()
        ranking = RankingConfig(
            intersection_boost=0.2,
            palace_weight=0.5,
            verbatim_weight=0.5,
            keyword_weight=0.8,
            semantic_weight=0.2,
            rank_decay=0.1,
            title_boost=2.0,
            bm25_k1=2.5,
            bm25_b=0.25,
            bm25_candidates=500,
            faiss_k=100,
        )

        palace = [
            PalaceSearchResult(
                object_id="obj-1",
                conversation_id="conv-1",
                project_id="proj",
                ply_start=0,
                ply_end=5,
                exchange_core="Test",
                specific_context="",
                files_touched=[],
                rooms=[],
                score=10.0,
            )
        ]

        verbatim = [
            SearchResult(
                conversation_id="conv-1",
                project_id="proj",
                title="Test",
                created_at=now,
                updated_at=now,
                message_count=5,
                file_path="/test.jsonl",
                snippet="Test",
                score=0.5,
            )
        ]

        # Mock UnifiedStorage
        mock_storage = Mock()
        conv_data = {
            "conversation_id": "conv-1",
            "title": "Test",
            "created_at": now,
            "updated_at": now,
            "message_count": 5,
            "file_path": "/test.jsonl",
        }
        mock_storage.get_conversation.return_value = conv_data
        mock_storage.get_conversations_batch.return_value = {"conv-1": conv_data}

        results = merge_results(palace, verbatim, mock_storage, ranking)

        # Combined score should be max 1.0 with intersection boost
        assert 0 < results[0].combined_score <= 1.0
