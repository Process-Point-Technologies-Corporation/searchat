"""Tests for unified search engine module."""
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import numpy as np
import pytest

from searchat.config import Config
from searchat.core.unified_search import UnifiedSearchEngine
from searchat.core.unified_storage import UnifiedStorage, EMBEDDING_DIM
from searchat.models.domain import SearchFilters
from searchat.models.enums import AlgorithmType


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock()
    config.embedding = MagicMock()
    config.embedding.model = "all-MiniLM-L6-v2"
    config.embedding.get_device.return_value = "cpu"
    config.performance = MagicMock()
    config.performance.query_cache_size = 10
    config.search = MagicMock()
    config.search.ranking = MagicMock()
    config.search.ranking.keyword_weight = 0.6
    config.search.ranking.semantic_weight = 0.4
    config.search.ranking.boost_multiplier = 1.2
    config.search.ranking.rank_decay = 0.1
    config.search.ranking.title_boost = 2.0
    return config


@pytest.fixture
def populated_storage():
    """Create an in-memory storage with test data."""
    conn = duckdb.connect(":memory:")
    storage = UnifiedStorage(Path("/tmp"), conn=conn)

    # Add test conversations
    for i in range(3):
        conv_id = f"test-conv-{i:03d}"
        storage.conn.execute("""
            INSERT INTO conversations (
                conversation_id, project_id, file_path, title,
                created_at, updated_at, message_count, full_text,
                file_hash, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            conv_id,
            f"project-{i % 2}",
            f"/path/to/{conv_id}.jsonl",
            f"Test Conversation {i} about Python and JavaScript",
            datetime(2025, 1, i + 1),
            datetime(2025, 1, i + 1),
            5,
            f"This is conversation {i} about Python programming and JavaScript development.",
            f"hash{i}",
            datetime.utcnow(),
        ])

        # Add exchanges
        exchange_id = str(uuid.uuid4())
        storage.conn.execute("""
            INSERT INTO exchanges (
                exchange_id, conversation_id, project_id, ply_start, ply_end,
                exchange_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            exchange_id,
            conv_id,
            f"project-{i % 2}",
            0,
            2,
            f"User: How do I use Python?\nAssistant: Python is great for {['data science', 'web development', 'automation'][i]}.",
            datetime.utcnow(),
        ])

        # Add embeddings
        embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        storage.conn.execute("""
            INSERT INTO verbatim_embeddings (exchange_id, embedding)
            VALUES (?, ?::FLOAT[])
        """, [exchange_id, embedding.tolist()])

    yield storage
    storage.close()


def _make_engine(mock_config, populated_storage, with_encode=False):
    """Create a UnifiedSearchEngine with a mock embedder."""
    mock_embedder = MagicMock()
    if with_encode:
        mock_embedder.encode.return_value = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    return UnifiedSearchEngine(
        search_dir=Path("/tmp"),
        config=mock_config,
        storage=populated_storage,
        embedder=mock_embedder,
    )


class TestUnifiedSearchEngine:
    """Tests for unified search engine."""

    def test_search_returns_results(self, mock_config, populated_storage):
        """Test that search returns results."""
        engine = _make_engine(mock_config, populated_storage, with_encode=True)
        results = engine.search("Python", algorithm=AlgorithmType.SEMANTIC)

        assert results is not None
        assert len(results.results) > 0

    def test_keyword_search(self, mock_config, populated_storage):
        """Test keyword search mode."""
        engine = _make_engine(mock_config, populated_storage)

        # Keyword search doesn't need embedder
        results = engine.search("Python", algorithm=AlgorithmType.KEYWORD)

        assert results is not None
        assert results.mode_used == "keyword"

    def test_hybrid_search(self, mock_config, populated_storage):
        """Test hybrid search mode."""
        engine = _make_engine(mock_config, populated_storage, with_encode=True)
        results = engine.search("Python programming", algorithm=AlgorithmType.HYBRID)

        assert results is not None
        assert results.mode_used == "hybrid"

    def test_search_with_project_filter(self, mock_config, populated_storage):
        """Test search with project filter."""
        engine = _make_engine(mock_config, populated_storage, with_encode=True)

        filters = SearchFilters(project_ids=["project-0"])
        results = engine.search("Python", algorithm=AlgorithmType.SEMANTIC, filters=filters)

        # All results should be from project-0
        for r in results.results:
            assert r.project_id == "project-0"

    def test_search_caching(self, mock_config, populated_storage):
        """Test that results are cached."""
        engine = _make_engine(mock_config, populated_storage, with_encode=True)

        # First search
        results1 = engine.search("Python", algorithm=AlgorithmType.SEMANTIC)

        # Second search (should use cache)
        results2 = engine.search("Python", algorithm=AlgorithmType.SEMANTIC)

        # Cache should have been used (second search faster, but we can't measure easily)
        # Just verify we get consistent results
        assert len(results1.results) == len(results2.results)

    def test_search_cache_key_includes_limit(self, mock_config, populated_storage):
        """Different limits should not reuse truncated cached results."""
        engine = _make_engine(mock_config, populated_storage)

        first = MagicMock(return_value=[{"conversation_id": "conv-1", "project_id": "p", "title": "A", "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(), "message_count": 1, "file_path": "/a", "exchange_text": "a", "score": 0.9, "exchange_id": "e1", "ply_start": 0, "ply_end": 1}])
        second = MagicMock(return_value=[
            {"conversation_id": "conv-1", "project_id": "p", "title": "A", "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(), "message_count": 1, "file_path": "/a", "exchange_text": "a", "score": 0.9, "exchange_id": "e1", "ply_start": 0, "ply_end": 1},
            {"conversation_id": "conv-2", "project_id": "p", "title": "B", "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(), "message_count": 1, "file_path": "/b", "exchange_text": "b", "score": 0.8, "exchange_id": "e2", "ply_start": 0, "ply_end": 1},
        ])

        engine.storage.search_verbatim_bm25 = first
        results1 = engine.search("Python", algorithm=AlgorithmType.KEYWORD, limit=1)
        engine.storage.search_verbatim_bm25 = second
        results2 = engine.search("Python", algorithm=AlgorithmType.KEYWORD, limit=2)

        assert len(results1.results) == 1
        assert len(results2.results) == 2

    def test_cached_results_are_same_instance(self, mock_config, populated_storage):
        """Cache returns the same instance for identical queries (no deepcopy overhead)."""
        engine = _make_engine(mock_config, populated_storage, with_encode=True)

        first = engine.search("Python", algorithm=AlgorithmType.SEMANTIC)
        second = engine.search("Python", algorithm=AlgorithmType.SEMANTIC)
        assert first is second

    def test_search_result_fields(self, mock_config, populated_storage):
        """Test that search results have expected fields."""
        engine = _make_engine(mock_config, populated_storage, with_encode=True)
        results = engine.search("Python", algorithm=AlgorithmType.SEMANTIC)

        if results.results:
            r = results.results[0]
            assert hasattr(r, "conversation_id")
            assert hasattr(r, "project_id")
            assert hasattr(r, "title")
            assert hasattr(r, "score")
            assert hasattr(r, "snippet")
            assert hasattr(r, "message_start_index")
            assert hasattr(r, "message_end_index")


class TestSnippetCreation:
    """Tests for snippet creation."""

    def test_create_snippet_centers_on_match(self, mock_config, populated_storage):
        """Test that snippets are centered on query match."""
        engine = _make_engine(mock_config, populated_storage)

        text = "A" * 100 + "SEARCHTERM" + "B" * 100
        parsed = engine.query_parser.parse("SEARCHTERM")
        snippet = engine._create_snippet(text, parsed, length=50)

        # Snippet should contain the search term
        assert "SEARCHTERM" in snippet

    def test_create_snippet_handles_short_text(self, mock_config, populated_storage):
        """Test snippet creation with text shorter than length."""
        engine = _make_engine(mock_config, populated_storage)

        text = "Short text"
        parsed = engine.query_parser.parse("query")
        snippet = engine._create_snippet(text, parsed, length=200)

        # Should return full text without ellipsis
        assert snippet == "Short text"

    def test_create_snippet_handles_empty_text(self, mock_config, populated_storage):
        """Test snippet creation with empty text."""
        engine = _make_engine(mock_config, populated_storage)

        parsed = engine.query_parser.parse("query")
        snippet = engine._create_snippet("", parsed)
        assert snippet == ""


class TestResultMerging:
    """Tests for result merging logic."""

    def test_merge_deduplicates_by_conversation(self, mock_config, populated_storage):
        """Test that merging deduplicates results by conversation ID."""
        engine = _make_engine(mock_config, populated_storage, with_encode=True)
        results = engine.search("Python", algorithm=AlgorithmType.HYBRID)

        # Check for unique conversation IDs
        conv_ids = [r.conversation_id for r in results.results]
        assert len(conv_ids) == len(set(conv_ids)), "Results should be deduplicated"


class TestTemporalDecaySearch:
    """Tests for temporal decay facet scoping behavior."""

    def test_temporal_decay_search_does_not_mutate_caller_filters(
        self, mock_config, populated_storage
    ):
        """Facet resolution should not rewrite the caller's SearchFilters."""
        engine = _make_engine(mock_config, populated_storage, with_encode=True)
        filters = SearchFilters(project_ids=["original-project"])

        engine._resolve_facets_temporal = MagicMock(return_value=["resolved-project"])
        engine.storage.search_verbatim_bm25 = MagicMock(return_value=[])
        engine.storage.search_verbatim_semantic = MagicMock(return_value=[])

        engine._temporal_decay_search("Python", filters, limit=5)

        assert filters.project_ids == ["original-project"]


class TestStatistics:
    """Tests for engine statistics."""

    def test_get_stats(self, mock_config, populated_storage):
        """Test getting storage statistics."""
        engine = _make_engine(mock_config, populated_storage)

        stats = engine.get_stats()
        assert "conversations" in stats
        assert "exchanges" in stats
        assert stats["conversations"] == 3
