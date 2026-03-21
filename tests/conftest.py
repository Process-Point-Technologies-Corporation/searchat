"""Pytest configuration and fixtures for searchat tests.

CRITICAL: This file mocks heavy dependencies (sentence-transformers, faiss)
at import time to avoid TensorFlow/CUDA requirements during testing.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


# ============================================================================
# IMPORT-TIME MOCKING (must happen before any searchat imports)
# ============================================================================

# Mock sentence-transformers to avoid tf_keras dependency
mock_st_module = MagicMock()
mock_st_class = MagicMock()
mock_st_instance = MagicMock()

# Configure mock embedding model behavior
def mock_encode(texts, **kwargs):
    """Return fake 384-dimensional embeddings."""
    if isinstance(texts, str):
        texts = [texts]
    return np.array([[0.1] * 384 for _ in texts])

mock_st_instance.encode = mock_encode
mock_st_instance.get_sentence_embedding_dimension.return_value = 384
mock_st_class.return_value = mock_st_instance
mock_st_module.SentenceTransformer = mock_st_class

# Inject mocked module into sys.modules BEFORE any imports
sys.modules['sentence_transformers'] = mock_st_module


# Mock FAISS to avoid GPU dependencies
class MockFaissIndex:
    """Mock FAISS index for testing."""

    def __init__(self, dimension: int = 384):
        self.d = dimension
        self.ntotal = 0
        self._vectors = []

    def add(self, vectors):
        """Mock adding vectors to index."""
        if len(vectors) > 0:
            self._vectors.extend(vectors)
            self.ntotal = len(self._vectors)

    def search(self, queries, k):
        """Mock search returning fake distances and indices."""
        n_queries = len(queries) if hasattr(queries, '__len__') else 1
        if n_queries == 0:
            return np.array([]), np.array([])
        # Return fake results
        distances = np.random.rand(n_queries, k).astype('float32')
        indices = np.random.randint(0, max(1, self.ntotal), (n_queries, k))
        return distances, indices

    def reset(self):
        """Reset the index."""
        self.ntotal = 0
        self._vectors = []


def mock_index_flat_l2(dimension):
    """Factory for creating mock FAISS index."""
    return MockFaissIndex(dimension)


mock_faiss_module = MagicMock()
mock_faiss_module.IndexFlatL2 = mock_index_flat_l2
mock_faiss_module.read_index = MagicMock(return_value=MockFaissIndex())
mock_faiss_module.write_index = MagicMock()

sys.modules['faiss'] = mock_faiss_module


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

# NOW safe to import searchat modules (after mocking)
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from searchat.services import BackupManager
from searchat.config import Config


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure for tests."""
    data_dir = tmp_path / "data"
    conversations_dir = data_dir / "conversations"
    indices_dir = data_dir / "indices"

    conversations_dir.mkdir(parents=True)
    indices_dir.mkdir(parents=True)

    return data_dir


@pytest.fixture
def temp_search_dir(tmp_path):
    """Create temporary search directory with full structure."""
    search_dir = tmp_path / ".searchat"
    data_dir = search_dir / "data"
    config_dir = search_dir / "config"
    backups_dir = search_dir / "backups"

    data_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    backups_dir.mkdir(parents=True)

    return search_dir


@pytest.fixture
def sample_claude_conversation(tmp_path) -> Path:
    """Create sample Claude Code conversation JSONL file."""
    conv_file = tmp_path / "test_conversation.jsonl"
    messages = [
        {
            "type": "user",
            "message": {"text": "How do I implement a binary search tree in Python?"},
            "timestamp": "2025-01-15T10:00:00",
            "uuid": "msg-1"
        },
        {
            "type": "assistant",
            "message": {
                "text": "Here's a binary search tree implementation:\n```python\nclass Node:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n```"
            },
            "timestamp": "2025-01-15T10:00:30",
            "uuid": "msg-2"
        },
        {
            "type": "user",
            "message": {"text": "Can you add insert and search methods?"},
            "timestamp": "2025-01-15T10:01:00",
            "uuid": "msg-3"
        },
    ]

    with open(conv_file, 'w', encoding='utf-8') as f:
        for msg in messages:
            f.write(json.dumps(msg) + '\n')

    return conv_file


@pytest.fixture
def sample_vibe_conversation(tmp_path) -> Path:
    """Create sample Mistral Vibe conversation JSON file."""
    conv_file = tmp_path / "session_20250115.json"
    data = {
        "id": "session-123",
        "created_at": "2025-01-15T10:00:00",
        "updated_at": "2025-01-15T10:05:00",
        "messages": [
            {
                "role": "user",
                "content": "Explain async/await in JavaScript",
                "timestamp": "2025-01-15T10:00:00"
            },
            {
                "role": "assistant",
                "content": "Async/await is syntactic sugar for promises in JavaScript. It allows you to write asynchronous code that looks synchronous.",
                "timestamp": "2025-01-15T10:00:15"
            },
            {
                "role": "user",
                "content": "Show me an example with error handling",
                "timestamp": "2025-01-15T10:02:00"
            },
        ]
    }

    with open(conv_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return conv_file


@pytest.fixture
def backup_manager(temp_search_dir):
    """BackupManager instance with temporary directory."""
    return BackupManager(temp_search_dir)


@pytest.fixture
def mock_embeddings():
    """Mock sentence transformer embeddings for testing."""
    import numpy as np

    class MockEmbedder:
        def encode(self, texts, batch_size=32, show_progress_bar=False):
            """Return random embeddings for testing."""
            if isinstance(texts, str):
                texts = [texts]
            # Return consistent dimensions (384 for MiniLM)
            return np.random.rand(len(texts), 384).astype('float32')

    return MockEmbedder()


@pytest.fixture
def sample_config(tmp_path) -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "batch_size": 32,
        },
        "search": {
            "default_mode": "hybrid",
            "max_results": 100,
            "min_score": 0.0,
        },
        "chunking": {
            "max_chunk_size": 1000,
            "overlap": 100,
        }
    }


# ============================================================================
# PALACE FIXTURES
# ============================================================================

@pytest.fixture
def palace_data_dir(tmp_path):
    """Temporary data directory for palace tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "indices").mkdir()
    return data_dir


@pytest.fixture
def palace_search_dir(tmp_path):
    """Temporary search directory with palace-ready structure."""
    search_dir = tmp_path / ".searchat"
    data_dir = search_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "indices").mkdir()
    (data_dir / "conversations").mkdir()
    return search_dir
