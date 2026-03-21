"""Tests for PalaceBM25Index."""
from datetime import datetime

import duckdb
import pytest

from searchat.models.domain import DistilledObject, FileTouched, Room, RoomObject
from searchat.palace.bm25_index import PalaceBM25Index
from searchat.palace.storage import PalaceStorage


@pytest.fixture
def in_memory_storage():
    """Create in-memory PalaceStorage for testing."""
    conn = duckdb.connect(":memory:")
    storage = PalaceStorage(data_dir=None, conn=conn)
    return storage


@pytest.fixture
def seeded_storage(in_memory_storage):
    """Storage with test data."""
    now = datetime(2026, 2, 1, 12, 0, 0)

    objects = [
        DistilledObject(
            object_id="obj-1",
            project_id="proj-1",
            conversation_id="conv-1",
            ply_start=0,
            ply_end=5,
            files_touched=[
                FileTouched(path="src/main.py", action="modified"),
                FileTouched(path="tests/test_main.py", action="created"),
            ],
            exchange_core="Implemented FAISS vector search integration",
            specific_context="Added IndexFlatL2 for semantic similarity",
            created_at=now,
            exchange_at=now,
            embedding_id=0,
            distilled_text="Implemented FAISS vector search integration\nAdded IndexFlatL2",
        ),
        DistilledObject(
            object_id="obj-2",
            project_id="proj-1",
            conversation_id="conv-1",
            ply_start=6,
            ply_end=10,
            files_touched=[
                FileTouched(path="src/search.py", action="modified"),
            ],
            exchange_core="Added BM25 keyword search for hybrid retrieval",
            specific_context="Using rank_bm25 library with Okapi variant",
            created_at=now,
            exchange_at=now,
            embedding_id=1,
            distilled_text="Added BM25 keyword search for hybrid retrieval",
        ),
        DistilledObject(
            object_id="obj-3",
            project_id="proj-2",
            conversation_id="conv-2",
            ply_start=0,
            ply_end=3,
            files_touched=[
                FileTouched(path="config/settings.toml", action="modified"),
            ],
            exchange_core="Configured database connection pooling",
            specific_context="Set max_connections to 50 with timeout",
            created_at=now,
            exchange_at=now,
            embedding_id=2,
            distilled_text="Configured database connection pooling",
        ),
    ]

    rooms = [
        Room(
            room_id="room-1",
            room_type="file",
            room_key="src/main.py",
            room_label="main.py",
            project_id="proj-1",
            created_at=now,
            updated_at=now,
            object_count=1,
        ),
        Room(
            room_id="room-2",
            room_type="concept",
            room_key="search",
            room_label="Search Implementation",
            project_id="proj-1",
            created_at=now,
            updated_at=now,
            object_count=2,
        ),
    ]

    junctions = [
        RoomObject(room_id="room-1", object_id="obj-1", relevance=0.9, placed_at=now),
        RoomObject(room_id="room-2", object_id="obj-1", relevance=0.8, placed_at=now),
        RoomObject(room_id="room-2", object_id="obj-2", relevance=0.9, placed_at=now),
    ]

    in_memory_storage.store_distillation_results(objects, rooms, junctions)
    return in_memory_storage


class TestBuildFromStorage:
    """Tests for build_from_storage method."""

    def test_build_indexes_all_objects(self, seeded_storage):
        """Build indexes all objects from storage."""
        index = PalaceBM25Index()
        count = index.build_from_storage(seeded_storage)

        assert count == 3
        assert index.size == 3
        assert len(index.object_ids) == 3
        assert len(index.corpus) == 3

    def test_build_empty_storage(self, in_memory_storage):
        """Build on empty storage returns zero."""
        index = PalaceBM25Index()
        count = index.build_from_storage(in_memory_storage)

        assert count == 0
        assert index.size == 0
        assert index.bm25 is None

    def test_build_includes_file_paths(self, seeded_storage):
        """Build includes file paths in searchable corpus."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        # Find obj-1's corpus entry
        obj1_idx = index.object_ids.index("obj-1")
        tokens = index.corpus[obj1_idx]

        # File paths should be tokenized
        assert "main" in tokens
        assert "py" in tokens
        assert "test" in tokens

    def test_build_includes_room_metadata(self, seeded_storage):
        """Build includes room keys and labels in searchable corpus."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        # Find obj-1's corpus entry (in rooms room-1 and room-2)
        obj1_idx = index.object_ids.index("obj-1")
        tokens = index.corpus[obj1_idx]

        # Room labels should be tokenized
        assert "search" in tokens
        assert "implementation" in tokens


class TestSearch:
    """Tests for search method."""

    def test_search_returns_scored_results(self, seeded_storage):
        """Search returns (object_id, score) pairs."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        results = index.search("FAISS vector")

        assert len(results) >= 1
        # First result should be obj-1 (has FAISS in exchange_core)
        assert results[0][0] == "obj-1"
        assert results[0][1] > 0

    def test_search_keyword_match(self, seeded_storage):
        """Search finds exact keyword matches."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        results = index.search("BM25 keyword")

        # obj-2 has BM25 in exchange_core
        object_ids = [r[0] for r in results]
        assert "obj-2" in object_ids

    def test_search_file_path_match(self, seeded_storage):
        """Search finds matches in file paths."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        results = index.search("settings.toml")

        # obj-3 has settings.toml in files_touched
        object_ids = [r[0] for r in results]
        assert "obj-3" in object_ids

    def test_search_room_label_match(self, seeded_storage):
        """Search finds matches in room labels."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        results = index.search("Search Implementation")

        # obj-1 and obj-2 are in "Search Implementation" room
        object_ids = [r[0] for r in results]
        assert any(oid in object_ids for oid in ["obj-1", "obj-2"])

    def test_search_respects_limit(self, seeded_storage):
        """Search respects the limit parameter."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        results = index.search("search", limit=1)

        assert len(results) <= 1

    def test_search_sorted_by_score(self, seeded_storage):
        """Search results are sorted by score descending."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        results = index.search("search")

        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self, seeded_storage):
        """Search with empty query returns empty results."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        results = index.search("")

        assert results == []

    def test_search_no_match(self, seeded_storage):
        """Search with no matching terms returns empty results."""
        index = PalaceBM25Index()
        index.build_from_storage(seeded_storage)

        results = index.search("xyznonexistentterm123")

        assert results == []

    def test_search_empty_index(self, in_memory_storage):
        """Search on empty index returns empty results."""
        index = PalaceBM25Index()
        index.build_from_storage(in_memory_storage)

        results = index.search("anything")

        assert results == []


class TestTokenize:
    """Tests for tokenization."""

    def test_tokenize_lowercase(self):
        """Tokenization lowercases text."""
        index = PalaceBM25Index()
        tokens = index._tokenize("FAISS Vector SEARCH")

        assert "faiss" in tokens
        assert "vector" in tokens
        assert "search" in tokens

    def test_tokenize_splits_paths(self):
        """Tokenization splits file paths."""
        index = PalaceBM25Index()
        tokens = index._tokenize("src/searchat/palace/query.py")

        assert "src" in tokens
        assert "searchat" in tokens
        assert "palace" in tokens
        assert "query" in tokens
        assert "py" in tokens

    def test_tokenize_splits_underscores(self):
        """Tokenization splits on underscores."""
        index = PalaceBM25Index()
        tokens = index._tokenize("search_engine_module")

        assert "search" in tokens
        assert "engine" in tokens
        assert "module" in tokens

    def test_tokenize_splits_hyphens(self):
        """Tokenization splits on hyphens."""
        index = PalaceBM25Index()
        tokens = index._tokenize("all-MiniLM-L6-v2")

        assert "all" in tokens
        assert "minilm" in tokens
        assert "l6" in tokens
        assert "v2" in tokens
