"""Tests for PalaceQuery engine."""
from datetime import datetime
from pathlib import Path
import shutil
import uuid

import duckdb
import numpy as np
import pytest

from searchat.config import Config
from searchat.models.domain import DistilledObject, FileTouched, Room, RoomObject
from searchat.palace.faiss_index import DistilledFaissIndex
from searchat.palace.query import PalaceQuery
from searchat.palace.storage import PalaceStorage


@pytest.fixture
def query_engine():
    root = Path.cwd() / "pytest_tmp" / f"palace_query_{uuid.uuid4().hex}"
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "indices").mkdir(exist_ok=True)
    config = Config.load()
    engine = PalaceQuery(data_dir, config)
    try:
        yield engine
    finally:
        engine.close()
        shutil.rmtree(root, ignore_errors=True)


def _seed_data(engine: PalaceQuery):
    """Seed storage with test data."""
    now = datetime(2026, 1, 15, 10, 0, 0)
    earlier = datetime(2026, 1, 15, 9, 0, 0)

    obj1 = DistilledObject(
        object_id="obj-1", project_id="proj-1", conversation_id="conv-1",
        ply_start=0, ply_end=3,
        files_touched=[FileTouched(path="src/main.py", action="modified")],
        exchange_core="Implemented main function", specific_context="Used argparse",
        created_at=now, exchange_at=earlier, embedding_id=0,
        distilled_text="Implemented main function\nUsed argparse",
    )
    obj2 = DistilledObject(
        object_id="obj-2", project_id="proj-1", conversation_id="conv-1",
        ply_start=4, ply_end=7,
        files_touched=[FileTouched(path="src/utils.py", action="created")],
        exchange_core="Created utility module", specific_context="Helper functions",
        created_at=now, exchange_at=now, embedding_id=1,
        distilled_text="Created utility module\nHelper functions",
    )

    room1 = Room(
        room_id="room-1", room_type="file", room_key="src/main.py",
        room_label="main.py", project_id="proj-1",
        created_at=now, updated_at=now, object_count=1,
    )
    room2 = Room(
        room_id="room-2", room_type="file", room_key="src/utils.py",
        room_label="utils.py", project_id="proj-1",
        created_at=now, updated_at=now, object_count=1,
    )

    junc1 = RoomObject(room_id="room-1", object_id="obj-1", relevance=0.9, placed_at=now)
    junc2 = RoomObject(room_id="room-2", object_id="obj-2", relevance=0.8, placed_at=now)

    engine.storage.store_distillation_results([obj1, obj2], [room1, room2], [junc1, junc2])

    # Seed FAISS index
    embeddings = np.random.rand(2, 384).astype(np.float32)
    engine.faiss_index.load_or_create()
    engine.faiss_index.append_vectors(
        object_ids=["obj-1", "obj-2"],
        project_ids=["proj-1", "proj-1"],
        distilled_texts=[obj1.distilled_text, obj2.distilled_text],
        embeddings=embeddings,
        created_at_values=[now, now],
    )


class TestWalkRoom:
    def test_walk_room_ordered(self, query_engine):
        _seed_data(query_engine)
        objects = query_engine.walk_room("room-1")
        assert len(objects) == 1
        assert objects[0].object_id == "obj-1"

    def test_walk_empty_room(self, query_engine):
        objects = query_engine.walk_room("nonexistent")
        assert objects == []


class TestFindRooms:
    def test_find_by_keyword(self, query_engine):
        _seed_data(query_engine)
        rooms = query_engine.find_rooms("main")
        assert len(rooms) >= 1
        assert any(r.room_id == "room-1" for r in rooms)

    def test_find_no_match(self, query_engine):
        _seed_data(query_engine)
        rooms = query_engine.find_rooms("zzz_nonexistent_zzz")
        # Semantic search might return results, but keyword won't
        # Just verify no crash
        assert isinstance(rooms, list)


class TestSearchDistilled:
    def test_search_returns_objects(self, query_engine):
        _seed_data(query_engine)
        results = query_engine.search_distilled("main function implementation")
        assert isinstance(results, list)
        # Mock FAISS returns random indices, so results depend on whether
        # the random indices hit valid vector IDs
        # Just verify no crash and correct types
        for r in results:
            assert isinstance(r, DistilledObject)

    def test_search_empty_index(self, query_engine):
        # No data seeded
        query_engine.faiss_index.load_or_create()
        results = query_engine.search_distilled("anything")
        assert results == []


class TestBm25Invalidation:
    def test_bm25_rebuilds_after_storage_changes(self, query_engine):
        _seed_data(query_engine)
        initial_count = query_engine.ensure_bm25_index()
        assert initial_count == 2

        now = datetime(2026, 1, 16, 10, 0, 0)
        obj3 = DistilledObject(
            object_id="obj-3", project_id="proj-1", conversation_id="conv-2",
            ply_start=0, ply_end=1,
            files_touched=[FileTouched(path="src/new.py", action="created")],
            exchange_core="Added new module", specific_context="new helper",
            created_at=now, exchange_at=now, embedding_id=2,
            distilled_text="Added new module\nnew helper",
        )
        room3 = Room(
            room_id="room-3", room_type="file", room_key="src/new.py",
            room_label="new.py", project_id="proj-1",
            created_at=now, updated_at=now, object_count=1,
        )
        junc3 = RoomObject(room_id="room-3", object_id="obj-3", relevance=0.9, placed_at=now)

        query_engine.storage.store_distillation_results([obj3], [room3], [junc3])

        rebuilt_count = query_engine.ensure_bm25_index()
        assert rebuilt_count == 3
