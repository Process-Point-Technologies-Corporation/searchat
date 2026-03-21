"""Tests for PalaceStorage DuckDB layer."""
import json
from datetime import datetime

import duckdb
import pytest

from searchat.models.domain import DistilledObject, FileTouched, Room, RoomObject
from searchat.palace.storage import PalaceStorage


@pytest.fixture
def storage():
    """In-memory DuckDB storage for fast isolated tests."""
    conn = duckdb.connect(":memory:")
    s = PalaceStorage(data_dir=None, conn=conn)
    yield s
    conn.close()


def _make_object(
    object_id: str = "obj-1",
    conversation_id: str = "conv-1",
    project_id: str = "proj-1",
    ply_start: int = 0,
    ply_end: int = 3,
) -> DistilledObject:
    return DistilledObject(
        object_id=object_id,
        project_id=project_id,
        conversation_id=conversation_id,
        ply_start=ply_start,
        ply_end=ply_end,
        files_touched=[FileTouched(path="src/main.py", action="modified")],
        exchange_core="Implemented the main function",
        specific_context="Used argparse for CLI arguments",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        exchange_at=datetime(2026, 1, 15, 9, 0, 0),
        embedding_id=0,
        distilled_text="Implemented the main function\nUsed argparse for CLI arguments",
    )


def _make_room(
    room_id: str = "room-1",
    room_type: str = "file",
    room_key: str = "src/main.py",
    object_count: int = 1,
) -> Room:
    return Room(
        room_id=room_id,
        room_type=room_type,
        room_key=room_key,
        room_label="main.py",
        project_id="proj-1",
        created_at=datetime(2026, 1, 15, 10, 0, 0),
        updated_at=datetime(2026, 1, 15, 10, 0, 0),
        object_count=object_count,
    )


def _make_junction(
    room_id: str = "room-1",
    object_id: str = "obj-1",
) -> RoomObject:
    return RoomObject(
        room_id=room_id,
        object_id=object_id,
        relevance=0.9,
        placed_at=datetime(2026, 1, 15, 10, 0, 0),
    )


class TestTableCreation:
    def test_tables_exist(self, storage):
        tables = storage.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "objects" in table_names
        assert "rooms" in table_names
        assert "room_objects" in table_names


class TestStoreAndRetrieve:
    def test_store_object_and_retrieve(self, storage):
        obj = _make_object()
        room = _make_room()
        junc = _make_junction()

        storage.store_distillation_results([obj], [room], [junc])

        retrieved = storage.get_object_by_id("obj-1")
        assert retrieved.object_id == "obj-1"
        assert retrieved.exchange_core == "Implemented the main function"
        assert len(retrieved.files_touched) == 1
        assert retrieved.files_touched[0].path == "src/main.py"

    def test_get_objects_in_room(self, storage):
        obj1 = _make_object(object_id="obj-1", ply_start=0, ply_end=3)
        obj2 = _make_object(
            object_id="obj-2", ply_start=4, ply_end=7,
        )
        obj2.exchange_at = datetime(2026, 1, 15, 10, 0, 0)
        room = _make_room(object_count=2)
        junc1 = _make_junction(object_id="obj-1")
        junc2 = _make_junction(object_id="obj-2")

        storage.store_distillation_results([obj1, obj2], [room], [junc1, junc2])

        objects = storage.get_objects_in_room("room-1")
        assert len(objects) == 2
        assert objects[0].object_id == "obj-1"
        assert objects[1].object_id == "obj-2"

    def test_find_rooms_by_keyword(self, storage):
        room = _make_room()
        storage.store_distillation_results([], [room], [])

        found = storage.find_rooms_by_keyword("main")
        assert len(found) == 1
        assert found[0].room_id == "room-1"

    def test_find_rooms_no_match(self, storage):
        room = _make_room()
        storage.store_distillation_results([], [room], [])

        found = storage.find_rooms_by_keyword("nonexistent")
        assert len(found) == 0

    def test_get_all_rooms(self, storage):
        room1 = _make_room(room_id="room-1")
        room2 = _make_room(room_id="room-2", room_key="src/utils.py")
        storage.store_distillation_results([], [room1, room2], [])

        all_rooms = storage.get_all_rooms()
        assert len(all_rooms) == 2

    def test_get_all_rooms_filtered(self, storage):
        room1 = _make_room(room_id="room-1")
        room2 = _make_room(room_id="room-2")
        room2.project_id = "other-proj"
        storage.store_distillation_results([], [room1, room2], [])

        filtered = storage.get_all_rooms(project_id="proj-1")
        assert len(filtered) == 1
        assert filtered[0].room_id == "room-1"


class TestDedup:
    def test_duplicate_object_ignored(self, storage):
        obj = _make_object()
        room = _make_room()
        junc = _make_junction()

        storage.store_distillation_results([obj], [room], [junc])
        # Same object_id again — ON CONFLICT DO NOTHING
        storage.store_distillation_results([obj], [room], [junc])

        keys = storage.get_existing_object_keys()
        assert len(keys) == 1

    def test_unique_constraint_on_ply(self, storage):
        obj1 = _make_object(object_id="obj-1")
        obj2 = _make_object(object_id="obj-2")  # Same conv+ply range, different ID
        room = _make_room()

        storage.store_distillation_results([obj1], [room], [_make_junction(object_id="obj-1")])

        # obj2 has same (conversation_id, ply_start, ply_end) — should fail unique constraint
        with pytest.raises(Exception):
            storage.store_distillation_results([obj2], [], [])


class TestUpsertRooms:
    def test_room_upsert_updates_count(self, storage):
        room_v1 = _make_room(object_count=1)
        storage.store_distillation_results([], [room_v1], [])

        room_v2 = _make_room(object_count=5)
        room_v2.updated_at = datetime(2026, 1, 16, 10, 0, 0)
        storage.store_distillation_results([], [room_v2], [])

        rooms = storage.get_all_rooms()
        assert len(rooms) == 1
        assert rooms[0].object_count == 5


class TestTransaction:
    def test_rollback_on_error(self, storage):
        obj = _make_object()
        room = _make_room()
        # Invalid junction — references nonexistent object
        bad_junc = _make_junction(object_id="nonexistent")

        with pytest.raises(Exception):
            storage.store_distillation_results([obj], [room], [bad_junc])

        # Object should not have been committed
        keys = storage.get_existing_object_keys()
        assert len(keys) == 0

    def test_storage_change_token_increments_on_commit(self, storage):
        obj = _make_object()
        room = _make_room()
        junc = _make_junction()

        before = storage.get_change_token()
        storage.store_distillation_results([obj], [room], [junc])

        assert storage.get_change_token() == before + 1


class TestGetExistingKeys:
    def test_empty_initially(self, storage):
        keys = storage.get_existing_object_keys()
        assert keys == set()

    def test_returns_stored_keys(self, storage):
        obj = _make_object()
        storage.store_distillation_results([obj], [], [])

        keys = storage.get_existing_object_keys()
        assert ("conv-1", 0, 3) in keys


class TestMigration:
    def test_migrate_adds_distilled_text_from_compact_text(self):
        """Migration adds distilled_text column and copies data from compact_text."""
        conn = duckdb.connect(":memory:")
        # Create old schema with compact_text only
        conn.execute("""
            CREATE TABLE objects (
                object_id VARCHAR PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                conversation_id VARCHAR NOT NULL,
                ply_start INTEGER NOT NULL,
                ply_end INTEGER NOT NULL,
                files_touched JSON,
                exchange_core VARCHAR NOT NULL,
                specific_context VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL,
                exchange_at TIMESTAMP NOT NULL,
                embedding_id BIGINT,
                compact_text VARCHAR NOT NULL,
                UNIQUE(conversation_id, ply_start, ply_end)
            )
        """)
        # Insert test data
        conn.execute("""
            INSERT INTO objects VALUES (
                'obj-1', 'proj-1', 'conv-1', 0, 3, '[]',
                'core', 'context', '2026-01-01', '2026-01-01', 0, 'old text'
            )
        """)

        # Initialize storage (should run migration)
        storage = PalaceStorage(data_dir=None, conn=conn)

        # Verify distilled_text column was added
        cols = conn.execute("PRAGMA table_info('objects')").fetchall()
        col_names = {r[1] for r in cols}

        assert "distilled_text" in col_names
        # compact_text remains (DuckDB constraints prevent dropping)

        # Verify data was copied
        obj = storage.get_object_by_id("obj-1")
        assert obj.distilled_text == "old text"

        conn.close()

    def test_no_migration_needed_for_new_schema(self, storage):
        """No error when schema already has distilled_text."""
        cols = storage.conn.execute("PRAGMA table_info('objects')").fetchall()
        col_names = {r[1] for r in cols}

        assert "distilled_text" in col_names
