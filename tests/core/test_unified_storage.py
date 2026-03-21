"""Tests for unified DuckDB storage with VSS and FTS."""
import uuid
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pytest

from searchat.core.unified_storage import UnifiedStorage, EMBEDDING_DIM
from searchat.models.domain import (
    ConversationRecord,
    DistilledObject,
    FileTouched,
    MessageRecord,
    Room,
    RoomObject,
)


@pytest.fixture
def in_memory_storage():
    """Create an in-memory unified storage for testing."""
    conn = duckdb.connect(":memory:")
    storage = UnifiedStorage(Path("/tmp"), conn=conn)
    yield storage
    storage.close()


@pytest.fixture
def sample_conversation():
    """Create a sample conversation record."""
    return ConversationRecord(
        conversation_id="test-conv-001",
        project_id="test-project",
        file_path="/path/to/test.jsonl",
        title="Test Conversation",
        created_at=datetime(2025, 1, 1, 10, 0, 0),
        updated_at=datetime(2025, 1, 1, 11, 0, 0),
        message_count=4,
        messages=[
            MessageRecord(
                sequence=0,
                role="user",
                content="Hello, can you help me?",
                timestamp=datetime(2025, 1, 1, 10, 0, 0),
                has_code=False,
            ),
            MessageRecord(
                sequence=1,
                role="assistant",
                content="Of course! What do you need help with?",
                timestamp=datetime(2025, 1, 1, 10, 1, 0),
                has_code=False,
            ),
            MessageRecord(
                sequence=2,
                role="user",
                content="I need to write a Python function.",
                timestamp=datetime(2025, 1, 1, 10, 2, 0),
                has_code=False,
            ),
            MessageRecord(
                sequence=3,
                role="assistant",
                content="Here's a Python function:\n```python\ndef hello():\n    print('Hello!')\n```",
                timestamp=datetime(2025, 1, 1, 10, 3, 0),
                has_code=True,
            ),
        ],
        full_text="Hello, can you help me? Of course! I need to write a Python function...",
        embedding_id=-1,
        file_hash="abc123",
        indexed_at=datetime(2025, 1, 1, 12, 0, 0),
    )


class TestUnifiedStorageInit:
    """Tests for storage initialization."""

    def test_tables_created(self, in_memory_storage):
        """Verify all tables are created on initialization."""
        tables = in_memory_storage.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}

        assert "conversations" in table_names
        assert "messages" in table_names
        assert "exchanges" in table_names
        assert "verbatim_embeddings" in table_names
        assert "palace_objects" in table_names
        assert "rooms" in table_names
        assert "room_objects" in table_names

    def test_extensions_loaded(self, in_memory_storage):
        """Check extension loading status."""
        # Extensions may or may not be available depending on environment
        stats = in_memory_storage.get_stats()
        assert "vss_available" in stats
        assert "fts_available" in stats


class TestConversationCRUD:
    """Tests for conversation CRUD operations."""

    def test_store_conversation(self, in_memory_storage, sample_conversation):
        """Test storing a conversation."""
        in_memory_storage.store_conversation(sample_conversation)

        # Verify conversation was stored
        conv = in_memory_storage.get_conversation("test-conv-001")
        assert conv is not None
        assert conv["title"] == "Test Conversation"
        assert conv["project_id"] == "test-project"
        assert conv["message_count"] == 4

    def test_store_conversation_messages(self, in_memory_storage, sample_conversation):
        """Test that messages are stored with the conversation."""
        in_memory_storage.store_conversation(sample_conversation)

        messages = in_memory_storage.get_conversation_messages("test-conv-001")
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, can you help me?"

    def test_conversation_exists(self, in_memory_storage, sample_conversation):
        """Test conversation existence check."""
        assert not in_memory_storage.conversation_exists("test-conv-001")

        in_memory_storage.store_conversation(sample_conversation)

        assert in_memory_storage.conversation_exists("test-conv-001")
        assert not in_memory_storage.conversation_exists("nonexistent")

    def test_get_all_conversation_ids(self, in_memory_storage, sample_conversation):
        """Test getting all conversation IDs."""
        in_memory_storage.store_conversation(sample_conversation)

        # Create another conversation
        conv2 = ConversationRecord(
            conversation_id="test-conv-002",
            project_id="other-project",
            file_path="/path/to/other.jsonl",
            title="Other Conversation",
            created_at=datetime(2025, 1, 2),
            updated_at=datetime(2025, 1, 2),
            message_count=1,
            messages=[
                MessageRecord(
                    sequence=0,
                    role="user",
                    content="Test",
                    timestamp=datetime(2025, 1, 2),
                    has_code=False,
                )
            ],
            full_text="Test",
            embedding_id=-1,
            file_hash="def456",
            indexed_at=datetime(2025, 1, 2),
        )
        in_memory_storage.store_conversation(conv2)

        # Get all
        all_ids = in_memory_storage.get_all_conversation_ids()
        assert len(all_ids) == 2
        assert "test-conv-001" in all_ids
        assert "test-conv-002" in all_ids

        # Filter by project
        project_ids = in_memory_storage.get_all_conversation_ids(project_id="test-project")
        assert len(project_ids) == 1
        assert "test-conv-001" in project_ids

    def test_mark_and_clear_source_file_state(self, in_memory_storage):
        """Invalid source-file state can be persisted and cleared."""
        in_memory_storage.mark_source_file_invalid(
            file_path="/tmp/bad.jsonl",
            conversation_id="bad",
            file_size=12,
            mtime_ns=34,
            error_message="bad json",
        )

        state = in_memory_storage.get_source_file_state(["/tmp/bad.jsonl"])
        assert state["/tmp/bad.jsonl"]["status"] == "invalid"
        assert state["/tmp/bad.jsonl"]["file_size"] == 12
        assert state["/tmp/bad.jsonl"]["mtime_ns"] == 34
        assert state["/tmp/bad.jsonl"]["error_message"] == "bad json"

        in_memory_storage.clear_source_file_state("/tmp/bad.jsonl")
        assert in_memory_storage.get_source_file_state(["/tmp/bad.jsonl"]) == {}


class TestExchangeCRUD:
    """Tests for exchange CRUD operations."""

    def test_store_exchange(self, in_memory_storage, sample_conversation):
        """Test storing an exchange."""
        in_memory_storage.store_conversation(sample_conversation)

        exchange_id = str(uuid.uuid4())
        in_memory_storage.store_exchange(
            exchange_id=exchange_id,
            conversation_id="test-conv-001",
            project_id="test-project",
            ply_start=0,
            ply_end=1,
            exchange_text="Hello, can you help me?\n\nOf course! What do you need help with?",
            created_at=datetime.utcnow(),
        )

        exchange = in_memory_storage.get_exchange(exchange_id)
        assert exchange is not None
        assert exchange["conversation_id"] == "test-conv-001"
        assert exchange["ply_start"] == 0
        assert exchange["ply_end"] == 1

    def test_get_existing_exchange_keys(self, in_memory_storage, sample_conversation):
        """Test getting existing exchange keys for deduplication."""
        in_memory_storage.store_conversation(sample_conversation)

        # Store two exchanges
        in_memory_storage.store_exchange(
            exchange_id=str(uuid.uuid4()),
            conversation_id="test-conv-001",
            project_id="test-project",
            ply_start=0,
            ply_end=1,
            exchange_text="Exchange 1",
            created_at=datetime.utcnow(),
        )
        in_memory_storage.store_exchange(
            exchange_id=str(uuid.uuid4()),
            conversation_id="test-conv-001",
            project_id="test-project",
            ply_start=2,
            ply_end=3,
            exchange_text="Exchange 2",
            created_at=datetime.utcnow(),
        )

        keys = in_memory_storage.get_existing_exchange_keys()
        assert len(keys) == 2
        assert ("test-conv-001", 0, 1) in keys
        assert ("test-conv-001", 2, 3) in keys

    def test_store_verbatim_embedding(self, in_memory_storage, sample_conversation):
        """Test storing embeddings for exchanges."""
        in_memory_storage.store_conversation(sample_conversation)

        exchange_id = str(uuid.uuid4())
        in_memory_storage.store_exchange(
            exchange_id=exchange_id,
            conversation_id="test-conv-001",
            project_id="test-project",
            ply_start=0,
            ply_end=1,
            exchange_text="Test exchange",
            created_at=datetime.utcnow(),
        )

        # Store embedding
        embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        in_memory_storage.store_verbatim_embedding(exchange_id, embedding)

        # Verify it was stored
        row = in_memory_storage.conn.execute(
            "SELECT exchange_id FROM verbatim_embeddings WHERE exchange_id = ?",
            [exchange_id],
        ).fetchone()
        assert row is not None

    def test_delete_exchange_data_rolls_back_when_called_in_transaction(
        self, in_memory_storage, sample_conversation
    ):
        """Changed-file reindex can delete and restore exchanges atomically."""
        in_memory_storage.store_conversation(sample_conversation)

        exchange_id = str(uuid.uuid4())
        in_memory_storage.store_exchange(
            exchange_id=exchange_id,
            conversation_id="test-conv-001",
            project_id="test-project",
            ply_start=0,
            ply_end=1,
            exchange_text="Exchange before reindex",
            created_at=datetime.utcnow(),
        )
        in_memory_storage.store_verbatim_embedding(
            exchange_id,
            np.ones(EMBEDDING_DIM, dtype=np.float32),
        )

        in_memory_storage._begin_transaction()
        try:
            in_memory_storage._delete_exchange_data(
                "test-conv-001", in_transaction=True,
            )
            assert in_memory_storage.get_exchange(exchange_id) is None
        finally:
            in_memory_storage._rollback()

        exchange = in_memory_storage.get_exchange(exchange_id)
        assert exchange is not None
        assert exchange["exchange_text"] == "Exchange before reindex"


class TestVectorSearch:
    """Tests for vector search functionality."""

    def test_semantic_search_brute_force(self, in_memory_storage, sample_conversation):
        """Test semantic search with brute-force fallback."""
        in_memory_storage.store_conversation(sample_conversation)

        # Store exchanges with embeddings
        for i in range(3):
            exchange_id = str(uuid.uuid4())
            in_memory_storage.store_exchange(
                exchange_id=exchange_id,
                conversation_id="test-conv-001",
                project_id="test-project",
                ply_start=i,
                ply_end=i,
                exchange_text=f"Exchange {i} content",
                created_at=datetime.utcnow(),
            )
            embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            in_memory_storage.store_verbatim_embedding(exchange_id, embedding)

        # Search
        query_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        results = in_memory_storage.search_verbatim_semantic(query_embedding, limit=10)

        # Should find results (brute force if VSS not available)
        assert len(results) > 0
        assert "exchange_id" in results[0]
        assert "distance" in results[0]
        assert "score" in results[0]


class TestPalaceObjectCRUD:
    """Tests for palace object CRUD operations."""

    def test_store_palace_object(self, in_memory_storage):
        """Test storing a palace object."""
        obj = DistilledObject(
            object_id=str(uuid.uuid4()),
            project_id="test-project",
            conversation_id="test-conv-001",
            ply_start=0,
            ply_end=1,
            files_touched=[FileTouched(path="src/main.py", action="modified")],
            exchange_core="User asked about Python functions",
            specific_context="Writing a hello function",
            created_at=datetime.utcnow(),
            exchange_at=datetime.utcnow(),
            embedding_id=-1,
            distilled_text="Python hello function implementation",
        )

        embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        in_memory_storage.store_palace_object(obj, embedding)

        # Retrieve and verify
        retrieved = in_memory_storage.get_palace_object(obj.object_id)
        assert retrieved is not None
        assert retrieved.exchange_core == "User asked about Python functions"
        assert len(retrieved.files_touched) == 1
        assert retrieved.files_touched[0].path == "src/main.py"

    def test_get_existing_palace_keys(self, in_memory_storage):
        """Test getting existing palace object keys."""
        obj = DistilledObject(
            object_id=str(uuid.uuid4()),
            project_id="test-project",
            conversation_id="test-conv-001",
            ply_start=0,
            ply_end=1,
            files_touched=[],
            exchange_core="Test",
            specific_context="Test",
            created_at=datetime.utcnow(),
            exchange_at=datetime.utcnow(),
            embedding_id=-1,
            distilled_text="Test",
        )
        in_memory_storage.store_palace_object(obj)

        keys = in_memory_storage.get_existing_palace_keys()
        assert ("test-conv-001", 0, 1) in keys


class TestRoomCRUD:
    """Tests for room CRUD operations."""

    def test_store_room(self, in_memory_storage):
        """Test storing a room."""
        room = Room(
            room_id="room-001",
            room_type="file",
            room_key="src/main.py",
            room_label="main.py",
            project_id="test-project",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            object_count=1,
        )
        in_memory_storage.store_room(room)

        # Verify via direct query
        row = in_memory_storage.conn.execute(
            "SELECT room_label FROM rooms WHERE room_id = ?",
            ["room-001"],
        ).fetchone()
        assert row is not None
        assert row[0] == "main.py"

    def test_store_room_object_junction(self, in_memory_storage):
        """Test storing room-object junctions."""
        # Create room
        room = Room(
            room_id="room-001",
            room_type="file",
            room_key="src/main.py",
            room_label="main.py",
            project_id="test-project",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            object_count=1,
        )
        in_memory_storage.store_room(room)

        # Create object
        obj = DistilledObject(
            object_id="obj-001",
            project_id="test-project",
            conversation_id="test-conv-001",
            ply_start=0,
            ply_end=1,
            files_touched=[],
            exchange_core="Test",
            specific_context="Test",
            created_at=datetime.utcnow(),
            exchange_at=datetime.utcnow(),
            embedding_id=-1,
            distilled_text="Test",
        )
        in_memory_storage.store_palace_object(obj)

        # Create junction
        junction = RoomObject(
            room_id="room-001",
            object_id="obj-001",
            relevance=0.9,
            placed_at=datetime.utcnow(),
        )
        in_memory_storage.store_room_object(junction)

        # Verify
        rooms = in_memory_storage.get_rooms_for_object("obj-001")
        assert len(rooms) == 1
        assert rooms[0].room_id == "room-001"


class TestBatchOperations:
    """Tests for batch operations."""

    def test_store_distillation_results(self, in_memory_storage):
        """Test batch storage of distillation results."""
        objects = [
            DistilledObject(
                object_id=f"obj-{i}",
                project_id="test-project",
                conversation_id="test-conv-001",
                ply_start=i * 2,
                ply_end=i * 2 + 1,
                files_touched=[],
                exchange_core=f"Core {i}",
                specific_context=f"Context {i}",
                created_at=datetime.utcnow(),
                exchange_at=datetime.utcnow(),
                embedding_id=-1,
                distilled_text=f"Distilled {i}",
            )
            for i in range(3)
        ]

        rooms = [
            Room(
                room_id=f"room-{i}",
                room_type="concept",
                room_key=f"concept-{i}",
                room_label=f"Concept {i}",
                project_id="test-project",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                object_count=1,
            )
            for i in range(2)
        ]

        junctions = [
            RoomObject(
                room_id="room-0",
                object_id="obj-0",
                relevance=0.9,
                placed_at=datetime.utcnow(),
            ),
            RoomObject(
                room_id="room-1",
                object_id="obj-1",
                relevance=0.8,
                placed_at=datetime.utcnow(),
            ),
        ]

        embeddings = np.random.rand(3, EMBEDDING_DIM).astype(np.float32)

        in_memory_storage.store_distillation_results(objects, rooms, junctions, embeddings)

        # Verify objects stored
        stats = in_memory_storage.get_stats()
        assert stats["palace_objects"] == 3
        assert stats["rooms"] == 2

    def test_store_distillation_results_rolls_back_invalid_junction(self, in_memory_storage):
        """Invalid room-object junctions should fail the whole transaction."""
        obj = DistilledObject(
            object_id="obj-1",
            project_id="test-project",
            conversation_id="test-conv-001",
            ply_start=0,
            ply_end=1,
            files_touched=[],
            exchange_core="Core",
            specific_context="Context",
            created_at=datetime.utcnow(),
            exchange_at=datetime.utcnow(),
            embedding_id=-1,
            distilled_text="Distilled",
        )
        room = Room(
            room_id="room-1",
            room_type="concept",
            room_key="concept-1",
            room_label="Concept 1",
            project_id="test-project",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            object_count=1,
        )
        bad_junction = RoomObject(
            room_id="room-1",
            object_id="missing-object",
            relevance=0.9,
            placed_at=datetime.utcnow(),
        )

        with pytest.raises(KeyError):
            in_memory_storage.store_distillation_results(
                [obj], [room], [bad_junction], None,
            )

        assert in_memory_storage.get_palace_object("obj-1") is None
        assert in_memory_storage.get_rooms_for_object("missing-object") == []


class TestStatistics:
    """Tests for storage statistics."""

    def test_get_stats(self, in_memory_storage, sample_conversation):
        """Test getting storage statistics."""
        in_memory_storage.store_conversation(sample_conversation)

        stats = in_memory_storage.get_stats()
        assert stats["conversations"] == 1
        assert stats["messages"] == 4
        assert "vss_available" in stats
        assert "fts_available" in stats
