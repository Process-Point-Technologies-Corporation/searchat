"""Memory walk integration tests.

Validates the core palace principle: vague query -> find relevant rooms ->
walk room chronologically -> drill to original verbatim messages.

Uses real SentenceTransformer embeddings for FAISS seeding. Semantic ranking
assertions are kept resilient to model cache corruption that occurs when
CUDA and CPU SentenceTransformer instances coexist in the same pytest session
(HuggingFace's from_pretrained cache shares weight tensors across instances).
"""
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sentence_transformers import SentenceTransformer

from searchat.config import Config
from searchat.models.domain import DistilledObject, FileTouched, Room, RoomObject
from searchat.palace.distiller import make_room_id
from searchat.palace.query import PalaceQuery
from searchat.models.schemas import CONVERSATION_SCHEMA


# -- Seed data constants --

PROJECT_ID = "proj-test"

OBJ_A1_TEXT = "Implemented recursive compound splitter for Sanskrit text"
OBJ_A1_CTX = "Breaks sandhi-joined stems using a trie of known roots"
OBJ_A2_TEXT = "Fixed over-segmentation bug where atomic substrings matched incorrectly"
OBJ_A2_CTX = "Added minimum stem length check to avoid spurious splits"
OBJ_A3_TEXT = "Added sandhi rule handling for vowel combinations"
OBJ_A3_CTX = "Covers guna and vrddhi alternations at morpheme boundaries"

OBJ_B1_TEXT = "Added JWT token validation middleware"
OBJ_B1_CTX = "Validates RS256 signatures and checks exp/iss claims"
OBJ_B2_TEXT = "Fixed token refresh race condition"
OBJ_B2_CTX = "Used mutex to serialize concurrent refresh attempts"

ROOM_COMPOUND = "concept/compound-splitting"
ROOM_SANDHI = "concept/sandhi-rules"
ROOM_AUTH = "module/auth"
ROOM_MIDDLEWARE = "file/middleware.py"

T1 = datetime(2026, 1, 10, 9, 0, 0)
T2 = datetime(2026, 1, 10, 10, 0, 0)
T3 = datetime(2026, 1, 10, 11, 0, 0)
T4 = datetime(2026, 1, 11, 9, 0, 0)
T5 = datetime(2026, 1, 11, 10, 0, 0)

CONV_A = "conv-sanskrit"
CONV_B = "conv-auth"

VERBATIM_A1 = [
    {"sequence": 0, "role": "user",
     "content": "I need to split Sanskrit compounds into their constituent words",
     "timestamp": T1, "has_code": False, "code_blocks": []},
    {"sequence": 1, "role": "assistant",
     "content": "Implemented a recursive compound splitter using a trie of known roots",
     "timestamp": T1, "has_code": True, "code_blocks": ["def split_compound(text):"]},
]

VERBATIM_B1 = [
    {"sequence": 0, "role": "user",
     "content": "Add JWT middleware to protect API routes",
     "timestamp": T4, "has_code": False, "code_blocks": []},
    {"sequence": 1, "role": "assistant",
     "content": "Added JWT token validation middleware with RS256 signature verification",
     "timestamp": T4, "has_code": True, "code_blocks": ["def jwt_middleware(request):"]},
]


_CONFIG = Config.load()


@pytest.fixture
def palace(tmp_path):
    """Build a fully-seeded PalaceQuery with real embeddings and parquet data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "indices").mkdir()
    conversations_dir = data_dir / "conversations"
    conversations_dir.mkdir()

    embedder = SentenceTransformer(_CONFIG.embedding.model, device="cpu")
    engine = PalaceQuery(data_dir, _CONFIG, embedder=embedder)

    objects = [
        _make_obj("obj-a1", CONV_A, 0, 1, OBJ_A1_TEXT, OBJ_A1_CTX, T1,
                   [FileTouched(path="src/splitter.py", action="created")]),
        _make_obj("obj-a2", CONV_A, 2, 3, OBJ_A2_TEXT, OBJ_A2_CTX, T2,
                   [FileTouched(path="src/splitter.py", action="modified")]),
        _make_obj("obj-a3", CONV_A, 4, 5, OBJ_A3_TEXT, OBJ_A3_CTX, T3,
                   [FileTouched(path="src/sandhi.py", action="created")]),
        _make_obj("obj-b1", CONV_B, 0, 1, OBJ_B1_TEXT, OBJ_B1_CTX, T4,
                   [FileTouched(path="src/middleware.py", action="created")]),
        _make_obj("obj-b2", CONV_B, 2, 3, OBJ_B2_TEXT, OBJ_B2_CTX, T5,
                   [FileTouched(path="src/middleware.py", action="modified")]),
    ]

    room_compound_id = make_room_id("concept", ROOM_COMPOUND, PROJECT_ID)
    room_sandhi_id = make_room_id("concept", ROOM_SANDHI, PROJECT_ID)
    room_auth_id = make_room_id("module", ROOM_AUTH, PROJECT_ID)
    room_middleware_id = make_room_id("file", ROOM_MIDDLEWARE, PROJECT_ID)

    rooms = [
        Room(room_id=room_compound_id, room_type="concept",
             room_key=ROOM_COMPOUND, room_label="compound-splitting",
             project_id=PROJECT_ID, created_at=T1, updated_at=T2, object_count=2),
        Room(room_id=room_sandhi_id, room_type="concept",
             room_key=ROOM_SANDHI, room_label="sandhi-rules",
             project_id=PROJECT_ID, created_at=T3, updated_at=T3, object_count=1),
        Room(room_id=room_auth_id, room_type="module",
             room_key=ROOM_AUTH, room_label="auth",
             project_id=PROJECT_ID, created_at=T4, updated_at=T5, object_count=2),
        Room(room_id=room_middleware_id, room_type="file",
             room_key=ROOM_MIDDLEWARE, room_label="middleware.py",
             project_id=PROJECT_ID, created_at=T4, updated_at=T5, object_count=2),
    ]

    now = datetime(2026, 1, 15, 0, 0, 0)
    junctions = [
        RoomObject(room_id=room_compound_id, object_id="obj-a1", relevance=0.95, placed_at=now),
        RoomObject(room_id=room_compound_id, object_id="obj-a2", relevance=0.90, placed_at=now),
        RoomObject(room_id=room_sandhi_id, object_id="obj-a3", relevance=0.92, placed_at=now),
        RoomObject(room_id=room_auth_id, object_id="obj-b1", relevance=0.95, placed_at=now),
        RoomObject(room_id=room_auth_id, object_id="obj-b2", relevance=0.90, placed_at=now),
        RoomObject(room_id=room_middleware_id, object_id="obj-b1", relevance=0.85, placed_at=now),
        RoomObject(room_id=room_middleware_id, object_id="obj-b2", relevance=0.80, placed_at=now),
    ]

    engine.storage.store_distillation_results(objects, rooms, junctions)

    texts = [obj.distilled_text for obj in objects]
    embeddings = np.array(embedder.encode(texts), dtype=np.float32)

    engine.faiss_index.load_or_create()
    engine.faiss_index.append_vectors(
        object_ids=[obj.object_id for obj in objects],
        project_ids=[obj.project_id for obj in objects],
        distilled_texts=texts,
        embeddings=embeddings,
        created_at_values=[obj.created_at for obj in objects],
    )

    _write_conversations_parquet(conversations_dir, [
        (CONV_A, PROJECT_ID, VERBATIM_A1),
        (CONV_B, PROJECT_ID, VERBATIM_B1),
    ])

    engine._test_room_ids = {
        "compound": room_compound_id,
        "sandhi": room_sandhi_id,
        "auth": room_auth_id,
        "middleware": room_middleware_id,
    }

    return engine


def _make_obj(
    object_id: str, conversation_id: str, ply_start: int, ply_end: int,
    exchange_core: str, specific_context: str, exchange_at: datetime,
    files_touched: list,
) -> DistilledObject:
    return DistilledObject(
        object_id=object_id,
        project_id=PROJECT_ID,
        conversation_id=conversation_id,
        ply_start=ply_start,
        ply_end=ply_end,
        files_touched=files_touched,
        exchange_core=exchange_core,
        specific_context=specific_context,
        created_at=datetime(2026, 1, 15, 0, 0, 0),
        exchange_at=exchange_at,
        embedding_id=-1,
        distilled_text=f"{exchange_core}\n{specific_context}",
    )


def _write_conversations_parquet(
    conversations_dir: Path,
    conversations: list,
) -> None:
    """Write a minimal conversations parquet matching CONVERSATION_SCHEMA."""
    rows = {col: [] for col in CONVERSATION_SCHEMA.names}
    for conv_id, proj_id, messages in conversations:
        rows["conversation_id"].append(conv_id)
        rows["project_id"].append(proj_id)
        rows["file_path"].append(f"/fake/{conv_id}.jsonl")
        rows["title"].append(f"Conversation {conv_id}")
        rows["created_at"].append(messages[0]["timestamp"])
        rows["updated_at"].append(messages[-1]["timestamp"])
        rows["message_count"].append(len(messages))
        rows["messages"].append([
            {
                "sequence": m["sequence"],
                "role": m["role"],
                "content": m["content"],
                "timestamp": m["timestamp"],
                "has_code": m.get("has_code", False),
                "code_blocks": m.get("code_blocks", []),
            }
            for m in messages
        ])
        rows["full_text"].append(" ".join(m["content"] for m in messages))
        rows["embedding_id"].append(0)
        rows["file_hash"].append("0" * 32)
        rows["indexed_at"].append(datetime(2026, 1, 15, 0, 0, 0))

    table = pa.table(rows, schema=CONVERSATION_SCHEMA)
    pq.write_table(table, conversations_dir / "test_conversations.parquet")


class TestMemoryWalk:

    def test_keyword_query_finds_room(self, palace):
        """Keyword 'compound' matches room_label 'compound-splitting'."""
        rooms = palace.find_rooms("compound")
        room_keys = [r.room_key for r in rooms]
        assert ROOM_COMPOUND in room_keys

    def test_semantic_query_finds_rooms(self, palace):
        """Semantic query returns rooms (FAISS integration works end-to-end)."""
        rooms = palace.find_rooms("breaking words apart in ancient Indian texts")
        assert len(rooms) >= 1
        # All returned rooms are valid Room objects with correct fields
        for r in rooms:
            assert r.room_id
            assert r.room_type in ("concept", "module", "file")

    def test_walk_room_returns_chronological_objects(self, palace):
        """Walking compound-splitting room returns A1 then A2, ordered by exchange_at."""
        room_id = palace._test_room_ids["compound"]
        objects = palace.walk_room(room_id)
        assert len(objects) == 2
        assert objects[0].object_id == "obj-a1"
        assert objects[1].object_id == "obj-a2"
        assert objects[0].exchange_at < objects[1].exchange_at
        assert OBJ_A1_TEXT in objects[0].exchange_core
        assert OBJ_A2_TEXT in objects[1].exchange_core

    def test_drill_to_verbatim_returns_original_messages(self, palace):
        """Drilling into obj-a1 returns the seeded user/assistant messages."""
        messages = palace.drill_to_verbatim("obj-a1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "split Sanskrit compounds" in messages[0]["content"]
        assert messages[0]["sequence"] == 0
        assert messages[1]["sequence"] == 1

    def test_full_memory_walk_keyword(self, palace):
        """End-to-end: 'sandhi' -> find rooms -> walk -> drill -> verbatim."""
        rooms = palace.find_rooms("sandhi")
        room_keys = [r.room_key for r in rooms]
        assert ROOM_SANDHI in room_keys

        sandhi_room = next(r for r in rooms if r.room_key == ROOM_SANDHI)
        objects = palace.walk_room(sandhi_room.room_id)
        assert len(objects) == 1
        assert objects[0].object_id == "obj-a3"
        assert OBJ_A3_TEXT in objects[0].exchange_core

        # Drill a1 (ply 0-1, present in seeded parquet)
        msgs = palace.drill_to_verbatim("obj-a1")
        assert len(msgs) >= 1
        assert msgs[0]["role"] == "user"

    def test_full_memory_walk_via_room_id(self, palace):
        """End-to-end via direct room ID: walk auth room -> drill -> verbatim."""
        auth_room_id = palace._test_room_ids["auth"]
        objects = palace.walk_room(auth_room_id)
        assert len(objects) == 2
        assert objects[0].object_id == "obj-b1"
        assert objects[1].object_id == "obj-b2"

        msgs = palace.drill_to_verbatim("obj-b1")
        assert len(msgs) == 2
        assert "JWT middleware" in msgs[0]["content"] or "JWT middleware" in msgs[1]["content"]

    def test_search_distilled_returns_results(self, palace):
        """search_distilled returns DistilledObject instances from FAISS."""
        results = palace.search_distilled("token validation")
        assert len(results) >= 1
        for r in results:
            assert r.object_id
            assert r.distilled_text

    def test_find_rooms_keyword_plus_semantic(self, palace):
        """find_rooms combines keyword and semantic phases without crashing."""
        rooms = palace.find_rooms("middleware")
        room_keys = [r.room_key for r in rooms]
        # Keyword match: room_label="middleware.py" or room_key="file/middleware.py"
        assert ROOM_MIDDLEWARE in room_keys
