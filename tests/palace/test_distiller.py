"""Tests for the Distiller engine."""
from datetime import datetime
from typing import List

import pytest

from searchat.config import Config
from searchat.palace.distiller import Distiller, extract_file_paths, make_room_id
from searchat.palace.llm import (
    DistillationInput,
    DistillationLLM,
    DistillationOutput,
    RoomAssignment,
)
from searchat.models.domain import DistilledObject, FileTouched, Room, RoomObject


class MockDistillationLLM(DistillationLLM):
    """Deterministic mock LLM for testing."""

    def distill(self, inputs: List[DistillationInput]) -> List[DistillationOutput]:
        results = []
        for inp in inputs:
            results.append(DistillationOutput(
                exchange_core=f"Distilled exchange {inp.ply_start}-{inp.ply_end}",
                specific_context=f"Context for {inp.conversation_id}",
                room_assignments=[
                    RoomAssignment(
                        room_type="file",
                        room_key="src/main.py",
                        room_label="main.py",
                        relevance=0.9,
                    ),
                ],
            ))
        return results


class TestMakeRoomId:
    def test_deterministic(self):
        id1 = make_room_id("file", "src/main.py", "proj-1")
        id2 = make_room_id("file", "src/main.py", "proj-1")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = make_room_id("file", "src/main.py", "proj-1")
        id2 = make_room_id("file", "src/utils.py", "proj-1")
        assert id1 != id2

    def test_16_char_hex(self):
        rid = make_room_id("concept", "testing", None)
        assert len(rid) == 16
        int(rid, 16)  # Validates hex


class TestExchangeSegmentation:
    @pytest.fixture
    def distiller(self, tmp_path):
        data_dir = tmp_path / ".searchat" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "indices").mkdir()
        config = Config.load()
        # Set min_exchange_chars to 0 so segmentation tests focus on boundary logic
        config.distillation.min_exchange_chars = 0
        return Distiller(tmp_path / ".searchat", config)

    def test_basic_segmentation(self, distiller):
        messages = [
            {"sequence": 0, "role": "user", "content": "hello"},
            {"sequence": 1, "role": "assistant", "content": "hi"},
            {"sequence": 2, "role": "user", "content": "next question"},
            {"sequence": 3, "role": "assistant", "content": "answer"},
        ]
        exchanges = distiller._segment_exchanges(messages)
        assert exchanges == [(0, 1), (2, 3)]

    def test_consecutive_user_messages_collapse(self, distiller):
        messages = [
            {"sequence": 0, "role": "user", "content": "hello"},
            {"sequence": 1, "role": "user", "content": "more context"},
            {"sequence": 2, "role": "assistant", "content": "response"},
        ]
        exchanges = distiller._segment_exchanges(messages)
        assert exchanges == [(0, 2)]

    def test_single_exchange(self, distiller):
        messages = [
            {"sequence": 0, "role": "user", "content": "hello"},
            {"sequence": 1, "role": "assistant", "content": "hi"},
        ]
        exchanges = distiller._segment_exchanges(messages)
        assert exchanges == [(0, 1)]

    def test_empty_messages(self, distiller):
        exchanges = distiller._segment_exchanges([])
        assert exchanges == []

    def test_max_ply_length_split(self, distiller):
        distiller.config.distillation.max_ply_length = 3
        messages = [
            {"sequence": i, "role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(6)
        ]
        # Without splitting: (0,1), (2,3), (4,5)
        # Each is 2 messages, under max_ply_length=3, so no splitting
        exchanges = distiller._segment_exchanges(messages)
        assert all(end - start + 1 <= 3 for start, end in exchanges)

    def test_assistant_only_start(self, distiller):
        messages = [
            {"sequence": 0, "role": "assistant", "content": "system message"},
            {"sequence": 1, "role": "user", "content": "hello"},
            {"sequence": 2, "role": "assistant", "content": "hi"},
        ]
        exchanges = distiller._segment_exchanges(messages)
        # Assistant-only message becomes its own exchange, then user+assistant
        assert len(exchanges) >= 1


class TestFlush:
    @pytest.fixture
    def distiller(self, tmp_path):
        data_dir = tmp_path / ".searchat" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "indices").mkdir()
        config = Config.load()
        return Distiller(tmp_path / ".searchat", config, llm=MockDistillationLLM())

    def test_flush_writes_to_storage_and_faiss(self, distiller):
        from searchat.models.domain import DistilledObject, FileTouched, Room, RoomObject

        now = datetime(2026, 1, 15, 10, 0, 0)
        obj = DistilledObject(
            object_id="obj-flush-1",
            project_id="proj-1",
            conversation_id="conv-1",
            ply_start=0,
            ply_end=3,
            files_touched=[FileTouched(path="src/main.py", action="modified")],
            exchange_core="Test exchange",
            specific_context="Test context",
            created_at=now,
            exchange_at=now,
            embedding_id=-1,
            distilled_text="Test exchange\nTest context",
        )
        room = Room(
            room_id="room-flush-1",
            room_type="file",
            room_key="src/main.py",
            room_label="main.py",
            project_id="proj-1",
            created_at=now,
            updated_at=now,
            object_count=1,
        )
        junc = RoomObject(
            room_id="room-flush-1",
            object_id="obj-flush-1",
            relevance=0.9,
            placed_at=now,
        )

        distiller.flush([obj], [room], [junc])

        # Verify object stored
        stored = distiller.storage.get_object_by_id("obj-flush-1")
        assert stored.exchange_core == "Test exchange"
        assert stored.embedding_id >= 0  # Should have been assigned

        # Verify room stored
        rooms = distiller.storage.get_all_rooms()
        assert len(rooms) == 1

        # Verify FAISS index has vectors
        assert distiller.faiss_index.index.ntotal == 1


class TestEmptyExchangeFiltering:
    """Empty exchange filtering is now integrated into _segment_exchanges."""

    @pytest.fixture
    def distiller(self, tmp_path):
        data_dir = tmp_path / ".searchat" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "indices").mkdir()
        config = Config.load()
        return Distiller(tmp_path / ".searchat", config)

    def test_drops_short_exchanges_at_segmentation(self, distiller):
        """Exchanges below min_exchange_chars are never produced."""
        distiller.config.distillation.min_exchange_chars = 50
        messages = [
            {"sequence": 0, "role": "user", "content": "hi"},
            {"sequence": 1, "role": "assistant", "content": "hello"},
            {"sequence": 2, "role": "user", "content": "x" * 60},
            {"sequence": 3, "role": "assistant", "content": "ok"},
        ]
        exchanges = distiller._segment_exchanges(messages)
        # (0,1) has 7 chars total ("hi" + "hello"), dropped at source
        # (2,3) has 62 chars total, kept
        assert len(exchanges) == 1
        assert exchanges[0] == (2, 3)

    def test_keeps_sufficient_exchanges(self, distiller):
        """Exchanges at or above threshold are all kept."""
        distiller.config.distillation.min_exchange_chars = 5
        messages = [
            {"sequence": 0, "role": "user", "content": "hello"},
            {"sequence": 1, "role": "assistant", "content": "world"},
            {"sequence": 2, "role": "user", "content": "query"},
            {"sequence": 3, "role": "assistant", "content": "reply"},
        ]
        exchanges = distiller._segment_exchanges(messages)
        assert len(exchanges) == 2

    def test_all_empty_messages_produce_no_exchanges(self, distiller):
        """Consecutive empty messages produce zero exchanges."""
        distiller.config.distillation.min_exchange_chars = 10
        messages = [
            {"sequence": 0, "role": "user", "content": ""},
            {"sequence": 1, "role": "assistant", "content": ""},
            {"sequence": 2, "role": "user", "content": ""},
            {"sequence": 3, "role": "assistant", "content": ""},
        ]
        exchanges = distiller._segment_exchanges(messages)
        assert exchanges == []


class TestExtractFilePaths:
    """Regex-based file path extraction from text."""

    def test_basic_paths(self):
        text = "Modified src/main.py and created tests/test_main.py"
        paths = extract_file_paths(text)
        assert "src/main.py" in paths
        assert "tests/test_main.py" in paths

    def test_backtick_wrapped_paths(self):
        text = "Edit `src/config.toml` and check `tests/test_config.py`"
        paths = extract_file_paths(text)
        assert "src/config.toml" in paths
        assert "tests/test_config.py" in paths

    def test_bare_filename(self):
        text = "The main.py file has a bug"
        paths = extract_file_paths(text)
        assert "main.py" in paths

    def test_deep_nested_path(self):
        text = "Changed src/deep/nested/module/file.py to fix import"
        paths = extract_file_paths(text)
        assert "src/deep/nested/module/file.py" in paths

    def test_ignores_natural_language(self):
        """LLM hallucination patterns produce no paths."""
        text = "output files (TOML and Markdown)"
        paths = extract_file_paths(text)
        assert paths == []

    def test_ignores_unknown_placeholder(self):
        text = "unknown"
        paths = extract_file_paths(text)
        assert paths == []

    def test_ignores_glob_patterns(self):
        text = "Zone.Identifier files (pattern: *:Zone.Identifier)"
        paths = extract_file_paths(text)
        assert paths == []

    def test_ignores_concept_descriptions(self):
        text = "Phase 37 results/training regime"
        paths = extract_file_paths(text)
        assert paths == []

    def test_deduplication(self):
        text = "Edit src/main.py then check src/main.py again and `src/main.py`"
        paths = extract_file_paths(text)
        assert paths.count("src/main.py") == 1

    def test_windows_backslash_paths(self):
        text = r"Modified src\palace\distiller.py"
        paths = extract_file_paths(text)
        assert "src/palace/distiller.py" in paths

    def test_dot_slash_prefix_stripped(self):
        text = "Run ./scripts/deploy.sh"
        paths = extract_file_paths(text)
        assert "scripts/deploy.sh" in paths

    def test_multiple_extensions(self):
        text = "Files: config.yaml, schema.sql, index.html, styles.css, app.tsx"
        paths = extract_file_paths(text)
        assert "config.yaml" in paths
        assert "schema.sql" in paths
        assert "index.html" in paths
        assert "styles.css" in paths
        assert "app.tsx" in paths

    def test_jsonl_and_parquet(self):
        text = "Read from data/sentences_train.jsonl and wrote output.parquet"
        paths = extract_file_paths(text)
        assert "data/sentences_train.jsonl" in paths
        assert "output.parquet" in paths

    def test_empty_text(self):
        assert extract_file_paths("") == []

    def test_no_paths_in_text(self):
        text = "The system uses a morpheme-based tokenization approach"
        assert extract_file_paths(text) == []

    def test_preserves_first_appearance_order(self):
        text = "First utils.py, then main.py, then config.toml"
        paths = extract_file_paths(text)
        assert paths == ["utils.py", "main.py", "config.toml"]
