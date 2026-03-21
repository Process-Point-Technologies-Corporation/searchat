"""Tests for unified indexer module."""
import uuid
import io
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pytest

from searchat.config import Config
from searchat.core.unified_indexer import UnifiedIndexer
from searchat.core.unified_storage import UnifiedStorage, EMBEDDING_DIM
from searchat.models.domain import ConversationRecord, MessageRecord


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock()
    config.embedding = MagicMock()
    config.embedding.model = "all-MiniLM-L6-v2"
    config.embedding.batch_size = 32
    config.embedding.get_device.return_value = "cpu"
    config.distillation = MagicMock()
    config.distillation.min_exchange_chars = 50
    config.distillation.max_ply_length = 20
    return config


@pytest.fixture
def in_memory_storage():
    """Create an in-memory unified storage for testing."""
    conn = duckdb.connect(":memory:")
    storage = UnifiedStorage(Path("/tmp"), conn=conn)
    yield storage
    storage.close()


class TestExchangeSegmentation:
    """Tests for message segmentation into exchanges."""

    def test_segment_empty_messages(self, mock_config, in_memory_storage):
        """Test segmentation with empty message list."""
        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)
            indexer.min_exchange_chars = 50
            indexer.max_ply_length = 20

            exchanges = indexer._segment_exchanges([])
            assert exchanges == []

    def test_segment_basic_exchange(self, mock_config, in_memory_storage):
        """Test segmentation with a basic user-assistant exchange."""
        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)
            indexer.min_exchange_chars = 10
            indexer.max_ply_length = 20

            messages = [
                {"sequence": 0, "role": "user", "content": "Hello, how are you?"},
                {"sequence": 1, "role": "assistant", "content": "I'm doing well, thank you!"},
            ]

            exchanges = indexer._segment_exchanges(messages)
            assert len(exchanges) == 1
            assert exchanges[0] == (0, 1)

    def test_segment_multiple_exchanges(self, mock_config, in_memory_storage):
        """Test segmentation with multiple user-assistant exchanges."""
        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)
            indexer.min_exchange_chars = 10
            indexer.max_ply_length = 20

            messages = [
                {"sequence": 0, "role": "user", "content": "First question?"},
                {"sequence": 1, "role": "assistant", "content": "First answer here."},
                {"sequence": 2, "role": "user", "content": "Second question?"},
                {"sequence": 3, "role": "assistant", "content": "Second answer here."},
            ]

            exchanges = indexer._segment_exchanges(messages)
            assert len(exchanges) == 2
            assert exchanges[0] == (0, 1)
            assert exchanges[1] == (2, 3)

    def test_segment_filters_short_exchanges(self, mock_config, in_memory_storage):
        """Test that exchanges below min_exchange_chars are filtered."""
        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)
            indexer.min_exchange_chars = 100  # High threshold
            indexer.max_ply_length = 20

            messages = [
                {"sequence": 0, "role": "user", "content": "Hi"},
                {"sequence": 1, "role": "assistant", "content": "Hello"},
            ]

            exchanges = indexer._segment_exchanges(messages)
            assert len(exchanges) == 0  # Should be filtered out

    def test_segment_splits_long_exchanges(self, mock_config, in_memory_storage):
        """Test that long exchanges are split by max_ply_length."""
        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)
            indexer.min_exchange_chars = 10
            indexer.max_ply_length = 3  # Short limit

            # Create a long exchange (5 messages without a user-assistant boundary)
            messages = [
                {"sequence": i, "role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} content"}
                for i in range(10)
            ]
            # Make sure there's no assistant content to trigger boundaries
            # Actually, let's create continuous user messages followed by one assistant
            messages = [
                {"sequence": 0, "role": "user", "content": "User message 0 with content"},
                {"sequence": 1, "role": "user", "content": "User message 1 with content"},
                {"sequence": 2, "role": "user", "content": "User message 2 with content"},
                {"sequence": 3, "role": "user", "content": "User message 3 with content"},
                {"sequence": 4, "role": "user", "content": "User message 4 with content"},
                {"sequence": 5, "role": "assistant", "content": "Finally an assistant response here"},
            ]

            exchanges = indexer._segment_exchanges(messages)
            # Should be split into chunks of max_ply_length
            # Original exchange is (0, 5) = 6 messages
            # With max_ply_length=3, should be: (0,2), (3,5)
            for start, end in exchanges:
                assert end - start + 1 <= indexer.max_ply_length

    def test_segment_handles_empty_assistant_messages(self, mock_config, in_memory_storage):
        """Test that empty assistant messages don't close exchanges."""
        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)
            indexer.min_exchange_chars = 10
            indexer.max_ply_length = 20

            messages = [
                {"sequence": 0, "role": "user", "content": "Question here?"},
                {"sequence": 1, "role": "assistant", "content": ""},  # Empty (tool call)
                {"sequence": 2, "role": "user", "content": "Follow up question"},
                {"sequence": 3, "role": "assistant", "content": "Actual response with content"},
            ]

            exchanges = indexer._segment_exchanges(messages)
            # Empty assistant message shouldn't close the exchange
            # All messages should be in one exchange
            assert len(exchanges) == 1
            assert exchanges[0] == (0, 3)


class TestMessageParsing:
    """Tests for message parsing from parquet format."""

    def test_parse_dict_messages(self, mock_config, in_memory_storage):
        """Test parsing messages in dict format."""
        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)

            messages_raw = [
                {"sequence": 0, "role": "user", "content": "Hello"},
                {"sequence": 1, "role": "assistant", "content": "Hi there"},
            ]

            messages = indexer._parse_messages(messages_raw)
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1]["content"] == "Hi there"

    def test_parse_none_messages(self, mock_config, in_memory_storage):
        """Test parsing None messages returns empty list."""
        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)

            messages = indexer._parse_messages(None)
            assert messages == []

    def test_detect_codex_agent_format(self, tmp_path):
        """Test detecting Codex sessions separately from Claude JSONL."""
        conv_file = tmp_path / "rollout.jsonl"
        conv_file.write_text(
            '{"type":"session_meta","payload":{"id":"codex-1","cwd":"D:\\\\projects\\\\searchat"}}\n',
            encoding="utf-8",
        )

        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)
            assert indexer._detect_agent_format(conv_file) == "codex"

    def test_parse_codex_session(self, tmp_path):
        """Test parsing Codex session JSONL transcripts."""
        codex_dir = tmp_path / ".codex" / "sessions" / "2025" / "01" / "01"
        codex_dir.mkdir(parents=True)
        conv_file = codex_dir / "rollout.jsonl"
        conv_file.write_text(
            "\n".join([
                '{"type":"session_meta","payload":{"id":"codex-1","cwd":"D:\\\\projects\\\\searchat","timestamp":"2025-01-01T10:00:00Z"}}',
                '{"timestamp":"2025-01-01T10:00:01Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Find the bug"}]}}',
                '{"timestamp":"2025-01-01T10:00:02Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Tracing the watcher now."}]}}',
            ]) + "\n",
            encoding="utf-8",
        )

        with patch.object(UnifiedIndexer, "__init__", lambda x, y, z, w: None):
            indexer = UnifiedIndexer.__new__(UnifiedIndexer)
            record = indexer._parse_source_file(conv_file, "codex-sessions", "codex")

            assert record.conversation_id == "codex-1"
            assert record.project_id == "codex-searchat"
            assert record.title == "Find the bug"
            assert record.message_count == 2
            assert record.messages[0].role == "user"
            assert record.messages[1].role == "assistant"


class TestIndexerIntegration:
    """Integration tests for the full indexer."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory structure."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        conv_dir = data_dir / "conversations"
        conv_dir.mkdir()
        indices_dir = data_dir / "indices"
        indices_dir.mkdir()
        return tmp_path

    def test_indexer_with_no_parquet_raises(self, temp_dir, mock_config):
        """Test that indexer raises error when no parquet files exist."""
        # Create storage but no parquet files
        storage = UnifiedStorage(temp_dir / "data")

        mock_embedder = MagicMock()
        indexer = UnifiedIndexer(
            search_dir=temp_dir,
            config=mock_config,
            storage=storage,
            embedder=mock_embedder,
        )

        with pytest.raises(FileNotFoundError, match="No parquet files"):
            indexer.index_from_parquet()

        storage.close()


class TestSourceFileState:
    """Tests for cached invalid transcript skipping."""

    def test_index_from_source_files_skips_unchanged_invalid_transcript(
        self, tmp_path, mock_config, in_memory_storage
    ):
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text("invalid json\n", encoding="utf-8")

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        indexer = UnifiedIndexer(
            search_dir=tmp_path,
            config=mock_config,
            storage=in_memory_storage,
            embedder=mock_embedder,
        )

        in_memory_storage.mark_source_file_invalid(
            file_path=str(bad_file),
            conversation_id=bad_file.stem,
            file_size=bad_file.stat().st_size,
            mtime_ns=bad_file.stat().st_mtime_ns,
            error_message="bad json",
        )

        stats = indexer.index_from_source_files([str(bad_file)])

        assert stats["skipped_known_invalid"] == 1
        assert stats["invalid_transcript_count"] == 0
        mock_embedder.encode.assert_not_called()

    def test_index_from_source_files_clears_invalid_state_after_successful_parse(
        self, tmp_path, mock_config, in_memory_storage
    ):
        good_file = tmp_path / "good.jsonl"
        good_file.write_text(
            "\n".join([
                '{"type":"user","message":{"content":"Hello there with enough chars"},"timestamp":"2025-01-01T10:00:00"}',
                '{"type":"assistant","message":{"content":"General Kenobi with enough chars"},"timestamp":"2025-01-01T10:00:01"}',
            ]) + "\n",
            encoding="utf-8",
        )

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.ones((1, EMBEDDING_DIM), dtype=np.float32)

        indexer = UnifiedIndexer(
            search_dir=tmp_path,
            config=mock_config,
            storage=in_memory_storage,
            embedder=mock_embedder,
        )

        in_memory_storage.mark_source_file_invalid(
            file_path=str(good_file),
            conversation_id=good_file.stem,
            file_size=good_file.stat().st_size,
            mtime_ns=good_file.stat().st_mtime_ns - 1,
            error_message="old error",
        )

        stats = indexer.index_from_source_files([str(good_file)])

        assert stats["new_conversations"] == 1
        assert stats["skipped_known_invalid"] == 0
        assert in_memory_storage.get_source_file_state([str(good_file)]) == {}

    def test_index_from_source_files_parallel_parse_aggregates_results(
        self, mock_config, in_memory_storage
    ):
        file_paths = [
            "D:/fake/one.jsonl",
            "D:/fake/two.jsonl",
        ]

        def fake_stat(self):
            return SimpleNamespace(st_size=100, st_mtime_ns=123456)

        provider = MagicMock()
        provider.agent_id = "claude"

        def parse_conversation(path, project_id):
            content = f"content from {Path(path).stem} with enough chars to pass"
            return SimpleNamespace(
                conversation_id=Path(path).stem,
                project_id=project_id,
                file_path=str(path),
                title=Path(path).stem,
                created_at=datetime(2025, 1, 1),
                updated_at=datetime(2025, 1, 1),
                message_count=2,
                messages=[
                    SimpleNamespace(sequence=0, role="user", content=content, timestamp=datetime(2025, 1, 1), has_code=False),
                    SimpleNamespace(sequence=1, role="assistant", content=content, timestamp=datetime(2025, 1, 1), has_code=False),
                ],
                full_text=content,
                embedding_id=-1,
                file_hash="",
                indexed_at=datetime(2025, 1, 1),
                file_size=100,
                mtime_ns=123456,
            )

        provider.parse_conversation.side_effect = parse_conversation

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.ones((2, EMBEDDING_DIM), dtype=np.float32)

        indexer = UnifiedIndexer(
            search_dir=Path("/tmp"),
            config=mock_config,
            storage=in_memory_storage,
            embedder=mock_embedder,
        )
        indexer.parse_workers = 2

        with patch("searchat.core.unified_indexer.detect_provider", return_value=provider):
            with patch("pathlib.Path.stat", fake_stat):
                stats = indexer.index_from_source_files(file_paths)

        assert stats["new_conversations"] == 2
        assert stats["exchanges_created"] == 2
        assert stats["parse_seconds"] >= 0
        assert stats["encode_seconds"] >= 0
        assert stats["store_seconds"] >= 0
        assert mock_embedder.encode.call_count == 1

    def test_changed_claude_jsonl_uses_append_only_reindex(
        self, mock_config, in_memory_storage
    ):
        conv_file = Path("D:/fake/append.jsonl")
        old_line = (
            '{"type":"user","message":{"content":"'
            'User asks a long enough question to survive exchange filtering."},'
            '"timestamp":"2025-01-01T10:00:00"}\n'
        )
        appended_line = (
            '{"type":"assistant","message":{"content":"'
            'Assistant follows up with a long enough answer for indexing."},'
            '"timestamp":"2025-01-01T10:00:01"}\n'
        )
        old_size = len(old_line.encode("utf-8"))
        full_bytes = (old_line + appended_line).encode("utf-8")

        record = ConversationRecord(
            conversation_id=conv_file.stem,
            project_id="tmp_path",
            file_path=str(conv_file),
            title="User asks a long enough question to survive exchange filtering.",
            created_at=datetime(2025, 1, 1, 10, 0, 0),
            updated_at=datetime(2025, 1, 1, 10, 0, 0),
            message_count=1,
            messages=[
                MessageRecord(
                    sequence=0,
                    role="user",
                    content="User asks a long enough question to survive exchange filtering.",
                    timestamp=datetime(2025, 1, 1, 10, 0, 0),
                    has_code=False,
                )
            ],
            full_text="User asks a long enough question to survive exchange filtering.",
            embedding_id=-1,
            file_hash="",
            indexed_at=datetime(2025, 1, 1, 10, 0, 0),
            file_size=old_size,
            mtime_ns=1,
        )
        in_memory_storage.store_conversation(record)
        in_memory_storage.store_exchanges_batch([
            {
                "exchange_id": str(uuid.uuid4()),
                "conversation_id": record.conversation_id,
                "project_id": record.project_id,
                "ply_start": 0,
                "ply_end": 0,
                "exchange_text": f"user: {record.messages[0].content}",
            }
        ], created_at=datetime(2025, 1, 1, 10, 0, 0))

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.ones((1, EMBEDDING_DIM), dtype=np.float32)

        provider = MagicMock()
        provider.agent_id = "claude"

        indexer = UnifiedIndexer(
            search_dir=Path("/tmp"),
            config=mock_config,
            storage=in_memory_storage,
            embedder=mock_embedder,
        )

        def fake_stat(self):
            if self == conv_file:
                return SimpleNamespace(st_size=len(full_bytes), st_mtime_ns=999)
            return SimpleNamespace(st_size=100, st_mtime_ns=123456)

        def fake_open(self, mode="r", *args, **kwargs):
            if self != conv_file or mode != "rb":
                raise AssertionError(f"Unexpected open for {self} mode={mode}")
            return io.BytesIO(full_bytes)

        with patch("searchat.core.unified_indexer.detect_provider", return_value=provider):
            with patch("pathlib.Path.stat", fake_stat):
                with patch("pathlib.Path.open", fake_open):
                    stats = indexer.index_from_source_files([], changed_file_paths=[str(conv_file)])

        exchanges = in_memory_storage.get_conversation_exchanges(record.conversation_id)
        assert stats["updated_conversations"] == 1
        assert stats["append_only_updates"] == 1
        assert stats["exchanges_created"] == 1
        assert len(exchanges) == 1
        assert exchanges[0]["ply_start"] == 0
        assert exchanges[0]["ply_end"] == 1
        stored = in_memory_storage.get_conversation(record.conversation_id)
        assert stored["message_count"] == 2
        assert mock_embedder.encode.call_count == 1
