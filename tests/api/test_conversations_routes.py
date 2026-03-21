"""Unit tests for conversations API routes (unified-only codebase)."""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from fastapi.testclient import TestClient

from searchat.api.app import app


PATCH_GET_ENGINE = 'searchat.api.routers.conversations.get_unified_search_engine'


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


def _make_engine(get_conversation_return=None, cursor_rows=None):
    """Build a mock UnifiedSearchEngine with storage mock.

    cursor_rows: list of tuples matching the SELECT column order:
        (conversation_id, project_id, title, created_at, updated_at,
         message_count, file_path)
    The mock handles two sequential execute calls:
        1. COUNT(*) → returns (len(cursor_rows),)
        2. Data query → returns cursor_rows
    """
    engine = Mock()
    cursor = MagicMock()
    rows = cursor_rows or []
    # Each execute() returns a new result mock; side_effect sequences them
    count_result = MagicMock()
    count_result.fetchone.return_value = (len(rows),)
    data_result = MagicMock()
    data_result.fetchall.return_value = rows
    cursor.execute.side_effect = [count_result, data_result]
    engine.storage._get_read_cursor.return_value = cursor
    engine.storage._get_cursor.return_value = Mock()
    engine.storage.get_conversation.return_value = get_conversation_return
    return engine


def _conv_rows(rows):
    """Convert list of dicts to list of tuples matching SELECT column order.

    Column order: conversation_id, project_id, title, created_at, updated_at,
                  message_count, file_path
    """
    return [
        (
            r['conversation_id'], r['project_id'], r['title'],
            r['created_at'], r['updated_at'], r['message_count'],
            r['file_path'],
        )
        for r in rows
        if r['message_count'] > 0  # SQL WHERE filters these
    ]


SAMPLE_ROWS_RAW = None  # set in fixture

@pytest.fixture
def sample_rows():
    """Default three-row dataset used by multiple test classes.

    conv-3 has message_count=0 and is filtered by the SQL WHERE clause,
    so _conv_rows() excludes it.
    """
    now = datetime.now()
    raw = [
        {
            'conversation_id': 'conv-1',
            'project_id': 'project-a',
            'title': 'Python Binary Search',
            'created_at': now,
            'updated_at': now,
            'message_count': 10,
            'file_path': '/home/user/.claude/conv-1.jsonl',
            'full_text': 'This is a conversation about implementing binary search in Python...',
        },
        {
            'conversation_id': 'conv-2',
            'project_id': 'project-b',
            'title': 'API Design',
            'created_at': now,
            'updated_at': now,
            'message_count': 5,
            'file_path': 'C:\\Users\\Test\\.claude\\conv-2.jsonl',
            'full_text': 'Discussion about REST API design patterns',
        },
        {
            'conversation_id': 'conv-3',
            'project_id': 'project-a',
            'title': 'Empty Conversation',
            'created_at': now,
            'updated_at': now,
            'message_count': 0,
            'file_path': '/home/user/.claude/conv-3.jsonl',
            'full_text': '',
        },
    ]
    return _conv_rows(raw)


@pytest.fixture
def mock_platform_manager():
    """Mock PlatformManager for terminal operations."""
    mock = Mock()
    mock.platform = "windows"
    mock.normalize_path = Mock(side_effect=lambda x: x)
    mock.open_terminal_with_command = Mock()
    return mock


def _conv_dict(conversation_id='conv-1', project_id='project-a',
               title='Python Binary Search',
               file_path='/home/user/.claude/conv-1.jsonl',
               message_count=10, **overrides):
    """Build a conversation dict matching UnifiedStorage.get_conversation() output."""
    now = datetime.now()
    base = {
        'conversation_id': conversation_id,
        'project_id': project_id,
        'title': title,
        'file_path': file_path,
        'created_at': now,
        'updated_at': now,
        'message_count': message_count,
        'full_text': '',
        'file_hash': 'abc123',
        'indexed_at': now,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# GET /api/conversations/all
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGetAllConversationsEndpoint:
    """Tests for GET /api/conversations/all endpoint."""

    def _get(self, client, rows, params=""):
        """rows: list of tuples (conv_id, project_id, title, created_at, updated_at, msg_count, file_path, snippet)"""
        engine = _make_engine(cursor_rows=rows)
        with patch(PATCH_GET_ENGINE, return_value=engine):
            return client.get(f"/api/conversations/all{params}")

    def test_get_all_conversations_default_sort(self, client, sample_rows):
        """Test getting all conversations with default sort (by length)."""
        response = self._get(client, sample_rows)

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "total" in data
        assert data["total"] == 2  # conv-3 filtered out (0 messages)

        # Sorted by message_count descending (conv-1: 10, conv-2: 5)
        assert data["results"][0]["conversation_id"] == "conv-1"
        assert data["results"][0]["message_count"] == 10
        assert data["results"][1]["conversation_id"] == "conv-2"
        assert data["results"][1]["message_count"] == 5

    def test_get_all_conversations_filters_zero_messages(self, client, sample_rows):
        """Test that conversations with 0 messages are filtered out."""
        response = self._get(client, sample_rows)

        assert response.status_code == 200
        data = response.json()

        conv_ids = [r["conversation_id"] for r in data["results"]]
        assert "conv-3" not in conv_ids
        assert len(data["results"]) == 2

    def test_get_all_conversations_sort_by_length(self, client, sample_rows):
        """Test sorting by message count (length)."""
        response = self._get(client, sample_rows, "?sort_by=length")

        assert response.status_code == 200
        data = response.json()

        assert data["results"][0]["message_count"] >= data["results"][1]["message_count"]

    def test_get_all_conversations_sort_by_date_newest(self, client):
        """Test sorting by newest date (SQL ORDER BY returns newest first)."""
        now = datetime.now()
        # SQL returns pre-sorted: newest first
        rows = _conv_rows([
            {
                'conversation_id': 'conv-new',
                'project_id': 'project-a',
                'title': 'New',
                'created_at': now,
                'updated_at': datetime(2025, 1, 31),
                'message_count': 5,
                'file_path': '/test/new.jsonl',
                'full_text': 'New conversation',
            },
            {
                'conversation_id': 'conv-old',
                'project_id': 'project-a',
                'title': 'Old',
                'created_at': now,
                'updated_at': datetime(2025, 1, 1),
                'message_count': 5,
                'file_path': '/test/old.jsonl',
                'full_text': 'Old conversation',
            },
        ])

        response = self._get(client, rows, "?sort_by=date_newest")

        assert response.status_code == 200
        data = response.json()

        assert data["results"][0]["conversation_id"] == "conv-new"
        assert data["results"][1]["conversation_id"] == "conv-old"

    def test_get_all_conversations_sort_by_date_oldest(self, client):
        """Test sorting by oldest date."""
        now = datetime.now()
        rows = _conv_rows([
            {
                'conversation_id': 'conv-old',
                'project_id': 'project-a',
                'title': 'Old',
                'created_at': now,
                'updated_at': datetime(2025, 1, 1),
                'message_count': 5,
                'file_path': '/test/old.jsonl',
                'full_text': 'Old conversation',
            },
            {
                'conversation_id': 'conv-new',
                'project_id': 'project-a',
                'title': 'New',
                'created_at': now,
                'updated_at': datetime(2025, 1, 31),
                'message_count': 5,
                'file_path': '/test/new.jsonl',
                'full_text': 'New conversation',
            },
        ])

        response = self._get(client, rows, "?sort_by=date_oldest")

        assert response.status_code == 200
        data = response.json()

        assert data["results"][0]["conversation_id"] == "conv-old"
        assert data["results"][1]["conversation_id"] == "conv-new"

    def test_get_all_conversations_sort_by_title(self, client):
        """Test sorting by title alphabetically (SQL ORDER BY returns sorted)."""
        now = datetime.now()
        # SQL returns pre-sorted by title ASC
        rows = _conv_rows([
            {
                'conversation_id': 'conv-2', 'project_id': 'project-b',
                'title': 'API Design', 'created_at': now, 'updated_at': now,
                'message_count': 5, 'file_path': '/test/conv-2.jsonl',
                'full_text': 'API Design discussion',
            },
            {
                'conversation_id': 'conv-1', 'project_id': 'project-a',
                'title': 'Python Binary Search', 'created_at': now, 'updated_at': now,
                'message_count': 10, 'file_path': '/test/conv-1.jsonl',
                'full_text': 'Binary search...',
            },
        ])
        response = self._get(client, rows, "?sort_by=title")

        assert response.status_code == 200
        data = response.json()

        titles = [r["title"] for r in data["results"]]
        assert titles == sorted(titles)

    def test_get_all_conversations_source_detection(self, client, sample_rows):
        """Test that source (WSL/WIN) is detected correctly."""
        response = self._get(client, sample_rows)

        assert response.status_code == 200
        data = response.json()

        # conv-1 has /home/ path (WSL)
        conv1 = next(r for r in data["results"] if r["conversation_id"] == "conv-1")
        assert conv1["source"] == "WSL"

        # conv-2 has C:\ path (Windows)
        conv2 = next(r for r in data["results"] if r["conversation_id"] == "conv-2")
        assert conv2["source"] == "WIN"

    def test_get_all_conversations_has_pagination_fields(self, client):
        """Test that response includes pagination metadata."""
        now = datetime.now()
        rows = _conv_rows([{
            'conversation_id': 'conv-1',
            'project_id': 'test',
            'title': 'Test',
            'created_at': now,
            'updated_at': now,
            'message_count': 5,
            'file_path': '/test/conv.jsonl',
        }])

        response = self._get(client, rows)

        assert response.status_code == 200
        data = response.json()

        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_more" in data
        assert data["has_more"] is False  # only 1 result, less than limit

    def test_get_all_conversations_error_handling(self, client):
        """Test error handling when DuckDB query raises."""
        engine = _make_engine()
        engine.storage._get_read_cursor.return_value.execute.side_effect = Exception("Database error")
        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversations/all")

            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]

    def test_get_all_conversations_filter_by_project(self, client):
        """Test filtering conversations by project."""
        now = datetime.now()
        # SQL WHERE filters to project-a only
        rows = [('conv-1', 'project-a', 'Python Binary Search', now, now, 10,
                 '/home/user/.claude/conv-1.jsonl', 'Binary search...')]
        response = self._get(client, rows, "?project=project-a")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert data["results"][0]["conversation_id"] == "conv-1"
        assert data["results"][0]["project_id"] == "project-a"

    def test_get_all_conversations_filter_by_project_no_results(self, client):
        """Test filtering by project with no matching conversations."""
        response = self._get(client, [], "?project=nonexistent-project")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 0
        assert len(data["results"]) == 0

    def test_get_all_conversations_filter_by_date_today(self, client):
        """Test filtering conversations from today (SQL handles filtering)."""
        now = datetime.now()
        # SQL WHERE filters to today only — mock returns pre-filtered result
        rows = _conv_rows([
            {
                'conversation_id': 'conv-today',
                'project_id': 'project-a',
                'title': 'Today',
                'created_at': now,
                'updated_at': now,
                'message_count': 5,
                'file_path': '/test/today.jsonl',
                'full_text': 'Today conversation',
            },
        ])

        response = self._get(client, rows, "?date=today")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert data["results"][0]["conversation_id"] == "conv-today"

    def test_get_all_conversations_filter_by_date_week(self, client):
        """Test filtering conversations from last 7 days (SQL handles filtering)."""
        now = datetime.now()
        five_days_ago = now - timedelta(days=5)
        # SQL WHERE filters to within last 7 days
        rows = _conv_rows([
            {
                'conversation_id': 'conv-recent',
                'project_id': 'project-a',
                'title': 'Recent',
                'created_at': five_days_ago,
                'updated_at': five_days_ago,
                'message_count': 5,
                'file_path': '/test/recent.jsonl',
                'full_text': 'Recent conversation',
            },
        ])

        response = self._get(client, rows, "?date=week")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert data["results"][0]["conversation_id"] == "conv-recent"

    def test_get_all_conversations_filter_by_date_month(self, client):
        """Test filtering conversations from last 30 days (SQL handles filtering)."""
        now = datetime.now()
        twenty_days_ago = now - timedelta(days=20)
        # SQL WHERE filters to within last 30 days
        rows = _conv_rows([
            {
                'conversation_id': 'conv-recent',
                'project_id': 'project-a',
                'title': 'Recent',
                'created_at': twenty_days_ago,
                'updated_at': twenty_days_ago,
                'message_count': 5,
                'file_path': '/test/recent.jsonl',
                'full_text': 'Recent conversation',
            },
        ])

        response = self._get(client, rows, "?date=month")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert data["results"][0]["conversation_id"] == "conv-recent"

    def test_get_all_conversations_filter_by_custom_date_range(self, client):
        """Test filtering conversations with custom date range (SQL handles filtering)."""
        # SQL WHERE filters to Jan 12-18 range
        rows = _conv_rows([
            {
                'conversation_id': 'conv-jan-15',
                'project_id': 'project-a',
                'title': 'Jan 15',
                'created_at': datetime(2025, 1, 15),
                'updated_at': datetime(2025, 1, 15),
                'message_count': 5,
                'file_path': '/test/jan15.jsonl',
                'full_text': 'Jan 15 conversation',
            },
        ])

        response = self._get(client, rows, "?date=custom&date_from=2025-01-12&date_to=2025-01-18")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert data["results"][0]["conversation_id"] == "conv-jan-15"

    def test_get_all_conversations_filter_combined(self, client):
        """Test filtering by both project and date (SQL handles filtering)."""
        now = datetime.now()
        # SQL WHERE filters to project-a AND today
        rows = _conv_rows([
            {
                'conversation_id': 'conv-a-today',
                'project_id': 'project-a',
                'title': 'A Today',
                'created_at': now,
                'updated_at': now,
                'message_count': 5,
                'file_path': '/test/a-today.jsonl',
                'full_text': 'Project A today',
            },
        ])

        response = self._get(client, rows, "?project=project-a&date=today")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert data["results"][0]["conversation_id"] == "conv-a-today"


# ---------------------------------------------------------------------------
# GET /api/conversation/{conversation_id}
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGetConversationEndpoint:
    """Tests for GET /api/conversation/{conversation_id} endpoint."""

    def test_get_conversation_success(self, client, tmp_path):
        """Test successfully retrieving a conversation."""
        conv_file = tmp_path / "conv-1.jsonl"
        messages = [
            {"type": "user", "message": {"content": "Hello"}, "timestamp": "2025-01-01T10:00:00"},
            {"type": "assistant", "message": {"content": "Hi there!"}, "timestamp": "2025-01-01T10:00:05"},
        ]
        with open(conv_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')

        engine = _make_engine(_conv_dict(
            conversation_id='conv-1',
            title='Python Binary Search',
            file_path=str(conv_file),
        ))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversation/conv-1")

            assert response.status_code == 200
            data = response.json()

            assert data["conversation_id"] == "conv-1"
            assert data["title"] == "Python Binary Search"
            assert data["tool"] == "claude"
            assert data["message_count"] == 2
            assert len(data["messages"]) == 2
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][0]["content"] == "Hello"
            assert data["messages"][1]["role"] == "assistant"
            assert data["messages"][1]["content"] == "Hi there!"

    def test_get_conversation_not_in_index(self, client):
        """Test error when conversation not found in index."""
        engine = _make_engine(get_conversation_return=None)

        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversation/nonexistent")

            assert response.status_code == 404
            assert "not found in index" in response.json()["detail"]

    def test_get_conversation_file_not_found(self, client):
        """Test error when conversation file doesn't exist."""
        engine = _make_engine(_conv_dict(
            file_path='/nonexistent/path/conv-1.jsonl',
        ))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversation/conv-1")

            assert response.status_code == 404
            assert "file not found" in response.json()["detail"].lower()

    def test_get_conversation_invalid_json(self, client, tmp_path):
        """Test error handling for invalid JSON in conversation file."""
        conv_file = tmp_path / "invalid.jsonl"
        with open(conv_file, 'w') as f:
            f.write("invalid json\n")

        engine = _make_engine(_conv_dict(file_path=str(conv_file)))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversation/conv-1")

            assert response.status_code == 500
            assert "No valid JSON objects" in response.json()["detail"]

    def test_get_conversation_rejects_partially_corrupt_jsonl(self, client, tmp_path):
        """Test that partially malformed JSONL files fail with line diagnostics."""
        conv_file = tmp_path / "partial-invalid.jsonl"
        with open(conv_file, 'w') as f:
            f.write(json.dumps({
                "type": "user",
                "message": {"content": "Hello"},
                "timestamp": "2025-01-01T10:00:00",
            }) + '\n')
            f.write('{"type": "assistant", "message": {"content": "broken"}\n')

        engine = _make_engine(_conv_dict(file_path=str(conv_file)))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversation/conv-1")

            assert response.status_code == 500
            assert "Malformed JSONL" in response.json()["detail"]
            assert "line 2" in response.json()["detail"]

    def test_get_conversation_with_list_content(self, client, tmp_path):
        """Test handling of content as list (text blocks)."""
        conv_file = tmp_path / "conv-list.jsonl"
        messages = [
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "text", "text": "First block"},
                        {"type": "text", "text": "Second block"},
                    ]
                },
                "timestamp": "2025-01-01T10:00:00",
            }
        ]
        with open(conv_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')

        engine = _make_engine(_conv_dict(file_path=str(conv_file)))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversation/conv-1")

            assert response.status_code == 200
            data = response.json()

            assert "First block\n\nSecond block" in data["messages"][0]["content"]

    def test_get_conversation_skips_non_user_assistant_messages(self, client, tmp_path):
        """Test that only user/assistant messages are included."""
        conv_file = tmp_path / "conv-mixed.jsonl"
        messages = [
            {"type": "user", "message": {"content": "User message"}, "timestamp": "2025-01-01T10:00:00"},
            {"type": "system", "message": {"content": "System message"}, "timestamp": "2025-01-01T10:00:01"},
            {"type": "assistant", "message": {"content": "Assistant message"}, "timestamp": "2025-01-01T10:00:02"},
        ]
        with open(conv_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')

        engine = _make_engine(_conv_dict(file_path=str(conv_file)))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversation/conv-1")

            assert response.status_code == 200
            data = response.json()

            assert len(data["messages"]) == 2
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][1]["role"] == "assistant"

    def test_get_conversation_with_duplicate_ids(self, client, tmp_path):
        """Test that get_conversation returns a single dict (no duplicate handling needed)."""
        conv_file = tmp_path / "conv-dup.jsonl"
        with open(conv_file, 'w') as f:
            f.write(json.dumps({"type": "user", "message": {"content": "Test"}, "timestamp": "2025-01-01T10:00:00"}) + '\n')

        # get_conversation returns a single dict — duplicates are a storage-layer concern
        engine = _make_engine(_conv_dict(
            conversation_id='dup-id',
            title='First',
            file_path=str(conv_file),
            message_count=1,
        ))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversation/dup-id")

            assert response.status_code == 200
            assert response.json()["title"] == "First"

    def test_get_conversation_codex_session_success(self, client, tmp_path):
        """Test successfully retrieving a Codex session conversation."""
        codex_dir = tmp_path / ".codex" / "sessions" / "2025" / "01" / "01"
        codex_dir.mkdir(parents=True)
        conv_file = codex_dir / "rollout.jsonl"
        messages = [
            {"type": "session_meta", "payload": {"id": "codex-1", "cwd": "D:\\projects\\searchat"}},
            {
                "type": "response_item",
                "timestamp": "2025-01-01T10:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Investigate watcher issue"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2025-01-01T10:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Tracing the agent provider path now."}],
                },
            },
        ]
        with open(conv_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')

        engine = _make_engine(_conv_dict(
            conversation_id='codex-1',
            title='Codex Session',
            file_path=str(conv_file),
        ))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            response = client.get("/api/conversation/codex-1")

            assert response.status_code == 200
            data = response.json()

            assert data["tool"] == "codex"
            assert data["message_count"] == 2
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][0]["content"] == "Investigate watcher issue"
            assert data["messages"][1]["role"] == "assistant"
            assert data["messages"][1]["content"] == "Tracing the agent provider path now."


# ---------------------------------------------------------------------------
# POST /api/resume
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestResumeSessionEndpoint:
    """Tests for POST /api/resume endpoint."""

    def test_resume_claude_session_success(self, client, mock_platform_manager, tmp_path):
        """Test successfully resuming a Claude Code session."""
        conv_file = tmp_path / "conv-1.jsonl"
        messages = [
            {"type": "user", "cwd": "/home/user/project", "message": {"content": "Test"}},
            {"type": "assistant", "message": {"content": "Response"}},
        ]
        with open(conv_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')

        engine = _make_engine(_conv_dict(
            conversation_id='conv-1',
            file_path=str(conv_file),
        ))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            with patch('searchat.api.routers.conversations.get_platform_manager', return_value=mock_platform_manager):
                response = client.post("/api/resume", json={"conversation_id": "conv-1"})

                assert response.status_code == 200
                data = response.json()

                assert data["success"] is True
                assert data["tool"] == "claude"
                assert data["cwd"] == "/home/user/project"
                assert "claude --resume conv-1" in data["command"]
                assert data["platform"] == "windows"

                mock_platform_manager.open_terminal_with_command.assert_called_once()

    def test_resume_vibe_session_success(self, client, mock_platform_manager, tmp_path):
        """Test successfully resuming a Vibe session."""
        conv_file = tmp_path / "session_123.json"
        vibe_data = {
            "metadata": {
                "environment": {
                    "working_directory": "/home/user/vibe-project"
                }
            },
            "messages": [],
        }
        with open(conv_file, 'w') as f:
            json.dump(vibe_data, f)

        engine = _make_engine(_conv_dict(
            conversation_id='session_123',
            project_id='vibe-project',
            title='Vibe Session',
            file_path=str(conv_file),
            message_count=5,
        ))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            with patch('searchat.api.routers.conversations.get_platform_manager', return_value=mock_platform_manager):
                response = client.post("/api/resume", json={"conversation_id": "session_123"})

                assert response.status_code == 200
                data = response.json()

                assert data["success"] is True
                assert data["tool"] == "vibe"
                assert data["cwd"] == "/home/user/vibe-project"
                assert "vibe --resume session_123" in data["command"]

    def test_resume_codex_session_success(self, client, mock_platform_manager, tmp_path):
        """Test successfully resuming a Codex session."""
        codex_dir = tmp_path / ".codex" / "sessions" / "2025" / "01" / "01"
        codex_dir.mkdir(parents=True)
        conv_file = codex_dir / "rollout.jsonl"
        lines = [
            {"type": "session_meta", "payload": {"id": "codex-1", "cwd": "D:\\projects\\searchat"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Test"}],
                },
            },
        ]
        with open(conv_file, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')

        engine = _make_engine(_conv_dict(
            conversation_id='codex-1',
            file_path=str(conv_file),
        ))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            with patch('searchat.api.routers.conversations.get_platform_manager', return_value=mock_platform_manager):
                response = client.post("/api/resume", json={"conversation_id": "codex-1"})

                assert response.status_code == 200
                data = response.json()

                assert data["success"] is True
                assert data["tool"] == "codex"
                assert data["cwd"] == "D:\\projects\\searchat"
                assert "codex resume codex-1" in data["command"]

    def test_resume_conversation_not_found(self, client, mock_platform_manager):
        """Test error when conversation doesn't exist."""
        engine = _make_engine(get_conversation_return=None)

        with patch(PATCH_GET_ENGINE, return_value=engine):
            with patch('searchat.api.routers.conversations.get_platform_manager', return_value=mock_platform_manager):
                response = client.post("/api/resume", json={"conversation_id": "nonexistent"})

                assert response.status_code == 404
                assert "not found" in response.json()["detail"]

    def test_resume_unknown_format(self, client, mock_platform_manager, tmp_path):
        """Test error for unknown conversation format."""
        conv_file = tmp_path / "conv.txt"
        conv_file.write_text("unknown format")

        engine = _make_engine(_conv_dict(
            conversation_id='conv-1',
            file_path=str(conv_file),
        ))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            with patch('searchat.api.routers.conversations.get_platform_manager', return_value=mock_platform_manager):
                response = client.post("/api/resume", json={"conversation_id": "conv-1"})

                assert response.status_code == 400
                assert "Unknown conversation format" in response.json()["detail"]

    def test_resume_with_path_normalization(self, client, mock_platform_manager, tmp_path):
        """Test that paths are normalized for the platform."""
        conv_file = tmp_path / "conv-1.jsonl"
        messages = [{"type": "user", "cwd": "/mnt/c/Users/Test/project", "message": {"content": "Test"}}]
        with open(conv_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')

        engine = _make_engine(_conv_dict(
            conversation_id='conv-1',
            file_path=str(conv_file),
        ))

        mock_platform_manager.normalize_path = Mock(return_value="C:\\Users\\Test\\project")

        with patch(PATCH_GET_ENGINE, return_value=engine):
            with patch('searchat.api.routers.conversations.get_platform_manager', return_value=mock_platform_manager):
                response = client.post("/api/resume", json={"conversation_id": "conv-1"})

                assert response.status_code == 200
                data = response.json()

                assert data["cwd"] == "C:\\Users\\Test\\project"
                mock_platform_manager.normalize_path.assert_called_once_with("/mnt/c/Users/Test/project")

    def test_resume_without_cwd(self, client, mock_platform_manager, tmp_path):
        """Test resuming when no cwd is found in conversation."""
        conv_file = tmp_path / "conv-1.jsonl"
        messages = [
            {"type": "user", "message": {"content": "Test"}},  # No cwd
            {"type": "assistant", "message": {"content": "Response"}},
        ]
        with open(conv_file, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg) + '\n')

        engine = _make_engine(_conv_dict(
            conversation_id='conv-1',
            file_path=str(conv_file),
        ))

        with patch(PATCH_GET_ENGINE, return_value=engine):
            with patch('searchat.api.routers.conversations.get_platform_manager', return_value=mock_platform_manager):
                response = client.post("/api/resume", json={"conversation_id": "conv-1"})

                assert response.status_code == 200
                data = response.json()

                assert data["cwd"] is None
                mock_platform_manager.open_terminal_with_command.assert_called_once()
