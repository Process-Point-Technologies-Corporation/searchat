"""Fixtures for hook tests."""
import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_transcript_path(tmp_path: Path) -> Path:
    """Valid JSONL with user-assistant exchange."""
    transcript = tmp_path / "transcript.jsonl"
    lines = [
        # Initial snapshot (skipped by reader)
        json.dumps({
            "type": "file-history-snapshot",
            "messageId": "msg-001",
            "snapshot": {"messageId": "msg-001", "timestamp": "2026-02-05T10:00:00Z"}
        }),
        # User message with string content
        json.dumps({
            "type": "user",
            "uuid": "user-001",
            "parentUuid": None,
            "sessionId": "session-123",
            "cwd": "/test/project",
            "message": {
                "role": "user",
                "content": "How do I implement binary search?"
            },
            "timestamp": "2026-02-05T10:00:01Z"
        }),
        # Assistant thinking (intermediate)
        json.dumps({
            "type": "assistant",
            "uuid": "assist-001",
            "parentUuid": "user-001",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "User wants binary search..."}
                ]
            },
            "timestamp": "2026-02-05T10:00:02Z"
        }),
        # Assistant final response
        json.dumps({
            "type": "assistant",
            "uuid": "assist-002",
            "parentUuid": "assist-001",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Binary search works by repeatedly dividing the search interval in half. Here's an implementation:\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"
                    }
                ]
            },
            "timestamp": "2026-02-05T10:00:05Z"
        }),
    ]
    transcript.write_text("\n".join(lines), encoding="utf-8")
    return transcript


@pytest.fixture
def sample_hook_input(sample_transcript_path: Path) -> dict:
    """Hook stdin JSON with transcript_path, stop_hook_active=False."""
    return {
        "transcript_path": str(sample_transcript_path),
        "stop_hook_active": False,
        "session_id": "session-123",
        "cwd": "/test/project"
    }


@pytest.fixture
def empty_transcript_path(tmp_path: Path) -> Path:
    """Empty file."""
    transcript = tmp_path / "empty.jsonl"
    transcript.write_text("", encoding="utf-8")
    return transcript


@pytest.fixture
def malformed_transcript_path(tmp_path: Path) -> Path:
    """Invalid JSON lines."""
    transcript = tmp_path / "malformed.jsonl"
    transcript.write_text("not json\n{broken: json}\n", encoding="utf-8")
    return transcript


@pytest.fixture
def tool_only_transcript_path(tmp_path: Path) -> Path:
    """Only tool_result/tool_use, no text content."""
    transcript = tmp_path / "tool_only.jsonl"
    lines = [
        json.dumps({
            "type": "user",
            "uuid": "user-001",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-001",
                        "content": "File written successfully."
                    }
                ]
            },
            "timestamp": "2026-02-05T10:00:01Z"
        }),
        json.dumps({
            "type": "assistant",
            "uuid": "assist-001",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-002",
                        "name": "Read",
                        "input": {"path": "/test/file.py"}
                    }
                ]
            },
            "timestamp": "2026-02-05T10:00:02Z"
        }),
    ]
    transcript.write_text("\n".join(lines), encoding="utf-8")
    return transcript


@pytest.fixture
def array_content_transcript_path(tmp_path: Path) -> Path:
    """User content as array of blocks (mixed text and tool_result)."""
    transcript = tmp_path / "array_content.jsonl"
    lines = [
        json.dumps({
            "type": "user",
            "uuid": "user-001",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Based on the file you read, "},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-001",
                        "content": "def foo(): pass"
                    },
                    {"type": "text", "text": "how should I improve it?"}
                ]
            },
            "timestamp": "2026-02-05T10:00:01Z"
        }),
        json.dumps({
            "type": "assistant",
            "uuid": "assist-001",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "You should add type hints and a docstring."}
                ]
            },
            "timestamp": "2026-02-05T10:00:02Z"
        }),
    ]
    transcript.write_text("\n".join(lines), encoding="utf-8")
    return transcript
