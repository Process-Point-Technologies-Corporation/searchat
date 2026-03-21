"""Unit tests for distill-turn hook (no external dependencies)."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add hooks module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestReadTranscript:
    """Tests for reading and parsing transcript JSONL files."""

    def test_read_transcript_valid(self, sample_transcript_path: Path):
        """Parse JSONL, return message list."""
        from hooks.distill_turn import read_transcript

        messages = read_transcript(sample_transcript_path)

        assert len(messages) >= 2
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(user_msgs) >= 1
        assert len(assistant_msgs) >= 1

    def test_read_transcript_empty(self, empty_transcript_path: Path):
        """Empty file returns empty list."""
        from hooks.distill_turn import read_transcript

        messages = read_transcript(empty_transcript_path)

        assert messages == []

    def test_read_transcript_malformed(self, malformed_transcript_path: Path):
        """Malformed JSON raises error."""
        from hooks.distill_turn import read_transcript

        with pytest.raises(json.JSONDecodeError):
            read_transcript(malformed_transcript_path)


class TestExtractLastExchange:
    """Tests for extracting the last user-assistant exchange."""

    def test_extract_last_exchange_string_content(self, sample_transcript_path: Path):
        """User content as string."""
        from hooks.distill_turn import read_transcript, extract_last_exchange

        messages = read_transcript(sample_transcript_path)
        user_text, assistant_text = extract_last_exchange(messages)

        assert "binary search" in user_text.lower()
        assert "def binary_search" in assistant_text

    def test_extract_last_exchange_array_content(self, array_content_transcript_path: Path):
        """User content as array of blocks."""
        from hooks.distill_turn import read_transcript, extract_last_exchange

        messages = read_transcript(array_content_transcript_path)
        user_text, assistant_text = extract_last_exchange(messages)

        # Should extract text blocks, skip tool_result
        assert "improve" in user_text.lower()
        assert "type hints" in assistant_text.lower()

    def test_extract_last_exchange_tool_only(self, tool_only_transcript_path: Path):
        """Skip tool_result/tool_use only messages."""
        from hooks.distill_turn import read_transcript, extract_last_exchange

        messages = read_transcript(tool_only_transcript_path)
        user_text, assistant_text = extract_last_exchange(messages)

        # No text content, should return empty strings
        assert user_text == ""
        assert assistant_text == ""

    def test_extract_last_exchange_empty(self):
        """No messages returns empty strings."""
        from hooks.distill_turn import extract_last_exchange

        user_text, assistant_text = extract_last_exchange([])

        assert user_text == ""
        assert assistant_text == ""


class TestParseResponse:
    """Tests for parsing LLM JSON responses."""

    def test_parse_response_plain_json(self):
        """Parse clean JSON."""
        from hooks.distill_turn import parse_response

        raw = '{"exchange_core": "Implemented binary search", "specific_context": "binary_search function"}'
        result = parse_response(raw)

        assert result["exchange_core"] == "Implemented binary search"
        assert result["specific_context"] == "binary_search function"

    def test_parse_response_markdown_fenced(self):
        """Strip ```json fencing."""
        from hooks.distill_turn import parse_response

        raw = """```json
{"exchange_core": "Fixed bug", "specific_context": "IndexError on line 42"}
```"""
        result = parse_response(raw)

        assert result["exchange_core"] == "Fixed bug"
        assert result["specific_context"] == "IndexError on line 42"

    def test_parse_response_markdown_fenced_no_lang(self):
        """Strip ``` fencing without language specifier."""
        from hooks.distill_turn import parse_response

        raw = """```
{"exchange_core": "Added feature", "specific_context": "new_function()"}
```"""
        result = parse_response(raw)

        assert result["exchange_core"] == "Added feature"

    def test_parse_response_malformed(self):
        """Raise JSONDecodeError on malformed input."""
        from hooks.distill_turn import parse_response

        with pytest.raises(json.JSONDecodeError):
            parse_response("not valid json")


class TestLoopPrevention:
    """Tests for stop_hook_active loop prevention."""

    def test_loop_prevention_stop_hook_active(self, sample_hook_input: dict):
        """Exit when stop_hook_active=True."""
        from hooks.distill_turn import should_exit_early

        hook_input = {**sample_hook_input, "stop_hook_active": True}

        assert should_exit_early(hook_input) is True

    def test_no_loop_prevention_when_inactive(self, sample_hook_input: dict):
        """Continue when stop_hook_active=False."""
        from hooks.distill_turn import should_exit_early

        hook_input = {**sample_hook_input, "stop_hook_active": False}

        assert should_exit_early(hook_input) is False

    def test_no_loop_prevention_when_missing(self, sample_hook_input: dict):
        """Continue when stop_hook_active is missing."""
        from hooks.distill_turn import should_exit_early

        hook_input = {k: v for k, v in sample_hook_input.items() if k != "stop_hook_active"}

        assert should_exit_early(hook_input) is False
