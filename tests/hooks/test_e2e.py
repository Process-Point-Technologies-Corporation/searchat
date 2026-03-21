"""End-to-end tests running the hook as a subprocess."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


HOOK_SCRIPT = Path(__file__).parent.parent.parent / "src" / "hooks" / "distill_turn.py"


class TestHookSubprocess:
    """Tests running the hook script as a subprocess."""

    def test_hook_subprocess_stop_hook_active(self, sample_transcript_path: Path):
        """Run script with stop_hook_active=True, verify immediate exit."""
        hook_input = {
            "transcript_path": str(sample_transcript_path),
            "stop_hook_active": True,
            "session_id": "test-session",
            "cwd": str(sample_transcript_path.parent),
        }

        result = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            input=json.dumps(hook_input),
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should exit cleanly with no output
        assert result.returncode == 0
        assert result.stdout == ""

    def test_hook_subprocess_missing_transcript(self, tmp_path: Path):
        """Missing file, verify clean exit."""
        hook_input = {
            "transcript_path": str(tmp_path / "nonexistent.jsonl"),
            "stop_hook_active": False,
            "session_id": "test-session",
            "cwd": str(tmp_path),
        }

        result = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            input=json.dumps(hook_input),
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should exit cleanly
        assert result.returncode == 0

    def test_hook_subprocess_empty_stdin(self):
        """Empty stdin, verify clean exit."""
        result = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0

    def test_hook_subprocess_malformed_transcript(self, malformed_transcript_path: Path):
        """Bad JSON in transcript should cause non-zero exit."""
        hook_input = {
            "transcript_path": str(malformed_transcript_path),
            "stop_hook_active": False,
            "session_id": "test-session",
            "cwd": str(malformed_transcript_path.parent),
        }

        result = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            input=json.dumps(hook_input),
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Script will fail on JSON parse error
        assert result.returncode != 0


class TestHookWithMockedHaiku:
    """Tests that mock the Haiku call."""

    def test_full_flow_mocked(self, sample_transcript_path: Path, tmp_path: Path):
        """Test full flow with mocked Haiku response."""
        from hooks.distill_turn import (
            read_transcript,
            extract_last_exchange,
            parse_response,
            store_distillation,
            USER_PLACEHOLDER,
            ASSISTANT_PLACEHOLDER,
            DISTILLATION_PROMPT,
        )

        # Read and extract
        messages = read_transcript(sample_transcript_path)
        user_text, assistant_text = extract_last_exchange(messages)

        assert user_text
        assert assistant_text

        # Simulate Haiku response
        mock_response = json.dumps({
            "exchange_core": "User asked about binary search implementation",
            "specific_context": "binary_search function",
            "tags": ["binary_search", "algorithm", "python"],
        })

        result = parse_response(mock_response)

        # Store
        cwd = str(tmp_path / "test_project")
        Path(cwd).mkdir(parents=True, exist_ok=True)

        # Override learnings path for test
        learnings_dir = tmp_path / ".claude" / "learnings"
        learnings_dir.mkdir(parents=True, exist_ok=True)

        with patch("hooks.distill_turn.get_learnings_path") as mock_path:
            mock_path.return_value = learnings_dir / "test_project.jsonl"
            store_distillation(result, cwd, "test-session")

        # Verify stored
        learnings_file = learnings_dir / "test_project.jsonl"
        assert learnings_file.exists()

        stored = json.loads(learnings_file.read_text(encoding="utf-8").strip())
        assert stored["exchange_core"] == "User asked about binary search implementation"
        assert "timestamp" in stored
        assert stored["session_id"] == "test-session"
