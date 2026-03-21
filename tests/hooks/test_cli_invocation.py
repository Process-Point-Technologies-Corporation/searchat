"""Tests for Haiku CLI invocation (mocked subprocess)."""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestInvokeHaiku:
    """Tests for invoking claude CLI with Haiku model."""

    def test_invoke_haiku_success(self):
        """Mock subprocess, verify command args."""
        from hooks.distill_turn import invoke_haiku

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"exchange_core": "Test", "specific_context": "test.py"}'

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("shutil.which", return_value="/usr/bin/claude"):
            result = invoke_haiku("Test prompt")

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            # Verify command structure
            assert "/usr/bin/claude" in call_args[0][0]
            assert "--print" in call_args[0][0]
            assert "--model" in call_args[0][0]
            assert "haiku" in call_args[0][0]
            # Verify prompt passed via input
            assert call_args.kwargs["input"] == "Test prompt"
            assert result == '{"exchange_core": "Test", "specific_context": "test.py"}'

    def test_invoke_haiku_not_found(self):
        """shutil.which returns None raises RuntimeError."""
        from hooks.distill_turn import invoke_haiku

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="claude CLI not found"):
                invoke_haiku("Test prompt")

    def test_invoke_haiku_cli_failure(self):
        """Non-zero exit raises RuntimeError."""
        from hooks.distill_turn import invoke_haiku

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Authentication failed"

        with patch("subprocess.run", return_value=mock_result), \
             patch("shutil.which", return_value="/usr/bin/claude"):
            with pytest.raises(RuntimeError, match="exit 1"):
                invoke_haiku("Test prompt")

    def test_invoke_haiku_timeout(self):
        """TimeoutExpired raised on timeout."""
        from hooks.distill_turn import invoke_haiku

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 60)), \
             patch("shutil.which", return_value="/usr/bin/claude"):
            with pytest.raises(subprocess.TimeoutExpired):
                invoke_haiku("Test prompt")

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_invoke_haiku_windows_cmd(self):
        """Windows .cmd -> shell=True."""
        from hooks.distill_turn import invoke_haiku

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"exchange_core": "Test", "specific_context": "test.py"}'

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("shutil.which", return_value="C:\\Program Files\\claude.cmd"):
            invoke_haiku("Test prompt")

            call_args = mock_run.call_args
            assert call_args.kwargs.get("shell") is True

    def test_stdin_piping(self):
        """Verify prompt passed via input= not -p flag."""
        from hooks.distill_turn import invoke_haiku

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"

        long_prompt = "A" * 10000  # Long prompt that would exceed command line limits

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("shutil.which", return_value="/usr/bin/claude"):
            invoke_haiku(long_prompt)

            call_args = mock_run.call_args
            # Prompt should be in input, not in command args
            assert call_args.kwargs["input"] == long_prompt
            cmd = call_args[0][0]
            assert "-p" not in cmd
            assert long_prompt not in cmd
